"""
GPU drawing callbacks for onion skin and anchor visualization.
"""

import bisect
import math

import blf
import bpy
import gpu
from bpy_extras.view3d_utils import location_3d_to_region_2d
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from .cache import get_cache, get_active_gp
from .anchors import get_all_anchor_positions
from .transforms import SURFACE_OFFSET, adjust_obj_to_surface
from .debug_log import log, log_onion_draw, log_bake, log_cursor


# Draw handler references
_draw_handler = None
_motion_path_handler = None

# Motion path cache
_motion_path_cache = None  # List of Vector positions
_motion_path_cache_gp = None  # GP object the cache is for
_motion_path_dirty = True
# Motion path GPU batch cache (avoids recreation every redraw)
_motion_path_coords = None  # Pre-built tuple coords for GPU
_motion_path_line_batch = None
_motion_path_point_batch = None

# Motion path visualization enhancements
_motion_path_arrows_batch = None  # GPU batch for direction arrows (triangles)
_motion_path_keyframe_data = None  # [(world_pos, frame_num, tangent), ...] for arrows/labels
_motion_path_labels_handler = None  # POST_PIXEL handler for frame number labels

# Timing chart batches - keyframes are circles, inbetweens are ticks
_motion_path_keyframe_circles_batch = None  # Filled circles at keyframe positions (TRIS)
_motion_path_inbetween_ticks_batch = None  # Perpendicular tick marks for inbetween frames (LINES)

# Baked shrinkwrap offsets - computed ONCE when shrinkwrap enabled or animation changes
# Structure: {frame: z_offset}  - stores the Z offset from raycast
_baked_shrinkwrap_offsets = {}
_baked_offset_valid = False  # True after successful bake

# v9.3: Context-aware driver management
# Driver setup can only happen in safe contexts (UI callbacks, operators, file load)
# NOT in frame_change or depsgraph_update handlers (restricted context)
_driver_setup_pending = False  # True if driver needs setup from next safe context
_bake_in_progress = False  # Guard against overlapping bakes

# Onion skin GPU batch cache - CRITICAL for performance near camera
# Structure: {(frame, z_offset): {'fill_batches': [batch, ...], 'stroke_batches': [batch, ...]}}
# Batches are keyed by (frame, z_offset) and built once, reused on every redraw
# v9.4: Added size limit to prevent unbounded memory growth
_onion_batch_cache = {}
_onion_cache_z_offset = None  # Track z_offset to detect changes
_onion_cache_gp = None  # Track GP object to detect changes
_ONION_CACHE_MAX_SIZE = 100  # Maximum number of cached batch entries (FIFO eviction)

# Keyframe list cache - avoid recomputing sorted keyframes on every draw
# Invalidated when GP object changes or onion batch cache is cleared
_keyframe_cache = None  # Sorted list of keyframe numbers
_keyframe_cache_gp = None  # GP object the cache is for


# v8.5: Timer code removed - now using modal operator in operators.py

# Cached shaders (lazy initialized to avoid GPU calls at import time)
_stroke_shader = None
_fill_shader = None


def _get_stroke_shader():
    """Get cached POLYLINE_UNIFORM_COLOR shader."""
    global _stroke_shader
    if _stroke_shader is None:
        _stroke_shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    return _stroke_shader


def _get_fill_shader():
    """Get cached UNIFORM_COLOR shader."""
    global _fill_shader
    if _fill_shader is None:
        _fill_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    return _fill_shader


# Point shader for POINTS primitive
_point_shader = None


def _get_point_shader():
    """Get cached POINT_UNIFORM_COLOR shader for drawing points."""
    global _point_shader
    if _point_shader is None:
        _point_shader = gpu.shader.from_builtin('POINT_UNIFORM_COLOR')
    return _point_shader


def invalidate_motion_path():
    """Mark motion path cache as dirty, triggering rebuild on next draw."""
    global _motion_path_dirty, _motion_path_line_batch, _motion_path_point_batch, _motion_path_coords
    global _motion_path_arrows_batch, _motion_path_keyframe_data
    global _motion_path_keyframe_circles_batch, _motion_path_inbetween_ticks_batch
    _motion_path_dirty = True
    # Also invalidate GPU batch cache
    _motion_path_line_batch = None
    _motion_path_point_batch = None
    _motion_path_coords = None
    # Invalidate visualization enhancement batches
    _motion_path_arrows_batch = None
    _motion_path_keyframe_data = None
    # Invalidate timing chart batches (keyframe circles + inbetween ticks)
    _motion_path_keyframe_circles_batch = None
    _motion_path_inbetween_ticks_batch = None


def invalidate_onion_batch_cache():
    """Clear onion skin GPU batches only.

    PERFORMANCE (P7): Does NOT clear keyframe cache.
    Call invalidate_keyframe_cache() separately when keyframes change.
    """
    global _onion_batch_cache, _onion_cache_z_offset, _onion_cache_gp
    _onion_batch_cache = {}
    _onion_cache_z_offset = None
    _onion_cache_gp = None


def invalidate_keyframe_cache():
    """Clear keyframe cache. Call when GP keyframes are added/removed/moved."""
    global _keyframe_cache, _keyframe_cache_gp
    _keyframe_cache = None
    _keyframe_cache_gp = None


def invalidate_baked_offsets():
    """Mark baked offsets as invalid, requiring re-bake."""
    global _baked_offset_valid
    _baked_offset_valid = False


def get_baked_offset(frame):
    """
    Get pre-baked shrinkwrap Z offset for a frame.
    Returns offset value, or None if not baked.
    """
    if not _baked_offset_valid:
        return None
    return _baked_shrinkwrap_offsets.get(frame)


def is_bake_valid():
    """Check if baked offsets are valid."""
    return _baked_offset_valid


def _is_safe_context():
    """
    Check if current context allows ID modifications (driver setup).

    Safe contexts: UI callbacks (update functions), operator execute(), on_load_post()
    Unsafe contexts: frame_change_post, depsgraph_update_post, timer callbacks during playback

    Returns True if driver setup can be attempted, False otherwise.
    """
    try:
        # During playback, handlers run in restricted contexts
        # Timer callbacks during playback are also restricted
        if bpy.context.screen and bpy.context.screen.is_animation_playing:
            return False
    except (AttributeError, RuntimeError):
        # Context unavailable = likely in restricted handler
        return False
    return True


def is_driver_setup_pending():
    """Check if driver setup is waiting for a safe context."""
    return _driver_setup_pending


def complete_pending_driver_setup(gp_obj):
    """
    Complete pending driver setup. Call from safe context only.

    Returns True if driver was set up, False if nothing was pending or context unsafe.
    """
    global _driver_setup_pending
    if not _driver_setup_pending:
        return False
    if not _is_safe_context():
        return False
    if gp_obj is None:
        return False

    _setup_shrinkwrap_driver(gp_obj)
    _driver_setup_pending = False
    log("Completed pending driver setup", "BAKE")
    return True


# ============================================================================
# v8.1: DRIVER NAMESPACE OFFSET SYSTEM
# Uses bpy.app.driver_namespace to register a lookup function.
# Driver expression calls this function - NO KEYFRAMES NEEDED!
# Clean timeline, reliable offset application.
# ============================================================================

def _get_shrinkwrap_offset_for_driver(frame):
    """
    Driver namespace function - called by driver expression.
    Returns the baked offset for the given frame.
    This is registered in bpy.app.driver_namespace["shrinkwrap_offset"].
    """
    if not _baked_offset_valid:
        return 0.0
    return _baked_shrinkwrap_offsets.get(int(frame), 0.0)


def register_driver_namespace():
    """Register our lookup function in the driver namespace."""
    bpy.app.driver_namespace["shrinkwrap_offset"] = _get_shrinkwrap_offset_for_driver


def unregister_driver_namespace():
    """Unregister from driver namespace."""
    if "shrinkwrap_offset" in bpy.app.driver_namespace:
        del bpy.app.driver_namespace["shrinkwrap_offset"]


def _setup_shrinkwrap_driver(gp_obj):
    """
    Add driver on delta_location.z that calls our namespace function.
    No keyframes needed - just a Python expression!

    The driver expression 'shrinkwrap_offset(frame)' calls our registered
    function which looks up the offset from our baked dict.

    v9.3: This function should ONLY be called from safe contexts.
    Context safety is checked by the caller (bake_shrinkwrap_offsets or
    complete_pending_driver_setup). No deferred timer - we use pending flag instead.
    """
    # Remove existing driver if present (avoid duplicates)
    try:
        gp_obj.driver_remove("delta_location", 2)  # index 2 = Z
    except RuntimeError:
        pass  # No driver existed

    # NOTE: Namespace function should be registered at addon load time
    # (in __init__.py register()), not here. But ensure it exists as safety.
    if "shrinkwrap_offset" not in bpy.app.driver_namespace:
        register_driver_namespace()

    # Add new driver
    fcurve = gp_obj.driver_add("delta_location", 2)
    driver = fcurve.driver
    driver.type = 'SCRIPTED'

    # Add frame variable that reads current frame from scene
    var_frame = driver.variables.new()
    var_frame.name = "frame"
    var_frame.type = 'SINGLE_PROP'
    var_frame.targets[0].id_type = 'SCENE'
    var_frame.targets[0].id = bpy.context.scene
    var_frame.targets[0].data_path = "frame_current"

    # Add z_offset variable that reads stroke_z_offset from settings
    # This allows the driver to include z_offset in its evaluation
    var_z_offset = driver.variables.new()
    var_z_offset.name = "z_offset"
    var_z_offset.type = 'SINGLE_PROP'
    var_z_offset.targets[0].id_type = 'SCENE'
    var_z_offset.targets[0].id = bpy.context.scene
    var_z_offset.targets[0].data_path = "world_onion.stroke_z_offset"

    # Expression: shrinkwrap offset + z_offset setting
    # When shrinkwrap is enabled, both offsets are combined
    driver.expression = "shrinkwrap_offset(frame) + z_offset"

    log("Setup driver on delta_location.z with shrinkwrap + z_offset", "BAKE")


def _has_shrinkwrap_driver(gp_obj):
    """
    Check if shrinkwrap driver exists on delta_location.z.
    Returns True if driver is present, False otherwise.
    """
    if gp_obj is None:
        return False
    if gp_obj.animation_data is None:
        return False
    for fc in gp_obj.animation_data.drivers:
        if fc.data_path == "delta_location" and fc.array_index == 2:
            return True
    return False


def ensure_shrinkwrap_valid(gp_obj, settings, scene):
    """
    Validate shrinkwrap data state. Called from frame_change handler.

    v9.3: This function does NOT attempt driver setup from handlers because
    handlers run in restricted contexts where ID modifications fail.
    Driver setup is handled separately from safe contexts (UI callbacks,
    operators, file load).

    This function only validates and fixes:
      - _baked_shrinkwrap_offsets dict (populated)
      - _baked_offset_valid flag (True)
      - Namespace function (registered)

    Driver existence is NOT validated here - driver setup happens from:
      - settings.update_realtime() when user enables shrinkwrap
      - __init__.on_load_post() after file load
      - operators.cursor_sync modal when playback stops

    Returns True if shrinkwrap DATA is valid, False otherwise.
    """
    if not settings.depth_interaction_enabled:
        return False

    if gp_obj is None:
        return False

    # Check data components (NOT driver - can't fix from handler)
    bake_valid = is_bake_valid()
    namespace_valid = "shrinkwrap_offset" in bpy.app.driver_namespace

    # If data components are valid, nothing to do - early exit (cheap path)
    if bake_valid and namespace_valid:
        return True

    # Data invalid - re-bake WITHOUT driver setup (we're in handler context)
    log(f"Shrinkwrap data validation: bake={bake_valid} namespace={namespace_valid} - re-baking", "BAKE")
    bake_shrinkwrap_offsets(gp_obj, settings, scene, setup_driver=False)

    # NOTE: No scene.frame_set() here - we're already in frame_change handler.
    # Calling frame_set would cause recursion.

    return is_bake_valid()


def remove_shrinkwrap_driver(gp_obj):
    """
    Remove the shrinkwrap driver when feature is disabled.
    Applies stroke_z_offset directly instead of resetting to 0.
    """
    if gp_obj is None:
        return

    try:
        gp_obj.driver_remove("delta_location", 2)
        log("Removed shrinkwrap driver from delta_location.z", "BAKE")
    except RuntimeError:
        pass  # Driver didn't exist

    # Apply z_offset from settings instead of resetting to 0
    # This ensures z_offset still works when shrinkwrap is disabled
    try:
        settings = bpy.context.scene.world_onion
        gp_obj.delta_location.z = settings.stroke_z_offset
    except (AttributeError, RuntimeError):
        # Fallback to 0 if settings unavailable (e.g., during unregister)
        try:
            gp_obj.delta_location.z = 0.0
        except (AttributeError, RuntimeError):
            pass


def bake_shrinkwrap_offsets(gp_obj, settings, scene, setup_driver=True):
    """
    Pre-compute shrinkwrap Z offsets for ENTIRE animation range.

    Called when shrinkwrap is enabled or animation changes.

    Args:
        gp_obj: The Grease Pencil object
        settings: WorldOnionSettings
        scene: The scene
        setup_driver: If True, attempt driver setup (only succeeds in safe context).
                      Pass False when calling from handlers to avoid context errors.

    v9.3: Driver setup is now context-aware. If setup_driver=True but context is
    unsafe, driver setup is marked as pending and will be completed from the
    next safe context (UI callback, operator, or when playback stops).
    """
    global _baked_shrinkwrap_offsets, _baked_offset_valid, _bake_in_progress, _driver_setup_pending

    # Guard against overlapping bakes (can happen with nested handler calls)
    if _bake_in_progress:
        log("Bake already in progress, skipping", "BAKE")
        return 0

    _bake_in_progress = True

    try:
        _baked_shrinkwrap_offsets = {}
        _baked_offset_valid = False

        if not gp_obj:
            return 0

        if not gp_obj.animation_data or not gp_obj.animation_data.action:
            # No animation - just compute current frame offset
            offset = _compute_single_frame_offset(gp_obj, scene, scene.frame_current)
            if offset is not None:
                _baked_shrinkwrap_offsets[scene.frame_current] = offset
                _baked_offset_valid = True
            # Still need driver setup for single-frame case
            if setup_driver:
                _handle_driver_setup(gp_obj)
            return 1

        # Get frame range from animation
        start_frame = int(gp_obj.animation_data.action.frame_range[0])
        end_frame = int(gp_obj.animation_data.action.frame_range[1])

        # Get location F-curves for evaluation
        fcurves = gp_obj.animation_data.action.fcurves
        fc_x = fcurves.find('location', index=0)
        fc_y = fcurves.find('location', index=1)
        fc_z = fcurves.find('location', index=2)

        if not fc_x or not fc_y or not fc_z:
            return 0

        try:
            depsgraph = bpy.context.evaluated_depsgraph_get()
        except (RuntimeError, AttributeError):
            return 0

        count = 0
        for frame in range(start_frame, end_frame + 1):
            # Evaluate F-curve position
            x = fc_x.evaluate(frame)
            y = fc_y.evaluate(frame)
            z = fc_z.evaluate(frame)

            # Raycast down to find mesh surface
            ray_origin = Vector((x, y, z + 1000.0))
            ray_dir = Vector((0, 0, -1))

            try:
                hit, location, normal, index, hit_obj, matrix = scene.ray_cast(
                    depsgraph, ray_origin, ray_dir
                )

                if hit and hit_obj != gp_obj:
                    # Offset = mesh_z - fcurve_z + small surface offset
                    # Minimum Z mode: only push UP, never down (preserves jump animations)
                    offset = max(0, (location.z + SURFACE_OFFSET) - z)
                    _baked_shrinkwrap_offsets[frame] = offset
                else:
                    # No mesh below - zero offset
                    _baked_shrinkwrap_offsets[frame] = 0.0
            except (RuntimeError, AttributeError):
                _baked_shrinkwrap_offsets[frame] = 0.0

            count += 1

        _baked_offset_valid = True

        # Also invalidate motion path so it rebuilds with baked offsets
        invalidate_motion_path()

        # DEBUG: Log bake results
        if _baked_shrinkwrap_offsets:
            offsets = list(_baked_shrinkwrap_offsets.values())
            offset_range = f"min={min(offsets):.4f} max={max(offsets):.4f}"
        else:
            offset_range = "empty"
        log_bake(count, offset_range)

        # v9.3: Context-aware driver setup
        if setup_driver:
            _handle_driver_setup(gp_obj)

        return count

    finally:
        _bake_in_progress = False


def _handle_driver_setup(gp_obj):
    """
    Handle driver setup with context awareness.
    If context is safe, set up driver immediately.
    If context is unsafe, mark as pending for later completion.
    """
    global _driver_setup_pending

    if _is_safe_context():
        _setup_shrinkwrap_driver(gp_obj)
        _driver_setup_pending = False
    else:
        # Mark as pending - will be completed from next safe context
        _driver_setup_pending = True
        log("Driver setup marked as pending (unsafe context)", "BAKE")


def _compute_single_frame_offset(gp_obj, scene, frame):
    """Compute shrinkwrap offset for a single frame."""
    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
    except (RuntimeError, AttributeError):
        return None

    pos = gp_obj.location.copy()

    ray_origin = Vector((pos.x, pos.y, pos.z + 1000.0))
    ray_dir = Vector((0, 0, -1))

    try:
        hit, location, normal, index, hit_obj, matrix = scene.ray_cast(
            depsgraph, ray_origin, ray_dir
        )
        if hit and hit_obj != gp_obj:
            return (location.z + SURFACE_OFFSET) - pos.z
    except (RuntimeError, AttributeError):
        pass

    return 0.0

# ============================================================================
# MOTION PATH VISUALIZATION HELPERS
# ============================================================================

def _build_spacing_dots(coords, count):
    """
    Build arc-length parameterized spacing tick marks along the motion path.

    This creates tick marks at uniform arc-length intervals, which naturally
    results in dense ticks where the object moves slowly and sparse ticks
    where it moves fast - like a traditional animation timing chart.

    Args:
        coords: List of (x, y, z) tuples representing the path
        count: Number of ticks to place along the path

    Returns:
        List of (position, tangent) tuples where:
        - position: Vector of tick center
        - tangent: Vector of path direction at that point (for perpendicular calc)
    """
    if len(coords) < 2 or count < 1:
        return []

    # 1. Compute cumulative arc length
    distances = []
    cumulative = [0.0]

    for i in range(1, len(coords)):
        p0 = Vector(coords[i - 1])
        p1 = Vector(coords[i])
        dist = (p1 - p0).length
        distances.append(dist)
        cumulative.append(cumulative[-1] + dist)

    total_length = cumulative[-1]
    if total_length < 0.0001:
        return []

    # 2. Place ticks at uniform arc-length intervals
    step = total_length / count
    ticks = []

    for i in range(count + 1):
        target_dist = i * step

        # Find segment containing this arc length using binary search
        j = bisect.bisect_right(cumulative, target_dist)
        if j >= len(cumulative):
            j = len(cumulative) - 1
        if j < 1:
            j = 1

        # Interpolate within segment
        seg_start = cumulative[j - 1]
        seg_len = distances[j - 1] if j - 1 < len(distances) else 0.0001

        if seg_len > 0.0001:
            t = (target_dist - seg_start) / seg_len
            t = max(0.0, min(1.0, t))

            p0 = Vector(coords[j - 1])
            p1 = Vector(coords[j]) if j < len(coords) else p0
            tick_pos = p0.lerp(p1, t)

            # Tangent is the direction of this segment
            tangent = (p1 - p0).normalized()

            ticks.append((tick_pos, tangent))

    return ticks


def _build_arrow_geometry(keyframe_data, size):
    """
    Build V/chevron geometry for a single direction arrow at end of path.

    Only draws one arrow at the last keyframe position, pointing in
    direction of travel. The motion path direction is clear from this
    single indicator.

    Args:
        keyframe_data: List of (world_pos, frame_num, tangent) tuples
        size: Size of arrow in world units

    Returns:
        List of (x, y, z) tuples for line vertices (4 vertices - 2 lines)
    """
    if not keyframe_data:
        return []

    # Only use the LAST keyframe (end of path)
    world_pos, frame_num, tangent = keyframe_data[-1]

    if tangent.length < 0.0001:
        return []

    # Arrow tip is at the end position
    # Arms extend backward (opposite to travel direction)
    backward = -tangent.normalized() * size

    # Perpendicular for the V spread
    up = Vector((0, 0, 1))
    right = tangent.cross(up)
    if right.length < 0.0001:
        right = Vector((1, 0, 0))
    right = right.normalized() * size * 0.5

    # V tip at end, arms go backward and outward
    tip = world_pos
    left_arm = world_pos + backward + right
    right_arm = world_pos + backward - right

    # Two lines: tip to left_arm, tip to right_arm
    return [
        (tip.x, tip.y, tip.z + 0.01),
        (left_arm.x, left_arm.y, left_arm.z + 0.01),
        (tip.x, tip.y, tip.z + 0.01),
        (right_arm.x, right_arm.y, right_arm.z + 0.01),
    ]


def _extract_keyframe_data(gp_obj, fc_x, fc_y, fc_z, settings):
    """
    Extract keyframe positions and tangents for arrows and labels.

    Detects actual keyframes from F-curves and computes tangent
    direction at each keyframe by sampling slightly before/after.

    Args:
        gp_obj: The Grease Pencil object
        fc_x, fc_y, fc_z: Location F-curves
        settings: WorldOnionSettings

    Returns:
        List of (world_pos, frame_num, tangent) tuples
    """
    if not fc_x or not fc_x.keyframe_points:
        return []

    keyframe_data = []
    z_offset = settings.stroke_z_offset
    depth_enabled = settings.depth_interaction_enabled

    # Get all keyframe frame numbers
    keyframe_frames = sorted(set(int(kp.co[0]) for kp in fc_x.keyframe_points))

    for frame in keyframe_frames:
        # Sample position at keyframe
        x = fc_x.evaluate(frame)
        y = fc_y.evaluate(frame)
        z = fc_z.evaluate(frame)
        pos = Vector((x, y, z))

        # Apply shrinkwrap offset if enabled
        if depth_enabled:
            baked_offset = get_baked_offset(frame)
            if baked_offset is not None:
                pos.z += baked_offset

        # Apply z_offset
        if z_offset > 0:
            pos.z += z_offset

        # Compute tangent by sampling before/after
        delta = 0.5
        x_before = fc_x.evaluate(frame - delta)
        y_before = fc_y.evaluate(frame - delta)
        z_before = fc_z.evaluate(frame - delta)

        x_after = fc_x.evaluate(frame + delta)
        y_after = fc_y.evaluate(frame + delta)
        z_after = fc_z.evaluate(frame + delta)

        tangent = Vector((x_after - x_before, y_after - y_before, z_after - z_before))

        keyframe_data.append((pos, frame, tangent))

    return keyframe_data


def get_draw_handlers():
    """Get current draw handler references."""
    return _draw_handler


def _build_onion_batches_for_frame(frame, strokes, z_offset, fill_shader, stroke_shader):
    """
    Build and cache GPU batches for a single onion skin frame.
    Returns dict with 'fill_batches' and 'stroke_batches' lists.
    """
    fill_batches = []
    stroke_batches = []

    for stroke_data in strokes:
        points = stroke_data.get('points')
        if not points or len(points) < 2:
            continue

        # Apply z_offset to points (points are tuples)
        coords = [(p[0], p[1], p[2] + z_offset) for p in points]

        # Build fill batch if has fill triangles
        fill_triangles = stroke_data.get('fill_triangles', [])
        if fill_triangles:
            tri_coords = []
            for i, j, k in fill_triangles:
                if i < len(coords) and j < len(coords) and k < len(coords):
                    tri_coords.extend([coords[i], coords[j], coords[k]])
            if tri_coords:
                fill_batches.append(batch_for_shader(fill_shader, 'TRIS', {"pos": tri_coords}))

        # Build stroke batch
        stroke_batches.append(batch_for_shader(stroke_shader, 'LINE_STRIP', {"pos": coords}))

    return {'fill_batches': fill_batches, 'stroke_batches': stroke_batches}


def draw_onion_callback():
    """
    GPU draw callback - renders cached onion skin strokes.
    Called every viewport redraw.

    PERFORMANCE: Uses batch caching to avoid recreating GPU geometry every frame.
    Batches are cached per (frame, z_offset) and only rebuilt when data changes.
    """
    global _onion_batch_cache, _onion_cache_z_offset, _onion_cache_gp

    try:
        scene = bpy.context.scene
    except (RuntimeError, AttributeError):
        # Context unavailable during render or background operations
        return

    if not hasattr(scene, 'world_onion'):
        return

    settings = scene.world_onion

    if not settings.enabled:
        return

    # Get active GP object (auto-detect)
    gp_obj = get_active_gp(bpy.context)
    if gp_obj is None:
        return

    current_frame = scene.frame_current
    cache = get_cache()

    # Determine which frames to show based on mode
    if settings.mode == 'KEYFRAMES':
        frames_to_show = get_keyframe_based_frames(gp_obj, settings, current_frame)
    else:
        frames_to_show = get_regular_frames(settings, current_frame)

    if not frames_to_show:
        return

    # Check if GP object changed -> invalidate batch cache
    if _onion_cache_gp != gp_obj:
        _onion_batch_cache = {}
        _onion_cache_gp = gp_obj

    # Calculate base z_offset (stroke_z_offset setting)
    base_z_offset = settings.stroke_z_offset if settings.stroke_z_offset > 0 else 0.0

    # Check if z_offset settings changed -> invalidate batch cache
    if _onion_cache_z_offset != base_z_offset:
        _onion_batch_cache = {}
        _onion_cache_z_offset = base_z_offset

    # Set up GPU state
    stroke_shader = _get_stroke_shader()
    fill_shader = _get_fill_shader()

    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(False)

    try:
        region = bpy.context.region
        if region is None:
            return

        # PERFORMANCE: Two-pass rendering reduces shader binds from 2N to 2
        # First pass: all fills with fill_shader bound once
        # Second pass: all strokes with stroke_shader bound once

        # Pre-compute draw data for all frames (cached_batches + colors)
        # This avoids recomputing colors in each pass
        draw_data = []  # List of (cached_batches, fill_color, stroke_color)

        # Cache common settings lookups
        depth_enabled = settings.depth_interaction_enabled
        bake_valid = is_bake_valid() if depth_enabled else False
        color_before = settings.color_before
        color_after = settings.color_after
        fill_opacity = settings.fill_opacity
        stroke_opacity = settings.opacity
        falloff = settings.falloff
        max_offset = max(settings.frames_before, settings.frames_after, 1)

        for frame_offset, frame in frames_to_show:
            if frame not in cache or frame == current_frame:
                continue

            strokes = cache[frame]
            if not strokes:
                continue

            # Calculate z_offset for this frame
            z_offset = base_z_offset
            if depth_enabled and bake_valid:
                baked_offset = get_baked_offset(frame)
                if baked_offset is not None:
                    z_offset += baked_offset

            # Cache key includes frame and z_offset
            cache_key = (frame, round(z_offset, 4))

            # Get or build cached batches
            if cache_key not in _onion_batch_cache:
                _onion_batch_cache[cache_key] = _build_onion_batches_for_frame(
                    frame, strokes, z_offset, fill_shader, stroke_shader
                )
                # v9.4: FIFO eviction to prevent unbounded cache growth
                while len(_onion_batch_cache) > _ONION_CACHE_MAX_SIZE:
                    oldest_key = next(iter(_onion_batch_cache))
                    del _onion_batch_cache[oldest_key]

            cached_batches = _onion_batch_cache[cache_key]

            # Calculate colors
            base_color = color_before if frame < current_frame else color_after
            abs_offset = abs(frame_offset)
            falloff_factor = 1.0 - (abs_offset / max_offset) * falloff if falloff > 0 else 1.0
            falloff_clamped = max(0.1, falloff_factor)

            fill_color = (base_color[0], base_color[1], base_color[2], fill_opacity * falloff_clamped)
            stroke_color = (base_color[0], base_color[1], base_color[2], stroke_opacity * falloff_clamped)

            draw_data.append((cached_batches, fill_color, stroke_color))

        # PASS 1: Draw all fills (bind shader ONCE)
        fill_shader.bind()
        for cached_batches, fill_color, _ in draw_data:
            if cached_batches['fill_batches']:
                fill_shader.uniform_float("color", fill_color)
                for batch in cached_batches['fill_batches']:
                    batch.draw(fill_shader)

        # PASS 2: Draw all strokes (bind shader ONCE)
        stroke_shader.bind()
        stroke_shader.uniform_float("viewportSize", (region.width, region.height))
        stroke_shader.uniform_float("lineWidth", settings.line_width)
        for cached_batches, _, stroke_color in draw_data:
            if cached_batches['stroke_batches']:
                stroke_shader.uniform_float("color", stroke_color)
                for batch in cached_batches['stroke_batches']:
                    batch.draw(stroke_shader)

    finally:
        # Reset GPU state (always runs, even if exception occurs)
        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('NONE')
        gpu.state.depth_mask_set(True)


def get_keyframe_based_frames(gp_obj, settings, current_frame):
    """
    Get frames to show in GP Keyframes mode.

    PERFORMANCE: Uses cached sorted keyframe list to avoid iterating all
    layers/keyframes on every viewport draw.
    """
    global _keyframe_cache, _keyframe_cache_gp

    frames_to_show = []

    if gp_obj is None:
        return frames_to_show

    # Check if we need to rebuild the keyframe cache
    if _keyframe_cache is None or _keyframe_cache_gp != gp_obj:
        # Collect all unique keyframe numbers (expensive - only do once)
        all_keyframes = set()
        for layer in gp_obj.data.layers:
            if layer.hide:
                continue
            for kf in layer.frames:
                all_keyframes.add(kf.frame_number)

        _keyframe_cache = sorted(all_keyframes)
        _keyframe_cache_gp = gp_obj

    sorted_keyframes = _keyframe_cache

    if not sorted_keyframes:
        return frames_to_show

    # Find current keyframe index using binary search (O(log n) instead of O(n))
    idx = bisect.bisect_right(sorted_keyframes, current_frame)
    current_idx = idx - 1 if idx > 0 else 0

    # Clamp to valid range
    if current_idx >= len(sorted_keyframes):
        current_idx = len(sorted_keyframes) - 1

    # Get keyframes before
    for i in range(1, settings.frames_before + 1):
        idx = current_idx - i
        if idx >= 0:
            frames_to_show.append((-i, sorted_keyframes[idx]))

    # Get keyframes after
    for i in range(1, settings.frames_after + 1):
        idx = current_idx + i
        if idx < len(sorted_keyframes):
            frames_to_show.append((i, sorted_keyframes[idx]))

    return frames_to_show


def get_regular_frames(settings, current_frame):
    """Get frames to show in Every Frame mode."""
    frames_to_show = []
    
    step = max(1, settings.frame_step)
    
    for i in range(1, settings.frames_before + 1):
        frame = current_frame - (i * step)
        frames_to_show.append((-i, frame))
    
    for i in range(1, settings.frames_after + 1):
        frame = current_frame + (i * step)
        frames_to_show.append((i, frame))
    
    return frames_to_show


def draw_motion_path_callback():
    """
    GPU draw callback - renders motion path connecting object locations.
    Samples F-Curves to draw path.
    Respects depth interaction (shrinkwrap) if enabled.

    Also renders visualization enhancements:
    - Arc/Ease spacing dots (uniform arc-length placement)
    - Direction arrows at keyframes
    """
    global _motion_path_cache, _motion_path_cache_gp, _motion_path_dirty
    global _motion_path_coords, _motion_path_line_batch, _motion_path_point_batch
    global _motion_path_arrows_batch, _motion_path_keyframe_data
    global _motion_path_keyframe_circles_batch, _motion_path_inbetween_ticks_batch

    try:
        scene = bpy.context.scene
    except (RuntimeError, AttributeError):
        return

    if not hasattr(scene, 'world_onion'):
        return

    settings = scene.world_onion

    if not settings.enabled:
        return

    if not settings.motion_path_enabled:
        return

    # Get active GP object
    gp_obj = get_active_gp(bpy.context)
    if gp_obj is None:
        return

    # Check if cache needs rebuild (dirty or different GP object)
    if _motion_path_dirty or _motion_path_cache is None or _motion_path_cache_gp != gp_obj:
        # Rebuild cache by sampling object animation data
        # Only if there is animation data
        if not gp_obj.animation_data or not gp_obj.animation_data.action:
            _motion_path_cache = None
            _motion_path_cache_gp = gp_obj
            _motion_path_dirty = False
            return

        # Find frame range of keyframes
        start_frame = int(gp_obj.animation_data.action.frame_range[0])
        end_frame = int(gp_obj.animation_data.action.frame_range[1])
        
        if start_frame == end_frame:
            # Single-frame animation - mark as handled to prevent rebuild loop
            _motion_path_cache = None
            _motion_path_cache_gp = gp_obj
            _motion_path_dirty = False
            return

        # Sample F-curves at fractional frames for smooth path
        # F-curves are Bezier curves - they interpolate perfectly at any frame value
        # No need for Catmull-Rom subdivision - just sample at high resolution
        MOTION_PATH_SAMPLES = 500  # Fixed sample count - enough for sharp Bezier curves

        points = []

        # Get location fcurves
        fcurves = gp_obj.animation_data.action.fcurves
        fc_x = fcurves.find('location', index=0)
        fc_y = fcurves.find('location', index=1)
        fc_z = fcurves.find('location', index=2)

        # Cache z_offset setting for motion path
        z_offset = settings.stroke_z_offset
        duration = end_frame - start_frame

        # Get depsgraph for raycasting (needed for shrinkwrap)
        # Define BEFORE if/else so it's available for timing ticks section too
        depsgraph = None
        if settings.depth_interaction_enabled:
            try:
                depsgraph = bpy.context.evaluated_depsgraph_get()
            except (RuntimeError, AttributeError):
                depsgraph = None

        if not fc_x or not fc_y or not fc_z:
            # Fallback to frame_set if F-curves incomplete
            original_frame = scene.frame_current
            step = max(1, duration // MOTION_PATH_SAMPLES)
            for f in range(start_frame, end_frame + 1, step):
                scene.frame_set(f)
                pos = gp_obj.location.copy()
                points.append(pos)
            scene.frame_set(original_frame)
        else:
            # Build keyframe list with positions and interpolation type
            # This handles CONSTANT interpolation by manually lerping for visualization
            keyframes = []
            for kp in fc_x.keyframe_points:
                frame = int(kp.co[0])
                if start_frame <= frame <= end_frame:
                    keyframes.append({
                        'frame': frame,
                        'x': kp.co[1],  # Direct value from keyframe
                        'y': fc_y.evaluate(frame),  # Y/Z may have different keyframe structure
                        'z': fc_z.evaluate(frame),
                        'interp': kp.interpolation,  # 'CONSTANT', 'BEZIER', 'LINEAR'
                    })
            keyframes.sort(key=lambda k: k['frame'])

            # Helper: apply shrinkwrap raycast to a position
            def apply_shrinkwrap(pos):
                if settings.depth_interaction_enabled and depsgraph:
                    ray_origin = Vector((pos.x, pos.y, pos.z + 1000.0))
                    ray_dir = Vector((0, 0, -1))
                    hit, location, normal, index, hit_obj, matrix = scene.ray_cast(
                        depsgraph, ray_origin, ray_dir
                    )
                    if hit and hit_obj != gp_obj:
                        pos.z = location.z + SURFACE_OFFSET
                # Apply z_offset
                if z_offset > 0:
                    pos.z += z_offset
                return pos

            # Sample between keyframes, handling constant interpolation specially
            if len(keyframes) < 2:
                # Single keyframe or none - just evaluate normally
                frame_step = duration / MOTION_PATH_SAMPLES if MOTION_PATH_SAMPLES > 0 else 1
                for i in range(MOTION_PATH_SAMPLES + 1):
                    f = start_frame + i * frame_step
                    pos = Vector((fc_x.evaluate(f), fc_y.evaluate(f), fc_z.evaluate(f)))
                    points.append(apply_shrinkwrap(pos))
            else:
                # Multiple keyframes - handle each segment
                for i, kf in enumerate(keyframes):
                    if i + 1 >= len(keyframes):
                        # Last keyframe - add final point
                        pos = Vector((kf['x'], kf['y'], kf['z']))
                        points.append(apply_shrinkwrap(pos))
                        break

                    next_kf = keyframes[i + 1]
                    segment_frames = next_kf['frame'] - kf['frame']

                    # v9.4: Guard against divide by zero if keyframes at same frame
                    if segment_frames <= 0:
                        continue

                    # Distribute samples proportionally across segments
                    samples_for_segment = max(2, int(MOTION_PATH_SAMPLES * segment_frames / duration))

                    for j in range(samples_for_segment):
                        t = j / samples_for_segment

                        if kf['interp'] == 'CONSTANT':
                            # CONSTANT interpolation: manually lerp for visualization
                            # This shows the visual travel path from A to B
                            x = kf['x'] + (next_kf['x'] - kf['x']) * t
                            y = kf['y'] + (next_kf['y'] - kf['y']) * t
                            z = kf['z'] + (next_kf['z'] - kf['z']) * t
                        else:
                            # BEZIER/LINEAR: use evaluate() (works correctly)
                            f = kf['frame'] + segment_frames * t
                            x = fc_x.evaluate(f)
                            y = fc_y.evaluate(f)
                            z = fc_z.evaluate(f)

                        pos = Vector((x, y, z))
                        points.append(apply_shrinkwrap(pos))

        _motion_path_cache = points
        _motion_path_cache_gp = gp_obj
        _motion_path_dirty = False

        # Build GPU batch directly - no smoothing needed, F-curves are already smooth
        coords = [(p.x, p.y, p.z) for p in points]
        _motion_path_coords = coords
        _motion_path_line_batch = batch_for_shader(
            _get_stroke_shader(), 'LINE_STRIP', {"pos": _motion_path_coords}
        )
        # Points batch uses original unsmoothed positions for keyframe markers
        _motion_path_point_batch = batch_for_shader(
            _get_point_shader(), 'POINTS', {"pos": [(p.x, p.y, p.z) for p in points]}
        )

        # Timing Chart: Keyframe dots + Inbetween dots
        # Both use 3D point sprites (POINTS primitive) - angle-independent rendering
        # Uses F-curve evaluation to match path exactly (respects Bezier interpolation)
        log(f"SPACING_DOTS: enabled={settings.motion_path_spacing_dots_enabled} coords_len={len(coords)}", "DOTS")
        if settings.motion_path_spacing_dots_enabled and fc_x and fc_x.keyframe_points:
            inbetween_point_coords = []  # 3D dots for inbetweens
            keyframe_point_coords = []   # 3D dots for keyframes

            # Get keyframe frames as a set for O(1) lookup
            keyframe_frames = set(int(kp.co[0]) for kp in fc_x.keyframe_points)

            # Build sorted keyframe list for constant interpolation handling
            kf_list = []
            for kp in fc_x.keyframe_points:
                kf_frame = int(kp.co[0])
                if start_frame <= kf_frame <= end_frame:
                    kf_list.append({
                        'frame': kf_frame,
                        'x': kp.co[1],
                        'y': fc_y.evaluate(kf_frame),
                        'z': fc_z.evaluate(kf_frame),
                        'interp': kp.interpolation,
                    })
            kf_list.sort(key=lambda k: k['frame'])

            # Helper: find Z from motion path points (already shrinkwrapped)
            def get_z_from_motion_path(xy_pos):
                """Find the nearest motion path point and return its Z value."""
                if not points:
                    return xy_pos.z
                # Find closest point by XY distance
                best_z = points[0].z
                best_dist_sq = float('inf')
                for p in points:
                    dist_sq = (p.x - xy_pos.x)**2 + (p.y - xy_pos.y)**2
                    if dist_sq < best_dist_sq:
                        best_dist_sq = dist_sq
                        best_z = p.z
                return best_z

            # Helper: get position for frame, handling constant interpolation
            def get_position_for_frame(frame):
                """Get interpolated position for frame, handling constant interp specially."""
                # Find which keyframe segment this frame is in
                for i, kf in enumerate(kf_list):
                    if i + 1 < len(kf_list):
                        next_kf = kf_list[i + 1]
                        # v9.4: Guard against degenerate keyframe pairs
                        if next_kf['frame'] <= kf['frame']:
                            continue
                        if kf['frame'] <= frame < next_kf['frame']:
                            if kf['interp'] == 'CONSTANT':
                                # Manually lerp for visualization
                                t = (frame - kf['frame']) / (next_kf['frame'] - kf['frame'])
                                return Vector((
                                    kf['x'] + (next_kf['x'] - kf['x']) * t,
                                    kf['y'] + (next_kf['y'] - kf['y']) * t,
                                    kf['z'] + (next_kf['z'] - kf['z']) * t,
                                ))
                            else:
                                # BEZIER/LINEAR: use evaluate()
                                return Vector((fc_x.evaluate(frame), fc_y.evaluate(frame), fc_z.evaluate(frame)))
                # Fallback to evaluate
                return Vector((fc_x.evaluate(frame), fc_y.evaluate(frame), fc_z.evaluate(frame)))

            # For each integer frame, get position (handling constant interp)
            for frame in range(start_frame, end_frame + 1):
                pos = get_position_for_frame(frame)

                # Get Z from motion path (already shrinkwrapped AND has z_offset applied)
                if settings.depth_interaction_enabled:
                    pos.z = get_z_from_motion_path(pos)
                    # Note: z_offset already included in motion path points, don't apply again
                else:
                    # Only apply z_offset when shrinkwrap disabled (motion path didn't apply it)
                    if z_offset > 0:
                        pos.z += z_offset

                pos.z += 0.01  # Render in front

                # Check if this is a keyframe or inbetween
                if frame in keyframe_frames:
                    # KEYFRAME: Store position for 3D dot (larger size)
                    keyframe_point_coords.append((pos.x, pos.y, pos.z))
                else:
                    # INBETWEEN: Store position for 3D dot (smaller size)
                    inbetween_point_coords.append((pos.x, pos.y, pos.z))

            # Build separate batches for keyframes and inbetweens (both use POINTS)
            point_shader = _get_point_shader()

            if keyframe_point_coords:
                _motion_path_keyframe_circles_batch = batch_for_shader(
                    point_shader, 'POINTS', {"pos": keyframe_point_coords}
                )
                log(f"TIMING: built {len(keyframe_point_coords)} keyframe dots", "DOTS")
            else:
                _motion_path_keyframe_circles_batch = None

            if inbetween_point_coords:
                _motion_path_inbetween_ticks_batch = batch_for_shader(
                    point_shader, 'POINTS', {"pos": inbetween_point_coords}
                )
                log(f"TIMING: built {len(inbetween_point_coords)} inbetween dots", "DOTS")
            else:
                _motion_path_inbetween_ticks_batch = None
        else:
            _motion_path_keyframe_circles_batch = None
            _motion_path_inbetween_ticks_batch = None

        # Direction Arrows and Keyframe Data - extract from F-curves
        if settings.motion_path_arrows_enabled or settings.motion_path_labels_enabled:
            _motion_path_keyframe_data = _extract_keyframe_data(gp_obj, fc_x, fc_y, fc_z, settings)

            if settings.motion_path_arrows_enabled and len(points) >= 2:
                # Use the path's actual last point for arrow position (already has shrinkwrap applied)
                # This ensures the arrow sits exactly on the path, not floating above it
                last_point = points[-1]
                second_last = points[-2]
                # Compute tangent from the last segment of the actual path
                tangent = Vector((
                    last_point.x - second_last.x,
                    last_point.y - second_last.y,
                    last_point.z - second_last.z
                ))
                # Create arrow data using actual path end point
                arrow_data = [(last_point, end_frame, tangent)]
                arrow_lines = _build_arrow_geometry(arrow_data, settings.motion_path_arrows_size)
                if arrow_lines:
                    # Use POLYLINE shader for controllable line width
                    stroke_shader = _get_stroke_shader()
                    _motion_path_arrows_batch = batch_for_shader(
                        stroke_shader, 'LINES', {"pos": arrow_lines}
                    )
                else:
                    _motion_path_arrows_batch = None
            else:
                _motion_path_arrows_batch = None
        else:
            _motion_path_keyframe_data = None
            _motion_path_arrows_batch = None

    path_points = _motion_path_cache
    if not path_points or len(path_points) < 2:
        return

    # Check if batches are available
    if _motion_path_line_batch is None:
        return

    # Lazy rebuild: if visualization enhancements are enabled but batches are None,
    # force a rebuild by marking dirty and returning (will render next frame)
    needs_rebuild = False
    # Timing chart uses keyframe circles + inbetween ticks (both must be checked)
    if settings.motion_path_spacing_dots_enabled:
        if _motion_path_keyframe_circles_batch is None and _motion_path_inbetween_ticks_batch is None:
            needs_rebuild = True
            log("LAZY_REBUILD: timing ticks enabled but batches are None", "DOTS")
    if settings.motion_path_arrows_enabled and _motion_path_arrows_batch is None:
        needs_rebuild = True
    if settings.motion_path_labels_enabled and _motion_path_keyframe_data is None:
        needs_rebuild = True

    if needs_rebuild:
        _motion_path_dirty = True
        log("LAZY_REBUILD: triggering rebuild", "DOTS")
        # Request a redraw so we rebuild next frame
        try:
            for area in bpy.context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
        except (RuntimeError, AttributeError):
            pass  # Context unavailable
        return

    # Check region before proceeding
    region = bpy.context.region
    if region is None:
        return

    # Set up GPU state
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(False)

    try:
        color = tuple(settings.motion_path_color)

        line_shader = _get_stroke_shader()  # Use cached shader
        line_shader.bind()  # Bind once
        line_shader.uniform_float("viewportSize", (region.width, region.height))
        line_shader.uniform_float("lineWidth", settings.motion_path_width)
        line_shader.uniform_float("color", color)

        # Draw path using cached batch (no recreation every frame!)
        _motion_path_line_batch.draw(line_shader)

        # Draw points using cached batch
        if settings.motion_path_show_points:
            point_shader = _get_point_shader()
            point_shader.bind()
            gpu.state.point_size_set(8.0)
            point_shader.uniform_float("color", color)
            _motion_path_point_batch.draw(point_shader)

        # Draw visualization enhancements using POLYLINE shader for line width control
        stroke_shader = _get_stroke_shader()

        # Draw Timing Chart: Inbetween ticks first, then keyframe circles on top
        log(f"DRAW_TIMING: enabled={settings.motion_path_spacing_dots_enabled} ticks={_motion_path_inbetween_ticks_batch is not None} circles={_motion_path_keyframe_circles_batch is not None}", "DOTS")
        if settings.motion_path_spacing_dots_enabled:
            # Use LESS_EQUAL depth test so ticks are occluded by mesh (like motion path line)
            gpu.state.depth_test_set('LESS_EQUAL')
            gpu.state.depth_mask_set(False)
            gpu.state.blend_set('ALPHA')

            # Draw inbetween dots (3D points) with tick color
            if _motion_path_inbetween_ticks_batch is not None:
                point_shader = _get_point_shader()
                point_shader.bind()
                tick_color = tuple(settings.motion_path_spacing_dots_color)
                point_shader.uniform_float("color", tick_color)
                # Point size scaled for visibility (setting is 1-10, points need ~4-40 pixels)
                gpu.state.point_size_set(settings.motion_path_spacing_dots_size * 4.0)
                _motion_path_inbetween_ticks_batch.draw(point_shader)

            # Draw keyframe dots (3D points) with keyframe color (on top, larger size)
            if _motion_path_keyframe_circles_batch is not None:
                point_shader = _get_point_shader()
                point_shader.bind()
                keyframe_color = tuple(settings.motion_path_keyframe_color)
                point_shader.uniform_float("color", keyframe_color)
                # Keyframe dots are larger than inbetweens (1.5x size)
                gpu.state.point_size_set(settings.motion_path_spacing_dots_size * 6.0)
                _motion_path_keyframe_circles_batch.draw(point_shader)

            gpu.state.depth_test_set('LESS_EQUAL')

        # Draw Direction Arrows (V/chevron)
        if settings.motion_path_arrows_enabled and _motion_path_arrows_batch is not None:
            # Use LESS_EQUAL depth test so arrows are occluded by mesh (like motion path)
            gpu.state.depth_test_set('LESS_EQUAL')
            gpu.state.blend_set('ALPHA')

            stroke_shader.bind()
            stroke_shader.uniform_float("viewportSize", (region.width, region.height))
            # Arrow size setting controls both V size and line thickness
            stroke_shader.uniform_float("lineWidth", settings.motion_path_arrows_size * 20.0)
            arrow_color = tuple(settings.motion_path_arrows_color)
            stroke_shader.uniform_float("color", arrow_color)
            _motion_path_arrows_batch.draw(stroke_shader)

            gpu.state.depth_test_set('LESS_EQUAL')

    finally:
        # Reset GPU state (always runs, even on exception)
        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('NONE')
        gpu.state.depth_mask_set(True)
        gpu.state.point_size_set(1.0)


def draw_motion_path_labels_callback():
    """
    POST_PIXEL draw callback - renders frame number labels at keyframe positions.

    This uses Blender's blf module for 2D text rendering in screen space.
    The labels are projected from 3D keyframe positions to 2D screen coords.
    """
    # Early exit if no keyframe data (check before accessing context)
    if _motion_path_keyframe_data is None or len(_motion_path_keyframe_data) == 0:
        return

    try:
        scene = bpy.context.scene
        region = bpy.context.region
        region_data = bpy.context.region_data
    except (RuntimeError, AttributeError, TypeError):
        return

    # Validate context objects
    if scene is None or region is None or region_data is None:
        return

    if not hasattr(scene, 'world_onion'):
        return

    settings = scene.world_onion

    if not settings.enabled:
        return

    if not settings.motion_path_enabled:
        return

    if not settings.motion_path_labels_enabled:
        return

    try:
        # Get label settings
        font_size = settings.motion_path_labels_size
        label_color = tuple(settings.motion_path_labels_color)

        # Font ID 0 is the default font
        font_id = 0

        # Set up blf
        blf.size(font_id, font_size)
        blf.color(font_id, label_color[0], label_color[1], label_color[2], label_color[3])

        # Draw label for each keyframe
        for world_pos, frame_num, tangent in _motion_path_keyframe_data:
            # Project 3D position to 2D screen coordinates
            screen_pos = location_3d_to_region_2d(region, region_data, world_pos)

            if screen_pos is None:
                continue  # Behind camera or outside viewport

            # Format label text
            label_text = f"F{frame_num}"

            # Offset label slightly from keyframe position to avoid overlap
            offset_x = 8
            offset_y = 8

            x = screen_pos.x + offset_x
            y = screen_pos.y + offset_y

            # Draw the label
            blf.position(font_id, x, y, 0)
            blf.draw(font_id, label_text)
    except (RuntimeError, AttributeError, TypeError):
        # Silently handle context errors during drawing
        pass


def register_draw_handlers():
    """Register GPU draw handlers."""
    global _draw_handler, _motion_path_handler, _motion_path_labels_handler
    global _motion_path_cache, _motion_path_cache_gp, _motion_path_dirty

    if _draw_handler is None:
        _draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_onion_callback, (), 'WINDOW', 'POST_VIEW'
        )

    if _motion_path_handler is None:
        _motion_path_handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_motion_path_callback, (), 'WINDOW', 'POST_VIEW'
        )

    # Frame labels use POST_PIXEL for screen-space 2D text rendering
    if _motion_path_labels_handler is None:
        _motion_path_labels_handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_motion_path_labels_callback, (), 'WINDOW', 'POST_PIXEL'
        )

    # Force motion path rebuild on registration (fixes reload not showing path)
    _motion_path_cache = None
    _motion_path_cache_gp = None
    _motion_path_dirty = True


def unregister_draw_handlers():
    """Unregister GPU draw handlers."""
    global _draw_handler, _motion_path_handler, _motion_path_labels_handler

    if _draw_handler is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, 'WINDOW')
        except ValueError:
            pass
        _draw_handler = None

    if _motion_path_handler is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_motion_path_handler, 'WINDOW')
        except ValueError:
            pass
        _motion_path_handler = None

    if _motion_path_labels_handler is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_motion_path_labels_handler, 'WINDOW')
        except ValueError:
            pass
        _motion_path_labels_handler = None
