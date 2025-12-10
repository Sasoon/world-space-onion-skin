"""
GPU drawing callbacks for onion skin and anchor visualization.
"""

import bisect

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from .cache import get_cache, get_active_gp
from .anchors import get_all_anchor_positions
from .transforms import SURFACE_OFFSET, catmull_rom_point, adjust_obj_to_surface
from .debug_log import log, log_onion_draw, log_bake, log_cursor


# Draw handler references
_draw_handler = None
_anchor_draw_handler = None
_motion_path_handler = None

# Motion path cache
_motion_path_cache = None  # List of Vector positions
_motion_path_cache_gp = None  # GP object the cache is for
_motion_path_dirty = True
# Motion path GPU batch cache (avoids recreation every redraw)
_motion_path_coords = None  # Pre-built tuple coords for GPU
_motion_path_line_batch = None
_motion_path_point_batch = None

# Baked shrinkwrap offsets - computed ONCE when shrinkwrap enabled or animation changes
# Structure: {frame: z_offset}  - stores the Z offset from raycast
_baked_shrinkwrap_offsets = {}
_baked_offset_valid = False  # True after successful bake

# Onion skin GPU batch cache - CRITICAL for performance near camera
# Structure: {frame: {'fill_batches': [batch, ...], 'stroke_batches': [batch, ...]}}
# Batches are keyed by frame and built once, reused on every redraw
_onion_batch_cache = {}
_onion_cache_z_offset = None  # Track z_offset to detect changes
_onion_cache_gp = None  # Track GP object to detect changes

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


def invalidate_motion_path():
    """Mark motion path cache as dirty, triggering rebuild on next draw."""
    global _motion_path_dirty, _motion_path_line_batch, _motion_path_point_batch, _motion_path_coords
    _motion_path_dirty = True
    # Also invalidate GPU batch cache
    _motion_path_line_batch = None
    _motion_path_point_batch = None
    _motion_path_coords = None


def invalidate_onion_batch_cache():
    """Clear all cached onion skin GPU batches. Call when stroke data changes."""
    global _onion_batch_cache, _onion_cache_z_offset, _onion_cache_gp
    global _keyframe_cache, _keyframe_cache_gp
    _onion_batch_cache = {}
    _onion_cache_z_offset = None
    _onion_cache_gp = None
    # Also clear keyframe cache since keyframes may have changed
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
    """
    def _do_setup():
        """Inner function to do the actual driver setup."""
        # Remove existing driver if present (avoid duplicates)
        try:
            gp_obj.driver_remove("delta_location", 2)  # index 2 = Z
        except:
            pass  # No driver existed

        # Ensure namespace function is registered
        register_driver_namespace()

        # Add new driver
        fcurve = gp_obj.driver_add("delta_location", 2)
        driver = fcurve.driver
        driver.type = 'SCRIPTED'

        # Add frame variable that reads current frame from scene
        var = driver.variables.new()
        var.name = "frame"
        var.type = 'SINGLE_PROP'
        var.targets[0].id_type = 'SCENE'
        var.targets[0].id = bpy.context.scene
        var.targets[0].data_path = "frame_current"

        # Expression calls our registered namespace function
        driver.expression = "shrinkwrap_offset(frame)"

        log("Setup driver on delta_location.z using namespace function", "BAKE")

    # Try direct setup first
    try:
        _do_setup()
    except AttributeError as e:
        if "Writing to ID classes" in str(e):
            # Context doesn't allow ID modifications - defer to timer
            log("Driver setup deferred to timer (context restriction)", "BAKE")
            def _deferred_setup():
                try:
                    _do_setup()
                except Exception as ex:
                    log(f"Deferred driver setup failed: {ex}", "ERROR")
                return None  # Don't repeat
            bpy.app.timers.register(_deferred_setup, first_interval=0.1)
        else:
            raise  # Re-raise unexpected errors


def remove_shrinkwrap_driver(gp_obj):
    """
    Remove the shrinkwrap driver when feature is disabled.
    Also resets delta_location.z to 0.
    """
    if gp_obj is None:
        return

    try:
        gp_obj.driver_remove("delta_location", 2)
        log("Removed shrinkwrap driver from delta_location.z", "BAKE")
    except:
        pass  # Driver didn't exist

    # Reset delta_location to avoid stuck offset
    try:
        gp_obj.delta_location.z = 0.0
    except:
        pass


def bake_shrinkwrap_offsets(gp_obj, settings, scene):
    """
    Pre-compute shrinkwrap Z offsets for ENTIRE animation range.

    Called when shrinkwrap is enabled or animation changes.
    MUST be called from a context where raycast is reliable (not during playback).

    This eliminates runtime raycast dependency - during playback we just
    read from this lookup table.
    """
    global _baked_shrinkwrap_offsets, _baked_offset_valid

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
                offset = (location.z + SURFACE_OFFSET) - z
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

    # v8.1: Setup driver that calls our namespace function
    # NO KEYFRAMES - just stores in dict and driver looks it up!
    _setup_shrinkwrap_driver(gp_obj)

    return count


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


def adjust_path_points_to_mesh(points, scene):
    """Adjust path points to sit on mesh surface (shrinkwrap effect)."""
    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
    except (RuntimeError, AttributeError):
        return points

    adjusted_points = []
    for pt in points:
        x, y, z = pt

        # Cast ray straight down from high above
        ray_origin = Vector((x, y, z + 1000))
        ray_dir = Vector((0, 0, -1))

        hit, location, normal, index, hit_obj, matrix = scene.ray_cast(
            depsgraph, ray_origin, ray_dir
        )

        if hit:
            # Adjust Z to be on mesh surface + small offset
            new_z = location.z + SURFACE_OFFSET
            adjusted_points.append((x, y, new_z))
        else:
            # No hit - keep original
            adjusted_points.append(pt)

    return adjusted_points


def get_draw_handlers():
    """Get current draw handler references."""
    return _draw_handler, _anchor_draw_handler


def _build_onion_batches_for_frame(frame, strokes, z_offset, fill_shader, stroke_shader):
    """
    Build and cache GPU batches for a single onion skin frame.
    Returns dict with 'fill_batches' and 'stroke_batches' lists.
    """
    fill_batches = []
    stroke_batches = []

    for stroke_data in strokes:
        points = stroke_data['points']
        if len(points) < 2:
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

        # Draw each cached frame
        for frame_offset, frame in frames_to_show:
            if frame not in cache:
                continue
            if frame == current_frame:
                continue

            strokes = cache[frame]
            if not strokes:
                continue

            # Calculate z_offset for this frame (includes per-frame baked offset)
            z_offset = base_z_offset
            if settings.depth_interaction_enabled and is_bake_valid():
                baked_offset = get_baked_offset(frame)
                if baked_offset is not None:
                    z_offset += baked_offset

            # Cache key includes frame and z_offset (rounded to avoid float precision issues)
            cache_key = (frame, round(z_offset, 4))

            # Check if batches are cached for this frame
            if cache_key not in _onion_batch_cache:
                # Build and cache batches
                _onion_batch_cache[cache_key] = _build_onion_batches_for_frame(
                    frame, strokes, z_offset, fill_shader, stroke_shader
                )

            cached_batches = _onion_batch_cache[cache_key]

            # DEBUG: Log onion draw
            log_onion_draw(current_frame, frame, z_offset, len(strokes))

            # Calculate color based on before/after
            if frame < current_frame:
                base_color = settings.color_before
            else:
                base_color = settings.color_after

            # Calculate opacity with falloff
            abs_offset = abs(frame_offset)
            max_offset = max(settings.frames_before, settings.frames_after, 1)
            falloff_factor = 1.0 - (abs_offset / max_offset) * settings.falloff if settings.falloff > 0 else 1.0

            # PASS 1: Draw fills (set color, draw all cached fill batches)
            fill_alpha = settings.fill_opacity * max(0.1, falloff_factor)
            fill_color = (base_color[0], base_color[1], base_color[2], fill_alpha)

            fill_shader.bind()
            fill_shader.uniform_float("color", fill_color)
            for batch in cached_batches['fill_batches']:
                batch.draw(fill_shader)

            # PASS 2: Draw strokes (set color, draw all cached stroke batches)
            stroke_alpha = settings.opacity * max(0.1, falloff_factor)
            stroke_color = (base_color[0], base_color[1], base_color[2], stroke_alpha)

            stroke_shader.bind()
            stroke_shader.uniform_float("viewportSize", (region.width, region.height))
            stroke_shader.uniform_float("lineWidth", settings.line_width)
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


def draw_anchor_callback():
    """
    GPU draw callback - renders anchor position indicators.
    """
    try:
        scene = bpy.context.scene
    except (RuntimeError, AttributeError):
        return

    if not hasattr(scene, 'world_onion'):
        return
    
    settings = scene.world_onion
    
    if not settings.enabled:
        return
    
    if not settings.anchor_enabled:
        return

    # Get active GP object (auto-detect)
    gp_obj = get_active_gp(bpy.context)
    if gp_obj is None:
        return

    # Get all anchor positions
    anchor_data = get_all_anchor_positions(gp_obj, settings)

    if not anchor_data:
        return

    shader = _get_fill_shader()  # Use cached shader

    gpu.state.blend_set('ALPHA')
    gpu.state.point_size_set(12.0)
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(False)

    shader.bind()  # Bind once at start

    # Draw each anchor
    for pos, is_current in anchor_data:
        if is_current:
            # Current keyframe's anchor - bright yellow
            color = (1.0, 0.9, 0.0, 0.9)
        else:
            # Other anchors - dimmer
            color = (0.8, 0.6, 0.0, 0.4)

        # Draw as a point
        shader.uniform_float("color", color)
        batch = batch_for_shader(shader, 'POINTS', {"pos": [pos]})
        batch.draw(shader)
        
        # Draw a small cross for visibility
        size = 0.05
        cross_lines = [
            (pos.x - size, pos.y, pos.z),
            (pos.x + size, pos.y, pos.z),
            (pos.x, pos.y - size, pos.z),
            (pos.x, pos.y + size, pos.z),
            (pos.x, pos.y, pos.z - size),
            (pos.x, pos.y, pos.z + size),
        ]
        batch = batch_for_shader(shader, 'LINES', {"pos": cross_lines})
        gpu.state.line_width_set(2.0 if is_current else 1.0)
        shader.uniform_float("color", color)
        batch.draw(shader)
    
    # Reset GPU state
    gpu.state.blend_set('NONE')
    gpu.state.depth_test_set('NONE')
    gpu.state.depth_mask_set(True)
    gpu.state.line_width_set(1.0)
    gpu.state.point_size_set(1.0)


def draw_motion_path_callback():
    """
    GPU draw callback - renders motion path connecting object locations.
    Samples F-Curves to draw path.
    Respects depth interaction (shrinkwrap) if enabled.
    """
    global _motion_path_cache, _motion_path_cache_gp, _motion_path_dirty
    global _motion_path_coords, _motion_path_line_batch, _motion_path_point_batch

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
            return

        # Sample path
        # Determine step based on duration to avoid too many points
        duration = end_frame - start_frame
        step = max(1, duration // 100)  # Limit to ~100 points
        
        points = []
        
        # Save current frame
        original_frame = scene.frame_current
        
        # Sample frames
        # Note: This is slow-ish but necessary to see evaluated positions
        # Faster way is evaluation via fcurves directly but that ignores constraints?
        # Actually we want to visualize the F-Curve + Raycast adjustment.
        # Since we adjust in frame_change_post, standard evaluation doesn't show it unless we run the handler logic here.
        
        # Optimization: Just evaluate F-Curves for raw position, then apply raycast logic
        # This avoids scene.frame_set() overhead
        
        # Get location fcurves
        fcurves = gp_obj.animation_data.action.fcurves
        fc_x = fcurves.find('location', index=0)
        fc_y = fcurves.find('location', index=1)
        fc_z = fcurves.find('location', index=2)
        
        if not fc_x or not fc_y or not fc_z:
             # Fallback to frame_set if incomplete
             for f in range(start_frame, end_frame + 1, step):
                 scene.frame_set(f)
                 # Our handler runs on frame_set, so gp_obj.location is adjusted
                 points.append(gp_obj.location.copy())
             scene.frame_set(original_frame)
        else:
             # Evaluate F-Curves + apply shrinkwrap using BAKED offsets (v7 BULLETPROOF)
             for f in range(start_frame, end_frame + 1, step):
                 x = fc_x.evaluate(f)
                 y = fc_y.evaluate(f)
                 z = fc_z.evaluate(f)

                 pos = Vector((x, y, z))

                 if settings.depth_interaction_enabled:
                     # v7.1: Use ONLY baked offsets - no runtime raycast
                     baked_offset = get_baked_offset(f)
                     if baked_offset is not None:
                         pos.z += baked_offset
                     # No fallback - if not baked, just use raw F-curve position

                 points.append(pos)

        _motion_path_cache = points
        _motion_path_cache_gp = gp_obj
        _motion_path_dirty = False

        # Build GPU batches once and cache them
        coords = [(p.x, p.y, p.z) for p in points]

        # Apply Catmull-Rom smoothing if enabled
        if settings.motion_path_smoothing > 0 and len(coords) >= 4:
            smoothed = []
            subdivisions = settings.motion_path_smoothing

            for i in range(len(coords) - 1):
                # Get 4 control points (clamp at boundaries)
                p0 = coords[max(0, i - 1)]
                p1 = coords[i]
                p2 = coords[min(len(coords) - 1, i + 1)]
                p3 = coords[min(len(coords) - 1, i + 2)]

                # Add start point
                smoothed.append(p1)

                # Add interpolated points between p1 and p2
                for j in range(1, subdivisions + 1):
                    t = j / (subdivisions + 1)
                    pt = catmull_rom_point(p0, p1, p2, p3, t)
                    smoothed.append((pt.x, pt.y, pt.z))

            # Add final point
            smoothed.append(coords[-1])
            coords = smoothed

        _motion_path_coords = coords
        _motion_path_line_batch = batch_for_shader(
            _get_stroke_shader(), 'LINE_STRIP', {"pos": _motion_path_coords}
        )
        # Points batch uses original unsmoothed positions for keyframe markers
        _motion_path_point_batch = batch_for_shader(
            _get_fill_shader(), 'POINTS', {"pos": [(p.x, p.y, p.z) for p in points]}
        )

    path_points = _motion_path_cache
    if not path_points or len(path_points) < 2:
        return

    # Check if batches are available
    if _motion_path_line_batch is None:
        return

    # Set up GPU state
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(False)

    color = tuple(settings.motion_path_color)
    region = bpy.context.region

    line_shader = _get_stroke_shader()  # Use cached shader
    line_shader.bind()  # Bind once
    line_shader.uniform_float("viewportSize", (region.width, region.height))
    line_shader.uniform_float("lineWidth", settings.motion_path_width)
    line_shader.uniform_float("color", color)

    # Draw path using cached batch (no recreation every frame!)
    _motion_path_line_batch.draw(line_shader)

    # Draw points using cached batch
    if settings.motion_path_show_points:
        point_shader = _get_fill_shader()
        point_shader.bind()
        gpu.state.point_size_set(8.0)
        point_shader.uniform_float("color", color)
        _motion_path_point_batch.draw(point_shader)

    # Reset GPU state
    gpu.state.blend_set('NONE')
    gpu.state.depth_test_set('NONE')
    gpu.state.depth_mask_set(True)
    gpu.state.point_size_set(1.0)


def register_draw_handlers():
    """Register GPU draw handlers."""
    global _draw_handler, _anchor_draw_handler, _motion_path_handler
    global _motion_path_cache, _motion_path_cache_gp, _motion_path_dirty

    if _draw_handler is None:
        _draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_onion_callback, (), 'WINDOW', 'POST_VIEW'
        )

    if _anchor_draw_handler is None:
        _anchor_draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_anchor_callback, (), 'WINDOW', 'POST_VIEW'
        )

    if _motion_path_handler is None:
        _motion_path_handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_motion_path_callback, (), 'WINDOW', 'POST_VIEW'
        )

    # Force motion path rebuild on registration (fixes reload not showing path)
    _motion_path_cache = None
    _motion_path_cache_gp = None
    _motion_path_dirty = True


def unregister_draw_handlers():
    """Unregister GPU draw handlers."""
    global _draw_handler, _anchor_draw_handler, _motion_path_handler

    if _draw_handler is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, 'WINDOW')
        except ValueError:
            pass
        _draw_handler = None

    if _anchor_draw_handler is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_anchor_draw_handler, 'WINDOW')
        except ValueError:
            pass
        _anchor_draw_handler = None

    if _motion_path_handler is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_motion_path_handler, 'WINDOW')
        except ValueError:
            pass
        _motion_path_handler = None
