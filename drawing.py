"""
GPU drawing callbacks for onion skin and anchor visualization.
"""

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from .cache import get_cache, get_active_gp
from .anchors import get_all_anchor_positions
from .transforms import SURFACE_OFFSET, catmull_rom_point, adjust_obj_to_surface


# Draw handler references
_draw_handler = None
_anchor_draw_handler = None
_motion_path_handler = None

# Motion path cache
_motion_path_cache = None  # List of (x, y, z) tuples
_motion_path_cache_gp = None  # GP object the cache is for
_motion_path_dirty = True

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
    global _motion_path_dirty
    _motion_path_dirty = True


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


def draw_onion_callback():
    """
    GPU draw callback - renders cached onion skin strokes.
    Called every viewport redraw.
    """
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

    # Set up GPU state (use cached shaders for performance)
    stroke_shader = _get_stroke_shader()
    fill_shader = _get_fill_shader()

    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(False)

    try:
        # Get viewport dimensions for line width
        region = bpy.context.region

        # Bind shaders once for the entire draw operation
        stroke_shader.bind()
        stroke_shader.uniform_float("viewportSize", (region.width, region.height))
        stroke_shader.uniform_float("lineWidth", settings.line_width)

        # Draw each cached frame
        for frame_offset, frame in frames_to_show:
            if frame not in cache:
                continue
            if frame == current_frame:
                continue

            strokes = cache[frame]
            if not strokes:
                continue

            # Calculate color based on before/after
            if frame < current_frame:
                base_color = settings.color_before
            else:
                base_color = settings.color_after

            # Calculate opacity with falloff
            abs_offset = abs(frame_offset)
            max_offset = max(settings.frames_before, settings.frames_after, 1)

            if settings.falloff > 0:
                falloff_factor = 1.0 - (abs_offset / max_offset) * settings.falloff
            else:
                falloff_factor = 1.0

            # Fill uses fill_opacity setting
            fill_alpha = settings.fill_opacity * max(0.1, falloff_factor)
            fill_color = (base_color[0], base_color[1], base_color[2], fill_alpha)

            # Stroke uses opacity setting
            stroke_alpha = settings.opacity * max(0.1, falloff_factor)
            stroke_color = (base_color[0], base_color[1], base_color[2], stroke_alpha)

            # PASS 1: Draw fills first (underneath strokes)
            # Bind fill shader once per frame with current color
            fill_shader.bind()
            fill_shader.uniform_float("color", fill_color)

            for stroke_data in strokes:
                fill_triangles = stroke_data.get('fill_triangles', [])
                if fill_triangles:
                    points = stroke_data['points']
                    # Using raw cached world points
                    coords = [(p.x, p.y, p.z) for p in points]

                    # Build triangle vertex list from indices
                    tri_coords = []
                    for i, j, k in fill_triangles:
                        if i < len(coords) and j < len(coords) and k < len(coords):
                            tri_coords.extend([coords[i], coords[j], coords[k]])

                    if tri_coords:
                        batch = batch_for_shader(fill_shader, 'TRIS', {"pos": tri_coords})
                        batch.draw(fill_shader)

            # PASS 2: Draw strokes on top
            # Re-bind stroke shader with current color (viewport/lineWidth already set)
            stroke_shader.bind()
            stroke_shader.uniform_float("viewportSize", (region.width, region.height))
            stroke_shader.uniform_float("lineWidth", settings.line_width)
            stroke_shader.uniform_float("color", stroke_color)

            for stroke_data in strokes:
                points = stroke_data['points']
                coords = [(p.x, p.y, p.z) for p in points]

                if len(coords) < 2:
                    continue

                batch = batch_for_shader(stroke_shader, 'LINE_STRIP', {"pos": coords})
                batch.draw(stroke_shader)

    finally:
        # Reset GPU state (always runs, even if exception occurs)
        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('NONE')
        gpu.state.depth_mask_set(True)


def get_keyframe_based_frames(gp_obj, settings, current_frame):
    """Get frames to show in GP Keyframes mode."""
    frames_to_show = []

    if gp_obj is None:
        return frames_to_show

    # Collect all unique keyframe numbers
    all_keyframes = set()
    for layer in gp_obj.data.layers:
        if layer.hide:
            continue
        
        for kf in layer.frames:
            all_keyframes.add(kf.frame_number)
    
    sorted_keyframes = sorted(all_keyframes)
    
    if not sorted_keyframes:
        return frames_to_show
    
    # Find current keyframe index
    current_idx = None
    for i, kf in enumerate(sorted_keyframes):
        if kf == current_frame:
            current_idx = i
            break
        elif kf > current_frame:
            current_idx = i - 1 if i > 0 else 0
            break
    
    if current_idx is None:
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
             # Evaluate F-Curves + apply shrinkwrap logic manually
             for f in range(start_frame, end_frame + 1, step):
                 x = fc_x.evaluate(f)
                 y = fc_y.evaluate(f)
                 z = fc_z.evaluate(f)
                 
                 pos = Vector((x, y, z))
                 
                 if settings.depth_interaction_enabled:
                     # Manual raycast logic (same as adjust_obj_to_surface)
                     ray_origin = Vector((x, y, z + 1000.0))
                     ray_dir = Vector((0, 0, -1))
                     # We need depsgraph. scene.ray_cast needs it.
                     # Ideally we use the evaluated depsgraph from context
                     try:
                         depsgraph = bpy.context.evaluated_depsgraph_get()
                         hit, location, _, _, hit_obj, _ = scene.ray_cast(depsgraph, ray_origin, ray_dir)
                         if hit and hit_obj != gp_obj:
                             pos.z = location.z + SURFACE_OFFSET
                     except:
                         pass
                 
                 points.append(pos)

        _motion_path_cache = points
        _motion_path_cache_gp = gp_obj
        _motion_path_dirty = False

    path_points = _motion_path_cache
    if not path_points or len(path_points) < 2:
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

    # Draw path
    # Convert Vectors to tuples for batch
    coords = [(p.x, p.y, p.z) for p in path_points]
    
    batch = batch_for_shader(line_shader, 'LINE_STRIP', {"pos": coords})
    batch.draw(line_shader)

    # Draw points
    if settings.motion_path_show_points:
        point_shader = _get_fill_shader()
        point_shader.bind()
        gpu.state.point_size_set(8.0)
        point_shader.uniform_float("color", color)
        batch = batch_for_shader(point_shader, 'POINTS', {"pos": coords})
        batch.draw(point_shader)

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
