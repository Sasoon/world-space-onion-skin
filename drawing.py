"""
GPU drawing callbacks for onion skin and anchor visualization.
"""

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from .cache import get_cache, get_active_gp
from .anchors import get_all_anchor_positions, get_all_locked_frames, get_lock_for_frame, get_interpolated_position


# Draw handler references
_draw_handler = None
_anchor_draw_handler = None
_motion_path_handler = None

# Motion path cache
_motion_path_cache = None  # List of (x, y, z) tuples
_motion_path_cache_gp = None  # GP object the cache is for
_motion_path_dirty = True


def invalidate_motion_path():
    """Mark motion path cache as dirty, triggering rebuild on next draw."""
    global _motion_path_dirty
    _motion_path_dirty = True


def adjust_path_points_to_mesh(points, scene):
    """Adjust path points to sit on mesh surface (shrinkwrap effect).

    For each point, casts a ray straight down to find any mesh surface,
    then adjusts Z to be on that surface plus a small offset for visibility.

    Args:
        points: List of (x, y, z) tuples
        scene: Current scene

    Returns:
        List of adjusted (x, y, z) tuples
    """
    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
    except (RuntimeError, AttributeError):
        return points

    # Small offset to ensure path sits visibly on top of mesh
    SURFACE_OFFSET = 0.01

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

    # Set up GPU state
    stroke_shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    fill_shader = gpu.shader.from_builtin('UNIFORM_COLOR')

    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(False)

    # Get viewport dimensions for line width
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

        # Compute offset to position strokes correctly on motion path
        offset = Vector((0, 0, 0))

        # Check if interpolation is enabled - use motion path positions
        if settings.interpolation_enabled:
            # Get current position on motion path for this frame
            interp_pos, interp_info = get_interpolated_position(gp_obj, frame)

            if interp_pos is None:
                # Frame is outside locked range - skip it
                continue

            # Compute anchor position of cached strokes (center X/Y, min Z)
            all_points = []
            for stroke_data in strokes:
                all_points.extend(stroke_data['points'])

            if all_points:
                sum_x, sum_y = 0.0, 0.0
                min_z = float('inf')
                for p in all_points:
                    sum_x += p.x
                    sum_y += p.y
                    min_z = min(min_z, p.z)
                n = len(all_points)
                cached_anchor = Vector((sum_x / n, sum_y / n, min_z))

                # Offset from cached anchor to motion path position
                offset = interp_pos - cached_anchor

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
        for stroke_data in strokes:
            fill_triangles = stroke_data.get('fill_triangles', [])
            if fill_triangles:
                # Apply offset for locked frames
                points = stroke_data['points']
                coords = [(p.x + offset.x, p.y + offset.y, p.z + offset.z) for p in points]

                # Build triangle vertex list from indices
                tri_coords = []
                for i, j, k in fill_triangles:
                    if i < len(coords) and j < len(coords) and k < len(coords):
                        tri_coords.extend([coords[i], coords[j], coords[k]])

                if tri_coords:
                    batch = batch_for_shader(fill_shader, 'TRIS', {"pos": tri_coords})
                    fill_shader.bind()
                    fill_shader.uniform_float("color", fill_color)
                    batch.draw(fill_shader)

        # PASS 2: Draw strokes on top
        stroke_shader.uniform_float("viewportSize", (region.width, region.height))
        stroke_shader.uniform_float("lineWidth", settings.line_width)

        for stroke_data in strokes:
            # Apply offset for locked frames
            points = stroke_data['points']
            coords = [(p.x + offset.x, p.y + offset.y, p.z + offset.z) for p in points]

            if len(coords) < 2:
                continue

            batch = batch_for_shader(stroke_shader, 'LINE_STRIP', {"pos": coords})
            stroke_shader.bind()
            stroke_shader.uniform_float("color", stroke_color)
            batch.draw(stroke_shader)

    # Reset GPU state
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
        if settings.skip_underscore and layer.name.startswith('_'):
            continue
        if settings.layer_filter and settings.layer_filter not in layer.name:
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
        # Context unavailable during render or background operations
        return

    if not hasattr(scene, 'world_onion'):
        return
    
    settings = scene.world_onion
    
    if not settings.enabled:
        return
    
    if not settings.anchor_enabled:
        return
    
    if not settings.anchor_show_indicators:
        return

    # Get active GP object (auto-detect)
    gp_obj = get_active_gp(bpy.context)
    if gp_obj is None:
        return

    # Get all anchor positions
    anchor_data = get_all_anchor_positions(gp_obj, settings)
    
    if not anchor_data:
        return
    
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    
    gpu.state.blend_set('ALPHA')
    gpu.state.point_size_set(12.0)
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(False)
    
    # Draw each anchor
    for pos, is_current in anchor_data:
        if is_current:
            # Current keyframe's anchor - bright yellow
            color = (1.0, 0.9, 0.0, 0.9)
        else:
            # Other anchors - dimmer
            color = (0.8, 0.6, 0.0, 0.4)
        
        # Draw as a point
        batch = batch_for_shader(shader, 'POINTS', {"pos": [pos]})
        shader.bind()
        shader.uniform_float("color", color)
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
        shader.bind()
        shader.uniform_float("color", color)
        batch.draw(shader)
    
    # Reset GPU state
    gpu.state.blend_set('NONE')
    gpu.state.depth_test_set('NONE')
    gpu.state.depth_mask_set(True)
    gpu.state.line_width_set(1.0)


def catmull_rom_spline(points, subdivisions):
    """Generate smooth curve through points using Catmull-Rom interpolation.

    Args:
        points: List of (x, y, z) tuples - control points
        subdivisions: Number of subdivisions between each pair of points

    Returns:
        List of (x, y, z) tuples - smoothed curve points
    """
    if len(points) < 2:
        return points

    if subdivisions <= 0:
        return points

    result = []

    # For Catmull-Rom, we need 4 points: p0, p1, p2, p3
    # We interpolate between p1 and p2
    # For endpoints, duplicate first/last points

    n = len(points)
    for i in range(n - 1):
        # Get 4 control points (duplicate at boundaries)
        p0 = points[max(0, i - 1)]
        p1 = points[i]
        p2 = points[min(n - 1, i + 1)]
        p3 = points[min(n - 1, i + 2)]

        # Add the starting point
        if i == 0:
            result.append(p1)

        # Interpolate between p1 and p2
        for j in range(1, subdivisions + 1):
            t = j / (subdivisions + 1)
            t2 = t * t
            t3 = t2 * t

            # Catmull-Rom matrix coefficients
            x = 0.5 * ((2 * p1[0]) +
                      (-p0[0] + p2[0]) * t +
                      (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t2 +
                      (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t3)

            y = 0.5 * ((2 * p1[1]) +
                      (-p0[1] + p2[1]) * t +
                      (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t2 +
                      (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t3)

            z = 0.5 * ((2 * p1[2]) +
                      (-p0[2] + p2[2]) * t +
                      (2*p0[2] - 5*p1[2] + 4*p2[2] - p3[2]) * t2 +
                      (-p0[2] + 3*p1[2] - 3*p2[2] + p3[2]) * t3)

            result.append((x, y, z))

        # Add endpoint of this segment
        result.append(p2)

    return result


def draw_motion_path_callback():
    """
    GPU draw callback - renders motion path connecting anchor positions.
    Shows the trajectory of movement across ALL frames with anchor data.
    - Solid line: segment starts from a LOCKED frame (strokes follow path)
    - Dashed line: segment starts from an UNLOCKED frame (strokes glued to camera)
    Uses caching for performance - only rebuilds when invalidated.
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
        # Rebuild cache - include ALL frames with anchor_world (locked AND unlocked)
        all_frames = get_all_locked_frames(gp_obj, include_data=True, include_unlocked_anchors=True)

        if len(all_frames) < 2:
            _motion_path_cache = None
            _motion_path_cache_gp = gp_obj
            _motion_path_dirty = False
            return

        # Sort by frame number and extract positions with lock status
        sorted_frames = sorted(all_frames.items(), key=lambda x: int(x[0]))

        # Cache structure: list of (position, is_locked) tuples
        path_data = []
        for frame_str, lock_data in sorted_frames:
            if 'anchor_world' in lock_data:
                pos = tuple(lock_data['anchor_world'])
                is_locked = lock_data.get('world_locked', False)
                path_data.append((pos, is_locked))

        _motion_path_cache = path_data if len(path_data) >= 2 else None
        _motion_path_cache_gp = gp_obj
        _motion_path_dirty = False

    path_data = _motion_path_cache
    if path_data is None or len(path_data) < 2:
        return

    # Extract just positions for smoothing/depth
    path_points = [p[0] for p in path_data]

    # Apply smoothing if enabled
    smoothing = settings.motion_path_smoothing

    # Collect shrinkwrapped anchor positions for point drawing
    shrinkwrapped_locked_points = []
    shrinkwrapped_unlocked_points = []

    if smoothing > 0:
        # For smoothing, we need to track which segment each smoothed point belongs to
        # Build segments with their lock status, then smooth each segment
        solid_segments = []  # List of point lists for solid (locked) segments
        dashed_segments = []  # List of point lists for dashed (unlocked) segments

        for i in range(len(path_data) - 1):
            start_pos, start_locked = path_data[i]
            end_pos, end_locked = path_data[i + 1]

            # Generate smoothed points for this segment
            if len(path_points) >= 2:
                # Get control points for Catmull-Rom
                p0 = path_points[max(0, i - 1)]
                p1 = start_pos
                p2 = end_pos
                p3 = path_points[min(len(path_points) - 1, i + 2)]

                segment_points = [p1]
                for j in range(1, smoothing + 1):
                    t = j / (smoothing + 1)
                    pt = catmull_rom_point(p0, p1, p2, p3, t)
                    segment_points.append(pt)
                segment_points.append(p2)
            else:
                segment_points = [start_pos, end_pos]

            # Apply depth interaction if enabled
            if settings.depth_interaction_enabled:
                segment_points = adjust_path_points_to_mesh(segment_points, scene)

            # Collect shrinkwrapped anchor positions
            if start_locked:
                shrinkwrapped_locked_points.append(segment_points[0])
                solid_segments.append(segment_points)
            else:
                shrinkwrapped_unlocked_points.append(segment_points[0])
                dashed_segments.append(segment_points)

            # Last anchor point (from final segment's end)
            if i == len(path_data) - 2:
                if end_locked:
                    shrinkwrapped_locked_points.append(segment_points[-1])
                else:
                    shrinkwrapped_unlocked_points.append(segment_points[-1])
    else:
        # No smoothing - just separate into segments by lock status
        solid_segments = []
        dashed_segments = []

        for i in range(len(path_data) - 1):
            start_pos, start_locked = path_data[i]
            end_pos, end_locked = path_data[i + 1]
            segment_points = [start_pos, end_pos]

            # Apply depth interaction if enabled
            if settings.depth_interaction_enabled:
                segment_points = adjust_path_points_to_mesh(segment_points, scene)

            # Collect shrinkwrapped anchor positions
            if start_locked:
                shrinkwrapped_locked_points.append(segment_points[0])
                solid_segments.append(segment_points)
            else:
                shrinkwrapped_unlocked_points.append(segment_points[0])
                dashed_segments.append(segment_points)

            # Last anchor point (from final segment's end)
            if i == len(path_data) - 2:
                if end_locked:
                    shrinkwrapped_locked_points.append(segment_points[-1])
                else:
                    shrinkwrapped_unlocked_points.append(segment_points[-1])

    # Set up GPU state
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(False)

    color = tuple(settings.motion_path_color)
    # Inactive (unlocked) color: red with same alpha
    inactive_color = (0.9, 0.2, 0.2, color[3] * 0.8)
    region = bpy.context.region

    line_shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    line_shader.uniform_float("viewportSize", (region.width, region.height))
    line_shader.uniform_float("lineWidth", settings.motion_path_width)

    # Draw solid segments (active/locked) - normal color
    for segment in solid_segments:
        if len(segment) >= 2:
            batch = batch_for_shader(line_shader, 'LINE_STRIP', {"pos": segment})
            line_shader.bind()
            line_shader.uniform_float("color", color)
            batch.draw(line_shader)

    # Draw inactive segments (unlocked) - red color
    for segment in dashed_segments:
        if len(segment) >= 2:
            batch = batch_for_shader(line_shader, 'LINE_STRIP', {"pos": segment})
            line_shader.bind()
            line_shader.uniform_float("color", inactive_color)
            batch.draw(line_shader)

    # Draw points at each anchor if enabled (using shrinkwrapped positions)
    if settings.motion_path_show_points:
        point_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.point_size_set(8.0)

        # Locked points - full color (shrinkwrapped)
        if shrinkwrapped_locked_points:
            batch = batch_for_shader(point_shader, 'POINTS', {"pos": shrinkwrapped_locked_points})
            point_shader.bind()
            point_shader.uniform_float("color", color)
            batch.draw(point_shader)

        # Unlocked points - red color (shrinkwrapped)
        if shrinkwrapped_unlocked_points:
            batch = batch_for_shader(point_shader, 'POINTS', {"pos": shrinkwrapped_unlocked_points})
            point_shader.bind()
            point_shader.uniform_float("color", inactive_color)
            batch.draw(point_shader)

    # Reset GPU state
    gpu.state.blend_set('NONE')
    gpu.state.depth_test_set('NONE')
    gpu.state.depth_mask_set(True)


def catmull_rom_point(p0, p1, p2, p3, t):
    """Calculate a single point on a Catmull-Rom spline."""
    t2 = t * t
    t3 = t2 * t

    x = 0.5 * ((2 * p1[0]) +
               (-p0[0] + p2[0]) * t +
               (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
               (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
    y = 0.5 * ((2 * p1[1]) +
               (-p0[1] + p2[1]) * t +
               (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
               (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
    z = 0.5 * ((2 * p1[2]) +
               (-p0[2] + p2[2]) * t +
               (2 * p0[2] - 5 * p1[2] + 4 * p2[2] - p3[2]) * t2 +
               (-p0[2] + 3 * p1[2] - 3 * p2[2] + p3[2]) * t3)

    return (x, y, z)


def register_draw_handlers():
    """Register GPU draw handlers."""
    global _draw_handler, _anchor_draw_handler, _motion_path_handler

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
