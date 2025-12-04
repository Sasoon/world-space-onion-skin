"""
GPU drawing callbacks for onion skin and anchor visualization.
"""

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from .cache import get_cache, get_active_gp
from .anchors import get_all_anchor_positions


# Draw handler references
_draw_handler = None
_anchor_draw_handler = None


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
                points = stroke_data['points']
                # Build triangle vertex list from indices
                tri_coords = []
                for i, j, k in fill_triangles:
                    if i < len(points) and j < len(points) and k < len(points):
                        tri_coords.extend([
                            (points[i].x, points[i].y, points[i].z),
                            (points[j].x, points[j].y, points[j].z),
                            (points[k].x, points[k].y, points[k].z),
                        ])

                if tri_coords:
                    batch = batch_for_shader(fill_shader, 'TRIS', {"pos": tri_coords})
                    fill_shader.bind()
                    fill_shader.uniform_float("color", fill_color)
                    batch.draw(fill_shader)

        # PASS 2: Draw strokes on top
        stroke_shader.uniform_float("viewportSize", (region.width, region.height))
        stroke_shader.uniform_float("lineWidth", settings.line_width)

        for stroke_data in strokes:
            points = stroke_data['points']
            if len(points) < 2:
                continue

            coords = [(p.x, p.y, p.z) for p in points]

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


def register_draw_handlers():
    """Register GPU draw handlers."""
    global _draw_handler, _anchor_draw_handler
    
    if _draw_handler is None:
        _draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_onion_callback, (), 'WINDOW', 'POST_VIEW'
        )
    
    if _anchor_draw_handler is None:
        _anchor_draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_anchor_callback, (), 'WINDOW', 'POST_VIEW'
        )


def unregister_draw_handlers():
    """Unregister GPU draw handlers."""
    global _draw_handler, _anchor_draw_handler

    if _draw_handler is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, 'WINDOW')
        except ValueError:
            # Handler already removed or not registered
            pass
        _draw_handler = None

    if _anchor_draw_handler is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_anchor_draw_handler, 'WINDOW')
        except ValueError:
            # Handler already removed or not registered
            pass
        _anchor_draw_handler = None
