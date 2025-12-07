"""
Timeline/Dopesheet drawing for world-lock indicators.

Draws sparklines (horizontal lines) showing the effective duration of each
world-locked keyframe in the Dopesheet editor.
"""

import bpy
import gpu
from gpu_extras.batch import batch_for_shader

from .cache import get_active_gp
from .anchors import get_all_locked_frames


_timeline_draw_handler = None

# Cached shader (lazy initialized)
_timeline_shader = None


def _get_timeline_shader():
    """Get cached UNIFORM_COLOR shader for timeline drawing."""
    global _timeline_shader
    if _timeline_shader is None:
        _timeline_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    return _timeline_shader


def get_lane_center_y(context, lane_index):
    """Return the view-space Y coordinate for the centre of a dopesheet lane.
    
    This uses absolute view coordinates where:
    - Y=0 is top of timeline (ruler)
    - Y goes negative downwards
    - Ruler is ~24 units high
    - Channels are ~22 units high
    
    All scaled by system UI scale.
    """
    prefs = getattr(context, "preferences", None)
    system_prefs = getattr(prefs, "system", None) if prefs else None
    ui_scale = getattr(system_prefs, "ui_scale", 1.0) if system_prefs else 1.0
    
    # Ruler height is roughly 24px at 1.0 scale (23px was slightly too high)
    ruler_height = 24.0 * ui_scale
    
    # Channel height (Lane height) is roughly 18-20px. 
    # Storyliner uses 18. Standard widget is 20.
    # Let's use 19.0 as a safer median or sticking to 18 if user reports it being too low.
    # If previous 22 was too high (offset 5 instead of 4), shrinking to 18/19 should fix it.
    channel_step = 18.0 * ui_scale * 1.0 # Scale by preferences if needed, but ui_scale covers it.
    
    # Padding/Start offset?
    # First lane starts immediately after ruler.
    
    # Distance to top of the requested lane
    lane_top_y = -ruler_height - (lane_index * channel_step)

    # Center of the lane (middle of the channel step)
    center_y = lane_top_y - (channel_step / 2.0)

    return center_y


def draw_timeline_callback():
    """GPU draw callback for world-lock indicators in Dopesheet."""
    try:
        context = bpy.context
        scene = context.scene
    except (RuntimeError, AttributeError):
        # Context unavailable during render or background operations
        return

    if not hasattr(scene, 'world_onion'):
        return

    settings = scene.world_onion
    if not settings.enabled or not settings.show_timeline_indicator:
        return

    gp_obj = get_active_gp(context)
    if gp_obj is None or gp_obj.data is None:
        return

    # Find Dopesheet region and draw
    for area in context.screen.areas:
        if area.type == 'DOPESHEET_EDITOR':
            for region in area.regions:
                if region.type == 'WINDOW':
                    draw_lock_indicators(context, area, region, gp_obj, settings)


def get_lock_spans(gp_obj):
    """Get spans showing each locked keyframe's effective duration.

    Uses object-level lock detection (all layers share lock state).
    Each locked keyframe spans from its frame to the next keyframe
    (regardless of whether that keyframe is locked or not).
    This shows "this lock is in effect for these frames".

    Returns list of (start_frame, end_frame) tuples.
    """
    # Get all keyframes across all layers
    all_keyframes = set()
    for layer in gp_obj.data.layers:
        for kf in layer.frames:
            all_keyframes.add(kf.frame_number)

    all_keyframes = sorted(all_keyframes)

    if not all_keyframes:
        return []

    # Get object-level locked frames
    locked_frames = set(get_all_locked_frames(gp_obj))

    if not locked_frames:
        return []

    spans = []
    for i, frame in enumerate(all_keyframes):
        if frame in locked_frames:
            # Find next keyframe (or use current frame if last)
            if i + 1 < len(all_keyframes):
                next_frame = all_keyframes[i + 1]
            else:
                next_frame = frame  # Last keyframe, just a point
            spans.append((frame, next_frame))

    return spans


def draw_lock_indicators(context, area, region, gp_obj, settings):
    """Draw sparklines showing world-lock effective duration.

    Uses view coordinates directly (POST_VIEW draw type):
    - X = frame numbers
    - Y = lane/channel index (negative going down)
    """
    view2d = region.view2d

    # Only draw for active layer
    active_layer = gp_obj.data.layers.active
    if active_layer is None:
        return

    # Get spans for locked keyframes (object-level)
    lock_spans = get_lock_spans(gp_obj)
    if not lock_spans:
        return

    # Get layer index
    if active_layer.hide:
        return

    # Identify the correct lane for the active layer.
    # We need to respect Dopesheet filtering and sorting (Top-to-Bottom).
    
    space = area.spaces.active
    dopesheet = space.dopesheet if hasattr(space, 'dopesheet') else None
    
    # 1. Determine Header Offset
    # Standard Hierarchy:
    # - Summary (Optional, Lane 0)
    # - Object (Lane 0 or 1)
    # - Data (Lane 1 or 2)
    # - Layers (Start at Lane 2 or 3)
    
    lane_offset = 2  # Object + Data rows
    if dopesheet and dopesheet.show_summary:
        lane_offset += 1

    # 2. Build list of visible layers in Visual Order (Top-to-Bottom)
    # GP Layers are stored Bottom-to-Top (Index 0 is Bottom).
    # So we iterate in reverse to match Dopesheet visual order.
    visible_layers = []
    
    # Note: gp_obj.data.layers is a collection, reversed() works on it.
    for l in reversed(gp_obj.data.layers):
        # Basic visibility
        if l.hide:
            continue
            
        if dopesheet:
            # Name filter
            if dopesheet.filter_text:
                # Simple case-insensitive containment check
                if dopesheet.filter_text.lower() not in l.name.lower():
                    continue
        
        visible_layers.append(l)

    # 3. Find Active Layer Index
    try:
        visual_index = visible_layers.index(active_layer)
    except ValueError:
        # Active layer is filtered out or hidden
        return

    # 4. Calculate Final Lane
    lane = lane_offset + visual_index
    view_y = get_lane_center_y(context, lane)

    # Calculate line thickness in view units
    # Get the scale factor from pixels to view units
    view_min_y = view2d.region_to_view(0, 0)[1]
    view_max_y = view2d.region_to_view(0, region.height)[1]
    view_height = abs(view_max_y - view_min_y)

    if view_height > 0 and region.height > 0:
        view_units_per_pixel = view_height / region.height
    else:
        view_units_per_pixel = 0.05  # fallback

    line_width = getattr(settings, 'timeline_line_width', 3.0)
    line_half_height = (line_width / 2.0) * view_units_per_pixel

    color = tuple(settings.timeline_lock_color) + (0.9,)

    # Set up GPU state
    gpu.state.blend_set('ALPHA')

    shader = _get_timeline_shader()  # Use cached shader
    shader.bind()
    shader.uniform_float("color", color)

    for start_frame, end_frame in lock_spans:
        # Draw rectangle in view coordinates (frames for X, lane for Y)
        coords = [
            (float(start_frame), view_y - line_half_height),
            (float(end_frame), view_y - line_half_height),
            (float(end_frame), view_y + line_half_height),
            (float(start_frame), view_y + line_half_height),
        ]

        batch = batch_for_shader(shader, 'TRI_FAN', {"pos": coords})
        batch.draw(shader)

    gpu.state.blend_set('NONE')


def register_timeline_handlers():
    """Register timeline draw handler."""
    global _timeline_draw_handler

    if _timeline_draw_handler is None:
        # PRE_VIEW draws before keyframe icons (behind them)
        # Uses view coordinates (frames/channels)
        _timeline_draw_handler = bpy.types.SpaceDopeSheetEditor.draw_handler_add(
            draw_timeline_callback,
            (),
            'WINDOW',
            'PRE_VIEW'
        )


def unregister_timeline_handlers():
    """Unregister timeline draw handler."""
    global _timeline_draw_handler

    if _timeline_draw_handler is not None:
        try:
            bpy.types.SpaceDopeSheetEditor.draw_handler_remove(_timeline_draw_handler, 'WINDOW')
        except ValueError:
            # Handler already removed or not registered
            pass
        _timeline_draw_handler = None
