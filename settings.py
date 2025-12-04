"""
Settings and property definitions for world-space onion skinning.
"""

import bpy
from mathutils import Vector

from .cache import clear_cache, cache_current_frame, get_active_gp
from .drawing import register_draw_handlers, unregister_draw_handlers
from .anchors import (
    get_anchors,
    get_anchor_for_frame,
    set_anchor_for_frame,
    calculate_anchor_from_strokes,
    get_current_keyframes_set
)
from .handlers import set_last_keyframe_set, set_last_active_layer_name


def update_enabled(self, context):
    """Called when addon is enabled/disabled."""
    if self.enabled:
        register_draw_handlers()

        # Cache current frame immediately if GP is active
        gp_obj = get_active_gp(context)
        if gp_obj is not None:
            cache_current_frame(gp_obj, self)
            set_last_keyframe_set(get_current_keyframes_set(gp_obj, self))
    else:
        unregister_draw_handlers()

    # Redraw viewports
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def update_setting(self, context):
    """Called when display settings change."""
    # Just redraw - no need to invalidate cache
    for area in context.screen.areas:
        if area.type in ('VIEW_3D', 'DOPESHEET_EDITOR', 'TIMELINE'):
            area.tag_redraw()


def update_filter(self, context):
    """Called when filter settings change."""
    # Filters affect what strokes are shown - need to rebuild cache
    clear_cache()

    gp_obj = get_active_gp(context)
    if self.enabled and gp_obj is not None:
        cache_current_frame(gp_obj, self)

    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()


class WorldOnionSettings(bpy.types.PropertyGroup):
    enabled: bpy.props.BoolProperty(
        name="Enable",
        description="Enable world-space onion skinning",
        default=False,
        update=update_enabled,
    )

    mode: bpy.props.EnumProperty(
        name="Mode",
        description="How to select which frames to show",
        items=[
            ('FRAMES', "Every Frame", "Show every Nth frame before/after"),
            ('KEYFRAMES', "GP Keyframes", "Show only actual GP keyframes"),
        ],
        default='FRAMES',
        update=update_setting,
    )
    
    frames_before: bpy.props.IntProperty(
        name="Before",
        description="Number of frames/keyframes to show before current",
        default=3, min=0, max=50,
        update=update_setting,
    )
    
    frames_after: bpy.props.IntProperty(
        name="After", 
        description="Number of frames/keyframes to show after current",
        default=3, min=0, max=50,
        update=update_setting,
    )
    
    frame_step: bpy.props.IntProperty(
        name="Step",
        description="Show every Nth frame",
        default=1, min=1, max=24,
        update=update_setting,
    )
    
    opacity: bpy.props.FloatProperty(
        name="Stroke Opacity",
        description="Opacity of onion skin strokes",
        default=0.5, min=0.0, max=1.0,
        update=update_setting,
    )
    
    falloff: bpy.props.FloatProperty(
        name="Falloff",
        description="Fade opacity for frames further from current",
        default=0.5, min=0.0, max=1.0,
        update=update_setting,
    )

    fill_opacity: bpy.props.FloatProperty(
        name="Fill Opacity",
        description="Opacity of fill areas in onion skin",
        default=0.25, min=0.0, max=1.0,
        update=update_setting,
    )
    
    color_before: bpy.props.FloatVectorProperty(
        name="Before Color",
        description="Color for frames before current",
        subtype='COLOR',
        default=(1.0, 0.5, 0.5),
        min=0.0, max=1.0,
        size=3,
        update=update_setting,
    )
    
    color_after: bpy.props.FloatVectorProperty(
        name="After Color",
        description="Color for frames after current",
        subtype='COLOR',
        default=(0.5, 0.8, 1.0),
        min=0.0, max=1.0,
        size=3,
        update=update_setting,
    )
    
    line_width: bpy.props.FloatProperty(
        name="Line Width",
        description="Width of onion skin strokes",
        default=2.0, min=0.5, max=10.0,
        update=update_setting,
    )
    
    skip_underscore: bpy.props.BoolProperty(
        name="Skip _ Layers",
        description="Skip layers whose names start with underscore",
        default=True,
        update=update_filter,
    )
    
    layer_filter: bpy.props.StringProperty(
        name="Layer Filter",
        description="Only show layers containing this text (empty = all)",
        default="",
        update=update_filter,
    )
    
    # Anchor system properties
    anchor_enabled: bpy.props.BoolProperty(
        name="Enable Anchors",
        description="Enable anchor system for world-space drawing",
        default=False,
        update=update_setting,
    )
    
    anchor_auto_cursor: bpy.props.BoolProperty(
        name="Auto-Move Cursor",
        description="Automatically move 3D cursor to anchor position on frame change",
        default=True,
        update=update_setting,
    )
    
    anchor_snap_to_stroke: bpy.props.BoolProperty(
        name="Snap to Stroke",
        description="Recalculate anchor to stroke bottom when drawing. If disabled, anchor stays at original cursor position",
        default=True,
        update=update_setting,
    )

    anchor_show_indicators: bpy.props.BoolProperty(
        name="Show Indicators",
        description="Show visual indicators at anchor positions",
        default=True,
        update=update_setting,
    )
    
    world_lock_inherit: bpy.props.BoolProperty(
        name="Inherit Lock",
        description="New keyframes automatically inherit world lock from previous keyframe",
        default=True,
    )

    # Timeline indicator properties
    show_timeline_indicator: bpy.props.BoolProperty(
        name="Timeline Indicator",
        description="Show world-lock indicators on keyframes in Dopesheet/Timeline",
        default=True,
        update=update_setting,
    )

    timeline_lock_color: bpy.props.FloatVectorProperty(
        name="Lock Color",
        description="Color for world-locked keyframe indicators in timeline",
        subtype='COLOR',
        size=3,
        default=(1.0, 0.6, 0.0),  # Orange
        min=0.0, max=1.0,
        update=update_setting,
    )

    timeline_line_width: bpy.props.FloatProperty(
        name="Line Width",
        description="Width of the timeline lock indicator line",
        default=3.0, min=1.0, max=10.0,
        update=update_setting,
    )

    # Motion path properties
    motion_path_enabled: bpy.props.BoolProperty(
        name="Show Motion Path",
        description="Draw a line connecting anchor positions across locked frames",
        default=False,
        update=update_setting,
    )

    motion_path_color: bpy.props.FloatVectorProperty(
        name="Path Color",
        description="Color of the motion path line",
        subtype='COLOR',
        size=4,
        default=(0.2, 0.8, 1.0, 0.8),  # Cyan with alpha
        min=0.0, max=1.0,
        update=update_setting,
    )

    motion_path_width: bpy.props.FloatProperty(
        name="Path Width",
        description="Width of the motion path line",
        default=2.0, min=1.0, max=10.0,
        update=update_setting,
    )

    motion_path_show_points: bpy.props.BoolProperty(
        name="Show Points",
        description="Show dots at each anchor position along the path",
        default=True,
        update=update_setting,
    )

    motion_path_smoothing: bpy.props.IntProperty(
        name="Smoothing",
        description="Subdivisions for smooth curves (0=sharp, higher=smoother)",
        default=0, min=0, max=10,
        update=update_setting,
    )

    # Interpolation properties
    interpolation_enabled: bpy.props.BoolProperty(
        name="Interpolate Positions",
        description="Interpolate GP position between locked frames. Uses GP keyframe type (T key): Breakdown=Linear, Moving Hold=Bezier",
        default=False,
        update=update_setting,
    )
