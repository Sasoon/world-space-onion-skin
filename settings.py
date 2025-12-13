"""
Settings and property definitions for world-space onion skinning.
"""

import bpy

from .cache import clear_cache, cache_current_frame, get_active_gp
from .drawing import (
    register_draw_handlers, unregister_draw_handlers,
    bake_shrinkwrap_offsets, invalidate_baked_offsets,
    invalidate_motion_path, remove_shrinkwrap_driver,
    invalidate_onion_batch_cache, complete_pending_driver_setup,
)
from .anchors import get_current_keyframes_set
from .handlers import set_last_keyframe_set, set_last_active_layer_name
from .transforms import align_canvas_to_cursor


def update_enabled(self, context):
    """Called when addon is enabled/disabled."""
    if self.enabled:
        register_draw_handlers()
        # v8.5: Start cursor sync modal operator for reliable cursor tracking
        # Import here to avoid circular import
        from .operators import is_cursor_sync_running
        if not is_cursor_sync_running():
            bpy.ops.world_onion.cursor_sync('INVOKE_DEFAULT')

        # Cache current frame immediately if GP is active
        gp_obj = get_active_gp(context)
        if gp_obj is not None:
            cache_current_frame(gp_obj, self)
            set_last_keyframe_set(get_current_keyframes_set(gp_obj, self))
    else:
        unregister_draw_handlers()
        # v8.5: Modal operator auto-cancels when addon is disabled (checks in modal())

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


def update_motion_path_setting(self, context):
    """Called when motion path geometry settings change (smoothing)."""
    # Invalidate motion path cache so it rebuilds with new smoothing
    invalidate_motion_path()
    # Redraw viewports
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()


def update_anchor_enabled(self, context):
    """Called when anchor system is enabled/disabled.

    v8.2: Canvas alignment is done HERE (once) instead of every frame change.
    This avoids 'Writing to ID classes not allowed' errors during timeline scrubbing.
    """
    if self.anchor_enabled:
        # Set canvas to follow cursor - this setting PERSISTS
        try:
            align_canvas_to_cursor(context)
        except (RuntimeError, AttributeError):
            pass  # May fail in some contexts, but that's OK - user can trigger manually

    # Redraw viewports
    for area in context.screen.areas:
        if area.type in ('VIEW_3D', 'DOPESHEET_EDITOR', 'TIMELINE'):
            area.tag_redraw()


def update_realtime(self, context):
    """Called when realtime settings change (Z offset, shrinkwrap) - apply immediately."""
    # NOTE: DO NOT clear stroke cache here!
    # Z offset is applied at draw time, not stored in cache.
    # Clearing cache on every slider adjustment was causing massive lag.
    # Only shrinkwrap state change requires special handling (baking).

    # Invalidate GPU batch cache since z_offset affects batch geometry
    invalidate_onion_batch_cache()

    # Auto-bake shrinkwrap offsets when shrinkwrap is enabled
    # This ensures we have baked data before playback starts
    # v9.3: UI callback is a safe context - driver setup will succeed here
    if self.depth_interaction_enabled:
        gp_obj = get_active_gp(context)
        if gp_obj:
            # Bake with setup_driver=True (default) - safe context allows it
            bake_shrinkwrap_offsets(gp_obj, self, context.scene, setup_driver=True)
            # Also complete any pending driver setup from previous attempts
            # (e.g., if user enabled shrinkwrap during playback, stopped, then adjusted settings)
            complete_pending_driver_setup(gp_obj)
    else:
        # Shrinkwrap disabled - invalidate baked data and remove driver
        invalidate_baked_offsets()
        gp_obj = get_active_gp(context)
        if gp_obj:
            remove_shrinkwrap_driver(gp_obj)

    # Invalidate motion path so it rebuilds
    invalidate_motion_path()

    # Force frame re-evaluation to apply the change
    scene = context.scene
    current_frame = scene.frame_current
    scene.frame_set(current_frame)

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
        default=3, min=0, max=500,
        update=update_setting,
    )

    frames_after: bpy.props.IntProperty(
        name="After",
        description="Number of frames/keyframes to show after current",
        default=3, min=0, max=500,
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
    
    # Anchor system properties
    anchor_enabled: bpy.props.BoolProperty(
        name="Enable Anchors",
        description="Enable anchor system for world-space drawing",
        default=False,
        update=update_anchor_enabled,  # v8.2: Canvas alignment done once here
    )
    
    anchor_sync_mode: bpy.props.EnumProperty(
        name="Sync Mode",
        description="How cursor and object position interact",
        items=[
            ('NONE', "Manual", "No automatic sync - use Set Anchor button"),
            ('CURSOR_FOLLOWS', "Cursor → Object", "Cursor follows object position"),
            ('OBJECT_FOLLOWS', "Object → Cursor", "Object follows cursor (Auto-Draw mode)"),
        ],
        default='NONE',
        update=update_setting,
    )

    align_to_view: bpy.props.BoolProperty(
        name="Align to View",
        description="Rotate strokes to face the camera when snapping",
        default=False,
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

    # Arc/Ease Spacing Dots - show timing visualization
    motion_path_spacing_dots_enabled: bpy.props.BoolProperty(
        name="Spacing Dots",
        description="Show dots along path with spacing based on velocity (dense=slow, sparse=fast)",
        default=False,
        update=update_motion_path_setting,
    )

    motion_path_spacing_dots_count: bpy.props.IntProperty(
        name="Dot Count",
        description="Number of spacing dots to show along the path",
        default=50, min=10, max=200,
        update=update_motion_path_setting,
    )

    motion_path_spacing_dots_size: bpy.props.FloatProperty(
        name="Boldness",
        description="Thickness/weight of timing chart tick marks",
        default=3.0, min=1.0, max=10.0,
        update=update_setting,  # Just redraws - lineWidth is GPU state
    )

    motion_path_spacing_dots_color: bpy.props.FloatVectorProperty(
        name="Tick Color",
        description="Color of timing chart tick marks",
        subtype='COLOR',
        size=4,
        default=(1.0, 0.0, 0.0, 1.0),  # Bright red, full opacity for visibility
        min=0.0, max=1.0,
        update=update_setting,
    )

    motion_path_keyframe_color: bpy.props.FloatVectorProperty(
        name="Key Color",
        description="Color of keyframe markers (circles) on timing chart",
        subtype='COLOR',
        size=4,
        default=(1.0, 1.0, 1.0, 1.0),  # White for visibility
        min=0.0, max=1.0,
        update=update_setting,
    )

    # Direction Arrows - show travel direction at keyframes
    motion_path_arrows_enabled: bpy.props.BoolProperty(
        name="Direction Arrows",
        description="Show arrows at keyframe positions indicating direction of travel",
        default=False,
        update=update_motion_path_setting,
    )

    motion_path_arrows_size: bpy.props.FloatProperty(
        name="Arrow Size",
        description="Size of direction arrows in world units",
        default=0.1, min=0.01, max=1.0,
        update=update_motion_path_setting,
    )

    motion_path_arrows_color: bpy.props.FloatVectorProperty(
        name="Arrow Color",
        description="Color of direction arrows",
        subtype='COLOR',
        size=4,
        default=(1.0, 0.5, 0.0, 0.9),  # Orange with alpha
        min=0.0, max=1.0,
        update=update_setting,
    )

    # Frame Labels - show frame numbers at keyframes
    motion_path_labels_enabled: bpy.props.BoolProperty(
        name="Frame Labels",
        description="Show frame numbers at keyframe positions",
        default=False,
        update=update_setting,
    )

    motion_path_labels_size: bpy.props.IntProperty(
        name="Label Size",
        description="Font size for frame labels",
        default=12, min=8, max=32,
        update=update_setting,
    )

    motion_path_labels_color: bpy.props.FloatVectorProperty(
        name="Label Color",
        description="Color of frame number labels",
        subtype='COLOR',
        size=4,
        default=(1.0, 1.0, 1.0, 0.9),  # White with alpha
        min=0.0, max=1.0,
        update=update_setting,
    )

    # Depth interaction (shrinkwrap) - strokes follow mesh surface
    depth_interaction_enabled: bpy.props.BoolProperty(
        name="Shrinkwrap",
        description="Motion path and strokes automatically follow mesh surfaces below them",
        default=False,
        update=update_realtime,
    )

    # Global Z offset for stroke placement
    stroke_z_offset: bpy.props.FloatProperty(
        name="Z Offset",
        description="Push strokes up on global Z to prevent clipping behind mesh",
        default=0.0, min=0.0, max=10.0,
        step=1,  # 0.01 increments
        precision=3,
        update=update_realtime,
    )
