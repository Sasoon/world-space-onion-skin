"""
UI Panel for world-space onion skinning.
"""

import bpy

from .cache import get_cache_stats, get_active_gp
from .anchors import get_anchors, is_object_locked_at_frame, get_visible_keyframe, find_visible_locked_frame


class WONION_PT_main_panel(bpy.types.Panel):
    """World Space Onion Skin panel"""
    bl_label = "World Onion Skin"
    bl_idname = "WONION_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Onion'

    def draw_header(self, context):
        settings = context.scene.world_onion
        self.layout.prop(settings, "enabled", text="")

    def draw(self, context):
        layout = self.layout
        settings = context.scene.world_onion

        # Check for active GP object
        gp_obj = get_active_gp(context)

        if not settings.enabled:
            layout.label(text="Disabled", icon='CHECKBOX_DEHLT')
            return

        if gp_obj is None:
            layout.label(text="Select a GP object", icon='INFO')
            return

        # Show active GP object name with world lock toggle
        current_frame = context.scene.frame_current

        # Find which locked frame is active (object-level)
        locked_frame = find_visible_locked_frame(gp_obj, current_frame)
        is_locked = locked_frame is not None

        row = layout.row()
        row.label(text=gp_obj.name, icon='GREASEPENCIL')

        # World lock toggle (object-level)
        icon = 'LOCKED' if is_locked else 'UNLOCKED'
        row.operator("world_onion.toggle_world_lock", icon=icon, text="")

        # Show parent chain info
        parent = gp_obj.parent
        if parent:
            chain = []
            p = parent
            while p:
                chain.append(p.name)
                if p.type == 'CAMERA':
                    break
                p = p.parent
            if chain:
                layout.label(text=f"  {' > '.join(chain)}", icon='LINKED')

        # Show world lock status (object-level)
        if is_locked:
            layout.label(text=f"World Locked (frame {locked_frame})", icon='PINNED')

        # Edit mode tools
        if context.mode == 'EDIT_GREASE_PENCIL':
            layout.separator()
            layout.operator("world_onion.snap_to_cursor", icon='CURSOR')


class WONION_PT_frames(bpy.types.Panel):
    """Frame settings sub-panel"""
    bl_label = "Frames"
    bl_idname = "WONION_PT_frames"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Onion'
    bl_parent_id = "WONION_PT_main_panel"

    @classmethod
    def poll(cls, context):
        settings = context.scene.world_onion
        return settings.enabled and get_active_gp(context) is not None

    def draw(self, context):
        layout = self.layout
        settings = context.scene.world_onion

        layout.prop(settings, "mode", text="")
        row = layout.row(align=True)
        row.prop(settings, "frames_before")
        row.prop(settings, "frames_after")
        # Only show step for Every Frame mode
        if settings.mode == 'FRAMES':
            layout.prop(settings, "frame_step")


class WONION_PT_appearance(bpy.types.Panel):
    """Appearance settings sub-panel"""
    bl_label = "Appearance"
    bl_idname = "WONION_PT_appearance"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Onion'
    bl_parent_id = "WONION_PT_main_panel"

    @classmethod
    def poll(cls, context):
        settings = context.scene.world_onion
        return settings.enabled and get_active_gp(context) is not None

    def draw(self, context):
        layout = self.layout
        settings = context.scene.world_onion

        # Stroke and Fill opacity on same row
        row = layout.row(align=True)
        row.prop(settings, "opacity")
        row.prop(settings, "fill_opacity")

        # Falloff on next row
        layout.prop(settings, "falloff")

        # Line width following row
        layout.prop(settings, "line_width")

        # Colors
        row = layout.row(align=True)
        row.prop(settings, "color_before", text="")
        row.prop(settings, "color_after", text="")


class WONION_PT_filters(bpy.types.Panel):
    """Filter settings sub-panel"""
    bl_label = "Filters"
    bl_idname = "WONION_PT_filters"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Onion'
    bl_parent_id = "WONION_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        settings = context.scene.world_onion
        return settings.enabled and get_active_gp(context) is not None

    def draw(self, context):
        layout = self.layout
        settings = context.scene.world_onion

        layout.prop(settings, "skip_underscore")
        layout.prop(settings, "layer_filter")


class WONION_PT_anchors(bpy.types.Panel):
    """Anchor system sub-panel"""
    bl_label = "Anchors"
    bl_idname = "WONION_PT_anchors"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Onion'
    bl_parent_id = "WONION_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        settings = context.scene.world_onion
        return settings.enabled and get_active_gp(context) is not None

    def draw_header(self, context):
        settings = context.scene.world_onion
        self.layout.prop(settings, "anchor_enabled", text="")

    def draw(self, context):
        layout = self.layout
        settings = context.scene.world_onion
        gp_obj = get_active_gp(context)

        layout.active = settings.anchor_enabled

        layout.prop(settings, "anchor_auto_cursor")
        layout.prop(settings, "anchor_snap_to_stroke")
        layout.prop(settings, "anchor_show_indicators")
        layout.prop(settings, "world_lock_inherit")

        # Timeline indicator
        row = layout.row(align=True)
        row.prop(settings, "show_timeline_indicator")
        if settings.show_timeline_indicator:
            row.prop(settings, "timeline_lock_color", text="")
            row.prop(settings, "timeline_line_width", text="W")

        # Motion path
        row = layout.row(align=True)
        row.prop(settings, "motion_path_enabled")
        if settings.motion_path_enabled:
            row.prop(settings, "motion_path_color", text="")
            row.prop(settings, "motion_path_width", text="W")
            layout.prop(settings, "motion_path_show_points")

        layout.separator()

        row = layout.row(align=True)
        row.operator("world_onion.auto_anchor", text="Auto", icon='CON_LOCLIKE')
        row.operator("world_onion.set_anchor", text="Cursor", icon='PINNED')
        row.operator("world_onion.clear_anchor", text="Clear", icon='UNPINNED')
        row.operator("world_onion.clear_all_anchors", text="", icon='X')

        # Show anchor count
        if gp_obj is not None:
            anchors = get_anchors(gp_obj)
            total = sum(len(layer_anchors) for layer_anchors in anchors.values())
            layout.label(text=f"{total} anchors stored")


class WONION_PT_cache(bpy.types.Panel):
    """Cache sub-panel"""
    bl_label = "Cache"
    bl_idname = "WONION_PT_cache"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Onion'
    bl_parent_id = "WONION_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        settings = context.scene.world_onion
        return settings.enabled and get_active_gp(context) is not None

    def draw(self, context):
        layout = self.layout

        layout.label(text=get_cache_stats())

        row = layout.row(align=True)
        row.operator("world_onion.build_cache", icon='FILE_REFRESH')
        row.operator("world_onion.clear_cache", icon='X', text="")


class WONION_PT_dev(bpy.types.Panel):
    """Development tools sub-panel"""
    bl_label = "Dev"
    bl_idname = "WONION_PT_dev"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Onion'
    bl_parent_id = "WONION_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        settings = context.scene.world_onion
        return settings.enabled and get_active_gp(context) is not None

    def draw(self, context):
        layout = self.layout
        layout.operator("world_onion.reload_addon", icon='FILE_REFRESH')
        layout.separator()
        layout.label(text="Testing:")
        layout.operator("world_onion.reset_test_state", icon='LOOP_BACK')
        layout.operator("world_onion.clear_all_locks", icon='UNLOCKED')


# List of panel classes for registration
panel_classes = (
    WONION_PT_main_panel,
    WONION_PT_frames,
    WONION_PT_appearance,
    WONION_PT_filters,
    WONION_PT_anchors,
    WONION_PT_cache,
    WONION_PT_dev,
)
