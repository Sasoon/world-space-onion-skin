"""
UI Panel for world-space onion skinning.
"""

import bpy

from .cache import get_cache_stats, get_active_gp
from .anchors import get_anchors


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

        row = layout.row()
        row.label(text=gp_obj.name, icon='GREASEPENCIL')

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

        # Edit mode tools
        if context.mode == 'EDIT_GREASE_PENCIL':
            layout.separator()
            row = layout.row(align=True)
            row.operator("world_onion.snap_to_cursor", text="Snap to Cursor", icon='CURSOR')
            row.operator("world_onion.snap_to_gp", text="Snap to GP", icon='GREASEPENCIL')

            # Align to View toggle and button
            row = layout.row(align=True)
            row.prop(settings, "align_to_view", toggle=True, icon='VIEW_CAMERA')
            row.operator("world_onion.align_to_view", text="", icon='CON_TRACKTO')


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


class WONION_PT_animation(bpy.types.Panel):
    """Animation and Anchor settings sub-panel"""
    bl_label = "Motion & Anchors"
    bl_idname = "WONION_PT_animation"
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
        gp_obj = get_active_gp(context)

        # Motion Path
        row = layout.row(align=True)
        row.prop(settings, "motion_path_enabled")
        if settings.motion_path_enabled:
            row.prop(settings, "motion_path_color", text="")
            row.prop(settings, "motion_path_width", text="W")

            row = layout.row(align=True)
            row.prop(settings, "motion_path_smoothing")

        layout.separator()

        # Shrinkwrap and Z Offset
        layout.prop(settings, "depth_interaction_enabled")
        if settings.depth_interaction_enabled:
            row = layout.row(align=True)
            row.operator("world_onion.bake_shrinkwrap", icon='FILE_REFRESH')
            # Show bake status
            from .drawing import is_bake_valid
            if is_bake_valid():
                row.label(text="", icon='CHECKMARK')
            else:
                row.label(text="", icon='ERROR')
        layout.prop(settings, "stroke_z_offset")

        layout.separator()

        # Anchor System
        layout.prop(settings, "anchor_enabled")
        
        if settings.anchor_enabled:
            box = layout.box()
            box.prop(settings, "anchor_auto_cursor")

        layout.separator()

        # Anchor data management
        row = layout.row(align=True)
        row.operator("world_onion.set_anchor", text="Set Anchor", icon='PINNED')
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


# List of panel classes for registration
panel_classes = (
    WONION_PT_main_panel,
    WONION_PT_frames,
    WONION_PT_appearance,
    WONION_PT_animation,
    WONION_PT_cache,
    WONION_PT_dev,
)
