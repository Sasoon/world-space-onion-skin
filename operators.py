"""
Operators for world-space onion skinning.
"""

import bpy
from mathutils import Vector, Matrix

from .cache import clear_cache, get_cache, get_cache_stats, get_active_gp
from .anchors import (
    get_anchors,
    set_anchors,
    get_anchor_for_frame,
    set_anchor_for_frame,
    remove_anchor_for_frame,
    calculate_anchor_from_strokes,
    get_current_keyframes_set,
    get_visible_keyframe,
)
from .transforms import get_layer_transform, get_camera_direction, align_canvas_to_cursor, ensure_billboard_constraint
from .drawing import invalidate_motion_path, get_baked_offset
from .debug_log import log

# v8.5: Track if cursor sync modal is running
_cursor_sync_running = False

# v8.5.2: Shared frame tracker for hybrid cursor sync (modal + handler coordination)
# This prevents double-updates which cause jitter
_last_cursor_synced_frame = None


def is_cursor_sync_running():
    """Check if cursor sync modal operator is running."""
    return _cursor_sync_running


def set_cursor_sync_running(value):
    """Set cursor sync modal operator running state."""
    global _cursor_sync_running
    _cursor_sync_running = value


def get_last_cursor_synced_frame():
    """Get the last frame cursor was synced for (used by handler to avoid double-update)."""
    return _last_cursor_synced_frame


def set_last_cursor_synced_frame(frame):
    """Set the last frame cursor was synced for."""
    global _last_cursor_synced_frame
    _last_cursor_synced_frame = frame


class WONION_OT_cursor_sync(bpy.types.Operator):
    """Background operator to sync cursor with GP object during playback"""
    bl_idname = "world_onion.cursor_sync"
    bl_label = "Cursor Sync"
    bl_options = {'INTERNAL'}

    _timer = None
    _last_frame = None

    def modal(self, context, event):
        # Check if addon still enabled
        if not hasattr(context.scene, 'world_onion'):
            self.cancel(context)
            return {'CANCELLED'}

        settings = context.scene.world_onion
        if not settings.enabled:
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            # Update cursor if needed
            if settings.anchor_enabled and settings.anchor_auto_cursor:
                current_frame = context.scene.frame_current
                # v8.5.3: Check BOTH instance state AND shared tracker
                # Skip if handler already updated this frame (prevents double-update jitter)
                if (current_frame != self._last_frame and
                    current_frame != get_last_cursor_synced_frame()):
                    self._last_frame = current_frame
                    gp_obj = get_active_gp(context)
                    if gp_obj:
                        try:
                            depsgraph = context.evaluated_depsgraph_get()
                            gp_obj_eval = gp_obj.evaluated_get(depsgraph)
                            context.scene.cursor.location = gp_obj_eval.matrix_world.translation
                            # Mark frame as synced so handler doesn't double-update
                            set_last_cursor_synced_frame(current_frame)
                            # Log for debugging
                            log(f"MODAL_CURSOR frame={current_frame}", "CURSOR")
                        except Exception as e:
                            log(f"MODAL_CURSOR frame={current_frame} FAILED: {e}", "ERROR")

        return {'PASS_THROUGH'}

    def execute(self, context):
        global _cursor_sync_running
        if _cursor_sync_running:
            # Already running
            return {'CANCELLED'}

        wm = context.window_manager
        # v8.5.1: Increased polling rate to 200fps (5ms) to reduce cursor lag during scrubbing
        # Higher frequency = less time between frame change and cursor update
        self._timer = wm.event_timer_add(0.005, window=context.window)
        wm.modal_handler_add(self)
        _cursor_sync_running = True
        log("Cursor sync modal started (5ms interval)", "INFO")
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        global _cursor_sync_running
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        _cursor_sync_running = False
        log("Cursor sync modal stopped", "INFO")


class WONION_OT_clear_cache(bpy.types.Operator):
    """Clear the onion skin cache"""
    bl_idname = "world_onion.clear_cache"
    bl_label = "Clear Cache"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        cache = get_cache()
        count = len(cache)
        clear_cache()
        self.report({'INFO'}, f"Cleared {count} cached frames")
        
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        
        return {'FINISHED'}


class WONION_OT_build_cache(bpy.types.Operator):
    """Scrub through timeline to build cache"""
    bl_idname = "world_onion.build_cache"
    bl_label = "Build Full Cache"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return get_active_gp(context) is not None

    def execute(self, context):
        settings = context.scene.world_onion

        if not settings.enabled:
            self.report({'WARNING'}, "Enable onion skin first")
            return {'CANCELLED'}

        gp_obj = get_active_gp(context)
        if gp_obj is None:
            self.report({'WARNING'}, "No active GP object")
            return {'CANCELLED'}
        
        scene = context.scene
        start = scene.frame_start
        end = scene.frame_end
        original = scene.frame_current
        
        context.window.cursor_set('WAIT')
        
        # Scrub through all frames to build onion skin cache
        for frame in range(start, end + 1):
            scene.frame_set(frame)
        
        # Return to original frame
        scene.frame_set(original)
        
        context.window.cursor_set('DEFAULT')
        
        cache = get_cache()
        msg = f"Cached {len(cache)} frames"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


def set_anchor_logic(context, gp_obj, scene, target_world_pos, move_selected_strokes_to_target=False):
    """
    Shared logic for setting anchor (moving object location).
    target_world_pos: Vector - where the object should move to.
    move_selected_strokes_to_target: bool - if True, selected strokes are moved to the target
                                            (which means they become (0,0,0) local).
                                            If False, ALL strokes are compensated to stay in world space.
    """
    current_frame = scene.frame_current
    
    # Get current object location (old anchor)
    old_location = gp_obj.location.copy()
    
    active_layer = gp_obj.data.layers.active
    if active_layer is None:
        return {'CANCELLED'}

    # Find active keyframe
    active_kf = None
    for kf in active_layer.frames:
        if kf.frame_number == current_frame:
            active_kf = kf
            break
    
    if active_kf is None:
        # Find visible
        visible_kf = None
        for kf in active_layer.frames:
            if kf.frame_number <= current_frame:
                if visible_kf is None or kf.frame_number > visible_kf.frame_number:
                    visible_kf = kf
        
        if visible_kf:
            # Create new keyframe at current frame
            active_kf = active_layer.frames.copy(visible_kf.frame_number, current_frame)
        else:
            return {'CANCELLED'}

    # Ensure billboard constraint exists (so matrix_world is correct)
    ensure_billboard_constraint(gp_obj, scene)
    context.view_layer.update()

    # Matrix before move
    matrix_world_old = gp_obj.matrix_world.copy()
    
    # Store world positions
    stroke_points = []
    drawing = active_kf.drawing
    for stroke in drawing.strokes:
        points = []
        for p in stroke.points:
            world_pos = matrix_world_old @ Vector(p.position)
            points.append(world_pos)
        stroke_points.append(points)
        
    # Move Object
    gp_obj.location = target_world_pos
    
    # Insert Keyframe for Location
    gp_obj.keyframe_insert(data_path="location", frame=current_frame)
    
    # Update view layer to get new matrix
    context.view_layer.update()
    matrix_world_new = gp_obj.matrix_world
    matrix_world_new_inv = matrix_world_new.inverted()
    
    # Transform strokes back to local space relative to new object position
    for i, stroke in enumerate(drawing.strokes):
        if move_selected_strokes_to_target and stroke.select:
            # Calculate bottom-center anchor point (center XY, lowest Z)
            # This ensures strokes sit ON the surface rather than clipping through
            points_world = stroke_points[i]
            if not points_world:
                continue
                
            sum_x = sum(p.x for p in points_world)
            sum_y = sum(p.y for p in points_world)
            min_z = min(p.z for p in points_world)
            n = len(points_world)
            bottom_center = Vector((sum_x/n, sum_y/n, min_z))
            
            # We want bottom_center to be at target_world_pos
            offset = target_world_pos - bottom_center
            
            # Apply offset to world points, then transform to new local
            for j, p_world in enumerate(points_world):
                new_world = p_world + offset
                new_local = matrix_world_new_inv @ new_world
                stroke.points[j].position = new_local
                
        else:
            # Preserve world position (compensate for object move)
            points_world = stroke_points[i]
            for j, p_world in enumerate(points_world):
                new_local = matrix_world_new_inv @ p_world
                stroke.points[j].position = new_local

    # Explicitly invalidate motion path and request redraw
    invalidate_motion_path()
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()

    return {'FINISHED'}


class WONION_OT_set_anchor(bpy.types.Operator):
    """Set anchor at current 3D cursor position (moves object, keeps strokes in place)"""
    bl_idname = "world_onion.set_anchor"
    bl_label = "Set Anchor"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return get_active_gp(context) is not None

    def execute(self, context):
        gp_obj = get_active_gp(context)
        if gp_obj is None:
            return {'CANCELLED'}
        
        scene = context.scene
        cursor_pos = scene.cursor.location.copy()
        
        result = set_anchor_logic(context, gp_obj, scene, cursor_pos, move_selected_strokes_to_target=False)
        
        if result == {'FINISHED'}:
            self.report({'INFO'}, "Anchor set (Object moved to Cursor)")
            active_layer = gp_obj.data.layers.active
            if active_layer:
                 cam_dir = get_camera_direction(scene)
                 set_anchor_for_frame(gp_obj, active_layer.name, scene.frame_current, cursor_pos, cam_dir)
            
        return result


class WONION_OT_snap_to_gp(bpy.types.Operator):
    """Snap cursor and anchor to stroke bottom-center (center XY, lowest Z)"""
    bl_idname = "world_onion.snap_to_gp"
    bl_label = "Snap to GP"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return get_active_gp(context) is not None

    def execute(self, context):
        gp_obj = get_active_gp(context)
        if gp_obj is None:
            return {'CANCELLED'}
            
        active_layer = gp_obj.data.layers.active
        if active_layer is None:
            return {'CANCELLED'}
            
        scene = context.scene
        settings = scene.world_onion
        current_frame = scene.frame_current
        
        # Calculate stroke bottom-center (center XY, lowest Z)
        anchor_pos = calculate_anchor_from_strokes(gp_obj, active_layer, current_frame)

        if anchor_pos is None:
            self.report({'WARNING'}, "No strokes found")
            return {'CANCELLED'}

        # Note: Z offset is NOT applied here - it's applied in on_frame_change
        # to avoid double-application (baked in keyframe + applied on frame change)

        result = set_anchor_logic(context, gp_obj, scene, anchor_pos, move_selected_strokes_to_target=False)
        
        if result == {'FINISHED'}:
            self.report({'INFO'}, "Snapped to GP (bottom-center)")
            scene.cursor.location = anchor_pos
            cam_dir = get_camera_direction(scene)
            set_anchor_for_frame(gp_obj, active_layer.name, scene.frame_current, anchor_pos, cam_dir)

            # v9.1.2: Auto-bake shrinkwrap after snap
            if settings.depth_interaction_enabled:
                from .drawing import bake_shrinkwrap_offsets
                bake_shrinkwrap_offsets(gp_obj, settings, scene)
                context.view_layer.update()

        return result


class WONION_OT_snap_to_cursor(bpy.types.Operator):
    """Snap selected strokes to cursor AND move object to cursor"""
    bl_idname = "world_onion.snap_to_cursor"
    bl_label = "Snap to Cursor"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        gp_obj = get_active_gp(context)
        return gp_obj is not None and context.mode == 'EDIT_GREASE_PENCIL'

    def execute(self, context):
        gp_obj = get_active_gp(context)
        if gp_obj is None:
            return {'CANCELLED'}

        scene = context.scene
        settings = scene.world_onion
        cursor_pos = scene.cursor.location.copy()

        # Note: Z offset is NOT applied here - it's applied in on_frame_change
        # to avoid double-application (baked in keyframe + applied on frame change)

        result = set_anchor_logic(context, gp_obj, scene, cursor_pos, move_selected_strokes_to_target=True)

        if result == {'FINISHED'}:
            self.report({'INFO'}, "Snapped strokes and Object to Cursor")
            active_layer = gp_obj.data.layers.active
            if active_layer:
                cam_dir = get_camera_direction(scene)
                set_anchor_for_frame(gp_obj, active_layer.name, scene.frame_current, cursor_pos, cam_dir)

            # v9.1.2: Auto-bake shrinkwrap after snap
            if settings.depth_interaction_enabled:
                from .drawing import bake_shrinkwrap_offsets
                bake_shrinkwrap_offsets(gp_obj, settings, scene)
                context.view_layer.update()

        return result


class WONION_OT_clear_anchor(bpy.types.Operator):
    """Clear Anchor Data"""
    bl_idname = "world_onion.clear_anchor"
    bl_label = "Clear Anchor Data"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return get_active_gp(context) is not None

    def execute(self, context):
        gp_obj = get_active_gp(context)
        if gp_obj is None:
            return {'CANCELLED'}
        
        active_layer = gp_obj.data.layers.active
        if active_layer:
            remove_anchor_for_frame(gp_obj, active_layer.name, context.scene.frame_current)
            self.report({'INFO'}, "Anchor metadata cleared")
            
        return {'FINISHED'}


class WONION_OT_clear_all_anchors(bpy.types.Operator):
    """Clear all anchor metadata"""
    bl_idname = "world_onion.clear_all_anchors"
    bl_label = "Clear All Anchors"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        gp_obj = get_active_gp(context)
        if gp_obj:
            set_anchors(gp_obj, {})
            self.report({'INFO'}, "All anchor metadata cleared")
        return {'FINISHED'}


class WONION_OT_reload_addon(bpy.types.Operator):
    """Reload the addon (for development) - handles full unregister/register cycle"""
    bl_idname = "world_onion.reload_addon"
    bl_label = "Reload Addon"
    bl_options = {'REGISTER'}

    def execute(self, context):
        import importlib
        import sys

        addon_name = __package__.split('.')[0] if '.' in __package__ else __package__
        print(f"RELOADING ADDON: {addon_name}")

        if addon_name not in sys.modules:
            self.report({'ERROR'}, f"Addon module '{addon_name}' not found")
            return {'CANCELLED'}

        main_module = sys.modules[addon_name]

        # 1. Unregister current state
        try:
            main_module.unregister()
            print("Unregistered successfully")
        except Exception as e:
            print(f"Unregister failed: {e}")

        # 2. Reload main module (triggers reload of submodules via __init__.py)
        try:
            importlib.reload(main_module)
            print("Reloaded successfully")
        except Exception as e:
            self.report({'ERROR'}, f"Reload failed: {e}")
            return {'CANCELLED'}

        # 3. Re-register
        try:
            main_module.register()
            print("Registered successfully")
        except Exception as e:
            self.report({'ERROR'}, f"Register failed: {e}")
            return {'CANCELLED'}

        # 4. Force Redraw
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()

        self.report({'INFO'}, "Addon Reloaded Successfully")
        return {'FINISHED'}


class WONION_OT_bake_shrinkwrap(bpy.types.Operator):
    """Bake shrinkwrap offsets for entire animation range"""
    bl_idname = "world_onion.bake_shrinkwrap"
    bl_label = "Bake Shrinkwrap"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        settings = context.scene.world_onion
        return (get_active_gp(context) is not None and
                settings.enabled and
                settings.depth_interaction_enabled)

    def execute(self, context):
        from .drawing import bake_shrinkwrap_offsets

        gp_obj = get_active_gp(context)
        settings = context.scene.world_onion

        count = bake_shrinkwrap_offsets(gp_obj, settings, context.scene)

        # v9.1.1: Force depsgraph update and cursor sync after bake
        # Without this, driver hasn't evaluated yet and cursor is at wrong position
        context.view_layer.update()

        # Sync cursor to GP object's new position (with offset applied)
        if settings.anchor_enabled and settings.anchor_auto_cursor:
            try:
                depsgraph = context.evaluated_depsgraph_get()
                gp_obj_eval = gp_obj.evaluated_get(depsgraph)
                context.scene.cursor.location = gp_obj_eval.matrix_world.translation
            except:
                pass

        self.report({'INFO'}, f"Baked shrinkwrap for {count} frames")

        # Force redraw
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

        return {'FINISHED'}


# List of operator classes for registration
operator_classes = (
    WONION_OT_cursor_sync,  # v8.5: Modal timer for cursor sync
    WONION_OT_clear_cache,
    WONION_OT_build_cache,
    WONION_OT_set_anchor,
    WONION_OT_snap_to_gp,
    WONION_OT_snap_to_cursor,
    WONION_OT_clear_anchor,
    WONION_OT_clear_all_anchors,
    WONION_OT_reload_addon,
    WONION_OT_bake_shrinkwrap,
)
