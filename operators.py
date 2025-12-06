"""
Operators for world-space onion skinning.
"""

import bpy
from mathutils import Vector

from .cache import clear_cache, get_cache, cache_current_frame, get_cache_stats, get_active_gp
from .anchors import (
    get_anchors,
    set_anchors,
    get_anchor_for_frame,
    set_anchor_for_frame,
    remove_anchor_for_frame,
    calculate_anchor_from_strokes,
    get_current_keyframes_set,
    get_visible_keyframe,
    # Object-level lock functions
    is_object_locked_at_frame,
    get_lock_for_frame,
    set_lock_for_frame,
    remove_lock_for_frame,
    update_lock_anchor,
)
from .transforms import get_layer_transform, get_camera_direction
from .handlers import apply_object_world_lock_for_frame, reset_object_world_lock, apply_world_lock_from_stored
from .drawing import invalidate_motion_path


class WONION_OT_clear_cache(bpy.types.Operator):
    """Clear the onion skin cache"""
    bl_idname = "world_onion.clear_cache"
    bl_label = "Clear Cache"
    bl_options = {'REGISTER'}
    
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
    """Scrub through timeline to build cache and calculate anchors"""
    bl_idname = "world_onion.build_cache"
    bl_label = "Build Full Cache"
    bl_options = {'REGISTER'}

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
        
        anchors_created = 0
        
        # Build anchors for all layers, all keyframes
        if settings.anchor_enabled:
            for layer in gp_obj.data.layers:
                if layer.hide:
                    continue
                if settings.skip_underscore and layer.name.startswith('_'):
                    continue
                if settings.layer_filter and settings.layer_filter not in layer.name:
                    continue
                
                for kf in layer.frames:
                    # Check if anchor already exists
                    existing = get_anchor_for_frame(gp_obj, layer.name, kf.frame_number)
                    if existing is None:
                        # Need to set frame to get correct world matrix
                        scene.frame_set(kf.frame_number)
                        
                        anchor_pos = calculate_anchor_from_strokes(gp_obj, layer, kf.frame_number)
                        if anchor_pos is not None:
                            cam_dir = get_camera_direction(scene)
                            set_anchor_for_frame(gp_obj, layer.name, kf.frame_number, anchor_pos, cam_dir)
                            anchors_created += 1
        
        # Scrub through all frames to build onion skin cache
        for frame in range(start, end + 1):
            scene.frame_set(frame)
        
        # Return to original frame
        scene.frame_set(original)
        
        context.window.cursor_set('DEFAULT')
        
        cache = get_cache()
        msg = f"Cached {len(cache)} frames"
        if anchors_created > 0:
            msg += f", created {anchors_created} anchors"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class WONION_OT_set_anchor(bpy.types.Operator):
    """Set anchor at current 3D cursor position for active layer's current keyframe"""
    bl_idname = "world_onion.set_anchor"
    bl_label = "Set Anchor"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return get_active_gp(context) is not None

    def execute(self, context):
        settings = context.scene.world_onion
        gp_obj = get_active_gp(context)
        scene = context.scene

        if gp_obj is None or gp_obj.data is None:
            self.report({'WARNING'}, "No active GP object")
            return {'CANCELLED'}
        
        active_layer = gp_obj.data.layers.active
        if active_layer is None:
            self.report({'WARNING'}, "No active layer")
            return {'CANCELLED'}
        
        current_frame = scene.frame_current
        
        # Find keyframe at or before current frame
        active_kf = None
        for kf in active_layer.frames:
            if kf.frame_number <= current_frame:
                if active_kf is None or kf.frame_number > active_kf.frame_number:
                    active_kf = kf
        
        if active_kf is None:
            self.report({'WARNING'}, "No keyframe found at or before current frame")
            return {'CANCELLED'}
        
        # Set anchor at cursor position
        cursor_pos = scene.cursor.location.copy()

        # Capture camera direction
        cam_dir = get_camera_direction(scene)

        set_anchor_for_frame(gp_obj, active_layer.name, active_kf.frame_number, cursor_pos, cam_dir)

        # Calculate anchor_local_offset
        gp_origin_world = gp_obj.matrix_world.to_translation()
        offset_world = cursor_pos - gp_origin_world
        anchor_local_offset = gp_obj.matrix_world.to_3x3().inverted() @ offset_world

        # Lock the frame (or update if already locked)
        if is_object_locked_at_frame(gp_obj, active_kf.frame_number):
            update_lock_anchor(gp_obj, active_kf.frame_number, cursor_pos, anchor_local_offset)
        else:
            # Lock the frame at this position
            matrix_local = gp_obj.matrix_local.copy()
            original_mpi = gp_obj.matrix_parent_inverse.copy()
            set_lock_for_frame(gp_obj, active_kf.frame_number, cursor_pos, anchor_local_offset, original_mpi, matrix_local)

        # Apply world lock with new anchor
        apply_object_world_lock_for_frame(gp_obj, scene)

        # Invalidate motion path so it updates
        invalidate_motion_path()

        self.report({'INFO'}, f"Anchor set and locked at frame {active_kf.frame_number}")

        # Redraw
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

        return {'FINISHED'}


class WONION_OT_auto_anchor(bpy.types.Operator):
    """Auto-calculate anchor from strokes (center XY, lowest Z)"""
    bl_idname = "world_onion.auto_anchor"
    bl_label = "Auto Anchor"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return get_active_gp(context) is not None

    def execute(self, context):
        settings = context.scene.world_onion
        gp_obj = get_active_gp(context)
        scene = context.scene

        if gp_obj is None or gp_obj.data is None:
            self.report({'WARNING'}, "No active GP object")
            return {'CANCELLED'}
        
        active_layer = gp_obj.data.layers.active
        if active_layer is None:
            self.report({'WARNING'}, "No active layer")
            return {'CANCELLED'}
        
        current_frame = scene.frame_current
        
        # Find keyframe at or before current frame
        active_kf = None
        for kf in active_layer.frames:
            if kf.frame_number <= current_frame:
                if active_kf is None or kf.frame_number > active_kf.frame_number:
                    active_kf = kf
        
        if active_kf is None:
            self.report({'WARNING'}, "No keyframe found at or before current frame")
            return {'CANCELLED'}
        
        # Calculate anchor from strokes
        anchor_pos = calculate_anchor_from_strokes(gp_obj, active_layer, active_kf.frame_number)
        
        if anchor_pos is None:
            self.report({'WARNING'}, "No strokes found to calculate anchor")
            return {'CANCELLED'}
        
        # Capture camera direction
        cam_dir = get_camera_direction(scene)

        set_anchor_for_frame(gp_obj, active_layer.name, active_kf.frame_number, anchor_pos, cam_dir)

        # Calculate anchor_local_offset
        gp_origin_world = gp_obj.matrix_world.to_translation()
        offset_world = anchor_pos - gp_origin_world
        anchor_local_offset = gp_obj.matrix_world.to_3x3().inverted() @ offset_world

        # Lock the frame (or update if already locked)
        if is_object_locked_at_frame(gp_obj, active_kf.frame_number):
            update_lock_anchor(gp_obj, active_kf.frame_number, anchor_pos, anchor_local_offset)
        else:
            # Lock the frame at this position
            matrix_local = gp_obj.matrix_local.copy()
            original_mpi = gp_obj.matrix_parent_inverse.copy()
            set_lock_for_frame(gp_obj, active_kf.frame_number, anchor_pos, anchor_local_offset, original_mpi, matrix_local)

        # Apply world lock with new anchor
        apply_object_world_lock_for_frame(gp_obj, scene)

        # Move cursor to anchor
        scene.cursor.location = anchor_pos

        # Invalidate motion path so it updates
        invalidate_motion_path()

        self.report({'INFO'}, f"Anchor auto-set and locked at frame {active_kf.frame_number}")

        # Redraw
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

        return {'FINISHED'}


class WONION_OT_clear_anchor(bpy.types.Operator):
    """Clear anchor for active layer's current keyframe"""
    bl_idname = "world_onion.clear_anchor"
    bl_label = "Clear Anchor"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return get_active_gp(context) is not None

    def execute(self, context):
        settings = context.scene.world_onion
        gp_obj = get_active_gp(context)
        scene = context.scene

        if gp_obj is None or gp_obj.data is None:
            self.report({'WARNING'}, "No active GP object")
            return {'CANCELLED'}
        
        active_layer = gp_obj.data.layers.active
        if active_layer is None:
            self.report({'WARNING'}, "No active layer")
            return {'CANCELLED'}
        
        current_frame = scene.frame_current
        
        # Find keyframe at or before current frame
        active_kf = None
        for kf in active_layer.frames:
            if kf.frame_number <= current_frame:
                if active_kf is None or kf.frame_number > active_kf.frame_number:
                    active_kf = kf
        
        if active_kf is None:
            self.report({'WARNING'}, "No keyframe found at or before current frame")
            return {'CANCELLED'}
        
        remove_anchor_for_frame(gp_obj, active_layer.name, active_kf.frame_number)
        
        self.report({'INFO'}, f"Anchor cleared for frame {active_kf.frame_number}")
        
        # Redraw
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        
        return {'FINISHED'}


class WONION_OT_clear_all_anchors(bpy.types.Operator):
    """Clear all anchors for this GP object"""
    bl_idname = "world_onion.clear_all_anchors"
    bl_label = "Clear All Anchors"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return get_active_gp(context) is not None

    def execute(self, context):
        gp_obj = get_active_gp(context)

        if gp_obj is None:
            self.report({'WARNING'}, "No active GP object")
            return {'CANCELLED'}
        
        set_anchors(gp_obj, {})
        
        self.report({'INFO'}, "All anchors cleared")
        
        # Redraw
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        
        return {'FINISHED'}


class WONION_OT_clear_all_locks(bpy.types.Operator):
    """Clear all world locks for this GP object and reset to normal parenting"""
    bl_idname = "world_onion.clear_all_locks"
    bl_label = "Clear All Locks"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return get_active_gp(context) is not None

    def execute(self, context):
        from .anchors import set_object_lock_data
        from .handlers import reset_object_world_lock

        gp_obj = get_active_gp(context)

        if gp_obj is None:
            self.report({'WARNING'}, "No active GP object")
            return {'CANCELLED'}

        # Clear all lock data
        set_object_lock_data(gp_obj, {})

        # Reset MPI to identity
        reset_object_world_lock(gp_obj, None)

        self.report({'INFO'}, "All world locks cleared, MPI reset to identity")

        # Redraw
        for area in context.screen.areas:
            if area.type in ('VIEW_3D', 'DOPESHEET_EDITOR'):
                area.tag_redraw()

        return {'FINISHED'}


class WONION_OT_reset_test_state(bpy.types.Operator):
    """Reset to clean state for testing: clear locks, anchors, cache, reset MPI"""
    bl_idname = "world_onion.reset_test_state"
    bl_label = "Reset Test State"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return get_active_gp(context) is not None

    def execute(self, context):
        from .anchors import set_object_lock_data, set_anchors
        from .handlers import reset_object_world_lock
        from mathutils import Matrix

        gp_obj = get_active_gp(context)

        if gp_obj is None:
            self.report({'WARNING'}, "No active GP object")
            return {'CANCELLED'}

        # Clear all lock data
        set_object_lock_data(gp_obj, {})

        # Clear all anchors
        set_anchors(gp_obj, {})

        # Clear cache
        clear_cache()

        # Reset MPI to identity
        gp_obj.matrix_parent_inverse = Matrix.Identity(4)

        self.report({'INFO'}, "Test state reset: locks, anchors, cache cleared; MPI = identity")

        # Redraw
        for area in context.screen.areas:
            if area.type in ('VIEW_3D', 'DOPESHEET_EDITOR'):
                area.tag_redraw()

        return {'FINISHED'}


class WONION_OT_snap_to_cursor(bpy.types.Operator):
    """Snap selected strokes to cursor position (preserves shape)"""
    bl_idname = "world_onion.snap_to_cursor"
    bl_label = "Snap to Cursor"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        gp_obj = get_active_gp(context)
        return gp_obj is not None and context.mode == 'EDIT_GREASE_PENCIL'

    def execute(self, context):
        gp_obj = get_active_gp(context)
        scene = context.scene

        if gp_obj is None or gp_obj.data is None:
            self.report({'WARNING'}, "No active GP object")
            return {'CANCELLED'}

        active_layer = gp_obj.data.layers.active
        if active_layer is None:
            self.report({'WARNING'}, "No active layer")
            return {'CANCELLED'}

        current_frame = scene.frame_current

        # Find keyframe at or before current frame
        active_kf = None
        for kf in active_layer.frames:
            if kf.frame_number <= current_frame:
                if active_kf is None or kf.frame_number > active_kf.frame_number:
                    active_kf = kf

        if active_kf is None or active_kf.drawing is None:
            self.report({'WARNING'}, "No keyframe found at or before current frame")
            return {'CANCELLED'}

        drawing = active_kf.drawing

        # Build transformation matrices
        world_matrix = gp_obj.matrix_world
        layer_matrix = get_layer_transform(active_layer)
        full_matrix = world_matrix @ layer_matrix
        full_matrix_inv = full_matrix.inverted()

        # Use compatibility API: drawing.strokes
        selected_strokes = [s for s in drawing.strokes if s.select]

        if not selected_strokes:
            self.report({'WARNING'}, "No strokes selected")
            return {'CANCELLED'}

        # Collect all world-space points from selected strokes
        world_points = []
        for stroke in selected_strokes:
            for point in stroke.points:
                local_pos = Vector(point.position)
                world_pos = full_matrix @ local_pos
                world_points.append(world_pos)

        if not world_points:
            self.report({'WARNING'}, "No points found in selected strokes")
            return {'CANCELLED'}

        # Calculate reference point (XY center, min Z)
        sum_x = sum(p.x for p in world_points)
        sum_y = sum(p.y for p in world_points)
        min_z = min(p.z for p in world_points)
        count = len(world_points)

        reference_point = Vector((sum_x / count, sum_y / count, min_z))

        # Get cursor position in world space
        cursor_pos = scene.cursor.location.copy()

        # Calculate offset in world space
        offset_world = cursor_pos - reference_point

        # Convert offset to local space
        offset_local = full_matrix_inv.to_3x3() @ offset_world

        # Apply offset to all points in selected strokes
        for stroke in selected_strokes:
            for point in stroke.points:
                old_pos = Vector(point.position)
                new_pos = old_pos + offset_local
                point.position = (new_pos.x, new_pos.y, new_pos.z)

        # Mark data as updated
        gp_obj.data.update_tag()

        # Set anchor at cursor position (where strokes now are)
        cam_dir = get_camera_direction(scene)
        set_anchor_for_frame(gp_obj, active_layer.name, active_kf.frame_number, cursor_pos, cam_dir)

        # Calculate anchor_local_offset
        gp_origin_world = gp_obj.matrix_world.to_translation()
        offset_world_anchor = cursor_pos - gp_origin_world
        anchor_local_offset = gp_obj.matrix_world.to_3x3().inverted() @ offset_world_anchor

        # Lock the frame (or update if already locked)
        if is_object_locked_at_frame(gp_obj, active_kf.frame_number):
            update_lock_anchor(gp_obj, active_kf.frame_number, cursor_pos, anchor_local_offset)
        else:
            # Lock the frame at this position
            matrix_local = gp_obj.matrix_local.copy()
            original_mpi = gp_obj.matrix_parent_inverse.copy()
            set_lock_for_frame(gp_obj, active_kf.frame_number, cursor_pos, anchor_local_offset, original_mpi, matrix_local)

        # Apply world lock with new anchor
        apply_object_world_lock_for_frame(gp_obj, scene)

        # Invalidate motion path so it updates
        invalidate_motion_path()

        # Redraw
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

        self.report({'INFO'}, f"Snapped {len(selected_strokes)} strokes to cursor and locked at frame {active_kf.frame_number}")
        return {'FINISHED'}


def get_selected_keyframe_frames(gp_obj):
    """Get all unique frame numbers that have selected keyframes.

    Returns sorted list of frame numbers.
    """
    frames = set()
    for layer in gp_obj.data.layers:
        for frame in layer.frames:
            if frame.select:
                frames.add(frame.frame_number)
    return sorted(frames)


class WONION_OT_toggle_world_lock(bpy.types.Operator):
    """Toggle world-space lock for current frame (object-level)"""
    bl_idname = "world_onion.toggle_world_lock"
    bl_label = "Toggle World Lock"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        gp_obj = get_active_gp(context)
        if gp_obj is None or gp_obj.data is None:
            return False
        # Check if there are any keyframes at or before current frame
        current_frame = context.scene.frame_current
        for layer in gp_obj.data.layers:
            for kf in layer.frames:
                if kf.frame_number <= current_frame:
                    return True
        return False

    def execute(self, context):
        gp_obj = get_active_gp(context)
        scene = context.scene

        if gp_obj is None or gp_obj.data is None:
            self.report({'WARNING'}, "No active GP object")
            return {'CANCELLED'}

        # Check for selected keyframes in dopesheet
        selected_frames = get_selected_keyframe_frames(gp_obj)

        if len(selected_frames) > 0:
            # Multi-select mode: operate on all selected frame numbers
            return self.execute_multi(context, gp_obj, selected_frames)
        else:
            # Single-frame mode: operate on visible keyframe at current frame
            return self.execute_single(context, gp_obj, scene)

    def execute_single(self, context, gp_obj, scene):
        """Toggle lock for the visible keyframe at current frame."""
        current_frame = scene.frame_current

        # Find the visible keyframe (at or before current frame)
        visible_frame = None
        for layer in gp_obj.data.layers:
            for kf in layer.frames:
                if kf.frame_number <= current_frame:
                    if visible_frame is None or kf.frame_number > visible_frame:
                        visible_frame = kf.frame_number

        if visible_frame is None:
            self.report({'WARNING'}, "No keyframe found")
            return {'CANCELLED'}

        currently_locked = is_object_locked_at_frame(gp_obj, visible_frame)

        if currently_locked:
            # Unlock
            print(f"\n=== UNLOCK DEBUG ===")
            print(f"Unlocking frame {visible_frame}")
            remove_lock_for_frame(gp_obj, visible_frame)
            reset_object_world_lock(gp_obj)
            print(f"=== END UNLOCK DEBUG ===\n")
            self.report({'INFO'}, f"World lock OFF for frame {visible_frame}")
        else:
            # Lock - store anchor data for pivot-based billboard effect
            print(f"\n=== LOCK DEBUG ===")
            print(f"Locking frame {visible_frame}")
            original_frame = scene.frame_current
            frame_changed = visible_frame != original_frame
            lock_payload = None

            try:
                if frame_changed:
                    scene.frame_set(visible_frame)

                # Ensure matrices are up to date for the visible keyframe
                context.view_layer.update()

                if gp_obj.parent is None:
                    self.report({'WARNING'}, "Cannot lock - GP has no parent")
                    return {'CANCELLED'}

                # Calculate anchor (stroke center) for pivot-based rotation
                # Try to get anchor from active layer's strokes
                active_layer = gp_obj.data.layers.active
                anchor_world = None
                if active_layer is not None:
                    anchor_world = calculate_anchor_from_strokes(gp_obj, active_layer, visible_frame)

                # Fallback to GP origin if no strokes
                gp_origin_world = gp_obj.matrix_world.to_translation()
                if anchor_world is None:
                    anchor_world = gp_origin_world.copy()

                # Calculate anchor offset in GP local coordinates
                offset_world = anchor_world - gp_origin_world
                anchor_local_offset = gp_obj.matrix_world.to_3x3().inverted() @ offset_world

                matrix_local = gp_obj.matrix_local.copy()
                original_mpi = gp_obj.matrix_parent_inverse.copy()

                print(f"anchor_world: {anchor_world}")
                print(f"anchor_local_offset: {anchor_local_offset}")
                print(f"matrix_local:\n{matrix_local}")

                # Store the lock data captured from the visible keyframe
                set_lock_for_frame(
                    gp_obj,
                    visible_frame,
                    anchor_world,
                    anchor_local_offset,
                    original_mpi,
                    matrix_local,
                )

                lock_payload = (anchor_world, anchor_local_offset, matrix_local)

            finally:
                if frame_changed:
                    scene.frame_set(original_frame)
                # Refresh depsgraph at the user's original frame
                context.view_layer.update()

            if lock_payload is None:
                return {'CANCELLED'}

            anchor_world, anchor_local_offset, matrix_local = lock_payload

            # Apply the lock immediately (with pivot-based billboard effect)
            apply_world_lock_from_stored(gp_obj, anchor_world, anchor_local_offset, matrix_local)

            # Force update and verify anchor position stays fixed
            context.view_layer.update()

            print(f"=== END LOCK DEBUG ===\n")
            self.report({'INFO'}, f"World lock ON for frame {visible_frame}")

        self._redraw_viewports(context)
        return {'FINISHED'}

    def execute_multi(self, context, gp_obj, selected_frames):
        """Multi-select mode: lock/unlock all selected frames.

        Each frame gets locked with its OWN anchor position (calculated from
        strokes at that frame), not a shared anchor.
        """
        scene = context.scene

        # Determine action: if ANY are unlocked, lock all; if ALL locked, unlock all
        any_unlocked = False
        for frame_num in selected_frames:
            if not is_object_locked_at_frame(gp_obj, frame_num):
                any_unlocked = True
                break

        lock_count = 0
        unlock_count = 0

        if any_unlocked:
            # Lock all selected frames - each with its own anchor

            if gp_obj.parent is None:
                self.report({'WARNING'}, "Cannot lock - GP has no parent")
                return {'CANCELLED'}

            original_frame = scene.frame_current
            original_mpi = gp_obj.matrix_parent_inverse.copy()
            active_layer = gp_obj.data.layers.active

            try:
                for frame_num in selected_frames:
                    if is_object_locked_at_frame(gp_obj, frame_num):
                        continue  # Skip already locked

                    # Jump to this frame to get correct matrices and strokes
                    scene.frame_set(frame_num)
                    context.view_layer.update()

                    # Calculate anchor for THIS frame's strokes
                    anchor_world = None
                    if active_layer is not None:
                        anchor_world = calculate_anchor_from_strokes(gp_obj, active_layer, frame_num)

                    # Fallback to GP origin if no strokes
                    gp_origin_world = gp_obj.matrix_world.to_translation()
                    if anchor_world is None:
                        anchor_world = gp_origin_world.copy()

                    # Calculate anchor offset in GP local coordinates
                    offset_world = anchor_world - gp_origin_world
                    anchor_local_offset = gp_obj.matrix_world.to_3x3().inverted() @ offset_world

                    matrix_local = gp_obj.matrix_local.copy()

                    # Store frame-specific lock data
                    set_lock_for_frame(gp_obj, frame_num, anchor_world, anchor_local_offset, original_mpi, matrix_local)
                    lock_count += 1

            finally:
                # Return to original frame
                scene.frame_set(original_frame)
                context.view_layer.update()

            # Apply lock for current frame
            apply_object_world_lock_for_frame(gp_obj, scene)
            self.report({'INFO'}, f"World lock ON for {lock_count} frames")
        else:
            # Unlock all selected frames
            for frame_num in selected_frames:
                remove_lock_for_frame(gp_obj, frame_num)
                unlock_count += 1
            # Reset to normal parenting
            reset_object_world_lock(gp_obj)
            self.report({'INFO'}, f"World lock OFF for {unlock_count} frames")

        self._redraw_viewports(context)
        return {'FINISHED'}

    def _redraw_viewports(self, context):
        """Request redraw of 3D viewports and timeline."""
        for area in context.screen.areas:
            if area.type in ('VIEW_3D', 'DOPESHEET_EDITOR', 'TIMELINE'):
                area.tag_redraw()


class WONION_OT_reload_addon(bpy.types.Operator):
    """Reload the addon (for development)"""
    bl_idname = "world_onion.reload_addon"
    bl_label = "Reload Addon"
    bl_options = {'REGISTER'}

    def execute(self, context):
        import importlib
        import sys

        # Auto-detect addon name from this module's package
        addon_name = __package__.split('.')[0] if '.' in __package__ else __package__
        print(f"Reloading addon: {addon_name}")

        # Get the main module
        if addon_name not in sys.modules:
            self.report({'ERROR'}, f"Addon module '{addon_name}' not found in sys.modules")
            return {'CANCELLED'}

        main_module = sys.modules[addon_name]

        # Preserve current settings before unregister
        saved_settings = {}
        if hasattr(context.scene, 'world_onion'):
            settings = context.scene.world_onion
            saved_settings = {
                'enabled': settings.enabled,
                'mode': settings.mode,
                'frames_before': settings.frames_before,
                'frames_after': settings.frames_after,
                'frame_step': settings.frame_step,
                'opacity': settings.opacity,
                'fill_opacity': getattr(settings, 'fill_opacity', 0.25),
                'falloff': settings.falloff,
                'color_before': tuple(settings.color_before),
                'color_after': tuple(settings.color_after),
                'line_width': settings.line_width,
                'skip_underscore': settings.skip_underscore,
                'layer_filter': settings.layer_filter,
                'anchor_enabled': settings.anchor_enabled,
                'anchor_auto_cursor': settings.anchor_auto_cursor,
                'anchor_snap_to_stroke': settings.anchor_snap_to_stroke,
                'anchor_show_indicators': settings.anchor_show_indicators,
                'world_lock_inherit': settings.world_lock_inherit,
            }

        # Step 1: Unregister the addon (cleanup handlers, classes, etc.)
        try:
            main_module.unregister()
        except Exception as e:
            self.report({'WARNING'}, f"Unregister warning: {e}")

        # Step 2: Get all submodules to reload
        modules_to_reload = [
            name for name in sys.modules.keys()
            if name.startswith(addon_name) and name != addon_name
        ]

        # Sort to reload submodules first (deeper modules before shallower)
        modules_to_reload.sort(key=lambda x: x.count('.'), reverse=True)

        # Step 3: Reload all submodules
        for mod_name in modules_to_reload:
            try:
                importlib.reload(sys.modules[mod_name])
            except Exception as e:
                self.report({'WARNING'}, f"Failed to reload {mod_name}: {e}")

        # Step 4: Reload main module last (so it picks up reloaded submodules)
        try:
            importlib.reload(main_module)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to reload main module: {e}")
            return {'CANCELLED'}

        # Step 5: Re-register the addon with fresh classes
        try:
            main_module.register()
        except Exception as e:
            self.report({'ERROR'}, f"Failed to re-register: {e}")
            return {'CANCELLED'}

        # Step 6: Restore saved settings
        if saved_settings and hasattr(context.scene, 'world_onion'):
            new_settings = context.scene.world_onion
            for key, value in saved_settings.items():
                try:
                    if hasattr(new_settings, key):
                        setattr(new_settings, key, value)
                except Exception:
                    pass  # Skip properties that fail (e.g., invalid object references)

        self.report({'INFO'}, f"Reloaded {len(modules_to_reload) + 1} modules")

        # Redraw
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

        return {'FINISHED'}


# List of operator classes for registration
operator_classes = (
    WONION_OT_clear_cache,
    WONION_OT_build_cache,
    WONION_OT_set_anchor,
    WONION_OT_auto_anchor,
    WONION_OT_clear_anchor,
    WONION_OT_clear_all_anchors,
    WONION_OT_clear_all_locks,
    WONION_OT_reset_test_state,
    WONION_OT_snap_to_cursor,
    WONION_OT_toggle_world_lock,
    WONION_OT_reload_addon,
)
