"""
Operators for world-space onion skinning.
"""

import bpy
from mathutils import Vector, Matrix

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
    calculate_anchor_local_offset,
    # Object-level lock functions
    is_object_locked_at_frame,
    get_lock_for_frame,
    set_lock_for_frame,
    remove_lock_for_frame,
    update_lock_anchor,
    get_object_lock_data,
    set_object_lock_data,
)
from .transforms import get_layer_transform, get_camera_direction, align_canvas_to_cursor
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

        # If we're at an interpolated frame (not exactly on a keyframe),
        # duplicate to create a new keyframe at current frame
        created_new_kf = False
        if active_kf.frame_number != current_frame:
            # Copy the keyframe to current frame (API: copy(from_frame, to_frame))
            new_kf = active_layer.frames.copy(active_kf.frame_number, current_frame)
            active_kf = new_kf
            created_new_kf = True

        # Set anchor at cursor position
        cursor_pos = scene.cursor.location.copy()

        # Get stroke local center for proper pivot calculation
        _, anchor_local = calculate_anchor_from_strokes(
            gp_obj, active_layer, active_kf.frame_number, return_local=True
        )
        if anchor_local is None:
            self.report({'WARNING'}, "No strokes found")
            return {'CANCELLED'}

        # Capture camera direction
        cam_dir = get_camera_direction(scene)

        set_anchor_for_frame(gp_obj, active_layer.name, active_kf.frame_number, cursor_pos, cam_dir)

        # Use stroke local center as offset, cursor as world anchor
        # This makes strokes "snap" to cursor position
        anchor_local_offset, matrix_local = calculate_anchor_local_offset(gp_obj, anchor_local)

        # Lock the frame (or update if already locked)
        if is_object_locked_at_frame(gp_obj, active_kf.frame_number):
            update_lock_anchor(gp_obj, active_kf.frame_number, cursor_pos, anchor_local_offset, matrix_local)
        else:
            # Lock the frame at this position
            original_mpi = gp_obj.matrix_parent_inverse.copy()
            set_lock_for_frame(gp_obj, active_kf.frame_number, cursor_pos, anchor_local_offset, original_mpi, matrix_local)

        # Apply world lock with new anchor
        apply_object_world_lock_for_frame(gp_obj, scene)
        context.view_layer.update()

        # Invalidate motion path so it updates
        invalidate_motion_path()

        msg = f"Anchor set and locked at frame {active_kf.frame_number}"
        if created_new_kf:
            msg += " (new keyframe created)"
        self.report({'INFO'}, msg)

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

        # CRITICAL: Calculate anchor BEFORE creating new keyframe!
        # On interpolated frames, matrix_world is at the interpolated position.
        # Creating a keyframe triggers depsgraph update which resets matrix_world.
        # We must capture the correct world position FIRST.
        anchor_pos, anchor_local = calculate_anchor_from_strokes(
            gp_obj, active_layer, active_kf.frame_number, return_local=True
        )

        if anchor_pos is None:
            self.report({'WARNING'}, "No strokes found to calculate anchor")
            return {'CANCELLED'}

        # NOW create new keyframe if needed (after anchor is calculated)
        created_new_kf = False
        if active_kf.frame_number != current_frame:
            # Copy the keyframe to current frame (API: copy(from_frame, to_frame))
            new_kf = active_layer.frames.copy(active_kf.frame_number, current_frame)
            active_kf = new_kf
            created_new_kf = True

        # Capture camera direction
        cam_dir = get_camera_direction(scene)

        set_anchor_for_frame(gp_obj, active_layer.name, active_kf.frame_number, anchor_pos, cam_dir)

        # Use the raw local anchor position (stable regardless of MPI)
        # IMPORTANT: If this frame is NOT YET locked, we're coming from an interpolated position.
        # The current matrix_local has the interpolation offset baked in, so we need to use
        # the SOURCE locked frame's matrix_local instead.
        source_matrix_local = None
        if not is_object_locked_at_frame(gp_obj, active_kf.frame_number):
            # This frame isn't locked yet - find the source locked frame's matrix_local
            # Use frame BEFORE current to find the source (since current frame now has a keyframe)
            from .anchors import get_all_locked_frames, get_lock_for_frame
            locked_frames = get_all_locked_frames(gp_obj)
            # Find the locked frame at or before current frame
            source_locked_frame = None
            for lf in sorted(locked_frames, reverse=True):
                if lf < current_frame:
                    source_locked_frame = lf
                    break
            if source_locked_frame is not None:
                source_lock_data = get_lock_for_frame(gp_obj, source_locked_frame)
                if source_lock_data and "matrix_local" in source_lock_data:
                    source_matrix_local = Matrix(source_lock_data["matrix_local"])

        anchor_local_offset, matrix_local = calculate_anchor_local_offset(gp_obj, anchor_local)
        if source_matrix_local is not None:
            matrix_local = source_matrix_local

        # Get original MPI to store (for unlock restore)
        from .anchors import get_object_lock_data
        lock_data = get_object_lock_data(gp_obj)
        original_mpi = None
        for frame_str, data in lock_data.items():
            if isinstance(data, dict) and "original_parent_inverse" in data:
                original_mpi = Matrix(data["original_parent_inverse"])
                break
        if original_mpi is None:
            original_mpi = gp_obj.matrix_parent_inverse.copy()

        # Lock the frame (or update if already locked)
        if is_object_locked_at_frame(gp_obj, active_kf.frame_number):
            update_lock_anchor(gp_obj, active_kf.frame_number, anchor_pos, anchor_local_offset, matrix_local)
        else:
            set_lock_for_frame(gp_obj, active_kf.frame_number, anchor_pos, anchor_local_offset, original_mpi, matrix_local)

        # Apply the lock - positions strokes at anchor_pos
        apply_object_world_lock_for_frame(gp_obj, scene)
        context.view_layer.update()

        # Move cursor to anchor
        scene.cursor.location = anchor_pos

        # Invalidate motion path so it updates
        invalidate_motion_path()

        msg = f"Anchor auto-set and locked at frame {active_kf.frame_number}"
        if created_new_kf:
            msg += " (new keyframe created)"
        self.report({'INFO'}, msg)

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
    """Snap selected strokes to cursor position (preserves shape). Creates new keyframe if at interpolated frame."""
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

        # Remember selected stroke indices before potentially copying keyframe
        selected_indices = [i for i, s in enumerate(drawing.strokes) if s.select]

        # If we're at an interpolated frame (not exactly on a keyframe),
        # duplicate to create a new keyframe at current frame
        created_new_kf = False
        if active_kf.frame_number != current_frame:
            # Copy the keyframe to current frame (API: copy(from_frame, to_frame))
            new_kf = active_layer.frames.copy(active_kf.frame_number, current_frame)
            active_kf = new_kf
            created_new_kf = True
            # Get drawing from new keyframe
            drawing = active_kf.drawing
            # Restore selection on copied strokes
            selected_strokes = []
            for i, stroke in enumerate(drawing.strokes):
                if i in selected_indices:
                    stroke.select = True
                    selected_strokes.append(stroke)
            # Update matrices for the new keyframe context
            full_matrix = world_matrix @ layer_matrix
            full_matrix_inv = full_matrix.inverted()

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

        # Recalculate anchor from moved strokes (now at cursor position)
        _, anchor_local = calculate_anchor_from_strokes(
            gp_obj, active_layer, active_kf.frame_number, return_local=True
        )
        if anchor_local is None:
            # Fallback - shouldn't happen since we just moved strokes
            anchor_local = Vector((0, 0, 0))

        # Use raw local anchor position (stable regardless of MPI)
        # IMPORTANT: If this frame is NOT YET locked, we're coming from an interpolated position.
        # The current matrix_local has the interpolation offset baked in, so we need to use
        # the SOURCE locked frame's matrix_local instead.
        source_matrix_local = None
        if not is_object_locked_at_frame(gp_obj, active_kf.frame_number):
            # This frame isn't locked yet - find the source locked frame's matrix_local
            from .anchors import get_all_locked_frames
            locked_frames = get_all_locked_frames(gp_obj)
            # Find the locked frame before current frame
            source_locked_frame = None
            for lf in sorted(locked_frames, reverse=True):
                if lf < current_frame:
                    source_locked_frame = lf
                    break
            if source_locked_frame is not None:
                source_lock_data = get_lock_for_frame(gp_obj, source_locked_frame)
                if source_lock_data and "matrix_local" in source_lock_data:
                    source_matrix_local = Matrix(source_lock_data["matrix_local"])

        anchor_local_offset, matrix_local = calculate_anchor_local_offset(gp_obj, anchor_local)
        if source_matrix_local is not None:
            matrix_local = source_matrix_local

        # Lock the frame (or update if already locked)
        if is_object_locked_at_frame(gp_obj, active_kf.frame_number):
            update_lock_anchor(gp_obj, active_kf.frame_number, cursor_pos, anchor_local_offset, matrix_local)
        else:
            # Get original MPI from existing locks (for unlock restore)
            from .anchors import get_object_lock_data
            lock_data = get_object_lock_data(gp_obj)
            original_mpi = None
            for frame_str, data in lock_data.items():
                if isinstance(data, dict) and "original_parent_inverse" in data:
                    original_mpi = Matrix(data["original_parent_inverse"])
                    break
            if original_mpi is None:
                original_mpi = gp_obj.matrix_parent_inverse.copy()
            set_lock_for_frame(gp_obj, active_kf.frame_number, cursor_pos, anchor_local_offset, original_mpi, matrix_local)

        # Apply world lock with new anchor
        apply_object_world_lock_for_frame(gp_obj, scene)

        # Force depsgraph update to ensure matrix_world is recalculated
        # This prevents issues with stale transforms on subsequent operations
        context.view_layer.update()

        # Invalidate motion path so it updates
        invalidate_motion_path()

        # Redraw
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

        msg = f"Snapped {len(selected_strokes)} strokes to cursor and locked at frame {active_kf.frame_number}"
        if created_new_kf:
            msg += " (new keyframe created)"
        self.report({'INFO'}, msg)
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
            # Unlock - return to vanilla camera-parented behavior
            # Keep anchor_world data for potential re-locking

            all_lock_data = get_object_lock_data(gp_obj)
            frame_str = str(visible_frame)
            frame_data = all_lock_data.get(frame_str, {})

            # Get original_parent_inverse to restore vanilla behavior
            if isinstance(frame_data, dict) and "original_parent_inverse" in frame_data:
                original_mpi = Matrix(frame_data["original_parent_inverse"])
            else:
                original_mpi = Matrix.Identity(4)

            # Mark as unlocked but keep anchor_world for re-locking
            if isinstance(frame_data, dict):
                frame_data["world_locked"] = False
                all_lock_data[frame_str] = frame_data
                set_object_lock_data(gp_obj, all_lock_data)
            else:
                remove_lock_for_frame(gp_obj, visible_frame)

            # Restore original MPI - vanilla camera-parented behavior
            gp_obj.matrix_parent_inverse = original_mpi

            self.report({'INFO'}, f"World lock OFF for frame {visible_frame}")
        else:
            # Lock - check if we have existing lock data to restore (even if currently unlocked)
            all_lock_data = get_object_lock_data(gp_obj)
            frame_str = str(visible_frame)
            existing_data = all_lock_data.get(frame_str)

            if existing_data and isinstance(existing_data, dict) and "anchor_world" in existing_data:
                # Restore existing lock data (re-locking after unlock)
                all_lock_data[frame_str]["world_locked"] = True
                set_object_lock_data(gp_obj, all_lock_data)

                # Apply the restored lock (already imported at top of file)
                apply_world_lock_from_stored(
                    gp_obj,
                    existing_data["anchor_world"],
                    existing_data["anchor_local_offset"],
                    existing_data["matrix_local"]
                )

                # Restore cursor position and canvas alignment
                anchor_world = existing_data["anchor_world"]
                scene.cursor.location = Vector(anchor_world)
                try:
                    align_canvas_to_cursor(context)
                except (RuntimeError, AttributeError):
                    pass

                self.report({'INFO'}, f"World lock RESTORED for frame {visible_frame}")
            else:
                # New lock - calculate anchor data for pivot-based billboard effect
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
                    anchor_local = None
                    if active_layer is not None:
                        anchor_world, anchor_local = calculate_anchor_from_strokes(
                            gp_obj, active_layer, visible_frame, return_local=True
                        )

                    # Fallback to GP origin if no strokes
                    if anchor_world is None:
                        anchor_world = gp_obj.matrix_world.to_translation().copy()
                        anchor_local = Vector((0, 0, 0))

                    # Use raw local anchor position (stable regardless of MPI)
                    anchor_local_offset, matrix_local = calculate_anchor_local_offset(gp_obj, anchor_local)
                    original_mpi = gp_obj.matrix_parent_inverse.copy()

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

                self.report({'INFO'}, f"World lock ON for frame {visible_frame}")

        # Invalidate motion path cache so dashed/solid segments update
        invalidate_motion_path()
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

                    # Check if we have existing lock data to restore (even if currently unlocked)
                    all_lock_data = get_object_lock_data(gp_obj)
                    frame_str = str(frame_num)
                    existing_data = all_lock_data.get(frame_str)
                    if existing_data and isinstance(existing_data, dict) and "anchor_world" in existing_data:
                        # Restore existing lock data (re-locking after unlock)
                        all_lock_data[frame_str]["world_locked"] = True
                        set_object_lock_data(gp_obj, all_lock_data)
                        lock_count += 1
                        continue

                    # New lock - jump to this frame to get correct matrices and strokes
                    scene.frame_set(frame_num)
                    context.view_layer.update()

                    # Calculate anchor for THIS frame's strokes
                    anchor_world = None
                    anchor_local = None
                    if active_layer is not None:
                        anchor_world, anchor_local = calculate_anchor_from_strokes(
                            gp_obj, active_layer, frame_num, return_local=True
                        )

                    # Fallback to GP origin if no strokes
                    if anchor_world is None:
                        anchor_world = gp_obj.matrix_world.to_translation().copy()
                        anchor_local = Vector((0, 0, 0))

                    # Use raw local anchor position (stable regardless of MPI)
                    anchor_local_offset, matrix_local = calculate_anchor_local_offset(gp_obj, anchor_local)

                    # Store frame-specific lock data
                    set_lock_for_frame(gp_obj, frame_num, anchor_world, anchor_local_offset, original_mpi, matrix_local)
                    lock_count += 1

            finally:
                # Return to original frame
                scene.frame_set(original_frame)
                context.view_layer.update()

            # Apply lock for current frame
            apply_object_world_lock_for_frame(gp_obj, scene)
            context.view_layer.update()

            # Restore cursor/canvas for the visible frame
            visible_frame = None
            for layer in gp_obj.data.layers:
                for kf in layer.frames:
                    if kf.frame_number <= original_frame:
                        if visible_frame is None or kf.frame_number > visible_frame:
                            visible_frame = kf.frame_number
            if visible_frame is not None:
                all_lock_data = get_object_lock_data(gp_obj)
                frame_str = str(visible_frame)
                frame_data = all_lock_data.get(frame_str, {})
                if isinstance(frame_data, dict) and "anchor_world" in frame_data and frame_data.get("world_locked", False):
                    scene.cursor.location = Vector(frame_data["anchor_world"])
                    try:
                        align_canvas_to_cursor(context)
                    except (RuntimeError, AttributeError):
                        pass

            self.report({'INFO'}, f"World lock ON for {lock_count} frames")
        else:
            # Unlock all selected frames - return to vanilla camera-parented behavior
            # Keep anchor_world data for potential re-locking
            all_lock_data = get_object_lock_data(gp_obj)

            # Get original_parent_inverse (same for all frames)
            original_mpi = Matrix.Identity(4)
            for frame_str, data in all_lock_data.items():
                if isinstance(data, dict) and "original_parent_inverse" in data:
                    original_mpi = Matrix(data["original_parent_inverse"])
                    break

            for frame_num in selected_frames:
                frame_str = str(frame_num)
                frame_data = all_lock_data.get(frame_str, {})

                if not isinstance(frame_data, dict):
                    continue

                # Mark as unlocked but keep anchor_world for re-locking
                frame_data["world_locked"] = False
                all_lock_data[frame_str] = frame_data
                unlock_count += 1

            # Save all changes in one go
            set_object_lock_data(gp_obj, all_lock_data)

            # Restore original MPI - vanilla camera-parented behavior
            gp_obj.matrix_parent_inverse = original_mpi

            self.report({'INFO'}, f"World lock OFF for {unlock_count} frames")

        # Invalidate motion path cache so dashed/solid segments update
        invalidate_motion_path()
        self._redraw_viewports(context)
        return {'FINISHED'}

    def _redraw_viewports(self, context):
        """Request redraw of 3D viewports and timeline."""
        for area in context.screen.areas:
            if area.type in ('VIEW_3D', 'DOPESHEET_EDITOR', 'TIMELINE'):
                area.tag_redraw()


class WONION_OT_reload_addon(bpy.types.Operator):
    """Reload the addon (for development) - robust version that handles GPU cleanup"""
    bl_idname = "world_onion.reload_addon"
    bl_label = "Reload Addon"
    bl_options = {'REGISTER'}

    def execute(self, context):
        import importlib
        import sys

        addon_name = __package__.split('.')[0] if '.' in __package__ else __package__
        print(f"\n{'='*50}")
        print(f"RELOADING ADDON: {addon_name}")
        print(f"{'='*50}")

        if addon_name not in sys.modules:
            self.report({'ERROR'}, f"Addon module '{addon_name}' not found")
            return {'CANCELLED'}

        main_module = sys.modules[addon_name]

        # Step 1: Save settings BEFORE any cleanup
        saved_settings = self._save_settings(context)

        # Step 2: Force cleanup of ALL GPU handlers and state BEFORE unregister
        # This prevents crashes from stale GPU references
        self._force_cleanup_gpu_handlers(addon_name)

        # Step 3: Clear all module-level caches and state
        self._clear_module_state(addon_name)

        # Step 4: Unregister addon
        try:
            main_module.unregister()
            print("  ✓ Unregistered addon")
        except Exception as e:
            print(f"  ! Unregister warning: {e}")

        # Step 5: Get submodules in correct reload order (dependencies first)
        modules_to_reload = self._get_ordered_submodules(addon_name)

        # Step 6: Reload all submodules
        reload_count = 0
        for mod_name in modules_to_reload:
            if mod_name in sys.modules:
                try:
                    importlib.reload(sys.modules[mod_name])
                    reload_count += 1
                except Exception as e:
                    print(f"  ! Failed to reload {mod_name}: {e}")

        # Step 7: Reload main module
        try:
            importlib.reload(main_module)
            reload_count += 1
            print(f"  ✓ Reloaded {reload_count} modules")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to reload main module: {e}")
            return {'CANCELLED'}

        # Step 8: Re-register
        try:
            main_module.register()
            print("  ✓ Re-registered addon")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to re-register: {e}")
            return {'CANCELLED'}

        # Step 9: Restore settings
        self._restore_settings(context, saved_settings)

        # Step 10: Force redraw all areas
        self._force_redraw_all(context)

        print(f"{'='*50}")
        print("RELOAD COMPLETE")
        print(f"{'='*50}\n")

        self.report({'INFO'}, f"Reloaded {reload_count} modules successfully")
        return {'FINISHED'}

    def _save_settings(self, context):
        """Save all current settings to restore after reload."""
        saved = {}
        if hasattr(context.scene, 'world_onion'):
            settings = context.scene.world_onion
            # Get all properties dynamically
            for prop_name in dir(settings):
                if prop_name.startswith('_') or prop_name.startswith('bl_'):
                    continue
                try:
                    val = getattr(settings, prop_name)
                    # Only save simple types
                    if isinstance(val, (bool, int, float, str)):
                        saved[prop_name] = val
                    elif hasattr(val, '__iter__') and not isinstance(val, str):
                        # Color/vector properties
                        saved[prop_name] = tuple(val)
                except Exception:
                    pass
        return saved

    def _restore_settings(self, context, saved_settings):
        """Restore saved settings after reload."""
        if not saved_settings:
            return
        if not hasattr(context.scene, 'world_onion'):
            return

        new_settings = context.scene.world_onion
        for key, value in saved_settings.items():
            try:
                if hasattr(new_settings, key):
                    setattr(new_settings, key, value)
            except Exception:
                pass

    def _force_cleanup_gpu_handlers(self, addon_name):
        """Forcefully remove ALL draw handlers to prevent GPU crashes."""
        import sys

        # Clean up drawing.py handlers
        drawing_module = f"{addon_name}.drawing"
        if drawing_module in sys.modules:
            drawing = sys.modules[drawing_module]
            # Call unregister directly on the module
            if hasattr(drawing, 'unregister_draw_handlers'):
                try:
                    drawing.unregister_draw_handlers()
                    print("  ✓ Cleaned up viewport draw handlers")
                except Exception as e:
                    print(f"  ! Draw handler cleanup: {e}")
            # Also clear any cached GPU batches/shaders
            if hasattr(drawing, '_motion_path_cache'):
                drawing._motion_path_cache = None
            if hasattr(drawing, '_motion_path_dirty'):
                drawing._motion_path_dirty = True

        # Clean up timeline_drawing.py handlers
        timeline_module = f"{addon_name}.timeline_drawing"
        if timeline_module in sys.modules:
            timeline = sys.modules[timeline_module]
            if hasattr(timeline, 'unregister_timeline_handlers'):
                try:
                    timeline.unregister_timeline_handlers()
                    print("  ✓ Cleaned up timeline draw handlers")
                except Exception as e:
                    print(f"  ! Timeline handler cleanup: {e}")

        # Clean up handlers.py event handlers
        handlers_module = f"{addon_name}.handlers"
        if handlers_module in sys.modules:
            handlers = sys.modules[handlers_module]
            if hasattr(handlers, 'unregister_handlers'):
                try:
                    handlers.unregister_handlers()
                    print("  ✓ Cleaned up event handlers")
                except Exception as e:
                    print(f"  ! Event handler cleanup: {e}")

    def _clear_module_state(self, addon_name):
        """Clear module-level caches and global state."""
        import sys

        # Clear cache
        cache_module = f"{addon_name}.cache"
        if cache_module in sys.modules:
            cache = sys.modules[cache_module]
            if hasattr(cache, 'clear_cache'):
                try:
                    cache.clear_cache()
                except Exception:
                    pass
            if hasattr(cache, '_cache'):
                cache._cache = {}

        # Reset handler state
        handlers_module = f"{addon_name}.handlers"
        if handlers_module in sys.modules:
            handlers = sys.modules[handlers_module]
            for attr in ['_last_keyframe_set', '_last_active_layer_name',
                         '_last_active_gp', '_in_depsgraph_handler']:
                if hasattr(handlers, attr):
                    try:
                        if attr == '_last_keyframe_set':
                            setattr(handlers, attr, set())
                        elif attr == '_in_depsgraph_handler':
                            setattr(handlers, attr, False)
                        else:
                            setattr(handlers, attr, None)
                    except Exception:
                        pass

    def _get_ordered_submodules(self, addon_name):
        """Get submodules in dependency order (leaves first, then parents)."""
        import sys

        submodules = [
            name for name in sys.modules.keys()
            if name.startswith(addon_name + '.') and name != addon_name
        ]

        # Order: deepest first, then by dependency
        # transforms, cache, anchors, handlers, drawing, timeline_drawing, operators, settings, ui
        priority = {
            'transforms': 0,
            'cache': 1,
            'anchors': 2,
            'handlers': 3,
            'drawing': 4,
            'timeline_drawing': 5,
            'operators': 6,
            'settings': 7,
            'ui': 8,
        }

        def sort_key(mod_name):
            base = mod_name.split('.')[-1]
            return priority.get(base, 50)

        submodules.sort(key=sort_key)
        return submodules

    def _force_redraw_all(self, context):
        """Force redraw of all relevant areas."""
        try:
            for window in context.window_manager.windows:
                for area in window.screen.areas:
                    area.tag_redraw()
        except Exception:
            pass


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
