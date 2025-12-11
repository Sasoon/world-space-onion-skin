"""
Operators for world-space onion skinning.
"""

import time

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
from .transforms import get_layer_transform, get_camera_direction, align_canvas_to_cursor, ensure_billboard_constraint, align_strokes_to_camera
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
    """Background operator to sync cursor with GP object and manage canvas visibility"""
    bl_idname = "world_onion.cursor_sync"
    bl_label = "Cursor Sync"
    bl_options = {'INTERNAL'}

    # Instance variables (reset in execute)
    _timer = None
    _last_frame = None
    _last_frame_time = None
    _is_animating = False  # Combined play+scrub state
    _scrub_cooldown = 0.0

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
            current_frame = context.scene.frame_current
            current_time = time.time()

            # Detect playback state
            try:
                is_playing = context.screen.is_animation_playing
            except (AttributeError, RuntimeError):
                is_playing = False

            # Detect scrubbing (rapid frame changes when NOT in playback)
            is_scrubbing = False
            if not is_playing:
                if current_frame != self._last_frame:
                    if self._last_frame_time is not None:
                        delta = current_time - self._last_frame_time
                        is_scrubbing = delta < 0.1  # <100ms = scrubbing
                    self._last_frame_time = current_time

            # Cooldown: keep scrubbing state for 200ms after last rapid change
            if is_scrubbing:
                self._scrub_cooldown = 0.2
            elif self._scrub_cooldown > 0:
                self._scrub_cooldown -= 0.016
                is_scrubbing = True

            # Combined animation state
            is_animating = is_playing or is_scrubbing

            # State transitions: hide/show canvas
            if is_animating and not self._is_animating:
                self._hide_canvas(context)
            elif not is_animating and self._is_animating:
                # CRITICAL FIX: Sync cursor BEFORE showing canvas
                # Without this, canvas shows at stale cursor position from before animation
                if settings.anchor_enabled and settings.anchor_auto_cursor:
                    gp_obj = get_active_gp(context)
                    if gp_obj:
                        try:
                            depsgraph = context.evaluated_depsgraph_get()
                            gp_obj_eval = gp_obj.evaluated_get(depsgraph)
                            context.scene.cursor.location = gp_obj_eval.matrix_world.translation
                            set_last_cursor_synced_frame(current_frame)
                            log(f"MODAL_CURSOR_ON_STOP frame={current_frame}", "CURSOR")
                        except Exception as e:
                            log(f"MODAL_CURSOR_ON_STOP FAILED: {e}", "ERROR")

                self._show_canvas(context)
                self._last_frame_time = None  # Reset for next scrub detection

            self._is_animating = is_animating

            # Track frame for scrub detection
            if current_frame != self._last_frame:
                self._last_frame = current_frame

                # ONLY update cursor when STATIONARY (not animating)
                # This eliminates jitter and lag during playback/scrub
                if not is_animating:
                    if settings.anchor_enabled and settings.anchor_auto_cursor:
                        if current_frame != get_last_cursor_synced_frame():
                            gp_obj = get_active_gp(context)
                            if gp_obj:
                                try:
                                    depsgraph = context.evaluated_depsgraph_get()
                                    gp_obj_eval = gp_obj.evaluated_get(depsgraph)
                                    context.scene.cursor.location = gp_obj_eval.matrix_world.translation
                                    set_last_cursor_synced_frame(current_frame)
                                    log(f"MODAL_CURSOR frame={current_frame}", "CURSOR")
                                except Exception as e:
                                    log(f"MODAL_CURSOR frame={current_frame} FAILED: {e}", "ERROR")

        return {'PASS_THROUGH'}

    def _hide_canvas(self, context):
        """Hide GP canvas grid during playback/scrub."""
        try:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    for space in area.spaces:
                        if space.type == 'VIEW_3D':
                            space.overlay.use_gpencil_grid = False
        except (AttributeError, RuntimeError):
            pass

    def _show_canvas(self, context):
        """Show GP canvas grid when stationary."""
        try:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    for space in area.spaces:
                        if space.type == 'VIEW_3D':
                            space.overlay.use_gpencil_grid = True
        except (AttributeError, RuntimeError):
            pass

    def execute(self, context):
        global _cursor_sync_running
        if _cursor_sync_running:
            return {'CANCELLED'}

        # Initialize all state variables
        self._last_frame = context.scene.frame_current
        self._last_frame_time = time.time()
        self._is_animating = False
        self._scrub_cooldown = 0.0

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.016, window=context.window)
        wm.modal_handler_add(self)
        _cursor_sync_running = True
        log("Cursor sync modal started (16ms interval)", "INFO")
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        global _cursor_sync_running
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        _cursor_sync_running = False
        # Ensure canvas is shown when modal stops
        self._show_canvas(context)
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
    constraint_modified = ensure_billboard_constraint(gp_obj, scene)

    # If constraint was newly created/modified, force complete depsgraph rebuild
    # view_layer.update() alone isn't enough - new constraints need full re-evaluation
    if constraint_modified:
        scene.frame_set(scene.frame_current)
    else:
        context.view_layer.update()

    # Get evaluated matrix (with constraints fully applied)
    try:
        depsgraph = context.evaluated_depsgraph_get()
        gp_obj_eval = gp_obj.evaluated_get(depsgraph)
        matrix_world_old = gp_obj_eval.matrix_world.copy()
    except (RuntimeError, AttributeError):
        # Fallback to raw matrix if depsgraph unavailable
        matrix_world_old = gp_obj.matrix_world.copy()
    layer_matrix = get_layer_transform(active_layer)
    full_matrix_old = matrix_world_old @ layer_matrix

    # Store world positions (using full transform: object + layer)
    stroke_points = []
    drawing = active_kf.drawing
    for stroke in drawing.strokes:
        points = []
        for p in stroke.points:
            world_pos = full_matrix_old @ Vector(p.position)
            points.append(world_pos)
        stroke_points.append(points)
        
    # Move Object
    gp_obj.location = target_world_pos

    # Insert Keyframe for Location
    gp_obj.keyframe_insert(data_path="location", frame=current_frame)

    # CRITICAL: Use scene.frame_set() to force complete depsgraph rebuild
    # view_layer.update() alone doesn't re-evaluate billboard constraint rotation
    # after position change. The constraint needs to recalculate the angle to camera.
    scene.frame_set(scene.frame_current)

    # Get evaluated matrix (with constraints fully applied after frame_set)
    try:
        depsgraph = context.evaluated_depsgraph_get()
        gp_obj_eval = gp_obj.evaluated_get(depsgraph)
        matrix_world_new = gp_obj_eval.matrix_world.copy()
    except (RuntimeError, AttributeError):
        matrix_world_new = gp_obj.matrix_world.copy()
    # Use full matrix (object + layer) for transforming back to local space
    full_matrix_new = matrix_world_new @ layer_matrix
    full_matrix_new_inv = full_matrix_new.inverted()

    # Transform strokes back to local space relative to new object position
    if move_selected_strokes_to_target:
        # GROUPED centering: collect all points from ALL selected strokes
        all_selected_world_points = []
        selected_stroke_indices = []

        for i, stroke in enumerate(drawing.strokes):
            if stroke.select:
                selected_stroke_indices.append(i)
                all_selected_world_points.extend(stroke_points[i])

        if all_selected_world_points:
            # Compute single bottom-center for the entire group (center XY, lowest Z)
            sum_x = sum(p.x for p in all_selected_world_points)
            sum_y = sum(p.y for p in all_selected_world_points)
            min_z = min(p.z for p in all_selected_world_points)
            n = len(all_selected_world_points)
            group_bottom_center = Vector((sum_x / n, sum_y / n, min_z))

            # Single offset applied to all selected strokes as a group
            offset = target_world_pos - group_bottom_center

            # Apply offset to selected strokes (and optionally align to view)
            settings = scene.world_onion
            for i in selected_stroke_indices:
                # Apply offset to get new world positions
                offset_points = [p_world + offset for p_world in stroke_points[i]]

                # Optionally align strokes to face camera
                if settings.align_to_view:
                    offset_points = align_strokes_to_camera(offset_points, target_world_pos, scene)

                # Transform to local and write
                for j, new_world in enumerate(offset_points):
                    new_local = full_matrix_new_inv @ new_world
                    drawing.strokes[i].points[j].position = new_local

            # Preserve world position for non-selected strokes
            for i, stroke in enumerate(drawing.strokes):
                if i not in selected_stroke_indices:
                    for j, p_world in enumerate(stroke_points[i]):
                        new_local = full_matrix_new_inv @ p_world
                        stroke.points[j].position = new_local
        else:
            # No selected strokes - preserve all world positions
            for i, stroke in enumerate(drawing.strokes):
                for j, p_world in enumerate(stroke_points[i]):
                    new_local = full_matrix_new_inv @ p_world
                    stroke.points[j].position = new_local
    else:
        # Preserve world position for ALL strokes (compensate for object move)
        # Optionally align to view if enabled
        settings = scene.world_onion
        for i, stroke in enumerate(drawing.strokes):
            world_points = stroke_points[i]

            # Optionally align strokes to face camera
            if settings.align_to_view:
                world_points = align_strokes_to_camera(world_points, target_world_pos, scene)

            for j, p_world in enumerate(world_points):
                new_local = full_matrix_new_inv @ p_world
                stroke.points[j].position = new_local

    # Explicitly invalidate motion path and onion cache, then request redraw
    invalidate_motion_path()
    # Also invalidate onion GPU batch cache since stroke local positions changed
    from .drawing import invalidate_onion_batch_cache
    invalidate_onion_batch_cache()
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


class WONION_OT_align_to_view(bpy.types.Operator):
    """Align strokes on active layer to face the camera"""
    bl_idname = "world_onion.align_to_view"
    bl_label = "Align to View"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        gp_obj = get_active_gp(context)
        if gp_obj is None:
            return False
        # Need a camera in the scene
        return context.scene.camera is not None

    def execute(self, context):
        gp_obj = get_active_gp(context)
        if gp_obj is None:
            return {'CANCELLED'}

        scene = context.scene
        active_layer = gp_obj.data.layers.active
        if active_layer is None:
            self.report({'WARNING'}, "No active layer")
            return {'CANCELLED'}

        current_frame = scene.frame_current

        # Find active keyframe
        active_kf = None
        for kf in active_layer.frames:
            if kf.frame_number == current_frame:
                active_kf = kf
                break

        if active_kf is None:
            # Find visible keyframe
            visible_kf = get_visible_keyframe(active_layer, current_frame)
            if visible_kf is None:
                self.report({'WARNING'}, "No keyframe found")
                return {'CANCELLED'}
            active_kf = visible_kf

        drawing = active_kf.drawing
        if drawing is None or len(drawing.strokes) == 0:
            self.report({'WARNING'}, "No strokes to align")
            return {'CANCELLED'}

        # Get transform matrices (object + layer)
        matrix_world = gp_obj.matrix_world
        layer_matrix = get_layer_transform(active_layer)
        full_matrix = matrix_world @ layer_matrix
        full_matrix_inv = full_matrix.inverted()

        # Calculate anchor position (rotation center) - use stroke bottom-center
        anchor_pos = calculate_anchor_from_strokes(gp_obj, active_layer, active_kf.frame_number)
        if anchor_pos is None:
            # Fallback to object location
            anchor_pos = gp_obj.location.copy()

        # Collect world positions for all strokes
        all_world_points = []
        stroke_point_counts = []
        for stroke in drawing.strokes:
            stroke_world_points = []
            for p in stroke.points:
                world_pos = full_matrix @ Vector(p.position)
                stroke_world_points.append(world_pos)
            all_world_points.append(stroke_world_points)
            stroke_point_counts.append(len(stroke.points))

        # Flatten all points for alignment (so all strokes rotate together as a group)
        flat_points = []
        for stroke_points in all_world_points:
            flat_points.extend(stroke_points)

        if len(flat_points) < 3:
            self.report({'WARNING'}, "Need at least 3 points to align")
            return {'CANCELLED'}

        # Align all points as a group
        aligned_flat = align_strokes_to_camera(flat_points, anchor_pos, scene)

        # Unflatten and write back to strokes
        idx = 0
        for i, stroke in enumerate(drawing.strokes):
            for j in range(stroke_point_counts[i]):
                new_local = full_matrix_inv @ aligned_flat[idx]
                stroke.points[j].position = new_local
                idx += 1

        # Invalidate caches and redraw
        invalidate_motion_path()
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

        self.report({'INFO'}, "Strokes aligned to camera view")
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
    WONION_OT_align_to_view,  # Rotate strokes to face camera
    WONION_OT_bake_shrinkwrap,
)
