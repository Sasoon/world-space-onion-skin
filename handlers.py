"""
Blender handlers for frame changes and depsgraph updates.
"""

import bpy
from bpy.app.handlers import persistent
from mathutils import Vector, Matrix

from .cache import cache_current_frame, clear_cache, get_cache, get_active_gp, get_all_world_locked_gp_objects
from .anchors import (
    get_anchor_for_frame,
    set_anchor_for_frame,
    calculate_anchor_from_strokes,
    calculate_anchor_local_offset,
    get_current_keyframes_set,
    get_visible_keyframe,
    migrate_anchor_data,
    # Object-level lock functions
    get_object_lock_data,
    is_object_locked_at_frame,
    get_lock_for_frame,
    set_lock_for_frame,
    find_visible_locked_frame,
    migrate_object_lock_frame,
    get_interpolated_position,
)
from .transforms import align_canvas_to_cursor, get_camera_direction
from .drawing import invalidate_motion_path


# Global tracking state
_last_keyframe_set = set()
_last_active_layer_name = None
_last_active_gp = None  # Track active GP object for change detection
_in_depsgraph_handler = False  # Prevent recursive handler calls


def get_last_keyframe_set():
    return _last_keyframe_set


def set_last_keyframe_set(value):
    global _last_keyframe_set
    _last_keyframe_set = value


def get_last_active_layer_name():
    return _last_active_layer_name


def set_last_active_layer_name(value):
    global _last_active_layer_name
    _last_active_layer_name = value


def reset_last_active_gp():
    global _last_active_gp
    _last_active_gp = None


# =============================================================================
# OBJECT-LEVEL WORLD LOCK SYSTEM
# =============================================================================
# Uses matrix_parent_inverse to control GP object's world position while
# keeping layer transforms untouched. This provides:
# - Full GP effects compatibility
# - True billboard effect (strokes always face camera)
# - Simpler math (no pivot calculations)
# =============================================================================

def apply_world_lock_from_stored(gp_obj, anchor_world, anchor_local_offset, matrix_local_stored):
    """Apply world lock with billboard effect using pivot-based rotation.

    Rotates around the anchor point (stroke center) so strokes stay planted
    while the GP rotates to face the camera.

    Args:
        gp_obj: The GP object to lock
        anchor_world: Vector or list [x,y,z] - world position of anchor (stays fixed)
        anchor_local_offset: Vector or list [x,y,z] - offset from GP origin to anchor in local coords
        matrix_local_stored: Matrix (4x4) - the matrix_local captured at lock time
    """
    if gp_obj is None:
        return

    # Convert from list if needed
    if isinstance(anchor_world, list):
        anchor_world = Vector(anchor_world)
    if isinstance(anchor_local_offset, list):
        anchor_local_offset = Vector(anchor_local_offset)
    if isinstance(matrix_local_stored, list):
        matrix_local_stored = Matrix(matrix_local_stored)

    parent = gp_obj.parent
    if parent is None:
        # No parent - apply transform directly using matrix_local rotation
        local_rot = matrix_local_stored.to_3x3()
        gp_position = anchor_world - local_rot @ anchor_local_offset
        gp_obj.location = gp_position
        return

    # Billboard rotation: camera rotation with local orientation preserved
    camera_rot = parent.matrix_world.to_3x3()
    local_rot = matrix_local_stored.to_3x3()
    R_desired = camera_rot @ local_rot

    # Pivot-based rotation: compute GP position to keep anchor fixed
    # gp_position = anchor_world - R_desired @ anchor_local_offset
    gp_position = anchor_world - R_desired @ anchor_local_offset

    # Compose desired world matrix
    desired_world = Matrix.Translation(gp_position) @ R_desired.to_4x4()

    # Solve for MPI:
    # matrix_world = parent.matrix_world @ mpi @ matrix_local
    # We want: desired_world = parent.matrix_world @ mpi @ matrix_local_stored
    # So: mpi = parent.matrix_world.inverted() @ desired_world @ matrix_local_stored.inverted()
    new_mpi = parent.matrix_world.inverted() @ desired_world @ matrix_local_stored.inverted()

    gp_obj.matrix_parent_inverse = new_mpi


def reset_object_world_lock(gp_obj, original_parent_inverse=None):
    """Reset GP object to normal camera-parented behavior.

    Args:
        gp_obj: The GP object to reset
        original_parent_inverse: Optional Matrix to restore, or None for identity
    """
    if gp_obj is None:
        return

    if original_parent_inverse is not None:
        if isinstance(original_parent_inverse, list):
            gp_obj.matrix_parent_inverse = Matrix(original_parent_inverse)
        else:
            gp_obj.matrix_parent_inverse = original_parent_inverse
    else:
        gp_obj.matrix_parent_inverse = Matrix.Identity(4)


def apply_object_world_lock_for_frame(gp_obj, scene):
    """Determine which lock applies at current frame and apply it.

    Uses pivot-based billboard rotation around the anchor point.
    Supports interpolation between locked frames when enabled.

    Args:
        gp_obj: The GP object to process
        scene: The current scene
    """
    if gp_obj is None:
        return

    current_frame = scene.frame_current
    settings = scene.world_onion

    # Find which locked frame is visible
    locked_frame = find_visible_locked_frame(gp_obj, current_frame)

    if locked_frame is not None:
        # Get lock data for this frame
        lock_data = get_lock_for_frame(gp_obj, locked_frame)
        if lock_data and "anchor_world" in lock_data and "anchor_local_offset" in lock_data and "matrix_local" in lock_data:
            # Check if interpolation is enabled
            anchor_world = lock_data["anchor_world"]
            anchor_local_offset = lock_data["anchor_local_offset"]
            matrix_local = lock_data["matrix_local"]

            if settings.interpolation_enabled:
                # Try to get interpolated position
                interp_pos, interp_info = get_interpolated_position(gp_obj, current_frame)
                if interp_pos is not None and interp_info is not None:
                    # Use interpolated position and data from the interpolation's prev frame
                    anchor_world = [interp_pos.x, interp_pos.y, interp_pos.z]
                    prev_data = interp_info['prev_data']
                    anchor_local_offset = prev_data["anchor_local_offset"]
                    matrix_local = prev_data["matrix_local"]

            # Apply world lock with pivot-based billboard effect
            apply_world_lock_from_stored(
                gp_obj,
                anchor_world,
                anchor_local_offset,
                matrix_local
            )
            return

    # No locked keyframe visible at current frame
    # BUT if interpolation is enabled, we may still be between locked frames
    # (e.g., new unlocked keyframe created via auto-keying at interpolated position)
    if settings.interpolation_enabled:
        interp_pos, interp_info = get_interpolated_position(gp_obj, current_frame)
        if interp_pos is not None and interp_info is not None:
            # We're between locked frames - apply interpolated position
            # This preserves the visual position when a new unlocked keyframe is created
            prev_data = interp_info['prev_data']
            anchor_world = [interp_pos.x, interp_pos.y, interp_pos.z]
            anchor_local_offset = prev_data["anchor_local_offset"]
            matrix_local = prev_data["matrix_local"]

            apply_world_lock_from_stored(
                gp_obj,
                anchor_world,
                anchor_local_offset,
                matrix_local
            )
            return  # Don't reset MPI - preserve interpolated position

    # Truly no lock context - reset to normal parenting
    lock_data = get_object_lock_data(gp_obj)
    # Try to find original_parent_inverse from any frame's data
    original_mpi = None
    for frame_str, data in lock_data.items():
        if isinstance(data, dict) and "original_parent_inverse" in data:
            original_mpi = data["original_parent_inverse"]
            break

    reset_object_world_lock(gp_obj, original_mpi)


@persistent
def on_frame_change(scene):
    """
    Called AFTER frame change is complete.
    Caches the current frame's world-space strokes.
    Also handles anchor auto-move cursor.
    Also applies world-lock transforms.
    """
    global _last_keyframe_set

    if not hasattr(scene, 'world_onion'):
        return

    settings = scene.world_onion

    if not settings.enabled:
        return

    # === OBJECT-LEVEL WORLD LOCK ===
    # Apply to ALL GP objects with world locks (not just active)
    # This ensures locks work even when GP object isn't selected
    for locked_gp in get_all_world_locked_gp_objects(scene):
        apply_object_world_lock_for_frame(locked_gp, scene)

    # Get active GP object for cache/anchor features
    gp_obj = get_active_gp(bpy.context)

    # Track if we handled cursor via interpolation (to skip normal anchor cursor)
    cursor_handled_by_interpolation = False

    # === CURSOR/CANVAS INTERPOLATION (active GP only) ===
    # Move cursor to interpolated position when scrubbing between locked frames
    if gp_obj is not None and settings.interpolation_enabled and settings.anchor_enabled and settings.anchor_auto_cursor:
        interp_pos, interp_info = get_interpolated_position(gp_obj, scene.frame_current)
        if interp_pos is not None and interp_info is not None:
            # We're interpolating between frames - move cursor to interpolated position
            scene.cursor.location = interp_pos.copy()
            cursor_handled_by_interpolation = True
            try:
                align_canvas_to_cursor(bpy.context)
            except (RuntimeError, AttributeError):
                pass

    if gp_obj is None:
        return

    # === ANCHOR SYSTEM ===
    if settings.anchor_enabled:
        # Update keyframe tracking set on frame change
        _last_keyframe_set = get_current_keyframes_set(gp_obj, settings)
        
        # Auto-move cursor to anchor position (active layer only)
        # Skip if cursor was already set by interpolation
        if settings.anchor_auto_cursor and not cursor_handled_by_interpolation:
            active_layer = gp_obj.data.layers.active
            if active_layer is not None:
                current_frame = scene.frame_current

                # Find keyframe at or before current frame for active layer
                active_kf = get_visible_keyframe(active_layer, current_frame)

                if active_kf is not None:
                    # Check if anchor exists
                    anchor_pos = get_anchor_for_frame(gp_obj, active_layer.name, active_kf.frame_number)

                    if anchor_pos is None and settings.anchor_snap_to_stroke:
                        # Only auto-calculate if snap enabled AND NOT world-locked
                        # World-locked frames should have anchor set at lock time, not during scrubbing
                        if not is_object_locked_at_frame(gp_obj, active_kf.frame_number):
                            anchor_pos = calculate_anchor_from_strokes(gp_obj, active_layer, active_kf.frame_number)

                            if anchor_pos is not None:
                                # Store for future with camera direction
                                cam_dir = get_camera_direction(scene)
                                set_anchor_for_frame(gp_obj, active_layer.name, active_kf.frame_number, anchor_pos, cam_dir)

                    # Move cursor to anchor
                    if anchor_pos is not None:
                        scene.cursor.location = anchor_pos

                        # Align canvas to cursor when anchor system is active
                        if settings.anchor_enabled:
                            try:
                                align_canvas_to_cursor(bpy.context)
                            except (RuntimeError, AttributeError):
                                # Context may be invalid during certain operations
                                pass

    # === ONION SKIN CACHE ===
    cache_current_frame(gp_obj, settings)

    # === MOTION PATH ===
    # Invalidate on every frame change so it stays fresh
    if settings.motion_path_enabled:
        invalidate_motion_path()

    # Request viewport redraw
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
    except (RuntimeError, AttributeError):
        # Window manager may not be accessible in some contexts
        pass


@persistent
def on_depsgraph_update(scene, depsgraph):
    """
    Invalidate cache when parent chain changes (camera moves).
    Also detect new keyframes for anchor auto-capture.
    Also detect layer selection changes for anchor cursor snapping.
    Also detect active object changes to clear cache.
    """
    global _last_keyframe_set, _last_active_layer_name, _last_active_gp, _in_depsgraph_handler

    # Prevent recursive calls (setting custom properties can trigger updates)
    if _in_depsgraph_handler:
        return
    _in_depsgraph_handler = True

    try:
        _on_depsgraph_update_impl(scene, depsgraph)
    finally:
        _in_depsgraph_handler = False


def _on_depsgraph_update_impl(scene, depsgraph):
    """Implementation of depsgraph update handler."""
    global _last_keyframe_set, _last_active_layer_name, _last_active_gp

    if not hasattr(scene, 'world_onion'):
        return

    settings = scene.world_onion
    if not settings.enabled:
        return

    # Get active GP object (auto-detect)
    gp_obj = get_active_gp(bpy.context)

    # Detect active GP object change - clear cache when switching
    if gp_obj != _last_active_gp:
        if _last_active_gp is not None:
            clear_cache()
            _last_active_layer_name = None
        _last_active_gp = gp_obj

    # Get all locked GP objects - we need to check their parent chains
    locked_gp_objects = get_all_world_locked_gp_objects(scene)

    gp_data_changed = False
    should_clear_cache = False
    should_update_locks = False

    # Get the GP data type name for comparison (active GP only)
    gp_data = gp_obj.data if (gp_obj and hasattr(gp_obj, 'data')) else None

    # Check what was updated
    for update in depsgraph.updates:
        update_id = update.id
        update_type = type(update_id).__name__

        # GP data changed - detect for anchor system but DON'T clear cache
        if gp_data is not None:
            if update_id == gp_data:
                gp_data_changed = True
                continue
            # Check by name but only for GP data types
            if update_type in ('GreasePencilv3', 'GreasePencil'):
                if hasattr(update_id, 'name') and update_id.name == gp_data.name:
                    gp_data_changed = True
                    continue

        # Check if update affects parent chain of ANY locked GP object
        for locked_gp in locked_gp_objects:
            parent = locked_gp.parent
            while parent is not None:
                if update_id == parent:
                    should_update_locks = True
                    # Only clear cache if active GP's parent changed
                    if gp_obj and locked_gp == gp_obj:
                        should_clear_cache = True
                    break
                if parent.animation_data and update_id == parent.animation_data.action:
                    should_update_locks = True
                    if gp_obj and locked_gp == gp_obj:
                        should_clear_cache = True
                    break
                parent = parent.parent
            if should_update_locks:
                break

    # Clear cache for active GP's parent changes
    if should_clear_cache:
        clear_cache()

    # Update world lock transforms for all locked objects if any parent moved
    if should_update_locks:
        for locked_gp in locked_gp_objects:
            apply_object_world_lock_for_frame(locked_gp, scene)
    
    # Check for layer selection change
    if settings.anchor_enabled and settings.anchor_auto_cursor and gp_data is not None:
        active_layer = gp_data.layers.active
        current_layer_name = active_layer.name if active_layer else None
        
        if current_layer_name != _last_active_layer_name:
            # Layer selection changed - snap cursor to new layer's anchor
            _last_active_layer_name = current_layer_name
            
            if active_layer is not None:
                current_frame = scene.frame_current

                # Find keyframe at or before current frame
                active_kf = get_visible_keyframe(active_layer, current_frame)

                if active_kf is not None:
                    # Get or calculate anchor
                    anchor_pos = get_anchor_for_frame(gp_obj, active_layer.name, active_kf.frame_number)

                    if anchor_pos is None and settings.anchor_snap_to_stroke:
                        # Only auto-calculate if snap enabled AND NOT world-locked
                        # World-locked frames should have anchor set at lock time
                        if not is_object_locked_at_frame(gp_obj, active_kf.frame_number):
                            anchor_pos = calculate_anchor_from_strokes(gp_obj, active_layer, active_kf.frame_number)

                            if anchor_pos is not None:
                                cam_dir = get_camera_direction(scene)
                                set_anchor_for_frame(gp_obj, active_layer.name, active_kf.frame_number, anchor_pos, cam_dir)

                    if anchor_pos is not None:
                        scene.cursor.location = anchor_pos

                        # Align canvas to cursor when anchor system is active
                        if settings.anchor_enabled:
                            try:
                                align_canvas_to_cursor(bpy.context)
                            except (RuntimeError, AttributeError):
                                # Context may be invalid during certain operations
                                pass

    # Handle anchor system - detect keyframe moves and NEW keyframes
    if gp_data_changed and settings.anchor_enabled:
        current_frame = scene.frame_current
        current_kf_set = get_current_keyframes_set(gp_obj, settings)

        # Detect keyframe MOVES (1 removed + 1 added on same layer)
        if _last_keyframe_set:
            removed_by_layer = {}
            added_by_layer = {}

            for layer_name, frame in (_last_keyframe_set - current_kf_set):
                removed_by_layer.setdefault(layer_name, []).append(frame)

            for layer_name, frame in (current_kf_set - _last_keyframe_set):
                added_by_layer.setdefault(layer_name, []).append(frame)

            # Invalidate motion path if any keyframes changed
            if removed_by_layer or added_by_layer:
                invalidate_motion_path()

            # Track which frames were moved (to distinguish from deletions)
            moved_frames = set()

            # Check for moves (1 removed + 1 added on same layer = move)
            for layer_name in removed_by_layer:
                if layer_name in added_by_layer:
                    removed = removed_by_layer[layer_name]
                    added = added_by_layer[layer_name]
                    if len(removed) == 1 and len(added) == 1:
                        old_frame = removed[0]
                        new_frame = added[0]
                        moved_frames.add(old_frame)
                        # Migrate both anchor data and object lock data
                        migrate_anchor_data(gp_obj, layer_name, old_frame, new_frame)
                        migrate_object_lock_frame(gp_obj, old_frame, new_frame)

            # Clean up orphaned lock data for deleted keyframes (not moves)
            from .anchors import remove_lock_for_frame
            for layer_name, removed_frames in removed_by_layer.items():
                for frame in removed_frames:
                    # If this frame wasn't moved, it's a deletion - clean up lock data
                    if frame not in moved_frames:
                        if is_object_locked_at_frame(gp_obj, frame):
                            remove_lock_for_frame(gp_obj, frame)

        # Update user-facing anchor from strokes if snap_to_stroke enabled
        active_layer = gp_obj.data.layers.active
        if active_layer is not None and settings.anchor_snap_to_stroke:
            visible_kf = get_visible_keyframe(active_layer, current_frame)
            if visible_kf is not None and visible_kf.frame_number == current_frame:
                # Only update anchor if NOT world-locked (locked frames keep their anchor)
                if not is_object_locked_at_frame(gp_obj, current_frame):
                    stroke_anchor = calculate_anchor_from_strokes(gp_obj, active_layer, current_frame)
                    if stroke_anchor is not None:
                        cam_dir = get_camera_direction(scene)
                        set_anchor_for_frame(gp_obj, active_layer.name, current_frame, stroke_anchor, cam_dir)

        # Find NEW keyframes (just created)
        if _last_keyframe_set:
            new_keyframes = current_kf_set - _last_keyframe_set

            # For new keyframes on current frame, capture cursor position
            for layer_name, frame_num in new_keyframes:
                if frame_num == current_frame:
                    # This is a brand new keyframe - capture cursor as anchor
                    existing_anchor = get_anchor_for_frame(gp_obj, layer_name, frame_num)
                    if existing_anchor is None:
                        cursor_pos = scene.cursor.location.copy()
                        cam_dir = get_camera_direction(scene)
                        set_anchor_for_frame(gp_obj, layer_name, frame_num, cursor_pos, cam_dir)

                    # World lock inherit: if previous keyframe was locked, lock this one too
                    if settings.world_lock_inherit:
                        # Find the previous keyframe on any layer
                        prev_kf_frame = None
                        for layer in gp_obj.data.layers:
                            for kf in layer.frames:
                                if kf.frame_number < frame_num:
                                    if prev_kf_frame is None or kf.frame_number > prev_kf_frame:
                                        prev_kf_frame = kf.frame_number

                        if prev_kf_frame is not None:
                            # Check if previous keyframe was world-locked (object level)
                            if is_object_locked_at_frame(gp_obj, prev_kf_frame):
                                # Auto-lock the new keyframe at current anchor position
                                if gp_obj.parent is not None:
                                    # Calculate anchor from strokes for pivot-based rotation
                                    active_layer = gp_obj.data.layers.active
                                    anchor_world = None
                                    anchor_local = None
                                    if active_layer is not None:
                                        anchor_world, anchor_local = calculate_anchor_from_strokes(
                                            gp_obj, active_layer, frame_num, return_local=True
                                        )

                                    if anchor_world is None:
                                        anchor_world = gp_obj.matrix_world.to_translation().copy()
                                        anchor_local = Vector((0, 0, 0))

                                    # Use raw local anchor position (stable regardless of MPI)
                                    anchor_local_offset, matrix_local = calculate_anchor_local_offset(gp_obj, anchor_local)
                                    original_mpi = gp_obj.matrix_parent_inverse.copy()
                                    set_lock_for_frame(gp_obj, frame_num, anchor_world, anchor_local_offset, original_mpi, matrix_local)

        # Update tracking set
        _last_keyframe_set = current_kf_set


@persistent
def on_load_post(dummy):
    """Clear cache when loading a new file."""
    clear_cache()


def register_handlers():
    """Register all handlers."""
    if on_frame_change not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(on_frame_change)
    
    if on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(on_depsgraph_update)
    
    if on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(on_load_post)


def unregister_handlers():
    """Unregister all handlers."""
    if on_frame_change in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(on_frame_change)
    
    if on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(on_depsgraph_update)
    
    if on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(on_load_post)
