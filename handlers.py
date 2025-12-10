"""
Blender handlers for frame changes and depsgraph updates.
"""

import bpy
from bpy.app.handlers import persistent
from mathutils import Vector

from .cache import cache_current_frame, clear_cache, get_active_gp
from .anchors import (
    get_anchor_for_frame,
    set_anchor_for_frame,
    calculate_anchor_from_strokes,
    get_current_keyframes_set,
    get_visible_keyframe,
    migrate_anchor_data,
)
from .transforms import align_canvas_to_cursor, get_camera_direction, ensure_billboard_constraint
from .drawing import invalidate_motion_path
from .debug_log import log, log_frame_change


# Global tracking state
_last_keyframe_set = set()
_last_active_layer_name = None
_last_active_gp = None  # Track active GP object for change detection
_in_depsgraph_handler = False  # Prevent recursive handler calls
# v8.5.2: Cursor sync uses shared tracker in operators.py for modal+handler coordination

# v8: Timer/handler-based offset application REMOVED
# The driver on delta_location.z handles offset automatically now!


def _tag_viewport_redraw():
    """Tag a single 3D viewport for redraw - early exit for efficiency."""
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
                    return  # Only need one tag - Blender redraws all viewports
    except (RuntimeError, AttributeError):
        pass


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


@persistent
def on_frame_change(scene):
    """
    Called AFTER frame change is complete.
    Caches the current frame's world-space strokes.
    Ensures billboard constraint.
    v8.5.2: Hybrid cursor sync - handler catches frames missed by modal during playback.
    """
    global _last_keyframe_set

    if not hasattr(scene, 'world_onion'):
        return

    settings = scene.world_onion

    if not settings.enabled:
        return

    gp_obj = get_active_gp(bpy.context)
    if gp_obj is None:
        return

    # Ensure billboard constraint is active
    ensure_billboard_constraint(gp_obj, scene)

    # === v8: DRIVER HANDLES OFFSET AUTOMATICALLY ===
    # The driver on delta_location.z reads from the animated custom property
    # "_shrinkwrap_z_offset" which was baked during the bake operation.
    # No Python code needed here - Blender's native driver system handles it!

    # === ONION SKIN CACHE ===
    # Cache strokes for onion skin drawing
    # Note: The driver applies offset via delta_location, but we cache the
    # "raw" stroke positions (from matrix_world without delta). The offset
    # is applied during GPU drawing via get_baked_offset(frame).
    cache_current_frame(gp_obj, settings)

    # === ANCHOR SYSTEM ===
    if settings.anchor_enabled:
        # Update keyframe tracking set on frame change
        _last_keyframe_set = get_current_keyframes_set(gp_obj, settings)

        # v8.5.2: Hybrid cursor sync - handler catches frames missed by modal
        # Only update if modal hasn't already handled this frame (prevents jitter)
        if settings.anchor_auto_cursor:
            from .operators import get_last_cursor_synced_frame, set_last_cursor_synced_frame
            current_frame = scene.frame_current
            if current_frame != get_last_cursor_synced_frame():
                try:
                    depsgraph = bpy.context.evaluated_depsgraph_get()
                    gp_obj_eval = gp_obj.evaluated_get(depsgraph)
                    scene.cursor.location = gp_obj_eval.matrix_world.translation
                    set_last_cursor_synced_frame(current_frame)
                    log(f"HANDLER_CURSOR frame={current_frame}", "CURSOR")
                except (RuntimeError, AttributeError):
                    pass  # Context unavailable

    # === MOTION PATH ===
    # NOTE: Motion path invalidation REMOVED from frame change handler.
    # The path only needs rebuilding when animation data changes (handled in
    # on_depsgraph_update) or shrinkwrap settings change (handled in settings.py).
    # Invalidating on every frame was causing massive performance overhead.

    # Request viewport redraw - use efficient helper with early exit
    _tag_viewport_redraw()


@persistent
def on_depsgraph_update(scene, depsgraph):
    """
    Invalidate cache when parent chain changes (camera moves).
    Also detect new keyframes for anchor auto-capture.
    Also detect active object changes to clear cache.
    """
    global _last_keyframe_set, _last_active_layer_name, _last_active_gp, _in_depsgraph_handler

    # Prevent recursive calls
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

    # Get active GP object
    gp_obj = get_active_gp(bpy.context)

    # v8.5: Cursor update moved to modal operator (WONION_OT_cursor_sync)
    # The modal handles cursor updates reliably - no backup needed here.

    # Detect active GP object change - clear cache when switching
    if gp_obj != _last_active_gp:
        if _last_active_gp is not None:
            clear_cache()
            _last_active_layer_name = None
        _last_active_gp = gp_obj

    if gp_obj is None:
        return

    gp_data = gp_obj.data
    gp_data_changed = False
    animation_changed = False

    # Check updates - with early exit once both flags found
    for update in depsgraph.updates:
        # Early exit optimization - stop iterating once we know both flags
        if gp_data_changed and animation_changed:
            break

        # GP stroke data changed
        if not gp_data_changed:
            if update.id == gp_data:
                gp_data_changed = True
            elif isinstance(update.id, bpy.types.GreasePencil) and update.id.name == gp_data.name:
                gp_data_changed = True

        # Animation data changed (Location keyframes added/deleted/moved)
        if not animation_changed:
            if gp_obj.animation_data and update.id == gp_obj.animation_data.action:
                animation_changed = True
            elif isinstance(update.id, bpy.types.Action):
                # Check if this action belongs to our GP object
                if gp_obj.animation_data and gp_obj.animation_data.action:
                    if update.id.name == gp_obj.animation_data.action.name:
                        animation_changed = True

    # Invalidate motion path on GP data OR animation change
    if gp_data_changed or animation_changed:
        invalidate_motion_path()
        # Force viewport redraw for immediate feedback
        _tag_viewport_redraw()

    # Detect keyframe changes
    if gp_data_changed and _last_keyframe_set:
        current_kf_set = get_current_keyframes_set(gp_obj, settings)
        
        # Track moved keyframes for anchor migration
        removed_by_layer = {}
        added_by_layer = {}

        for layer_name, frame in (_last_keyframe_set - current_kf_set):
            removed_by_layer.setdefault(layer_name, []).append(frame)

        for layer_name, frame in (current_kf_set - _last_keyframe_set):
            added_by_layer.setdefault(layer_name, []).append(frame)

        # Check for moves
        for layer_name in removed_by_layer:
            if layer_name in added_by_layer:
                removed = removed_by_layer[layer_name]
                added = added_by_layer[layer_name]
                if len(removed) == 1 and len(added) == 1:
                    old_frame = removed[0]
                    new_frame = added[0]
                    migrate_anchor_data(gp_obj, layer_name, old_frame, new_frame)
                elif len(removed) == len(added):
                    removed_sorted = sorted(removed)
                    added_sorted = sorted(added)
                    for old_frame, new_frame in zip(removed_sorted, added_sorted):
                        migrate_anchor_data(gp_obj, layer_name, old_frame, new_frame)

        # Handle new keyframes
        if settings.anchor_enabled:
            current_frame = scene.frame_current
            new_keyframes = current_kf_set - _last_keyframe_set

            for layer_name, frame_num in new_keyframes:
                if frame_num == current_frame:
                    # Capture cursor as anchor for new keyframes
                    existing_anchor = get_anchor_for_frame(gp_obj, layer_name, frame_num)
                    if existing_anchor is None:
                        cursor_pos = scene.cursor.location.copy()
                        cam_dir = get_camera_direction(scene)
                        set_anchor_for_frame(gp_obj, layer_name, frame_num, cursor_pos, cam_dir)

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
