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
from .transforms import align_canvas_to_cursor, ensure_billboard_constraint
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

    PERFORMANCE: Keyframe set update moved to depsgraph handler (P5).
    Only recomputed when GP data actually changes, not every frame.
    """
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

    # === SHRINKWRAP VALIDATION ===
    # Single point of truth - if shrinkwrap is enabled, ensure all components are valid.
    # This catches ALL scenarios: enable toggle, file load, GP switch, addon reload, etc.
    # Cheap when already valid (just boolean checks), fixes automatically when not.
    if settings.depth_interaction_enabled:
        from .drawing import ensure_shrinkwrap_valid
        ensure_shrinkwrap_valid(gp_obj, settings, scene)

    # === ONION SKIN CACHE ===
    # Cache strokes for onion skin drawing
    cache_current_frame(gp_obj, settings)

    # NOTE: Keyframe set update moved to depsgraph handler (P5 optimization)
    # Only updates when gp_data_changed, not on every frame scrub.

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
    """Implementation of depsgraph update handler.

    PERFORMANCE (P8): Uses identity checks and cached attribute lookups.
    PERFORMANCE (P5): Keyframe set update moved here from frame_change handler.
    """
    global _last_keyframe_set, _last_active_layer_name, _last_active_gp

    if not hasattr(scene, 'world_onion'):
        return

    settings = scene.world_onion
    if not settings.enabled:
        return

    # Get active GP object
    gp_obj = get_active_gp(bpy.context)

    # Detect active GP object change - clear cache when switching
    if gp_obj is not _last_active_gp:  # Identity check (P8)
        if _last_active_gp is not None:
            clear_cache()
            _last_active_layer_name = None
        _last_active_gp = gp_obj

    if gp_obj is None:
        return

    # Cache lookups before loop (P8 optimization)
    gp_data = gp_obj.data
    anim_data = gp_obj.animation_data
    action = anim_data.action if anim_data else None

    gp_data_changed = False
    animation_changed = False

    # Check updates - with early exit once both flags found
    for update in depsgraph.updates:
        if gp_data_changed and animation_changed:
            break

        update_id = update.id

        # GP stroke data changed - use identity check (P8)
        if not gp_data_changed:
            if update_id is gp_data:
                gp_data_changed = True
            elif isinstance(update_id, bpy.types.GreasePencil) and update_id.name == gp_data.name:
                gp_data_changed = True

        # Animation data changed - use identity check with cached action (P8)
        if not animation_changed and action is not None:
            if update_id is action:
                animation_changed = True
            elif isinstance(update_id, bpy.types.Action) and update_id.name == action.name:
                animation_changed = True

    # Invalidate motion path AND onion cache on GP data OR animation change
    if gp_data_changed or animation_changed:
        invalidate_motion_path()
        # Also invalidate onion GPU batch cache so strokes refresh immediately
        # This fixes the "stale onion skin while editing" bug
        from .drawing import invalidate_onion_batch_cache, invalidate_keyframe_cache
        invalidate_onion_batch_cache()
        # P7: Also invalidate keyframe cache since keyframes may have changed
        invalidate_keyframe_cache()
        # Force viewport redraw for immediate feedback
        _tag_viewport_redraw()

        # Re-bake shrinkwrap when animation changes (keyframes added/deleted/moved)
        # Unlike motion path (lazy rebuild), shrinkwrap uses pre-baked dictionary
        # that the driver reads, so stale data persists without explicit re-bake
        # v9.3: Use setup_driver=False since we're in handler context (restricted)
        if animation_changed and settings.depth_interaction_enabled:
            from .drawing import bake_shrinkwrap_offsets
            bake_shrinkwrap_offsets(gp_obj, settings, scene, setup_driver=False)
            # NOTE: No scene.frame_set() here - it can cause recursive handler calls
            # and we're already in a handler context. Driver will pick up new baked
            # values on the next frame naturally.

    # Detect keyframe changes (P5: only when GP data changes, not every frame)
    if gp_data_changed:
        current_kf_set = get_current_keyframes_set(gp_obj, settings)

        # Only do comparison if we have a previous set to compare against
        # On first run, _last_keyframe_set is empty - just initialize it
        if _last_keyframe_set:
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
                            set_anchor_for_frame(gp_obj, layer_name, frame_num, cursor_pos)

        _last_keyframe_set = current_kf_set


@persistent
def on_load_post(dummy):
    """Clear cache when loading a new file."""
    clear_cache()


@persistent
def on_undo_post(scene):
    """Clear caches when user performs undo - ensures visual state matches data.

    This is critical for snap operators which use scene.frame_set() during execution.
    frame_set() triggers cache updates mid-operator, so on undo the stroke DATA is
    restored but cached onion skin data still has old positions. Clearing caches
    here forces them to rebuild from the restored (undone) data.
    """
    clear_cache()
    invalidate_motion_path()
    from .drawing import invalidate_onion_batch_cache, invalidate_keyframe_cache
    invalidate_onion_batch_cache()
    invalidate_keyframe_cache()  # P7: Keyframes may have been undone
    _tag_viewport_redraw()
    log("UNDO detected - cleared all caches", "INFO")


def register_handlers():
    """Register all handlers."""
    if on_frame_change not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(on_frame_change)

    if on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(on_depsgraph_update)

    if on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(on_load_post)

    if on_undo_post not in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.append(on_undo_post)


def unregister_handlers():
    """Unregister all handlers."""
    if on_frame_change in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(on_frame_change)

    if on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(on_depsgraph_update)

    if on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(on_load_post)

    if on_undo_post in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.remove(on_undo_post)
