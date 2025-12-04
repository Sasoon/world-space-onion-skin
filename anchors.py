"""
Anchor system for world-space drawing positions and world lock data.
"""

import bpy
import json
from mathutils import Vector, Matrix

from .transforms import get_layer_transform


def get_anchors(gp_obj):
    """Get anchor data from GP object custom property."""
    if gp_obj is None:
        return {}
    
    if "world_onion_anchors" not in gp_obj:
        return {}
    
    try:
        data = json.loads(gp_obj["world_onion_anchors"])

        # Convert legacy format (list) to new format (dict with pos and cam_dir)
        for layer_name in data:
            for frame_str in data[layer_name]:
                anchor_data = data[layer_name][frame_str]
                if isinstance(anchor_data, list):
                    # Legacy format: just position as list
                    data[layer_name][frame_str] = {"pos": anchor_data}

        return data
    except (json.JSONDecodeError, TypeError, KeyError):
        # Invalid or corrupted anchor data
        return {}


def set_anchors(gp_obj, anchors):
    """Save anchor data to GP object custom property."""
    if gp_obj is None:
        return
    gp_obj["world_onion_anchors"] = json.dumps(anchors)


def get_anchor_for_frame(gp_obj, layer_name, frame):
    """Get the anchor position for a specific layer and frame.
    
    Returns Vector or None.
    """
    anchors = get_anchors(gp_obj)
    
    if layer_name not in anchors:
        return None
    
    frame_str = str(frame)
    if frame_str not in anchors[layer_name]:
        return None
    
    data = anchors[layer_name][frame_str]
    
    if isinstance(data, dict) and "pos" in data:
        return Vector(data["pos"])
    elif isinstance(data, list):
        return Vector(data)
    
    return None


def get_anchor_camera_dir(gp_obj, layer_name, frame):
    """Get the camera direction stored with anchor for a specific layer and frame.

    Returns Vector or None.
    """
    anchors = get_anchors(gp_obj)

    if layer_name not in anchors:
        return None

    frame_str = str(frame)
    if frame_str not in anchors[layer_name]:
        return None

    data = anchors[layer_name][frame_str]

    if isinstance(data, dict) and "cam_dir" in data:
        return Vector(data["cam_dir"])

    return None


def get_lock_anchor_for_frame(gp_obj, layer_name, frame):
    """Get the lock anchor position for world-lock calculations.

    This is always calculated from stroke geometry and used as the rotation
    pivot for world-lock transforms. Falls back to user anchor if lock_anchor
    is not set.

    Returns Vector or None.
    """
    anchors = get_anchors(gp_obj)

    if layer_name not in anchors:
        return None

    frame_str = str(frame)
    if frame_str not in anchors[layer_name]:
        return None

    data = anchors[layer_name][frame_str]

    if isinstance(data, dict):
        # Prefer lock_anchor (stroke-based), fall back to pos (user anchor)
        if "lock_anchor" in data:
            return Vector(data["lock_anchor"])
        elif "pos" in data:
            return Vector(data["pos"])
    elif isinstance(data, list):
        return Vector(data)

    return None


def set_lock_anchor_for_frame(gp_obj, layer_name, frame, position):
    """Set the lock anchor position for world-lock calculations.

    This should always be calculated from stroke geometry.

    position: Vector or tuple (x, y, z)
    """
    anchors = get_anchors(gp_obj)

    if layer_name not in anchors:
        anchors[layer_name] = {}

    frame_str = str(frame)

    # Preserve existing data
    if frame_str in anchors[layer_name]:
        anchor_data = anchors[layer_name][frame_str]
        if not isinstance(anchor_data, dict):
            anchor_data = {"pos": anchor_data} if isinstance(anchor_data, list) else {}
    else:
        anchor_data = {}

    anchor_data["lock_anchor"] = [position[0], position[1], position[2]]

    anchors[layer_name][frame_str] = anchor_data
    set_anchors(gp_obj, anchors)


def set_anchor_for_frame(gp_obj, layer_name, frame, position, camera_dir=None):
    """Set the anchor position and camera direction for a specific layer and frame.

    position: Vector or tuple (x, y, z)
    camera_dir: Vector or tuple (x, y, z) - direction camera is facing
    """
    anchors = get_anchors(gp_obj)

    if layer_name not in anchors:
        anchors[layer_name] = {}

    frame_str = str(frame)

    # Preserve existing data (like world_locked, lock_matrix)
    if frame_str in anchors[layer_name]:
        anchor_data = anchors[layer_name][frame_str]
        if not isinstance(anchor_data, dict):
            anchor_data = {}
    else:
        anchor_data = {}

    anchor_data["pos"] = [position[0], position[1], position[2]]

    if camera_dir is not None:
        anchor_data["cam_dir"] = [camera_dir[0], camera_dir[1], camera_dir[2]]

    anchors[layer_name][frame_str] = anchor_data
    set_anchors(gp_obj, anchors)


def remove_anchor_for_frame(gp_obj, layer_name, frame):
    """Remove the anchor for a specific layer and frame."""
    anchors = get_anchors(gp_obj)

    if layer_name not in anchors:
        return

    frame_str = str(frame)
    if frame_str in anchors[layer_name]:
        del anchors[layer_name][frame_str]
        set_anchors(gp_obj, anchors)


def migrate_anchor_data(gp_obj, layer_name, old_frame, new_frame):
    """Move anchor/world lock data from old frame to new frame.

    Called when a keyframe is moved in the timeline to preserve
    anchor and world lock state.
    """
    anchors = get_anchors(gp_obj)

    if layer_name not in anchors:
        return

    old_frame_str = str(old_frame)
    new_frame_str = str(new_frame)

    if old_frame_str in anchors[layer_name]:
        # Move data to new frame
        anchors[layer_name][new_frame_str] = anchors[layer_name][old_frame_str]
        del anchors[layer_name][old_frame_str]
        set_anchors(gp_obj, anchors)


def calculate_anchor_from_strokes(gp_obj, layer, frame_number):
    """Calculate anchor position from strokes: center XY, lowest Z.

    IMPORTANT: Uses identity layer transform to get raw stroke positions
    without any world lock compensation. This prevents circular dependency
    where world lock sets layer.translation and anchor calculation uses it.

    Returns Vector or None if no strokes found.
    """
    if gp_obj is None or layer is None:
        return None

    # Find the keyframe
    keyframe = None
    for kf in layer.frames:
        if kf.frame_number == frame_number:
            keyframe = kf
            break

    if keyframe is None or keyframe.drawing is None:
        return None

    drawing = keyframe.drawing

    if not hasattr(drawing, 'attributes') or 'position' not in drawing.attributes:
        return None

    pos_attr = drawing.attributes['position']
    if len(pos_attr.data) == 0:
        return None

    # Get world matrix WITHOUT layer transform
    # This gives raw stroke positions, unaffected by world lock compensation
    # DO NOT use layer_matrix here - it contains world lock offset
    world_matrix = gp_obj.matrix_world
    full_matrix = world_matrix  # Raw positions only
    
    # Collect all world-space points
    min_z = float('inf')
    sum_x = 0.0
    sum_y = 0.0
    count = 0
    
    for p in pos_attr.data:
        local_pos = Vector(p.vector)
        world_pos = full_matrix @ local_pos
        
        sum_x += world_pos.x
        sum_y += world_pos.y
        if world_pos.z < min_z:
            min_z = world_pos.z
        count += 1
    
    if count == 0:
        return None
    
    # Center XY, lowest Z
    anchor = Vector((sum_x / count, sum_y / count, min_z))
    return anchor


def get_current_keyframes_set(gp_obj, settings):
    """Get a set of (layer_name, frame_number) for all current keyframes."""
    result = set()
    
    if gp_obj is None or gp_obj.data is None:
        return result
    
    for layer in gp_obj.data.layers:
        if layer.hide:
            continue
        if settings.skip_underscore and layer.name.startswith('_'):
            continue
        if settings.layer_filter and settings.layer_filter not in layer.name:
            continue
        
        for kf in layer.frames:
            result.add((layer.name, kf.frame_number))
    
    return result


def is_world_locked(gp_obj, layer_name, frame):
    """Check if a keyframe is world-locked.

    Returns bool.
    """
    anchors = get_anchors(gp_obj)

    if layer_name not in anchors:
        return False

    frame_str = str(frame)
    if frame_str not in anchors[layer_name]:
        return False

    data = anchors[layer_name][frame_str]

    if isinstance(data, dict):
        return data.get("world_locked", False)

    return False


def get_world_locked_frames_for_layer(gp_obj, layer_name):
    """Get list of frame numbers that are world-locked for a layer.

    Returns sorted list of frame numbers.
    """
    anchors = get_anchors(gp_obj)

    if layer_name not in anchors:
        return []

    locked_frames = []
    for frame_str, data in anchors[layer_name].items():
        if isinstance(data, dict) and data.get("world_locked", False):
            locked_frames.append(int(frame_str))

    return sorted(locked_frames)


def get_lock_matrix(gp_obj, layer_name, frame):
    """Get the lock matrix for a world-locked keyframe.

    Returns Matrix or None.
    """
    anchors = get_anchors(gp_obj)

    if layer_name not in anchors:
        return None

    frame_str = str(frame)
    if frame_str not in anchors[layer_name]:
        return None

    data = anchors[layer_name][frame_str]

    if isinstance(data, dict) and "lock_matrix" in data:
        # Convert nested list back to Matrix
        return Matrix(data["lock_matrix"])

    return None


def set_world_lock(gp_obj, layer_name, frame, locked, lock_matrix=None):
    """Set the world lock state for a keyframe.

    locked: bool - whether the keyframe is world-locked
    lock_matrix: Matrix - the object's world matrix at lock time
    """
    anchors = get_anchors(gp_obj)

    if layer_name not in anchors:
        anchors[layer_name] = {}

    frame_str = str(frame)

    # Get or create anchor data
    if frame_str in anchors[layer_name]:
        anchor_data = anchors[layer_name][frame_str]
        if not isinstance(anchor_data, dict):
            anchor_data = {"pos": anchor_data} if isinstance(anchor_data, list) else {}
    else:
        anchor_data = {}

    anchor_data["world_locked"] = locked

    if lock_matrix is not None:
        # Convert Matrix to nested list for JSON serialization
        anchor_data["lock_matrix"] = [list(row) for row in lock_matrix]
    elif not locked and "lock_matrix" in anchor_data:
        # Remove lock_matrix when unlocking
        del anchor_data["lock_matrix"]

    anchors[layer_name][frame_str] = anchor_data
    set_anchors(gp_obj, anchors)


def get_visible_keyframe(layer, current_frame):
    """Get the keyframe that is visible at the current frame.

    This is the keyframe at or before current_frame.
    Returns the keyframe object or None.
    """
    visible_kf = None
    for kf in layer.frames:
        if kf.frame_number <= current_frame:
            if visible_kf is None or kf.frame_number > visible_kf.frame_number:
                visible_kf = kf
    return visible_kf


def get_active_anchor_world_position(gp_obj, settings, scene):
    """Get the anchor position for the current frame's active keyframe.
    
    Searches backward from current frame to find the active keyframe,
    then returns its anchor position.
    
    Returns Vector or None.
    """
    if gp_obj is None or gp_obj.data is None:
        return None
    
    current_frame = scene.frame_current
    
    # Check all layers for the one with anchor data closest to current frame
    best_anchor = None
    best_frame = -1
    
    for layer in gp_obj.data.layers:
        if layer.hide:
            continue
        if settings.skip_underscore and layer.name.startswith('_'):
            continue
        if settings.layer_filter and settings.layer_filter not in layer.name:
            continue
        
        # Find keyframe at or before current frame
        for kf in layer.frames:
            if kf.frame_number <= current_frame:
                if kf.frame_number > best_frame:
                    anchor = get_anchor_for_frame(gp_obj, layer.name, kf.frame_number)
                    if anchor is not None:
                        best_anchor = anchor
                        best_frame = kf.frame_number
    
    return best_anchor


def get_all_anchor_positions(gp_obj, settings):
    """Get all anchor positions for display.

    Returns list of (Vector position, bool is_current_frame).
    """
    if gp_obj is None:
        return []

    result = []
    current_frame = bpy.context.scene.frame_current
    anchors = get_anchors(gp_obj)

    for layer_name, layer_anchors in anchors.items():
        # Check if layer passes filters
        layer = None
        for l in gp_obj.data.layers:
            if l.name == layer_name:
                layer = l
                break

        if layer is None:
            continue
        if layer.hide:
            continue
        if settings.skip_underscore and layer.name.startswith('_'):
            continue
        if settings.layer_filter and settings.layer_filter not in layer.name:
            continue

        for frame_str, anchor_data in layer_anchors.items():
            frame = int(frame_str)

            if isinstance(anchor_data, dict) and "pos" in anchor_data:
                pos = Vector(anchor_data["pos"])
            elif isinstance(anchor_data, list):
                pos = Vector(anchor_data)
            else:
                continue

            is_current = (frame == current_frame)
            result.append((pos, is_current))

    return result


# =============================================================================
# OBJECT-LEVEL WORLD LOCK SYSTEM
# =============================================================================
# New simplified lock system that works at the object level using
# matrix_parent_inverse instead of layer transforms. This provides:
# - Full GP effects compatibility (layer transforms untouched)
# - True billboard effect (strokes always face camera)
# - Simpler math (no pivot calculations)
# =============================================================================

def get_object_lock_data(gp_obj):
    """Get object-level world lock data from GP object.

    Returns dict keyed by frame number (as string):
    {
        "42": {
            "world_locked": True,
            "lock_position": [x, y, z],
            "original_parent_inverse": [[row], [row], [row], [row]]
        }
    }
    """
    if gp_obj is None:
        return {}

    # Check for new property first
    if "world_onion_locks" in gp_obj:
        try:
            return json.loads(gp_obj["world_onion_locks"])
        except (json.JSONDecodeError, TypeError):
            # Invalid or corrupted lock data
            return {}

    # Check for old data to migrate
    if "world_onion_anchors" in gp_obj:
        migrate_layer_locks_to_object_locks(gp_obj)
        if "world_onion_locks" in gp_obj:
            try:
                return json.loads(gp_obj["world_onion_locks"])
            except (json.JSONDecodeError, TypeError):
                # Migration produced invalid data
                pass

    return {}


def set_object_lock_data(gp_obj, data):
    """Save object-level world lock data to GP object."""
    if gp_obj is None:
        return
    gp_obj["world_onion_locks"] = json.dumps(data)


def is_object_locked_at_frame(gp_obj, frame):
    """Check if a specific frame is world-locked at the object level.

    Returns bool.
    """
    lock_data = get_object_lock_data(gp_obj)
    frame_str = str(frame)

    if frame_str not in lock_data:
        return False

    data = lock_data[frame_str]
    return isinstance(data, dict) and data.get("world_locked", False)


def get_lock_for_frame(gp_obj, frame):
    """Get the lock data for a specific frame.

    Returns dict with lock_position and original_parent_inverse, or None.
    """
    lock_data = get_object_lock_data(gp_obj)
    frame_str = str(frame)

    if frame_str not in lock_data:
        return None

    data = lock_data[frame_str]
    if isinstance(data, dict) and data.get("world_locked", False):
        return data

    return None


def set_lock_for_frame(gp_obj, frame, anchor_world, anchor_local_offset,
                       original_parent_inverse=None, matrix_local_at_lock=None):
    """Set world lock for a specific frame with pivot-based billboard.

    anchor_world: Vector - world position of anchor (stroke center) that stays fixed
    anchor_local_offset: Vector - offset from GP origin to anchor in GP local coordinates
    original_parent_inverse: Matrix - original matrix_parent_inverse for restore on unlock
    matrix_local_at_lock: Matrix (4x4) - the matrix_local at lock time
    """
    lock_data = get_object_lock_data(gp_obj)
    frame_str = str(frame)

    frame_data = {
        "world_locked": True,
        # Anchor position in world space (stays fixed during billboard rotation)
        "anchor_world": [anchor_world[0], anchor_world[1], anchor_world[2]],
        # Offset from GP origin to anchor in local coordinates (for pivot rotation)
        "anchor_local_offset": [anchor_local_offset[0], anchor_local_offset[1], anchor_local_offset[2]],
    }

    if original_parent_inverse is not None:
        frame_data["original_parent_inverse"] = [list(row) for row in original_parent_inverse]

    if matrix_local_at_lock is not None:
        frame_data["matrix_local"] = [list(row) for row in matrix_local_at_lock]

    lock_data[frame_str] = frame_data
    set_object_lock_data(gp_obj, lock_data)


def remove_lock_for_frame(gp_obj, frame):
    """Remove world lock for a specific frame (unlock)."""
    lock_data = get_object_lock_data(gp_obj)
    frame_str = str(frame)

    if frame_str in lock_data:
        # Keep the original_parent_inverse for reference but mark as unlocked
        if isinstance(lock_data[frame_str], dict):
            lock_data[frame_str]["world_locked"] = False
        else:
            del lock_data[frame_str]
        set_object_lock_data(gp_obj, lock_data)


def find_visible_locked_frame(gp_obj, current_frame):
    """Find which locked frame is visible at the current frame.

    Searches for the most recent keyframe that is world-locked.
    Returns the frame number or None if no lock applies.
    """
    if gp_obj is None or gp_obj.data is None:
        return None

    lock_data = get_object_lock_data(gp_obj)
    if not lock_data:
        return None

    # Get all keyframe numbers from the GP object
    all_keyframes = set()
    for layer in gp_obj.data.layers:
        for kf in layer.frames:
            all_keyframes.add(kf.frame_number)

    if not all_keyframes:
        return None

    # Find the visible keyframe (at or before current frame)
    visible_kf = None
    for kf_num in sorted(all_keyframes):
        if kf_num <= current_frame:
            visible_kf = kf_num
        else:
            break

    if visible_kf is None:
        return None

    # Check if this keyframe is locked
    if is_object_locked_at_frame(gp_obj, visible_kf):
        return visible_kf

    return None


def get_all_locked_frames(gp_obj, include_data=False):
    """Get all world-locked frames.

    Args:
        gp_obj: Grease Pencil object
        include_data: If True, return dict {frame_str: lock_data}
                      If False, return sorted list of frame numbers (default)

    Returns:
        If include_data=False: Sorted list of frame numbers
        If include_data=True: Dict of {frame_str: lock_data} for locked frames
    """
    lock_data = get_object_lock_data(gp_obj)

    if include_data:
        return {
            frame_str: data
            for frame_str, data in lock_data.items()
            if isinstance(data, dict) and data.get("world_locked", False)
        }
    else:
        locked_frames = []
        for frame_str, data in lock_data.items():
            if isinstance(data, dict) and data.get("world_locked", False):
                locked_frames.append(int(frame_str))
        return sorted(locked_frames)


def migrate_layer_locks_to_object_locks(gp_obj):
    """One-time migration from layer-level locks to object-level locks.

    Called automatically when accessing object lock data if old format exists.
    Preserves anchor data (for cursor workflow) while extracting lock data.
    """
    if gp_obj is None:
        return

    if "world_onion_anchors" not in gp_obj:
        return

    # Already migrated?
    if "world_onion_locks" in gp_obj:
        return

    try:
        old_data = json.loads(gp_obj["world_onion_anchors"])
    except (json.JSONDecodeError, TypeError):
        # Invalid or corrupted anchor data
        return

    new_lock_data = {}

    # Extract lock data from all layers
    # If any layer at a frame was locked, the frame is locked
    for layer_name, layer_data in old_data.items():
        if not isinstance(layer_data, dict):
            continue

        for frame_str, frame_data in layer_data.items():
            if not isinstance(frame_data, dict):
                continue

            if frame_data.get("world_locked", False):
                # Only migrate if not already have data for this frame
                if frame_str not in new_lock_data:
                    lock_matrix = frame_data.get("lock_matrix")
                    if lock_matrix:
                        try:
                            pos = Matrix(lock_matrix).to_translation()
                            new_lock_data[frame_str] = {
                                "world_locked": True,
                                "lock_position": list(pos),
                            }
                        except (TypeError, ValueError):
                            # Invalid matrix format
                            pass

    if new_lock_data:
        gp_obj["world_onion_locks"] = json.dumps(new_lock_data)


def migrate_object_lock_frame(gp_obj, old_frame, new_frame):
    """Migrate object lock data when a keyframe is moved in the timeline."""
    lock_data = get_object_lock_data(gp_obj)

    old_frame_str = str(old_frame)
    new_frame_str = str(new_frame)

    if old_frame_str in lock_data:
        lock_data[new_frame_str] = lock_data[old_frame_str]
        del lock_data[old_frame_str]
        set_object_lock_data(gp_obj, lock_data)
