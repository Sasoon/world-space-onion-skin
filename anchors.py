"""
Anchor system for world-space drawing positions.
Removes legacy lock system. Anchors now just map to keyframe locations.
"""

import bpy
import json
from mathutils import Vector, Matrix

from .transforms import get_layer_transform


# JSON cache - avoid re-parsing JSON on every anchor lookup
_anchor_json_cache = None
_anchor_json_cache_gp = None


def invalidate_anchor_json_cache():
    """Clear anchor JSON cache. Call when anchor data changes."""
    global _anchor_json_cache, _anchor_json_cache_gp
    _anchor_json_cache = None
    _anchor_json_cache_gp = None


def _invalidate_all_anchor_caches():
    """Invalidate both JSON cache and GPU batch cache."""
    invalidate_anchor_json_cache()
    # Also invalidate GPU batch cache (import here to avoid circular import)
    try:
        from .drawing import invalidate_anchor_batch_cache
        invalidate_anchor_batch_cache()
    except ImportError:
        pass


def get_anchors(gp_obj, use_cache=True):
    """Get anchor data from GP object custom property.

    PERFORMANCE: Uses module-level cache to avoid JSON parsing on every call.
    Pass use_cache=False to bypass cache (e.g., after external modification).
    """
    global _anchor_json_cache, _anchor_json_cache_gp

    if gp_obj is None:
        return {}

    # Check cache first (identity comparison for speed)
    if use_cache and gp_obj is _anchor_json_cache_gp and _anchor_json_cache is not None:
        return _anchor_json_cache

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

        # Update cache
        _anchor_json_cache = data
        _anchor_json_cache_gp = gp_obj

        return data
    except (json.JSONDecodeError, TypeError, KeyError):
        # Invalid or corrupted anchor data
        return {}


def set_anchors(gp_obj, anchors):
    """Save anchor data to GP object custom property."""
    if gp_obj is None:
        return
    gp_obj["world_onion_anchors"] = json.dumps(anchors)
    # Invalidate caches since anchor data changed
    _invalidate_all_anchor_caches()


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


def set_anchor_for_frame(gp_obj, layer_name, frame, position, camera_dir=None):
    """Set the anchor position and camera direction for a specific layer and frame.

    position: Vector or tuple (x, y, z)
    camera_dir: Vector or tuple (x, y, z) - direction camera is facing
    """
    anchors = get_anchors(gp_obj)

    if layer_name not in anchors:
        anchors[layer_name] = {}

    frame_str = str(frame)

    # Preserve existing data
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
    """Move anchor data from old frame to new frame."""
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


def calculate_anchor_from_strokes(gp_obj, layer, frame_number, return_local=False):
    """Calculate anchor position from strokes: center XY, lowest Z in WORLD space.

    Transforms all points to world space first, then computes:
    - X: average of all world X coordinates
    - Y: average of all world Y coordinates
    - Z: minimum of all world Z coordinates (so strokes sit ON surfaces)

    Args:
        gp_obj: The GP object
        layer: The layer to get strokes from
        frame_number: Frame number to find keyframe
        return_local: If True, returns (anchor_world, anchor_local) tuple

    Returns:
        If return_local=False: Vector (world position) or None
        If return_local=True: (Vector world, Vector local) or (None, None)
    """
    if gp_obj is None or layer is None:
        return (None, None) if return_local else None

    # Find the visible keyframe (at or before frame_number)
    # This matches the logic in set_anchor_logic() for consistency
    keyframe = None
    for kf in layer.frames:
        if kf.frame_number <= frame_number:
            if keyframe is None or kf.frame_number > keyframe.frame_number:
                keyframe = kf

    if keyframe is None or keyframe.drawing is None:
        return (None, None) if return_local else None

    drawing = keyframe.drawing

    if not hasattr(drawing, 'attributes') or 'position' not in drawing.attributes:
        return (None, None) if return_local else None

    pos_attr = drawing.attributes['position']
    if len(pos_attr.data) == 0:
        return (None, None) if return_local else None

    # Get full transform matrix (object + layer) for transforming local to world
    # This matches how strokes are transformed in cache.py
    matrix_world = gp_obj.matrix_world
    layer_matrix = get_layer_transform(layer)
    full_matrix = matrix_world @ layer_matrix

    # Compute anchor in WORLD coordinates (center XY, lowest Z)
    world_min_z = float('inf')
    world_sum_x = 0.0
    world_sum_y = 0.0
    count = 0

    for p in pos_attr.data:
        local_pos = Vector(p.vector)
        world_pos = full_matrix @ local_pos
        world_sum_x += world_pos.x
        world_sum_y += world_pos.y
        if world_pos.z < world_min_z:
            world_min_z = world_pos.z
        count += 1

    if count == 0:
        return (None, None) if return_local else None

    # Anchor in world coordinates
    anchor_world = Vector((world_sum_x / count, world_sum_y / count, world_min_z))

    if return_local:
        # Transform back to local if needed
        anchor_local = gp_obj.matrix_world.inverted() @ anchor_world
        return anchor_world, anchor_local
    return anchor_world


def get_current_keyframes_set(gp_obj, settings):
    """Get a set of (layer_name, frame_number) for all current keyframes."""
    result = set()
    
    if gp_obj is None or gp_obj.data is None:
        return result
    
    for layer in gp_obj.data.layers:
        if layer.hide:
            continue
        
        for kf in layer.frames:
            result.add((layer.name, kf.frame_number))
    
    return result


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


def get_all_anchor_positions(gp_obj, settings):
    """Get all anchor positions for display.

    Returns list of (Vector position, bool is_current_frame).

    PERFORMANCE: Uses dict lookup for layers (O(1)) instead of O(n) search per anchor.
    """
    if gp_obj is None:
        return []

    result = []
    current_frame = bpy.context.scene.frame_current
    anchors = get_anchors(gp_obj)

    if not anchors:
        return result

    # Build layer lookup dict ONCE (O(n) instead of O(n*m))
    layer_dict = {l.name: l for l in gp_obj.data.layers}

    for layer_name, layer_anchors in anchors.items():
        # O(1) layer lookup instead of O(n) loop
        layer = layer_dict.get(layer_name)

        if layer is None or layer.hide:
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
