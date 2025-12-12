"""
Cache management for world-space onion skinning.
"""

import bisect
from collections import OrderedDict

import bpy
from mathutils import Vector
from mathutils.geometry import tessellate_polygon

from .transforms import get_layer_transform


def _find_active_keyframe(frames, current_frame):
    """
    Binary search for the active keyframe at or before current_frame.
    Returns the keyframe object or None if no keyframe is at/before current_frame.
    O(log n) instead of O(n) linear search.

    Note: Blender keeps layer.frames sorted by frame_number internally,
    so we skip the O(n log n) sort and just build the frame_numbers list.
    """
    if not frames:
        return None

    # Blender keeps frames sorted - just extract frame numbers for bisect
    # This is O(n) list creation vs O(n log n) sorting on every call
    frame_numbers = [kf.frame_number for kf in frames]

    # Find insertion point - the index where current_frame would be inserted
    idx = bisect.bisect_right(frame_numbers, current_frame)

    # idx-1 gives us the keyframe at or before current_frame
    if idx > 0:
        return frames[idx - 1]
    return None


def get_active_gp(context):
    """Get active GP object, or None if not a GP object."""
    obj = context.active_object
    if obj is not None and obj.type == 'GREASEPENCIL':
        return obj
    return None


def triangulate_fill(world_points):
    """
    Triangulate a polygon for fill rendering.
    Works with any polygon - Blender implicitly closes open strokes for fills.
    Returns list of triangle indices [(i, j, k), ...] or empty list if not fillable.
    """
    if len(world_points) < 3:
        return []

    try:
        # tessellate_polygon expects a list of loops (for polygons with holes)
        # Each loop is a list of 3D points as tuples
        poly = [[tuple(p) for p in world_points]]
        triangles = tessellate_polygon(poly)
        return triangles
    except Exception:
        # Triangulation can fail for degenerate polygons
        return []


# Global cache - OrderedDict for O(1) eviction of oldest entries
_cache = OrderedDict()  # {frame_number: [stroke_points_list, ...]}


def get_cache():
    """Get the global cache dict."""
    return _cache


def clear_cache():
    """Clear all cached frames and invalidate GPU batch cache."""
    global _cache
    _cache = OrderedDict()
    # Also invalidate onion batch and keyframe cache since stroke data changed
    try:
        from .drawing import invalidate_onion_batch_cache, invalidate_keyframe_cache
        invalidate_onion_batch_cache()
        invalidate_keyframe_cache()  # P7: Keyframes may have changed
    except ImportError:
        pass  # drawing module not loaded yet


def get_cache_stats():
    """Get cache statistics string."""
    return f"{len(_cache)} frames cached"


def extract_strokes_at_current_frame(gp_obj, settings):
    """
    Extract world-space stroke data from the GP object at current frame.
    Returns list of stroke data dicts.
    """
    strokes_data = []

    if gp_obj is None or gp_obj.data is None:
        return strokes_data

    materials = gp_obj.data.materials  # Get materials list for fill detection
    world_matrix = gp_obj.matrix_world

    for layer in gp_obj.data.layers:
        if layer.hide:
            continue
        
        layer_matrix = get_layer_transform(layer)
        full_matrix = world_matrix @ layer_matrix
        
        current_frame = bpy.context.scene.frame_current
        # Use binary search for O(log n) keyframe lookup
        active_kf = _find_active_keyframe(layer.frames, current_frame)

        if active_kf is None:
            continue
        
        if active_kf.drawing is None:
            continue
            
        drawing = active_kf.drawing
        
        if not hasattr(drawing, 'attributes') or 'position' not in drawing.attributes:
            continue
            
        pos_attr = drawing.attributes['position']
        
        if not hasattr(drawing, 'curve_offsets'):
            continue
        
        curve_offsets = list(drawing.curve_offsets)
        if not curve_offsets:
            continue

        curve_offsets_values = [co.value for co in curve_offsets]
        num_points = len(pos_attr.data)

        # Get material indices (CURVE domain - one per stroke)
        if 'material_index' in drawing.attributes:
            mat_idx_attr = drawing.attributes['material_index']
            material_indices = [mi.value for mi in mat_idx_attr.data]
        else:
            # No material_index attribute = all curves use index 0
            material_indices = [0] * len(curve_offsets_values)

        for i, start_idx in enumerate(curve_offsets_values):
            if i + 1 < len(curve_offsets_values):
                end_idx = curve_offsets_values[i + 1]
            else:
                end_idx = num_points

            if start_idx >= end_idx:
                continue

            # Extract world points as tuples (not Vectors) for GPU efficiency
            # This eliminates tuple conversion overhead during every viewport redraw
            world_points = []
            for p_idx in range(start_idx, end_idx):
                local_pos = Vector(pos_attr.data[p_idx].vector)
                world_pos = full_matrix @ local_pos
                world_points.append((world_pos.x, world_pos.y, world_pos.z))

            if len(world_points) >= 2:
                # Check material fill setting (not geometric closure)
                mat_idx = material_indices[i] if i < len(material_indices) else 0
                has_fill = False
                if mat_idx >= 0 and mat_idx < len(materials) and materials[mat_idx] is not None:
                    # GP material settings are under material.grease_pencil
                    gp_mat = materials[mat_idx].grease_pencil
                    if gp_mat is not None:
                        has_fill = gp_mat.show_fill

                # Triangulate if material has fill enabled
                fill_triangles = []
                if has_fill and len(world_points) >= 3:
                    fill_triangles = triangulate_fill(world_points)

                stroke_data = {
                    'points': world_points,
                    'layer': layer.name,
                    'frame': active_kf.frame_number,
                    'fill_triangles': fill_triangles,
                }

                strokes_data.append(stroke_data)
    
    return strokes_data


def cache_current_frame(gp_obj, settings):
    """Cache strokes for the current frame.

    PERFORMANCE: In KEYFRAMES mode, uses binary search to check if current
    frame is a keyframe, instead of O(layers × keyframes) full scan.
    """
    global _cache
    frame = bpy.context.scene.frame_current

    # In KEYFRAMES mode, only cache if this is effectively a keyframe
    if settings.mode == 'KEYFRAMES':
        # Use binary search per layer instead of full scan
        is_keyframe = False
        for layer in gp_obj.data.layers:
            if layer.hide:
                continue
            # Blender keeps layer.frames sorted - use binary search
            frames = layer.frames
            if frames:
                # Quick check: binary search for exact frame match
                frame_numbers = [kf.frame_number for kf in frames]
                idx = bisect.bisect_left(frame_numbers, frame)
                if idx < len(frame_numbers) and frame_numbers[idx] == frame:
                    is_keyframe = True
                    break

        if not is_keyframe:
            return

    strokes = extract_strokes_at_current_frame(gp_obj, settings)
    _cache[frame] = strokes

    # Limit cache size - O(1) eviction with OrderedDict
    max_cached = 2000
    while len(_cache) > max_cached:
        _cache.popitem(last=False)  # Remove oldest entry (FIFO)
