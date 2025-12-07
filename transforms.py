"""
Transform and matrix utilities for world-space onion skinning.
"""

import bpy
from mathutils import Matrix, Vector


# Shared constants
SURFACE_OFFSET = 0.01  # Small offset to keep strokes visible on mesh surfaces


def catmull_rom_point(p0, p1, p2, p3, t):
    """Calculate a single point on a Catmull-Rom spline.

    Args:
        p0, p1, p2, p3: Control points (tuples or vectors with x, y, z)
        t: Parameter from 0 to 1 (interpolates between p1 and p2)

    Returns:
        Vector (x, y, z) of the interpolated point
    """
    t2 = t * t
    t3 = t2 * t

    x = 0.5 * ((2 * p1[0]) +
               (-p0[0] + p2[0]) * t +
               (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
               (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
    y = 0.5 * ((2 * p1[1]) +
               (-p0[1] + p2[1]) * t +
               (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
               (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
    z = 0.5 * ((2 * p1[2]) +
               (-p0[2] + p2[2]) * t +
               (2 * p0[2] - 5 * p1[2] + 4 * p2[2] - p3[2]) * t2 +
               (-p0[2] + 3 * p1[2] - 3 * p2[2] + p3[2]) * t3)

    return Vector((x, y, z))


def get_layer_transform(layer):
    """Build transformation matrix for a GP layer."""
    try:
        t = Matrix.Translation(layer.translation)
        r = layer.rotation.to_matrix().to_4x4()
        s = Matrix.Diagonal((*layer.scale, 1.0))
        return t @ r @ s
    except (AttributeError, TypeError, ValueError):
        # Layer missing expected properties or invalid values
        return Matrix.Identity(4)


def get_world_matrix_at_frame(obj, scene, frame):
    """Get object's world matrix at a specific frame."""
    original_frame = scene.frame_current
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    matrix = obj.matrix_world.copy()
    scene.frame_set(original_frame)
    bpy.context.view_layer.update()
    return matrix


def get_camera_direction(scene):
    """Get the current camera's forward direction in world space.

    Returns Vector or None if no scene camera.
    """
    camera = scene.camera
    if camera is None:
        return None
    cam_matrix = camera.matrix_world
    return -(cam_matrix.to_3x3() @ Vector((0, 0, 1)))


def align_canvas_to_cursor(context):
    """Align the GP canvas to the 3D cursor position.
    
    This sets the GP stroke placement to use the 3D cursor location,
    which makes the canvas grid appear at the cursor position.
    Then offsets Y so cursor aligns with canvas bottom edge.
    """
    scene = context.scene
    
    # Set stroke placement to 3D cursor - this moves canvas origin to cursor
    ts = scene.tool_settings
    ts.gpencil_stroke_placement_view3d = 'CURSOR'
    
    # Offset canvas so cursor is at bottom, not center
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    overlay = space.overlay
                    
                    # Get canvas scale to calculate half height
                    scale_y = overlay.gpencil_grid_scale[1] if hasattr(overlay, 'gpencil_grid_scale') else 1.0
                    half_height = scale_y / 2.0
                    
                    # X stays centered, Y offset by half height to put bottom at cursor
                    overlay.gpencil_grid_offset[0] = 0.0
                    overlay.gpencil_grid_offset[1] = half_height
                    break
