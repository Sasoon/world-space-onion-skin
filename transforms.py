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


def ensure_billboard_constraint(gp_obj, scene):
    """Ensure the GP object has a billboard constraint targeting the active camera.
    
    Uses COPY_ROTATION to match camera orientation so strokes always face the viewer.
    Returns True if constraint was added or modified.
    """
    if gp_obj is None:
        return False

    camera = scene.camera
    if camera is None:
        return False

    CONSTRAINT_NAME = "WorldOnion_Billboard"
    modified = False

    # Find existing constraint
    constraint = gp_obj.constraints.get(CONSTRAINT_NAME)
    
    if constraint is None:
        constraint = gp_obj.constraints.new(type='COPY_ROTATION')
        constraint.name = CONSTRAINT_NAME
        modified = True
    
    # Ensure all settings are correct
    if constraint.use_x != True:
        constraint.use_x = True
        modified = True
    if constraint.use_y != True:
        constraint.use_y = True
        modified = True
    if constraint.use_z != True:
        constraint.use_z = True
        modified = True
    if constraint.mix_mode != 'REPLACE':
        constraint.mix_mode = 'REPLACE'
        modified = True
    if constraint.target_space != 'WORLD':
        constraint.target_space = 'WORLD'
        modified = True
    if constraint.owner_space != 'WORLD':
        constraint.owner_space = 'WORLD'
        modified = True
    if constraint.influence != 1.0:
        constraint.influence = 1.0
        modified = True
    if constraint.mute:
        constraint.mute = False
        modified = True
    
    # Ensure target is correct
    if constraint.target != camera:
        constraint.target = camera
        modified = True
    
    # Force depsgraph update if we made changes
    if modified:
        try:
            bpy.context.view_layer.update()
        except (RuntimeError, AttributeError):
            pass
    
    return modified


def align_strokes_to_camera(stroke_points, anchor_pos, scene):
    """
    Rotate stroke points around anchor to face the camera.

    For 2.5D workflows, strokes should lie on a plane perpendicular to the
    camera's view direction. This function rotates stroke points so they
    face the camera, preserving their shape.

    Args:
        stroke_points: List of world-space Vector points
        anchor_pos: Vector - center of rotation
        scene: Blender scene (to get camera)

    Returns:
        List of rotated world-space Vector points
    """
    camera = scene.camera
    if camera is None:
        return stroke_points  # No camera, can't align

    if len(stroke_points) < 3:
        return stroke_points  # Need at least 3 points to determine plane

    # 1. Get camera's forward direction (negative Z in camera space)
    cam_matrix = camera.matrix_world
    cam_forward = -(cam_matrix.to_3x3() @ Vector((0, 0, 1)))
    cam_forward.normalize()

    # Target normal = camera forward (strokes should face camera)
    target_normal = cam_forward

    # 2. Compute current stroke plane normal using cross product of edges
    # Use first, middle, and last points to estimate plane
    p0 = stroke_points[0]
    p1 = stroke_points[len(stroke_points) // 2]
    p2 = stroke_points[-1]

    edge1 = p1 - p0
    edge2 = p2 - p0
    current_normal = edge1.cross(edge2)

    if current_normal.length < 0.0001:
        return stroke_points  # Degenerate (collinear points)

    current_normal.normalize()

    # 3. Calculate rotation from current normal to target normal
    rotation = current_normal.rotation_difference(target_normal)

    # 4. Rotate all points around anchor
    aligned_points = []
    for p in stroke_points:
        # Translate to anchor origin, rotate, translate back
        relative = p - anchor_pos
        rotated = rotation @ relative
        aligned_points.append(anchor_pos + rotated)

    return aligned_points


def adjust_obj_to_surface(gp_obj, scene):
    """
    Adjust GP object location to sit on mesh surface (raycast down).
    Used for shrinkwrap behavior without baking keys or constraints.
    Returns True if adjusted.
    """
    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
    except (RuntimeError, AttributeError):
        return False

    # Current location (likely from F-Curve)
    current_pos = gp_obj.location
    
    # Raycast down from above
    ray_origin = Vector((current_pos.x, current_pos.y, current_pos.z + 1000.0))
    ray_dir = Vector((0, 0, -1))
    
    hit, location, normal, index, hit_obj, matrix = scene.ray_cast(
        depsgraph, ray_origin, ray_dir
    )
    
    # Ignore self (the GP object itself won't be hit by ray_cast usually, but to be safe)
    if hit and hit_obj != gp_obj:
        new_z = location.z + SURFACE_OFFSET
        # Only update if significant change to avoid float jitter fighting F-Curve
        if abs(new_z - current_pos.z) > 0.0001:
            try:
                gp_obj.location.z = new_z
                return True
            except AttributeError:
                # Writing not allowed in this context (render, playback, etc.)
                return False

    return False
