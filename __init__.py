"""
World Space Onion Skin - Blender Addon

Provides world-space onion skinning for Grease Pencil objects,
particularly useful for storyboarding with camera-parented GP objects.

Features:
- World-space onion skin display (strokes stay in world position)
- Anchor system for cursor positioning per frame/layer
- Runtime world lock (toggle per keyframe, no baking required)
- Canvas alignment to cursor

Author: Claude + Sasoon
Version: 8.0.0
"""

bl_info = {
    "name": "World Space Onion Skin",
    "author": "Claude + Sasoon",
    "version": (8, 0, 0),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar > Onion",
    "description": "World-space onion skinning for GP storyboarding",
    "category": "Animation",
}

import bpy
from bpy.app.handlers import persistent

# Handle reloading for development
if "bpy" in locals():
    import importlib
    if "transforms" in locals():
        importlib.reload(transforms)
    if "cache" in locals():
        importlib.reload(cache)
    if "anchors" in locals():
        importlib.reload(anchors)
    if "handlers" in locals():
        importlib.reload(handlers)
    if "drawing" in locals():
        importlib.reload(drawing)
    if "timeline_drawing" in locals():
        importlib.reload(timeline_drawing)
    if "operators" in locals():
        importlib.reload(operators)
    if "settings" in locals():
        importlib.reload(settings)
    if "ui" in locals():
        importlib.reload(ui)

from . import transforms
from . import cache
from . import anchors
from . import handlers
from . import drawing
from . import timeline_drawing
from . import operators
from . import settings
from . import ui


# All classes to register
classes = (
    settings.WorldOnionSettings,
    *operators.operator_classes,
    *ui.panel_classes,
)


@persistent
def on_load_post(dummy):
    """Called after a .blend file is loaded. Re-register draw handlers if addon is enabled."""
    # Check if any scene has the addon enabled
    for scene in bpy.data.scenes:
        if hasattr(scene, 'world_onion') and scene.world_onion.enabled:
            drawing.register_draw_handlers()
            break


def register():
    """Register the addon."""
    # Register classes
    for cls in classes:
        bpy.utils.register_class(cls)

    # Register property group
    bpy.types.Scene.world_onion = bpy.props.PointerProperty(type=settings.WorldOnionSettings)

    # Register handlers
    handlers.register_handlers()

    # Register timeline drawing
    timeline_drawing.register_timeline_handlers()

    # Register load handler to restore draw callbacks after file load
    if on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(on_load_post)

    print("World Space Onion Skin registered")


def unregister():
    """Unregister the addon."""
    # Unregister load handler
    if on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(on_load_post)

    # Unregister draw handlers
    drawing.unregister_draw_handlers()

    # Unregister timeline drawing
    timeline_drawing.unregister_timeline_handlers()

    # Unregister handlers
    handlers.unregister_handlers()

    # Unregister property group
    if hasattr(bpy.types.Scene, 'world_onion'):
        del bpy.types.Scene.world_onion

    # Unregister classes
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            # Class may not be registered
            pass

    print("World Space Onion Skin unregistered")


if __name__ == "__main__":
    register()
