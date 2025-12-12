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
Version: 9.1.0
"""

bl_info = {
    "name": "World Space Onion Skin",
    "author": "Claude + Sasoon",
    "version": (9, 1, 0),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar > Onion",
    "description": "World-space onion skinning for GP storyboarding",
    "category": "Animation",
}

import bpy
from bpy.app.handlers import persistent

# Handle reloading for development - MUST reload in dependency order
# debug_log first (no deps), transforms, cache, anchors, drawing, handlers, operators, settings, ui
if "debug_log" in locals():
    import importlib
    importlib.reload(debug_log)
    importlib.reload(transforms)
    importlib.reload(cache)
    importlib.reload(anchors)
    importlib.reload(drawing)
    importlib.reload(handlers)
    importlib.reload(operators)
    importlib.reload(settings)
    importlib.reload(ui)

from . import debug_log
from . import transforms
from . import cache
from . import anchors
from . import drawing  # drawing before handlers (handlers imports from drawing)
from . import handlers
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
    # v8.1: Always re-register driver namespace on file load (drivers persist but namespace doesn't)
    drawing.register_driver_namespace()

    # Check if any scene has the addon enabled
    for scene in bpy.data.scenes:
        if hasattr(scene, 'world_onion') and scene.world_onion.enabled:
            drawing.register_draw_handlers()

            # v9.3: Re-bake shrinkwrap if it was enabled
            # Module globals are reset on file load, so baked offset dictionary is empty.
            # Without this, driver returns 0.0 and shrinkwrap doesn't work until user scrubs.
            # on_load_post() is a safe context - driver setup will succeed here.
            if scene.world_onion.depth_interaction_enabled:
                gp_obj = cache.get_active_gp(bpy.context)
                if gp_obj:
                    drawing.bake_shrinkwrap_offsets(gp_obj, scene.world_onion, scene, setup_driver=True)
                    scene.frame_set(scene.frame_current)

            # v8.5: Start cursor sync modal operator
            from .operators import is_cursor_sync_running
            if not is_cursor_sync_running():
                bpy.ops.world_onion.cursor_sync('INVOKE_DEFAULT')
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

    # Register load handler to restore draw callbacks after file load
    if on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(on_load_post)

    # v8.1: Register driver namespace function for shrinkwrap offset lookup
    drawing.register_driver_namespace()

    print("World Space Onion Skin registered")


def unregister():
    """Unregister the addon."""
    # Unregister load handler
    if on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(on_load_post)

    # v8.1: Unregister driver namespace function
    drawing.unregister_driver_namespace()

    # v8.5: Modal operator auto-cancels when addon disables (no explicit stop needed)

    # Unregister draw handlers
    drawing.unregister_draw_handlers()

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
