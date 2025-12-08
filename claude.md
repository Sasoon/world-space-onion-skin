# World Space Onion Skin - Blender Addon

## Overview

A Blender addon that provides **world-space onion skinning** for Grease Pencil objects. Unlike Blender's built-in onion skinning (which displays strokes in local/object space), this addon keeps reference frames fixed in world space - essential for camera-parented GP objects in storyboarding and animation workflows.

**Total codebase:** ~2,500 lines across 9 Python modules

## Key Features

- World-space onion skins with customizable before/after colors and opacity falloff
- Two frame selection modes: KEYFRAMES (N keyframes before/after) or FRAMES (every Nth frame)
- Anchor system for storing/restoring drawing positions per frame/layer
- Motion path visualization showing object movement across keyframes
- Depth interaction (shrinkwrap) - strokes follow mesh surfaces via raycasting
- Auto-cursor positioning for efficient world-space drawing
- Automatic billboard constraint (GP always faces camera)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      USER / TIMELINE                        │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
                   ┌───────────────────┐
                   │  handlers.py      │  Frame change & depsgraph events
                   └─────────┬─────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
   ┌───────────┐      ┌────────────┐      ┌─────────────┐
   │ cache.py  │      │transforms.py│     │ anchors.py  │
   │           │      │            │      │             │
   │ Extract & │      │ Matrix ops │      │ Position    │
   │ cache     │      │ Raycast    │      │ metadata    │
   │ strokes   │      │ Billboard  │      │ (JSON)      │
   └─────┬─────┘      └────────────┘      └─────────────┘
         │
         ▼
   ┌───────────────────────────────────────┐
   │            drawing.py                 │
   │                                       │
   │  GPU callbacks (onion, anchor, path)  │
   └───────────────────────────────────────┘
```

## File Structure

| File | Purpose |
|------|---------|
| `__init__.py` | Addon registration, handler setup, bl_info |
| `settings.py` | PropertyGroup with all user settings + update callbacks |
| `cache.py` | Frame data caching, stroke extraction from GP objects |
| `transforms.py` | Matrix utilities, raycast, billboard constraint |
| `anchors.py` | Anchor metadata management (persistent JSON per frame/layer) |
| `handlers.py` | Event handlers for frame changes, depsgraph updates |
| `drawing.py` | GPU draw callbacks for onion skins, anchors, motion paths |
| `operators.py` | User-facing operators (cache, anchor, snapping) |
| `ui.py` | UI panels in View3D sidebar |

## Core Data Structures

### Stroke Data (cached per frame)
```python
{
    'points': [Vector(...), ...],      # World-space positions
    'layer': str,                       # Layer name
    'frame': int,                       # Frame number
    'fill_triangles': [(i, j, k), ...], # Triangulation indices for fills
}
```

### Anchor Data (JSON in GP object custom property)
```python
{
    "layer_name": {
        "123": {                         # Frame number as string key
            "pos": [x, y, z],           # Cursor position
            "cam_dir": [x, y, z]        # Camera direction snapshot
        }
    }
}
```

### Global Cache
```python
_cache = {
    frame_number: [stroke_data, stroke_data, ...],
    ...
}
```

## Key Implementation Details

### 1. World-Space Stroke Transform
Strokes must be transformed through the full matrix chain:
```python
# cache.py
layer_matrix = get_layer_transform(layer)  # Layer TRS
full_matrix = gp_obj.matrix_world @ layer_matrix
world_pos = full_matrix @ local_pos
```

### 2. GPU Rendering Pipeline
Three separate draw handlers registered to `SpaceView3D`:
- `draw_onion_callback()` - Renders onion skin strokes/fills
- `draw_anchor_callback()` - Renders anchor position indicators
- `draw_motion_path_callback()` - Renders motion path lines

Uses shaders:
- `POLYLINE_UNIFORM_COLOR` for strokes (respects line width)
- `UNIFORM_COLOR` for fills and anchor crosses

### 3. Frame Selection Modes
```python
if settings.mode == 'KEYFRAMES':
    # Show N keyframes before/after current
    frames_to_show = get_keyframe_based_frames(...)
else:
    # Show every Nth frame in range
    frames_to_show = get_regular_frames(...)
```

### 4. Shrinkwrap (Depth Interaction)
Raycast from above to find mesh surface:
```python
# transforms.py
ray_origin = Vector((pos.x, pos.y, pos.z + 1000.0))
ray_dir = Vector((0, 0, -1))
hit, location, _, _, hit_obj, _ = scene.ray_cast(depsgraph, ray_origin, ray_dir)
if hit and hit_obj != gp_obj:
    gp_obj.location.z = location.z + SURFACE_OFFSET
```

### 5. Anchor Migration
When keyframes are moved in timeline, anchor metadata automatically follows:
```python
# handlers.py - on_depsgraph_update
migrate_anchor_data(gp_obj, layer_name, old_frame, new_frame)
```

## Settings (WorldOnionSettings PropertyGroup)

| Category | Properties |
|----------|------------|
| **Enable** | `enabled`, `mode` (FRAMES/KEYFRAMES) |
| **Range** | `frames_before`, `frames_after`, `frame_step` |
| **Appearance** | `opacity`, `falloff`, `fill_opacity`, `line_width`, `color_before`, `color_after` |
| **Motion Path** | `motion_path_enabled`, `motion_path_color`, `motion_path_width`, `motion_path_show_points`, `motion_path_smoothing` |
| **Depth** | `depth_interaction_enabled`, `stroke_z_offset` |
| **Anchors** | `anchor_enabled`, `anchor_auto_cursor` |

Update callbacks:
- `update_enabled` - Registers/unregisters draw handlers
- `update_setting` - Marks viewport for redraw
- `update_realtime` - Forces frame re-evaluation

## Operators

| Operator | Description |
|----------|-------------|
| `WONION_OT_clear_cache` | Clear stroke cache |
| `WONION_OT_build_cache` | Pre-cache all frames (scrubs timeline) |
| `WONION_OT_set_anchor` | Move object to cursor, preserve stroke positions |
| `WONION_OT_snap_to_gp` | Snap cursor to stroke center (XY) / lowest point (Z) |
| `WONION_OT_snap_to_cursor` | Move selected strokes AND object to cursor |
| `WONION_OT_clear_anchor` | Clear anchor for current frame |
| `WONION_OT_clear_all_anchors` | Clear all anchor data |
| `WONION_OT_reload_addon` | Development: full reload cycle |

## Design Patterns

### 1. Persistent Handlers with Global State
```python
_last_keyframe_set = set()  # Track previous state for change detection

@persistent
def on_frame_change(scene):
    ...
```

### 2. Lazy Shader Caching
```python
_stroke_shader = None

def _get_stroke_shader():
    global _stroke_shader
    if _stroke_shader is None:
        _stroke_shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    return _stroke_shader
```

### 3. Metadata via Custom Properties
```python
gp_obj["world_onion_anchors"] = json.dumps(anchors)  # Persists with .blend
```

### 4. Two-Phase Cache
1. **Extraction** (on frame change) - Transform strokes to world space
2. **Rendering** (on viewport redraw) - Fetch pre-transformed data

## Dependencies

**Blender APIs:**
- `bpy.types`, `bpy.props` - Properties and registration
- `bpy.app.handlers` - Event handlers (frame_change_post, depsgraph_update_post, load_post)
- Grease Pencil API (GPencilLayer, GPencilFrame, GPencilDrawing, curve_offsets)
- `gpu`, `gpu.shader`, `gpu_extras.batch` - GPU rendering
- `mathutils` - Vector, Matrix, tessellate_polygon

**Python stdlib:**
- `json` - Anchor serialization
- `importlib` - Module reloading (dev)

## Development Notes

### Adding New Settings
1. Add property to `WorldOnionSettings` in `settings.py`
2. Add UI in `ui.py` panel
3. Use appropriate update callback (`update_setting` for display, `update_realtime` for transforms)

### Debugging
- Use `WONION_OT_reload_addon` operator to hot-reload during development
- Check Blender's System Console for print statements
- Cache issues: call `clear_cache()` from `cache.py`

### Common Issues
- **Strokes not showing**: Check layer visibility, verify cache has data
- **Wrong positions**: Layer transforms not being applied - check `get_layer_transform()`
- **Anchors not persisting**: JSON serialization issue in custom property

## Version History

Recent focus areas:
- Object-level implementation (refactored from layer-level)
- Motion path visualization with raycast support
- Shrinkwrap/depth interaction
- Billboard constraint reliability
