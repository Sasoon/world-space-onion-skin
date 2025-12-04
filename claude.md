# World Space Onion Skin - Developer Documentation

> **Version:** 9.1.0 | **Blender:** 4.5+ | **Authors:** Claude + Sasoon

## What This Addon Does

This is a **specialized animation tool for Grease Pencil storyboarding** that provides world-space onion skinning. Traditional onion skinning shows previous/future frames in local space (following parent transformations). This addon keeps frames visible in **world space**, making it ideal for storyboards with animated cameras where strokes should stay where they were drawn in the world.

### Core Features

1. **World-Space Onion Skin Display** - Shows previous/future GP frames at their world positions
2. **World Lock System** - Locks keyframes to world space so strokes don't follow camera movement
3. **Anchor System** - Tracks reference positions per-keyframe for precise stroke placement
4. **Timeline Indicators** - Visual spans in Dopesheet showing when world-locks are active
5. **Auto-Cursor Movement** - Moves 3D cursor to anchor positions on frame change
6. **True Billboard Effect** - Strokes always face the camera from any viewing angle
7. **Full GP Effects Compatibility** - Layer transforms are untouched, effects work normally

### The Vision

The addon solves a specific storyboarding workflow: drawing on camera-parented Grease Pencil objects where you want strokes to remain at their world position even as the camera animates. Traditional GP onion skin would show previous frames following the camera transform, making it impossible to see where strokes actually are in world space.

### Architecture Note (v9.0.0)

The addon uses an **object-level** world-lock system via `matrix_parent_inverse`. This approach:
- Keeps all layer transforms at identity (effects work perfectly)
- Provides true billboard rendering (strokes face camera from any angle)
- Uses simpler matrix math (no pivot calculations)
- Locks entire frames at object level (not per-layer)

---

## Architecture Overview

```
REGISTRATION (__init__.py)
       │
       v
SETTINGS (settings.py) ──> Property callbacks
       │
       v
UI/OPERATORS (ui.py, operators.py)
       │
       v
BUSINESS LOGIC
├── cache.py      - Stroke extraction & caching
├── anchors.py    - Anchor/world-lock data persistence
├── transforms.py - Matrix math utilities
└── handlers.py   - Event handlers (frame change, depsgraph)
       │
       v
GPU RENDERING
├── drawing.py          - Viewport onion skin rendering
└── timeline_drawing.py - Dopesheet world-lock indicators
```

---

## File Structure

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | ~150 | Registration, module reload, handler setup |
| `settings.py` | ~180 | PropertyGroup with all user settings |
| `ui.py` | ~280 | Panel layouts (main + 6 sub-panels) |
| `operators.py` | ~450 | 9 operators for cache, anchors, world-lock |
| `cache.py` | ~200 | Stroke extraction, triangulation, caching |
| `anchors.py` | ~250 | Anchor read/write, world-lock state |
| `handlers.py` | ~400 | Frame change + depsgraph handlers |
| `drawing.py` | ~250 | GPU onion skin rendering |
| `timeline_drawing.py` | ~200 | Dopesheet indicator drawing |
| `transforms.py` | ~100 | Matrix utilities |

---

## Key Data Structures

### Cache Entry (per-stroke)
```python
{
    'points': [Vector(x,y,z), ...],      # World-space coordinates
    'layer': "LayerName",                 # Source layer name
    'frame': 42,                          # Frame number
    'fill_triangles': [(0,1,2), ...]      # Triangle indices for fill rendering
}
```

### Object-Level Lock Data (JSON in `gp_obj["world_onion_locks"]`)
```python
{
    "42": {                                  # Frame number as string key
        "world_locked": true,                # Is this frame world-locked?
        "anchor_world": [x, y, z],           # Stroke center world position (pivot, stays fixed)
        "anchor_local_offset": [x, y, z],    # Offset from GP origin in local coords
        "original_parent_inverse": [[...]],  # Original matrix_parent_inverse for restore
        "matrix_local": [[...]]              # GP local matrix at lock time
    }
}
```

### Anchor Data (JSON in `gp_obj["world_onion_anchors"]`) - for cursor workflow
```python
{
    "LayerName": {
        "42": {                            # Frame number as string key
            "pos": [x, y, z],              # User-set anchor (from cursor)
            "cam_dir": [x, y, z],          # Camera direction at set time
        }
    }
}
```

### Global State (module-level in handlers.py)
```python
_last_keyframe_set = set()       # Track (layer, frame) tuples for change detection
_last_active_layer_name = None   # Detect layer selection changes
_last_active_gp = None           # Detect active object changes
_in_depsgraph_handler = False    # Prevent recursive handler calls
```

---

## Core Algorithms

### 1. World-Lock via Pivot-Based Billboard Rotation

**Problem:** Camera-parented GP rotates with camera. Strokes should stay at world position AND face camera (billboard effect). However, strokes are typically offset from the GP origin - naive rotation around the origin causes strokes to swing instead of staying planted.

**Solution:** Use pivot-based rotation around the stroke center (anchor), not the GP origin:

```
WRONG (rotate around GP origin):     CORRECT (rotate around anchor):

  Camera rotates →   Strokes swing      Camera rotates →   Strokes stay planted
       ↻                  ↻                  ↻
    [Origin]●────○ Strokes             [Origin]●────● Anchor (fixed)
                  ↘ (moves!)                ↘              │
                                       (moves to          └── Strokes stay here
                                        compensate)
```

**Pivot Rotation Formula:**
```
gp_position = anchor_world - R_desired @ anchor_local_offset
```

Where:
- `anchor_world` = stroke center in world space (stays fixed)
- `anchor_local_offset` = offset from GP origin to anchor in local coords
- `R_desired = camera_rot @ local_rot` (billboard with preserved orientation)

```python
# In handlers.py: apply_world_lock_from_stored()

def apply_world_lock_from_stored(gp_obj, anchor_world, anchor_local_offset, matrix_local_stored):
    parent = gp_obj.parent
    if parent is None:
        gp_obj.location = Vector(anchor_world)
        return

    # Billboard rotation: camera rotation with local orientation preserved
    camera_rot = parent.matrix_world.to_3x3()
    local_rot = matrix_local_stored.to_3x3()
    R_desired = camera_rot @ local_rot

    # Pivot-based rotation: compute GP position to keep anchor fixed
    gp_position = Vector(anchor_world) - R_desired @ Vector(anchor_local_offset)

    # Compose desired world matrix
    desired_world = Matrix.Translation(gp_position) @ R_desired.to_4x4()

    # Solve for MPI: world = parent @ mpi @ local
    # mpi = parent⁻¹ @ desired_world @ local⁻¹
    new_mpi = parent.matrix_world.inverted() @ desired_world @ matrix_local_stored.inverted()
    gp_obj.matrix_parent_inverse = new_mpi
```

**Key insights:**
1. Strokes face camera (billboard) via `R_desired = camera_rot @ local_rot`
2. Strokes stay planted via pivot rotation around `anchor_world`
3. No layer transforms modified - GP effects work perfectly
4. `matrix_local` preserves the GP's original local orientation

### 2. Anchor Calculation from Strokes

```python
# In anchors.py: calculate_anchor_from_strokes()

# Center XY, lowest Z - gives "bottom center" reference point
sum_x, sum_y = 0, 0
min_z = float('inf')

for world_point in all_stroke_points:
    sum_x += world_point.x
    sum_y += world_point.y
    min_z = min(min_z, world_point.z)

anchor = Vector((sum_x / count, sum_y / count, min_z))
```

### 3. Falloff Opacity

```python
# Linear fade based on frame distance
max_distance = max(frames_before, frames_after)
falloff_factor = 1.0 - (abs(frame_offset) / max_distance) * falloff_strength
final_alpha = base_opacity * max(0.1, falloff_factor)
```

### 4. World-Space Stroke Extraction

```python
# In cache.py: extract_strokes_at_current_frame()

for layer in gp.layers:
    layer_matrix = get_layer_transform(layer)       # T @ R @ S (identity when locked)
    full_matrix = world_matrix @ layer_matrix       # world_matrix includes MPI

    for point in stroke.points:
        world_point = full_matrix @ Vector(point.position)
```

**Note:** With object-level locking, `layer_matrix` is always identity (layer transforms untouched). The world position comes from `gp_obj.matrix_world` which incorporates `matrix_parent_inverse`.

---

## Edge Cases Handled

| Edge Case | Solution | Location |
|-----------|----------|----------|
| No active GP object | Return `CANCELLED`, hide panels, skip handlers | operators.py, ui.py, handlers.py |
| No keyframes on layer | Skip layer, use fallback frame | anchors.py, cache.py |
| GP not parented to camera | Set `gp_obj.location` directly instead of MPI | handlers.py |
| Recursive depsgraph handler | Guard with `_in_depsgraph_handler` flag | handlers.py |
| Legacy layer-level locks | Auto-migrate to object-level on first access | anchors.py |
| Legacy anchor format | Auto-convert `[x,y,z]` to `{"pos":[x,y,z]}` | anchors.py |
| Missing camera | Use default direction `(0,0,1)` | anchors.py |
| Degenerate polygon triangulation | try/except returns empty triangles | cache.py |
| Parent chain changes | Detect via depsgraph, recalc all world-locks | handlers.py |
| Layer selection change | Snap cursor to new layer's anchor | handlers.py |
| Keyframe moved in timeline | Migrate anchor data to new frame | handlers.py |
| Unlock restoration | Store/restore `original_parent_inverse` | anchors.py, operators.py |

---

## Handler Flow

### on_frame_change (frame_change_post)
```
1. For ALL GP objects with world-locks:
   └── apply_object_world_lock_for_frame()
       ├── Find which locked frame is visible at current frame
       ├── If locked: apply_object_world_lock(lock_position)
       └── If unlocked: reset_object_world_lock(original_mpi)

2. If anchor system enabled:
   ├── Get/calculate anchor for current frame
   ├── Move cursor to anchor position
   └── Align canvas grid to cursor

3. Cache current frame's strokes

4. Trigger viewport redraw
```

### on_depsgraph_update (depsgraph_update_post)
```
1. Guard against recursion (_in_depsgraph_handler)

2. Detect active GP change → clear cache

3. Detect parent chain changes:
   ├── Re-apply matrix_parent_inverse for ALL locked objects
   └── Clear cache only if active GP's parent changed

4. Anchor system features:
   ├── Detect keyframe moves → migrate anchor data
   ├── Detect new keyframes → capture cursor as anchor
   ├── Detect layer selection change → snap cursor
   └── If world_lock_inherit enabled → auto-lock new keyframes
```

---

## GPU Rendering Pipeline

### Viewport Drawing (drawing.py)
```
draw_onion_callback()
├── Determine frames to show (FRAMES or KEYFRAMES mode)
├── Set GPU state (ALPHA blend, LESS_EQUAL depth)
├── For each frame:
│   ├── Calculate falloff alpha
│   ├── Draw fills (TRI_FAN, UNIFORM_COLOR shader)
│   └── Draw strokes (LINE_STRIP, POLYLINE_UNIFORM_COLOR shader)
└── Reset GPU state
```

### Timeline Drawing (timeline_drawing.py)
```
draw_timeline_callback()
├── Find active GP and layer
├── Calculate lane Y position (handles UI scale, channels)
├── Get world-locked frame spans (object-level, via get_all_locked_frames())
├── Convert frame → view X coordinates
└── Draw rectangles at lane position
```

---

## Extension Points

### Adding a New Setting
1. Add property to `WorldOnionSettings` in `settings.py`
2. Add update callback if needed (use `update_setting` for redraw-only)
3. Add UI in appropriate panel in `ui.py`
4. Use `settings.your_property` where needed

### Adding a New Operator
1. Create class in `operators.py` inheriting `bpy.types.Operator`
2. Add to `classes` list in `operators.py`
3. Add button in `ui.py`
4. Register in `__init__.py` classes list

### Adding New Anchor Data
1. Modify dict structure in `get_anchors()` / `set_anchor_for_frame()` in `anchors.py`
2. Add migration logic for legacy data
3. Update any handlers that read/write anchor data

---

## Common Patterns

### Property Update Callbacks
```python
def update_setting(self, context):
    """Redraw without cache invalidation."""
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()

def update_filter(self, context):
    """Invalidate cache (filters affect visible strokes)."""
    _cache.clear()
    update_setting(self, context)
```

### Custom Property Persistence
```python
# Store
gp_obj["world_onion_anchors"] = json.dumps(data)

# Load with fallback
if "world_onion_anchors" in gp_obj:
    data = json.loads(gp_obj["world_onion_anchors"])
else:
    data = {}
```

### Handler Registration
```python
@persistent  # Survives file load
def my_handler(scene):
    pass

# In register():
if my_handler not in bpy.app.handlers.frame_change_post:
    bpy.app.handlers.frame_change_post.append(my_handler)

# In unregister():
if my_handler in bpy.app.handlers.frame_change_post:
    bpy.app.handlers.frame_change_post.remove(my_handler)
```

---

## Gotchas and Pitfalls

### 1. Depsgraph Handler Recursion
Setting custom properties triggers depsgraph update. Always guard:
```python
_in_depsgraph_handler = False

def on_depsgraph_update(scene, depsgraph):
    global _in_depsgraph_handler
    if _in_depsgraph_handler:
        return
    _in_depsgraph_handler = True
    try:
        # ... do work ...
    finally:
        _in_depsgraph_handler = False
```

### 2. Keyframe Detection Timing
Keyframe changes must be detected by comparing sets, not by checking update flags:
```python
current_keyframes = {(layer.name, frame.frame_number) for layer in gp.layers for frame in layer.frames}
added = current_keyframes - _last_keyframe_set
removed = _last_keyframe_set - current_keyframes
_last_keyframe_set = current_keyframes
```

### 3. Layer Index vs Visual Order
GP layers are stored bottom-to-top but displayed top-to-bottom in Dopesheet:
```python
visual_index = len(visible_layers) - 1 - layer_list_index
```

### 4. Anchor System (Cursor Workflow)
Anchors are for **cursor positioning only** (where the user wants to draw). Lock positions are stored separately in `world_onion_locks` and represent the GP object's world position when locked.

### 5. Cache Invalidation
Cache must be cleared when:
- Active GP object changes
- Parent chain changes (for active GP only)
- Layer filters change
- (NOT on every frame change - that defeats the purpose)

### 6. World-Lock Applies to ALL Objects
`apply_object_world_lock_for_frame()` iterates ALL GP objects with world-locked keyframes, not just the active one. This is intentional - multiple GP objects can be locked simultaneously.

### 7. Matrix Parent Inverse Persistence
The `matrix_parent_inverse` is NOT stored in custom properties - it's part of the object data. The lock system stores `lock_position` and `original_parent_inverse` in custom properties, then recomputes `matrix_parent_inverse` on every frame change. This handles undo/redo and ensures billboard rotation stays current.

---

## Development Workflow

### Hot Reload
Use the "Reload Addon" button in Dev panel (or call `WONION_OT_reload_addon`):
1. Saves current settings
2. Unregisters addon
3. Reloads all modules in dependency order
4. Re-registers with fresh classes
5. Restores settings

### Debugging
- Add print statements (visible in Blender's terminal/console)
- Cache stats visible in Cache panel
- Anchor count visible in Anchors panel
- Use Blender's Python console for inspection:
  ```python
  import json
  gp = bpy.context.active_object
  print(json.dumps(json.loads(gp["world_onion_anchors"]), indent=2))
  ```

### Testing World-Lock
1. Create GP object parented to camera
2. Draw strokes on frame 1
3. Toggle world-lock (click lock icon or use operator)
4. Move to frame 2, rotate camera
5. Strokes should stay in world position

---

## Module Dependencies

```
__init__.py
├── settings.py (no deps)
├── transforms.py (no deps)
├── anchors.py → transforms
├── cache.py → transforms, anchors
├── handlers.py → cache, anchors, transforms, drawing
├── drawing.py → cache
├── timeline_drawing.py → anchors
├── operators.py → cache, anchors, handlers, transforms
└── ui.py → operators, handlers
```

When reloading, order matters: reload dependencies before dependents.

---

## Performance Notes

- **Cache limit:** 2000 frames max, oldest evicted
- **GPU batching:** All strokes drawn in batched `LINE_STRIP` calls
- **Handler guards:** Early exits prevent expensive recalculations
- **MPI recalculation:** `matrix_parent_inverse` recomputed each frame for billboard effect

---

## API Patterns Used

| Pattern | Example |
|---------|---------|
| Custom properties | `gp_obj["world_onion_locks"]`, `gp_obj["world_onion_anchors"]` |
| GPU shaders | `gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')` |
| Batch rendering | `batch_for_shader(shader, 'LINE_STRIP', {"pos": coords})` |
| Persistent handlers | `@persistent` decorator |
| Matrix parent inverse | `gp_obj.matrix_parent_inverse = parent.inverted() @ desired_world` |
| GP data access | `layer.frames[0].drawing.attributes['position'].data` |

---

## Summary

This addon is a **production-quality specialized tool** for GP storyboarding on animated cameras. Key architectural decisions:

1. **Modular design** - Clear separation of concerns
2. **JSON persistence** - Avoids baking data into keyframes
3. **Dual handler system** - Frame change for rendering, depsgraph for state
4. **Global state tracking** - Enables change detection across handler calls
5. **GPU rendering** - Modern shader-based drawing
6. **Object-level locking** - Uses `matrix_parent_inverse` instead of layer transforms

The world-lock algorithm uses **pivot-based rotation** around the stroke center (anchor) with `matrix_parent_inverse` to maintain world-space positioning while providing true billboard rendering. The formula `gp_position = anchor_world - R_desired @ anchor_local_offset` keeps strokes planted while the GP origin moves to compensate for camera rotation. Layer transforms remain untouched, ensuring full GP effects compatibility. The anchor system provides separate cursor positioning for the drawing workflow.
