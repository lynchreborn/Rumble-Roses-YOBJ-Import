# Rumble Roses PS2 YOBJ - Blender Import Plugin

A Blender add-on that imports **Rumble Roses (PS2)** `.yobj` models: fighters, hair, clothing, accessories and whole arenas, with skeletons, skin weights, UVs, normals and materials.

The goal is simple: get the game's models into Blender for visual art, renders and study. It is **import only**. There is no exporter (someone is welcome to build one on top of this code).

**Requires:** Blender 5.2 or newer  |  **Version:** 1.0.0

> This project contains no game assets. You need your own files extracted from your own copy of the game.

---

## Features

**Both modes**
- Reads `.yobj` files: meshes, bones, skin weights, UVs, custom normals
- Builds armatures and merges accessory skeletons (hair, clothes) into the body rig
- Builds materials from the game's textures (`.tga`), reusing identical materials instead of making `.001` copies
- Batch import: select many files at once
- Seam weight check, import log and optional debug extras (Debug Mode)

**Character Mode**
- Hair strands and lashes with alpha cutouts
- Sheen / shine layer (the PS2's normal-based reflection) rebuilt as a MatCap
- Cloth alpha used as a specular mask
- Rigs are named after the folder, so several fighters can live in one scene without sharing bones

**Arena Mode**
- Reads `C351.dat` for the game's mirrored copies, alpha blending and additive meshes
- Tiling textures set to repeat, transparent textures detected from their alpha
- Optional **Place Crowd**: fills the ringside seats using `526C.dat` / `526F.dat`, retextured and facing the ring

---

## Install

1. Download `rumble_roses_yobj_importer-1.0.0.zip` from the Releases page.
2. In Blender: **Edit > Preferences > Get Extensions > (top-right menu) > Install from Disk**, then pick the zip.
3. Enable **Rumble Roses YOBJ Importer**.

Prefer not to install? Open `__init__.py` in Blender's Text Editor and press **Run Script**.

---

## Use

1. **File > Import > Rumble Roses YOBJ (.yobj)**
2. Select one or more `.yobj` files. To bring in a whole fighter, select every file in that fighter's folder.
3. Leave **Mode** on *Auto Detect* (or choose Character / Arena yourself).
4. Click **Import**.

### Expected folder layout

```
1810/                  <- one fighter (or arena) per folder, as exported from the game
  0002.yobj            <- body
  4E20.yobj ...        <- hair, clothes, accessories
  textures/            <- your .tga files (found automatically)
```

- **Textures:** put the `.tga` files in a folder named `textures` beside the `.yobj` files, or set **Texture Folder** by hand.
- **Arena data files** (`C351.dat`, `526C.dat`, `526F.dat`): keep them in the same folder as the `.yobj` files, or one folder up.

---

## Options

| Option | What it does |
|---|---|
| **Mode** | *Auto Detect* (default), *Character*, or *Arena* |
| **Materials / Texture Folder** | Build materials and say where the `.tga` files are |
| **Import into New Collection** | One collection per file, or use the active collection |
| **Axis / Scale** | Y-up to Z-up rotation and overall scale |
| **Join Parts** | One mesh per file instead of one per part (off by default) |

**Expert Settings** (collapsed panel): mirror fix, bone direction, skeleton merging and tolerance, material reuse, sheen layer and strength (default 0.25), hair options, cloth alpha as specular (on by default), and the arena options (C351 mirroring and transparency, dynamic collections, Place Crowd).

**Debug Mode** (tick the box in the panel header): log detail, debug attributes and heat-maps, bone empties, seam check, a `_import_log.txt` next to the first `.yobj`, and the sheen experiments (lookup, alpha scale, blend, flip V). Attach that log when reporting a bug.

---

## Tips

- **Hair or clothes attached to the wrong rig?** Skeletons only merge between files from the *same folder*. Keep each fighter's files together.
- **Textures look stretched in an arena?** Make sure Materials is on; tiling surfaces are switched to repeat automatically.
- **Auto Detect guessed wrong?** Pick Character or Arena Mode manually. This can happen on a lone arena weapon file.
- **Crowd facing the wrong way?** Use *Extra Turn* (try 180) under Place Crowd.

---

## Known limitations

- Import only; no export.
- No animations or poses. Crowd and props import static.
- Ring props, weapons, animated lights and stand (upper-level) crowd spawn at the origin; the game positions them from code and data that are not decoded yet.
- Cameras and lights from `C355` / `EAC4` / `EAC5` are not converted.
- Cloth sheen is a close approximation of the PS2's look, not an exact match.
- Shader IDs 5 and 7 are not decoded.

---

## Credits and legal

Reverse-engineered from research on the PS2 file formats. "RROExporter" by Mike from 2007 was used as reference for yobj vs obj comparison to get a baseline. No code was stolen or reverse-engineered from the software itself.

Rumble Roses is a trademark of Konami. This is an unofficial fan project, not affiliated with or endorsed by Konami, and it includes no game data.

**License:** GPL-3.0-or-later 
