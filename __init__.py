"""import_yobj.py - Rumble Roses (PS2) .yobj native Blender importer.  Version 1.0.0 (Blender 5.2)

Install as an add-on (Edit > Preferences > Add-ons > Install from Disk) or just open
it in Blender's Text Editor and hit Run Script. Then: File > Import > Rumble Roses YOBJ.

Layout of this file (top to bottom):
    1. PARSE LAYER   pure Python, zero bpy. Reads bytes, returns plain data.
                     You can run it without Blender:  python import_yobj.py some.yobj
    2. DEBUG HELPERS pure Python too (log formatting, seam check)
    3. BUILD LAYER   bpy. Armature, meshes, weights, materials, debug attributes.
    4. OPERATOR/UI   the File > Import menu entry and its options.

Spec: YOBJ_Reverse_Engineering_Spec.md (section numbers in comments point at it).

1.0.0 (release):
    * Mode dropdown has a new default, "Auto Detect" (looks for C351/526x data files, bone counts
      and texture-name patterns). Character Mode / Arena Mode are still there as manual overrides.
    * The armature is named after the folder the .yobj files sit in (body_armature_1810), and
      a file only merges into an armature from the SAME folder, so two fighters in one scene
      keep their own rigs.
    * The body file (0002) is preferred as the base skeleton of a batch.
    * New "Join Parts" option (off by default): one mesh object per file instead of one per part.
    * Defaults: Cloth Alpha as Specular on, Sheen Strength 0.25.
    * Sheen Lookup / Alpha Scale / Blend / Flip V moved into Debug Mode; Bone Index Base and
      4-Influence Lane Order are hidden (still settable from Python).
    * Vertex weights are assigned in bulk (one call per bone + weight instead of per vertex).
    * "Imported N files" no longer undercounts when a file fails to parse.
"""

bl_info = {
    "name": "Rumble Roses YOBJ Importer",
    "author": "LyonessDen/Claude",
    "version": (1, 0, 0),
    "blender": (5, 2, 0),
    "location": "File > Import > Rumble Roses YOBJ (.yobj)",
    "description": "Native importer for PS2 Rumble Roses .yobj models (armature, meshes, skin weights)",
    "category": "Import-Export",
}

import hashlib
import math
import os
import re
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

# =============================================================================
# 1. PARSE LAYER  (no bpy in here, on purpose)
# =============================================================================

NONE_U32 = 0xFFFFFFFF
PART_SIZE = 0x40       # one record in the part table
GROUP_SIZE = 0x20      # one record in a part's group table
BONE_SIZE = 80
MATERIAL_SIZE = 208
VU_BASE = 0x280        # weight tag imm = 0x280 + local index of first vertex it covers (spec 4.3)
SUM_TOL = 1e-4         # "weights sum to 1" tolerance from the spec
DROP_TOL = 1e-6        # weights smaller than this get thrown away when normalizing

# Lane permutations for 4-influence groups (spec 4.7a). Entry i says "weight slot i
# goes with bone_indices[perm[i]]". IDENTITY is what the spec calls confirmed; the
# rest are the odd winners from the permutation test, here for A/B testing only.
LANE_ORDERS = {
    'IDENTITY': (0, 1, 2, 3),
    'ALT_0321': (0, 3, 2, 1),   # best on 0002 (4-influence)
    'ALT_0213': (0, 2, 1, 3),   # best on 271B (4-influence)
    'REVERSE': (3, 2, 1, 0),
}


def _u32(d, o):
    return struct.unpack_from('<I', d, o)[0]


# ---- tiny matrix helpers (plain lists, so no mathutils needed here) ---------

def _mul3(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _mul4(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def _rot(axis, a):
    c, s = math.cos(a), math.sin(a)
    if axis == 0:
        return [[1, 0, 0], [0, c, -s], [0, s, c]]
    if axis == 1:
        return [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]]


def _euler_zyx(rot):
    # intrinsic Z then Y then X, column vectors: R = Rz * Ry * Rx (spec 2.1)
    return _mul3(_mul3(_rot(2, rot[2]), _rot(1, rot[1])), _rot(0, rot[0]))


def _mat4(r, t):
    return [[r[0][0], r[0][1], r[0][2], t[0]],
            [r[1][0], r[1][1], r[1][2], t[1]],
            [r[2][0], r[2][1], r[2][2], t[2]],
            [0.0, 0.0, 0.0, 1.0]]


_S = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]   # the Y reflection (old OBJ exporter convention)
# Y flip + X flip together = diag(-1,-1,1). Two reflections cancel into a plain 180 degree
# turn about Z, so this one is a proper rotation (det +1): no mirroring, no winding change.
_FXS = [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def _conv(mirror_x):
    return _FXS if mirror_x else _S


def world_pos(model, p):
    """File (x,y,z) -> importer space. Every consumer goes through here so mesh, bones,
    seam check and heat maps can never disagree about which flip is active."""
    x, y, z = p
    return (-x, -y, z) if model.mirror_x else (x, -y, z)


def world_normal(model, n):
    """File normal -> importer space. Verified on every sample: the file's triangle winding
    agrees with its stored normals, so a proper rotation keeps winding as-is and just
    rotates the normal. The old Y-flip-only mode needs the extra sign dance."""
    nx, ny, nz = n
    return (-nx, -ny, nz) if model.mirror_x else (-nx, ny, -nz)


# ---- data containers ---------------------------------------------------------

@dataclass
class BoneData:
    index: int
    name: str
    parent: int                 # -1 = root
    local_pos: tuple
    local_rot: tuple
    g_fix: list = None          # 4x4, un-reflected (det +1), S * G_raw * S
    head: tuple = (0, 0, 0)     # translation of g_fix (same as yobj_bones global_pos)
    children: list = field(default_factory=list)


@dataclass
class GroupData:
    count: int
    influence_count: int
    ptr_vertices: int
    ptr_normals: int
    bone_indices: list          # raw u32 values from the file, -1 where 0xFFFFFFFF


@dataclass
class MaterialData:
    slot: int
    name: str
    kd: tuple
    ks: tuple
    shader: int
    strips: list                # list of strips; strip = list of (vertex_index, u, v)
    raw_words: tuple = ()       # the whole 208-byte record as 52 u32, only used by the debug dump
    # Optional second texture layer (record bytes +0x30..+0x5F, all zero when absent).
    # layer2_kind (+0x50): 1 and 2 both turned out to be a sheen/reflection overlay whose UVs the
    # engine makes at runtime from the view-space normal (GS dump fit, R^2 ~0.98). Seen as:
    #   kind 1 = 4E84 clothing (light.tga, an env sphere), kind 2 = 4E20 hair (light3.tga) and bikes.
    # The UVs are NOT in the file, which is why there's no second UV set anywhere.
    layer2_kind: int = 0
    layer2_slot: int = -1       # texture slot of that layer (+0x58), -1 = no second layer
    layer2_name: str = ''
    layer2_shader: int = 0      # +0x54
    # What the PS2 GIF tag in front of this material's strips says each vertex carries
    gif_regs: tuple = ()        # distinct register lists, e.g. (('ST', 'RGBAQ', 'XYZF2'),)
    gif_prims: tuple = ()       # distinct PRIM words (tristrip + texture + alpha blend flags...)
    role: str = ''              # 'HAIR_STRAND' / 'HAIR_BODY' / 'LASH' / 'CLOTH', set by assign_roles() in the build layer


@dataclass
class VertexInfo:
    group: int
    infl: int
    raw_sum: float = 1.0        # sum of the raw row BEFORE we normalize (handy for debugging)
    notes: list = field(default_factory=list)


@dataclass
class PartData:
    index: int
    groups: list
    positions: list             # raw file (x, y, z), part-local order
    normals: list               # raw file (nx, ny, nz)
    materials: list
    total_verts: int
    tags: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    multi_vertex_idx: list = field(default_factory=list)   # local vertex index per weight row
    vinfo: list = field(default_factory=list)              # VertexInfo per vertex
    weights: list = field(default_factory=list)            # {bone_index: weight} per vertex, normalized
    # triangles: parallel lists, one entry per kept triangle
    tris: list = field(default_factory=list)               # (a, b, c) part-local vertex indices
    tri_uvs: list = field(default_factory=list)            # ((u,v),(u,v),(u,v)) raw file UVs
    tri_mat: list = field(default_factory=list)            # index into self.materials
    # One entry per UV set; each is a list parallel to `tris`. Always exactly one set today:
    # the GIF tags declare a single ST register per vertex and no spare bytes exist (see notes).
    uv_sets: list = field(default_factory=list)
    uv_set_count: int = 1                                  # how many ST registers the GIF tags declare
    issues: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    # Optional per-part bone palette: global_bone_id = palette[raw_bone_id]. None = indices are
    # already global ids. No palette has been found in any sample file (see notes), so this
    # stays None until someone decodes one.
    palette: list = None


@dataclass
class YobjModel:
    path: str
    model_name: str
    names: list
    bones: list
    parts: list
    bone_base: int
    lane_order: str
    mirror_x: bool = False


# ---- low level readers --------------------------------------------------------

def read_data_block(path):
    raw = Path(path).read_bytes()
    if raw[:4] != b'YOBJ':
        raise ValueError('%s is not a YOBJ file' % path)
    size = _u32(raw, 4)
    return raw[8:8 + size]       # every pointer in the file counts from here


def walk_unpack_chain(d, pos, max_rows):
    """Follow the chain of VIF UNPACK V4_32 blocks that holds the weight rows.

    Each slot is 16 bytes: three NOP words, then the tag word at slot+12. The tag
    is cmd(31:24) | num(23:16) | imm(15:0). Rows follow the slot, 16 bytes each.

    Returns (tags, rows). We stop early once we've got max_rows, because the
    material block sits right after the stream and we don't want to wander into it.
    """
    tags, rows = [], []
    while pos + 16 <= len(d) and len(rows) < max_rows:
        w = _u32(d, pos + 12)
        cmd = w >> 24
        # UNPACK is 0x60..0x7F, V4_32 has 0xC in the low nibble
        if (cmd & 0x60) != 0x60 or (cmd & 0x0F) != 0x0C:
            break
        num = (w >> 16) & 0xFF
        base = pos + 16
        if base + num * 16 > len(d):
            break
        tags.append({
            'word': w, 'cmd': cmd, 'num': num,
            'imm': w & 0xFFFF,
            'addr': w & 0x3FF,
            'masked': bool(cmd & 0x10),      # the "m" bit in the UNPACK command
            'first_row': len(rows),
            'offset': pos + 12,
        })
        for i in range(num):
            rows.append(struct.unpack_from('<4f', d, base + i * 16))
        pos = base + num * 16
    return tags, rows


def parse_bones(d, mirror_x=False):
    count = _u32(d, 0x14)
    table = _u32(d, 0x20)
    bones = []
    for i in range(count):
        o = table + i * BONE_SIZE
        parent = _u32(d, o + 0x30)
        bones.append(BoneData(
            index=i,
            name=d[o:o + 16].split(b'\0')[0].decode('ascii', 'replace'),
            parent=-1 if parent == NONE_U32 or parent >= count else parent,
            local_pos=struct.unpack_from('<3f', d, o + 0x10),
            local_rot=struct.unpack_from('<3f', d, o + 0x20),
        ))

    # world = parent_world @ local, walked recursively (spec 2.1), then conjugated by the
    # coordinate flip C:  G_fix = C * G_raw * C  (C = S normally, X-flip*S when mirror_x)
    C = _conv(mirror_x)
    raw = {}

    def solve(i):
        if i in raw:
            return raw[i]
        b = bones[i]
        local = _mat4(_euler_zyx(b.local_rot), b.local_pos)
        raw[i] = local if b.parent < 0 else _mul4(solve(b.parent), local)
        return raw[i]

    # recursion depth could bite on a really deep chain, so bump the limit a bit
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 4 * count + 1000))
    for b in bones:
        b.g_fix = _mul4(_mul4(C, solve(b.index)), C)
        b.head = (b.g_fix[0][3], b.g_fix[1][3], b.g_fix[2][3])
        if b.parent >= 0:
            bones[b.parent].children.append(b.index)
    return bones


GIF_REG_NAMES = {0: 'PRIM', 1: 'RGBAQ', 2: 'ST', 3: 'UV', 4: 'XYZF2', 5: 'XYZ2', 0xE: 'A+D', 0xF: 'NOP'}


def decode_gif_tag(d, entries_ptr, entry_count):
    """Each strip's entries are preceded by a 16-byte PS2 GIF tag that says which registers
    the GPU receives per vertex. Returns (register names, PRIM word or -1), or None when the
    16 bytes don't look like a GIF tag (NLOOP must equal the entry count; it does for all
    2761 strips in the sample files)."""
    pos = entries_ptr - 16
    if pos < 0:
        return None
    w0, w1, w2, w3 = struct.unpack_from('<4I', d, pos)
    if (w0 & 0x7FFF) != entry_count:
        return None
    nreg = (w1 >> 28) & 15 or 16
    regs64 = w2 | (w3 << 32)
    regs = tuple(GIF_REG_NAMES.get((regs64 >> (4 * i)) & 15, '?%d' % ((regs64 >> (4 * i)) & 15))
                 for i in range(nreg))
    prim = (w1 >> 15) & 0x7FF if (w1 >> 14) & 1 else -1
    return regs, prim


def parse_part(d, p, part_table, bone_count, names, bone_base, lane_order, palette=None):
    o = part_table + p * PART_SIZE
    group_count = _u32(d, o + 0x00)
    mat_count = _u32(d, o + 0x04)
    group_ptr = _u32(d, o + 0x08)
    mat_ptr = _u32(d, o + 0x0C)
    stream_ptr = _u32(d, o + 0x1C)
    total_verts = _u32(d, o + 0x28)

    groups, positions, normals = [], [], []
    for g in range(group_count):
        go = group_ptr + g * GROUP_SIZE
        cnt, infl, ptr_v, ptr_n = struct.unpack_from('<4I', d, go)
        bones = [-1 if b == NONE_U32 else b for b in struct.unpack_from('<4I', d, go + 0x10)]
        groups.append(GroupData(cnt, infl, ptr_v, ptr_n, bones))
        for v in range(cnt):
            x, y, z, _w = struct.unpack_from('<4f', d, ptr_v + v * 16)
            positions.append((x, y, z))
            nx, ny, nz, _w = struct.unpack_from('<4f', d, ptr_n + v * 16)
            normals.append((nx, ny, nz))

    part = PartData(index=p, groups=groups, positions=positions, normals=normals,
                    materials=[], total_verts=total_verts)

    # ---- materials + strips -> triangles (spec 1.5, 3.2) ----
    for k in range(mat_count):
        mo = mat_ptr + k * MATERIAL_SIZE
        kd = struct.unpack_from('<3f', d, mo)
        ks = struct.unpack_from('<3f', d, mo + 0x10)
        shader = _u32(d, mo + 0x24)
        slot = _u32(d, mo + 0x28)
        n_strips, p_strips = struct.unpack_from('<2I', d, mo + 0xC4)
        strips = []
        gif_seen = set()
        for s in range(n_strips):
            n_ent, p_ent = struct.unpack_from('<2I', d, p_strips + s * 16 + 8)
            g = decode_gif_tag(d, p_ent, n_ent)
            if g:
                gif_seen.add(g)
            entries = []
            for e in range(n_ent):
                eo = p_ent + e * 32
                u, v = struct.unpack_from('<2f', d, eo)
                entries.append((_u32(d, eo + 12), u, v))
            strips.append(entries)
        raw = struct.unpack_from('<52I', d, mo)
        l2_kind, l2_shader, l2_slot = 0, 0, -1
        if any(raw[0x30 // 4:0x60 // 4]):
            l2_kind, l2_shader, l2_slot = raw[0x50 // 4], raw[0x54 // 4], raw[0x58 // 4]
        part.materials.append(MaterialData(
            slot=slot, name=names[slot] if slot < len(names) else 'slot%d' % slot,
            kd=kd, ks=ks, shader=shader, strips=strips, raw_words=raw,
            layer2_kind=l2_kind, layer2_shader=l2_shader,
            layer2_slot=l2_slot if l2_slot < len(names) else -1,
            layer2_name=names[l2_slot] if 0 <= l2_slot < len(names) else '',
            gif_regs=tuple(sorted({g[0] for g in gif_seen})),
            gif_prims=tuple(sorted({g[1] for g in gif_seen}))))

    nverts = len(positions)
    degenerate = out_of_range = 0
    for mi, mat in enumerate(part.materials):
        for strip in mat.strips:
            for i in range(len(strip) - 2):
                a, b, c = i, i + 1, i + 2
                if i % 2 == 1:          # odd triangles flip to keep winding consistent
                    b, c = c, b
                ids = (strip[a][0], strip[b][0], strip[c][0])
                if max(ids) >= nverts:
                    out_of_range += 1
                    continue
                if len(set(ids)) < 3:   # degenerate, strips use these as "restarts"
                    degenerate += 1
                    continue
                part.tris.append(ids)
                part.tri_uvs.append(tuple((strip[k][1], strip[k][2]) for k in (a, b, c)))
                part.tri_mat.append(mi)
    # UV sets: the entry layout only has one ST pair per vertex, so one set. If a GIF tag ever
    # declares more ST registers than that, say so instead of silently losing a set.
    part.uv_sets = [part.tri_uvs]
    st_counts = [r.count('ST') for m_ in part.materials for r in m_.gif_regs]
    part.uv_set_count = max(st_counts) if st_counts else 1
    if part.uv_set_count > len(part.uv_sets):
        part.issues.append('GIF tags declare %d ST registers but entries only carry %d UV pair(s)'
                           % (part.uv_set_count, len(part.uv_sets)))
    part.stats['degenerate_tris'] = degenerate
    part.stats['bad_index_tris'] = out_of_range
    if out_of_range:
        part.issues.append('%d triangles referenced a vertex index past the end' % out_of_range)

    # ---- weight stream (spec 4) ----
    last = groups[-1] if groups else None
    normals_end = last.ptr_normals + last.count * 16 if last else stream_ptr
    if normals_end != stream_ptr:
        part.issues.append('normals end 0x%X != stream ptr 0x%X (using normals end)' % (normals_end, stream_ptr))
    expected = sum(g.count for g in groups if g.influence_count > 1)
    if expected:
        part.tags, part.rows = walk_unpack_chain(d, normals_end, expected)
    if len(part.rows) != expected:
        part.issues.append('weight rows %d != expected %d' % (len(part.rows), expected))
    if sum(g.count for g in groups) != total_verts:
        part.issues.append('group vertex sum %d != part total %d' % (sum(g.count for g in groups), total_verts))
    if any(t['masked'] for t in part.tags):
        part.issues.append('MASKED UNPACK tag seen (cmd bit 4). Rows are read flat, which is only right '
                           'if there is no cycle/fill-write setup. Check this part by hand.')

    part.palette = palette
    resolve_weights(part, bone_count, bone_base, lane_order)

    # Check the spec's imm rule: imm = 0x280 + local index of first vertex a tag covers.
    for t in part.tags:
        if t['first_row'] < len(part.multi_vertex_idx):
            t['expected_first_vertex'] = part.multi_vertex_idx[t['first_row']]
        else:
            t['expected_first_vertex'] = None
        t['imm_ok'] = (t['addr'] - VU_BASE) == t['expected_first_vertex']
    return part


def resolve_weights(part, bone_count, bone_base, lane_order):
    """Turn groups + rows into one {bone_index: weight} dict per vertex.

    Bone indices are 0-based table indices after subtracting bone_base. Weights are
    normalized so every vertex sums to exactly 1.0 (within float rounding).
    """
    perm = LANE_ORDERS[lane_order]
    palette = part.palette

    def map_bone(raw_idx):
        """raw file value -> 0-based global bone index, or None if it doesn't land anywhere."""
        if raw_idx < 0:
            return None
        if palette is not None:               # palette hook: global = palette[raw]
            b = palette[raw_idx] if raw_idx < len(palette) else -1
        else:
            b = raw_idx - bone_base
        return b if 0 <= b < bone_count else None
    part.vinfo, part.weights, part.multi_vertex_idx = [], [], []
    cursor = 0
    unmapped = used_perm = fallback = 0
    local = 0
    for gi, g in enumerate(part.groups):
        infl = g.influence_count
        for _ in range(g.count):
            vi = VertexInfo(group=gi, infl=infl)
            if infl <= 1:
                row = (1.0, 0.0, 0.0, 0.0)
            else:
                part.multi_vertex_idx.append(local)
                if cursor < len(part.rows):
                    row = part.rows[cursor]
                    cursor += 1
                else:
                    row = None
                    vi.notes.append('missing weight row')

            w = {}
            if row is not None:
                vi.raw_sum = sum(x for x in row if x == x)
                lanes = perm if infl == 4 else (0, 1, 2, 3)
                if infl == 4 and lane_order != 'IDENTITY':
                    used_perm += 1
                for slot in range(4):
                    val = row[slot]
                    if val == 0.0:
                        continue
                    if val < 0.0 or val != val:
                        vi.notes.append('bad weight slot %d = %r' % (slot, val))
                        continue
                    if slot >= infl:
                        vi.notes.append('weight in unused slot %d' % slot)
                    raw_idx = g.bone_indices[lanes[slot]]
                    b = map_bone(raw_idx)
                    if b is None:
                        vi.notes.append('slot %d -> unmapped bone %d' % (slot, raw_idx))
                        unmapped += 1
                        continue
                    w[b] = w.get(b, 0.0) + val
                if abs(vi.raw_sum - 1.0) > SUM_TOL:
                    vi.notes.append('raw weight sum %.6f' % vi.raw_sum)

            # normalize; if nothing usable is left, glue the vertex to the group's first bone
            w = {b: x for b, x in w.items() if x > DROP_TOL}
            total = sum(w.values())
            if total <= 0.0:
                first = map_bone(g.bone_indices[0])
                first = first if first is not None else 0
                w = {first: 1.0}
                vi.notes.append('no usable weights, fell back to bone %d' % first)
                fallback += 1
            else:
                w = {b: x / total for b, x in w.items()}
                heavy = max(w, key=w.get)
                w[heavy] += 1.0 - sum(w.values())     # push float dust onto the biggest one
            part.vinfo.append(vi)
            part.weights.append(w)
            local += 1

    part.stats['unmapped_slots'] = unmapped
    part.stats['fallback_verts'] = fallback
    part.stats['permuted_verts'] = used_perm


def parse_yobj(path, bone_base=1, lane_order='IDENTITY', mirror_x=False, palettes=None):
    """Main entry point of the parse layer. Returns a YobjModel."""
    d = read_data_block(path)
    part_count = _u32(d, 0x10)
    bone_count = _u32(d, 0x14)
    name_count = _u32(d, 0x18)
    part_table = _u32(d, 0x1C)
    name_table = _u32(d, 0x24)
    name_ptr = _u32(d, 0x28)

    model_name = d[name_ptr:d.index(b'\0', name_ptr)].decode('ascii', 'replace')
    names = [d[name_table + i * 16:name_table + i * 16 + 16].split(b'\0')[0].decode('ascii', 'replace')
             for i in range(name_count)]
    bones = parse_bones(d, mirror_x)
    palettes = palettes or {}
    parts = [parse_part(d, p, part_table, bone_count, names, bone_base, lane_order, palettes.get(p))
             for p in range(part_count)]
    return YobjModel(path=str(path), model_name=model_name, names=names, bones=bones,
                     parts=parts, bone_base=bone_base, lane_order=lane_order, mirror_x=mirror_x)


def detect_texture_dir(path):
    """If a folder called 'textures' (any case) sits beside the given .yobj file or folder, return it."""
    if not path:
        return ''
    base = path if os.path.isdir(path) else os.path.dirname(path)
    if not base or not os.path.isdir(base):
        return ''
    for fn in os.listdir(base):
        full = os.path.join(base, fn)
        if fn.lower() == 'textures' and os.path.isdir(full):
            return full
    return ''


def find_texture(name, search_dirs, exts=('.tga', '.png', '.dds', '.bmp')):
    """Case-insensitive hunt for <name>.<ext>. Returns a path or None."""
    wanted = {(name + e).lower() for e in exts}
    for folder in search_dirs:
        if not folder or not os.path.isdir(folder):
            continue
        for fn in os.listdir(folder):
            if fn.lower() in wanted:
                return os.path.join(folder, fn)
    return None


# =============================================================================
# 2. DEBUG HELPERS  (still no bpy)
# =============================================================================

class ImportLog:
    """Prints to the console and remembers every line so we can dump it into a
    Blender text block afterwards. level: 0 quiet, 1 summary, 2 detailed, 3 verbose."""

    def __init__(self, level=1, echo=True):
        self.level = level
        self.echo = echo
        self.lines = []

    def __call__(self, msg, level=1):
        if level > self.level:
            return
        self.lines.append(msg)
        if self.echo:
            print(msg)


# Words of the 208-byte material record we know the meaning of (offsets in bytes):
# 0x00-0x1F colours, 0x24 shader, 0x28 texture slot, 0xC4/0xC8 strip count/pointer.
# Anything else non-zero is "unknown" and gets listed in the debug dump, in case one of
# them turns out to be a UV scale/offset or a wrap mode.
_KNOWN_MATERIAL_WORDS = set(range(0x00, 0x20, 4)) | {0x24, 0x28, 0xC4, 0xC8} | set(range(0x30, 0x60, 4))


def material_uv_report(mat):
    """Plain-python numbers about one material's UV data, for the debug dump."""
    us = [e[1] for strip in mat.strips for e in strip]
    vs = [e[2] for strip in mat.strips for e in strip]
    out_of_range = sum(1 for u, v in zip(us, vs) if u < -0.001 or u > 1.001 or v < -0.001 or v > 1.001)

    def area(t):
        (a, b), (c, d), (e, f) = t
        return (c - a) * (f - b) - (e - a) * (d - b)

    # After the odd-triangle swap every triangle in a strip should have the same UV
    # orientation. A triangle that disagrees with its strip is a "twisted" UV face.
    twisted = tris = 0
    for strip in mat.strips:
        signs = []
        for i in range(len(strip) - 2):
            a, b, c = i, i + 1, i + 2
            if i % 2 == 1:
                b, c = c, b
            if len({strip[k][0] for k in (a, b, c)}) < 3:
                continue
            ar = area([(strip[k][1], strip[k][2]) for k in (a, b, c)])
            if abs(ar) > 1e-12:
                signs.append(ar > 0)
        tris += len(signs)
        if signs:
            majority = sum(signs) * 2 >= len(signs)
            twisted += sum(1 for x in signs if x != majority)
    unknown = []
    for i, w in enumerate(mat.raw_words):
        off = i * 4
        if w and off not in _KNOWN_MATERIAL_WORDS and off < 0xC0:
            unknown.append('+0x%02X=%08X' % (off, w))
    return {
        'entries': len(us), 'strips': len(mat.strips), 'uv_tris': tris, 'twisted': twisted,
        'u': (min(us), max(us)) if us else (0, 0), 'v': (min(vs), max(vs)) if vs else (0, 0),
        'out_of_range': out_of_range, 'unknown_words': unknown,
        'flag_word': mat.raw_words[0xC0 // 4] if mat.raw_words else 0,
    }


def log_model(model, log):
    """Per-part report: group counts, influence histogram, VIF tags, row counts, bone 0 usage."""
    log('[YOBJ] %s  model=%r  bones=%d  parts=%d  BONE_BASE=%d  lane_order=%s' % (
        Path(model.path).name, model.model_name, len(model.bones), len(model.parts),
        model.bone_base, model.lane_order))
    log('[YOBJ] texture names: %s' % ', '.join(model.names), 2)
    if len({b.name for b in model.bones}) != len(model.bones):
        log('[YOBJ] WARNING duplicate bone names in the file, Blender will rename some')

    for part in model.parts:
        hist = {}
        for g in part.groups:
            hist[g.influence_count] = hist.get(g.influence_count, 0) + g.count
        used = sorted({b for w in part.weights for b in w})
        bone0 = [(vi, w[0]) for vi, w in enumerate(part.weights) if 0 in w]
        bone0_heavy = [vi for vi, wgt in bone0 if wgt > 0.5]
        tri_count = len(part.tris)
        part.stats.update(bone0_verts=len(bone0), bone0_heavy=len(bone0_heavy))

        log('[YOBJ] part %02d: groups=%d verts=%d tris=%d (dropped %d degenerate) mats=%d' % (
            part.index, len(part.groups), len(part.positions), tri_count,
            part.stats['degenerate_tris'], len(part.materials)))
        log('         influence histogram (verts per influence count): %s' % (
            ', '.join('%d-bone: %d' % (k, v) for k, v in sorted(hist.items()))))
        log('         bone palette: %s' % ('custom, %d entries' % len(part.palette) if part.palette
                                          else 'none in file, raw indices used as global ids'), 2)
        log('         weight rows %d/%d   bones used: %d   weight on root(bone 0): %d verts (%d above 0.5)' % (
            len(part.rows), sum(g.count for g in part.groups if g.influence_count > 1),
            len(used), len(bone0), len(bone0_heavy)))
        for mi, mat in enumerate(part.materials):
            r = material_uv_report(mat)
            log('         material %d: slot=%d %r shader=%d kd=%s ks=%s strips=%d entries=%d' % (
                mi, mat.slot, mat.name, mat.shader, tuple(round(c, 3) for c in mat.kd),
                tuple(round(c, 3) for c in mat.ks), r['strips'], r['entries']), 2)
            log('            UV channels in file: %d   GIF registers per vertex: %s   PRIM words: %s' % (
                len(part.uv_sets), ' | '.join(','.join(g) for g in mat.gif_regs) or 'undecoded',
                ' '.join('0x%X' % x for x in mat.gif_prims) or 'none'), 2)
            log('            U[%.3f..%.3f] V[%.3f..%.3f] entries outside 0..1: %d' % (
                r['u'][0], r['u'][1], r['v'][0], r['v'][1], r['out_of_range']), 2)
            if mat.layer2_slot >= 0:
                log('            layer 2: kind=%d slot=%d %r shader=%d  (%s)' % (
                    mat.layer2_kind, mat.layer2_slot, mat.layer2_name, mat.layer2_shader,
                    'sheen overlay, matcap UVs from normal' if mat.layer2_kind == 1 else
                    'sheen/reflection overlay, matcap UVs from normal' if mat.layer2_kind == 2 else
                    'unknown kind'), 2)
            log('            UV orientation: %d of %d triangles disagree with their strip   '
                'record flag word +0xC0=%08X   unknown non-zero words: %s' % (
                    r['twisted'], r['uv_tris'], r['flag_word'],
                    ' '.join(r['unknown_words']) or 'none'), 2)
        for t in part.tags:
            log('         VIF tag 0x%08X cmd=0x%02X masked=%s num=%d imm=0x%03X first_vertex(from imm)=%d '
                'first_vertex(by count)=%s %s' % (
                    t['word'], t['cmd'], t['masked'], t['num'], t['imm'], t['addr'] - VU_BASE,
                    t['expected_first_vertex'], 'OK' if t['imm_ok'] else 'MISMATCH'), 2)
        for gi, g in enumerate(part.groups):
            log('         group %02d: count=%d infl=%d bones(raw)=%s' % (
                gi, g.count, g.influence_count, g.bone_indices), 2)
        if part.stats['unmapped_slots'] or part.stats['fallback_verts']:
            log('         !! unmapped weight slots: %d   fallback verts: %d' % (
                part.stats['unmapped_slots'], part.stats['fallback_verts']))
        if part.stats['permuted_verts']:
            log('         lane order %s applied to %d 4-influence verts' % (
                model.lane_order, part.stats['permuted_verts']))
        for msg in part.issues:
            log('         ! %s' % msg)
        for vi in bone0_heavy[:10]:
            log('         root-heavy vertex %d (group %d): w=%.3f  raw bones=%s' % (
                vi, part.vinfo[vi].group, part.weights[vi][0], part.groups[part.vinfo[vi].group].bone_indices), 2)
        if level_ok(log, 3):
            for vi, (wd, info) in enumerate(zip(part.weights, part.vinfo)):
                if info.infl > 1 or info.notes:
                    log('         v%03d g%02d %s %s' % (
                        vi, info.group, {model.bones[b].name: round(x, 4) for b, x in wd.items()},
                        '; '.join(info.notes)), 3)


def level_ok(log, lvl):
    return getattr(log, 'level', 1) >= lvl


def seam_check(model, tol=1e-4, weight_tol=0.01, max_examples=15):
    """Find vertices that sit on the same spot in DIFFERENT parts and compare their weights.

    Parts aren't welded (spec 3.4), so a seam is just two vertices with the same
    position. If their weights differ the seam can split open when you pose it.
    Returns a dict with the counts and some worst-case examples.
    """
    cell = max(tol, 1e-9)
    grid = {}   # integer cell -> list of (part_index, vertex_index, pos)
    for part in model.parts:
        for vi, (x, y, z) in enumerate(part.positions):
            pos = world_pos(model, (x, y, z))
            key = (round(pos[0] / cell), round(pos[1] / cell), round(pos[2] / cell))
            grid.setdefault(key, []).append((part.index, vi, pos))

    pairs = mismatched = 0
    examples = []
    seen = set()
    for key, items in grid.items():
        # look at this cell and its neighbours, because a point near a cell edge can land next door
        candidates = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    candidates += grid.get((key[0] + dx, key[1] + dy, key[2] + dz), [])
        for (pa, va, posa) in items:
            for (pb, vb, posb) in candidates:
                if pa >= pb:            # each cross-part pair only once, skip same-part
                    continue
                if (pa, va, pb, vb) in seen:
                    continue
                if max(abs(posa[i] - posb[i]) for i in range(3)) > tol:
                    continue
                seen.add((pa, va, pb, vb))
                pairs += 1
                wa, wb = model.parts[pa].weights[va], model.parts[pb].weights[vb]
                diff = max(abs(wa.get(b, 0.0) - wb.get(b, 0.0)) for b in set(wa) | set(wb))
                if diff > weight_tol:
                    mismatched += 1
                    examples.append((diff, pa, va, pb, vb))
    examples.sort(reverse=True)
    return {'pairs': pairs, 'mismatched': mismatched, 'worst': examples[:max_examples]}


def log_seam_report(model, report, log):
    log('[YOBJ] seam check: %d coincident vertex pairs across parts, %d with differing weights' % (
        report['pairs'], report['mismatched']))
    for diff, pa, va, pb, vb in report['worst']:
        fa = {model.bones[b].name: round(x, 3) for b, x in model.parts[pa].weights[va].items()}
        fb = {model.bones[b].name: round(x, 3) for b, x in model.parts[pb].weights[vb].items()}
        log('         part %02d v%d vs part %02d v%d  maxdiff=%.3f\n           %s\n           %s' % (
            pa, va, pb, vb, diff, fa, fb))


# ---- skeleton matching (pure python, no bpy) ---------------------------------
# Used to decide whether the bones of a new file can be glued onto an armature that's already
# in the scene. "Same bone" = same name AND same world transform (within a tolerance).
#
# 0.9: the old rule was "share >= 3 bones". The small accessory files (ribbons, hair clips)
# only carry a bit of trunk, usually just {root, koshi}, so they always failed it and got a
# brand new armature each. The anchor rule below lets them in when they are clearly hanging
# off the trunk, and the whole decision is explained in the log so a rejection is debuggable.

SKEL_POS_TOL = 0.01       # head positions closer than this count as the same spot (was 1e-3)
SKEL_ROT_TOL = 0.01       # every cell of the 3x3 rotation block must be this close (was 1e-3)
# Files that share this many bones (with the root among them) merge, no questions asked.
MERGE_MIN_MATCHES = 3


def flatten4(m):
    return [m[i][j] for i in range(4) for j in range(4)]


def unflatten4(f):
    return [list(f[i * 4:i * 4 + 4]) for i in range(4)]


def transform_delta(g1, g2):
    """(position distance, biggest rotation-cell difference) between two 4x4 transforms."""
    pos = math.dist((g1[0][3], g1[1][3], g1[2][3]), (g2[0][3], g2[1][3], g2[2][3]))
    rot = max(abs(g1[i][j] - g2[i][j]) for i in range(3) for j in range(3))
    return pos, rot


def same_transform(g1, g2, pos_tol=SKEL_POS_TOL, rot_tol=SKEL_ROT_TOL):
    pos, rot = transform_delta(g1, g2)
    return pos <= pos_tol and rot <= rot_tol


def skeleton_of_model(model):
    """{bone name: g_fix} for a parsed model"""
    return {b.name: b.g_fix for b in model.bones}


def evaluate_merge(skel, model, pos_tol=SKEL_POS_TOL, rot_tol=SKEL_ROT_TOL):
    """Compare a model's bones to a skeleton ({name: g_fix}) and decide if they should merge.

    Returns a dict:
      mapping   file bone index -> existing bone name (name AND transform match) or None (= unique)
      reasons   file bone index -> why a bone is unique ('new name' / 'same name but different transform')
      deltas    file bone index -> (pos error, rot error) for every same-name bone, matched or not
      shared / unique   lists of file bone indices
      root_ok   every root bone of the file matches an existing bone
      closed    every shared bone's parent is shared too (the shared part is one piece from the root)
      attached  every unique bone eventually hangs off a shared bone
      accept    the decision, and `rule` says which rule made it
    """
    mapping, reasons, deltas, parent_override = {}, {}, {}, {}
    # An armature can already hold name.001, name.002 ... made by an earlier merge (same file
    # name, different rest pose). Group them so a file is compared against every variant.
    variants = {}
    for bname, g in skel.items():
        variants.setdefault(re.sub(r'\.\d{3}$', '', bname), []).append((bname, g))
    for b in model.bones:
        cands = variants.get(b.name)
        if not cands:
            mapping[b.index], reasons[b.index] = None, 'new name'
            continue
        scored = [(transform_delta(g, b.g_fix), bn) for bn, g in cands]
        (dp, dr), bn = min(scored, key=lambda t: t[0][0] + t[0][1])
        deltas[b.index] = (dp, dr)
        hit = next((bn2 for (p2, r2), bn2 in scored if p2 <= pos_tol and r2 <= rot_tol), None)
        if hit is not None:
            mapping[b.index] = hit
        else:
            mapping[b.index], reasons[b.index] = None, 'same name but different transform'
            # fallback: the duplicate hangs off the ORIGINAL bone, not the original's parent
            parent_override[b.index] = min((bn2 for bn2 in (c[0] for c in cands)), key=len)

    shared = [i for i, v in mapping.items() if v is not None]
    unique = [i for i, v in mapping.items() if v is None]
    roots = [b.index for b in model.bones if b.parent < 0]
    root_ok = bool(roots) and all(mapping[r] is not None for r in roots)
    closed = all(model.bones[i].parent < 0 or mapping[model.bones[i].parent] is not None for i in shared)

    def hangs_off_shared(i):
        hops = 0
        while i >= 0 and hops <= len(model.bones):
            if mapping[i] is not None:
                return True
            i, hops = model.bones[i].parent, hops + 1
        return False

    attached = all(hangs_off_shared(i) for i in unique)
    n = len(shared)

    accept, rule = False, ''
    if n >= MERGE_MIN_MATCHES and root_ok:
        accept, rule = True, 'A: %d shared bones including the root' % n
    elif n >= MERGE_MIN_MATCHES:
        accept, rule = True, 'B: %d shared bones (root not among them, old 0.8 behaviour)' % n
    elif (n >= 1 and root_ok and closed and attached and (unique or n >= 2)
          and (n >= 2 or len(model.bones) < len(skel))):
        # Anchor rule for small accessory files (trunk of root + koshi, then their own chain).
        # A prop with its own lone "root" at the origin would also match with 1 shared bone, so in
        # that case the file has to be smaller than the armature. Every extra bone has to hang off
        # the shared trunk either way.
        accept, rule = True, 'C: anchor rule, trunk of %d shared bone(s) from the root, %d unique bone(s) attached' % (
            n, len(unique))
    else:
        why = []
        if n == 0:
            why.append('no shared bones')
        if n and not root_ok:
            why.append('file root does not match')
        if n and not closed:
            why.append('shared bones are not a connected piece from the root')
        if n and not attached:
            why.append('some unique bones do not attach to the shared trunk')
        if n == 1 and not unique:
            why.append('a lone shared bone and nothing to add (could be a prop)')
        if n == 1 and len(model.bones) >= len(skel):
            why.append('only the root is shared and the file is not smaller than the armature (could be a prop)')
        rule = 'rejected: ' + (', '.join(why) or 'fewer than %d shared bones' % MERGE_MIN_MATCHES)
    return {'mapping': mapping, 'reasons': reasons, 'deltas': deltas,
            'parent_override': parent_override, 'shared': shared,
            'unique': unique, 'root_ok': root_ok, 'closed': closed, 'attached': attached,
            'accept': accept, 'rule': rule}


def pick_main(models):
    """Index of the model that should supply the base skeleton of a batch.

    A character body (file 0002 with a real skeleton) wins, so the rig is named after the
    body and not after some accessory that happens to carry two extra bones. Otherwise it is
    the model with the most bones (first one wins a tie).
    """
    for i, m in enumerate(models):
        if Path(m.path).stem.upper() == '0002' and len(m.bones) >= 50:
            return i
    return max(range(len(models)), key=lambda i: (len(models[i].bones), -i))


def folder_tag(path):
    """Name of the folder a .yobj sits in ('1810', '0010' ...).

    The game exports one fighter (or one arena) per folder, so the folder tells us which
    character a file belongs to, which is more reliable than guessing from material names.
    """
    try:
        return os.path.basename(os.path.dirname(os.path.abspath(str(path)))).strip()
    except Exception:
        return ''


def peek_yobj(path):
    """Cheap look at a file: (bone count, list of names in its name table). No full parse."""
    d = read_data_block(path)
    bone_count = _u32(d, 0x14)
    name_count = _u32(d, 0x18)
    name_table = _u32(d, 0x24)
    names = [d[name_table + i * 16:name_table + i * 16 + 16].split(b'\0')[0].decode('ascii', 'replace')
             for i in range(name_count)]
    return bone_count, names


def detect_mode(paths):
    """Guess 'CHARACTER' or 'ARENA' for a set of .yobj files. Returns (mode, reason).

    Order of evidence, strongest first:
      1. arena data files (C351.dat, 526C/526F.dat) beside the files -> arena
      2. any file with 50+ bones (a fighter's body)                  -> character
      3. texture names like 1810_belt (fighter) or aud3d_01a (crowd)
      4. every file has one bone or none (static scenery)            -> arena
      5. anything else                                               -> character
    """
    if not paths:
        return 'CHARACTER', 'no files'
    for finder, label in ((find_c351, 'C351.dat'), (find_crowd_dat, '526C/526F.dat')):
        try:
            if finder(paths[0]):
                return 'ARENA', '%s found next to the files' % label
        except Exception:
            pass
    peeks = []
    for p in paths:
        try:
            peeks.append(peek_yobj(p))
        except Exception:
            pass
    if not peeks:
        return 'CHARACTER', 'could not read the files'
    most = max(b for b, _ in peeks)
    if most >= 50:
        return 'CHARACTER', 'a file has %d bones (fighter body)' % most
    names = [n for _, ns in peeks for n in ns]
    if any(re.match(r'\d{4}_', n) for n in names):
        return 'CHARACTER', 'fighter-style texture names (e.g. 1810_belt)'
    if any(n.lower().startswith('aud3d') for n in names):
        return 'ARENA', 'crowd texture names (aud3d_...)'
    if most <= 1:
        return 'ARENA', 'every file is a single-bone static mesh'
    return 'CHARACTER', 'nothing says arena, assuming a fighter file'



# =============================================================================
# 3. BUILD LAYER  (bpy lives from here down)
# =============================================================================

try:
    import bpy
    from mathutils import Matrix, Vector
except ImportError:      # running from a normal Python, just parse + print
    bpy = None


@dataclass
class ImportOptions:
    bone_base: int = 1
    lane_order: str = 'IDENTITY'
    bone_orient: str = 'FILE_AXES'      # or POINT_TO_CHILD
    tail_length: float = 0.1
    use_custom_normals: bool = True
    import_materials: bool = True
    texture_dir: str = ''
    axis: str = 'Z_UP'                  # or NONE
    handedness: str = 'BAKE'            # BAKE | OBJECT | NONE  (how the mirror gets fixed)
    flip_u: bool = False                # last resort if geometry is right but textures read backwards
    global_scale: float = 1.0
    log_level: int = 1
    debug_attributes: bool = True
    debug_heatmaps: bool = True
    debug_bone_empties: bool = False
    seam_check: bool = True
    seam_tolerance: float = 1e-4
    write_log_text: bool = True
    new_collection: bool = True         # False = drop objects into the active collection
    secondary_layers: bool = True       # build the sheen (layer 2) overlay as a matcap, where a material has one
    sheen_flip_v: bool = False          # flip the matcap vertically if highlights sit upside down
    sheen_strength: float = 0.25        # emission strength of the sheen layer (1.0 = full)
    cloth_alpha_spec: bool = True       # clothing alpha -> specular level + roughness
    hair_body_cutoff: float = -1.0      # hair body alpha cutoff, negative = leave it opaque (what the PS2 does)
    reuse_materials: bool = True        # reuse materials from earlier imports instead of making .001 copies
    merge_skeletons: bool = True        # reuse / extend a matching armature instead of making a new one
    merge_tolerance: float = SKEL_POS_TOL   # how close (position and rotation) "the same bone" has to be
    hair_soft_fringe: bool = False      # hair strands: soft alpha blend instead of the hard 0.44 cutout
    sheen_mapping: str = 'CAMERA'       # CAMERA = view-space normal, REFLECT = reflection vector (A/B test)
    sheen_alpha_scale: float = 1.0      # multiplier on the base-alpha mask (2.0 if the TGA alpha is raw PS2, 0x80 = 1.0)
    sheen_blend: str = 'EMISSION'       # EMISSION = linear add, GAMMA_ADD = emulate the GS add in sRGB space
    # ---- Arena Mode (0.9.3) ----
    mode: str = 'CHARACTER'             # CHARACTER | ARENA (Auto Detect is resolved before this is filled in)
    mode_reason: str = ''               # why the mode was picked, for the log
    join_parts: bool = False            # one mesh object per file instead of one per part
    mirror_c351: bool = False           # ARENA: add the flipped / rotated copies the C351 'r' and 'y' flags ask for
    alpha_c351: bool = False            # ARENA: alpha blending from the C351 's' / 'k' / 'a' flags
    dynamic_collections: bool = False   # ARENA: park 'e' (event only) and 'c' (excluded) meshes in their own collections
    c351_flags: object = None           # filled in by run_import_batch: {yobj number: info dict}, or None
    # ---- Crowd (0.9.5) ----
    place_crowd: bool = False           # ARENA: spread the 4E2x crowd files over the seats in 526C/526F.dat
    crowd_density: str = 'C'            # 'C' = 526C.dat (full crowd), 'F' = 526F.dat (lighter crowd)
    crowd_hip_height: float = 8.0       # height of the crowd model's origin (its hips) above the floor
    crowd_face_ring: bool = True        # turn every crowd member to look at the ring centre
    crowd_yaw_offset: float = 0.0       # extra turn in degrees, in case the models face the wrong way
    crowd_data: object = None           # filled in by run_import_batch: {'seats', 'types', 'plan', 'path'}


DIST_HEAT_RANGE = 2.0   # vertex-to-bone distance that maps to full red in the heat map


def root_matrix(opts):
    """World matrix for the armature object (meshes are parented, so they follow).

    Y-up -> Z-up is +90 degrees about X (not -90, whatever the comment in the original note
    says). OBJECT handedness mode also drops in the X scale of -1, but a negative scale on an
    armature makes posing and exporting awkward, so BAKE (default) does the flip in the data.
    """
    m = Matrix.Identity(4)
    if opts.axis == 'Z_UP':
        m = Matrix.Rotation(math.radians(90.0), 4, 'X')
    if opts.handedness == 'OBJECT':
        m = m @ Matrix.Scale(-1, 4, (1, 0, 0))
    return m @ Matrix.Scale(opts.global_scale, 4)


def root_signature(opts):
    """Armatures can only be merged if they were made with the same flip/axis/scale,
    otherwise their bone positions live in different spaces."""
    return '%s|%s|%.6f' % (opts.handedness, opts.axis, opts.global_scale)


def _shape_edit_bone(eb, b, bones, opts):
    """Give one edit bone its length, orientation and (maybe) a tail aimed at its children."""
    kids = b.children
    if kids:
        nearest = min(math.dist(b.head, bones[k].head) for k in kids)
        length = max(nearest * 0.5, opts.tail_length * 0.25)
    else:
        length = opts.tail_length

    # Give it a sane starting shape, THEN set the matrix. A zero-length bone
    # makes Blender throw away the orientation.
    eb.head = (0.0, 0.0, 0.0)
    eb.tail = (0.0, length, 0.0)
    eb.matrix = Matrix(b.g_fix)     # head = translation, axes = G_fix rotation
    eb.length = length

    if opts.bone_orient == 'POINT_TO_CHILD' and kids:
        avg = [sum(bones[k].head[i] for k in kids) / len(kids) for i in range(3)]
        if math.dist(avg, b.head) > 1e-5:
            z_axis = Vector((b.g_fix[0][2], b.g_fix[1][2], b.g_fix[2][2]))
            eb.tail = avg
            eb.align_roll(z_axis)   # keep the file's local Z as the roll reference


def _add_bones(arm_obj, model, indices, names, opts, parent_override=None):
    """Create edit bones for the given file bone indices inside an existing armature object.

    `names` maps file bone index -> real Blender bone name for bones that already exist.
    It gets extended in place with the bones we create here, so parents can point at
    either kind. Returns the worst head error vs the parsed global position.
    """
    arm_data = arm_obj.data
    bones = model.bones

    # edit mode needs us to be in object mode first, and the armature to be active
    ctx_obj = bpy.context.object
    if ctx_obj is not None and ctx_obj.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')

    created = {}
    for i in indices:
        b = bones[i]
        eb = arm_data.edit_bones.new(b.name)
        names[i] = eb.name              # Blender may have renamed it, trust this one
        _shape_edit_bone(eb, b, bones, opts)
        created[i] = eb

    # parents after everything exists (a parent can come later in the table).
    # The parent can be a brand new bone or one that was already in the armature.
    parent_override = parent_override or {}
    for i in indices:
        p = bones[i].parent
        if i in parent_override and parent_override[i] in arm_data.edit_bones:
            created[i].parent = arm_data.edit_bones[parent_override[i]]   # kubi.001 under kubi
            created[i].use_connect = False
        elif p >= 0 and p in names:
            created[i].parent = arm_data.edit_bones[names[p]]
            created[i].use_connect = False

    bpy.ops.object.mode_set(mode='OBJECT')

    # custom props only work reliably on the real bones, not edit bones.
    # yobj_g_fix is what later imports compare against, so it works whatever bone direction was used.
    worst = 0.0
    for i in indices:
        b = bones[i]
        pb = arm_data.bones[names[i]]
        pb['yobj_index'] = b.index
        pb['yobj_file_name'] = b.name       # the name in the file, before Blender's .001
        pb['yobj_local_pos'] = list(b.local_pos)
        pb['yobj_local_rot'] = list(b.local_rot)
        pb['yobj_g_fix'] = flatten4(b.g_fix)
        worst = max(worst, math.dist(pb.head_local, b.head))
    return worst


def build_armature(model, opts, log, collection):
    """Make a fresh armature from G_fix. Returns (armature_object, {file bone index: real bone name})."""
    # suffix with the folder name (body_armature_1810) so every character/arena rig is easy to tell apart
    folder = folder_tag(model.path)
    arm_name = model.model_name + '_armature' + ('_' + folder.lstrip('_') if folder else '')
    arm_data = bpy.data.armatures.new(arm_name)
    arm_obj = bpy.data.objects.new(arm_name, arm_data)
    collection.objects.link(arm_obj)

    names = {}
    worst = _add_bones(arm_obj, model, range(len(model.bones)), names, opts)
    log('[YOBJ] armature: %d bones, orient=%s, worst head error vs parsed global_pos = %.2e%s' % (
        len(model.bones), opts.bone_orient, worst, '' if worst < 1e-5 else '   <-- TOO BIG'))

    # axis + handedness + scale all live on the object matrix
    arm_obj.matrix_world = root_matrix(opts)
    arm_obj['yobj_armature'] = True                 # marks it as ours so later imports can find it
    arm_obj['yobj_root_sig'] = root_signature(opts)
    arm_obj['yobj_source'] = os.path.basename(model.path)
    arm_obj['yobj_folder'] = folder      # merging only happens between files from the same folder
    return arm_obj, names


def armature_skeleton(arm_obj):
    """Read {bone name: g_fix} back from the bone custom props. Bones we didn't make are skipped."""
    skel = {}
    for bone in arm_obj.data.bones:
        flat = bone.get('yobj_g_fix')
        if flat is not None and len(flat) == 16:
            skel[bone.name] = unflatten4(list(flat))
    return skel


def describe_merge(model, ev, arm_name, log, level=1):
    """Spell out a merge decision: which bones matched, which were off and by how much, which rule fired."""
    names = [model.bones[i].name for i in ev['shared']]
    log('[YOBJ] skeleton check vs %r: %d shared / %d unique of %d bones  ->  %s' % (
        arm_name, len(ev['shared']), len(ev['unique']), len(model.bones), ev['rule']), level)
    log('         shared: %s' % (', '.join(names) or 'none'), level + 1)
    for i, (dp, dr) in sorted(ev['deltas'].items()):
        log('         %-14s %s   dpos=%.2e drot=%.2e' % (
            model.bones[i].name, 'match   ' if ev['mapping'][i] is not None else 'MISMATCH', dp, dr), level + 1)
    log('         file root matches: %s   shared part connected: %s   unique bones attached: %s' % (
        ev['root_ok'], ev['closed'], ev['attached']), level + 1)


def find_merge_target(model, opts, log=None):
    """Look through the scene for one of OUR armatures this model can be merged into.

    Needs the same root signature and an accepting rule from evaluate_merge() (see there). If
    several qualify, the armature the user currently has active wins, then whichever shares the
    most bones. Returns (arm_obj, evaluation) or None. Every candidate is explained in the log.
    """
    sig = root_signature(opts)
    active = bpy.context.view_layer.objects.active
    active_arm = None
    if active is not None:
        if active.type == 'ARMATURE':
            active_arm = active
        elif active.parent is not None and active.parent.type == 'ARMATURE':
            active_arm = active.parent

    best, rejected = None, []
    my_folder = folder_tag(model.path).lower()
    for obj in bpy.data.objects:
        if obj.type != 'ARMATURE' or not obj.get('yobj_armature'):
            continue
        # a different folder means a different fighter: never share a rig with it.
        # (rigs made by older versions have no folder stored, those stay allowed)
        their_folder = str(obj.get('yobj_folder') or '').lower()
        if my_folder and their_folder and my_folder != their_folder:
            if log:
                log('[YOBJ] skeleton: skipping %r, it came from folder %r and this file is from %r' % (
                    obj.name, their_folder, my_folder))
            continue
        if obj.get('yobj_root_sig') != sig:
            if log:
                log('[YOBJ] skeleton: skipping %r, different axis/flip/scale settings (%s vs %s)' % (
                    obj.name, obj.get('yobj_root_sig'), sig))
            continue
        ev = evaluate_merge(armature_skeleton(obj), model, opts.merge_tolerance, opts.merge_tolerance)
        if not ev['accept']:
            rejected.append((obj, ev))
            continue
        key = (active_arm is not None and obj == active_arm, len(ev['shared']))
        if best is None or key > best[0]:
            best = (key, obj, ev)
    if log:
        for obj, ev in rejected:
            describe_merge(model, ev, obj.name, log, 1 if best is None else 2)
    return (best[1], best[2]) if best else None


def attach_skeleton(model, opts, log, collection):
    """Decide: reuse/extend an existing armature, or build a new one.
    Returns (arm_obj, bone_names {file index: Blender bone name})."""
    found = find_merge_target(model, opts, log) if opts.merge_skeletons else None
    if found is None:
        if opts.merge_skeletons:
            log('[YOBJ] skeleton: no compatible armature in the scene, building a new one')
        return build_armature(model, opts, log, collection)

    arm_obj, ev = found
    mapping, reasons = ev['mapping'], ev['reasons']
    names = {i: n for i, n in mapping.items() if n is not None}
    unique = ev['unique']
    describe_merge(model, ev, arm_obj.name, log, 1)
    log('[YOBJ] skeleton: merging into %r  shared bones reused: %d   unique bones to add: %d' % (
        arm_obj.name, len(names), len(unique)))
    for i in unique:
        if reasons.get(i, '').startswith('same name'):
            dp, dr = ev['deltas'][i]
            log('         ! bone %r differs from the existing one (dpos=%.2e drot=%.2e): added as a '
                'duplicate parented to %r' % (model.bones[i].name, dp, dr, ev['parent_override'][i]))
    if unique:
        worst = _add_bones(arm_obj, model, unique, names, opts, ev['parent_override'])
        log('[YOBJ] added bones: %s   (worst head error %.2e)' % (
            ', '.join(names[i] for i in unique), worst), 2)
    return arm_obj, names


ALPHA_OPAQUE_BELOW = 0.99   # any pixel with alpha under this means "not fully opaque"


def analyze_alpha(img):
    """Look at a loaded image and say whether its alpha channel actually does anything.

    Returns (has_alpha, binary, a_min, a_max). `binary` means every alpha value is
    basically 0 or 1 (a cutout, like a fence), as opposed to soft translucency.
    A 32-bit TGA is not proof of transparency (plenty are 32-bit and solid), and the
    Rumble Roses TGAs all claim 0 alpha bits in the header even though they are 32 bpp,
    so we don't trust depth at all and just read the pixels (Blender always hands back
    RGBA, with alpha 1.0 for images that have none).
    The answer is cached on the image so repeat imports don't re-read it.
    """
    # Custom property arrays in Blender must be ONE type (all floats), so the cache is
    # four floats: [has_alpha 0/1, binary 0/1, min, max]. Mixing bools and floats here
    # throws "only floats, ints, booleans and dicts are allowed in ID property arrays".
    try:
        cached = img.get('yobj_alpha')
        if cached is not None and len(cached) == 4:
            return cached[0] > 0.5, cached[1] > 0.5, float(cached[2]), float(cached[3])
    except Exception:
        pass

    result = (False, False, 1.0, 1.0)
    if len(img.pixels) > 0:
        try:
            import numpy as np          # ships with Blender, and makes this fast
            buf = np.empty(len(img.pixels), dtype=np.float32)
            img.pixels.foreach_get(buf)
            a = buf[3::4]
            a_min, a_max = float(a.min()), float(a.max())
            binary = bool(np.all((a < 0.02) | (a > 0.98)))
        except ImportError:             # slow path, just in case numpy is missing
            a = list(img.pixels[3::4])
            a_min, a_max = min(a), max(a)
            binary = all(v < 0.02 or v > 0.98 for v in a)
        result = (a_min < ALPHA_OPAQUE_BELOW, binary, a_min, a_max)

    try:
        img['yobj_alpha'] = [1.0 if result[0] else 0.0, 1.0 if result[1] else 0.0,
                             float(result[2]), float(result[3])]
    except Exception:
        pass        # the cache is a nice-to-have, never worth failing an import over
    return result


def enable_transparency(mat, binary):
    """Turn on whatever transparency switch this Blender version has.

    Newer versions use surface_render_method, older ones blend_method. Both are
    poked inside try blocks because one or the other may be gone (4.2+ replaced the
    old one, and later versions may drop it entirely).
    """
    try:
        mat.surface_render_method = 'DITHERED'
    except Exception:
        pass
    try:
        mat.blend_method = 'CLIP' if binary else 'HASHED'
    except Exception:
        pass


# ---- material rules ---------------------------------------------------------------------------
# Where these come from: PCSX2 GS dumps of the character select screen (Reiko + Noble Rose).
#
# The +0xC0 low word of a material record is the PASS TYPE, and it decides how the PS2 draws it:
#   1 = normal alpha blend WITH an alpha test (hair strands, lashes, ribbons, skin/clothes)
#   2 = "replace" blend, alpha test OFF, plus a sheen pass on top (hair body, shiny clothing)
# For type 2 the alpha channel is NOT transparency, it's a mask that the sheen pass uses.
# That's why the hair body has no real threshold: the engine just draws it opaque.
#
# Every draw in both dumps also uses CLAMP texture addressing (never repeat), so every Image
# Texture node here is set to EXTEND. Blender's default (Repeat) wraps the few UVs that stray
# past 0..1 onto the opposite edge of the texture, which is where the dark padding came from.
#
# 0.9: roles come from the material name + pass type ONLY. The old "file is called 271X so it's
# a lash" rule is gone, because it made the same ribbon texture a LASH material in one file and a
# CLOTH material in another, and that is what produced ribon.001 on every batch import.

HAIR_NAME_HINTS = ('hair',)             # if a character's hair texture isn't called something-hair, add it here
LASH_NAME_HINTS = ('matuge', 'matge')   # eyelash material names
HAIR_STRAND_CUTOFF = 0.44               # hair strands: 112/255, the PS2's own alpha-test reference
SPEC_ROUGH_DROP = 0.65                  # experimental spec mask: roughness goes from 1.0 down to 0.35
IDENT_VERSION = 'v9'


def pass_type(m):
    """The +0xC0 low word, see the big comment above. 0 if the record words weren't kept."""
    return (m.raw_words[0xC0 // 4] & 0xFFFF) if len(m.raw_words) > 0xC0 // 4 else 0


def pass_hi(m):
    """The +0xC0 high word. Meaning still unknown, only logged and stored on the material."""
    return (m.raw_words[0xC0 // 4] >> 16) if len(m.raw_words) > 0xC0 // 4 else 0


def classify_role(m):
    """Sort a material record into 'LASH', 'HAIR_STRAND', 'HAIR_BODY' or 'CLOTH' (everything else).

    Lashes and hair are spotted by material (texture) name, there's no "this is hair" flag in the
    record that I've found. Strand vs body uses the pass type: body is the one drawn with the
    replace blend (type 2), strands are the alpha-blended ones (type 1).
    """
    name = m.name.lower()
    if any(h in name for h in LASH_NAME_HINTS):
        return 'LASH'
    if not any(h in name for h in HAIR_NAME_HINTS):
        return 'CLOTH'
    if pass_type(m) == 2 or m.layer2_kind != 0:
        return 'HAIR_BODY'
    return 'HAIR_STRAND'


def assign_roles(model):
    """Stamp every material record with its role. Done once up front so material_key() (which has
    no idea what file it's looking at) can just read it back."""
    for part in model.parts:
        for m in part.materials:
            m.role = classify_role(m)


def material_role(m):
    return m.role or classify_role(m)


def sheen_wanted(m, opts):
    """True when this record has a layer 2 texture AND the user left the sheen option on."""
    return (opts.secondary_layers and m.layer2_kind in (1, 2)
            and m.layer2_slot >= 0 and bool(m.layer2_name))


def material_key(m, opts):
    """What makes two records the same Blender material inside one file: texture slot, sheen
    layer (if we're building it) and role. Role is in there because the two hair records share one
    texture slot but need different alpha rules, so they MUST end up as two separate materials."""
    return (m.slot, m.layer2_slot if sheen_wanted(m, opts) else -1, material_role(m))


# ---- material identity (0.9) -------------------------------------------------------------------
# Two materials are "the same" when they use the same texture CONTENT and have the same node
# graph. Not the same name, not the same file path, not the same role label. So:
#     ident = v9 | hash of the base texture bytes | fingerprint of what the graph does
#     fingerprint = alpha handling (none / link / cutoff) + sheen (hash, strength, ...) + spec flag
# Materials made by older versions have no stored ident, so their node graph gets read back and
# turned into the same string.

_HASH_CACHE = {}


def file_hash(path):
    """Short SHA-1 of a file's bytes, '' if it can't be read. Cached by (path, mtime, size)."""
    if not path:
        return ''
    try:
        st = os.stat(path)
        key = (os.path.normcase(os.path.abspath(path)), st.st_mtime_ns, st.st_size)
        h = _HASH_CACHE.get(key)
        if h is None:
            with open(path, 'rb') as f:
                h = hashlib.sha1(f.read()).hexdigest()[:16]
            _HASH_CACHE[key] = h
        return h
    except OSError:
        return ''


def image_hash(img):
    """Same hash for an image datablock: its file on disk, or the packed bytes."""
    try:
        path = bpy.path.abspath(img.filepath) if img.filepath else ''
        h = file_hash(path)
        if h:
            return h
        if img.packed_file is not None:
            return hashlib.sha1(bytes(img.packed_file.data)).hexdigest()[:16]
    except Exception:
        pass
    return ''


def load_image(path):
    """Load a texture, but hand back an image that's already in the file if one has the same
    bytes (even from another folder), so repeat imports don't pile up name.001 images."""
    h = file_hash(path)
    if h:
        for img in bpy.data.images:
            if img.source == 'FILE' and image_hash(img) == h:
                return img
    return bpy.data.images.load(path, check_existing=True)


def alpha_token(kind, thr=0.0, binary=False):
    if kind == 'cut':
        return 'cut%.3f' % thr
    if kind == 'link':
        return 'linkB' if binary else 'link'
    return 'none'


def sheen_token(l2_hash, opts):
    if not l2_hash:
        return '-'
    return '%s:%.3f:%d:%s:%.3f:%s' % (l2_hash, opts.sheen_strength, 1 if opts.sheen_flip_v else 0,
                                      opts.sheen_mapping, opts.sheen_alpha_scale, opts.sheen_blend)


def make_ident(base_hash, kd, alpha_tok, sheen_tok, spec):
    base = base_hash or ('kd%.3f,%.3f,%.3f' % tuple(kd[:3]))    # no texture: the flat colour is all there is
    return '%s|%s|a=%s|s=%s|sp=%d' % (IDENT_VERSION, base, alpha_tok, sheen_tok, 1 if spec else 0)


def ident_diff(a, b):
    """Which parts of two ident strings differ, for the log."""
    names = ('version', 'base texture', 'alpha', 'sheen', 'spec mask')
    return ', '.join('%s: %s -> %s' % (n, x, y) for n, x, y in zip(names, a.split('|'), b.split('|')) if x != y) or 'identical'


def find_principled(tree):
    for n in tree.nodes:
        if n.type == 'BSDF_PRINCIPLED':
            return n
    return None


def _src(sock):
    """The node feeding a socket, or None."""
    try:
        if sock.is_linked and sock.links:
            return sock.links[0].from_node
    except Exception:
        pass
    return None


def graph_ident(mat):
    """Rebuild a material's ident from its node graph. Used for materials made by 0.6 - 0.8 (no
    stored ident). Returns None for anything that doesn't look like one of ours."""
    tree = getattr(mat, 'node_tree', None)
    bsdf = find_principled(tree) if tree else None
    if bsdf is None:
        return None
    try:
        base = _src(bsdf.inputs['Base Color'])
        if base is None or base.type != 'TEX_IMAGE' or base.image is None:
            return None
        # not obviously ours (no yobj_ props)? then the material name has to start with the texture name
        if not any(k.startswith('yobj_') for k in mat.keys()):
            stem = os.path.splitext(os.path.basename(base.image.filepath))[0].lower()
            if not stem or not mat.name.lower().startswith(stem):
                return None
        base_hash = image_hash(base.image)
        if not base_hash:
            return None

        a = _src(bsdf.inputs['Alpha'])
        if a is None:
            atok = 'none'
        elif a.type == 'MATH' and a.operation == 'GREATER_THAN':
            atok = alpha_token('cut', a.inputs[1].default_value)
        elif a.type == 'TEX_IMAGE' and a.image is not None:
            atok = alpha_token('link', binary=analyze_alpha(a.image)[1])
        else:
            return None

        stok = '-'
        e = _src(bsdf.inputs['Emission Color'])
        if e is not None:
            tex = _src(e.inputs[0]) if e.type == 'VECTOR_MATH' else e
            if tex is None or tex.type != 'TEX_IMAGE' or tex.image is None:
                return None
            flip = 0
            mp = _src(tex.inputs['Vector'])
            if mp is not None and mp.type == 'MAPPING' and mp.inputs['Scale'].default_value[1] < 0:
                flip = 1
            stok = '%s:%.3f:%d:CAMERA:1.000:EMISSION' % (
                image_hash(tex.image), bsdf.inputs['Emission Strength'].default_value, flip)
        spec = False
        try:
            spec = bsdf.inputs['Specular IOR Level'].is_linked
        except KeyError:
            pass
        return make_ident(base_hash, None, atok, stok, spec)
    except Exception:
        return None


def collect_existing_materials(log):
    """{ident: material} for everything an earlier import (any version) left in this file."""
    found = {}
    for old in bpy.data.materials:
        ident, source = old.get('yobj_ident'), 'stored'
        if not ident:
            ident, source = graph_ident(old), 'graph'
        if not ident:
            continue
        # if several qualify, a stored ident beats a graph guess, then the shortest name (no .001)
        rank = (source != 'stored', len(old.name))
        if ident not in found or rank < found[ident][0]:
            found[ident] = (rank, old, source)
    log('[YOBJ] materials: %d reusable material(s) already in this file' % len(found), 2)
    return {k: (v[1], v[2]) for k, v in found.items()}


def add_alpha_cutout(tree, tex_node, bsdf, threshold):
    """Texture Alpha -> Math (Greater Than, threshold) -> BSDF Alpha.
    The Math node turns soft alpha into a hard 0/1, so the result is a clean cutout."""
    cut = tree.nodes.new('ShaderNodeMath')
    cut.operation = 'GREATER_THAN'
    cut.use_clamp = False
    cut.inputs[1].default_value = threshold     # input 0 is the value, input 1 is the threshold
    cut.location = (-50, 150)
    tree.links.new(tex_node.outputs['Alpha'], cut.inputs[0])
    tree.links.new(cut.outputs['Value'], bsdf.inputs['Alpha'])


def add_cloth_spec_mask(tree, tex_node, bsdf, ks):
    """EXPERIMENTAL. Treat the clothing alpha as a shine mask on the normal Principled specular:
        alpha * Ks -> Specular IOR Level      (more alpha = more highlight)
        1 - 0.65*alpha -> Roughness           (more alpha = glossier, down to 0.35)
    This is a guess at what the alpha is for, it's here so it can be compared against the sheen."""
    spec = tree.nodes.new('ShaderNodeMath')
    spec.operation = 'MULTIPLY'
    spec.inputs[1].default_value = float(ks)
    spec.location = (-50, 0)
    tree.links.new(tex_node.outputs['Alpha'], spec.inputs[0])
    tree.links.new(spec.outputs['Value'], bsdf.inputs['Specular IOR Level'])

    rough = tree.nodes.new('ShaderNodeMath')
    rough.operation = 'MULTIPLY_ADD'            # value * multiplier + addend
    rough.inputs[1].default_value = -SPEC_ROUGH_DROP
    rough.inputs[2].default_value = 1.0
    rough.location = (-50, -150)
    tree.links.new(tex_node.outputs['Alpha'], rough.inputs[0])
    tree.links.new(rough.outputs['Value'], bsdf.inputs['Roughness'])


def _vmath(nodes, op, loc):
    n = nodes.new('ShaderNodeVectorMath')
    n.operation = op
    n.location = loc
    return n


def add_sheen_layer(mat, tree, bsdf, base_tex, m, l2_path, opts, log):
    """Layer 2 as a MatCap, added as emission.

    Default chain:  Geometry Normal -> Vector Transform (World to Camera) -> Mapping (scale 0.5, loc 0.5)
            -> sheen Image Texture -> x base texture alpha -> Emission Color (x Sheen Strength)
    The Mapping step is the "0.5 + 0.5 * normal.xy" the engine does, it squeezes the -1..1 normal
    into 0..1 texture space. Multiplying by the base alpha copies what the GS blend does
    (sheen * alpha already sitting in the frame buffer).

    A/B switches (cloth sheen is still not matching the game, see the session notes):
      sheen_mapping REFLECT    use the reflection vector instead of the plain normal
      sheen_alpha_scale        multiply the alpha mask (2.0 = if TGA alpha is raw PS2 where 0x80 is 1.0)
      sheen_blend GAMMA_ADD    do the add in sRGB space like the GS does, instead of adding linear light
    """
    if not l2_path:
        log('[YOBJ] material %-12s sheen texture %r not found' % (m.name, m.layer2_name))
        return
    img = load_image(l2_path)
    nodes, links = tree.nodes, tree.links

    geo = nodes.new('ShaderNodeNewGeometry')
    geo.location = (-1700, -250)

    to_cam = nodes.new('ShaderNodeVectorTransform')
    to_cam.vector_type = 'NORMAL'       # it's a direction, not a position
    to_cam.convert_from = 'WORLD'
    to_cam.convert_to = 'CAMERA'
    to_cam.location = (-1500, -250)
    links.new(geo.outputs['Normal'], to_cam.inputs['Vector'])
    lookup_vec = to_cam.outputs['Vector']

    if opts.sheen_mapping == 'REFLECT':
        # R = 2 (N.V) N - V, everything in camera space. V = Incoming (surface -> camera).
        inc = nodes.new('ShaderNodeVectorTransform')
        inc.vector_type = 'NORMAL'
        inc.convert_from = 'WORLD'
        inc.convert_to = 'CAMERA'
        inc.location = (-1500, -450)
        links.new(geo.outputs['Incoming'], inc.inputs['Vector'])
        dot = _vmath(nodes, 'DOT_PRODUCT', (-1350, -350))
        links.new(to_cam.outputs['Vector'], dot.inputs[0])
        links.new(inc.outputs['Vector'], dot.inputs[1])
        dbl = nodes.new('ShaderNodeMath')
        dbl.operation = 'MULTIPLY'
        dbl.inputs[1].default_value = 2.0
        dbl.location = (-1250, -350)
        links.new(dot.outputs['Value'], dbl.inputs[0])
        sc = _vmath(nodes, 'SCALE', (-1150, -300))
        links.new(to_cam.outputs['Vector'], sc.inputs[0])
        links.new(dbl.outputs['Value'], sc.inputs[3])
        sub = _vmath(nodes, 'SUBTRACT', (-1050, -300))
        links.new(sc.outputs['Vector'], sub.inputs[0])
        links.new(inc.outputs['Vector'], sub.inputs[1])
        lookup_vec = sub.outputs['Vector']

    mapping = nodes.new('ShaderNodeMapping')
    mapping.vector_type = 'POINT'
    mapping.inputs['Location'].default_value = (0.5, 0.5, 0.0)
    # flip_v is only there as an escape hatch, I couldn't check the vertical direction visually
    mapping.inputs['Scale'].default_value = (0.5, -0.5 if opts.sheen_flip_v else 0.5, 1.0)
    mapping.location = (-900, -250)
    links.new(lookup_vec, mapping.inputs['Vector'])

    tex = nodes.new('ShaderNodeTexImage')
    tex.image = img
    tex.extension = 'EXTEND'            # the lookup never leaves 0..1, EXTEND just avoids edge wrap
    tex.label = 'yobj_sheen'
    tex.location = (-700, -250)
    links.new(mapping.outputs['Vector'], tex.inputs['Vector'])

    mode = 'masked by base alpha' if base_tex is not None else 'unmasked'
    if base_tex is None:
        # no base texture to take an alpha from, so just show the sheen unmasked
        links.new(tex.outputs['Color'], bsdf.inputs['Emission Color'])
    else:
        alpha_out = base_tex.outputs['Alpha']
        if abs(opts.sheen_alpha_scale - 1.0) > 1e-6:
            scale = nodes.new('ShaderNodeMath')
            scale.operation = 'MULTIPLY'
            scale.inputs[1].default_value = opts.sheen_alpha_scale
            scale.location = (-700, -450)
            links.new(alpha_out, scale.inputs[0])
            alpha_out = scale.outputs['Value']
            mode += ', alpha x%.2f' % opts.sheen_alpha_scale

        if opts.sheen_blend == 'GAMMA_ADD':
            # out = lin( min(1, srgb(base) + srgb(sheen)*alpha) ) - base      (gamma 2.2 approximation)
            # i.e. the extra light the GS add would have put on top of the base colour.
            to_srgb, to_lin = (1 / 2.2,) * 3, (2.2,) * 3
            b_s = _vmath(nodes, 'POWER', (-500, -100))
            b_s.inputs[1].default_value = to_srgb
            links.new(base_tex.outputs['Color'], b_s.inputs[0])
            s_s = _vmath(nodes, 'POWER', (-500, -250))
            s_s.inputs[1].default_value = to_srgb
            links.new(tex.outputs['Color'], s_s.inputs[0])
            sa = _vmath(nodes, 'SCALE', (-350, -250))
            links.new(s_s.outputs['Vector'], sa.inputs[0])
            links.new(alpha_out, sa.inputs[3])
            add = _vmath(nodes, 'ADD', (-200, -150))
            links.new(b_s.outputs['Vector'], add.inputs[0])
            links.new(sa.outputs['Vector'], add.inputs[1])
            clamp = _vmath(nodes, 'MINIMUM', (-50, -150))
            clamp.inputs[1].default_value = (1.0, 1.0, 1.0)
            links.new(add.outputs['Vector'], clamp.inputs[0])
            back = _vmath(nodes, 'POWER', (100, -150))
            back.inputs[1].default_value = to_lin
            links.new(clamp.outputs['Vector'], back.inputs[0])
            diff = _vmath(nodes, 'SUBTRACT', (250, -150))
            links.new(back.outputs['Vector'], diff.inputs[0])
            links.new(base_tex.outputs['Color'], diff.inputs[1])
            links.new(diff.outputs['Vector'], bsdf.inputs['Emission Color'])
            mode += ', gamma-space add'
        else:
            # Vector Math "Scale" = colour * a single number. Slightly odd tool for the job but the
            # socket names are stable across versions, unlike the Mix node.
            mask = _vmath(nodes, 'SCALE', (-450, -250))
            links.new(tex.outputs['Color'], mask.inputs[0])
            links.new(alpha_out, mask.inputs[3])      # inputs[3] is the Scale value
            links.new(mask.outputs['Vector'], bsdf.inputs['Emission Color'])
    bsdf.inputs['Emission Strength'].default_value = opts.sheen_strength
    mat['yobj_layer2_slot'] = m.layer2_slot
    mat['yobj_layer2_name'] = m.layer2_name
    log('[YOBJ] material %-12s sheen %s (%s lookup) -> emission x%.2f, %s' % (
        m.name, m.layer2_name, opts.sheen_mapping.lower(), opts.sheen_strength, mode))


def plan_material(m, key, opts, search):
    """Work out everything that decides what the material for this record looks like, WITHOUT
    building it, so the result can be compared with materials that already exist.
    Returns a dict (see the end of the function for the fields)."""
    role = key[2]
    name = m.name + {'HAIR_STRAND': '_strands', 'HAIR_BODY': '_body'}.get(role, '')
    if key[1] >= 0:
        name += '+' + m.layer2_name
    tex_path = find_texture(m.name, search)
    l2_path = find_texture(m.layer2_name, search) if key[1] >= 0 else None

    atok, img, alpha_info = alpha_token('none'), None, None
    if tex_path:
        img = load_image(tex_path)
        if role == 'HAIR_STRAND':
            atok = alpha_token('link') if opts.hair_soft_fringe else alpha_token('cut', HAIR_STRAND_CUTOFF)
        elif role == 'HAIR_BODY':
            if opts.hair_body_cutoff >= 0.0:
                atok = alpha_token('cut', opts.hair_body_cutoff)
        elif role == 'LASH':
            alpha_info = analyze_alpha(img)         # (has_alpha, binary, min, max)
            if alpha_info[0]:
                atok = alpha_token('link', binary=alpha_info[1])
        elif opts.mode == 'ARENA' and pass_type(m) != 2:
            # no transparency flag exists in the yobj (every arena record is pass 1) and C351 misses
            # chairs, fences etc, so the texture decides: real alpha in the pixels -> use it.
            # Pass 2 is skipped: there the alpha is a sheen mask, not transparency.
            alpha_info = analyze_alpha(img)
            if alpha_info[0]:
                atok = alpha_token('link', binary=alpha_info[1])
    spec = bool(tex_path) and role == 'CLOTH' and opts.cloth_alpha_spec and pass_type(m) == 2
    stok = sheen_token(file_hash(l2_path) if l2_path else '', opts)
    base_hash = file_hash(tex_path)
    return {'role': role, 'name': name, 'tex_path': tex_path, 'l2_path': l2_path, 'img': img,
            'alpha_tok': atok, 'alpha_info': alpha_info, 'spec': spec,
            'ident': make_ident(base_hash, m.kd, atok, stok, spec), 'base_hash': base_hash}


def ensure_nodes(mat):
    """Blender 5 is always nodes (use_nodes is deprecated); older builds need the switch flipped."""
    if getattr(mat, 'node_tree', None) is not None:
        return
    try:
        mat.use_nodes = True
    except Exception:
        pass


def build_material(plan, m, opts, log):
    """Make the Blender material described by a plan_material() result."""
    role, tex_path, atok = plan['role'], plan['tex_path'], plan['alpha_tok']
    mat = bpy.data.materials.new(plan['name'])
    ensure_nodes(mat)
    mat['yobj_kd'] = list(m.kd)
    mat['yobj_ks'] = list(m.ks)
    mat['yobj_shader'] = m.shader
    mat['yobj_role'] = role
    mat['yobj_pass'] = pass_type(m)
    mat['yobj_pass_hi'] = pass_hi(m)        # unknown meaning, kept for whoever figures it out
    mat['yobj_ident'] = plan['ident']
    tree = mat.node_tree
    bsdf = find_principled(tree) if tree else None
    if bsdf is None:
        return mat
    bsdf.inputs['Base Color'].default_value = (m.kd[0], m.kd[1], m.kd[2], 1.0)
    # matte PS2 look: no metal, fully rough
    try:
        bsdf.inputs['Metallic'].default_value = 0.0
        bsdf.inputs['Roughness'].default_value = 1.0
    except Exception as exc:
        log('[YOBJ] material %-12s could not set matte defaults (%s)' % (m.name, exc))
    try:
        bsdf.inputs['Specular Tint'].default_value = (m.ks[0], m.ks[1], m.ks[2], 1.0)
    except Exception:
        pass   # socket got renamed or retyped in some version, not worth dying over

    base_node = None
    if tex_path:
        base_node = tree.nodes.new('ShaderNodeTexImage')
        base_node.image = plan['img']
        base_node.extension = 'EXTEND'      # the GS clamps every draw, see the top comment
        base_node.label = 'yobj_base'
        base_node.location = (-300, 300)
        tree.links.new(base_node.outputs['Color'], bsdf.inputs['Base Color'])
        log('[YOBJ] material %-12s [%s] texture %s' % (m.name, role, tex_path), 2)

        # Alpha rules per plan (try/except: a shader hiccup must never stop the meshes from loading)
        try:
            if atok.startswith('cut'):
                thr = float(atok[3:])
                add_alpha_cutout(tree, base_node, bsdf, thr)
                enable_transparency(mat, True)
                log('[YOBJ] material %-12s %s: alpha > %.2f cutout' % (m.name, role.lower(), thr))
            elif atok.startswith('link'):
                tree.links.new(base_node.outputs['Alpha'], bsdf.inputs['Alpha'])
                enable_transparency(mat, atok == 'linkB')
                info = plan['alpha_info']
                log('[YOBJ] material %-12s %s alpha linked (%s%s)' % (
                    m.name, role.lower(), 'cutout' if atok == 'linkB' else 'soft',
                    ', range %.2f..%.2f' % (info[2], info[3]) if info else ''))
            elif role == 'HAIR_BODY':
                log('[YOBJ] material %-12s hair body: opaque, like the PS2 (alpha is the sheen mask)' % m.name)
            else:
                log('[YOBJ] material %-12s opaque, alpha ignored (shine mask / no see-through pixels)' % m.name, 2)
            if plan['spec']:
                add_cloth_spec_mask(tree, base_node, bsdf, m.ks[0])
                log('[YOBJ] material %-12s EXPERIMENTAL: alpha -> specular level + roughness' % m.name)
        except Exception as exc:
            log('[YOBJ] material %-12s alpha setup failed (%s), leaving alpha unconnected' % (m.name, exc))
    else:
        log('[YOBJ] material %-12s no texture file found (looked for %s.tga)' % (m.name, m.name))
    if plan['l2_path']:
        try:
            add_sheen_layer(mat, tree, bsdf, base_node, m, plan['l2_path'], opts, log)
        except Exception as exc:
            log('[YOBJ] material %-12s sheen layer failed (%s), left out' % (m.name, exc))
    elif m.layer2_name and plan['ident'].split('|')[3] == 's=-' and sheen_wanted(m, opts):
        log('[YOBJ] material %-12s sheen texture %r not found' % (m.name, m.layer2_name))
    return mat


def build_materials(model, opts, log):
    """One Blender material per distinct (texture content, node graph). Returns {material_key: material}.

    A new material is only created when nothing in the file already matches its ident, so a
    batch of files that all use ribon.tga ends with ONE ribon material, and re-importing later
    reuses it. Every decision is logged: which existing material was reused (and how it was
    recognised) or, when a new one is made, how it differs from the closest existing one."""
    mats = {}
    assign_roles(model)
    if not opts.import_materials:
        return mats
    search = [opts.texture_dir, detect_texture_dir(model.path), os.path.dirname(model.path)]
    existing = {}
    if opts.reuse_materials:
        try:
            existing = collect_existing_materials(log)
        except Exception as exc:
            log('[YOBJ] materials: could not scan existing materials (%s), making fresh ones' % exc)
            existing = {}

    for part in model.parts:
        for m in part.materials:
            key = material_key(m, opts)
            if key in mats:
                continue
            plan = plan_material(m, key, opts, search)
            ident = plan['ident']
            hit = existing.get(ident)
            if hit is not None:
                mats[key] = hit[0]
                log('[YOBJ] material %-12s [%s] reusing %r (%s ident match)' % (m.name, plan['role'], hit[0].name, hit[1]))
                continue
            near = [k for k in existing if k.split('|')[1] == ident.split('|')[1]]
            if near:
                log('[YOBJ] material %-12s [%s] creating new: same texture as %r but %s' % (
                    m.name, plan['role'], existing[near[0]][0].name, ident_diff(near[0], ident)))
            else:
                log('[YOBJ] material %-12s [%s] creating new (no existing material uses this texture content)' % (
                    m.name, plan['role']), 2)
            log('         ident %s' % ident, 2)
            try:
                mat = build_material(plan, m, opts, log)
            except Exception as exc:
                # last resort: a bare material so the mesh still gets built
                log('[YOBJ] material %-12s build failed (%s), using a plain material' % (m.name, exc))
                mat = bpy.data.materials.new(plan['name'])
            mats[key] = mat
            if opts.reuse_materials:
                existing[ident] = (mat, 'this import')
    return mats


def _heat(t):
    t = max(0.0, min(1.0, t))
    return (t, 0.0, 1.0 - t, 1.0)       # blue -> red


# ---- Arena Mode: C351.dat -------------------------------------------------------------------
# C351.dat is a plain text table (Shift-JIS). One line per mesh: "<number> <flags...> ;comment".
# The number is the yobj file number in DECIMAL, so 0041.yobj (hex 0x41) is line 65.
# Single-letter flags we care about (legend is at the bottom of the file itself):
#   r = also draw a copy flipped across the YZ plane (X -> -X)      y = also draw a copy turned 180 deg about Y
#   s / k = normal alpha blend    a = additive blend    z = doesn't write Z
#   e = only shown during the entrance event     c = excluded (the Japanese is "nuki", left out)
# Tokens with brackets like l(125) / m(10003) are light / motion ids, we just keep them as text.

C351_FLAGS = set('sakzrynegwc')


def parse_c351(path):
    """Read C351.dat -> {line number: {'flags': set of letters, 'extra': [tokens], 'note': str}}."""
    out = {}
    with open(path, 'rb') as f:
        text = f.read().decode('cp932', errors='replace')
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(';'):
            continue
        body, _, note = line.partition(';')
        toks = body.split()
        if not toks or not toks[0].isdigit():
            continue
        info = out.setdefault(int(toks[0]), {'flags': set(), 'extra': [], 'note': note.strip()})
        for t in toks[1:]:
            if t in C351_FLAGS:
                info['flags'].add(t)
            else:
                info['extra'].append(t)       # l(125), m(10003), lg(129) ...
    return out


def find_c351(any_path):
    """Look for C351.dat next to the yobj, then one folder up. Returns a path or ''."""
    here = Path(any_path).resolve().parent
    for folder in (here, here.parent):
        try:
            for fn in os.listdir(str(folder)):
                if fn.lower() == 'c351.dat':
                    return os.path.join(str(folder), fn)
        except OSError:
            pass
    return ''


def c351_info(model, opts):
    """C351 entry for this model's file, or None. 0041.yobj -> int('0041', 16) -> line 65."""
    if not opts.c351_flags:
        return None
    stem = Path(model.path).stem
    try:
        return opts.c351_flags.get(int(stem, 16))
    except ValueError:
        return None


def variant_material(mat, tag):
    """Get or make a per-mesh variant of `mat` (tag: 'rep', 'c351blend', 'c351add' + optional 'z').
    Variants are found again by name + ident, so repeat imports reuse them instead of piling up
    copies. Returns (material, is_new)."""
    name = '%s.%s' % (mat.name.split('.' + tag)[0], tag)
    ident = '%s|%s' % (tag, mat.get('yobj_ident', mat.name))
    old = bpy.data.materials.get(name)
    if old is not None and old.get('yobj_ident') == ident:
        return old, False
    new = mat.copy()
    new.name = name
    new['yobj_ident'] = ident          # unique, so the reuse system never mistakes it for the plain one
    new['yobj_variant'] = tag
    return new, True


def _base_tex(tree):
    return next((n for n in tree.nodes if n.type == 'TEX_IMAGE' and n.label == 'yobj_base'), None) if tree else None


def alpha_is_linked(mat):
    bsdf = find_principled(mat.node_tree) if mat.node_tree else None
    return bool(bsdf and bsdf.inputs['Alpha'].is_linked)


def c351_material(mat, kind, no_zwrite, log):
    """Variant of `mat` for alpha blend ('blend') or additive ('add')."""
    new, fresh = variant_material(mat, 'c351' + kind + ('z' if no_zwrite else ''))
    if not fresh:
        return new
    tree = new.node_tree
    bsdf = find_principled(tree) if tree else None
    tex = _base_tex(tree)
    try:
        if bsdf is not None:
            if tex is not None and not bsdf.inputs['Alpha'].is_linked:
                tree.links.new(tex.outputs['Alpha'], bsdf.inputs['Alpha'])
            if kind == 'add':
                # normal Principled setup PLUS emission on top; the texture alpha is both the
                # real alpha and the emission strength, so dark / empty texels add nothing
                if tex is not None:
                    tree.links.new(tex.outputs['Color'], bsdf.inputs['Emission Color'])
                    tree.links.new(tex.outputs['Alpha'], bsdf.inputs['Emission Strength'])
                else:
                    bsdf.inputs['Emission Color'].default_value = bsdf.inputs['Base Color'].default_value
                    bsdf.inputs['Emission Strength'].default_value = 1.0
    except Exception as exc:
        log('[YOBJ] C351 material %s: node setup failed (%s)' % (new.name, exc))
    enable_transparency(new, False)
    if kind == 'add':
        for attr, val in (('surface_render_method', 'BLENDED'), ('blend_method', 'BLEND')):
            try:
                setattr(new, attr, val)
            except Exception:
                pass
    if no_zwrite:
        # 'z' flag: doesn't write depth. Blender's closest switch is "show backface"
        for attr in ('show_transparent_back', 'use_transparency_overlap'):
            try:
                setattr(new, attr, True)
            except Exception:
                pass
    return new


def repeat_slots(part):
    """Material indices of this part whose UVs leave 0..1. Arena meshes tile their textures (UVs
    run to 50 and below -20); EXTEND would smear those into streaks. The GS dump uses both CLAMP
    and REPEAT draws and the yobj has no field for it, so the UV range decides."""
    out = set()
    for tri, tm in zip(part.tri_uvs, part.tri_mat):
        if tm not in out and any(u < -0.02 or u > 1.02 or v < -0.02 or v > 1.02 for (u, v) in tri):
            out.add(tm)
    return out


def swap_slots(model, objs, per_part, make, log=None, what=''):
    """For each object (originals or copies, in part order) replace the material of every slot
    listed in per_part[part] with make(material). Returns how many slots changed."""
    n = len(model.parts)
    swapped = 0
    for i, obj in enumerate(objs):
        wanted = per_part[i % n]
        if not wanted:
            continue
        for si, slot in enumerate(obj.material_slots):
            if si in wanted and slot.material is not None:
                slot.material = make(slot.material)
                swapped += 1
    return swapped


def repeat_material(mat):
    new, fresh = variant_material(mat, 'rep')
    if fresh and new.node_tree:
        for n in new.node_tree.nodes:
            if n.type == 'TEX_IMAGE':
                n.extension = 'REPEAT'
    return new


# ---- Crowd (0.9.5) ------------------------------------------------------------------------------
# What the table files look like (526C.dat / 526F.dat, Shift-JIS text):
#
#     A (40,  90,  9.3, 0)          seat group "A": starts at x=40, y=90, and every next seat is
#                                   9.3 further in x, 0 further in y.  (Checked against the chair
#                                   meshes: x and y here are plain world X and Z of the yobj files.)
#     [0] 10                        crowd type 0, 10 people per texture (A, B, C = 30 total)
#     A00(01)                       seat A00 gets one of those people. "(01)" is an extra number
#                                   I can't decode yet, it's stored on the object and nothing else.
#
# Type N is crowd file 4E20 + N. The file's own model name can't be trusted for this (arena 0009
# has "050big" sitting in 4E22), the file number can. Texture names: the crowd file's own material
# is e.g. "aud3d_02b", so type 1 with texture C wants "aud3d_02c".

CROWD_ID_FIRST = 0x4E20
CROWD_ID_LAST = 0x4E2F

_CROWD_SEAT_RE = re.compile(r'^([A-Za-z])\s*\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)')
_CROWD_TYPE_RE = re.compile(r'^\[(\d+)\]\s*(\d+)')
_CROWD_ENTRY_RE = re.compile(r'^([A-Za-z])(\d+)(?:\((\d+)\))?$')
# "texture type" in Japanese, followed by A / B / C. Written as escapes so the .py stays plain ASCII
_CROWD_TEX_RE = re.compile('\u30c6\u30af\u30b9\u30c1\u30e3\u30bf\u30a4\u30d7\\s*([A-Za-z])')


def parse_crowd_dat(path):
    """Read 526C.dat / 526F.dat.
    Returns {'seats': {letter: (x, y, dx, dy)},
             'types': {type number: {'per_tex': int, 'entries': [(letter, number, variant or None, tex letter)]}}}"""
    with open(path, 'rb') as f:
        text = f.read().decode('cp932', errors='replace')
    seats, types = {}, {}
    cur = None              # the [n] block we're inside right now
    tex = None              # texture letter from the last "texture type X" comment
    in_block = False
    for raw in text.splitlines():
        body, _, comment = raw.partition(';')
        body = body.strip()
        # the A/B/C marker lives in a comment, sometimes on the entry's own line, sometimes alone
        mt = _CROWD_TEX_RE.search(comment)
        if mt and cur is not None:
            tex = mt.group(1).upper()
        if not body:
            continue
        if body == '{':
            in_block = True
            continue
        if body == '}':
            in_block = False
            cur, tex = None, None
            continue
        mh = _CROWD_TYPE_RE.match(body)
        if mh:
            cur = int(mh.group(1))
            types[cur] = {'per_tex': int(mh.group(2)), 'entries': []}
            tex = None
            continue
        if not in_block:
            continue
        if cur is None:
            ms = _CROWD_SEAT_RE.match(body)
            if ms:
                seats[ms.group(1)] = tuple(float(ms.group(i)) for i in range(2, 6))
            continue
        me = _CROWD_ENTRY_RE.match(body)
        if me:
            entries = types[cur]['entries']
            t = tex
            if t is None:
                # no marker seen: first block of per_tex people is A, next is B, then C
                per = max(1, types[cur]['per_tex'])
                t = 'ABC'[min(2, len(entries) // per)]
            variant = int(me.group(3)) if me.group(3) is not None else None
            entries.append((me.group(1), int(me.group(2)), variant, t))
    return {'seats': seats, 'types': types}


def crowd_plan(data, log=None):
    """Turn the parsed table into {type: [slot, ...]} with real positions.
    slot = {'seat': 'A00', 'tex': 'B', 'variant': 1 or None, 'x': .., 'z': ..}
    A seat that two entries claim is given to whoever asked first (0000 has 'h11' listed twice)."""
    plan, taken, clashes, unknown = {}, set(), 0, 0
    for t in sorted(data['types']):
        slots = []
        for (letter, num, variant, tex) in data['types'][t]['entries']:
            if letter not in data['seats']:
                unknown += 1
                continue
            if (letter, num) in taken:
                clashes += 1
                continue
            taken.add((letter, num))
            x0, y0, dx, dy = data['seats'][letter]
            slots.append({'seat': '%s%02d' % (letter, num), 'tex': tex, 'variant': variant,
                          'x': x0 + num * dx, 'z': y0 + num * dy})
        plan[t] = slots
    if log:
        log('[YOBJ] crowd: %d seat(s) planned over %d type(s)%s%s' % (
            len(taken), sum(1 for s in plan.values() if s),
            ', %d double-booked seat(s) skipped' % clashes if clashes else '',
            ', %d entr(ies) used a seat letter with no definition' % unknown if unknown else ''))
    return plan


def find_crowd_dat(any_path, density='C'):
    """526C.dat (or 526F.dat) beside the yobj, then one folder up. If the wanted one is missing the
    other one is better than nothing. Returns a path or ''."""
    wanted = ['526%s.dat' % density] + [n for n in ('526c.dat', '526f.dat') if n != '526%s.dat' % density.lower()]
    here = Path(any_path).resolve().parent
    for folder in (here, here.parent):
        try:
            names = {fn.lower(): fn for fn in os.listdir(str(folder))}
        except OSError:
            continue
        for w in wanted:
            if w.lower() in names:
                return os.path.join(str(folder), names[w.lower()])
    return ''


def crowd_type_index(model):
    """Crowd type number for a model (4E20 -> 0, 4E21 -> 1 ...), None if it isn't a crowd file."""
    try:
        n = int(Path(model.path).stem, 16)
    except ValueError:
        return None
    return n - CROWD_ID_FIRST if CROWD_ID_FIRST <= n <= CROWD_ID_LAST else None


def crowd_name_type(model_name):
    """What the model's own name claims: '030big01' -> 2, 'm03' -> 8. None if it doesn't fit."""
    m = re.match(r'^(m?)(\d\d)', model_name or '')
    if not m:
        return None
    return int(m.group(2)) - 1 + (6 if m.group(1) else 0)


def crowd_pose(model, slot, opts):
    """(position, yaw) of one crowd member, both in importer space (Y is up, before the Z-up turn).
    The seat gives X and Z, the height comes from the option. Going through world_pos() means the
    crowd gets exactly the same flips as every other mesh in the arena."""
    px, _, pz = world_pos(model, (slot['x'], 0.0, slot['z']))
    # height is just the Hip Height option (floor assumed at 0)
    py = opts.crowd_hip_height
    yaw = 0.0
    if opts.crowd_face_ring:
        # atan2(-px, -pz) would point local +Z at the ring. In-game test: that came out backwards,
        # the models really look down -Z, so we add half a turn
        yaw = math.atan2(-px, -pz) + math.pi
    return (px, py, pz), yaw + math.radians(opts.crowd_yaw_offset)


def crowd_texture_variant(rec_name, base_mat, letter, search, log, cache):
    """The base material, but wearing texture A / B / C. Falls back to the base material when the
    texture file can't be found (logged once)."""
    key = (rec_name, base_mat.name, letter)
    if key in cache:
        return cache[key]
    result = base_mat
    stem = re.sub(r'[A-Za-z]$', '', rec_name)            # aud3d_02b -> aud3d_02
    tex_name = stem + letter.lower()
    path = find_texture(tex_name, search)
    if not path:
        log('[YOBJ] crowd: texture %s not found, those members keep %s' % (tex_name, rec_name))
    else:
        base_ident = base_mat.get('yobj_ident', '')
        parts = base_ident.split('|')
        h = file_hash(path)
        if len(parts) >= 3:
            parts[1] = h
            ident = '|'.join(parts)
        else:
            ident = 'crowd|%s' % h
        if base_ident == ident:
            result = base_mat                            # it already IS this texture
        else:
            hit = next((m for m in bpy.data.materials if m.get('yobj_ident') == ident), None)
            if hit is not None:
                result = hit                             # an earlier import made this one
            else:
                new = base_mat.copy()
                new.name = tex_name
                new['yobj_ident'] = ident
                new['yobj_crowd'] = True
                if new.node_tree:
                    for n in new.node_tree.nodes:
                        if n.type == 'TEX_IMAGE' and n.label == 'yobj_base':
                            n.image = load_image(path)
                result = new
                log('[YOBJ] crowd: made material %s from %s' % (new.name, path), 2)
    cache[key] = result
    return result


def clear_old_crowd(idx):
    """A re-import shouldn't stack a second crowd on top of the first, so delete the earlier
    objects of this type first (they're tagged when they're made)."""
    gone = [o for o in bpy.data.objects if o.get('yobj_crowd_type') == idx]
    for o in gone:
        bpy.data.objects.remove(o, do_unlink=True)
    return len(gone)


def import_crowd_model(model, idx, opts, log):
    """Build one crowd file as a hidden template, then one linked copy per seat that the table
    gives this type. Everything lands in the top level collection 'Crowd'."""
    crowd = dynamic_collection('Crowd')
    removed = clear_old_crowd(idx)
    if removed:
        log('[YOBJ] crowd: removed %d object(s) from an earlier import of type %d' % (removed, idx))
    claimed = crowd_name_type(model.model_name)
    if claimed is not None and claimed != idx:
        log('[YOBJ] crowd: %s is called %r, which says type %d, but the file number says %d. Going with the file number.' % (
            Path(model.path).name, model.model_name, claimed, idx))

    mats = build_materials(model, opts, log)
    bone_names = {i: b.name for i, b in enumerate(model.bones)}
    root = root_matrix(opts)

    # ---- templates: the normal mesh build, minus the armature (the crowd is static) ----
    templates = []
    for part in model.parts:
        obj = build_part_mesh(model, part, None, bone_names, mats, opts, log, crowd)
        for mod in list(obj.modifiers):
            obj.modifiers.remove(mod)
        obj.vertex_groups.clear()
        obj.name = 'crowd_template_%s_part%02d' % (model.model_name, part.index)
        obj.matrix_world = root
        obj['yobj_crowd_type'] = idx
        obj['yobj_crowd_template'] = True
        templates.append(obj)

    slots = opts.crowd_data['plan'].get(idx, [])
    search = [opts.texture_dir, detect_texture_dir(model.path), os.path.dirname(model.path)]
    cache = {}
    for s in slots:
        pos, yaw = crowd_pose(model, s, opts)
        mtx = root @ Matrix.Translation(Vector(pos)) @ Matrix.Rotation(yaw, 4, 'Y')
        for tpl, part in zip(templates, model.parts):
            inst = bpy.data.objects.new('crowd_%d_%s' % (idx, s['seat']) + ('_p%d' % part.index if len(model.parts) > 1 else ''), tpl.data)
            crowd.objects.link(inst)
            inst.matrix_world = mtx
            inst['yobj_crowd_type'] = idx
            inst['yobj_crowd_seat'] = s['seat']
            inst['yobj_crowd_tex'] = s['tex']
            if s['variant'] is not None:
                inst['yobj_crowd_variant'] = s['variant']     # meaning unknown, kept for later
            if opts.import_materials:
                for si, rec in enumerate(part.materials):
                    base = tpl.data.materials[si] if si < len(tpl.data.materials) else None
                    if base is None:
                        continue
                    mat = crowd_texture_variant(rec.name, base, s['tex'], search, log, cache)
                    if mat is not base:
                        slot = inst.material_slots[si]
                        slot.link = 'OBJECT'            # only this copy gets the other texture
                        slot.material = mat

    # the templates would just sit in the middle of the ring as a pile, so hide them
    for tpl in templates:
        tpl.hide_render = True
        try:
            tpl.hide_set(True)
        except Exception:
            tpl.hide_viewport = True
    log('[YOBJ] crowd: %s (type %d, %r) -> %d seat(s) filled' % (Path(model.path).name, idx, model.model_name, len(slots)))
    if not slots:
        log('[YOBJ] crowd: the table has no seats for type %d, only the hidden template was made' % idx)
    return None


def dynamic_collection(name):
    """Get-or-make a top level collection (visible, nothing is hidden for now)."""
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    return coll


def move_objects(objs, coll):
    for obj in objs:
        for old in list(obj.users_collection):
            old.objects.unlink(obj)
        coll.objects.link(obj)


def build_part_mesh(model, part, arm_obj, bone_names, mats, opts, log, collection, variant=''):
    """variant: '' = the mesh as stored, 'MIRROR' = X flipped copy (winding reversed so it still
    faces outward), 'ROT180' = copy turned 180 degrees about the up axis (a proper rotation)."""
    name = '%s_part%02d' % (model.model_name, part.index)
    if variant == 'MIRROR':
        name += '_mirror'
    elif variant == 'ROT180':
        name += '_rot180'
    verts = [world_pos(model, p) for p in part.positions]    # flip(s) live in one place
    # importer space has Y as the up axis of the mesh (the armature object does the Z-up turn)
    if variant == 'MIRROR':
        verts = [(-x, y, z) for (x, y, z) in verts]
    elif variant == 'ROT180':
        verts = [(-x, y, -z) for (x, y, z) in verts]
    tris = [list(t) for t in part.tris]
    if variant == 'MIRROR':
        tris = [[t[0], t[2], t[1]] for t in tris]       # a reflection flips winding, so flip it back

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], tris)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)

    # ---- materials: one slot per material record in this part ----
    for m in part.materials:
        mesh.materials.append(mats.get(material_key(m, opts)))
    if len(mesh.polygons) == len(part.tris):
        mesh.polygons.foreach_set('material_index', part.tri_mat)

        # ---- UVs: one per corner, (u, 1-v). Loops come out in face order. ----
        if part.tris:
            for si, set_uvs in enumerate(part.uv_sets):
                uv_layer = mesh.uv_layers.new(name='UVMap_%d' % si)
                flat = []
                for tri in set_uvs:
                    if variant == 'MIRROR':
                        tri = (tri[0], tri[2], tri[1])  # same corner order change as the triangles
                    for (u, v) in tri:
                        flat += ((1.0 - u) if opts.flip_u else u, 1.0 - v)
                uv_layer.data.foreach_set('uv', flat)

        # ---- smoothing + custom split normals (Blender 4.1+/5.0 way, no auto smooth) ----
        mesh.polygons.foreach_set('use_smooth', [True] * len(mesh.polygons))
        if opts.use_custom_normals and verts:
            normals = []
            for fn in part.normals:
                nx, ny, nz = world_normal(model, fn)
                if variant == 'MIRROR':
                    nx = -nx
                elif variant == 'ROT180':
                    nx, nz = -nx, -nz
                n = Vector((nx, ny, nz))
                if n.length > 1e-8:
                    n.normalize()
                normals.append(n)
            try:
                mesh.normals_split_custom_set_from_vertices(normals)
            except Exception as exc:
                log('[YOBJ] part %02d: custom normals failed (%s), leaving them auto' % (part.index, exc))
    else:
        log('[YOBJ] part %02d: polygon count %d != expected %d, skipped UVs/normals/materials' % (
            part.index, len(mesh.polygons), len(part.tris)))

    # ---- vertex groups + weights ----
    # Collect first: {(bone, weight): [vertex ids]}. Rigid meshes (all weight 1.0) then cost
    # one vg.add call per bone instead of one per vertex. Groups are still created in the
    # order they are first met, so the group list looks exactly like before.
    groups = {}
    buckets = {}
    total_assigned = 0
    for vi, wd in enumerate(part.weights):
        for b, w in wd.items():
            if b not in groups:
                groups[b] = obj.vertex_groups.new(name=bone_names[b])
            buckets.setdefault((b, w), []).append(vi)
            total_assigned += 1
    for (b, w), verts in buckets.items():
        groups[b].add(verts, w, 'REPLACE')
    log('[YOBJ] part %02d: %d weight entries across %d vertex groups' % (
        part.index, total_assigned, len(groups)), 2)
    if 0 in groups:
        log('[YOBJ] part %02d: root bone (Bone 0 "%s") holds weight on %d verts' % (
            part.index, bone_names[0], part.stats.get('bone0_verts', 0)), 1 if part.stats.get('bone0_heavy') else 2)

    # ---- debug attributes ----
    if opts.debug_attributes and verts:
        for attr, kind, vals in (
                ('yobj_group', 'INT', [i.group for i in part.vinfo]),
                ('yobj_infl', 'INT', [i.infl for i in part.vinfo]),
                ('yobj_part_local_index', 'INT', list(range(len(verts)))),
                ('yobj_weight_sum', 'FLOAT', [i.raw_sum for i in part.vinfo])):   # raw sum, pre-normalize
            a = mesh.attributes.new(name=attr, type=kind, domain='POINT')
            a.data.foreach_set('value', vals)
    if opts.debug_heatmaps and verts:
        bones = model.bones
        root_cols, dist_cols = [], []
        for (pos, wd) in zip(verts, part.weights):
            root_cols += _heat(wd.get(0, 0.0))
            top = max(wd, key=wd.get)
            dist_cols += _heat(math.dist(pos, bones[top].head) / DIST_HEAT_RANGE)
        for attr, cols in (('yobj_root_weight', root_cols), ('yobj_bone_distance', dist_cols)):
            ca = mesh.color_attributes.new(name=attr, type='FLOAT_COLOR', domain='POINT')
            ca.data.foreach_set('color', cols)

    # ---- parent + armature modifier ----
    # Vertices are already in rest-pose armature space, so the parent inverse stays identity (spec 2.4).
    obj.parent = arm_obj
    mod = obj.modifiers.new('Armature', 'ARMATURE')
    mod.object = arm_obj
    obj['yobj_part_index'] = part.index
    return obj


def build_bone_empties(model, arm_obj, collection):
    """Little plain-axes empties at each bone's global_pos for eyeballing."""
    for b in model.bones:
        e = bpy.data.objects.new('dbg_%s' % b.name, None)
        e.empty_display_type = 'PLAIN_AXES'
        e.empty_display_size = 0.02
        e.location = b.head
        e.parent = arm_obj
        collection.objects.link(e)


def target_collection(model, opts):
    """Where this file's objects go: a fresh collection named after the model, or the active one."""
    if opts.new_collection:
        coll = bpy.data.collections.new(model.model_name)
        bpy.context.scene.collection.children.link(coll)
        return coll
    return bpy.context.view_layer.active_layer_collection.collection


def arena_post_process(model, objs, arm_obj, bone_names, mats, opts, log, coll):
    """Everything Arena Mode adds on top of the normal build. Returns originals + copies."""
    parts = model.parts
    info = c351_info(model, opts)
    flags = info['flags'] if info else set()
    if info:
        log('[YOBJ] C351 %s: flags %s %s %s' % (Path(model.path).name, ' '.join(sorted(flags)) or '-',
                                               ' '.join(info['extra']), info['note']))
    elif opts.c351_flags:
        log('[YOBJ] C351: no line for %s (drawn once, no flags)' % Path(model.path).name)
    tag = ' '.join(sorted(flags)) + ' ' + ' '.join(info['extra']) if info else ''

    # 1. copies the mirror data asks for
    all_objs = list(objs)
    if opts.mirror_c351:
        for letter, variant in (('r', 'MIRROR'), ('y', 'ROT180')):
            if letter in flags:
                copies = [build_part_mesh(model, p, arm_obj, bone_names, mats, opts, log, coll, variant)
                          for p in parts]
                for o in copies:
                    o['yobj_c351_copy'] = variant
                all_objs += copies
                log('[YOBJ] C351 %r flag: added %d %s copy object(s)' % (letter, len(copies), variant.lower()))
    for o in all_objs:
        if tag:
            o['yobj_c351_flags'] = tag

    # 2. materials (same swap for originals and copies, so they share variants)
    if opts.import_materials:
        rep = [repeat_slots(p) for p in parts]
        n = swap_slots(model, all_objs, rep, repeat_material)
        if n:
            log('[YOBJ] arena UVs: %d material slot(s) use REPEAT (UVs outside 0..1)' % n)
        kind = 'add' if 'a' in flags else ('blend' if flags & {'s', 'k'} else '')
        if opts.alpha_c351 and kind:
            nz = 'z' in flags
            everything = [set(range(len(p.materials))) for p in parts]
            def make(m):
                # a texture-alpha material (found by the alpha scan) is already blended
                if kind == 'blend' and not nz and alpha_is_linked(m):
                    return m
                return c351_material(m, kind, nz, log)
            swap_slots(model, all_objs, everything, make)

    # 3. collections
    if opts.dynamic_collections:
        if 'c' in flags:
            move_objects(all_objs, dynamic_collection('YOBJ Excluded (c)'))
        elif 'e' in flags:
            move_objects(all_objs, dynamic_collection('YOBJ Events Only (e)'))
    return all_objs


def join_part_objects(model, objs, log):
    """Join the per-part mesh objects of one file into a single object (Join Parts option).

    Mirror / rotated copies from the C351 flags stay separate from the originals, so you get
    one object for the original and one per copy. Vertex groups, materials, UVs and the
    armature link all survive Blender's own join. If anything goes wrong the parts are
    simply left as they were.
    """
    groups = {}
    for o in objs:
        if o.type == 'MESH':
            groups.setdefault(o.get('yobj_c351_copy', ''), []).append(o)
    result = []
    for variant, group in groups.items():
        if len(group) < 2:
            result += group
            continue
        keep = group[0]
        try:
            for o in list(bpy.context.view_layer.objects.selected):
                o.select_set(False)
            for o in group:
                o.select_set(True)
            bpy.context.view_layer.objects.active = keep
            bpy.ops.object.join()
        except Exception as exc:
            log('[YOBJ] join parts failed for %s (%s), leaving the parts separate' % (model.model_name, exc))
            result += group
            continue
        suffix = {'MIRROR': '_mirror', 'ROT180': '_rot180'}.get(variant, '')
        keep.name = model.model_name + suffix
        keep.data.name = keep.name
        if 'yobj_part_index' in keep:
            del keep['yobj_part_index']      # meaningless once the parts are one mesh
        result.append(keep)
        log('[YOBJ] join parts: %d part object(s) -> %r' % (len(group), keep.name))
    return result


def import_model(model, opts, log):
    """Build everything for one already-parsed model. Skeleton is merged or created first."""
    log_model(model, log)
    # crowd files take their own road: no armature, no per-file collection, many copies
    if opts.mode == 'ARENA' and opts.place_crowd and opts.crowd_data:
        crowd_idx = crowd_type_index(model)
        if crowd_idx is not None:
            return import_crowd_model(model, crowd_idx, opts, log)
    coll = target_collection(model, opts)
    arm_obj, bone_names = attach_skeleton(model, opts, log, coll)
    mats = build_materials(model, opts, log)
    objs = [build_part_mesh(model, part, arm_obj, bone_names, mats, opts, log, coll)
            for part in model.parts]
    if opts.mode == 'ARENA':
        objs = arena_post_process(model, objs, arm_obj, bone_names, mats, opts, log, coll)
    if opts.join_parts:
        objs = join_part_objects(model, objs, log)
    if opts.debug_bone_empties:
        dbg = bpy.data.collections.new(model.model_name + '_debug')
        coll.children.link(dbg)
        build_bone_empties(model, arm_obj, dbg)

    # post-import diagnostic: do duplicate seam vertices agree on weights?
    if opts.seam_check:
        log_seam_report(model, seam_check(model, tol=opts.seam_tolerance), log)
    log('[YOBJ] done: %s -> %d mesh objects on armature %r' % (model.model_name, len(objs), arm_obj.name))
    return arm_obj


def run_import_batch(filepaths, opts, log=None):
    """Import one or many files. Returns (models, failures[(path, error)]).

    With several files the one with the most bones is built first and becomes the main
    armature, everything else merges into it. With a single file the same merge logic
    looks for an armature already sitting in the scene (body first, hair later works).
    """
    log = log or ImportLog(opts.log_level)
    models, failures = [], []
    for path in filepaths:
        try:
            models.append(parse_yobj(path, bone_base=opts.bone_base, lane_order=opts.lane_order,
                                     mirror_x=(opts.handedness == 'BAKE')))
        except Exception as exc:
            log('[YOBJ] could not parse %s: %s' % (path, exc))
            failures.append((path, str(exc)))
    if not models:
        return [], failures
    if opts.mode_reason:
        log('[YOBJ] mode: %s (%s)' % (opts.mode, opts.mode_reason))

    opts.c351_flags = None
    if opts.mode == 'ARENA' and (opts.mirror_c351 or opts.alpha_c351 or opts.dynamic_collections):
        c351 = find_c351(models[0].path)
        if c351:
            try:
                opts.c351_flags = parse_c351(c351)
                log('[YOBJ] C351: read %d line(s) from %s' % (len(opts.c351_flags), c351))
            except Exception as exc:
                log('[YOBJ] C351: could not read %s (%s), C351 options skipped' % (c351, exc))
        else:
            log('[YOBJ] C351: C351.dat not found beside the yobj files (or one folder up), C351 options skipped')

    # ---- crowd table (Place Crowd) ----
    opts.crowd_data = None
    if opts.mode == 'ARENA' and opts.place_crowd:
        crowd_models = [m for m in models if crowd_type_index(m) is not None]
        if not crowd_models:
            log('[YOBJ] crowd: Place Crowd is on, but there are no 4E20..4E2F crowd files in this import')
        else:
            dat = find_crowd_dat(crowd_models[0].path, opts.crowd_density)
            if not dat:
                log('[YOBJ] crowd: no 526C.dat / 526F.dat beside the yobj files (or one folder up), crowd files import as they are')
            else:
                try:
                    data = parse_crowd_dat(dat)
                    data['plan'] = crowd_plan(data, log)
                    data['path'] = dat
                    opts.crowd_data = data
                    log('[YOBJ] crowd: read %s (%d seat group(s), %d type(s))' % (dat, len(data['seats']), len(data['types'])))
                except Exception as exc:
                    log('[YOBJ] crowd: could not read %s (%s), crowd files import as they are' % (dat, exc))

    if opts.merge_skeletons and len(models) > 1:
        main = pick_main(models)
        # main armature first, then the rest biggest skeleton first (bones a big file adds can
        # be the anchor a smaller one needs)
        rest = sorted((m for i, m in enumerate(models) if i != main), key=lambda m: -len(m.bones))
        order = [models[main]] + rest
        log('[YOBJ] batch of %d files, main armature source: %s (%d bones)' % (
            len(models), Path(models[main].path).name, len(models[main].bones)))
    else:
        order = models

    if opts.import_materials:
        tdir = opts.texture_dir or (detect_texture_dir(order[0].path) if order else '')
        if tdir and os.path.isdir(tdir):
            n = sum(1 for fn in os.listdir(tdir) if fn.lower().endswith('.tga'))
            log('[YOBJ] texture folder: %s  (%d .tga file(s))' % (tdir, n))
        else:
            log('[YOBJ] texture folder: none found (no Texture Folder set, no "textures" folder beside the yobj)')

    for model in order:
        log('[YOBJ] ---- importing %s ----' % model.path)
        try:
            import_model(model, opts, log)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            failures.append((model.path, str(exc)))

    if opts.write_log_text and order:
        # debug log goes next to the first yobj as <name>_import_log.txt
        stem = os.path.splitext(order[0].path)[0]
        try:
            with open(stem + '_import_log.txt', 'w', encoding='utf-8') as f:
                f.write('\n'.join(log.lines) + '\n')
            print('[YOBJ] log written to %s_import_log.txt' % stem)
        except OSError as exc:
            print('[YOBJ] could not write log txt (%s)' % exc)
    return order, failures


def run_import(filepath, opts, log=None):
    """Single-file convenience wrapper (handy from the Python console)."""
    models, failures = run_import_batch([filepath], opts, log)
    if failures:
        raise RuntimeError(failures[0][1])
    return models[0]


# =============================================================================
# 4. OPERATOR / UI
# =============================================================================

if bpy is not None:
    from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty,
                           StringProperty)
    from bpy_extras.io_utils import ImportHelper

    class IMPORT_OT_yobj(bpy.types.Operator, ImportHelper):
        """Import Rumble Roses PS2 .yobj model(s)"""
        bl_idname = "import_scene.yobj"
        bl_label = "Import Rumble Roses YOBJ"
        bl_options = {'REGISTER', 'UNDO'}

        filename_ext = ".yobj"
        filter_glob: StringProperty(default="*.yobj", options={'HIDDEN'})
        # these two let you multi-select files in the browser
        files: CollectionProperty(type=bpy.types.OperatorFileListElement, options={'HIDDEN', 'SKIP_SAVE'})
        directory: StringProperty(subtype='DIR_PATH')
        # remembers what the auto-detect filled in, so it can be refreshed when you browse elsewhere
        auto_texture_dir: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

        # ---- main panel ----
        mode: EnumProperty(
            name="Mode", default='AUTO',
            items=[('AUTO', "Auto Detect", "Look at the files (and any C351/526x data beside them) and pick Character or Arena Mode"),
                   ('CHARACTER', "Character Mode", "Fighters, hair, clothing and accessories"),
                   ('ARENA', "Arena Mode", "Arena / stage files, with the options that only matter there")])
        import_materials: BoolProperty(name="Materials", default=True)
        texture_dir: StringProperty(name="Texture Folder", subtype='DIR_PATH', default="")
        axis: EnumProperty(name="Axis", default='Z_UP',
                           items=[('Z_UP', "Y-up to Z-up", "Rotate +90 degrees about X so models stand up"),
                                  ('NONE', "As in file", "No rotation")])
        handedness: EnumProperty(
            name="Mirror Fix", default='BAKE',
            items=[('BAKE', "Bake into data", "X flip baked into verts, bones and normals. No negative scale"),
                   ('OBJECT', "Object matrix", "Keep old Y-flip data, put X scale -1 on the armature object"),
                   ('NONE', "None", "Old behaviour (Y flip only)")])
        global_scale: FloatProperty(name="Scale", default=1.0, min=1e-6)
        join_parts: BoolProperty(
            name="Join Parts", default=False,
            description="Merge the parts of each file into one mesh object (one object per file instead of one per part)")
        new_collection: BoolProperty(
            name="Import into New Collection", default=True,
            description="On: each file gets its own collection named after the model. "
                        "Off: objects go straight into the active collection")

        # ---- expert ----
        bone_base: EnumProperty(
            name="Bone Index Base", default='1', options={'HIDDEN'},
            items=[('1', "1 (spec default)", "File bone indices are 1-based"),
                   ('0', "0", "Treat file bone indices as 0-based (A/B test for root leakage)")])
        lane_order: EnumProperty(
            name="4-Influence Lane Order", default='IDENTITY', options={'HIDDEN'},
            items=[('IDENTITY', "Identity", "Weight slot i -> bone slot i"),
                   ('ALT_0321', "0,3,2,1", "Best permutation seen on 0002"),
                   ('ALT_0213', "0,2,1,3", "Best permutation seen on 271B"),
                   ('REVERSE', "Reverse", "3,2,1,0")])
        bone_orient: EnumProperty(
            name="Bone Direction", default='FILE_AXES',
            items=[('FILE_AXES', "File axes", "Bones point down the PS2 local Y axis (faithful)"),
                   ('POINT_TO_CHILD', "Point to child", "Tail at child bones, roll from the file's Z axis")])
        tail_length: FloatProperty(name="Leaf Bone Length", default=0.1, min=0.001, soft_max=1.0)
        merge_skeletons: BoolProperty(
            name="Merge Skeletons", default=True,
            description="Reuse a matching armature already in the scene (or from this batch) and only "
                        "add the bones it doesn't have, instead of making a new armature per file")
        secondary_layers: BoolProperty(
            name="Sheen Layer (MatCap)", default=True,
            description="Build the optional 2nd texture layer some materials have (hair sheen, "
                        "clothing shine). The PS2 made its UVs from the surface normal, so it is built "
                        "as a MatCap lookup and added as emission, masked by the base texture's alpha")
        sheen_flip_v: BoolProperty(
            name="Flip Sheen V", default=False,
            description="Only if the sheen highlights look upside down")
        sheen_strength: FloatProperty(
            name="Sheen Strength", default=0.25, min=0.0, soft_max=2.0,
            description="How strong the sheen/reflection layer is. Lower it if clothes look too shiny")
        cloth_alpha_spec: BoolProperty(
            name="Cloth Alpha as Specular", default=True,
            description="Treat the alpha of shiny clothing textures as a specular/roughness "
                        "mask on the normal shader. For comparing against the sheen layer")
        hair_body_cutoff: FloatProperty(
            name="Hair Body Alpha Cutoff", default=-1.0, min=-1.0, max=1.0,
            description="-1 leaves the hair body fully opaque, which is what the PS2 does. "
                        "A value like 0.1 cuts away very transparent texels (the old behaviour)")
        reuse_materials: BoolProperty(
            name="Reuse Existing Materials", default=True,
            description="If a material this importer made earlier matches (same texture CONTENT, same "
                        "node setup), use it instead of creating name.001, name.002 ...")
        merge_tolerance: FloatProperty(
            name="Merge Tolerance", default=SKEL_POS_TOL, min=1e-6, max=0.5, precision=4,
            description="How close position and rotation must be for two bones with the same name to "
                        "count as the same bone when merging skeletons")
        hair_soft_fringe: BoolProperty(
            name="Soft Hair Fringe", default=False,
            description="Hair strands: plain alpha blend instead of the hard 0.44 cutout. The PS2 still "
                        "blends strand pixels below the threshold (it only skips Z/alpha writes)")
        sheen_mapping: EnumProperty(
            name="Sheen Lookup", default='CAMERA',
            items=[('CAMERA', "View-space normal", "Current best fit"),
                   ('REFLECT', "Reflection vector", "A/B test for clothing sheen")])
        sheen_alpha_scale: FloatProperty(
            name="Sheen Alpha Scale", default=1.0, min=0.0, soft_max=4.0,
            description="Multiplier on the base-alpha mask. 2.0 if the TGA alpha is raw PS2 (0x80 = 1.0)")
        sheen_blend: EnumProperty(
            name="Sheen Blend", default='EMISSION',
            items=[('EMISSION', "Linear emission", "Add the sheen as linear light"),
                   ('GAMMA_ADD', "sRGB-space add", "Emulate the GS additive blend in gamma space")])
        # ---- arena only (shown in Expert Settings when Mode = Arena) ----
        mirror_c351: BoolProperty(
            name="Mirror Data From C351", default=True,
            description="Read C351.dat and add the extra copies it asks for: 'r' = copy flipped across "
                        "the YZ plane (X mirrored), 'y' = copy rotated 180 degrees about Y")
        alpha_c351: BoolProperty(
            name="Transparency From C351", default=False,
            description="Use the C351 flags for alpha: 's'/'k' = alpha blend, 'a' = additive, "
                        "'z' = doesn't write depth. Works on copies of the materials")
        dynamic_collections: BoolProperty(
            name="Dynamic Collections", default=False,
            description="Move meshes flagged 'e' (event only) and 'c' (excluded) in C351 into their "
                        "own collections. Nothing is hidden yet")
        place_crowd: BoolProperty(
            name="Place Crowd", default=False,
            description="Read the seat table (526C.dat / 526F.dat) and fill the seats with copies of the "
                        "4E20..4E2B crowd files, re-textured, facing the ring, all inside one 'Crowd' "
                        "collection. Import the crowd files together with the rest of the arena")
        crowd_density: EnumProperty(
            name="Crowd Table", default='C',
            items=[('C', "526C (full)", "The full crowd table"),
                   ('F', "526F (lighter)", "The lighter table: fewer people, no front rows")])
        crowd_hip_height: FloatProperty(
            name="Hip Height", default=8.0, soft_min=0.0, soft_max=12.0,
            description="Height of the crowd model's middle (its hips) above the floor. About 6 sits them "
                        "on the chairs (their legs go into the floor), 10 stands them on the floor")
        crowd_face_ring: BoolProperty(
            name="Face The Ring", default=True,
            description="Turn every crowd member to look at the middle of the ring")
        crowd_yaw_offset: FloatProperty(
            name="Extra Turn", default=0.0, min=-180.0, max=180.0,
            description="Extra turn in degrees. Use 180 if everyone is looking away from the ring")
        use_custom_normals: BoolProperty(name="Use File Normals", default=True)
        flip_u: BoolProperty(name="Flip U", default=False,
                             description="Only if geometry looks right but textures read backwards")

        # ---- debug (everything below only matters when debug_mode is on) ----
        debug_mode: BoolProperty(name="Debug Mode", default=False,
                                 description="Turn on the diagnostic extras below")
        log_level: EnumProperty(name="Console Log", default='2',
                                items=[('1', "Summary", ""),
                                       ('2', "Detailed", "Tags, groups, root-heavy vertices"),
                                       ('3', "Verbose", "Every multi-influence vertex")])
        debug_attributes: BoolProperty(name="Debug Attributes", default=True)
        debug_heatmaps: BoolProperty(name="Heat-Map Colours", default=True)
        debug_bone_empties: BoolProperty(name="Bone Empties", default=False)
        seam_check: BoolProperty(name="Seam Weight Check", default=True)
        seam_tolerance: FloatProperty(name="Seam Distance", default=1e-4, min=1e-8, precision=6)
        write_log_text: BoolProperty(name="Log to TXT", default=True,
                                    description="Write the console log to a .txt next to the .yobj")

        def draw(self, context):
            # only the everyday stuff lives here; the rest is in the sub-panels below
            layout = self.layout
            layout.use_property_split = True
            layout.use_property_decorate = False
            layout.prop(self, 'mode')
            layout.prop(self, 'import_materials')
            col = layout.column()
            col.enabled = self.import_materials
            col.prop(self, 'new_collection')
            col.prop(self, 'texture_dir')
            layout.prop(self, 'axis')
            layout.prop(self, 'global_scale')
            layout.prop(self, 'join_parts')

        def check(self, context):
            # runs while browsing: blank Texture Folder -> use a 'textures' folder beside the selection
            if self.texture_dir.strip() and self.texture_dir != self.auto_texture_dir:
                return False        # user typed their own, leave it alone
            found = detect_texture_dir(self.filepath) or detect_texture_dir(self.directory)
            if found != self.texture_dir:
                self.texture_dir = self.auto_texture_dir = found
                return True
            return False

        def execute(self, context):
            dm = self.debug_mode        # debug toggles are ignored unless Debug Mode is on
            paths = [os.path.join(self.directory, f.name) for f in self.files] or [self.filepath]
            mode, mode_reason = self.mode, ''
            if mode == 'AUTO':
                mode, mode_reason = detect_mode(paths)
                self.report({'INFO'}, 'Auto Detect chose %s Mode (%s)' % (mode.title(), mode_reason))
            arena = (mode == 'ARENA')
            opts = ImportOptions(
                bone_base=int(self.bone_base), lane_order=self.lane_order,
                bone_orient=self.bone_orient, tail_length=self.tail_length,
                use_custom_normals=self.use_custom_normals, import_materials=self.import_materials,
                texture_dir=bpy.path.abspath(self.texture_dir), axis=self.axis,
                handedness=self.handedness, flip_u=self.flip_u,
                global_scale=self.global_scale, new_collection=self.new_collection,
                merge_skeletons=self.merge_skeletons,
                # arena files have no sheen / hair / cloth-spec, so those are forced off there
                # (the hidden checkboxes could still be ticked from an earlier character import)
                secondary_layers=self.secondary_layers and not arena,
                sheen_flip_v=self.sheen_flip_v and dm, sheen_strength=self.sheen_strength,
                cloth_alpha_spec=self.cloth_alpha_spec and not arena,
                hair_body_cutoff=-1.0 if arena else self.hair_body_cutoff,
                reuse_materials=self.reuse_materials, merge_tolerance=self.merge_tolerance,
                hair_soft_fringe=self.hair_soft_fringe and not arena,
                # the three sheen experiments only count while Debug Mode is on
                sheen_mapping=self.sheen_mapping if dm else 'CAMERA',
                sheen_alpha_scale=self.sheen_alpha_scale if dm else 1.0,
                sheen_blend=self.sheen_blend if dm else 'EMISSION',
                join_parts=self.join_parts,
                mode=mode, mode_reason=mode_reason, mirror_c351=arena and self.mirror_c351,
                alpha_c351=arena and self.alpha_c351,
                dynamic_collections=arena and self.dynamic_collections,
                place_crowd=arena and self.place_crowd, crowd_density=self.crowd_density,
                crowd_hip_height=self.crowd_hip_height, crowd_face_ring=self.crowd_face_ring,
                crowd_yaw_offset=self.crowd_yaw_offset,
                log_level=int(self.log_level) if dm else 1,
                debug_attributes=dm and self.debug_attributes,
                debug_heatmaps=dm and self.debug_heatmaps,
                debug_bone_empties=dm and self.debug_bone_empties,
                seam_check=dm and self.seam_check, seam_tolerance=self.seam_tolerance,
                write_log_text=dm and self.write_log_text)

            if self.import_materials and not opts.texture_dir.strip():
                opts.texture_dir = detect_texture_dir(paths[0])     # still blank: look beside the yobj
            models, failures = run_import_batch(paths, opts)
            for path, err in failures:
                self.report({'ERROR'}, '%s: %s' % (os.path.basename(path), err))
            if not models:
                return {'CANCELLED'}
            # a file that failed to parse is in `failures` but not in `models`, so only
            # count the failures that belong to files we did get as far as building
            built = {m.path for m in models}
            ok = len(models) - sum(1 for p, _ in failures if p in built)
            self.report({'INFO'}, 'Imported %d YOBJ file(s)' % ok)
            return {'FINISHED'}

    class _YobjSubPanel:
        """Shared bits for the two collapsible sections in the file browser sidebar."""
        bl_space_type = 'FILE_BROWSER'
        bl_region_type = 'TOOL_PROPS'
        bl_parent_id = "FILE_PT_operator"
        bl_options = {'DEFAULT_CLOSED'}

        @classmethod
        def poll(cls, context):
            op = context.space_data.active_operator
            return op is not None and op.bl_idname == "IMPORT_SCENE_OT_yobj"

    class YOBJ_PT_expert(_YobjSubPanel, bpy.types.Panel):
        bl_label = "Expert Settings"

        def draw(self, context):
            op = context.space_data.active_operator
            layout = self.layout
            layout.use_property_split = True
            layout.use_property_decorate = False
            common = ('handedness', 'bone_orient', 'tail_length',
                      'merge_skeletons', 'merge_tolerance', 'reuse_materials')
            character_only = ('secondary_layers', 'sheen_strength', 'hair_soft_fringe',
                              'hair_body_cutoff', 'cloth_alpha_spec')
            arena_only = ('mirror_c351', 'alpha_c351', 'dynamic_collections', 'place_crowd')
            crowd_extras = ('crowd_density', 'crowd_hip_height', 'crowd_face_ring', 'crowd_yaw_offset')
            tail = ('use_custom_normals', 'flip_u')
            if op.mode == 'AUTO':
                # we don't know yet which one it will be, so show both sets
                names = (common + character_only + arena_only
                         + (crowd_extras if op.place_crowd else ()) + tail)
            elif op.mode == 'ARENA':
                # sheen / hair / cloth options are hidden; the crowd sub-options only show once Place Crowd is on
                names = common + arena_only + (crowd_extras if op.place_crowd else ()) + tail
            else:
                names = common + character_only + tail
            for name in names:
                layout.prop(op, name)

    class YOBJ_PT_debug(_YobjSubPanel, bpy.types.Panel):
        bl_label = "Debug Mode"

        def draw_header(self, context):
            self.layout.prop(context.space_data.active_operator, 'debug_mode', text="")

        def draw(self, context):
            op = context.space_data.active_operator
            layout = self.layout
            layout.use_property_split = True
            layout.use_property_decorate = False
            layout.enabled = op.debug_mode      # greyed out until the header box is ticked
            for name in ('log_level', 'sheen_mapping', 'sheen_alpha_scale', 'sheen_blend', 'sheen_flip_v',
                         'debug_attributes', 'debug_heatmaps', 'debug_bone_empties',
                         'seam_check', 'seam_tolerance', 'write_log_text'):
                layout.prop(op, name)

    _CLASSES = (IMPORT_OT_yobj, YOBJ_PT_expert, YOBJ_PT_debug)

    def menu_func_import(self, context):
        self.layout.operator(IMPORT_OT_yobj.bl_idname, text="Rumble Roses YOBJ (.yobj)")

    def register():
        for cls in _CLASSES:
            bpy.utils.register_class(cls)
        bpy.types.TOPBAR_MT_file_import.append(menu_func_import)

    def unregister():
        bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
        for cls in reversed(_CLASSES):
            bpy.utils.unregister_class(cls)


# ---- entry points -------------------------------------------------------------

if __name__ == "__main__":
    if bpy is not None:
        # Run Script from Blender's Text Editor: register, then pop the file browser
        try:
            unregister()          # stale copy from a previous Run Script
        except Exception:
            pass
        register()
        bpy.ops.import_scene.yobj('INVOKE_DEFAULT')
    else:
        # plain Python: just parse and print the report so you can sanity-check files
        paths = sys.argv[1:]
        if not paths:
            print('usage: python import_yobj.py file.yobj [more.yobj ...]   (parse + report only, no Blender)')
            sys.exit(1)
        for p in paths:
            m = parse_yobj(p)
            lg = ImportLog(2)
            log_model(m, lg)
            log_seam_report(m, seam_check(m), lg)
