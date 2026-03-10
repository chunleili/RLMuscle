"""Mesh I/O for RLMuscle — .usd and .geo formats only.

Provides:
- load_mesh(path)       — muscle TetMesh (positions, tets, fibers, tendon_mask, geo)
- load_bone_mesh(path)  — bone surface mesh (geo, positions, indices, muscle_ids)
- build_surface_tris()  — extract boundary triangles from tetrahedra
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

# Hardcoded Y-up -> Z-up rotation: 90 deg around X axis.
_Y_TO_Z = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Surface triangle extraction
# ---------------------------------------------------------------------------

def build_surface_tris(tets: np.ndarray, positions: np.ndarray = None) -> np.ndarray:
    """Extract boundary triangles from tetrahedra.

    If *positions* is given, orient each triangle so its outward normal points
    away from the tet interior (needed for correct backface culling).
    """
    tet_faces = (
        (1, 2, 3, 0),
        (0, 3, 2, 1),
        (0, 1, 3, 2),
        (0, 2, 1, 3),
    )
    counts: dict[tuple, list] = {}
    for tet in tets:
        for f in tet_faces:
            tri = (int(tet[f[0]]), int(tet[f[1]]), int(tet[f[2]]))
            opp = int(tet[f[3]])
            key = tuple(sorted(tri))
            counts.setdefault(key, []).append((tri, opp))

    surface = []
    for entries in counts.values():
        if len(entries) == 1:
            tri, opp = entries[0]
            if positions is not None:
                p0, p1, p2 = positions[tri[0]], positions[tri[1]], positions[tri[2]]
                n = np.cross(p1 - p0, p2 - p0)
                d = positions[opp] - p0
                if np.dot(n, d) > 0:
                    tri = (tri[0], tri[2], tri[1])  # flip winding
            surface.append(tri)
    return np.asarray(surface, dtype=np.int32)


# ---------------------------------------------------------------------------
# Internal: format detection
# ---------------------------------------------------------------------------

def _is_usd(path) -> bool:
    return str(path).lower().endswith((".usd", ".usdc", ".usda"))


# ---------------------------------------------------------------------------
# Internal: USD helpers
# ---------------------------------------------------------------------------

def _read_primvar(pv_api, name, dtype=np.float32):
    """Read a named primvar, return numpy array or None."""
    pv = pv_api.GetPrimvar(name)
    if pv and pv.Get() is not None:
        return np.asarray(pv.Get(), dtype=dtype)
    return None


def _load_mesh_usd(path: Path, y_up_to_z_up: bool = False):
    """Load muscle TetMesh from USD.

    Returns (positions, tets, fibers, tendon_mask, geo_namespace).
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise FileNotFoundError(f"Cannot open USD: {path}")

    tet_prim = None
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.TetMesh):
            tet_prim = prim
            break
    if tet_prim is None:
        raise ValueError(f"No TetMesh prim found in {path}")

    tm = UsdGeom.TetMesh(tet_prim)
    positions = np.asarray(tm.GetPointsAttr().Get(), dtype=np.float32)
    tets = np.asarray(tm.GetTetVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 4)

    pv_api = UsdGeom.PrimvarsAPI(tet_prim)
    fibers = _read_primvar(pv_api, "materialW", np.float32)
    tendon_mask = _read_primvar(pv_api, "tendonmask", np.float32)

    if y_up_to_z_up:
        positions = (positions @ _Y_TO_Z.T).astype(np.float32)
        if fibers is not None and fibers.ndim == 2 and fibers.shape[1] == 3:
            fibers = (fibers @ _Y_TO_Z.T).astype(np.float32)

    # Build a geo-like namespace so constraint code can do getattr(self.geo, mask_name)
    geo = SimpleNamespace()
    geo.positions = positions.tolist()
    geo.vert = tets.tolist()
    geo.pointattr = {}

    for pv in pv_api.GetPrimvars():
        name = pv.GetName().replace("primvars:", "")
        if pv.GetInterpolation() != "vertex":
            continue
        val = pv.Get()
        if val is None:
            continue
        arr = np.asarray(val)
        if arr.ndim in (1, 2):
            setattr(geo, name, arr.tolist())
            geo.pointattr[name] = arr.tolist()

    print(f"Loaded USD TetMesh from {path}: {positions.shape[0]} verts, {tets.shape[0]} tets")
    return positions, tets, fibers, tendon_mask, geo


def _load_mesh_geo(path: Path):
    """Load muscle TetMesh from .geo file.

    Returns (positions, tets, fibers, tendon_mask, geo_object).
    """
    from VMuscle.geo import Geo
    geo = Geo(str(path))
    positions = np.asarray(geo.positions, dtype=np.float32)
    tets = np.asarray(geo.vert, dtype=np.int32)
    fibers = np.asarray(geo.materialW, dtype=np.float32) if hasattr(geo, "materialW") else None
    tendon_mask = np.asarray(geo.tendonmask, dtype=np.float32) if hasattr(geo, "tendonmask") else None
    return positions, tets, fibers, tendon_mask, geo


def _load_bone_usd(usd_path: Path, bone_root: str = "/character/bone"):
    """Load bone mesh from USD (multiple Mesh prims under *bone_root*).

    Returns (None, positions, indices, muscle_ids).
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise FileNotFoundError(f"Cannot open USD: {usd_path}")

    root = stage.GetPrimAtPath(bone_root)
    if not root or not root.IsValid():
        return None, np.zeros((0, 3), np.float32), np.zeros(0, np.int32), {}

    all_pos, all_idx, all_mid = [], [], []
    offset = 0

    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        m = UsdGeom.Mesh(prim)
        pts_raw = m.GetPointsAttr().Get()
        if pts_raw is None:
            continue
        pts = np.asarray(pts_raw, dtype=np.float32)
        idx = np.asarray(m.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)

        parent = prim.GetParent()
        bone_name = str(parent.GetName()) if parent and parent.IsValid() else str(prim.GetName())

        all_pos.append(pts)
        all_idx.append(idx + offset)
        all_mid.extend([bone_name] * len(pts))
        offset += len(pts)

    if len(all_pos) == 0:
        return None, np.zeros((0, 3), np.float32), np.zeros(0, np.int32), {}

    positions = np.vstack(all_pos).astype(np.float32)
    indices = np.concatenate(all_idx).astype(np.int32)
    muscle_ids = _build_muscle_id_mapping(all_mid)
    return None, positions, indices, muscle_ids


def _load_bone_geo(target_path: Path):
    """Load bone mesh from .geo file.

    Returns (bone_geo, positions, indices, muscle_ids).
    """
    from VMuscle.geo import Geo
    bone_geo = Geo(str(target_path))
    if len(bone_geo.positions) == 0:
        print(f"Warning: No vertices found in {target_path}")
        return bone_geo, np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int32), {}

    bone_pos = np.asarray(bone_geo.positions, dtype=np.float32)
    if hasattr(bone_geo, 'indices'):
        bone_indices = np.asarray(bone_geo.indices, dtype=np.int32)
    elif hasattr(bone_geo, 'vert'):
        bone_indices = np.array(bone_geo.vert, dtype=np.int32).flatten()
    else:
        bone_indices = np.zeros(0, dtype=np.int32)

    muscle_ids = {}
    if hasattr(bone_geo, 'pointattr') and 'muscle_id' in bone_geo.pointattr:
        muscle_ids = _build_muscle_id_mapping(bone_geo.pointattr['muscle_id'])

    return bone_geo, bone_pos, bone_indices, muscle_ids


def _build_muscle_id_mapping(muscle_ids_list):
    """Build dict mapping muscle_id -> vertex index array from per-vertex id list."""
    mapping = {}
    for v_idx, mid in enumerate(muscle_ids_list):
        if mid not in mapping:
            mapping[mid] = []
        mapping[mid].append(v_idx)
    for mid in mapping:
        mapping[mid] = np.array(mapping[mid], dtype=np.int32)
    print(f"Bone muscle_id groups: {list(mapping.keys())}")
    return mapping


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_mesh(path: Path):
    """Load muscle TetMesh from .usd or .geo file.

    Returns (positions, tets, fibers, tendon_mask, geo).
    """
    if path is None:
        raise ValueError("No mesh path provided.")
    if _is_usd(path):
        return _load_mesh_usd(path, y_up_to_z_up=False)
    else:
        return _load_mesh_geo(path)


def load_bone_mesh(path: Path):
    """Load bone surface mesh from .usd or .geo file.

    Returns (bone_geo, bone_pos, bone_indices, bone_muscle_ids) where:
    - bone_geo: Geo object or None (USD)
    - bone_pos: np.ndarray shape (N, 3)
    - bone_indices: np.ndarray flat int32
    - bone_muscle_ids: dict mapping muscle_id -> vertex index array
    """
    if _is_usd(path):
        return _load_bone_usd(path)
    else:
        return _load_bone_geo(path)
