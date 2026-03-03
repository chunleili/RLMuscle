"""Shared mesh I/O utilities for RLMuscle.

Provides USD and .geo mesh loading, plus surface extraction helpers.
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
    # Tet-face indexing: each tuple gives the 3 vertices of one face,
    # and the 4th vertex (opposite) is the remaining index.
    # Winding is set so that the outward normal (right-hand rule) points
    # away from the opposite vertex.
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
                # Ensure outward winding: normal should point away from opposite vertex
                p0, p1, p2 = positions[tri[0]], positions[tri[1]], positions[tri[2]]
                n = np.cross(p1 - p0, p2 - p0)
                d = positions[opp] - p0
                if np.dot(n, d) > 0:
                    tri = (tri[0], tri[2], tri[1])  # flip winding
            surface.append(tri)
    return np.asarray(surface, dtype=np.int32)


# ---------------------------------------------------------------------------
# USD mesh loading
# ---------------------------------------------------------------------------

def _read_primvar(pv_api, name, dtype=np.float32):
    """Read a named primvar, return numpy array or None."""
    pv = pv_api.GetPrimvar(name)
    if pv and pv.Get() is not None:
        return np.asarray(pv.Get(), dtype=dtype)
    return None


def load_mesh_usd(path: Path, y_up_to_z_up: bool = False):
    """Load muscle TetMesh from a USD file.

    Returns (positions, tets, fibers, tendon_mask, geo_like) matching the
    same 5-tuple interface as load_mesh_geo().

    When *y_up_to_z_up* is True, positions and fiber directions are rotated to Z-up automatically.

    The *geo_like* object is a SimpleNamespace carrying every vertex-interpolated
    primvar as an attribute (e.g. ``geo_like.muscletobonemask``), so existing
    constraint code that does ``getattr(self.geo, mask_name)`` keeps working.
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise FileNotFoundError(f"Cannot open USD: {path}")

    apply_rotation = y_up_to_z_up

    # Find the first TetMesh prim
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

    # Apply Y-up -> Z-up conversion if needed
    if apply_rotation:
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
        if arr.ndim == 1:
            setattr(geo, name, arr.tolist())
            geo.pointattr[name] = arr.tolist()
        elif arr.ndim == 2:
            setattr(geo, name, arr.tolist())
            geo.pointattr[name] = arr.tolist()

    print(f"Loaded USD TetMesh from {path}: {positions.shape[0]} verts, {tets.shape[0]} tets")
    return positions, tets, fibers, tendon_mask, geo


def load_bone_usd_data(usd_path: Path, bone_root: str = "/character/bone"):
    """Load bone mesh data from USD (multiple Mesh prims under *bone_root*).

    Returns (positions, face_indices_flat, muscle_id_per_vertex) where
    all bone meshes are concatenated into single arrays.
    muscle_id_per_vertex is a list of strings, one per vertex.
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise FileNotFoundError(f"Cannot open USD: {usd_path}")

    root = stage.GetPrimAtPath(bone_root)
    if not root or not root.IsValid():
        return np.zeros((0, 3), np.float32), np.zeros(0, np.int32), []

    all_pos = []
    all_idx = []
    all_mid = []  # per-vertex muscle_id string
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

        # Determine bone name from parent Xform prim (e.g. /character/bone/L_radius)
        parent = prim.GetParent()
        bone_name = str(parent.GetName()) if parent and parent.IsValid() else str(prim.GetName())

        all_pos.append(pts)
        all_idx.append(idx + offset)
        all_mid.extend([bone_name] * len(pts))
        offset += len(pts)

    if len(all_pos) == 0:
        return np.zeros((0, 3), np.float32), np.zeros(0, np.int32), []

    positions = np.vstack(all_pos).astype(np.float32)
    indices = np.concatenate(all_idx).astype(np.int32)
    return positions, indices, all_mid


__all__ = ["build_surface_tris", "load_mesh_usd", "load_bone_usd_data"]
