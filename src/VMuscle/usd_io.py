from __future__ import annotations

"""Minimal USD read/write facade for RLMuscle.

Public API:
- `UsdIO`: one object for read + layered edit operations.
- `usd_args`: CLI argument helper for teaching examples.
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

ColorRgb = tuple[float, float, float]
PrimvarsMap = dict[str, np.ndarray | None]

# Hardcoded Y-up -> Z-up rotation: 90 deg around X axis.
_Y_TO_Z = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)


@dataclass
class UsdMesh:
    mesh_path: str
    vertices: np.ndarray          # (N, 3) float32
    faces: np.ndarray             # (T, 3) int32
    color: ColorRgb = (0.7, 0.7, 0.7)
    tets: np.ndarray | None = None          # (M, 4) int32
    primvars: PrimvarsMap = field(default_factory=dict)


def usd_args(
    usd_path: str = "data/muscle/model/bicep.usd",
    output_path: str = "output.usd",
    parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    """Add shared USD-layering CLI arguments for teaching examples."""
    if parser is None:
        import newton.examples
        parser = newton.examples.create_parser()
    parser.set_defaults(output_path=output_path)
    parser.add_argument("--usd-path", type=str, default=usd_path,
                        help="data/muscle/model/bicep.usd")
    parser.add_argument("--usd-root-path", type=str, default="/",
                        help="USD prim path to load.")
    parser.add_argument("--use_layered_usd", "--use-layered-usd",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Write runtime edits into a layered USD file.")
    parser.add_argument("--copy_usd", "--copy-usd",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Copy source USD to output directory before writing layered edits.")
    return parser


# ---------------------------------------------------------------------------
# Internal helpers – mesh loading
# ---------------------------------------------------------------------------

def _triangulate_faces(face_counts: np.ndarray, face_indices: np.ndarray) -> np.ndarray:
    """Fan-triangulate polygonal faces into (T, 3) int32 triangle array."""
    tris: list[tuple[int, int, int]] = []
    offset = 0
    for count in face_counts.tolist():
        count = int(count)
        face = face_indices[offset : offset + count]
        offset += count
        if count < 3:
            continue
        a = int(face[0])
        for i in range(1, count - 1):
            tris.append((a, int(face[i]), int(face[i + 1])))
    return np.asarray(tris, dtype=np.int32)


def _fix_tet_winding(vertices: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Ensure all tets have positive signed volume."""
    p0, p1, p2, p3 = (vertices[tets[:, i]] for i in range(4))
    signed6v = np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), p3 - p0)
    inv = signed6v < 0.0
    if np.any(inv):
        tets[inv, 0], tets[inv, 1] = tets[inv, 1].copy(), tets[inv, 0].copy()
    return tets


def _extract_surface_tris(tets: np.ndarray) -> np.ndarray:
    """Extract boundary triangles from tet mesh (faces shared by exactly one tet)."""
    tet_faces = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
    counts: dict[tuple, list] = {}
    for tet in tets:
        for f in tet_faces:
            tri = (int(tet[f[0]]), int(tet[f[1]]), int(tet[f[2]]))
            counts.setdefault(tuple(sorted(tri)), []).append(tri)
    return np.asarray([v[0] for v in counts.values() if len(v) == 1], dtype=np.int32)


def _read_primvars(prim) -> PrimvarsMap:
    """Read all authored primvars from a prim via PrimvarsAPI."""
    from pxr import UsdGeom
    result: PrimvarsMap = {}
    for pv in UsdGeom.PrimvarsAPI(prim).GetAuthoredPrimvars():
        name = str(pv.GetPrimvarName())
        raw = pv.Get()
        if raw is None:
            result[name] = None
            continue
        arr = np.asarray(raw)
        # Expand indexed primvars
        indices = pv.GetIndices()
        if indices is not None and len(indices) > 0:
            idx = np.asarray(indices, dtype=np.int32)
            if idx.max() < arr.shape[0]:
                arr = arr[idx]
        result[name] = arr
    return result


def _color_from_primvars(primvars: PrimvarsMap) -> ColorRgb:
    """Extract display color from primvars dict, default gray."""
    dc = primvars.get("displayColor")
    if dc is None:
        return (0.7, 0.7, 0.7)
    arr = np.asarray(dc, dtype=np.float32)
    if arr.size < 3:
        return (0.7, 0.7, 0.7)
    rgb = np.clip(arr.reshape(-1, 3).mean(axis=0) if arr.ndim > 1 else arr[:3], 0.0, 1.0)
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]))


def _load_meshes(
    usd_path: str,
    *,
    root_path: str = "/",
    y_up_to_z_up: bool = False,
) -> list[UsdMesh]:
    from pxr import Gf, Usd, UsdGeom

    root_path = root_path.strip() or "/"
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise FileNotFoundError(f"Failed to open USD file: {usd_path}")

    apply_rotation = y_up_to_z_up and str(UsdGeom.GetStageUpAxis(stage)).upper() == "Y"
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    # Iterate prims under root
    if root_path == "/":
        prim_iter = stage.Traverse()
    else:
        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim.IsValid():
            raise ValueError(f"USD root path does not exist: {root_path}")
        prim_iter = Usd.PrimRange(root_prim)

    meshes: list[UsdMesh] = []
    for prim in prim_iter:
        is_mesh = prim.IsA(UsdGeom.Mesh)
        is_tet = prim.IsA(UsdGeom.TetMesh)
        if not is_mesh and not is_tet:
            continue

        tets = None
        if is_mesh:
            # Try newton.usd.get_mesh (handles triangulation, normals)
            verts, tris = None, None
            try:
                import newton.usd as nusd
                nm = nusd.get_mesh(prim)
                verts = np.asarray(nm.vertices, dtype=np.float32)
                idx = np.asarray(nm.indices, dtype=np.int32)
                tris = idx.reshape(-1, 3) if idx.ndim == 1 else idx
            except Exception:
                pass
            # Fallback: direct pxr
            if verts is None:
                m = UsdGeom.Mesh(prim)
                pts, cnts, idx = m.GetPointsAttr().Get(), m.GetFaceVertexCountsAttr().Get(), m.GetFaceVertexIndicesAttr().Get()
                if pts is None or cnts is None or idx is None:
                    continue
                verts = np.asarray(pts, dtype=np.float32)
                tris = _triangulate_faces(np.asarray(cnts, dtype=np.int32), np.asarray(idx, dtype=np.int32))
                if tris.size == 0:
                    continue

        elif is_tet:
            # Direct pxr for TetMesh
            tm = UsdGeom.TetMesh(prim)
            pts_val = tm.GetPointsAttr().Get()
            if pts_val is None:
                continue
            verts = np.asarray(pts_val, dtype=np.float32)

            # Surface triangles
            tris = None
            surf_attr = tm.GetSurfaceFaceVertexIndicesAttr()
            if surf_attr and surf_attr.IsValid():
                sv = surf_attr.Get()
                if sv is not None:
                    raw = np.asarray(sv, dtype=np.int32)
                    if raw.size > 0:
                        tris = raw.reshape(-1, 3) if raw.ndim == 1 else raw

            # Tet indices
            tet_attr = tm.GetTetVertexIndicesAttr()
            if tet_attr and tet_attr.IsValid():
                tv = tet_attr.Get()
                if tv is not None:
                    raw = np.asarray(tv, dtype=np.int32)
                    if raw.size > 0:
                        tets = raw.reshape(-1, 4) if raw.ndim == 1 else raw

            # Compute surface from tets if needed
            if (tris is None or tris.size == 0) and tets is not None and tets.shape[0] > 0:
                valid = np.all((tets >= 0) & (tets < verts.shape[0]), axis=1)
                tets = _fix_tet_winding(verts, tets[valid].copy())
                tris = _extract_surface_tris(tets)

            if tris is None or tris.size == 0:
                continue

        if verts.ndim != 2 or verts.shape[1] != 3 or verts.shape[0] == 0:
            continue

        # World transform
        world = xform_cache.GetLocalToWorldTransform(prim)
        verts = np.asarray(
            [world.Transform(Gf.Vec3d(*p.astype(float))) for p in verts],
            dtype=np.float32,
        )
        if apply_rotation:
            verts = (verts @ _Y_TO_Z.T).astype(np.float32)
        if tets is not None and tets.size > 0:
            tets = _fix_tet_winding(verts, tets.copy())

        primvars = _read_primvars(prim)
        meshes.append(UsdMesh(
            mesh_path=str(prim.GetPath()),
            vertices=verts,
            faces=np.asarray(tris, dtype=np.int32),
            color=_color_from_primvars(primvars),
            tets=tets,
            primvars=primvars,
        ))

    if not meshes:
        raise ValueError(f"No renderable Mesh/TetMesh found under '{root_path}' in: {usd_path}")
    return meshes


def _center_meshes(meshes: list[UsdMesh], up_axis: int) -> list[UsdMesh]:
    all_pts = np.vstack([m.vertices for m in meshes])
    anchor = 0.5 * (all_pts.min(axis=0) + all_pts.max(axis=0))
    anchor[up_axis] = all_pts[:, up_axis].min()
    for m in meshes:
        m.vertices = m.vertices - anchor
    return meshes


# ---------------------------------------------------------------------------
# _UsdLayerEditor – layered USD editing (direct pxr API)
# ---------------------------------------------------------------------------

class _UsdLayerEditor:
    """Layered USD editor: overlay stage on top of source USD via sublayers."""

    def __init__(self, source_usd_path: str, output_path: str, *, copy_usd: bool = True):
        from pxr import Gf, Sdf, Usd, UsdGeom
        self.Gf, self.Sdf, self.Usd, self.UsdGeom = Gf, Sdf, Usd, UsdGeom

        self.source_usd_path = os.path.abspath(source_usd_path)
        self.output_path = os.path.abspath(output_path)

        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        if copy_usd:
            self._copy_source()

        # Open overlay stage with source as sublayer
        layer = Sdf.Layer.FindOrOpen(self.output_path) or Sdf.Layer.CreateNew(self.output_path)
        src_rel = os.path.relpath(self.source_usd_path, os.path.dirname(self.output_path)).replace("\\", "/")
        if src_rel not in list(layer.subLayerPaths):
            layer.subLayerPaths = list(layer.subLayerPaths) + [src_rel]
            layer.Save()
        self.stage = Usd.Stage.Open(layer)
        self.stage.SetEditTarget(layer)
        self._layer = layer
        self._time_start: int | None = None
        self._time_end: int | None = None

        # Metadata scope
        dp = self.stage.GetDefaultPrim()
        scope_path = f"{dp.GetPath()}/anim" if dp and dp.IsValid() else "/anim"
        scope = self.stage.DefinePrim(scope_path, "Scope")
        self._frame_attr = scope.CreateAttribute("frameNum", Sdf.ValueTypeNames.Int, custom=True)
        scope.CreateAttribute("sourceUsd", Sdf.ValueTypeNames.String, custom=True).Set(
            self.source_usd_path.replace("\\", "/"))
        scope.CreateAttribute("copyUsd", Sdf.ValueTypeNames.Bool, custom=True).Set(copy_usd)

    def _copy_source(self):
        import shutil
        dst = os.path.join(os.path.dirname(self.output_path), os.path.basename(self.source_usd_path))
        dst = os.path.abspath(dst)
        if os.path.normcase(dst) != os.path.normcase(self.source_usd_path):
            shutil.copy2(self.source_usd_path, dst)
        self.source_usd_path = dst

    def _touch_time(self, frame: int):
        if self._time_start is None or frame < self._time_start:
            self._time_start = frame
        if self._time_end is None or frame > self._time_end:
            self._time_end = frame
        for layer in (self.stage.GetRootLayer(), self._layer):
            if layer:
                layer.startTimeCode = float(self._time_start)
                layer.endTimeCode = float(self._time_end)

    def mark_frame(self, frame: int):
        self._frame_attr.Set(frame, frame)
        self._touch_time(frame)

    # -- Scalar / custom property writing --

    def _sdf_type(self, value, value_type: str | None):
        """Resolve Sdf type: explicit name or infer from Python type."""
        if value_type:
            return getattr(self.Sdf.ValueTypeNames, value_type)
        if isinstance(value, bool):
            return self.Sdf.ValueTypeNames.Bool
        if isinstance(value, int):
            return self.Sdf.ValueTypeNames.Int
        if isinstance(value, float):
            return self.Sdf.ValueTypeNames.Float
        if isinstance(value, str):
            return self.Sdf.ValueTypeNames.String
        raise ValueError(f"Cannot infer USD type from {type(value).__name__}. Pass value_type=.")

    def _coerce(self, value, sdf_type):
        """Coerce Python value to USD-compatible type."""
        tn = str(sdf_type)
        if "Float3" in tn or "Color3f" in tn:
            a = np.asarray(value, dtype=np.float32).ravel()
            return self.Gf.Vec3f(float(a[0]), float(a[1]), float(a[2]))
        if "Bool" in tn:
            return bool(value)
        if "Int" in tn:
            return int(value)
        if "Float" in tn or "Double" in tn:
            return float(value)
        if "String" in tn:
            return str(value)
        return value

    def set_custom_property(self, prim_path: str, name: str, value, *,
                            value_type: str | None = None, frame: int | None = None,
                            create_prim: str | None = None) -> bool:
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid() and create_prim:
            self.stage.DefinePrim(prim_path, create_prim)
            prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return False

        sdf_t = self._sdf_type(value, value_type)
        attr = prim.CreateAttribute(name, sdf_t, custom=True)
        v = self._coerce(value, sdf_t)
        if frame is None:
            attr.Set(v)
        else:
            attr.Set(v, int(frame))
            self.mark_frame(int(frame))
        return True

    # -- Primvar writing (direct PrimvarsAPI) --

    def _sdf_primvar_type(self, value, value_type: str | None):
        """Resolve Sdf type for array primvar values."""
        if value_type:
            return getattr(self.Sdf.ValueTypeNames, value_type)
        arr = np.asarray(value)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return self.Sdf.ValueTypeNames.Float3Array
        if arr.ndim >= 1 and arr.dtype.kind == "f":
            return self.Sdf.ValueTypeNames.FloatArray
        if arr.ndim >= 1 and arr.dtype.kind in ("i", "u"):
            return self.Sdf.ValueTypeNames.IntArray
        return self._sdf_type(value, None)

    def _coerce_primvar(self, value, sdf_type):
        """Coerce value for primvar Set()."""
        tn = str(sdf_type)
        if "Array" not in tn:
            return self._coerce(value, sdf_type)
        arr = np.asarray(value, dtype=np.float32 if "Float" in tn or "Color" in tn else None)
        if "Color3fArray" in tn or "Float3Array" in tn or "Normal3fArray" in tn or "Point3fArray" in tn:
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return [self.Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in arr]
        if "IntArray" in tn:
            return np.asarray(value, dtype=np.int32).ravel().tolist()
        if "FloatArray" in tn or "DoubleArray" in tn:
            return np.asarray(value, dtype=np.float32).ravel().astype(float).tolist()
        if "BoolArray" in tn:
            return [bool(v) for v in np.asarray(value).ravel()]
        return value

    def set_primvar(self, prim_path: str, name: str, value, *,
                    value_type: str | None = None, interpolation: str = "constant",
                    frame: int | None = None) -> bool:
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return False

        sdf_t = self._sdf_primvar_type(value, value_type)
        interp = getattr(self.UsdGeom.Tokens, interpolation, self.UsdGeom.Tokens.constant)
        api = self.UsdGeom.PrimvarsAPI(prim)
        pv = api.GetPrimvar(name)
        if not pv:
            pv = api.CreatePrimvar(name, sdf_t, interp)
        else:
            pv.SetInterpolation(interp)

        coerced = self._coerce_primvar(value, sdf_t)
        if frame is None:
            pv.Set(coerced)
        else:
            pv.Set(coerced, int(frame))
            self.mark_frame(int(frame))
        return True

    def set_display_color(self, prim_path: str, color: ColorRgb, *, frame: int | None = None) -> bool:
        rgb = np.clip(color, 0.0, 1.0).tolist() if isinstance(color, np.ndarray) else [max(0, min(1, c)) for c in color]
        return self.set_primvar(prim_path, "displayColor", [rgb],
                                value_type="Color3fArray", interpolation="constant", frame=frame)

    # -- Save / close --

    def save(self):
        for layer in (self.stage.GetRootLayer(), self._layer):
            if layer:
                layer.Save()

    def close(self):
        self.save()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# UsdIO – public API
# ---------------------------------------------------------------------------

class UsdIO:
    """Minimal public USD API for mesh reading and layered editing.

    Quick start::

        usd = UsdIO("data/muscle/model/bicep.usd").read()
        for mesh in usd.meshes:
            print(mesh.mesh_path, mesh.vertices.shape, mesh.primvars.keys())

        with usd.start("output/bicep.anim.usda"):
            usd.set_runtime("activation", 0.5, frame=0)
            usd.set_primvar("/character/muscle", "displayColor", [[1,0,0]], value_type="Color3fArray")
    """

    def __init__(self, source_usd_path: str, *, root_path: str = "/",
                 y_up_to_z_up: bool = False, center_model: bool = False):
        self.source_usd_path = os.path.abspath(str(source_usd_path))
        self.root_path = root_path.strip() or "/"
        self.y_up_to_z_up = bool(y_up_to_z_up)
        self.center_model = bool(center_model)
        self._up_axis = 2 if y_up_to_z_up else 1
        self._meshes: list[UsdMesh] | None = None
        self._editor: _UsdLayerEditor | None = None
        self._runtime_prim = "/anim/runtime"

        if not os.path.isfile(self.source_usd_path):
            raise FileNotFoundError(f"USD source file not found: {self.source_usd_path}")

    def read(self) -> "UsdIO":
        if self._meshes is not None:
            return self
        self._meshes = _load_meshes(
            self.source_usd_path, root_path=self.root_path, y_up_to_z_up=self.y_up_to_z_up)
        if self.center_model:
            _center_meshes(self._meshes, self._up_axis)
        return self

    @property
    def meshes(self) -> list[UsdMesh]:
        if self._meshes is None:
            raise RuntimeError("Call read() first.")
        return self._meshes

    @property
    def focus_points(self) -> np.ndarray:
        m = self.meshes
        return np.vstack([x.vertices for x in m]).astype(np.float32) if m else np.zeros((0, 3), np.float32)

    def start(self, output_path: str | None = None, *,
              copy_usd: bool = True, runtime_prim: str = "/anim/runtime") -> "UsdIO":
        if output_path is None:
            stem = os.path.splitext(os.path.basename(self.source_usd_path))[0]
            output_path = os.path.abspath(os.path.join("output", f"{stem}.anim.usda"))
        self.close()
        self._editor = _UsdLayerEditor(self.source_usd_path, output_path, copy_usd=copy_usd)
        self._runtime_prim = runtime_prim.strip() or "/anim/runtime"
        self._editor.stage.DefinePrim(self._runtime_prim, "Scope")
        return self

    @property
    def output_path(self) -> str | None:
        return self._editor.output_path if self._editor else None

    def set_custom(self, prim_path: str, name: str, value, *,
                   value_type: str | None = None, frame: int | None = None) -> bool:
        return self._editor.set_custom_property(
            prim_path, name, value, value_type=value_type, frame=frame, create_prim="Scope")

    def set_runtime(self, name: str, value, *,
                    value_type: str | None = None, frame: int | None = None) -> bool:
        return self.set_custom(self._runtime_prim, name, value, value_type=value_type, frame=frame)

    def set_primvar(self, prim_path: str, name: str, value, *,
                    value_type: str | None = None, interpolation: str = "constant",
                    frame: int | None = None) -> bool:
        return self._editor.set_primvar(
            prim_path, name, value, value_type=value_type, interpolation=interpolation, frame=frame)

    def set_color(self, prim_path: str, color: ColorRgb, *, frame: int | None = None) -> bool:
        return self._editor.set_display_color(prim_path, color, frame=frame)

    def save(self):
        if self._editor:
            self._editor.save()

    def close(self):
        if self._editor:
            self._editor.close()
            self._editor = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


__all__ = ["UsdIO", "usd_args", "UsdMesh"]
