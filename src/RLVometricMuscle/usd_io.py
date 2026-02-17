from __future__ import annotations

"""Minimal USD read/write facade for RLMuscle.

Public API:
- `UsdIO`: one object for read + layered edit operations.
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

ColorRgb = tuple[float, float, float]
PrimvarsMap = dict[str, np.ndarray | None]
_MeshTuple = tuple[str, np.ndarray, np.ndarray, ColorRgb, PrimvarsMap]
_USE_NEWTON_USD_GET_MESH = True


def usd_args(
    usd_path: str,
    output_path: str = "output.usd",
    parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    """Add shared USD-layering CLI arguments for teaching examples."""
    if parser is None:
        import newton.examples
        parser = newton.examples.create_parser()

    parser.set_defaults(output_path=output_path)
    parser.add_argument("--usd-path", 
                        type=str, 
                        default=usd_path,
                        help="data/muscle/model/bicep.usd")
    parser.add_argument("--usd-root-path", 
                        type=str,
                        default="/",
                        help="USD prim path to load.")
    parser.add_argument(
        "--use_layered_usd",
        "--use-layered-usd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write runtime edits into a layered USD file.",
    )
    parser.add_argument(
        "--copy_usd",
        "--copy-usd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy source USD to output directory before writing layered edits.",
    )
    return parser


@dataclass(frozen=True)
class UsdMesh:
    path: str
    vertices: np.ndarray
    faces: np.ndarray
    color: ColorRgb
    primvars: PrimvarsMap


@dataclass
class UsdMeshSet:
    meshes: list[UsdMesh]
    center_shift: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    def __post_init__(self) -> None:
        self.center_shift = np.asarray(self.center_shift, dtype=np.float32)

    @property
    def focus_points(self) -> np.ndarray:
        if not self.meshes:
            return np.zeros((0, 3), dtype=np.float32)
        return np.vstack([mesh.vertices for mesh in self.meshes]).astype(np.float32)

    @property
    def base_colors(self) -> dict[str, ColorRgb]:
        return {mesh.path: mesh.color for mesh in self.meshes}

    @property
    def primvars(self) -> dict[str, PrimvarsMap]:
        return {mesh.path: mesh.primvars for mesh in self.meshes}

    @property
    def primvar_summary(self) -> dict[str, dict[str, Any]]:
        return _summarize_primvars(self.primvars)


def _axis_index_from_any(axis: Any | None) -> int | None:
    if axis is None:
        return None

    if isinstance(axis, str):
        name = axis.strip().upper()
        if name in ("X", "Y", "Z"):
            return "XYZ".index(name)
        raise ValueError(f"Invalid axis string: {axis!r}. Expected one of 'X', 'Y', 'Z'.")

    try:
        value = int(axis)
    except Exception:
        name = getattr(axis, "name", None)
        if isinstance(name, str):
            name = name.strip().upper()
            if name in ("X", "Y", "Z"):
                return "XYZ".index(name)
        value_attr = getattr(axis, "value", None)
        if value_attr is None:
            raise ValueError(f"Unsupported axis value: {axis!r}.")
        value = int(value_attr)

    if value not in (0, 1, 2):
        raise ValueError(f"Invalid axis index: {value}. Expected 0/1/2 for X/Y/Z.")
    return int(value)


def _normalize_root_path(root_path: str | None) -> str:
    value = "/" if root_path is None else str(root_path).strip()
    if not value:
        value = "/"
    if not value.startswith("/"):
        raise ValueError("USD root path must start with '/'.")
    return value


def _resolve_up_axis_index(up_axis: Any | None, *, default: int = 2) -> int:
    resolved = _axis_index_from_any(up_axis)
    if resolved is None:
        return int(default)
    return int(resolved)


def _rotation_matrix_align_up(source_up_axis: int, target_up_axis: int) -> np.ndarray:
    if source_up_axis == target_up_axis:
        return np.eye(3, dtype=np.float32)

    src = np.zeros(3, dtype=np.float64)
    dst = np.zeros(3, dtype=np.float64)
    src[source_up_axis] = 1.0
    dst[target_up_axis] = 1.0

    v = np.cross(src, dst)
    s = float(np.linalg.norm(v))
    c = float(np.dot(src, dst))

    if s < 1.0e-12:
        if c > 0.0:
            return np.eye(3, dtype=np.float32)
        ortho = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(src[0]) > 0.9:
            ortho = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(src, ortho)
        axis /= np.linalg.norm(axis)
        K = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ],
            dtype=np.float64,
        )
        R = np.eye(3, dtype=np.float64) + 2.0 * (K @ K)
        return R.astype(np.float32)

    K = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )
    R = np.eye(3, dtype=np.float64) + K + (K @ K) * ((1.0 - c) / (s * s))
    return R.astype(np.float32)


def _triangulate_faces(face_counts: np.ndarray, face_indices: np.ndarray) -> np.ndarray:
    if face_counts.ndim != 1 or face_indices.ndim != 1:
        raise ValueError("Invalid USD mesh face arrays.")
    if int(face_counts.sum()) != int(face_indices.size):
        raise ValueError("USD mesh face counts and indices are inconsistent.")

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
    p0 = vertices[tets[:, 0]]
    p1 = vertices[tets[:, 1]]
    p2 = vertices[tets[:, 2]]
    p3 = vertices[tets[:, 3]]
    signed6v = np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), p3 - p0)
    inverted = signed6v < 0.0
    if np.any(inverted):
        t0 = tets[inverted, 0].copy()
        tets[inverted, 0] = tets[inverted, 1]
        tets[inverted, 1] = t0
    return tets


def _extract_surface_tris(tets: np.ndarray) -> np.ndarray:
    faces = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
    counts: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    for tet in tets:
        for f in faces:
            tri = (int(tet[f[0]]), int(tet[f[1]]), int(tet[f[2]]))
            key = tuple(sorted(tri))
            counts.setdefault(key, []).append(tri)
    surface = [tris[0] for tris in counts.values() if len(tris) == 1]
    return np.asarray(surface, dtype=np.int32)


def _indices_from_value(value, width: int) -> np.ndarray | None:
    if value is None:
        return None
    raw = np.asarray(value, dtype=np.int32)
    if raw.size == 0:
        return None
    if raw.ndim == 1 and raw.size % width == 0:
        out = raw.reshape(-1, width)
    elif raw.ndim == 2 and raw.shape[1] == width:
        out = raw
    else:
        return None
    return np.asarray(out, dtype=np.int32)


def _points_from_point_based(point_based) -> np.ndarray | None:
    points_attr = point_based.GetPointsAttr()
    if not points_attr or not points_attr.IsValid():
        return None
    points_value = points_attr.Get()
    if points_value is None:
        return None
    points = np.asarray(points_value, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        return None
    return points


def _tet_indices_from_tetmesh(tetmesh) -> np.ndarray | None:
    attr = tetmesh.GetTetVertexIndicesAttr()
    if not attr or not attr.IsValid():
        return None
    return _indices_from_value(attr.Get(), width=4)


def _surface_tris_from_tetmesh(tetmesh) -> np.ndarray | None:
    attr = tetmesh.GetSurfaceFaceVertexIndicesAttr()
    if not attr or not attr.IsValid():
        return None
    return _indices_from_value(attr.Get(), width=3)


def _mesh_color_from_path(path: str) -> ColorRgb:
    name = path.lower()
    if "bone" in name:
        return (0.75, 0.78, 0.82)
    if "muscle" in name or "bicep" in name:
        return (0.92, 0.35, 0.35)
    return (0.35, 0.62, 0.95)


def _mesh_color_from_display_color(display_color: np.ndarray | None) -> ColorRgb | None:
    if display_color is None:
        return None
    arr = np.asarray(display_color, dtype=np.float32)
    if arr.size == 0:
        return None

    if arr.ndim == 1:
        if arr.shape[0] < 3:
            return None
        rgb = arr[:3]
    else:
        if arr.shape[-1] < 3:
            return None
        rgb = arr[..., :3].reshape(-1, 3).mean(axis=0)

    rgb = np.clip(rgb, 0.0, 1.0)
    return float(rgb[0]), float(rgb[1]), float(rgb[2])


def _read_primvar(prim, name: str) -> np.ndarray | None:
    attr = prim.GetAttribute(f"primvars:{name}")
    if not attr or not attr.IsValid():
        return None
    value = attr.Get()
    if value is None:
        return None
    values = np.asarray(value)
    if values.ndim == 0:
        values = np.asarray(list(value))
    if values.size == 0:
        return None

    idx_attr = prim.GetAttribute(f"primvars:{name}:indices")
    if idx_attr and idx_attr.IsValid():
        idx_value = idx_attr.Get()
        if idx_value is not None:
            indices = np.asarray(idx_value, dtype=np.int32)
            if indices.ndim > 0 and indices.size > 0:
                if np.any(indices < 0) or np.any(indices >= values.shape[0]):
                    raise ValueError(f"Invalid primvar indices for {name} on {prim.GetPath()}")
                return np.asarray(values[indices])
    return np.asarray(values)


def _list_primvar_names(prim) -> list[str]:
    names = set()
    for attr in prim.GetAttributes():
        attr_name = attr.GetName()
        if not attr_name.startswith("primvars:"):
            continue
        suffix = attr_name[len("primvars:") :]
        base = suffix.split(":", 1)[0]
        if base:
            names.add(base)
    return sorted(names)


def _read_all_primvars(prim) -> PrimvarsMap:
    primvars: PrimvarsMap = {}
    for name in _list_primvar_names(prim):
        primvars[name] = _read_primvar(prim, name)
    return primvars


def _load_mesh_with_newton_usd(prim) -> tuple[np.ndarray, np.ndarray] | None:
    global _USE_NEWTON_USD_GET_MESH
    if not _USE_NEWTON_USD_GET_MESH:
        return None

    try:
        import newton.usd as newton_usd
    except Exception:
        _USE_NEWTON_USD_GET_MESH = False
        return None

    try:
        mesh = newton_usd.get_mesh(prim)
    except Exception:
        _USE_NEWTON_USD_GET_MESH = False
        return None

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        return None

    indices = np.asarray(mesh.indices, dtype=np.int32)
    if indices.size == 0:
        return None
    if indices.ndim == 1:
        if indices.size % 3 != 0:
            return None
        tris = indices.reshape(-1, 3)
    elif indices.ndim == 2 and indices.shape[1] == 3:
        tris = indices
    else:
        return None

    return vertices, np.asarray(tris, dtype=np.int32)


def _load_mesh_fallback(prim, UsdGeom) -> tuple[np.ndarray, np.ndarray] | None:
    mesh_prim = UsdGeom.Mesh(prim)
    points_value = mesh_prim.GetPointsAttr().Get()
    counts_value = mesh_prim.GetFaceVertexCountsAttr().Get()
    indices_value = mesh_prim.GetFaceVertexIndicesAttr().Get()
    if points_value is None or counts_value is None or indices_value is None:
        return None

    points = np.asarray(points_value, dtype=np.float32)
    face_counts = np.asarray(counts_value, dtype=np.int32)
    face_indices = np.asarray(indices_value, dtype=np.int32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        return None

    tris = _triangulate_faces(face_counts, face_indices)
    if tris.size == 0:
        return None
    return points, tris


def _load_mesh_geometry(prim, UsdGeom) -> tuple[np.ndarray, np.ndarray] | None:
    parsed = _load_mesh_with_newton_usd(prim)
    if parsed is not None:
        return parsed
    return _load_mesh_fallback(prim, UsdGeom)


def _load_tetmesh_geometry(prim, UsdGeom) -> tuple[np.ndarray, np.ndarray] | None:
    tetmesh = UsdGeom.TetMesh(prim)
    if not tetmesh:
        return None

    points = _points_from_point_based(tetmesh)
    if points is None:
        return None

    tris = _surface_tris_from_tetmesh(tetmesh)
    tet_indices = _tet_indices_from_tetmesh(tetmesh)

    if (tris is None or tris.size == 0) and hasattr(UsdGeom.TetMesh, "ComputeSurfaceFaces"):
        try:
            tris = _indices_from_value(UsdGeom.TetMesh.ComputeSurfaceFaces(tetmesh), width=3)
        except Exception:
            tris = None

    if (tris is None or tris.size == 0) and tet_indices is not None and tet_indices.shape[0] > 0:
        valid = np.all((tet_indices >= 0) & (tet_indices < points.shape[0]), axis=1)
        tet_indices = tet_indices[valid]
        if tet_indices.shape[0] > 0:
            tet_indices = _fix_tet_winding(points, tet_indices.copy())
            tris = _extract_surface_tris(tet_indices)

    if tris is None or tris.size == 0:
        return None
    return points, np.asarray(tris, dtype=np.int32)


def _transform_points_world(points: np.ndarray, prim, xform_cache, Gf) -> np.ndarray:
    world_xform = xform_cache.GetLocalToWorldTransform(prim)
    return np.asarray(
        [world_xform.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2]))) for p in points],
        dtype=np.float32,
    )


def _iter_prims_under_root(stage, root_path: str, Usd):
    if root_path == "/":
        return stage.Traverse()
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        raise ValueError(f"USD root path does not exist: {root_path}")
    return Usd.PrimRange(root_prim)


def _load_mesh_tuples(
    usd_path: str,
    *,
    root_path: str = "/",
    y_up_to_z_up: bool = False,
) -> list[_MeshTuple]:
    try:
        from pxr import Gf, Usd, UsdGeom
    except ImportError as exc:
        raise ImportError(
            "USD import requires `pxr` (`usd-core`). Install with `uv add usd-core` or `pip install usd-core`."
        ) from exc

    root_path = _normalize_root_path(root_path)
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise FileNotFoundError(f"Failed to open USD file: {usd_path}")

    stage_up_axis = _axis_index_from_any(str(UsdGeom.GetStageUpAxis(stage)))
    up_axis_rotation = None
    if y_up_to_z_up and stage_up_axis == 1:
        up_axis_rotation = _rotation_matrix_align_up(1, 2)

    prim_iter = _iter_prims_under_root(stage, root_path, Usd)
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    meshes: list[_MeshTuple] = []
    for prim in prim_iter:
        is_mesh = prim.IsA(UsdGeom.Mesh)
        is_tetmesh = prim.IsA(UsdGeom.TetMesh)
        if not is_mesh and not is_tetmesh:
            continue

        primvars = _read_all_primvars(prim)
        if is_mesh:
            parsed = _load_mesh_geometry(prim, UsdGeom)
        else:
            parsed = _load_tetmesh_geometry(prim, UsdGeom)
        if parsed is None:
            continue

        points, tris = parsed
        points_world = _transform_points_world(points, prim, xform_cache, Gf)
        if up_axis_rotation is not None:
            points_world = np.asarray(points_world @ up_axis_rotation.T, dtype=np.float32)

        mesh_path = str(prim.GetPath())
        mesh_color = _mesh_color_from_display_color(primvars.get("displayColor"))
        if mesh_color is None:
            mesh_color = _mesh_color_from_path(mesh_path)
        meshes.append((mesh_path, points_world, np.asarray(tris, dtype=np.int32), mesh_color, primvars))

    if not meshes:
        raise ValueError(f"No renderable Mesh/TetMesh found under '{root_path}' in USD file: {usd_path}")
    return meshes


def _center_mesh_tuples(meshes: list[_MeshTuple], up_axis: int) -> tuple[list[_MeshTuple], np.ndarray]:
    all_points = np.vstack([vertices for _, vertices, _, _, _ in meshes])
    bbox_min, bbox_max = all_points.min(axis=0), all_points.max(axis=0)
    anchor = 0.5 * (bbox_min + bbox_max)
    anchor[int(up_axis)] = bbox_min[int(up_axis)]
    shifted: list[_MeshTuple] = []
    for mesh_path, vertices, faces, color, primvars in meshes:
        shifted.append((mesh_path, vertices - anchor, faces, color, primvars))
    return shifted, anchor.astype(np.float32)


def _mesh_tuples_to_objects(meshes: list[_MeshTuple]) -> list[UsdMesh]:
    return [
        UsdMesh(
            path=mesh_path,
            vertices=np.asarray(vertices, dtype=np.float32),
            faces=np.asarray(faces, dtype=np.int32),
            color=(float(color[0]), float(color[1]), float(color[2])),
            primvars=primvars,
        )
        for mesh_path, vertices, faces, color, primvars in meshes
    ]


def _summarize_primvars(primvars_by_path: dict[str, PrimvarsMap]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for path, primvars in primvars_by_path.items():
        info: dict[str, Any] = {}
        for key, value in sorted(primvars.items()):
            if value is None:
                info[key] = "None"
                continue
            arr = np.asarray(value)
            shape = tuple(arr.shape)
            if arr.dtype.kind in {"U", "S", "O"}:
                info[key] = f"shape={shape}, unique={np.unique(arr).tolist()}"
            else:
                info[key] = f"shape={shape}, dtype={arr.dtype}"
        summary[path] = info
    return summary


def _as_vec3(value, Gf):
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"Expected 3 values for vec3, got {arr.size}.")
    return Gf.Vec3f(float(arr[0]), float(arr[1]), float(arr[2]))


class _UsdLayerEditor:
    """Layered USD editor with simple prim/property editing methods."""

    def __init__(self, source_usd_path: str, output_path: str, copy_usd: bool = True):
        try:
            from pxr import Gf, Sdf, Usd, UsdGeom
        except ImportError as exc:
            raise ImportError(
                "USD export requires `pxr` (`usd-core`). Install with `uv add usd-core` or `pip install usd-core`."
            ) from exc

        self.Gf = Gf
        self.Sdf = Sdf
        self.Usd = Usd
        self.UsdGeom = UsdGeom

        self.source_usd_path = os.path.abspath(source_usd_path)
        self.output_path = os.path.abspath(output_path)
        self.copy_usd = bool(copy_usd)

        if not os.path.isfile(self.source_usd_path):
            raise FileNotFoundError(f"USD source file not found: {self.source_usd_path}")

        self._ensure_output_dir()
        if self.copy_usd:
            self._copy_source_to_output_dir()

        self.stage, self._edit_layer = self._open_overlay_stage()
        self._time_start: int | None = None
        self._time_end: int | None = None
        self._init_metadata_scope()

    def _ensure_output_dir(self) -> None:
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def _copy_source_to_output_dir(self) -> None:
        import shutil

        dst = os.path.join(os.path.dirname(self.output_path), os.path.basename(self.source_usd_path))
        dst = os.path.abspath(dst)
        if os.path.normcase(dst) != os.path.normcase(self.source_usd_path):
            shutil.copy2(self.source_usd_path, dst)
        self.source_usd_path = dst

    @staticmethod
    def _to_layer_path(path: str, relative_to: str | None = None) -> str:
        abs_path = os.path.abspath(path)
        if relative_to is not None:
            abs_path = os.path.relpath(abs_path, os.path.dirname(os.path.abspath(relative_to)))
        return abs_path.replace("\\", "/")

    def _open_overlay_stage(self):
        root_layer = self.Sdf.Layer.FindOrOpen(self.output_path)
        if root_layer is None:
            root_layer = self.Sdf.Layer.CreateNew(self.output_path)
        if root_layer is None:
            raise RuntimeError(f"Failed to create USD layer: {self.output_path}")

        src_layer = self._to_layer_path(self.source_usd_path, self.output_path)
        sublayers = list(root_layer.subLayerPaths)
        if src_layer not in sublayers:
            sublayers.append(src_layer)
            root_layer.subLayerPaths = sublayers
            root_layer.Save()

        stage = self.Usd.Stage.Open(root_layer)
        if stage is None:
            raise RuntimeError(f"Failed to open composed USD stage: {root_layer.identifier}")
        stage.SetEditTarget(root_layer)
        return stage, root_layer

    def _init_metadata_scope(self) -> None:
        default_prim = self.stage.GetDefaultPrim()
        if default_prim is not None and default_prim.IsValid():
            scope_path = f"{default_prim.GetPath()}/anim"
        else:
            scope_path = "/anim"
        scope = self.stage.DefinePrim(scope_path, "Scope")
        self._frame_attr = scope.CreateAttribute("frameNum", self.Sdf.ValueTypeNames.Int, custom=True)
        src_attr = scope.CreateAttribute("sourceUsd", self.Sdf.ValueTypeNames.String, custom=True)
        copy_attr = scope.CreateAttribute("copyUsd", self.Sdf.ValueTypeNames.Bool, custom=True)
        src_attr.Set(self.source_usd_path.replace("\\", "/"))
        copy_attr.Set(self.copy_usd)

    def _touch_time(self, frame: int) -> None:
        if self._time_start is None or frame < self._time_start:
            self._time_start = frame
        if self._time_end is None or frame > self._time_end:
            self._time_end = frame
        root_layer = self.stage.GetRootLayer()
        if root_layer is not None:
            root_layer.startTimeCode = float(self._time_start)
            root_layer.endTimeCode = float(self._time_end)
        if self._edit_layer is not None and self._edit_layer != root_layer:
            self._edit_layer.startTimeCode = float(self._time_start)
            self._edit_layer.endTimeCode = float(self._time_end)

    def mark_frame(self, frame: int) -> None:
        f = int(frame)
        self._frame_attr.Set(f, f)
        self._touch_time(f)

    def define_prim(self, prim_path: str, prim_type: str = "Scope") -> bool:
        path = str(prim_path).strip()
        if not path.startswith("/"):
            raise ValueError(f"Invalid prim path: {prim_path!r}")
        prim = self.stage.DefinePrim(path, prim_type)
        return bool(prim and prim.IsValid())

    def remove_prim(self, prim_path: str) -> bool:
        path = str(prim_path).strip()
        if not path.startswith("/"):
            raise ValueError(f"Invalid prim path: {prim_path!r}")
        prim = self.stage.GetPrimAtPath(path)
        if prim is None or not prim.IsValid():
            return False
        self.stage.RemovePrim(path)
        return True

    def _resolve_value_type(self, value: Any, value_type: str | None):
        if value_type:
            t = getattr(self.Sdf.ValueTypeNames, str(value_type), None)
            if t is None:
                raise ValueError(f"Unsupported value_type: {value_type!r}")
            return t
        if isinstance(value, bool):
            return self.Sdf.ValueTypeNames.Bool
        if isinstance(value, int) and not isinstance(value, bool):
            return self.Sdf.ValueTypeNames.Int
        if isinstance(value, float):
            return self.Sdf.ValueTypeNames.Float
        if isinstance(value, str):
            return self.Sdf.ValueTypeNames.String
        arr = np.asarray(value)
        if arr.ndim == 1 and arr.size == 3:
            return self.Sdf.ValueTypeNames.Float3
        raise ValueError("Cannot infer USD type from value. Pass value_type explicitly.")

    def _coerce_value(self, value: Any, value_type):
        type_name = str(value_type)
        if "Float3" in type_name or "Color3f" in type_name:
            return _as_vec3(value, self.Gf)
        if "Bool" in type_name:
            return bool(value)
        if "Int" in type_name:
            return int(value)
        if "Float" in type_name or "Double" in type_name:
            return float(value)
        if "String" in type_name:
            return str(value)
        return value

    def _resolve_primvar_type(self, value: Any, value_type: str | None):
        if value_type:
            t = getattr(self.Sdf.ValueTypeNames, str(value_type), None)
            if t is None:
                raise ValueError(f"Unsupported value_type: {value_type!r}")
            return t

        arr = np.asarray(value)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return self.Sdf.ValueTypeNames.Float3Array
        if arr.ndim == 1 and arr.size > 0:
            if arr.dtype.kind in {"b"}:
                return self.Sdf.ValueTypeNames.BoolArray
            if arr.dtype.kind in {"i", "u"}:
                return self.Sdf.ValueTypeNames.IntArray
            if arr.dtype.kind in {"f"}:
                return self.Sdf.ValueTypeNames.FloatArray

        return self._resolve_value_type(value, None)

    def _resolve_interpolation_token(self, interpolation: str | None):
        if interpolation is None:
            return None
        key = str(interpolation).strip()
        if not key:
            return None
        token = getattr(self.UsdGeom.Tokens, key, None)
        if token is None:
            token = getattr(self.UsdGeom.Tokens, key.lower(), None)
        if token is None and key.lower() == "facevarying":
            token = self.UsdGeom.Tokens.faceVarying
        if token is None:
            raise ValueError(
                f"Unsupported interpolation: {interpolation!r}. "
                "Use one of constant/uniform/varying/vertex/faceVarying."
            )
        return token

    def _coerce_primvar_value(self, value: Any, value_type):
        type_name = str(value_type)
        if "Array" not in type_name:
            return self._coerce_value(value, value_type)

        if any(k in type_name for k in ("Color3fArray", "Float3Array", "Vector3fArray", "Point3fArray", "Normal3fArray")):
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError(f"Expected Nx3 array for {type_name}, got shape {tuple(arr.shape)}.")
            return [self.Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in arr]

        arr = np.asarray(value)
        if "BoolArray" in type_name:
            return [bool(v) for v in arr.reshape(-1).tolist()]
        if "IntArray" in type_name:
            return np.asarray(value, dtype=np.int32).reshape(-1).tolist()
        if "FloatArray" in type_name or "DoubleArray" in type_name:
            return np.asarray(value, dtype=np.float32).reshape(-1).astype(float).tolist()
        if "StringArray" in type_name or "TokenArray" in type_name:
            return [str(v) for v in arr.reshape(-1).tolist()]
        return value

    def set_custom_property(
        self,
        prim_path: str,
        name: str,
        value: Any,
        *,
        value_type: str | None = None,
        frame: int | None = None,
        custom: bool = True,
        create_prim_type: str | None = None,
    ) -> bool:
        prim = self.stage.GetPrimAtPath(prim_path)
        if (prim is None or not prim.IsValid()) and create_prim_type is not None:
            self.define_prim(prim_path, create_prim_type)
            prim = self.stage.GetPrimAtPath(prim_path)
        if prim is None or not prim.IsValid():
            return False

        usd_type = self._resolve_value_type(value, value_type)
        attr = prim.CreateAttribute(name, usd_type, custom=bool(custom))
        if not attr or not attr.IsValid():
            return False

        v = self._coerce_value(value, usd_type)
        if frame is None:
            attr.Set(v)
        else:
            f = int(frame)
            attr.Set(v, f)
            self.mark_frame(f)
        return True

    def set_primvar(
        self,
        prim_path: str,
        name: str,
        value: Any,
        *,
        value_type: str | None = None,
        interpolation: str | None = None,
        frame: int | None = None,
        create_prim_type: str | None = None,
    ) -> bool:
        prim = self.stage.GetPrimAtPath(prim_path)
        if (prim is None or not prim.IsValid()) and create_prim_type is not None:
            self.define_prim(prim_path, create_prim_type)
            prim = self.stage.GetPrimAtPath(prim_path)
        if prim is None or not prim.IsValid():
            return False

        gprim = self.UsdGeom.Gprim(prim)
        if not gprim:
            return False

        usd_type = self._resolve_primvar_type(value, value_type)
        interpolation_token = self._resolve_interpolation_token(interpolation)

        primvars_api = self.UsdGeom.PrimvarsAPI(prim)
        pv = primvars_api.GetPrimvar(name)
        if not pv:
            if interpolation_token is None:
                interpolation_token = self.UsdGeom.Tokens.constant
            pv = primvars_api.CreatePrimvar(name, usd_type, interpolation_token)
        elif interpolation_token is not None:
            pv.SetInterpolation(interpolation_token)

        coerced = self._coerce_primvar_value(value, usd_type)
        if frame is None:
            pv.Set(coerced)
        else:
            f = int(frame)
            pv.Set(coerced, f)
            self.mark_frame(f)
        return True

    def set_display_color(self, prim_path: str, color: ColorRgb, *, frame: int | None = None) -> bool:
        rgb = np.clip(np.asarray(color, dtype=np.float32), 0.0, 1.0)
        return self.set_primvar(
            prim_path=prim_path,
            name="displayColor",
            value=[[float(rgb[0]), float(rgb[1]), float(rgb[2])]],
            value_type="Color3fArray",
            interpolation="constant",
            frame=frame,
        )

    def set_display_colors(self, colors_by_prim: dict[str, ColorRgb], *, frame: int | None = None) -> dict[str, int]:
        written = 0
        missing = 0
        for prim_path, color in colors_by_prim.items():
            if self.set_display_color(prim_path, color, frame=frame):
                written += 1
            else:
                missing += 1
        return {"written": written, "missing": missing}

    def save(self) -> None:
        root_layer = self.stage.GetRootLayer()
        if root_layer is not None:
            root_layer.Save()
        if self._edit_layer is not None and self._edit_layer != root_layer:
            self._edit_layer.Save()

    def close(self) -> None:
        self.save()

    def __enter__(self) -> "_UsdLayerEditor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        self.close()


class UsdIO:
    """Minimal public USD API for mesh reading and layered editing.

    Quick start:
        from RLVometricMuscle.usd_io import UsdIO

        usd = UsdIO("data/muscle/model/bicep.usd").read()
        meshes = usd.meshes
        focus_points = usd.focus_points

        with usd.start("output/bicep.anim.usda", copy_usd=True):
            usd.add_prim("/anim/debug", "Scope")
            usd.set_custom("/anim/debug", "note", "hello")
            usd.set_runtime("activation", 0.5, frame=0)
            usd.set_primvar("/character/muscle/bicep", "displayColor", [[1.0, 0.2, 0.2]], value_type="Color3fArray")
            usd.set_color("/character/muscle/bicep", (1.0, 0.2, 0.2), frame=0)

    Reading API:
    - read(): load meshes from source USD.
    - meshes / mesh_count / focus_points / base_colors / center_shift: loaded data.

    Layer editing API:
    - start(output_path, copy_usd=True): open a layered output stage.
    - add_prim(path, prim_type="Scope")
    - remove_prim(path)
    - set_custom(path, name, value, ...)
    - set_primvar(path, name, value, ...)
    - set_runtime(name, value, ...): write under runtime scope.
    - set_color(path, color, frame=None)
    - set_colors({path: color}, frame=None)
    - save() / close()
    """

    def __init__(
        self,
        source_usd_path: str,
        *,
        root_path: str = "/",
        y_up_to_z_up: bool = False,
        center_model: bool = False,
        up_axis: Any = 2,
    ):
        self.source_usd_path = os.path.abspath(str(source_usd_path))
        self.root_path = _normalize_root_path(root_path)
        self.y_up_to_z_up = bool(y_up_to_z_up)
        self.center_model = bool(center_model)
        self.up_axis = _resolve_up_axis_index(up_axis, default=2)
        self._cached: UsdMeshSet | None = None
        self._editor: _UsdLayerEditor | None = None
        self._runtime_prim = "/anim/runtime"

        if not os.path.isfile(self.source_usd_path):
            raise FileNotFoundError(f"USD source file not found: {self.source_usd_path}")

    @staticmethod
    def is_usd_path(path: str) -> bool:
        return str(path).lower().endswith((".usd", ".usda", ".usdc", ".usdz"))

    @staticmethod
    def default_anim_path(source_usd_path: str, *, output_dir: str = "output") -> str:
        stem = os.path.splitext(os.path.basename(str(source_usd_path)))[0]
        return os.path.abspath(os.path.join(str(output_dir), f"{stem}.anim.usda"))

    def read(self) -> "UsdIO":
        if self._cached is not None:
            return self

        mesh_tuples = _load_mesh_tuples(
            self.source_usd_path,
            root_path=self.root_path,
            y_up_to_z_up=self.y_up_to_z_up,
        )
        center_shift = np.zeros(3, dtype=np.float32)
        if self.center_model:
            mesh_tuples, center_shift = _center_mesh_tuples(mesh_tuples, self.up_axis)

        self._cached = UsdMeshSet(
            meshes=_mesh_tuples_to_objects(mesh_tuples),
            center_shift=center_shift,
        )
        return self

    def _require_data(self) -> UsdMeshSet:
        if self._cached is None:
            raise RuntimeError("USD meshes are not loaded. Call read() first.")
        return self._cached

    @property
    def meshes(self) -> list[UsdMesh]:
        return self._require_data().meshes

    @property
    def mesh_count(self) -> int:
        return len(self.meshes)

    @property
    def focus_points(self) -> np.ndarray:
        return self._require_data().focus_points

    @property
    def base_colors(self) -> dict[str, ColorRgb]:
        return self._require_data().base_colors

    @property
    def center_shift(self) -> np.ndarray:
        return self._require_data().center_shift

    def start(
        self,
        output_path: str | None = None,
        *,
        copy_usd: bool = True,
        runtime_prim: str = "/anim/runtime",
    ) -> "UsdIO":
        if output_path is None:
            output_path = self.default_anim_path(self.source_usd_path)
        self.close()
        self._editor = _UsdLayerEditor(
            source_usd_path=self.source_usd_path,
            output_path=output_path,
            copy_usd=copy_usd,
        )
        self._runtime_prim = _normalize_root_path(runtime_prim)
        self.add_prim(self._runtime_prim, "Scope")
        return self

    @property
    def output_path(self) -> str | None:
        if self._editor is None:
            return None
        return self._editor.output_path

    @property
    def source_path(self) -> str:
        if self._editor is None:
            return self.source_usd_path
        return self._editor.source_usd_path

    def _require_editor(self) -> _UsdLayerEditor:
        if self._editor is None:
            raise RuntimeError("Layer editor is not open. Call start(output_path, copy_usd=...) first.")
        return self._editor

    def mark_frame(self, frame: int) -> None:
        self._require_editor().mark_frame(frame)

    def add_prim(self, prim_path: str, prim_type: str = "Scope") -> bool:
        return self._require_editor().define_prim(prim_path, prim_type)

    def remove_prim(self, prim_path: str) -> bool:
        return self._require_editor().remove_prim(prim_path)

    def set_custom(
        self,
        prim_path: str,
        name: str,
        value: Any,
        *,
        value_type: str | None = None,
        frame: int | None = None,
        custom: bool = True,
        create_prim_type: str | None = "Scope",
    ) -> bool:
        return self._require_editor().set_custom_property(
            prim_path=prim_path,
            name=name,
            value=value,
            value_type=value_type,
            frame=frame,
            custom=custom,
            create_prim_type=create_prim_type,
        )

    def set_runtime(
        self,
        name: str,
        value: Any,
        *,
        value_type: str | None = None,
        frame: int | None = None,
        custom: bool = True,
    ) -> bool:
        return self.set_custom(
            self._runtime_prim,
            name,
            value,
            value_type=value_type,
            frame=frame,
            custom=custom,
            create_prim_type="Scope",
        )

    def set_primvar(
        self,
        prim_path: str,
        name: str,
        value: Any,
        *,
        value_type: str | None = None,
        interpolation: str | None = None,
        frame: int | None = None,
        create_prim_type: str | None = None,
    ) -> bool:
        return self._require_editor().set_primvar(
            prim_path=prim_path,
            name=name,
            value=value,
            value_type=value_type,
            interpolation=interpolation,
            frame=frame,
            create_prim_type=create_prim_type,
        )

    def set_color(self, prim_path: str, color: ColorRgb, *, frame: int | None = None) -> bool:
        rgb = np.clip(np.asarray(color, dtype=np.float32), 0.0, 1.0)
        return self.set_primvar(
            prim_path=prim_path,
            name="displayColor",
            value=[[float(rgb[0]), float(rgb[1]), float(rgb[2])]],
            value_type="Color3fArray",
            interpolation="constant",
            frame=frame,
        )

    def set_colors(self, colors_by_prim: dict[str, ColorRgb], *, frame: int | None = None) -> dict[str, int]:
        return self._require_editor().set_display_colors(colors_by_prim, frame=frame)

    def save(self) -> None:
        if self._editor is None:
            return
        self._editor.save()

    def close(self) -> None:
        if self._editor is None:
            return
        self._editor.close()
        self._editor = None

    def __enter__(self) -> "UsdIO":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        self.close()


__all__ = [
    "UsdIO",
    "add_usd_arguments",
]
