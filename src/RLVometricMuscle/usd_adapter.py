from __future__ import annotations

import os
from typing import Any

import numpy as np

_USE_NEWTON_USD_GET_MESH = True


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
        # Fallback for opposite vectors.
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
        # Rodrigues at pi: R = I + 2*K*K
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
    # Oriented boundary of a positively oriented tetrahedron [0,1,2,3]:
    # (1,2,3), (0,3,2), (0,1,3), (0,2,1)
    faces = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
    counts = {}
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


def _mesh_color_from_path(path: str) -> tuple[float, float, float]:
    name = path.lower()
    if "bone" in name:
        return (0.75, 0.78, 0.82)
    if "muscle" in name or "bicep" in name:
        return (0.92, 0.35, 0.35)
    return (0.35, 0.62, 0.95)


def _mesh_color_from_display_color(display_color: np.ndarray | None) -> tuple[float, float, float] | None:
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


def _read_all_primvars(prim) -> dict[str, np.ndarray | None]:
    primvars: dict[str, np.ndarray | None] = {}
    for name in _list_primvar_names(prim):
        primvars[name] = _read_primvar(prim, name)
    return primvars


def _load_mesh_with_newton_usd(prim) -> tuple[np.ndarray, np.ndarray] | None:
    # Reuse Newton public USD mesh parser where possible.
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
        # If parsing fails once in this runtime (e.g. missing optional kernels),
        # disable this path and rely on the local fallback parser.
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

    # Fallback if surface faces are not authored and ComputeSurfaceFaces is unavailable/failed.
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


def load_usd_render_meshes(
    usd_path: str,
    root_path: str = "/",
    y_up_to_z_up: bool = False,
):
    try:
        from pxr import Gf, Usd, UsdGeom
    except ImportError as exc:
        raise ImportError(
            "USD import requires `pxr` (`usd-core`). Install with `uv add usd-core` or `pip install usd-core`."
        ) from exc

    if not root_path:
        root_path = "/"
    if not root_path.startswith("/"):
        raise ValueError("USD root path must start with '/'.")

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise FileNotFoundError(f"Failed to open USD file: {usd_path}")

    stage_up_axis = _axis_index_from_any(str(UsdGeom.GetStageUpAxis(stage)))
    up_axis_rotation = None
    if y_up_to_z_up and stage_up_axis == 1:
        up_axis_rotation = _rotation_matrix_align_up(1, 2)

    if root_path == "/":
        prim_iter = stage.Traverse()
    else:
        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim.IsValid():
            raise ValueError(f"USD root path does not exist: {root_path}")
        prim_iter = Usd.PrimRange(root_prim)

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    meshes: list[tuple[str, np.ndarray, np.ndarray, tuple[float, float, float], dict[str, np.ndarray | None]]] = []
    for prim in prim_iter:
        is_mesh = prim.IsA(UsdGeom.Mesh)
        is_tetmesh = prim.IsA(UsdGeom.TetMesh)
        if not is_mesh and not is_tetmesh:
            continue

        primvars = _read_all_primvars(prim)

        if is_mesh:
            parsed = _load_mesh_geometry(prim, UsdGeom)
        elif is_tetmesh:
            parsed = _load_tetmesh_geometry(prim, UsdGeom)
        else:
            parsed = None

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


def summarize_primvars(primvars_by_path: dict[str, dict[str, np.ndarray | None]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for path, primvars in primvars_by_path.items():
        path_summary: dict[str, Any] = {}
        for key, value in sorted(primvars.items()):
            if value is None:
                path_summary[key] = "None"
                continue
            arr = np.asarray(value)
            shape = tuple(arr.shape)
            if arr.dtype.kind in {"U", "S", "O"}:
                unique = np.unique(arr).tolist()
                path_summary[key] = f"shape={shape}, unique={unique}"
            else:
                path_summary[key] = f"shape={shape}, dtype={arr.dtype}"
        summary[path] = path_summary
    return summary


def is_usd_path(path: str) -> bool:
    """Return *True* if *path* has a recognised USD file extension."""
    return str(path).lower().endswith((".usd", ".usda", ".usdc", ".usdz"))


def shape_key_from_path(path: str, mesh_index: int) -> str:
    """Derive a Newton shape key from a USD prim path."""
    sanitized = path.strip("/").replace("/", "_") or "mesh"
    return f"usd_mesh_{mesh_index}_{sanitized}"


def center_model_bbox_to_origin(
    *point_sets: np.ndarray,
    up_axis: int = 2,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Shift point sets so their combined bounding-box bottom-centre sits at the origin."""
    all_points = np.vstack(point_sets)
    bbox_min, bbox_max = all_points.min(axis=0), all_points.max(axis=0)
    anchor = 0.5 * (bbox_min + bbox_max)
    anchor[int(up_axis)] = bbox_min[int(up_axis)]
    shifted = [pts - anchor for pts in point_sets]
    return shifted, anchor.astype(np.float32)


class LayeredUsdExporter:
    """Layer-based USD writer that edits an existing USD scene instead of rebuilding it."""

    def __init__(self, source_usd_path: str, output_path: str, copy_usd: bool = False):
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

        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if self.copy_usd:
            self._copy_source_to_output_dir()

        self.stage, self._edit_layer = self._open_overlay_stage()

        default_prim = self.stage.GetDefaultPrim()
        if default_prim is not None and default_prim.IsValid():
            scope_path = f"{default_prim.GetPath()}/rlmuscle_export"
        else:
            scope_path = "/rlmuscle_export"

        self._scope = self.stage.DefinePrim(scope_path, "Scope")
        self._frame_attr = self._scope.CreateAttribute("frameNum", self.Sdf.ValueTypeNames.Int, custom=True)
        source_attr = self._scope.CreateAttribute("sourceUsd", self.Sdf.ValueTypeNames.String, custom=True)
        copy_attr = self._scope.CreateAttribute("copyUsd", self.Sdf.ValueTypeNames.Bool, custom=True)
        source_attr.Set(self.source_usd_path.replace("\\", "/"))
        copy_attr.Set(bool(self.copy_usd))
        self._time_code_start: int | None = None
        self._time_code_end: int | None = None

    def _copy_source_to_output_dir(self) -> None:
        """Copy the source USD file into the same directory as *output_path*."""
        import shutil

        dest = os.path.join(os.path.dirname(self.output_path), os.path.basename(self.source_usd_path))
        dest = os.path.abspath(dest)
        if os.path.normcase(dest) != os.path.normcase(self.source_usd_path):
            shutil.copy2(self.source_usd_path, dest)
        self.source_usd_path = dest

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

        source_layer_path = self._to_layer_path(self.source_usd_path, self.output_path)
        sublayers = list(root_layer.subLayerPaths)
        if source_layer_path not in sublayers:
            sublayers.append(source_layer_path)
            root_layer.subLayerPaths = sublayers
            root_layer.Save()

        stage = self.Usd.Stage.Open(root_layer)
        if stage is None:
            raise RuntimeError(f"Failed to open composed USD stage: {root_layer.identifier}")
        stage.SetEditTarget(root_layer)
        return stage, root_layer

    def ensure_scope(self, prim_path: str):
        path = str(prim_path).strip()
        if not path.startswith("/"):
            raise ValueError(f"Invalid USD prim path: {prim_path!r}")
        return self.stage.DefinePrim(path, "Scope")

    def set_prim_active(self, prim_path: str, active: bool) -> bool:
        path = str(prim_path).strip()
        if not path.startswith("/"):
            raise ValueError(f"Invalid USD prim path: {prim_path!r}")
        prim = self.stage.OverridePrim(path)
        if prim is None or not prim.IsValid():
            return False
        prim.SetActive(bool(active))
        return True

    def _set_display_color(self, prim_path: str, color: tuple[float, float, float], frame_index: int) -> bool:
        prim = self.stage.GetPrimAtPath(prim_path)
        if prim is None or not prim.IsValid():
            return False

        gprim = self.UsdGeom.Gprim(prim)
        if not gprim:
            return False

        primvars_api = self.UsdGeom.PrimvarsAPI(prim)
        primvar = primvars_api.GetPrimvar("displayColor")
        if not primvar:
            primvar = primvars_api.CreatePrimvar(
                "displayColor",
                self.Sdf.ValueTypeNames.Color3fArray,
                self.UsdGeom.Tokens.constant,
            )
        else:
            primvar.SetInterpolation(self.UsdGeom.Tokens.constant)

        rgb = np.clip(np.asarray(color, dtype=np.float32), 0.0, 1.0)
        primvar.Set([self.Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2]))], int(frame_index))
        return True

    def _update_time_code_range(self, frame: int) -> None:
        if self._time_code_start is None or frame < self._time_code_start:
            self._time_code_start = frame
        if self._time_code_end is None or frame > self._time_code_end:
            self._time_code_end = frame

        root_layer = self.stage.GetRootLayer()
        start = float(self._time_code_start)
        end = float(self._time_code_end)
        if root_layer is not None:
            root_layer.startTimeCode = start
            root_layer.endTimeCode = end
        if self._edit_layer is not None and self._edit_layer != root_layer:
            self._edit_layer.startTimeCode = start
            self._edit_layer.endTimeCode = end

    def write_frame(self, frame_index: int, mesh_colors: dict[str, tuple[float, float, float]]) -> dict[str, int]:
        frame = int(frame_index)
        self._frame_attr.Set(frame, frame)
        self._update_time_code_range(frame)

        written = 0
        missing = 0
        for prim_path, color in mesh_colors.items():
            if self._set_display_color(prim_path, color, frame):
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
