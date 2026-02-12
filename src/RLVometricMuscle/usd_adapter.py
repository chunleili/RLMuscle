from __future__ import annotations

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


def _tet_indices_from_prim(prim) -> np.ndarray | None:
    for attr_name in ("tetraIndices", "tetVertexIndices", "tetIndices"):
        attr = prim.GetAttribute(attr_name)
        if not attr or not attr.IsValid():
            continue
        value = attr.Get()
        if value is None:
            continue
        raw = np.asarray(value, dtype=np.int32)
        if raw.size == 0:
            continue
        if raw.ndim == 1 and raw.size % 4 == 0:
            tets = raw.reshape(-1, 4)
        elif raw.ndim == 2 and raw.shape[1] == 4:
            tets = raw
        else:
            continue
        return np.asarray(tets, dtype=np.int32)
    return None


def _surface_tris_from_prim(prim) -> np.ndarray | None:
    attr = prim.GetAttribute("surfaceFaceVertexIndices")
    if not attr or not attr.IsValid():
        return None

    value = attr.Get()
    if value is None:
        return None

    raw = np.asarray(value, dtype=np.int32)
    if raw.size == 0:
        return None
    if raw.ndim == 1 and raw.size % 3 == 0:
        tris = raw.reshape(-1, 3)
    elif raw.ndim == 2 and raw.shape[1] == 3:
        tris = raw
    else:
        return None
    return np.asarray(tris, dtype=np.int32)


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
        surface_tris = _surface_tris_from_prim(prim)
        tet_indices = _tet_indices_from_prim(prim)
        if not is_mesh and surface_tris is None and tet_indices is None:
            continue

        primvars = _read_all_primvars(prim)

        points = None
        tris = None

        if is_mesh:
            parsed = _load_mesh_with_newton_usd(prim)
            if parsed is None:
                parsed = _load_mesh_fallback(prim, UsdGeom)
            if parsed is not None:
                points, tris = parsed
        else:
            points_attr = prim.GetAttribute("points")
            if points_attr and points_attr.IsValid():
                points_value = points_attr.Get()
                if points_value is not None:
                    points = np.asarray(points_value, dtype=np.float32)
                    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
                        points = None

            if points is not None:
                if surface_tris is not None:
                    tris = surface_tris

                # For TetMesh-like assets, extract surface triangles from tetrahedra.
                if (tris is None or tris.size == 0) and tet_indices is not None and tet_indices.shape[0] > 0:
                    valid = np.all((tet_indices >= 0) & (tet_indices < points.shape[0]), axis=1)
                    tet_indices = tet_indices[valid]
                    if tet_indices.shape[0] > 0:
                        tet_indices = _fix_tet_winding(points, tet_indices.copy())
                        tris = _extract_surface_tris(tet_indices)

        if points is None or tris is None or tris.size == 0:
            continue

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
