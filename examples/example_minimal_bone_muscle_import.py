import argparse
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton
import newton.examples
from RLVometricMuscle.geo import Geo


@dataclass(frozen=True)
class DemoConfig:
    model_up_axis: newton.Axis = newton.Axis.Y
    gravity: float = -9.81
    center_model: bool = True
    show_ground: bool = True
    show_origin_gizmo: bool = False
    focus_keep_view: bool = True
    disable_shadows_after_first_frame: bool = True
    muscle_color_bins: int = 6
    light_color: tuple[float, float, float] = (0.5, 0.5, 0.5)
    sky_upper: tuple[float, float, float] = (0.85, 0.90, 0.95)
    sky_lower: tuple[float, float, float] = (0.95, 0.96, 0.97)


DEFAULT_CONFIG = DemoConfig()


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


def _load_tri_faces(geo: Geo) -> np.ndarray:
    if hasattr(geo, "vert"):
        faces = np.asarray(geo.vert, dtype=np.int32)
        if faces.ndim == 2 and faces.shape[1] == 3:
            return faces
    if hasattr(geo, "indices"):
        indices = np.asarray(geo.indices, dtype=np.int32)
        if indices.size % 3 != 0:
            raise ValueError("Bone mesh triangle index count must be divisible by 3.")
        return indices.reshape(-1, 3)
    raise ValueError("Failed to read triangle faces from geo file.")


def _split_bone_parts(geo: Geo, vertices: np.ndarray | None = None):
    if vertices is None:
        vertices = np.asarray(geo.positions, dtype=np.float32)
    faces = _load_tri_faces(geo)

    if hasattr(geo, "pointattr") and "muscle_id" in geo.pointattr:
        muscle_ids = np.asarray(geo.pointattr["muscle_id"])
        if muscle_ids.shape[0] == vertices.shape[0]:
            parts = []
            for bone_name in sorted(np.unique(muscle_ids).tolist()):
                vertex_mask = muscle_ids == bone_name
                tri_mask = np.all(vertex_mask[faces], axis=1)
                if not np.any(tri_mask):
                    continue
                part_faces = faces[tri_mask]
                used_vertices = np.unique(part_faces.reshape(-1))
                remap = np.full(vertices.shape[0], -1, dtype=np.int32)
                remap[used_vertices] = np.arange(used_vertices.shape[0], dtype=np.int32)
                local_vertices = vertices[used_vertices]
                local_faces = remap[part_faces]
                parts.append((str(bone_name), local_vertices, local_faces))
            if parts:
                return parts

    return [("bone", vertices, faces)]


def _bbox_min_max(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return points.min(axis=0), points.max(axis=0)


def center_model_bbox_to_origin(*point_sets: np.ndarray, up_axis: int = 2) -> tuple[list[np.ndarray], np.ndarray]:
    """Shift all point sets so their combined bbox bottom-center moves to the origin."""
    all_points = np.vstack(point_sets)
    bbox_min, bbox_max = _bbox_min_max(all_points)
    anchor = 0.5 * (bbox_min + bbox_max)
    anchor[int(up_axis)] = bbox_min[int(up_axis)]
    shifted = [pts - anchor for pts in point_sets]
    return shifted, anchor.astype(np.float32)


def _muscle_mask_to_rgb(mask_value: float) -> tuple[float, float, float]:
    # Match muscle.py tendonmask mapping: red (belly) -> white (tendon)
    r = 0.9 + mask_value * 0.1
    g = 0.1 + mask_value * 0.9
    b = 0.1 + mask_value * 0.9
    return float(r), float(g), float(b)


def _build_muscle_color_meshes(
    geo: Geo,
    vertices: np.ndarray,
    surface_tris: np.ndarray,
    num_bins: int = 6,
):
    tendon_mask = None
    if hasattr(geo, "pointattr") and "tendonmask" in geo.pointattr:
        tendon_mask = np.asarray(geo.pointattr["tendonmask"], dtype=np.float32)
    elif hasattr(geo, "tendonmask"):
        tendon_mask = np.asarray(geo.tendonmask, dtype=np.float32)

    if tendon_mask is None or tendon_mask.shape[0] != vertices.shape[0]:
        return [("muscle", vertices, surface_tris, (0.2, 0.6, 1.0))]

    tri_mask = tendon_mask[surface_tris].mean(axis=1)
    bin_ids = np.clip((tri_mask * num_bins).astype(np.int32), 0, num_bins - 1)

    meshes = []
    for b in range(num_bins):
        selected = surface_tris[bin_ids == b]
        if selected.shape[0] == 0:
            continue
        color_mask = (b + 0.5) / num_bins
        color = _muscle_mask_to_rgb(float(color_mask))
        meshes.append((f"muscle_bin_{b}", vertices, selected, color))

    if not meshes:
        return [("muscle", vertices, surface_tris, (0.2, 0.6, 1.0))]
    return meshes


def _camera_angles_from_look_dir(viewer, look_dir: np.ndarray) -> tuple[float, float]:
    d = look_dir / (np.linalg.norm(look_dir) + 1.0e-12)
    up_axis = getattr(getattr(viewer, "camera", None), "up_axis", 2)

    if up_axis == 0:  # X-up
        pitch = np.degrees(np.arcsin(np.clip(d[0], -1.0, 1.0)))
        yaw = np.degrees(np.arctan2(d[2], d[1]))
    elif up_axis == 1:  # Y-up
        pitch = np.degrees(np.arcsin(np.clip(d[1], -1.0, 1.0)))
        yaw = np.degrees(np.arctan2(d[2], d[0]))
    else:  # Z-up
        pitch = np.degrees(np.arcsin(np.clip(d[2], -1.0, 1.0)))
        yaw = np.degrees(np.arctan2(d[1], d[0]))
    return float(pitch), float(yaw)


def _create_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--muscle-geo",
        type=str,
        default="data/muscle/model/bicep.geo",
        help="Path to muscle geo file.",
    )
    parser.add_argument(
        "--bone-geo",
        type=str,
        default="data/muscle/model/bicep_bone.geo",
        help="Path to bone geo file.",
    )
    return parser


class Example:
    def __init__(self, viewer, args, cfg: DemoConfig = DEFAULT_CONFIG):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.viewer = viewer
        self.cfg = cfg

        self._prev_f_down = False
        self.is_first_frame = True
        self.model_center_shift = np.zeros(3, dtype=np.float32)

        muscle_geo, muscle_vertices, muscle_tets, bone_geo, bone_vertices = self._load_geometry(args)
        muscle_vertices, bone_vertices, muscle_meshes, bone_parts = self._prepare_geometry(
            muscle_geo,
            muscle_vertices,
            muscle_tets,
            bone_geo,
            bone_vertices,
        )
        self._build_model(muscle_meshes, bone_parts)

        self._muscle_points = muscle_vertices
        self._all_points = np.vstack([muscle_vertices, bone_vertices])
        self._origin_axis_length = max(0.1, 0.4 * float(np.linalg.norm(np.ptp(self._all_points, axis=0))))

        self.viewer.set_model(self.model)
        self._init_origin_gizmo()
        self._configure_visual_defaults()

    def _load_geometry(self, args):
        muscle_geo = Geo(args.muscle_geo)
        muscle_vertices = np.asarray(muscle_geo.positions, dtype=np.float32)
        muscle_tets = np.asarray(muscle_geo.vert, dtype=np.int32)
        if muscle_tets.ndim != 2 or muscle_tets.shape[1] != 4:
            raise ValueError(f"Expected tetrahedral muscle mesh, got shape {muscle_tets.shape}.")

        bone_geo = Geo(args.bone_geo)
        bone_vertices = np.asarray(bone_geo.positions, dtype=np.float32)
        return muscle_geo, muscle_vertices, muscle_tets, bone_geo, bone_vertices

    def _prepare_geometry(self, muscle_geo, muscle_vertices, muscle_tets, bone_geo, bone_vertices):
        if self.cfg.center_model:
            shifted_sets, center_shift = center_model_bbox_to_origin(
                muscle_vertices,
                bone_vertices,
                up_axis=int(self.cfg.model_up_axis),
            )
            muscle_vertices, bone_vertices = shifted_sets
            self.model_center_shift = center_shift

        muscle_tets = _fix_tet_winding(muscle_vertices, muscle_tets.copy())
        muscle_surface_tris = _extract_surface_tris(muscle_tets)
        muscle_meshes = _build_muscle_color_meshes(
            muscle_geo,
            muscle_vertices,
            muscle_surface_tris,
            num_bins=self.cfg.muscle_color_bins,
        )
        bone_parts = _split_bone_parts(bone_geo, vertices=bone_vertices)

        return muscle_vertices, bone_vertices, muscle_meshes, bone_parts

    def _build_model(self, muscle_meshes, bone_parts):
        builder = newton.ModelBuilder(up_axis=self.cfg.model_up_axis, gravity=self.cfg.gravity)

        for mesh_name, mesh_vertices, mesh_faces, mesh_color in muscle_meshes:
            muscle_mesh = newton.Mesh(
                vertices=mesh_vertices,
                indices=mesh_faces.reshape(-1),
                compute_inertia=False,
                is_solid=True,
                color=mesh_color,
                roughness=0.65,
                metallic=0.0,
            )
            builder.add_shape_mesh(body=-1, xform=wp.transform(), mesh=muscle_mesh, key=mesh_name)

        bone_colors = [
            (0.75, 0.78, 0.82),
            (0.68, 0.72, 0.76),
            (0.80, 0.83, 0.86),
        ]
        for i, (bone_name, part_vertices, part_faces) in enumerate(bone_parts):
            part_mesh = newton.Mesh(
                vertices=part_vertices,
                indices=part_faces.reshape(-1),
                compute_inertia=False,
                is_solid=True,
                color=bone_colors[i % len(bone_colors)],
                roughness=0.8,
                metallic=0.02,
            )
            builder.add_shape_mesh(body=-1, xform=wp.transform(), mesh=part_mesh, key=f"bone_{bone_name}")

        if self.cfg.show_ground:
            builder.add_ground_plane()

        self.model = builder.finalize()
        self.state_0 = self.model.state()

    def _configure_visual_defaults(self):
        self._set_camera_wide()
        self._set_light()
        self.focus_camera_on_model()

    def _init_origin_gizmo(self):
        self._origin_starts = None
        self._origin_ends = None
        self._origin_colors = None
        if not self.cfg.show_origin_gizmo:
            return

        starts = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        L = self._origin_axis_length
        ends = np.array(
            [
                [L, 0.0, 0.0],
                [0.0, L, 0.0],
                [0.0, 0.0, L],
            ],
            dtype=np.float32,
        )
        colors = np.array(
            [
                [1.0, 0.2, 0.2],  # X
                [0.2, 1.0, 0.2],  # Y
                [0.2, 0.4, 1.0],  # Z
            ],
            dtype=np.float32,
        )

        device = getattr(self.viewer, "device", None)
        self._origin_starts = wp.array(starts, dtype=wp.vec3, device=device)
        self._origin_ends = wp.array(ends, dtype=wp.vec3, device=device)
        self._origin_colors = wp.array(colors, dtype=wp.vec3, device=device)

    def _set_light(self):
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer._light_color = self.cfg.light_color
            self.viewer.renderer.sky_upper = self.cfg.sky_upper
            self.viewer.renderer.sky_lower = self.cfg.sky_lower

    def _set_camera_wide(self):
        if not hasattr(self.viewer, "set_camera"):
            return

        bbox_min, bbox_max = _bbox_min_max(self._all_points)
        center = 0.5 * (bbox_min + bbox_max)
        diag = float(np.linalg.norm(bbox_max - bbox_min))
        dist = max(1.2, 2.0 * diag)
        cam_pos = np.array(
            [center[0] + dist, center[1] + 0.35 * dist, center[2] + 0.4 * dist],
            dtype=np.float32,
        )
        look_dir = center - cam_pos
        pitch, yaw = _camera_angles_from_look_dir(self.viewer, look_dir)
        self.viewer.set_camera(pos=wp.vec3(*cam_pos.tolist()), pitch=pitch, yaw=yaw)

        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 45.0

    def focus_camera_on_model(self):
        if not hasattr(self.viewer, "set_camera"):
            return

        bbox_min, bbox_max = _bbox_min_max(self._muscle_points)
        center = 0.5 * (bbox_min + bbox_max)
        diag = float(np.linalg.norm(bbox_max - bbox_min))
        camera = getattr(self.viewer, "camera", None)

        if self.cfg.focus_keep_view and camera is not None:
            front = np.array(camera.get_front(), dtype=np.float32)
            front_norm = float(np.linalg.norm(front))
            if front_norm > 1.0e-6:
                front = front / front_norm
                fov = float(getattr(camera, "fov", 45.0))
                fov = float(np.clip(fov, 15.0, 90.0))
                tan_half = max(np.tan(np.radians(0.5 * fov)), 1.0e-3)
                dist = max(0.6, 0.75 * diag / tan_half)
                cam_pos = center - front * dist
                self.viewer.set_camera(
                    pos=wp.vec3(*cam_pos.tolist()),
                    pitch=float(camera.pitch),
                    yaw=float(camera.yaw),
                )
                return

        up_axis = int(getattr(camera, "up_axis", 2)) if camera is not None else 2
        side_axis = (up_axis + 1) % 3
        cam_pos = center.copy()
        cam_pos[up_axis] += max(0.5, 1.2 * diag)
        cam_pos[side_axis] += 0.35 * max(diag, 0.2)
        look_dir = center - cam_pos
        pitch, yaw = _camera_angles_from_look_dir(self.viewer, look_dir)
        self.viewer.set_camera(pos=wp.vec3(*cam_pos.tolist()), pitch=pitch, yaw=yaw)

    def _handle_first_frame_visual_overrides(self):
        if not self.is_first_frame:
            return

        if self.cfg.disable_shadows_after_first_frame and hasattr(self.viewer, "renderer"):
            if hasattr(self.viewer.renderer, "draw_shadows"):
                self.viewer.renderer.draw_shadows = False

        self.is_first_frame = False

    def _log_debug_visuals(self):
        if self.cfg.show_origin_gizmo and self._origin_starts is not None:
            self.viewer.log_lines(
                "/debug/origin_axes",
                self._origin_starts,
                self._origin_ends,
                self._origin_colors,
                width=0.01,
                hidden=False,
            )

    def gui(self, ui):
        ui.text("visualization-only mode")
        ui.text("simulate() is pass")
        ui.text("Press F to focus camera on muscle")
        ui.text(f"center_model={self.cfg.center_model}, up_axis={self.cfg.model_up_axis.name}")
        ui.text(f"show_ground={self.cfg.show_ground}, show_origin_gizmo={self.cfg.show_origin_gizmo}")
        ui.text(
            f"center_model_shift=({self.model_center_shift[0]:.3f}, "
            f"{self.model_center_shift[1]:.3f}, {self.model_center_shift[2]:.3f})"
        )

    def simulate(self):
        pass

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        f_down = bool(self.viewer.is_key_down("f")) if hasattr(self.viewer, "is_key_down") else False
        if f_down and not self._prev_f_down:
            self.focus_camera_on_model()
        self._prev_f_down = f_down

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self._log_debug_visuals()
        self.viewer.end_frame()

        self._handle_first_frame_visual_overrides()


def main():
    parser = _create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args, cfg=DEFAULT_CONFIG)
    newton.examples.run(example, args)


if __name__ == "__main__":
    main()
