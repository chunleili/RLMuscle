import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

import newton
import newton.examples
from RLVometricMuscle.usd_adapter import (
    LayeredUsdExporter,
    center_model_bbox_to_origin,
    is_usd_path,
    load_usd_render_meshes,
    shape_key_from_path,
    summarize_primvars,
)
from RLVometricMuscle.visualization import (
    ViewerVisualization,
    viewer_visual_config_from,
)


@dataclass()
class DemoConfig:
    # Default source data is Y-up. Enable conversion to improve Newton GL viewer lighting.
    y_up_to_z_up: bool = True
    gravity: float = -9.81
    center_model: bool = True
    show_ground: bool = True
    show_origin_gizmo: bool = True
    focus_keep_view: bool = True
    activation_demo_hz: float = 0.1


DEFAULT_CONFIG = DemoConfig()


def _create_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--usd-path",
        type=str,
        default="data/muscle/model/bicep.usd",
        help="Path to USD file.",
    )
    parser.add_argument(
        "--usd-root-path",
        type=str,
        default="/",
        help="USD prim path to load.",
    )
    parser.add_argument(
        "--use_my_usd_io",
        "--use-my-usd-io",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use local layer-based USD exporter instead of Newton USD viewer exporter.",
    )
    parser.add_argument(
        "--copy_usd",
        "--copy-usd",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When using --use_my_usd_io, copy source USD before writing overlay edits.",
    )
    return parser


class Example:
    def __init__(self, viewer, args, cfg: DemoConfig = DEFAULT_CONFIG):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.viewer = viewer
        self.cfg = cfg
        self.usd_path = args.usd_path
        self.usd_root_path = args.usd_root_path
        self.output_path = str(getattr(args, "output_path", "output.usd"))

        self._use_my_usd_io = bool(getattr(args, "use_my_usd_io", False))
        self._copy_usd = bool(getattr(args, "copy_usd", False))
        self._export_status: str = ""
        self._custom_usd_exporter: LayeredUsdExporter | None = None
        self._export_frame_num = 0
        self._export_first_mesh_path: str | None = None
        self._export_first_mesh_active = True

        if self._use_my_usd_io:
            if not is_usd_path(self.usd_path):
                raise ValueError("--use_my_usd_io requires --usd-path to be a USD file.")
            if str(getattr(args, "viewer", "")).lower() == "usd":
                raise ValueError("--use_my_usd_io cannot be combined with --viewer usd. Use --viewer gl or --viewer null.")

        self.model_center_shift = np.zeros(3, dtype=np.float32)
        self._activation_demo_value = 0.0
        self._activation_enabled = True
        self._activation_auto = True
        self._activation_manual_value = 0.5
        self._activation_demo_hz = float(self.cfg.activation_demo_hz)
        self._ui_panel_width = 520.0
        self._shape_id_by_path: dict[str, int] = {}
        self._activation_shape_ids: list[int] = []

        self._model_up_axis = newton.Axis.Z if self.cfg.y_up_to_z_up else newton.Axis.Y
        usd_meshes = load_usd_render_meshes(
            args.usd_path,
            root_path=args.usd_root_path,
            y_up_to_z_up=self.cfg.y_up_to_z_up,
        )
        self._mesh_base_color_by_path: dict[str, tuple[float, float, float]] = {
            mesh_path: (float(mesh_color[0]), float(mesh_color[1]), float(mesh_color[2]))
            for mesh_path, _, _, mesh_color, _ in usd_meshes
        }
        self._export_first_mesh_path = usd_meshes[0][0] if usd_meshes else None
        self.usd_primvars: dict[str, dict[str, np.ndarray | None]] = {
            mesh_path: primvars for mesh_path, _, _, _, primvars in usd_meshes
        }
        usd_meshes = self._prepare_meshes(usd_meshes)
        self._build_model(usd_meshes)

        self.usd_mesh_count = len(usd_meshes)
        self._focus_points = np.vstack([vertices for _, vertices, _, _, _ in usd_meshes])
        self._origin_axis_length = max(0.1, 0.4 * float(np.linalg.norm(np.ptp(self._focus_points, axis=0))))
        self._primvar_summary = self._build_primvar_summary()

        self.viewer.set_model(self.model)
        self._visualization = self._build_visualization()
        self._init_custom_usd_exporter()

    def _init_custom_usd_exporter(self) -> None:
        if not self._use_my_usd_io:
            return
        self._custom_usd_exporter = LayeredUsdExporter(
            source_usd_path=self.usd_path,
            output_path=self.output_path,
            copy_usd=self._copy_usd,
        )
        self._export_status = (
            f"my_usd_io enabled, output={self._custom_usd_exporter.output_path}, "
            f"copy_usd={self._copy_usd}"
        )

    def _prepare_meshes(self, meshes):
        if not self.cfg.center_model:
            return meshes

        point_sets = [vertices for _, vertices, _, _, _ in meshes]
        shifted_sets, center_shift = center_model_bbox_to_origin(
            *point_sets,
            up_axis=int(self._model_up_axis),
        )

        shifted_meshes = []
        for i, (mesh_path, _, mesh_faces, mesh_color, primvars) in enumerate(meshes):
            shifted_meshes.append((mesh_path, shifted_sets[i], mesh_faces, mesh_color, primvars))

        self.model_center_shift = center_shift
        return shifted_meshes

    def _build_model(self, meshes):
        builder = newton.ModelBuilder(up_axis=self._model_up_axis, gravity=self.cfg.gravity)

        for i, (mesh_path, mesh_vertices, mesh_faces, mesh_color, _primvars) in enumerate(meshes):
            render_mesh = newton.Mesh(
                vertices=mesh_vertices,
                indices=mesh_faces.reshape(-1),
                compute_inertia=False,
                is_solid=True,
                color=mesh_color,
                roughness=0.72,
                metallic=0.02,
            )
            shape_id = builder.add_shape_mesh(body=-1, xform=wp.transform(), mesh=render_mesh, key=shape_key_from_path(mesh_path, i))
            self._shape_id_by_path[mesh_path] = int(shape_id)
            lower = mesh_path.lower()
            if "muscle" in lower or "bicep" in lower:
                self._activation_shape_ids.append(int(shape_id))

        if not self._activation_shape_ids:
            self._activation_shape_ids = list(self._shape_id_by_path.values())

        if self.cfg.show_ground:
            builder.add_ground_plane()

        self.model = builder.finalize()
        self.state_0 = self.model.state()

    def _build_primvar_summary(self) -> dict[str, dict[str, Any]]:
        return summarize_primvars(self.usd_primvars)

    @staticmethod
    def _activation_to_color(activation: float) -> tuple[float, float, float]:
        a = float(np.clip(activation, 0.0, 1.0))
        cold = np.array([1, 1, 1], dtype=np.float32)
        hot = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        rgb = (1.0 - a) * cold + a * hot
        return float(rgb[0]), float(rgb[1]), float(rgb[2])

    def _compute_activation_demo_value(self) -> float:
        if self._activation_auto:
            omega = 2.0 * np.pi * float(self._activation_demo_hz)
            self._activation_demo_value = 0.5 * (1.0 + float(np.sin(omega * self.sim_time)))
        else:
            self._activation_demo_value = float(np.clip(self._activation_manual_value, 0.0, 1.0))
        return self._activation_demo_value

    def _compute_mesh_colors(self) -> dict[str, tuple[float, float, float]]:
        colors_by_path: dict[str, tuple[float, float, float]] = {}
        activation_ids = set(self._activation_shape_ids)
        activation_color = self._activation_to_color(self._compute_activation_demo_value())

        for mesh_path, shape_id in self._shape_id_by_path.items():
            base_color = self._mesh_base_color_by_path.get(mesh_path, (1.0, 1.0, 1.0))
            if self._activation_enabled and shape_id in activation_ids:
                colors_by_path[mesh_path] = activation_color
            else:
                colors_by_path[mesh_path] = base_color
        return colors_by_path

    def _update_display_colors(self, mesh_colors: dict[str, tuple[float, float, float]]) -> None:
        if not hasattr(self.viewer, "update_shape_colors"):
            return

        updates: dict[int, tuple[float, float, float]] = {}
        for mesh_path, shape_id in self._shape_id_by_path.items():
            updates[shape_id] = mesh_colors.get(mesh_path, (1.0, 1.0, 1.0))

        if updates:
            self.viewer.update_shape_colors(updates)

    def _write_custom_export_frame(self, mesh_colors: dict[str, tuple[float, float, float]]) -> None:
        if self._custom_usd_exporter is None:
            return

        if self._export_first_mesh_path:
            self._custom_usd_exporter.set_prim_active(self._export_first_mesh_path, self._export_first_mesh_active)

        stats = self._custom_usd_exporter.write_frame(self._export_frame_num, mesh_colors)
        self._export_status = (
            f"my_usd_io frame={self._export_frame_num}, written={stats['written']}, missing={stats['missing']}"
        )
        self._export_frame_num += 1

    def _save_custom_export(self) -> None:
        if self._custom_usd_exporter is None:
            return
        self._custom_usd_exporter.save()
        self._export_status = f"saved usd layer: {self._custom_usd_exporter.output_path}"

    def _build_visualization(self) -> ViewerVisualization:
        visual_cfg = viewer_visual_config_from(
            self.cfg,
            orbit_around_focus=True,
        )
        visuals = ViewerVisualization(viewer=self.viewer, config=visual_cfg)
        visuals.setup(
            wide_points=self._focus_points,
            focus_points=self._focus_points,
            origin_axis_length=self._origin_axis_length,
        )
        return visuals

    def gui(self, ui):
        _text = getattr(ui, "text_wrapped", ui.text)

        try:
            current_size = ui.get_window_size()
            current_h = float(getattr(current_size, "y", 0.0))
            cond_always = getattr(getattr(ui, "Cond_", None), "always", 0)
            ui.set_window_size(ui.ImVec2(float(self._ui_panel_width), current_h), cond_always)
        except Exception:
            pass

        ui.text("visualization-only mode")
        ui.text("simulate() is pass")
        ui.text("Press F to focus camera")
        ui.text(f"y_up_to_z_up={self.cfg.y_up_to_z_up}")
        ui.text(f"model_up_axis={self._model_up_axis.name}")
        _text(f"usd_path={self.usd_path}")
        _text(f"usd_root_path={self.usd_root_path}")
        ui.text(f"mesh_count={self.usd_mesh_count}")

        if self._custom_usd_exporter is not None:
            ui.text("my_usd_io=enabled")
            _text(f"output_path={self.output_path}")
            if self._export_first_mesh_path:
                changed, value = ui.checkbox("First Mesh Active (Export)", self._export_first_mesh_active)
                if changed:
                    self._export_first_mesh_active = bool(value)
            if ui.button("Save USD Layer"):
                self._save_custom_export()
                if self._export_status:
                    print(self._export_status)
            if self._export_status:
                _text(self._export_status)

        changed, value = ui.checkbox("Activation Color", self._activation_enabled)
        if changed:
            self._activation_enabled = bool(value)
        changed, value = ui.slider_float("Panel Width", float(self._ui_panel_width), 320.0, 900.0, "%.0f")
        if changed:
            self._ui_panel_width = float(np.clip(value, 320.0, 900.0))

        if self._activation_enabled:
            changed, value = ui.checkbox("Auto Activation", self._activation_auto)
            if changed:
                self._activation_auto = bool(value)
            changed, value = ui.slider_float("Activation", float(self._activation_manual_value), 0.0, 1.0, "%.3f")
            if changed:
                self._activation_manual_value = float(np.clip(value, 0.0, 1.0))
            changed, value = ui.slider_float("Frequency", float(self._activation_demo_hz), 0.05, 2.0, "%.2f")
            if changed:
                self._activation_demo_hz = float(value)
            ui.text(f"activation_demo={self._activation_demo_value:.3f}")

        ui.text("Primvar summary:")
        ui.begin_child("usd_primvar_summary_scroll", ui.ImVec2(0.0, 280.0))
        if hasattr(ui, "push_text_wrap_pos"):
            ui.push_text_wrap_pos(0.0)
        for path, primvar_info in self._primvar_summary.items():
            ui.text(f"  {path}:")
            for key, value in primvar_info.items():
                _text(f"    {key}: {value}")
        if hasattr(ui, "pop_text_wrap_pos"):
            ui.pop_text_wrap_pos()
        ui.end_child()

        _text(
            f"center_model_shift=({self.model_center_shift[0]:.3f}, "
            f"{self.model_center_shift[1]:.3f}, {self.model_center_shift[2]:.3f})"
        )

    def simulate(self):
        pass

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self._visualization.update_focus_hotkey()

        mesh_colors = self._compute_mesh_colors()

        self.viewer.begin_frame(self.sim_time)
        # Keep rendered primvars:displayColor in sync with runtime color state.
        self._update_display_colors(mesh_colors)
        self.viewer.log_state(self.state_0)
        self._visualization.log_debug_visuals()
        self.viewer.end_frame()

        self._write_custom_export_frame(mesh_colors)
        self._visualization.handle_post_frame()

    def close(self):
        if self._custom_usd_exporter is None:
            return
        self._custom_usd_exporter.close()


def main():
    parser = _create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args, cfg=DEFAULT_CONFIG)
    try:
        newton.examples.run(example, args)
    finally:
        example.close()


if __name__ == "__main__":
    main()
