"""Teaching example for USD IO with a minimal, step-by-step workflow.

This example demonstrates three core tasks:
1. Read render meshes from a source USD file using `UsdIO.read()`.
2. Build a Newton model from those meshes for visualization.
3. Write runtime edits into a layered `.anim.usda` file:
   - per-frame display colors
   - per-muscle activation properties

Usage (GUI):
    .\\.venv\\Scripts\\python.exe examples\\example_usd_io.py --viewer gl

Usage (headless layered export):
    .\\.venv\\Scripts\\python.exe examples\\example_usd_io.py --viewer null --headless --num-frames 120 --use-layered-usd
"""

import argparse
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton
import newton.examples
from RLVometricMuscle.usd_io import UsdIO
from RLVometricMuscle.visualization import ViewerVisualization, viewer_visual_config_from


def _shape_key_from_path(path: str, mesh_index: int) -> str:
    """Build a stable debug-friendly key from a USD prim path."""
    name = path.strip("/").replace("/", "_") or "mesh"
    return f"usd_mesh_{mesh_index}_{name}"


@dataclass()
class DemoConfig:
    """Rendering/demo options for this teaching script."""

    y_up_to_z_up: bool = True
    gravity: float = -9.81
    center_model: bool = True
    show_ground: bool = True
    show_origin_gizmo: bool = True
    focus_keep_view: bool = True
    activation_demo_hz: float = 0.1


DEFAULT_CONFIG = DemoConfig()


def _create_parser() -> argparse.ArgumentParser:
    """Create CLI arguments used by `newton.examples`."""
    parser = newton.examples.create_parser()
    parser.set_defaults(output_path="output/example_usd_io.anim.usda")
    parser.add_argument("--usd-path", type=str, default="data/muscle/model/bicep.usd", help="Path to USD file.")
    parser.add_argument("--usd-root-path", type=str, default="/", help="USD prim path to load.")
    parser.add_argument(
        "--use_layered_usd",
        "--use-layered-usd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write runtime edits into a layered USD file (requires --viewer gl/null).",
    )
    parser.add_argument(
        "--copy_usd",
        "--copy-usd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy source USD to output directory before writing layered edits.",
    )
    return parser


class Example:
    """Minimal simulation loop that demonstrates `UsdIO` end-to-end."""

    def __init__(self, viewer, args, cfg: DemoConfig = DEFAULT_CONFIG):
        self.viewer = viewer
        self.cfg = cfg
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_frame = 0

        self.usd_path = str(args.usd_path)
        self.usd_root_path = str(args.usd_root_path)
        self.output_path = str(getattr(args, "output_path", "output/example_usd_io.anim.usda"))
        self._use_layered_usd = bool(getattr(args, "use_layered_usd", True))
        self._copy_usd = bool(getattr(args, "copy_usd", True))
        self._export_status = ""

        # Step 0: Validate CLI choices for layered editing.
        if self._use_layered_usd and not UsdIO.is_usd_path(self.usd_path):
            raise ValueError("--use_layered_usd requires --usd-path to be a USD file.")
        if self._use_layered_usd and str(getattr(args, "viewer", "")).lower() == "usd":
            raise ValueError("--use_layered_usd cannot be combined with --viewer usd. Use --viewer gl or --viewer null.")

        self._activation_enabled = True
        self._activation_hz = float(self.cfg.activation_demo_hz)
        self._activation_value = 0.0
        self.model_center_shift = np.zeros(3, dtype=np.float32)
        self._shape_id_by_path: dict[str, int] = {}
        self._activation_mesh_paths: list[str] = []
        self._mesh_base_color_by_path: dict[str, tuple[float, float, float]] = {}

        # Step 1: Read USD meshes with a single entry point.
        model_up_axis = newton.Axis.Z if self.cfg.y_up_to_z_up else newton.Axis.Y
        self._usd_io = UsdIO(
            source_usd_path=self.usd_path,
            root_path=self.usd_root_path,
            y_up_to_z_up=self.cfg.y_up_to_z_up,
            center_model=self.cfg.center_model,
            up_axis=int(model_up_axis),
        ).read()

        # Step 2: Access loaded data through simple properties.
        self.model_center_shift = self._usd_io.center_shift
        self._mesh_base_color_by_path = self._usd_io.base_colors
        self.usd_mesh_count = self._usd_io.mesh_count
        self._focus_points = self._usd_io.focus_points
        self._origin_axis_length = max(0.1, 0.4 * float(np.linalg.norm(np.ptp(self._focus_points, axis=0))))

        # Step 3: Build Newton render model from USD meshes.
        self._build_model(self._usd_io.meshes, model_up_axis)
        self.viewer.set_model(self.model)
        self._visualization = self._build_visualization()

        # Step 4: Optionally start layered USD writing.
        self._init_layered_usd()

    def _build_model(self, meshes, model_up_axis) -> None:
        """Create render shapes from loaded USD meshes."""
        builder = newton.ModelBuilder(up_axis=model_up_axis, gravity=self.cfg.gravity)
        for i, mesh in enumerate(meshes):
            render_mesh = newton.Mesh(
                vertices=mesh.vertices,
                indices=mesh.faces.reshape(-1),
                compute_inertia=False,
                is_solid=True,
                color=mesh.color,
                roughness=0.72,
                metallic=0.02,
            )
            shape_id = builder.add_shape_mesh(
                body=-1,
                xform=wp.transform(),
                mesh=render_mesh,
                key=_shape_key_from_path(mesh.path, i),
            )
            self._shape_id_by_path[mesh.path] = int(shape_id)
            if "muscle" in mesh.path.lower() or "bicep" in mesh.path.lower():
                self._activation_mesh_paths.append(mesh.path)

        if not self._activation_mesh_paths:
            self._activation_mesh_paths = list(self._shape_id_by_path.keys())
        if self.cfg.show_ground:
            builder.add_ground_plane()
        self.model = builder.finalize()
        self.state_0 = self.model.state()

    def _init_layered_usd(self) -> None:
        """Open output layer and write constant metadata."""
        if not self._use_layered_usd:
            return
        self._usd_io.start(self.output_path, copy_usd=self._copy_usd)
        self._usd_io.set_runtime("meshCount", self.usd_mesh_count)
        self._export_status = (
            f"layered_usd enabled, output={self._usd_io.output_path}, copy_usd={self._copy_usd}"
        )

    def _activation_to_color(self, value: float) -> tuple[float, float, float]:
        """Map activation in [0,1] to a color for quick visual feedback."""
        a = float(np.clip(value, 0.0, 1.0))
        cold = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        hot = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        rgb = (1.0 - a) * cold + a * hot
        return float(rgb[0]), float(rgb[1]), float(rgb[2])

    def _compute_activation(self) -> float:
        """Generate a simple sinusoidal activation signal."""
        if not self._activation_enabled:
            self._activation_value = 0.0
            return self._activation_value
        omega = 2.0 * np.pi * float(self._activation_hz)
        self._activation_value = 0.5 * (1.0 + float(np.sin(omega * self.sim_time)))
        return self._activation_value

    def _compute_mesh_colors(self) -> dict[str, tuple[float, float, float]]:
        """Apply activation color to selected meshes."""
        colors = dict(self._mesh_base_color_by_path)
        if not self._activation_enabled:
            return colors
        activation_color = self._activation_to_color(self._compute_activation())
        activation_paths = set(self._activation_mesh_paths)
        for mesh_path in self._shape_id_by_path.keys():
            if mesh_path in activation_paths:
                colors[mesh_path] = activation_color
        return colors

    def _compute_activation_by_mesh(self) -> dict[str, float]:
        """Build per-muscle activation values (demo currently uses one shared signal)."""
        value = float(self._activation_value)
        if not self._activation_enabled:
            value = 0.0
        return {mesh_path: value for mesh_path in self._activation_mesh_paths}

    def _update_viewer_colors(self, mesh_colors: dict[str, tuple[float, float, float]]) -> None:
        """Mirror runtime color state to the active viewer."""
        if not hasattr(self.viewer, "update_shape_colors"):
            return
        updates: dict[int, tuple[float, float, float]] = {}
        for mesh_path, shape_id in self._shape_id_by_path.items():
            updates[shape_id] = mesh_colors.get(mesh_path, (1.0, 1.0, 1.0))
        if updates:
            self.viewer.update_shape_colors(updates)

    def _write_layer_frame(
        self,
        frame: int,
        mesh_colors: dict[str, tuple[float, float, float]],
        activation_by_mesh: dict[str, float],
    ) -> None:
        """Write per-frame colors and per-muscle activation into output USD layer."""
        if not self._use_layered_usd:
            return
        stats = self._usd_io.set_colors(mesh_colors, frame=frame)
        activation_written = 0
        activation_missing = 0
        for mesh_path, activation in activation_by_mesh.items():
            ok = self._usd_io.set_custom(
                mesh_path,
                "activation",
                float(activation),
                frame=frame,
                custom=True,
                create_prim_type=None,
            )
            if ok:
                activation_written += 1
            else:
                activation_missing += 1
        self._export_status = (
            f"layered_usd frame={frame}, "
            f"color_written={stats['written']}, color_missing={stats['missing']}, "
            f"activation_written={activation_written}, activation_missing={activation_missing}"
        )

    def _build_visualization(self) -> ViewerVisualization:
        """Create helper visuals (focus points, axes)."""
        visual_cfg = viewer_visual_config_from(self.cfg, orbit_around_focus=True)
        visuals = ViewerVisualization(viewer=self.viewer, config=visual_cfg)
        visuals.setup(
            wide_points=self._focus_points,
            focus_points=self._focus_points,
            origin_axis_length=self._origin_axis_length,
        )
        return visuals

    def gui(self, ui):
        """Simple UI to show parameters and force-save the USD layer."""
        ui.text("USD IO teaching example")
        ui.text("Press F to focus camera")
        ui.text(f"usd_path={self.usd_path}")
        ui.text(f"mesh_count={self.usd_mesh_count}")
        if self._use_layered_usd:
            ui.text("layered_usd=enabled")
            ui.text(f"output_path={self.output_path}")
            if ui.button("Save USD Layer"):
                self._usd_io.save()
                print(f"USD layer saved to {self._usd_io.output_path} at frame {self.sim_frame}")

        changed, value = ui.checkbox("Activation Color", self._activation_enabled)
        if changed:
            self._activation_enabled = bool(value)
        changed, value = ui.slider_float("Frequency", float(self._activation_hz), 0.05, 2.0, "%.2f")
        if changed:
            self._activation_hz = float(value)
        ui.text(f"activation={self._activation_value:.3f}")

    def step(self):
        self.sim_frame += 1
        self.sim_time += self.frame_dt

    def render(self):
        self._visualization.update_focus_hotkey()
        mesh_colors = self._compute_mesh_colors()
        activation_by_mesh = self._compute_activation_by_mesh()

        self.viewer.begin_frame(self.sim_time)
        self._update_viewer_colors(mesh_colors)
        self.viewer.log_state(self.state_0)
        self._visualization.log_debug_visuals()
        self.viewer.end_frame()

        self._write_layer_frame(self.sim_frame, mesh_colors, activation_by_mesh)
        self._visualization.handle_post_frame()

    def close(self):
        self._usd_io.close()


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
