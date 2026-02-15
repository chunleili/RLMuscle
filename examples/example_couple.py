"""Teaching example: simple muscle-bone coupling with Warp muscle core + Newton solver.

This demo uses:
1. Newton `SolverFeatherstone` for rigid-body pendulum dynamics.
2. `SolverVolumetricMuscle` (Warp) for volumetric muscle particles.
3. A light coupling loop: muscle activation -> motor torque on the pendulum.
4. Layered USD output for per-frame runtime signals.

Usage (GUI):
    .venv/Scripts/python.exe examples/example_couple.py --viewer gl

Usage (headless):
    .venv/Scripts/python.exe examples/example_couple.py --viewer null --headless --num-frames 120 --use-layered-usd
"""

import argparse

import numpy as np
import warp as wp

import newton
import newton.examples
from RLVometricMuscle.solver_muscle_bone_coupled import SolverMuscleBoneCoupled
from RLVometricMuscle.usd_io import UsdIO, add_usd_arguments


def _create_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    add_usd_arguments(
        parser,
        usd_path="data/pendulum/pendulum.usda",
        output_path="output/example_couple.anim.usda",
    )
    parser.add_argument("--activation-hz", type=float, default=0.25, help="Activation signal frequency.")
    parser.add_argument("--max-motor", type=float, default=600.0, help="Max motor torque from activation.")
    return parser


class Example:
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.sim_frame = 0

        self.activation_hz = float(args.activation_hz)
        self.max_motor = float(args.max_motor)
        self.activation = 0.0

        self.output_path = str(getattr(args, "output_path", "output/example_couple.anim.usda"))
        self._use_layered_usd = bool(getattr(args, "use_layered_usd", True))
        self._copy_usd = bool(getattr(args, "copy_usd", True))

        builder = newton.ModelBuilder()
        builder.add_articulation(key="pendulum")
        hx, hy, hz = 1.0, 0.1, 0.1

        link_0 = builder.add_body()
        builder.add_shape_box(link_0, hx=hx, hy=hy, hz=hz)
        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -wp.pi * 0.5)

        j0 = builder.add_joint_revolute(
            parent=-1,
            child=link_0,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 3.0), q=rot),
            child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
        )

        builder.add_ground_plane()

        self.model = builder.finalize()
        self.solver = SolverMuscleBoneCoupled(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self._motor_buf = np.zeros(self.model.joint_dof_count, dtype=np.float32)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.collision_pipeline = None
        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)

        self._usd_io = UsdIO(source_usd_path=args.usd_path)
        self._init_layered_usd()

    def _init_layered_usd(self) -> None:
        if not self._use_layered_usd:
            return
        self._usd_io.start(self.output_path, copy_usd=self._copy_usd)
        self._usd_io.set_runtime("fps", self.fps)

    def _compute_activation(self) -> float:
        omega = 2.0 * np.pi * self.activation_hz
        self.activation = 0.5 * (1.0 + float(np.sin(omega * self.sim_time)))
        return self.activation

    def _write_layer_frame(self, frame: int) -> None:
        if not self._use_layered_usd:
            return
        angle_rad = float(self.state_0.joint_q.numpy()[0])
        angle_deg = float(np.degrees(angle_rad))
        muscle_pos = self.solver.muscle_solver.core.pos.numpy()
        centroid = np.mean(muscle_pos, axis=0)

        self._usd_io.set_runtime("joint_angle", angle_rad, frame=frame)
        self._usd_io.set_runtime("muscle_activation", float(self.activation), frame=frame)
        self._usd_io.set_runtime("muscle_centroid_y", float(centroid[1]), frame=frame)
        self._usd_io.set_custom("/pendulum/joint0", "xformOp:rotateY", angle_deg, frame=frame)

    def simulate(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            a = self._compute_activation()
            self.solver.muscle_solver.core.set_activation_from_tets(np.array([a], dtype=np.float32))

            motor = (2.0 * a - 1.0) * self.max_motor
            self._motor_buf[0] = motor
            self.control.joint_f = wp.array(self._motor_buf, dtype=wp.float32, device=self.model.device)

            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def gui(self, ui) -> None:
        ui.text("Muscle-bone coupling demo")
        changed, value = ui.slider_float("max_motor", self.max_motor, 0.0, 1500.0, "%.1f")
        if changed:
            self.max_motor = float(value)
        ui.text(f"activation = {self.activation:.3f}")
        ui.text(f"joint_angle = {float(self.state_0.joint_q.numpy()[0]):.3f} rad")

    def step(self) -> None:
        self.simulate()
        self.sim_time += self.frame_dt
        self.sim_frame += 1

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()
        self._write_layer_frame(self.sim_frame)

    def close(self) -> None:
        self._usd_io.close()


def main():
    parser = _create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    try:
        newton.examples.run(example)
    finally:
        example.close()


if __name__ == "__main__":
    main()
