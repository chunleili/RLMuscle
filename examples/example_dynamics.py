"""Teaching example: pendulum dynamics with motor torque control and layered USD output.

This example demonstrates:
1. A single-link pendulum simulated with Newton rigid-body dynamics.
2. A GUI slider named 'motor' that applies a counter-clockwise torque to joint 0.
3. Per-frame USD output (joint angle + motor value) via `UsdIO` layered editing.

Usage (GUI):
    .venv/Scripts/python.exe examples/example_dynamics.py --viewer gl

Usage (headless layered export):
    .venv/Scripts/python.exe examples/example_dynamics.py --viewer null --headless --num-frames 120 --use-layered-usd
"""

import argparse

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverFeatherstone
from RLVometricMuscle.usd_io import UsdIO, add_usd_arguments


def _create_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    add_usd_arguments(
        parser,
        usd_path="data/pendulum/pendulum.usda",
        output_path="output/example_dynamics.anim.usda",
    )
    return parser


class Example:
    """Single-link pendulum with motor torque, writing state to layered USD."""

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.sim_frame = 0

        self.output_path = str(getattr(args, "output_path", "output/example_dynamics.anim.usda"))
        self._use_layered_usd = bool(getattr(args, "use_layered_usd", True))
        self._copy_usd = bool(getattr(args, "copy_usd", True))

        # Motor torque applied to joint 0 (positive = counter-clockwise).
        # User controls this via the GUI slider.
        self.motor = 0.0

        # --- Build pendulum model ---
        builder = newton.ModelBuilder()

        hx = 1.0  # half-length of the pendulum rod
        hy = 0.1
        hz = 0.1

        link_0 = builder.add_link()
        builder.add_shape_box(link_0, hx=hx, hy=hy, hz=hz)

        # Revolute joint: world → link_0, rotates around Y axis.
        # Parent anchor is at (0, 0, 3); child attaches at the rod's left end.
        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -wp.pi * 0.5)
        j0 = builder.add_joint_revolute(
            parent=-1,
            child=link_0,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 3.0), q=rot),
            child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
        )
        builder.add_articulation([j0], key="pendulum")
        builder.add_ground_plane()

        self.model = builder.finalize()
        self.solver = SolverFeatherstone(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Pre-allocate motor buffer (size = joint_dof_count) to avoid per-substep allocation.
        self._motor_buf = np.zeros(self.model.joint_dof_count, dtype=np.float32)

        # Evaluate forward kinematics to initialise body transforms.
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, args)
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        self.viewer.set_model(self.model)

        # --- Initialise layered USD output ---
        # We skip read() — geometry is defined in the source USD, not loaded as Newton meshes.
        self._usd_io = UsdIO(source_usd_path=args.usd_path)
        self._init_layered_usd()

    # ------------------------------------------------------------------
    # Layered USD helpers
    # ------------------------------------------------------------------

    def _init_layered_usd(self) -> None:
        if not self._use_layered_usd:
            return
        self._usd_io.start(self.output_path, copy_usd=self._copy_usd)
        self._usd_io.set_runtime("fps", self.fps)

    def _write_layer_frame(self, frame: int) -> None:
        if not self._use_layered_usd:
            return
        angle_rad = float(self.state_0.joint_q.numpy()[0])
        angle_deg = float(np.degrees(angle_rad))

        # Runtime scalars for data analysis.
        self._usd_io.set_runtime("joint_angle", angle_rad, frame=frame)
        self._usd_io.set_runtime("motor", float(self.motor), frame=frame)

        # Animate the joint Xform so DCC tools can replay the motion.
        self._usd_io.set_custom(
            "/pendulum/joint0", "xformOp:rotateY", angle_deg, frame=frame,
        )

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            # Apply motor torque to joint 0 (the single revolute DOF).
            # Zero first to avoid stale values, then set motor.
            self._motor_buf[0] = self.motor
            self.control.joint_f = wp.array(self._motor_buf, dtype=wp.float32, device=self.model.device)

            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    # ------------------------------------------------------------------
    # Newton example interface
    # ------------------------------------------------------------------

    def gui(self, ui) -> None:
        ui.text("Pendulum dynamics example")
        ui.text("Move the slider to apply torque to joint 0")
        ui.separator()

        changed, value = ui.slider_float("motor", float(self.motor), -1000.0, 1000.0, "%.1f")
        if changed:
            self.motor = float(value)

        angle = float(self.state_0.joint_q.numpy()[0])
        ui.text(f"joint_angle = {angle:.3f} rad")

        if self._use_layered_usd:
            ui.separator()
            ui.text(f"layered_usd: {self.output_path}")
            if ui.button("Save USD Layer"):
                self._usd_io.save()
                print(f"USD layer saved to {self._usd_io.output_path} at frame {self.sim_frame}")

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
        newton.examples.run(example, args)
    finally:
        example.close()


if __name__ == "__main__":
    main()
