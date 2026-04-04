"""Load Human.usd skeleton and display in OpenGL viewer.

Demonstrates loading a full USD scene into Newton ModelBuilder
with MuJoCo solver and interactive viewer (pause/play/reset).

Usage:
  RUN=example_human_import uv run main.py
  uv run python -m examples.example_human_import
"""

from pathlib import Path
import warp as wp
import newton

try:
    import newton_usd_schemas
except Exception:
    pass

import newton.usd

INPUT_USD = (Path(__file__).resolve().parents[1] / "data/Human/Human.usd")
OUTPUT_USD = "newton_sim_out.usd"


def main():
    # 1) Import USD -> Newton Model
    builder = newton.ModelBuilder()

    builder.add_usd(
        source=INPUT_USD.as_posix(),
        root_path="/Human",
        skip_mesh_approximation=True,
    )
    builder.add_ground_plane()
    model = builder.finalize()

    # 2) States / contacts / control
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.collide(state_0)

    # 3) Solver
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True)
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    initial_state = model.state()
    initial_state.assign(state_0)

    # 4) Simulation parameters
    fps = 60
    frame_dt = 1.0 / fps
    substeps = 10
    dt = frame_dt / substeps

    # 5) Viewer
    viewer = newton.viewer.ViewerGL()
    viewer.set_model(model)
    viewer._paused = True

    t = 0.0
    num_frames = 10_000_000
    for _ in range(num_frames):
        for _ in range(substeps):
            state_0.clear_forces()
            contacts = model.collide(state_0)
            if not viewer.is_paused():
                solver.step(state_in=state_0, state_out=state_1, control=control, contacts=contacts, dt=dt)
            if viewer.is_key_down("r"):
                state_0.assign(initial_state)
                state_1.assign(initial_state)
                viewer._paused = True
            state_0, state_1 = state_1, state_0

        viewer.begin_frame(t)
        viewer.log_state(state_0)
        viewer.end_frame()
        t += frame_dt

    viewer.close()
    print(f"Saved: {Path(OUTPUT_USD).resolve()}")


if __name__ == "__main__":
    main()
