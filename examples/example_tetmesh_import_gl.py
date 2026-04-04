"""Load TetMesh muscles from Human.usd and display in OpenGL viewer.

Usage:
  RUN=example_tetmesh_import_gl uv run main.py
  uv run python -m examples.example_tetmesh_import_gl
"""

from pathlib import Path

import warp as wp
import newton
import newton.usd

try:
    import newton_usd_schemas
except Exception:
    pass

from pxr import Usd

INPUT_USD = Path(__file__).resolve().parents[1] / "data" / "Human" / "Human.usd"

TEST_MUSCLES = [
    "/Human/muscle/arms/R_bicepsBrachialis/R_bicepsBrachialis_Tet",
    "/Human/muscle/arms/L_bicepsBrachialis/L_bicepsBrachialis_Tet",
    "/Human/muscle/legs/R_rectusFemoris/R_rectusFemoris_Tet",
    "/Human/muscle/legs/L_rectusFemoris/L_rectusFemoris_Tet",
]


def main():
    stage = Usd.Stage.Open(str(INPUT_USD))
    tet_prims = newton.usd.find_tetmesh_prims(stage)
    print(f"Found {len(tet_prims)} TetMesh prims (muscles)")

    builder = newton.ModelBuilder()

    for prim_path in TEST_MUSCLES:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"  SKIP: {prim_path} not found")
            continue

        tetmesh = newton.usd.get_tetmesh(prim)
        name = prim_path.split("/")[-1]
        print(f"  {name}: {len(tetmesh.vertices)} verts, "
              f"{len(tetmesh.tet_indices)//4} tets, "
              f"{len(tetmesh.surface_tri_indices)//3} surf tris")

        tetmesh.custom_attributes = None

        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            mesh=tetmesh,
            density=1000.0,
            k_mu=5000.0,
            k_lambda=5000.0,
            k_damp=10.0,
        )

    # Build model
    model = builder.finalize()
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    print(f"\nModel: {model.particle_count} particles, "
          f"{model.tet_count} tets, {model.tri_count} tris")

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    initial_state = model.state()
    initial_state.assign(state_0)

    # OpenGL viewer loop
    fps = 60
    frame_dt = 1.0 / fps
    substeps = 10
    dt = frame_dt / substeps

    solver = newton.solvers.SolverXPBD(model)

    viewer = newton.viewer.ViewerGL()
    viewer.set_model(model)
    viewer._paused = True

    t = 0.0
    for _ in range(10_000_000):
        for _ in range(substeps):
            state_0.clear_forces()
            contacts = model.collide(state_0)
            if not viewer.is_paused():
                solver.step(state_in=state_0, state_out=state_1,
                            control=control, contacts=contacts, dt=dt)
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


if __name__ == "__main__":
    main()
