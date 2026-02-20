"""Couple MuscleSim (Taichi PBD) with Newton rigid-body skeleton.

Uses Taichi GGUI for visualization. Newton SolverMuJoCo drives the radius
bone; bilateral attach constraints couple muscle forces back to the joint.
Both systems operate in Y-up coordinate space.
"""

import logging
import sys

import numpy as np
import taichi as ti
import newton
import warp as wp

from VMuscle.muscle import MuscleSim, load_config
from VMuscle.usd_io import UsdIO
from VMuscle.solver_muscle_bone_coupled import SolverMuscleBoneCoupled

# Elbow joint parameters (Y-up space)
ELBOW_PIVOT = np.array([0.328996, 1.16379, -0.0530352], dtype=np.float32)
ELBOW_AXIS = np.array([-0.788895, -0.45947, -0.408086], dtype=np.float32)

log = logging.getLogger("couple")


def setup_logging(to_file=False):
    """Configure 'couple' logger. Optionally write to log.md for debugging."""
    root = logging.getLogger()
    root.setLevel(logging.WARNING)

    couple_log = logging.getLogger("couple")
    couple_log.setLevel(logging.DEBUG)
    couple_log.propagate = False

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if to_file:
        couple_log.addHandler(logging.FileHandler("log.md", mode="w", encoding="utf-8"))
    couple_log.addHandler(logging.StreamHandler())
    for h in couple_log.handlers:
        h.setFormatter(fmt)


def build_elbow_model(usd):
    """Build a minimal Newton model: radius body + elbow revolute joint.

    Returns (model, state, radius_link, joint_index).
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

    radius_link = None
    joint = None
    for mesh in usd.meshes:
        if "radius" not in mesh.mesh_path.lower():
            continue
        radius_link = builder.add_link(xform=wp.transform())
        builder.add_shape_mesh(
            body=radius_link,
            xform=wp.transform(),
            mesh=newton.Mesh(
                vertices=mesh.vertices,
                indices=mesh.faces.reshape(-1),
                compute_inertia=True,
                is_solid=True,
            ),
        )
        joint = builder.add_joint_revolute(
            parent=-1,
            child=radius_link,
            axis=wp.vec3(ELBOW_AXIS),
            parent_xform=wp.transform(p=wp.vec3(ELBOW_PIVOT)),
            child_xform=wp.transform(p=wp.vec3(ELBOW_PIVOT)),
            limit_lower=-1.0, limit_upper=3.0,
            armature=1.0, friction=0.9,
            target_ke=5.0, target_kd=5.0,
        )
        # Alternative: ball joint (3 DOF)
        # joint = builder.add_joint_ball(
        #     parent=-1, child=radius_link,
        #     parent_xform=wp.transform(p=wp.vec3(ELBOW_PIVOT)),
        #     child_xform=wp.transform(p=wp.vec3(ELBOW_PIVOT)),
        #     armature=0.1, friction=0.9,
        # )
        builder.add_articulation([joint], key="elbow")
        break

    model = builder.finalize()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    return model, state, radius_link, joint


def create_joint_debug_visuals():
    """Create Taichi fields for visualizing joint pivot and axis."""
    axis_dir = ELBOW_AXIS / np.linalg.norm(ELBOW_AXIS)
    pivot_field = ti.Vector.field(3, dtype=ti.f32, shape=1)
    pivot_field[0] = ELBOW_PIVOT.tolist()
    axis_field = ti.Vector.field(3, dtype=ti.f32, shape=2)
    axis_field[0] = ELBOW_PIVOT.tolist()
    axis_field[1] = (ELBOW_PIVOT + axis_dir * 0.15).tolist()
    return pivot_field, axis_field


def _run_auto_test(solver, state, cfg, dt, n_steps=200):
    """Headless test: ramp activation 0 -> 0.5 -> 1.0 over n_steps."""
    for step in range(1, n_steps + 1):
        t = step / n_steps
        if t <= 0.33:
            cfg.activation = 0.0
        elif t <= 0.66:
            cfg.activation = 0.5
        else:
            cfg.activation = 1.0
        solver.step(state, state, dt=dt)
        if step % 50 == 1:
            body_q = state.body_q.numpy()[0]
            log.info(f"step={step:3d} act={cfg.activation:.1f} "
                     f"pos={body_q[:3].round(4)} "
                     f"|tau|={np.linalg.norm(solver._muscle_torque):.4f}")
    log.info("Auto test done.")


def _run_interactive(solver, sim, state, cfg, dt):
    """Interactive loop with Taichi GGUI."""
    pivot_field, axis_field = create_joint_debug_visuals()
    step_cnt = 0
    fps = 0.0
    cfg.pause = False

    while sim.vis.window.running:
        sim.vis._render_control()
        if not cfg.pause:
            with wp.ScopedTimer("Couple", synchronize=True, print=False) as timer:
                solver.step(state, state, dt=dt)
            fps = 1000.0 / timer.elapsed if timer.elapsed > 0 else 0.0
            step_cnt += 1
        if cfg.gui:
            sim.vis.scene.particles(pivot_field, radius=0.01, color=(1.0, 1.0, 0.0))
            sim.vis.scene.lines(axis_field, width=3, color=(0.0, 1.0, 0.0))
            sim.vis._render_frame(step_cnt)
            sim.vis.extra_text = f"Couple FPS: {fps:.1f}" if fps else ""


def main():
    wp.init()
    wp.set_device("cpu")

    auto_test = "--auto" in sys.argv
    setup_logging(to_file=auto_test)

    # 1. Muscle simulation
    cfg = load_config("data/muscle/config/bicep.json")
    if auto_test:
        cfg.gui = False
        cfg.render_mode = None
    else:
        cfg.gui = True
        cfg.render_mode = "human"
    sim = MuscleSim(cfg)

    # 2. Load radius mesh from USD
    usd = UsdIO(
        source_usd_path="data/muscle/model/bicep.usd",
        root_path="/character",
        y_up_to_z_up=False,
        center_model=False,
        up_axis=int(newton.Axis.Y),
    ).read()

    # 3. Build Newton skeleton
    dt = 1.0 / 60.0
    model, state, radius_link, joint = build_elbow_model(usd)

    # 4. Coupled solver
    solver = SolverMuscleBoneCoupled(
        model, sim, k_coupling=5000.0, max_torque=50.0, torque_smoothing=0.3,
    )
    if radius_link is not None and "L_radius" in sim.bone_muscle_ids:
        indices = sim.bone_muscle_ids["L_radius"]
        solver.configure_coupling(
            bone_body_id=radius_link,
            bone_rest_verts=sim.bone_pos[indices].astype(np.float32),
            bone_vertex_indices=indices,
            joint_index=joint,
            joint_pivot=ELBOW_PIVOT,
            joint_axis=ELBOW_AXIS,
        )

    log.info(f"dt={dt} muscle_substeps={cfg.num_substeps} "
             f"bone_substeps={solver.bone_substeps}")

    # 5. Run
    if auto_test:
        _run_auto_test(solver, state, cfg, dt)
    else:
        _run_interactive(solver, sim, state, cfg, dt)


if __name__ == "__main__":
    main()
