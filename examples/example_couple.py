"""Couple MuscleSim (Taichi PBD) with Newton rigid-body skeleton.

Uses Taichi GGUI for visualization. Newton SolverMuJoCo (Y-up) drives
the radius bone under gravity; attach constraints pull the muscle to follow.
Both systems operate in the same Y-up coordinate space — no conversion needed.
"""

import numpy as np
import taichi as ti
import newton
import warp as wp

from VMuscle.muscle import MuscleSim, load_config
from VMuscle.usd_io import UsdIO
from VMuscle.solver_muscle_bone_coupled import SolverMuscleBoneCoupled

# Elbow pivot in Y-up space 
ELBOW_PIVOT_NP = np.array([0.328996, 1.16379, -0.0530352], dtype=np.float32)
ELBOW_PIVOT = wp.vec3(ELBOW_PIVOT_NP)
# Elbow flexion axis in Y-up space 
ELBOW_AXIS_NP = np.array([-0.788895, -0.45947, -0.408086], dtype=np.float32)
ELBOW_AXIS = wp.vec3(ELBOW_AXIS_NP)


def vis_joint_init():
    # Debug visualization: pivot point + axis line
    axis_len = 0.15
    axis_dir = ELBOW_AXIS_NP / np.linalg.norm(ELBOW_AXIS_NP)
    axis_end = ELBOW_PIVOT_NP + axis_dir * axis_len
    # Pivot sphere (single particle)
    debug_pivot = ti.Vector.field(3, dtype=ti.f32, shape=1)
    debug_pivot[0] = ELBOW_PIVOT_NP.tolist()
    # Axis line (2 vertices = 1 line segment)
    debug_axis_verts = ti.Vector.field(3, dtype=ti.f32, shape=2)
    debug_axis_verts[0] = ELBOW_PIVOT_NP.tolist()
    debug_axis_verts[1] = axis_end.tolist()
    return debug_pivot, debug_axis_verts


def main():
    wp.init()
    wp.set_device("cpu")
    
    # 1. MuscleSim (Y-up, Taichi GGUI)
    cfg = load_config("data/muscle/config/bicep.json")
    cfg.gui = True
    cfg.render_mode = "human"
    sim = MuscleSim(cfg)

    # 2. Read radius mesh from USD for Newton model (Y-up, no centering)
    usd = UsdIO(
        source_usd_path="data/muscle/model/bicep.usd",
        root_path="/character",
        y_up_to_z_up=False,
        center_model=False,
        up_axis=int(newton.Axis.Y),
    ).read()

    # 3. Minimal Newton model: radius body + elbow revolute joint
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.8)
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    radius_link = None
    for mesh in usd.meshes:
        if "radius" in mesh.mesh_path.lower():
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
                axis=ELBOW_AXIS,
                parent_xform=wp.transform(p=ELBOW_PIVOT),
                child_xform=wp.transform(p=ELBOW_PIVOT),
                limit_lower=-0.5,
                limit_upper=2.0,
                armature=0.1,
                friction=0.5,
                target_kd=5.0,
                target_ke=5.0
            )
            builder.add_articulation([joint], key="elbow")
            break

    model = builder.finalize()
    state_0 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    # 4. Coupled solver
    solver = SolverMuscleBoneCoupled(model, sim)
    if radius_link is not None and "L_radius" in sim.bone_muscle_ids:
        indices = sim.bone_muscle_ids["L_radius"]
        rest_verts = sim.bone_pos[indices].astype(np.float32)
        solver.configure_coupling(
            radius_link, rest_verts, indices,
            joint_index=joint,
            joint_pivot=ELBOW_PIVOT_NP,
            joint_axis=ELBOW_AXIS_NP,
        )

    # 5. Debug visualization: pivot point + axis line
    debug_pivot,debug_axis_verts=vis_joint_init()

    # 6. Main loop (Taichi GGUI)
    step_cnt = 0
    dt = 1.0 / 60.0
    fps = 0.0
    while sim.vis.window.running:
        sim.vis._render_control()
        if not cfg.pause:
            with wp.ScopedTimer("Couple", synchronize=True, print=False) as timer:
                solver.step(state_0, state_0, dt=dt)
            fps = 1000.0 / timer.elapsed if timer.elapsed > 0 else 0.0
            step_cnt += 1
        if cfg.gui:
            # Draw pivot (yellow sphere) and axis (green line) before render
            sim.vis.scene.particles(debug_pivot, radius=0.01, color=(1.0, 1.0, 0.0))
            sim.vis.scene.lines(debug_axis_verts, width=3, color=(0.0, 1.0, 0.0))
            sim.vis._render_frame(step_cnt)
            sim.vis.extra_text = f"Couple FPS: {fps:.1f}" if fps else ""


if __name__ == "__main__":
    main()
