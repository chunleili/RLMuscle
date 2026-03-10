"""Couple Warp MuscleSim with Newton rigid-body skeleton (no Taichi).

This is a non-destructive alternative to `example_couple.py`.

Usage:
    uv run python examples/example_couple2.py --auto
    uv run python examples/example_couple2.py --auto --usd
    uv run python examples/example_couple2.py --steps 300
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import warp as wp

import newton

from VMuscle.config import load_config
from VMuscle.muscle_warp import MuscleSim
from VMuscle.solver_muscle_bone_coupled_warp import SolverMuscleBoneCoupledWarp
from VMuscle.usd_io import UsdIO

# Elbow joint parameters (Y-up space)
ELBOW_PIVOT = np.array([0.328996, 1.16379, -0.0530352], dtype=np.float32)
ELBOW_AXIS = np.array([-0.788895, -0.45947, -0.408086], dtype=np.float32)

log = logging.getLogger("couple")


def setup_logging(to_file: bool = False):
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
    for handler in couple_log.handlers:
        handler.setFormatter(fmt)


def _extract_radius_mesh(sim: MuscleSim):
    """Extract radius-only triangle mesh from bone data if available."""
    bone_pos = np.asarray(sim.bone_pos, dtype=np.float32)
    if bone_pos.size == 0:
        raise ValueError("Bone geometry is empty in MuscleSim.")

    if not hasattr(sim, "bone_indices_np"):
        raise ValueError("Bone triangle indices are missing in MuscleSim.")

    faces = np.asarray(sim.bone_indices_np, dtype=np.int32).reshape(-1, 3)

    group_name = None
    selected = None
    for key, indices in getattr(sim, "bone_muscle_ids", {}).items():
        if "radius" in str(key).lower():
            group_name = str(key)
            selected = np.asarray(indices, dtype=np.int32)
            break

    if selected is None or selected.size == 0:
        group_name = "all_bones"
        selected = np.arange(bone_pos.shape[0], dtype=np.int32)

    selected_set = set(selected.tolist())
    mask = np.array([all(int(v) in selected_set for v in tri) for tri in faces], dtype=bool)
    part_faces = faces[mask]
    if part_faces.size == 0:
        # Fallback: use full bone mesh if per-part extraction produced no triangles.
        part_faces = faces
        selected = np.arange(bone_pos.shape[0], dtype=np.int32)

    used = np.unique(part_faces.reshape(-1))
    remap = np.full(bone_pos.shape[0], -1, dtype=np.int32)
    remap[used] = np.arange(used.shape[0], dtype=np.int32)

    local_vertices = bone_pos[used]
    local_faces = remap[part_faces]

    return group_name, selected, local_vertices, local_faces


def build_elbow_model(sim: MuscleSim):
    """Build a minimal Newton model: radius body + elbow revolute joint.

    Returns:
        model, state, radius_link, joint_index, selected_global_indices
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

    group_name, selected_indices, radius_vertices, radius_faces = _extract_radius_mesh(sim)

    radius_link = builder.add_link(xform=wp.transform())
    builder.add_shape_mesh(
        body=radius_link,
        xform=wp.transform(),
        mesh=newton.Mesh(
            vertices=radius_vertices,
            indices=radius_faces.reshape(-1),
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
        limit_lower=-3.0,
        limit_upper=3.0,
        armature=1.0,
        friction=0.9,
        target_ke=5.0,
        target_kd=5.0,
    )
    builder.add_articulation([joint], label="elbow")

    model = builder.finalize()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    log.info("Using bone group '%s' with %d vertices", group_name, len(selected_indices))

    return model, state, radius_link, joint, selected_indices


def _activation_schedule(step: int, total: int) -> float:
    t = step / max(total, 1)
    if t <= 0.2:
        return 0.0
    if t <= 0.3:
        return 0.5
    if t <= 0.5:
        return 1.0
    if t <= 0.7:
        return 0.7
    if t <= 0.8:
        return 0.3
    return 0.0


def _build_bone_prim_map(usd: UsdIO, sim: MuscleSim) -> dict[str, np.ndarray]:
    """Map each bone USD prim path to its vertex indices in sim.bone_pos_field.

    UsdIO bone meshes have prim paths like ``/character/bone/L_radius/L_radiusShape``.
    MuscleSim stores all bone vertices concatenated in ``bone_pos_field`` and groups
    them via ``bone_muscle_ids`` (e.g. ``{"L_radius": [3,4,5,...]}``).
    """
    mapping: dict[str, np.ndarray] = {}
    for bm in usd.bone_meshes:
        # Match prim path keyword against bone_muscle_ids group name
        for group_name, indices in sim.bone_muscle_ids.items():
            if group_name.lower() in bm.mesh_path.lower():
                mapping[bm.mesh_path] = np.asarray(indices, dtype=np.int32)
                break
    return mapping


def run_loop(solver, state, cfg, dt: float, n_steps: int, auto: bool,
             usd: UsdIO | None = None, bone_prim_map: dict | None = None):
    """Headless run loop with optional scheduled activation and USD export."""
    for step in range(1, n_steps + 1):
        if auto:
            cfg.activation = _activation_schedule(step, n_steps)

        solver.step(state, state, dt=dt)

        # Write USD frame
        if usd is not None:
            # Muscle deformed vertices
            usd.set_points(usd.muscle_mesh.mesh_path, solver.core.pos, frame=step)
            # Bone vertices per group
            bone_np = solver.core.bone_pos_field.numpy()
            for prim_path, indices in (bone_prim_map or {}).items():
                usd.set_points(prim_path, bone_np[indices], frame=step)
            # Runtime metadata
            usd.set_runtime("activation", float(cfg.activation), frame=step)
            joint_q = state.joint_q.numpy()
            usd.set_runtime("joint_angle", float(joint_q[0]) if len(joint_q) > 0 else 0.0, frame=step)

        if step % 25 == 0 or step == 1:
            body_q = state.body_q.numpy()[0]
            joint_q = state.joint_q.numpy()
            joint_angle = float(joint_q[0]) if len(joint_q) > 0 else 0.0
            tau = solver._muscle_torque
            log.info(
                "step=%4d act=%.2f pos=(%.4f, %.4f, %.4f) q=%.4f |tau|=%.4f",
                step,
                float(cfg.activation),
                float(body_q[0]),
                float(body_q[1]),
                float(body_q[2]),
                joint_angle,
                float(np.linalg.norm(tau)),
            )


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Warp-only muscle-bone coupling demo")
    parser.add_argument("--auto", action="store_true", help="Use built-in activation schedule")
    parser.add_argument("--steps", type=int, default=300, help="Number of simulation steps")
    parser.add_argument(
        "--config",
        type=str,
        default="data/muscle/config/bicep.json",
        help="Path to muscle config JSON",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Warp device, e.g. cpu or cuda:0")
    parser.add_argument("--usd-source", type=str, default="data/muscle/model/bicep.usd",
                        help="Source USD for layered export")
    return parser


def main():
    args = _create_parser().parse_args()

    setup_logging(to_file=True)

    wp.init()
    wp.set_device(args.device)

    cfg = load_config(args.config)
    cfg.gui = False
    cfg.render_mode = None

    # Override cfg to load from USD source so MuscleSim and USD export
    # read the same asset, preventing data inconsistency.
    usd_source = args.usd_source
    cfg.geo_path = usd_source
    cfg.bone_geo_path = usd_source
    cfg.muscle_prim_path = "/character/muscle/bicep"
    cfg.bone_prim_paths = {
        "L_scapula": "/character/bone/L_scapula/L_scapulaShape",
        "L_radius": "/character/bone/L_radius/L_radiusShape",
        "L_humerus": "/character/bone/L_humerus/L_humerusShape",
    }
    if cfg.constraints:
        for c in cfg.constraints:
            if "target_path" in c:
                c["target_path"] = usd_source

    sim = MuscleSim(cfg)

    dt = 1.0 / 60.0
    model, state, radius_link, joint, selected_indices = build_elbow_model(sim)

    solver = SolverMuscleBoneCoupledWarp(
        model,
        sim,
        k_coupling=5000.0,
        max_torque=50.0,
    )

    if radius_link is not None and selected_indices.size > 0:
        solver.configure_coupling(
            bone_body_id=radius_link,
            bone_rest_verts=sim.bone_pos[selected_indices].astype(np.float32),
            bone_vertex_indices=selected_indices,
            joint_index=joint,
            joint_pivot=ELBOW_PIVOT,
            joint_axis=ELBOW_AXIS,
        )

    log.info(
        "dt=%.6f muscle_substeps=%d bone_substeps=%d",
        dt,
        int(cfg.num_substeps),
        int(solver.bone_substeps),
    )

    # USD export
    usd = UsdIO(usd_source, y_up_to_z_up=False)
    usd.muscle_mesh = usd.find_mesh("muscle")
    usd.bone_meshes = usd.find_meshes("bone")
    usd.start("output/example_couple2.anim.usd", copy_usd=True)
    usd.set_runtime("fps", 60)
    bone_prim_map = _build_bone_prim_map(usd, sim)
    log.info("USD export: %s  bone_groups=%s",
             usd.output_path, list(bone_prim_map.keys()))

    try:
        run_loop(solver, state, cfg, dt=dt, n_steps=int(max(1, args.steps)),
                 auto=bool(args.auto), usd=usd, bone_prim_map=bone_prim_map)
    finally:
        usd.close()
        log.info("USD saved: %s", usd.output_path)


if __name__ == "__main__":
    main()
