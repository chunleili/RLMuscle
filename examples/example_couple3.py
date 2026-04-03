"""Couple XPBD-Millard MuscleSim with Newton rigid-body skeleton.

Uses explicit active fiber force (sigma0 > 0) with Millard 2012 curves
instead of TETFIBERNORM constraint-based contraction.

Usage:
    RUN=example_couple3 uv run main.py -- --auto
    RUN=example_couple3 uv run main.py -- --auto --render
    RUN=example_couple3 uv run main.py -- --steps 300
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("WARP_CACHE_PATH", str((PROJECT_ROOT / ".cache" / "warp").resolve()))

import warp as wp

import newton

from VMuscle.config import load_config
from VMuscle.controllability import (
    DEFAULT_SWEEP_LEVELS,
    build_coupling_config,
    config_to_dict,
    list_presets,
    run_activation_sweep,
    solver_sample,
    write_sweep_report,
)
from VMuscle.muscle_warp import MuscleSim
from VMuscle.solver_muscle_bone_coupled_warp import SolverMuscleBoneCoupled
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
        part_faces = faces
        selected = np.arange(bone_pos.shape[0], dtype=np.int32)

    used = np.unique(part_faces.reshape(-1))
    remap = np.full(bone_pos.shape[0], -1, dtype=np.int32)
    remap[used] = np.arange(used.shape[0], dtype=np.int32)

    local_vertices = bone_pos[used]
    local_faces = remap[part_faces]

    return group_name, selected, local_vertices, local_faces


def build_elbow_model(sim: MuscleSim):
    """Build a minimal Newton model: radius body + elbow revolute joint."""
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


def _create_muscle_surface_mesh(usd: UsdIO, sim: MuscleSim) -> str | None:
    """Create a renderable surface Mesh prim from the TetMesh boundary triangles."""
    from pxr import Sdf, UsdGeom, Vt

    tet_prim_path = usd.muscle_mesh.mesh_path
    stage = usd._stage
    tet_prim = stage.GetPrimAtPath(tet_prim_path)
    if not tet_prim.IsValid():
        log.warning("TetMesh prim not found: %s", tet_prim_path)
        return None

    # Read surface triangle indices from the TetMesh
    sfvi_attr = tet_prim.GetAttribute("surfaceFaceVertexIndices")
    sfvi = sfvi_attr.Get() if sfvi_attr else None
    if sfvi is None or len(sfvi) == 0:
        log.warning("No surfaceFaceVertexIndices on %s", tet_prim_path)
        return None

    tri_indices = np.array(sfvi, dtype=np.int32).reshape(-1)  # flatten Vec3i -> flat ints
    n_tris = len(sfvi)

    # Create Mesh prim as sibling of the TetMesh
    surf_path = tet_prim_path + "_surface"
    surf_prim = stage.DefinePrim(surf_path, "Mesh")
    mesh = UsdGeom.Mesh(surf_prim)
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * n_tris))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(tri_indices))

    # Set rest-pose points
    pts_np = np.ascontiguousarray(sim.pos.numpy(), dtype=np.float32)
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts_np))

    # Copy displayColor from TetMesh if available
    src_pv = UsdGeom.PrimvarsAPI(tet_prim).GetPrimvar("displayColor")
    if src_pv and src_pv.Get() is not None:
        dst_pv = UsdGeom.PrimvarsAPI(surf_prim).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
        dst_pv.Set(src_pv.Get())

    log.info("Created muscle surface mesh: %s (%d triangles)", surf_path, n_tris)
    return surf_path


def _build_bone_prim_map(usd: UsdIO, sim: MuscleSim) -> dict[str, np.ndarray]:
    """Map each bone USD prim path to its vertex indices in sim.bone_pos_field."""
    mapping: dict[str, np.ndarray] = {}
    for bm in usd.bone_meshes:
        for group_name, indices in sim.bone_muscle_ids.items():
            if group_name.lower() in bm.mesh_path.lower():
                mapping[bm.mesh_path] = np.asarray(indices, dtype=np.int32)
                break
    return mapping


def _parse_levels(text: str) -> tuple[float, ...]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    return tuple(float(np.clip(value, 0.0, 1.0)) for value in values) if values else DEFAULT_SWEEP_LEVELS


def _run_eval_sweep(solver, sim: MuscleSim, state, cfg, dt: float, args):
    def reset_state():
        sim.reset()
        solver.reset_bone(state)

    def step_once():
        solver.step(state, state, dt=dt)

    def set_excitation(value: float):
        cfg.activation = float(value)

    report = run_activation_sweep(
        label=f"example_couple3:{args.preset}",
        dt=dt,
        step_fn=step_once,
        reset_fn=reset_state,
        set_excitation_fn=set_excitation,
        sample_fn=lambda: solver_sample(solver, state),
        levels=_parse_levels(args.eval_levels),
        hold_steps=args.eval_hold_steps,
        release_steps=args.eval_release_steps,
        warmup_steps=args.eval_warmup_steps,
    )
    report["control_config"] = config_to_dict(solver.control_config)
    report_path = PROJECT_ROOT / "output" / f"example_couple3_eval_{args.preset}.json"
    write_sweep_report(report_path, report)
    log.info("Evaluation sweep saved: %s", report_path)
    for episode in report["episodes"]:
        log.info(
            "eval act=%.2f steady_tau=%.4f steady_q=%.4f overshoot_tau=%.4f settle=%s",
            episode["activation"],
            episode["steady_axis_torque"],
            episode["steady_joint_angle"],
            episode["overshoot_axis_torque"],
            episode["settle_steps_after_release"],
        )
    log.info(
        "monotonic torque=%s angle=%s",
        report["monotonic_steady_torque"],
        report["monotonic_steady_angle"],
    )


def run_loop(solver, state, cfg, dt: float, n_steps: int, auto: bool,
             usd: UsdIO | None = None, bone_prim_map: dict | None = None,
             sim: MuscleSim | None = None,
             muscle_surface_path: str | None = None):
    """Run loop with optional rendering, scheduled activation and USD export."""
    for step in range(1, n_steps + 1):
        if sim and sim.renderer and not sim.renderer.is_running():
            break
        if auto:
            cfg.activation = _activation_schedule(step, n_steps)

        solver.step(state, state, dt=dt)
        if sim:
            sim.render()

        # Write USD frame
        if usd is not None:
            usd.set_points(usd.muscle_mesh.mesh_path, solver.core.pos, frame=step)
            if muscle_surface_path:
                usd.set_points(muscle_surface_path, solver.core.pos, frame=step)
            bone_np = solver.core.bone_pos_field.numpy()
            for prim_path, indices in (bone_prim_map or {}).items():
                usd.set_points(prim_path, bone_np[indices], frame=step)
            usd.set_runtime("activation", float(cfg.activation), frame=step)
            joint_q = state.joint_q.numpy()
            usd.set_runtime("joint_angle", float(joint_q[0]) if len(joint_q) > 0 else 0.0, frame=step)

        if step % 25 == 0 or step == 1:
            body_q = state.body_q.numpy()[0]
            joint_q = state.joint_q.numpy()
            joint_angle = float(joint_q[0]) if len(joint_q) > 0 else 0.0
            tau = solver._muscle_torque
            log.info(
                "step=%4d exc=%.2f act=%.2f pos=(%.4f, %.4f, %.4f) q=%.4f axis_tau=%.4f |tau|=%.4f",
                step,
                float(cfg.activation),
                float(solver._effective_activation),
                float(body_q[0]),
                float(body_q[1]),
                float(body_q[2]),
                joint_angle,
                float(solver._axis_torque),
                float(np.linalg.norm(tau)),
            )


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="XPBD-Millard muscle-bone coupling (explicit active fiber force)")
    parser.add_argument("--auto", action="store_true", help="Use built-in activation schedule")
    parser.add_argument("--steps", type=int, default=300, help="Number of simulation steps")
    parser.add_argument(
        "--config",
        type=str,
        default="data/muscle/config/bicep_xpbd_millard.json",
        help="Path to muscle config JSON",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Warp device, e.g. cpu or cuda:0")
    parser.add_argument("--render", action="store_true", help="Enable OpenGL interactive rendering")
    parser.add_argument("--no-usd", action="store_true", help="Disable default USD export")
    parser.add_argument("--usd-source", type=str, default="data/muscle/model/bicep.usd",
                        help="Source USD for layered export")
    parser.add_argument("--preset", type=str, default="smooth_nonlinear", choices=list_presets(),
                        help="Controllability preset")
    parser.add_argument("--coupling-mode", type=str, choices=["linear", "smooth_nonlinear", "rate_limited"],
                        help="Override coupling mode")
    parser.add_argument("--k-coupling", type=float, default=None, help="Override coupling stiffness")
    parser.add_argument("--max-torque", type=float, default=None, help="Override torque clamp")
    parser.add_argument("--passive-scale", type=float, default=None, help="Override passive torque floor")
    parser.add_argument("--torque-ema", type=float, default=None, help="Override torque smoothing factor")
    parser.add_argument("--torque-slew-rate", type=float, default=None, help="Override torque slew rate")
    parser.add_argument("--nonlinear-gamma", type=float, default=None, help="Override nonlinear activation gamma")
    parser.add_argument("--disable-activation-dynamics", action="store_true",
                        help="Disable first-order activation dynamics")
    parser.add_argument("--eval", action="store_true", help="Run activation sweep evaluation instead of time schedule")
    parser.add_argument("--eval-levels", type=str, default="0,0.1,0.3,0.5,0.7,1.0",
                        help="Comma-separated activation levels for sweep")
    parser.add_argument("--eval-hold-steps", type=int, default=90, help="Steps to hold each activation level")
    parser.add_argument("--eval-release-steps", type=int, default=90,
                        help="Steps to run after returning activation to zero")
    parser.add_argument("--eval-warmup-steps", type=int, default=20,
                        help="Zero-activation warmup before each sweep episode")
    return parser


def main():
    args = _create_parser().parse_args()

    setup_logging(to_file=True)

    wp.init()
    wp.set_device(args.device)

    cfg = load_config(args.config)
    cfg.gui = bool(args.render)
    cfg.render_mode = "human" if args.render else None

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

    control_config = build_coupling_config(
        args.preset,
        mode=args.coupling_mode,
        k_coupling=args.k_coupling,
        max_torque=args.max_torque,
        passive_scale=args.passive_scale,
        torque_ema=args.torque_ema,
        torque_slew_rate=args.torque_slew_rate,
        nonlinear_gamma=args.nonlinear_gamma,
        use_activation_dynamics=False if args.disable_activation_dynamics else None,
    )

    solver = SolverMuscleBoneCoupled(
        model,
        sim,
        control_config=control_config,
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
        "dt=%.6f muscle_substeps=%d bone_substeps=%d control=%s",
        dt,
        int(cfg.num_substeps),
        int(solver.bone_substeps),
        config_to_dict(control_config),
    )

    if args.eval:
        _run_eval_sweep(solver, sim, state, cfg, dt, args)
        return

    usd = None
    bone_prim_map = None
    muscle_surface_path = None
    if not args.no_usd:
        usd = UsdIO(usd_source, y_up_to_z_up=False)
        usd.muscle_mesh = usd.find_mesh("muscle")
        usd.bone_meshes = usd.find_meshes("bone")
        usd.start("output/example_couple3.anim.usd", copy_usd=True)
        usd.set_runtime("fps", 60)
        bone_prim_map = _build_bone_prim_map(usd, sim)

        # Create a renderable surface Mesh from the TetMesh's boundary triangles.
        muscle_surface_path = _create_muscle_surface_mesh(usd, sim)

        log.info("USD export: %s  bone_groups=%s",
                 usd.output_path, list(bone_prim_map.keys()))

    try:
        run_loop(solver, state, cfg, dt=dt, n_steps=int(max(1, args.steps)),
                 auto=bool(args.auto), usd=usd, bone_prim_map=bone_prim_map,
                 sim=sim, muscle_surface_path=muscle_surface_path)
    finally:
        if usd is not None:
            usd.close()
            log.info("USD saved: %s", usd.output_path)


if __name__ == "__main__":
    main()
