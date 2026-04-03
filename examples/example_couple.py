"""Couple MuscleSim (Taichi PBD) with Newton rigid-body skeleton.

Uses Taichi GGUI for visualization. Newton SolverMuJoCo drives the radius
bone; bilateral attach constraints couple muscle forces back to the joint.
Both systems operate in Y-up coordinate space.
"""

# -- Fix LLVM CommandLine option conflict between Taichi and Warp on macOS --
# `import taichi` immediately loads Taichi's LLVM C-extension and registers
# LLVM cl::opt options. If warp is imported/initialized first, Warp's LLVM is
# already resident and Taichi reuses it without re-registering. Reversed order
# triggers "Assertion failed: Option already exists!" because Warp's LLVM init
# asserts an option doesn't already exist (registered by Taichi's earlier load).
# Fix: import and init warp BEFORE importing taichi.
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("WARP_CACHE_PATH", str((PROJECT_ROOT / ".cache" / "warp").resolve()))
os.environ.setdefault("TI_OFFLINE_CACHE_FILE_PATH", str((PROJECT_ROOT / ".cache" / "taichi").resolve()))

import warp as wp

# Initialize Warp's LLVM runtime BEFORE taichi is imported.
# On macOS this prevents the "Option already exists!" assertion.
wp.init()
wp.set_device("cpu")

import taichi as ti
import newton

from VMuscle.controllability import (
    DEFAULT_SWEEP_LEVELS,
    build_coupling_config,
    config_to_dict,
    list_presets,
    run_activation_sweep,
    solver_sample,
    write_sweep_report,
)
from VMuscle.muscle_taichi import MuscleSim, load_config
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
            limit_lower=-3.0, limit_upper=3.0,
            armature=1.0, friction=0.05,
            target_ke=5.0, target_kd=5.0,
        )
        # Alternative: ball joint (3 DOF)
        # joint = builder.add_joint_ball(
        #     parent=-1, child=radius_link,
        #     parent_xform=wp.transform(p=wp.vec3(ELBOW_PIVOT)),
        #     child_xform=wp.transform(p=wp.vec3(ELBOW_PIVOT)),
        #     armature=0.1, friction=0.9,
        # )
        builder.add_articulation([joint])
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


def _parse_levels(text: str) -> tuple[float, ...]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    return tuple(float(np.clip(value, 0.0, 1.0)) for value in values) if values else DEFAULT_SWEEP_LEVELS


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Taichi muscle-bone coupling demo")
    parser.add_argument("--auto", action="store_true", help="Use built-in activation schedule")
    parser.add_argument("--steps", type=int, default=300, help="Number of simulation steps")
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
    parser.add_argument("--eval", action="store_true", help="Run activation sweep evaluation")
    parser.add_argument("--eval-levels", type=str, default="0,0.1,0.3,0.5,0.7,1.0",
                        help="Comma-separated activation levels for sweep")
    parser.add_argument("--eval-hold-steps", type=int, default=90, help="Steps to hold each activation level")
    parser.add_argument("--eval-release-steps", type=int, default=90,
                        help="Steps to run after returning activation to zero")
    parser.add_argument("--eval-warmup-steps", type=int, default=20,
                        help="Zero-activation warmup before each sweep episode")
    return parser


def _run_auto_test(solver, state, cfg, dt, n_steps=300):
    """Headless test: ramp activation 0 -> 0.5 -> 1.0 over n_steps."""
    for step in range(1, n_steps + 1):
        t = step / n_steps
        if t <= 0.2:
            cfg.activation = 0.0
        elif t <= 0.3:
            cfg.activation = 0.5
        elif t<=0.5:
            cfg.activation = 1.0
        elif t<=0.7:
            cfg.activation = 0.7
        elif t<=0.8:
            cfg.activation = 0.3
        else:
            cfg.activation = 0.0
        solver.step(state, state, dt=dt)
        if step % 50 == 1:
            body_q = state.body_q.numpy()[0]
            log.info(f"step={step:3d} exc={cfg.activation:.2f} act={solver._effective_activation:.2f} "
                     f"pos={body_q[:3].round(4)} "
                     f"axis_tau={solver._axis_torque:.4f} "
                     f"|tau|={np.linalg.norm(solver._muscle_torque):.4f}")
    log.info("Auto test done.")


def _run_eval_sweep(solver, sim, state, cfg, dt: float, args):
    def reset_state():
        sim.reset()
        solver.reset_bone(state)

    def step_once():
        solver.step(state, state, dt=dt)

    def set_excitation(value: float):
        cfg.activation = float(value)

    report = run_activation_sweep(
        label=f"example_couple:{args.preset}",
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
    report_path = PROJECT_ROOT / "output" / f"example_couple_eval_{args.preset}.json"
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


def _run_interactive(solver, sim, state, cfg, dt):
    """Interactive loop with Taichi GGUI."""
    pivot_field, axis_field = create_joint_debug_visuals()
    step_cnt = 0
    fps = 0.0
    cfg.pause = False

    while sim.vis.window.running:
        sim.vis._render_control()
        if cfg.reset:
            sim.reset()
            solver.reset_bone(state)
            step_cnt = 0
            cfg.reset = False
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
    args = _create_parser().parse_args()
    auto_test = bool(args.auto)
    setup_logging(to_file=True)

    # 1. Muscle simulation
    cfg = load_config("data/muscle/config/bicep.json")
    if auto_test or args.eval:
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
    ).read()

    # 3. Build Newton skeleton
    dt = 1.0 / 60.0
    model, state, radius_link, joint = build_elbow_model(usd)

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

    # 4. Coupled solver
    solver = SolverMuscleBoneCoupled(
        model, sim, control_config=control_config,
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

    log.info(
        "dt=%s muscle_substeps=%s bone_substeps=%s control=%s",
        dt,
        cfg.num_substeps,
        solver.bone_substeps,
        config_to_dict(control_config),
    )

    # 5. Run
    if args.eval:
        _run_eval_sweep(solver, sim, state, cfg, dt, args)
    elif auto_test:
        _run_auto_test(solver, state, cfg, dt, n_steps=int(max(1, args.steps)))
    else:
        _run_interactive(solver, sim, state, cfg, dt)


if __name__ == "__main__":
    main()
