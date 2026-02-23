"""Parameter study for coupled muscle-bone simulation stability & controllability.

Uses real physics engine (Newton + Taichi PBD) to test 5 bone solvers
with varying coupling parameters.  Measures stability (MAE, peak angle,
settling) and controllability (activation-angle/torque correlation,
monotonicity, overshoot, settling speed).

Scene is built ONCE (Taichi limitation), bone solver swapped per config.

Usage:
    uv run python tests/test_solver_parameters.py                    # full study
    uv run python tests/test_solver_parameters.py --quick            # quick sanity
    uv run python tests/test_solver_parameters.py --solvers xpbd     # single solver
    uv run python tests/test_solver_parameters.py --study stability  # stability only
    uv run python tests/test_solver_parameters.py --study control    # control only
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import warp as wp

logging.getLogger("couple").setLevel(logging.WARNING)
logging.getLogger("taichi").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ELBOW_PIVOT = np.array([0.328996, 1.16379, -0.0530352], dtype=np.float32)
ELBOW_AXIS = np.array([-0.788895, -0.45947, -0.408086], dtype=np.float32)

# Baseline coupling parameters
BASELINE = dict(
    k_coupling=5000.0,
    max_torque=50.0,
    dt=1.0 / 60.0,
)

# Parameter sweeps — one param changed at a time, rest = baseline
PARAM_SWEEPS = {
    "k_coupling":       [1000, 3000, 5000, 8000, 12000],
    "max_torque":       [10, 25, 50, 100, 200],
    "dt":               [1.0 / 30, 1.0 / 60, 1.0 / 120],
}

QUICK_PARAM_SWEEPS = {
    "k_coupling":       [1000, 5000, 12000],
    "max_torque":       [10, 50, 200],
    "dt":               [1.0 / 30, 1.0 / 60],
}

# Solver registry
_SOLVER_MAP = {}
_CUDA_SOLVERS = {"mujoco_cuda"}


def _init_solver_map():
    from newton.solvers import (
        SolverFeatherstone,
        SolverMuJoCo,
        SolverSemiImplicit,
        SolverXPBD,
    )
    _SOLVER_MAP.update({
        "featherstone": lambda m: SolverFeatherstone(
            m, angular_damping=0.3, friction_smoothing=2.0, use_tile_gemm=False),
        "mujoco_cpu": lambda m: SolverMuJoCo(m, solver="cg", use_mujoco_cpu=True),
        "mujoco_cuda": lambda m: SolverMuJoCo(m, solver="cg", integrator="euler"),
        "xpbd": lambda m: SolverXPBD(m, iterations=2),
        "semi_implicit": lambda m: SolverSemiImplicit(m),
    })


# ---------------------------------------------------------------------------
# Scene setup (mirrors test_perf_couple.py — built ONCE)
# ---------------------------------------------------------------------------

def _build_scene_once():
    import newton
    from newton.solvers import SolverMuJoCo

    from VMuscle.muscle import MuscleSim, load_config
    from VMuscle.solver_muscle_bone_coupled import SolverMuscleBoneCoupled
    from VMuscle.usd_io import UsdIO

    _init_solver_map()

    cfg = load_config("data/muscle/config/bicep.json")
    cfg.gui = False
    cfg.render_mode = None
    sim = MuscleSim(cfg)

    usd = UsdIO(
        source_usd_path="data/muscle/model/bicep.usd",
        root_path="/character",
        y_up_to_z_up=False,
        center_model=False,
        up_axis=int(newton.Axis.Y),
    ).read()

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    SolverMuJoCo.register_custom_attributes(builder)

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
            armature=1.0, friction=0.9,
            target_ke=5.0, target_kd=5.0,
        )
        builder.add_articulation([joint], key="elbow")
        break

    model = builder.finalize()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    coupled = SolverMuscleBoneCoupled(
        model, sim,
        k_coupling=BASELINE["k_coupling"],
        max_torque=BASELINE["max_torque"],
    )
    if radius_link is not None and "L_radius" in sim.bone_muscle_ids:
        indices = sim.bone_muscle_ids["L_radius"]
        coupled.configure_coupling(
            bone_body_id=radius_link,
            bone_rest_verts=sim.bone_pos[indices].astype(np.float32),
            bone_vertex_indices=indices,
            joint_index=joint,
            joint_pivot=ELBOW_PIVOT,
            joint_axis=ELBOW_AXIS,
        )

    return coupled, model, cfg


def _reset_state(model):
    import newton
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    return state


def _reset_muscle(coupled):
    """Reset muscle sim to initial state (positions, velocities, activation)."""
    coupled.core.reset()
    # Re-sync bone positions from initial bone mesh data
    if hasattr(coupled.core, 'bone_pos_field') and coupled.core.bone_pos is not None:
        coupled.core.bone_pos_field.from_numpy(coupled.core.bone_pos)


# ---------------------------------------------------------------------------
# Activation profiles
# ---------------------------------------------------------------------------

def constant_activation(level: float):
    """Return an activation function that always returns a constant level."""
    def _act(step: int, dt: float) -> float:
        return level
    return _act


def step_response_activation(step: int, dt: float) -> float:
    """Step response: rest -> activate -> deactivate.

    t=0.0-1.0s:  act=0.0   (rest baseline)
    t=1.0-3.0s:  act=0.8   (activate)
    t=3.0-6.0s:  act=0.0   (deactivate, measure settling)
    """
    t = step * dt
    if t < 1.0:
        return 0.0
    elif t < 3.0:
        return 0.8
    else:
        return 0.0


def stability_activation(step: int, dt: float) -> float:
    """Pulse activation for stability testing (~3s total)."""
    t = step * dt
    if t < 0.5:
        return 0.0
    elif t < 1.5:
        return 0.8
    else:
        return 0.0


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

def _apply_params(coupled, params: dict):
    """Apply coupling parameters to the coupled solver."""
    if "k_coupling" in params:
        coupled.k_coupling = params["k_coupling"]
    if "max_torque" in params:
        coupled.max_torque = params["max_torque"]


def _restore_baseline(coupled):
    """Restore baseline coupling parameters."""
    coupled.k_coupling = BASELINE["k_coupling"]
    coupled.max_torque = BASELINE["max_torque"]


def run_trial(coupled, model, cfg, dt: float, n_steps: int,
              act_fn) -> dict:
    """Run one trial, recording angle, torque, and activation at each step.

    Resets both Newton state and muscle sim before running.
    Uses eval_ik to extract joint angles from body poses (needed for
    body-space solvers like XPBD/SemiImplicit).
    Stops early and fills NaN on divergence.

    Returns dict with keys: angles, torques, activations, diverged_step.
    """
    import newton

    _reset_muscle(coupled)
    state = _reset_state(model)
    coupled._step_count = 0

    angles = np.full(n_steps, np.nan, dtype=np.float64)
    torques = np.full(n_steps, np.nan, dtype=np.float64)
    activations = np.zeros(n_steps, dtype=np.float64)
    diverged_step = -1

    for step in range(n_steps):
        act = act_fn(step, dt)
        cfg.activation = act
        activations[step] = act

        coupled.step(state, state, dt=dt)

        # Check for NaN in body state
        body_q = state.body_q.numpy()
        if np.isnan(body_q).any():
            diverged_step = step
            break

        # Use eval_ik to compute joint_q from body_q (handles all solver types)
        newton.eval_ik(model, state, state.joint_q, state.joint_qd)

        angle = float(state.joint_q.numpy()[coupled._joint_dof_index])
        tau_mag = float(np.linalg.norm(coupled._muscle_torque))

        angles[step] = angle
        torques[step] = tau_mag

    return {
        "angles": angles,
        "torques": torques,
        "activations": activations,
        "diverged_step": diverged_step,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_stability_metrics(data: dict) -> dict:
    """Compute stability metrics from trial data."""
    diverged = data["diverged_step"]
    if diverged >= 0:
        return {
            "peak_angle": float("inf"),
            "mae": float("inf"),
            "tail_mae": float("inf"),
            "energy_ratio": float("inf"),
            "diverged": True,
            "diverged_step": diverged,
        }

    angles = data["angles"]
    # Filter out NaN values
    valid = ~np.isnan(angles)
    if not valid.any():
        return {
            "peak_angle": float("inf"),
            "mae": float("inf"),
            "tail_mae": float("inf"),
            "energy_ratio": float("inf"),
            "diverged": True,
            "diverged_step": 0,
        }

    angles_valid = angles[valid]
    n = len(angles_valid)

    peak_angle = float(np.max(np.abs(angles_valid)))
    mae = float(np.mean(np.abs(angles_valid)))

    tail_start = int(n * 0.8)
    tail_mae = float(np.mean(np.abs(angles_valid[tail_start:])))

    # Approximate angular velocity from finite differences
    if n > 1:
        omega = np.diff(angles_valid)
        max_omega = float(np.max(np.abs(omega)))
        tail_omega = float(np.mean(np.abs(omega[-10:]))) if n > 10 else max_omega
        energy_ratio = tail_omega / max(max_omega, 1e-12)
    else:
        energy_ratio = 0.0

    return {
        "peak_angle": peak_angle,
        "mae": mae,
        "tail_mae": tail_mae,
        "energy_ratio": energy_ratio,
        "diverged": False,
        "diverged_step": -1,
    }


def compute_controllability_metrics(level_data: list[dict],
                                    step_data: dict,
                                    dt: float) -> dict:
    """Compute controllability metrics from separate-level runs + step response.

    Args:
        level_data: List of trial results at activation levels [0, 0.25, 0.5, 0.75, 1.0].
                    Each from a constant-activation run.
        step_data: Trial result from step_response_activation profile.
        dt: Time step.
    """
    # --- Steady-state angles/torques per activation level ---
    activation_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    steady_angles = []
    steady_torques = []
    valid_levels = []
    any_diverged = False

    for i, ld in enumerate(level_data):
        if ld["diverged_step"] >= 0:
            any_diverged = True
            steady_angles.append(float("nan"))
            steady_torques.append(float("nan"))
            continue
        angles = ld["angles"]
        torques_arr = ld["torques"]
        valid = ~np.isnan(angles)
        if not valid.any():
            steady_angles.append(float("nan"))
            steady_torques.append(float("nan"))
            continue
        # Use last 30% of valid data for steady state
        valid_angles = angles[valid]
        valid_torques = torques_arr[valid]
        ss_start = int(len(valid_angles) * 0.7)
        steady_angles.append(float(np.mean(np.abs(valid_angles[ss_start:]))))
        steady_torques.append(float(np.mean(valid_torques[ss_start:])))
        valid_levels.append(i)

    # If all diverged
    if len(valid_levels) < 2:
        return {
            "monotonicity": 0.0,
            "act_angle_corr": 0.0,
            "act_torque_corr": 0.0,
            "overshoot": float("inf"),
            "settling_steps": 999,
            "rest_stable": float("inf"),
            "reversible": False,
            "steady_angles": steady_angles,
            "steady_torques": steady_torques,
            "diverged": True,
            "diverged_step": 0,
        }

    # Filter to valid levels for correlation
    sa = np.array([steady_angles[i] for i in valid_levels])
    st = np.array([steady_torques[i] for i in valid_levels])
    al = np.array([activation_levels[i] for i in valid_levels])

    # Monotonicity: adjacent pairs where higher act -> higher |angle|
    correct = 0
    total = 0
    for i in range(len(valid_levels) - 1):
        idx_a, idx_b = valid_levels[i], valid_levels[i + 1]
        if not (np.isnan(steady_angles[idx_a]) or np.isnan(steady_angles[idx_b])):
            total += 1
            if steady_angles[idx_b] > steady_angles[idx_a]:
                correct += 1
    monotonicity = correct / total if total > 0 else 0.0

    # Correlation
    if len(sa) >= 3 and np.std(sa) > 1e-12 and np.std(al) > 1e-12:
        act_angle_corr = float(np.corrcoef(al, sa)[0, 1])
    else:
        act_angle_corr = 0.0

    if len(st) >= 3 and np.std(st) > 1e-12 and np.std(al) > 1e-12:
        act_torque_corr = float(np.corrcoef(al, st)[0, 1])
    else:
        act_torque_corr = 0.0

    # --- Step response metrics ---
    overshoot = 0.0
    settling_steps = 0
    rest_stable = 0.0
    reversible = False

    if step_data["diverged_step"] < 0:
        angles = step_data["angles"]
        valid = ~np.isnan(angles)
        if valid.any():
            valid_angles = angles[valid]
            n = len(valid_angles)

            # Steady state during activation (t=1.0-3.0s, use last 30%)
            act_start = int(1.0 / dt)
            act_end = int(3.0 / dt)
            act_end = min(act_end, n)
            if act_end > act_start:
                act_region = np.abs(valid_angles[act_start:act_end])
                steady_act = float(np.mean(act_region[int(len(act_region) * 0.7):]))
                peak_act = float(np.max(act_region))
                if steady_act > 1e-6:
                    overshoot = (peak_act - steady_act) / steady_act

            # Settling after deactivation (t=3.0s onward)
            settle_start = int(3.0 / dt)
            settle_start = min(settle_start, n - 1)
            if settle_start < n - 1:
                settle_region = valid_angles[settle_start:]
                if len(settle_region) > 5:
                    omega = np.abs(np.diff(settle_region))
                    threshold = 0.001
                    settling_steps = len(omega)
                    for i in range(len(omega) - 4):
                        if np.all(omega[i:i + 5] < threshold):
                            settling_steps = i
                            break

            # Rest stability: last 30 steps
            rest_region = valid_angles[-30:] if n >= 30 else valid_angles
            rest_stable = float(np.std(rest_region))

            # Reversible: angle at end should be smaller than during activation
            if act_end < n and steady_act > 1e-6:
                final_angle = float(np.mean(np.abs(valid_angles[-20:])))
                reversible = final_angle < steady_act * 0.8

    return {
        "monotonicity": monotonicity,
        "act_angle_corr": act_angle_corr,
        "act_torque_corr": act_torque_corr,
        "overshoot": overshoot,
        "settling_steps": settling_steps,
        "rest_stable": rest_stable,
        "reversible": reversible,
        "steady_angles": steady_angles,
        "steady_torques": steady_torques,
        "diverged": False,
        "diverged_step": -1,
    }


def grade_controllability(m: dict) -> str:
    """Assign A/B/C/D/F grade based on controllability metrics."""
    if m.get("diverged"):
        return "F"

    mono = m["monotonicity"]
    corr = m["act_angle_corr"]
    overshoot = m["overshoot"]
    settle = m["settling_steps"]
    rest = m["rest_stable"]
    rev = m["reversible"]

    if mono >= 1.0 and corr > 0.9 and overshoot < 0.2 and settle < 60 and rest < 0.01 and rev:
        return "A"
    if mono >= 0.75 and corr > 0.7 and overshoot < 0.5 and settle < 90:
        return "B"
    if mono >= 0.5 and corr > 0.5:
        return "C"
    if mono > 0 or corr > 0.3:
        return "D"
    return "F"


# ---------------------------------------------------------------------------
# Solver switching
# ---------------------------------------------------------------------------

def _switch_solver(coupled, model, solver_name: str):
    """Switch bone solver, handling CUDA device switching."""
    need_cuda = solver_name in _CUDA_SOLVERS
    if need_cuda:
        if not wp.is_cuda_available():
            print("  SKIP: CUDA not available")
            return False
        wp.set_device("cuda:0")

    coupled.bone_solver = _SOLVER_MAP[solver_name](model)

    if need_cuda:
        wp.set_device("cpu")

    return True


# ---------------------------------------------------------------------------
# Studies
# ---------------------------------------------------------------------------

def run_stability_study(coupled, model, cfg, solvers: list[str],
                        param_sweeps: dict, quick: bool):
    """Run parameter sweeps for stability, one param at a time per solver."""
    n_steps = 90 if quick else 180  # 1.5s or 3s @ 60Hz
    results = {}  # {solver: {param: [{value, metrics}, ...]}}

    for solver_name in solvers:
        print(f"\n{'='*60}")
        print(f"  Stability Study: {solver_name}")
        print(f"{'='*60}")

        if not _switch_solver(coupled, model, solver_name):
            continue

        results[solver_name] = {}

        for param_name, values in param_sweeps.items():
            print(f"\n  --- sweep: {param_name} ---")
            param_results = []

            for val in values:
                # Set param override
                params = dict(BASELINE)
                params[param_name] = val
                _apply_params(coupled, params)
                dt = params["dt"]
                actual_steps = n_steps if param_name != "dt" else int(3.0 / dt)

                data = run_trial(
                    coupled, model, cfg, dt, actual_steps,
                    stability_activation,
                )
                metrics = compute_stability_metrics(data)

                param_results.append({"value": val, **metrics})
                if metrics.get("diverged"):
                    print(f"    {param_name}={val:<8} DIVERGED at step {metrics['diverged_step']}")
                else:
                    print(f"    {param_name}={val:<8} peak={metrics['peak_angle']:.4f} "
                          f"mae={metrics['mae']:.4f} tail={metrics['tail_mae']:.4f} "
                          f"energy={metrics['energy_ratio']:.4f}")

            results[solver_name][param_name] = param_results
            _restore_baseline(coupled)

    return results


def run_controllability_study(coupled, model, cfg, solvers: list[str],
                              quick: bool):
    """Run controllability test for each solver at baseline params.

    1. Separate constant-activation runs at [0, 0.25, 0.5, 0.75, 1.0]
    2. Step-response test (rest -> activate -> deactivate)
    """
    dt = BASELINE["dt"]
    level_steps = 60 if quick else 120    # 1s or 2s per activation level
    step_steps = 180 if quick else 360    # 3s or 6s for step response
    activation_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = {}

    for solver_name in solvers:
        print(f"\n{'='*60}")
        print(f"  Controllability Study: {solver_name}")
        print(f"{'='*60}")

        if not _switch_solver(coupled, model, solver_name):
            continue

        _restore_baseline(coupled)

        # 1. Separate constant-activation runs
        level_data = []
        for act_level in activation_levels:
            data = run_trial(
                coupled, model, cfg, dt, level_steps,
                constant_activation(act_level),
            )
            level_data.append(data)
            if data["diverged_step"] >= 0:
                print(f"    act={act_level:.2f}: DIVERGED at step {data['diverged_step']}")
            else:
                valid = ~np.isnan(data["angles"])
                if valid.any():
                    ss = data["angles"][valid]
                    ss_mean = float(np.mean(np.abs(ss[int(len(ss) * 0.7):])))
                    tau_mean = float(np.mean(data["torques"][valid][int(len(ss) * 0.7):]))
                    print(f"    act={act_level:.2f}: steady_angle={ss_mean:.4f} "
                          f"steady_torque={tau_mean:.4f}")

        # 2. Step response test
        print("    Step response...", end="", flush=True)
        step_data = run_trial(
            coupled, model, cfg, dt, step_steps,
            step_response_activation,
        )
        if step_data["diverged_step"] >= 0:
            print(f" DIVERGED at step {step_data['diverged_step']}")
        else:
            print(" done")

        # Compute metrics
        metrics = compute_controllability_metrics(level_data, step_data, dt)
        grade = grade_controllability(metrics)
        metrics["grade"] = grade
        results[solver_name] = metrics

        if metrics.get("diverged"):
            print(f"  Grade: F (diverged)")
        else:
            print(f"  Grade: {grade}")
            print(f"  Monotonicity: {metrics['monotonicity']:.2f}")
            print(f"  Act-Angle Corr: {metrics['act_angle_corr']:.3f}")
            print(f"  Act-Torque Corr: {metrics['act_torque_corr']:.3f}")
            print(f"  Overshoot: {metrics['overshoot']:.3f}")
            print(f"  Settling Steps: {metrics['settling_steps']}")
            print(f"  Rest Stable (std): {metrics['rest_stable']:.6f}")
            print(f"  Reversible: {metrics['reversible']}")
            if metrics.get("steady_angles"):
                sa = metrics["steady_angles"]
                labels = ["act=0", "act=0.25", "act=0.5", "act=0.75", "act=1.0"]
                pairs = []
                for l, a in zip(labels, sa):
                    pairs.append(f"{l}:{'nan' if np.isnan(a) else f'{a:.4f}'}")
                print(f"  Steady Angles: {', '.join(pairs)}")

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_stability_summary(results: dict):
    """Print stability results as markdown tables."""
    print(f"\n{'='*70}")
    print("  STABILITY SUMMARY")
    print(f"{'='*70}")

    # Collect all param names
    all_params = set()
    for solver_results in results.values():
        all_params.update(solver_results.keys())

    for param_name in sorted(all_params):
        print(f"\n### Parameter: {param_name}\n")
        print("| solver | value | peak_angle | mae | tail_mae | energy_ratio |")
        print("|---|---:|---:|---:|---:|---:|")

        for solver_name, solver_results in results.items():
            if param_name not in solver_results:
                continue
            for r in solver_results[param_name]:
                val_str = f"{r['value']:.4f}" if isinstance(r['value'], float) else str(r['value'])
                if r.get("diverged"):
                    print(f"| {solver_name} | {val_str} | "
                          f"DIVERGED (step {r['diverged_step']}) | - | - | - |")
                else:
                    print(f"| {solver_name} | {val_str} | "
                          f"{r['peak_angle']:.4f} | {r['mae']:.4f} | "
                          f"{r['tail_mae']:.4f} | {r['energy_ratio']:.4f} |")


def print_controllability_summary(results: dict):
    """Print controllability results as markdown table."""
    print(f"\n{'='*70}")
    print("  CONTROLLABILITY SUMMARY")
    print(f"{'='*70}\n")
    print("| solver | grade | mono | angle_corr | torque_corr | overshoot | settle | rest_std | reversible |")
    print("|---|:---:|---:|---:|---:|---:|---:|---:|:---:|")

    for solver_name, m in results.items():
        if m.get("diverged"):
            print(f"| {solver_name} | **F** | "
                  f"DIVERGED (step {m['diverged_step']}) | - | - | - | - | - | - |")
        else:
            print(f"| {solver_name} | **{m['grade']}** | "
                  f"{m['monotonicity']:.2f} | {m['act_angle_corr']:.3f} | "
                  f"{m['act_torque_corr']:.3f} | {m['overshoot']:.3f} | "
                  f"{m['settling_steps']} | {m['rest_stable']:.6f} | "
                  f"{'Y' if m['reversible'] else 'N'} |")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parameter study: coupled muscle-bone stability & controllability")
    parser.add_argument("--solvers", nargs="+",
                        default=["featherstone", "mujoco_cpu", "mujoco_cuda",
                                 "xpbd", "semi_implicit"],
                        help="Bone solvers to test")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer steps and parameter values")
    parser.add_argument("--study", choices=["stability", "control", "both"],
                        default="both",
                        help="Which study to run (default: both)")
    args = parser.parse_args()

    wp.init()
    wp.set_device("cpu")

    print("Building scene (one-time init)...", flush=True)
    t0 = time.perf_counter()
    coupled, model, cfg = _build_scene_once()
    print(f"Scene ready in {time.perf_counter() - t0:.1f}s")

    sweeps = QUICK_PARAM_SWEEPS if args.quick else PARAM_SWEEPS

    stability_results = None
    control_results = None

    if args.study in ("stability", "both"):
        stability_results = run_stability_study(
            coupled, model, cfg, args.solvers, sweeps, args.quick)
        if stability_results:
            print_stability_summary(stability_results)

    if args.study in ("control", "both"):
        control_results = run_controllability_study(
            coupled, model, cfg, args.solvers, args.quick)
        if control_results:
            print_controllability_summary(control_results)


if __name__ == "__main__":
    main()
