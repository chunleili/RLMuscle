"""Real performance benchmark for coupled muscle-bone simulation.

Measures wall-clock time of the core solver.step() loop, excluding
initialization, model loading, and visualization.  Each configuration
is run multiple times and averaged to reduce noise.

IMPORTANT: Taichi can only be initialized once per process, so the scene
(MuscleSim + Newton model) is built once.  Different bone solvers are
benchmarked by swapping `coupled.bone_solver` on the same coupled solver.

Usage:
    uv run python tests/test_perf_couple.py
    uv run python tests/test_perf_couple.py --steps 300 --trials 5
    uv run python tests/test_perf_couple.py --solvers featherstone mujoco
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import warp as wp

# Suppress noisy loggers during benchmark
logging.getLogger("couple").setLevel(logging.WARNING)
logging.getLogger("taichi").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Scene setup (mirrors example_couple.py, built ONCE per process)
# ---------------------------------------------------------------------------

ELBOW_PIVOT = np.array([0.328996, 1.16379, -0.0530352], dtype=np.float32)
ELBOW_AXIS = np.array([-0.788895, -0.45947, -0.408086], dtype=np.float32)

# Solver factory — maps name to a constructor lambda
_SOLVER_MAP = {}
# Solvers that require CUDA device (mujoco_warp GPU backend)
_CUDA_SOLVERS = {"mujoco_cuda"}


def _init_solver_map():
    from newton.solvers import (
        SolverFeatherstone,
        SolverMuJoCo,
        SolverSemiImplicit,
        SolverVBD,
        SolverXPBD,
    )
    _SOLVER_MAP.update({
        "featherstone": lambda m: SolverFeatherstone(
            m, angular_damping=0.3, friction_smoothing=2.0, use_tile_gemm=False),
        "mujoco": lambda m: SolverMuJoCo(m, solver="cg"),
        "mujoco_cpu": lambda m: SolverMuJoCo(m, solver="cg", use_mujoco_cpu=True),
        # mujoco_warp GPU backend; integrator="euler" avoids tile_matmul LTO bug
        "mujoco_cuda": lambda m: SolverMuJoCo(m, solver="cg", integrator="euler"),
        "xpbd": lambda m: SolverXPBD(m, iterations=2),
        "vbd": lambda m: SolverVBD(m, iterations=10),
        "semi_implicit": lambda m: SolverSemiImplicit(m),
    })


def _build_scene_once():
    """Build MuscleSim + Newton model + coupled solver ONCE.

    Returns (coupled_solver, model, cfg, dt, radius_link, joint).
    The bone_solver inside coupled_solver is the default (Featherstone).
    Callers swap it before benchmarking each solver type.
    """
    import newton
    from newton.solvers import SolverMuJoCo

    from VMuscle.muscle import MuscleSim, load_config
    from VMuscle.solver_muscle_bone_coupled import SolverMuscleBoneCoupled
    from VMuscle.usd_io import UsdIO

    _init_solver_map()

    # Muscle sim (headless)
    cfg = load_config("data/muscle/config/bicep.json")
    cfg.gui = False
    cfg.render_mode = None
    sim = MuscleSim(cfg)

    # USD mesh
    usd = UsdIO(
        source_usd_path="data/muscle/model/bicep.usd",
        root_path="/character",
        y_up_to_z_up=False,
        center_model=False,
        up_axis=int(newton.Axis.Y),
    ).read()

    # Newton model
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

    # Coupled solver (default bone_solver = Featherstone)
    coupled = SolverMuscleBoneCoupled(
        model, sim, k_coupling=5000.0, max_torque=50.0,
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

    dt = 1.0 / 60.0
    return coupled, model, cfg, dt


def _reset_state(model):
    """Create a fresh Newton state (resets bone positions / velocities)."""
    import newton
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    return state


def _activation_at(step: int, n_steps: int) -> float:
    """Same activation ramp as example_couple._run_auto_test."""
    t = step / n_steps
    if t <= 0.2:
        return 0.0
    elif t <= 0.3:
        return 0.5
    elif t <= 0.5:
        return 1.0
    elif t <= 0.7:
        return 0.7
    elif t <= 0.8:
        return 0.3
    else:
        return 0.0


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def bench_solver(coupled, model, cfg, dt,
                 solver_name: str, n_steps: int, n_trials: int, warmup: int):
    """Benchmark a single bone solver. Returns dict with timing stats."""
    print(f"\n{'='*60}")
    print(f"  Solver: {solver_name}")
    print(f"  Steps: {n_steps}, Trials: {n_trials}, Warmup: {warmup}")
    print(f"{'='*60}")

    if solver_name not in _SOLVER_MAP:
        raise ValueError(f"Unknown solver: {solver_name}. "
                         f"Choose from {list(_SOLVER_MAP.keys())}")

    # Switch warp device for CUDA solvers
    need_cuda = solver_name in _CUDA_SOLVERS
    if need_cuda:
        if not wp.is_cuda_available():
            print("  SKIP: CUDA not available")
            return None
        wp.set_device("cuda:0")
        print("  Device: cuda:0")

    # Swap bone solver
    print(f"  Creating {solver_name} solver...", end="", flush=True)
    t0 = time.perf_counter()
    coupled.bone_solver = _SOLVER_MAP[solver_name](model)
    print(f" done ({time.perf_counter() - t0:.1f}s)")

    # Warmup — let JIT / kernel caches settle
    print(f"  Warmup ({warmup} steps)...", end="", flush=True)
    state = _reset_state(model)
    coupled._step_count = 0
    for step in range(1, warmup + 1):
        cfg.activation = _activation_at(step, warmup)
        coupled.step(state, state, dt=dt)
    print(" done")

    # Timed trials
    trial_times = []
    for trial in range(n_trials):
        state = _reset_state(model)
        coupled._step_count = 0

        t_start = time.perf_counter()
        for step in range(1, n_steps + 1):
            cfg.activation = _activation_at(step, n_steps)
            coupled.step(state, state, dt=dt)
        t_end = time.perf_counter()

        elapsed = t_end - t_start
        trial_times.append(elapsed)
        ms_per_step = elapsed / n_steps * 1000
        fps = n_steps / elapsed
        print(f"  Trial {trial+1}/{n_trials}: {elapsed:.3f}s "
              f"({ms_per_step:.2f} ms/step, {fps:.1f} fps)")

    avg_time = sum(trial_times) / len(trial_times)
    min_time = min(trial_times)
    max_time = max(trial_times)
    avg_ms = avg_time / n_steps * 1000
    avg_fps = n_steps / avg_time

    # Restore CPU device after CUDA solver
    if need_cuda:
        wp.set_device("cpu")

    return {
        "solver": solver_name,
        "n_steps": n_steps,
        "n_trials": n_trials,
        "trial_times": trial_times,
        "avg_s": avg_time,
        "min_s": min_time,
        "max_s": max_time,
        "avg_ms_per_step": avg_ms,
        "avg_fps": avg_fps,
    }


def print_summary(results: list[dict]):
    """Print a markdown table summarizing all results."""
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}\n")
    print("| solver | avg ms/step | avg fps | min_s | max_s | trials |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in results:
        print(f"| {r['solver']} | {r['avg_ms_per_step']:.2f} | "
              f"{r['avg_fps']:.1f} | {r['min_s']:.3f} | {r['max_s']:.3f} | "
              f"{r['n_trials']} |")


def main():
    parser = argparse.ArgumentParser(
        description="Real performance benchmark for coupled muscle-bone simulation")
    parser.add_argument("--solvers", nargs="+",
                        default=["featherstone", "mujoco_cpu", "mujoco_cuda", "xpbd", "semi_implicit"],
                        help="Bone solvers to benchmark (default: featherstone mujoco_cpu mujoco_cuda xpbd semi_implicit)")
    parser.add_argument("--steps", type=int, default=200,
                        help="Simulation steps per trial (default: 200)")
    parser.add_argument("--trials", type=int, default=3,
                        help="Number of trials per solver (default: 3)")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Warmup steps before timing (default: 20)")
    args = parser.parse_args()

    wp.init()
    wp.set_device("cpu")

    # Build scene ONCE (Taichi can only init once per process)
    print("Building scene (one-time init)...", flush=True)
    t0 = time.perf_counter()
    coupled, model, cfg, dt = _build_scene_once()
    print(f"Scene ready in {time.perf_counter() - t0:.1f}s\n")

    results = []
    for solver_name in args.solvers:
        try:
            r = bench_solver(coupled, model, cfg, dt,
                             solver_name, args.steps, args.trials, args.warmup)
            if r is not None:
                results.append(r)
        except Exception as e:
            print(f"  ERROR: {solver_name} failed: {e}")
            import traceback
            traceback.print_exc()

    if results:
        print_summary(results)


if __name__ == "__main__":
    main()
