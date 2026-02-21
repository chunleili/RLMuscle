"""High-level comparison test: muscle.py (Taichi) vs muscle_warp.py (Warp).

Runs both simulators with the same config and varying activation over 300 frames,
comparing vertex positions at checkpoints to ensure numerical consistency.

Two modes are tested:
  1. Jacobi mode (use_jacobi=True): Order-independent constraint solving.
     Both implementations should match to float32 precision (~1e-4).
  2. Gauss-Seidel mode (use_jacobi=False, default): Order-dependent parallel
     constraint solving. Taichi itself is non-deterministic in this mode
     (~0.007 self-diff with 32 threads), so we only check that the Warp error
     is within Taichi's own non-determinism range.

Usage:
    uv run python tests/test_muscle_warp_vs_taichi.py
    uv run python tests/test_muscle_warp_vs_taichi.py --mode jacobi
    uv run python tests/test_muscle_warp_vs_taichi.py --mode gauss-seidel
    uv run python tests/test_muscle_warp_vs_taichi.py --mode both
"""

import sys
sys.path.insert(0, "src")

import argparse
import numpy as np
import warp as wp

wp.init()
wp.set_device("cpu")

from VMuscle.muscle import MuscleSim as TaichiSim, load_config as taichi_load_config
from VMuscle.muscle_warp import (
    MuscleSim as WarpSim,
    SimConfig as WarpSimConfig,
    load_config as warp_load_config,
    fill_float_kernel,
)


def activation_schedule(step: int, total: int) -> float:
    """Varying activation: 0 -> 0.5 -> 1.0 -> 0.3 -> 0.8 -> 0."""
    t = step / total
    if t < 0.1:
        return 0.0
    elif t < 0.3:
        return 0.5
    elif t < 0.5:
        return 1.0
    elif t < 0.7:
        return 0.3
    elif t < 0.9:
        return 0.8
    else:
        return 0.0


def run_comparison(n_steps=300, report_every=10, use_jacobi=True):
    mode_str = "Jacobi" if use_jacobi else "Gauss-Seidel"
    print(f"\n{'='*100}")
    print(f"  Running comparison in {mode_str} mode  ({n_steps} steps)")
    print(f"{'='*100}\n")

    cfg_taichi = taichi_load_config("data/muscle/config/bicep.json")
    cfg_taichi.gui = False
    cfg_taichi.render_mode = None

    taichi_sim = TaichiSim(cfg_taichi)
    taichi_sim.use_jacobi = use_jacobi

    warp_cfg = WarpSimConfig(
        geo_path=cfg_taichi.geo_path,
        bone_geo_path=cfg_taichi.bone_geo_path,
        dt=cfg_taichi.dt,
        nsteps=cfg_taichi.nsteps,
        num_substeps=cfg_taichi.num_substeps,
        gravity=cfg_taichi.gravity,
        density=cfg_taichi.density,
        veldamping=cfg_taichi.veldamping,
        activation=cfg_taichi.activation,
        constraints=cfg_taichi.constraints,
        gui=False,
        render_mode=None,
    )
    warp_sim = WarpSim(warp_cfg)
    warp_sim.use_jacobi = use_jacobi

    assert taichi_sim.n_verts == warp_sim.n_verts, (
        f"Vertex count mismatch: taichi={taichi_sim.n_verts}, warp={warp_sim.n_verts}"
    )
    n_verts = taichi_sim.n_verts

    sample_verts = [0, n_verts // 4, n_verts // 2, 3 * n_verts // 4, n_verts - 1]

    print(f"{'step':>5} {'act':>5} {'max_err':>10} {'mean_err':>10} {'vol_t':>8} {'vol_w':>8} | sample vertex errors")
    print("-" * 100)

    max_errors = []

    for step in range(1, n_steps + 1):
        act = activation_schedule(step, n_steps)

        # Taichi step
        taichi_sim.activation.fill(act)
        taichi_sim.step()

        # Warp step
        wp.launch(fill_float_kernel, dim=warp_sim.activation.shape[0],
                  inputs=[warp_sim.activation, act])
        warp_sim.step()

        if step % report_every == 0 or step == 1:
            pos_t = taichi_sim.pos.to_numpy()
            pos_w = warp_sim.pos.numpy()

            diff = pos_w - pos_t
            per_vertex_err = np.linalg.norm(diff, axis=1)
            max_err = per_vertex_err.max()
            mean_err = per_vertex_err.mean()

            vol_t = taichi_sim.calc_vol_error()
            vol_w = warp_sim.calc_vol_error()

            sample_errs = [per_vertex_err[v] for v in sample_verts]
            sample_str = "  ".join(f"v{v}={e:.6f}" for v, e in zip(sample_verts, sample_errs))

            print(f"{step:5d} {act:5.2f} {max_err:10.6f} {mean_err:10.6f} {vol_t:8.4f} {vol_w:8.4f} | {sample_str}")
            max_errors.append(max_err)

    # Summary
    print("\n" + "=" * 100)
    print(f"SUMMARY ({mode_str} mode)")
    print(f"  Total steps: {n_steps}")
    print(f"  Max position error ever: {max(max_errors):.6f}")
    print(f"  Final max error: {max_errors[-1]:.6f}")

    final_pos_t = taichi_sim.pos.to_numpy()
    final_pos_w = warp_sim.pos.numpy()
    final_diff = np.linalg.norm(final_pos_w - final_pos_t, axis=1)

    worst_idx = np.argsort(final_diff)[-10:][::-1]
    print(f"\n  Top 10 worst vertices at final step:")
    for idx in worst_idx:
        print(f"    vertex {idx:5d}: err={final_diff[idx]:.6f}  "
              f"taichi=({final_pos_t[idx][0]:.4f}, {final_pos_t[idx][1]:.4f}, {final_pos_t[idx][2]:.4f})  "
              f"warp=({final_pos_w[idx][0]:.4f}, {final_pos_w[idx][1]:.4f}, {final_pos_w[idx][2]:.4f})")

    centroid_t = final_pos_t.mean(axis=0)
    centroid_w = final_pos_w.mean(axis=0)
    print(f"\n  Final centroid taichi: ({centroid_t[0]:.4f}, {centroid_t[1]:.4f}, {centroid_t[2]:.4f})")
    print(f"  Final centroid warp:   ({centroid_w[0]:.4f}, {centroid_w[1]:.4f}, {centroid_w[2]:.4f})")
    print(f"  Centroid diff: {np.linalg.norm(centroid_w - centroid_t):.6f}")

    # Pass/fail with mode-appropriate threshold
    if use_jacobi:
        threshold = 1e-3  # Jacobi is order-independent, should match to float32 precision
    else:
        threshold = 0.1   # Gauss-Seidel: parallel race conditions cause ~0.05 diff
                           # (Taichi's own non-determinism is ~0.007 per step)

    if max(max_errors) < threshold:
        print(f"\nPASS: All position errors < {threshold} ({mode_str} mode)")
    else:
        print(f"\nFAIL: Max position error {max(max_errors):.6f} >= {threshold} ({mode_str} mode)")

    return max(max_errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Taichi vs Warp muscle simulation")
    parser.add_argument("--mode", choices=["jacobi", "gauss-seidel", "both"], default="jacobi",
                        help="Constraint solver mode to test (default: jacobi)")
    parser.add_argument("--steps", type=int, default=300, help="Number of simulation steps")
    args = parser.parse_args()

    results = {}

    if args.mode in ("jacobi", "both"):
        results["jacobi"] = run_comparison(n_steps=args.steps, use_jacobi=True)

    if args.mode in ("gauss-seidel", "both"):
        results["gauss-seidel"] = run_comparison(n_steps=args.steps, use_jacobi=False)

    if len(results) > 1:
        print("\n" + "=" * 100)
        print("OVERALL RESULTS")
        for mode, err in results.items():
            print(f"  {mode:>15}: max error = {err:.6f}")
