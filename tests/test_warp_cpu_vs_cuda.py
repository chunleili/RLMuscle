"""Warp CPU vs CUDA comparison: check if GPU race conditions cause divergence.

Runs the same Warp simulation on CPU and CUDA, compares vertex positions
at checkpoints. Detects NaN/Inf and reports per-step error.

Usage:
    uv run python tests/test_warp_cpu_vs_cuda.py
"""
import sys
sys.path.insert(0, "src")

import numpy as np
import warp as wp

wp.init()

from VMuscle.muscle_warp import MuscleSim, SimConfig, load_config, fill_float_kernel

ACTIVATION = 0.3
N_STEPS = 50

def run_sim(device_name):
    wp.set_device(device_name)
    cfg = load_config("data/muscle/config/bicep.json")
    cfg.gui = False
    cfg.render_mode = None
    sim = MuscleSim(cfg)
    print(f"\n[{device_name}] use_jacobi={sim.use_jacobi}, n_verts={sim.n_verts}")

    positions = []
    for step in range(1, N_STEPS + 1):
        wp.launch(fill_float_kernel, dim=sim.activation.shape[0],
                  inputs=[sim.activation, ACTIVATION])
        sim.step()
        if step in [1, 10, 25, 50]:
            pos = sim.pos.numpy().copy()
            n_nan = (~np.isfinite(pos).all(axis=1)).sum()
            centroid = pos[np.isfinite(pos).all(axis=1)].mean(axis=0) if np.isfinite(pos).any() else [0,0,0]
            print(f"  step {step:3d}: nan={n_nan}/{pos.shape[0]}  "
                  f"centroid=({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})")
            positions.append((step, pos))
    return positions


def run_test():
    """Run CPU vs CUDA comparison.

    Returns dict with keys: cpu_nans, cuda_nans, max_diff, passed.
    """
    print("=== Running Warp on CPU ===")
    cpu_pos = run_sim("cpu")

    print("\n=== Running Warp on CUDA ===")
    cuda_pos = run_sim("cuda:0")

    print("\n=== Comparison ===")
    cpu_nans = 0
    cuda_nans = 0
    max_diff = 0.0
    for (step_c, pc), (step_g, pg) in zip(cpu_pos, cuda_pos):
        nan_cpu = int((~np.isfinite(pc).all(axis=1)).sum())
        nan_cuda = int((~np.isfinite(pg).all(axis=1)).sum())
        cpu_nans += nan_cpu
        cuda_nans += nan_cuda
        finite = np.isfinite(pc).all(axis=1) & np.isfinite(pg).all(axis=1)
        if finite.any():
            diff = np.abs(pc[finite] - pg[finite])
            step_max = float(diff.max())
            max_diff = max(max_diff, step_max)
            print(f"step {step_c}: max_diff={step_max:.6f}  mean_diff={diff.mean():.6f}  "
                  f"nan_cpu={nan_cpu}  nan_cuda={nan_cuda}")
        else:
            print(f"step {step_c}: all NaN")

    passed = cuda_nans == 0
    return {"cpu_nans": cpu_nans, "cuda_nans": cuda_nans, "max_diff": max_diff, "passed": passed}


if __name__ == "__main__":
    result = run_test()
    if result["passed"]:
        print("\nPASS: No NaN on CUDA side.")
    else:
        print(f"\nFAIL: {result['cuda_nans']} NaN vertices on CUDA side.")
