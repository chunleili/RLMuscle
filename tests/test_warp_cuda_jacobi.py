"""Verify Warp on CUDA with Jacobi mode is stable.

Runs Warp simulation on CUDA with use_jacobi=True for 100 steps.
Reports NaN count, volume error, and centroid at checkpoints.

Usage:
    uv run python tests/test_warp_cuda_jacobi.py
"""
import sys
sys.path.insert(0, "src")

import numpy as np
import warp as wp

wp.init()
wp.set_device("cuda:0")

from VMuscle.muscle_warp import MuscleSim, SimConfig, load_config, fill_float_kernel

ACTIVATION = 0.3
N_STEPS = 100


def run_test(n_steps=N_STEPS, activation=ACTIVATION):
    """Run CUDA Jacobi stability test.

    Returns dict with keys: n_nan, vol_err, passed.
    """
    cfg = load_config("data/muscle/config/bicep.json")
    cfg.gui = False
    cfg.render_mode = None
    sim = MuscleSim(cfg)
    sim.use_jacobi = True
    print(f"use_jacobi={sim.use_jacobi}, device=cuda:0, n_verts={sim.n_verts}")

    n_nan = 0
    vol_err = 0.0
    for step in range(1, n_steps + 1):
        wp.launch(fill_float_kernel, dim=sim.activation.shape[0],
                  inputs=[sim.activation, activation])
        sim.step()
        if step in [1, 10, 25, 50, 100]:
            pos = sim.pos.numpy()
            n_nan = int((~np.isfinite(pos).all(axis=1)).sum())
            centroid = pos.mean(axis=0)
            vol_err = sim.calc_vol_error()
            print(f"  step {step:3d}: nan={n_nan}  vol_err={vol_err:.4f}  "
                  f"centroid=({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})")

    passed = n_nan == 0
    print("\nDone. Jacobi + CUDA test passed." if passed else "\nFAILED: NaN detected!")
    return {"n_nan": n_nan, "vol_err": vol_err, "passed": passed}


if __name__ == "__main__":
    run_test()
