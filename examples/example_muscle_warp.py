"""Warp-based muscle simulation with Taichi GGUI visualization.

Uses Warp MuscleSim for PBD computation and copies results to Taichi fields
for rendering via the existing Taichi Visualizer. This ensures visual
consistency with the Taichi version while validating Warp numerical results.

Usage (GUI):
    uv run python examples/example_muscle_warp.py

Usage (headless auto-test):
    uv run python examples/example_muscle_warp.py --auto
"""

import sys
import time

sys.path.insert(0, "src")

import numpy as np
import warp as wp

from VMuscle.muscle import MuscleSim as TaichiMuscleSim, SimConfig, load_config
from VMuscle.muscle_warp import MuscleSim as WarpMuscleSim, SimConfig as WarpSimConfig, fill_float_kernel


def main():
    wp.init()
    wp.set_device("cpu")

    auto_test = "--auto" in sys.argv

    # Load shared config (contains constraint definitions, mesh paths, etc.)
    cfg = load_config("data/muscle/config/bicep.json")
    if auto_test:
        cfg.gui = False
        cfg.render_mode = None
    else:
        cfg.gui = True
        cfg.render_mode = "human"

    # 1. Taichi MuscleSim: provides visualization (Visualizer + Taichi fields)
    taichi_sim = TaichiMuscleSim(cfg)

    # 2. Warp MuscleSim: does the actual computation (no GUI)
    warp_cfg = WarpSimConfig(
        geo_path=cfg.geo_path,
        bone_geo_path=cfg.bone_geo_path,
        dt=cfg.dt,
        nsteps=cfg.nsteps,
        num_substeps=cfg.num_substeps,
        gravity=cfg.gravity,
        density=cfg.density,
        veldamping=cfg.veldamping,
        activation=cfg.activation,
        constraints=cfg.constraints,
        gui=False,
        render_mode=None,
    )
    warp_sim = WarpMuscleSim(warp_cfg)

    # Verify mesh compatibility
    assert taichi_sim.n_verts == warp_sim.n_verts, (
        f"Vertex count mismatch: taichi={taichi_sim.n_verts}, warp={warp_sim.n_verts}"
    )

    def sync_warp_to_taichi():
        """Copy Warp positions to Taichi field for visualization."""
        pos_np = warp_sim.pos.numpy()
        taichi_sim.pos.from_numpy(pos_np)

    # Copy initial positions
    sync_warp_to_taichi()

    if auto_test:
        _run_auto_test(warp_sim, taichi_sim, cfg)
    else:
        _run_interactive(warp_sim, taichi_sim, cfg, sync_warp_to_taichi)


def _run_auto_test(warp_sim, taichi_sim, cfg):
    """Headless test: run nsteps, print diagnostics."""
    print(f"Running {cfg.nsteps} steps (headless)...")
    for step in range(1, cfg.nsteps + 1):
        # Set activation
        wp.launch(fill_float_kernel, dim=warp_sim.activation.shape[0],
                  inputs=[warp_sim.activation, cfg.activation])
        warp_sim.step()

        if step % 50 == 0 or step == 1:
            pos_np = warp_sim.pos.numpy()
            centroid = pos_np.mean(axis=0)
            vol_err = warp_sim.calc_vol_error()
            print(f"  step={step:4d}  act={cfg.activation:.2f}  "
                  f"centroid=({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})  "
                  f"vol_err={vol_err:.4f}")

    print("Auto test done.")


def _run_interactive(warp_sim, taichi_sim, cfg, sync_fn):
    """Interactive loop: Warp step -> copy to Taichi -> render."""
    step_cnt = 0

    while taichi_sim.vis.window.running:
        taichi_sim.vis._render_control()

        if not cfg.pause:
            # Set activation from GUI slider (cfg.activation is updated by Visualizer GUI)
            wp.launch(fill_float_kernel, dim=warp_sim.activation.shape[0],
                      inputs=[warp_sim.activation, cfg.activation])

            taichi_sim.step_start_time = time.perf_counter()
            warp_sim.step()
            taichi_sim.step_end_time = time.perf_counter()

            # Copy Warp results to Taichi fields for rendering
            sync_fn()
            step_cnt += 1
            taichi_sim.step_cnt = step_cnt

        if cfg.gui:
            taichi_sim.vis._render_frame(step_cnt, cfg.save_image)


if __name__ == "__main__":
    main()
