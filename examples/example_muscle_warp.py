"""Warp-based muscle simulation with OpenGL or USD rendering.

Uses Warp MuscleSim for PBD computation and WarpRenderer for visualization.

Usage (OpenGL interactive):
    RUN=example_muscle_warp uv run main.py

Usage (headless auto-test, no rendering):
    uv run python examples/example_muscle_warp.py --auto --render none

Usage (auto-test with USD output):
    uv run python examples/example_muscle_warp.py --auto --render usd

Usage (USD output):
    uv run python examples/example_muscle_warp.py --render usd
"""

import argparse
import logging
import sys
import time

sys.path.insert(0, "src")

import numpy as np
import warp as wp

from VMuscle.log import setup_logging
from VMuscle.muscle_warp import MuscleSim, SimConfig, load_config, fill_float_kernel, WarpRenderer

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Warp muscle simulation example.")
    parser.add_argument("--auto", action="store_true", help="Headless auto-test mode.")
    parser.add_argument("--render", type=str, default=None, choices=["human", "usd", "none"],
                        help="Render mode: 'human' (OpenGL), 'usd' (USD file), 'none'. "
                             "Default: 'human' for interactive, 'usd' for --auto.")
    parser.add_argument("--device", type=str, default="cpu", help="Warp device (cpu or cuda:0).")
    args = parser.parse_args()

    setup_logging()

    wp.init()
    wp.set_device(args.device)

    # Resolve render mode: default is 'usd' for --auto, 'human' for interactive
    if args.render is None:
        render_mode = "usd" if args.auto else "human"
    elif args.render == "none":
        render_mode = None
    else:
        render_mode = args.render

    cfg = load_config("data/muscle/config/bicep.json")
    cfg.gui = render_mode is not None
    cfg.render_mode = render_mode

    # OpenGL interactive: run until window closed (effectively unlimited)
    if render_mode == "human" and not args.auto:
        cfg.nsteps = 1_000_000

    sim = MuscleSim(cfg)
    log.info(f"n_verts={sim.n_verts}, use_jacobi={sim.use_jacobi}, render_mode={render_mode}")

    # USD mode is non-interactive, always use auto test path
    if args.auto or render_mode == "usd":
        _run_auto_test(sim, cfg)
    else:
        _run_interactive(sim, cfg)


def _activation_ramp(t: float) -> float:
    """Ramp activation over normalized time [0,1]: 0 -> 0.5 -> 1.0 -> 0.7 -> 0.3 -> 0."""
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


def _run_auto_test(sim, cfg):
    """Headless test: ramp activation over nsteps, render each frame, print diagnostics."""
    log.info(f"Running {cfg.nsteps} steps (auto, render={cfg.render_mode})...")
    for step in range(1, cfg.nsteps + 1):
        act = _activation_ramp(step / cfg.nsteps)
        wp.launch(fill_float_kernel, dim=sim.activation.shape[0],
                  inputs=[sim.activation, act])
        sim.step()
        sim.step_cnt = step
        sim.render()

        if step % 50 == 0 or step == 1:
            pos_np = sim.pos.numpy()
            centroid = pos_np.mean(axis=0)
            vol_err = sim.calc_vol_error()
            log.info(f"step={step:4d}  act={act:.2f}  "
                     f"centroid=({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})  "
                     f"vol_err={vol_err:.4f}")

    if sim.renderer is not None:
        sim.renderer.save()
    log.info("Auto test done.")


def _run_interactive(sim, cfg):
    """Interactive loop with WarpRenderer."""
    step_cnt = 0

    while sim.renderer is None or sim.renderer.is_running():
        if step_cnt >= cfg.nsteps:
            break

        if not cfg.pause:
            wp.launch(fill_float_kernel, dim=sim.activation.shape[0],
                      inputs=[sim.activation, cfg.activation])

            sim.step_start_time = time.perf_counter()
            sim.step()
            sim.step_end_time = time.perf_counter()
            step_cnt += 1
            sim.step_cnt = step_cnt

        sim.render()

    # Save USD if applicable
    if sim.renderer is not None:
        sim.renderer.save()

    log.info(f"Done. {step_cnt} steps completed.")


if __name__ == "__main__":
    main()
