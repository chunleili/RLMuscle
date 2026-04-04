"""Mesh quality test harness for bicep XPBD experiments.

Runs standalone MuscleSim (no coupling) with activation ramp,
reports inverted tet count and det(F) statistics per frame.

Usage:
    uv run python scripts/test_mesh_quality.py --config <config.json> [--frames 50] [--label test]
"""
import argparse
import json
import os
import sys

os.environ.setdefault("WARP_CACHE_PATH", ".cache/warp")

import numpy as np
import warp as wp

from VMuscle.config import load_config
from VMuscle.muscle_warp import MuscleSim, fill_float_kernel


def build_rest_matrices(pos, tet_np):
    n_tets = len(tet_np)
    rm = np.zeros((n_tets, 3, 3), dtype=np.float32)
    for e in range(n_tets):
        i0, i1, i2, i3 = tet_np[e]
        M = np.column_stack([pos[i0] - pos[i3], pos[i1] - pos[i3], pos[i2] - pos[i3]])
        if abs(np.linalg.det(M)) > 1e-30:
            rm[e] = np.linalg.inv(M)
    return rm


def check_quality(pos_np, tet_np, rm):
    n_tets = len(tet_np)
    dets = np.zeros(n_tets, dtype=np.float32)
    for e in range(n_tets):
        i0, i1, i2, i3 = tet_np[e]
        Ds = np.column_stack([pos_np[i0] - pos_np[i3], pos_np[i1] - pos_np[i3], pos_np[i2] - pos_np[i3]])
        dets[e] = np.linalg.det(Ds @ rm[e])
    n_inv = int((dets <= 0).sum())
    return n_inv, dets


def run_test(config_path, n_frames=50, label="test", extra_constraints=None,
             override_params=None, post_step_fn=None):
    wp.init()
    wp.set_device("cpu")

    cfg = load_config(config_path)
    cfg.gui = False
    cfg.render_mode = None
    cfg.arch = "cpu"

    if extra_constraints:
        for ec in extra_constraints:
            cfg.constraints.append(ec)

    if override_params:
        for k, v in override_params.items():
            setattr(cfg, k, v)

    sim = MuscleSim(cfg)
    n_tets = len(sim.tet_np)
    tet_np = sim.tet_np.astype(np.int32)
    rest_pos = sim.pos.numpy().copy()
    rm = build_rest_matrices(rest_pos, tet_np)

    results = []
    for frame in range(1, n_frames + 1):
        # Gradual activation ramp
        act = min(frame * 0.05, 1.0) if frame >= 3 else 0.0
        cfg.activation = act
        wp.launch(fill_float_kernel, dim=n_tets,
                  inputs=[sim.activation, wp.float32(act)])
        sim.step()

        if post_step_fn:
            post_step_fn(sim)

        n_inv, dets = check_quality(sim.pos.numpy(), tet_np, rm)
        valid_dets = dets[dets > 0]
        results.append({
            "frame": frame, "activation": act, "n_inv": n_inv,
            "min_J": float(dets.min()), "max_J": float(dets.max()),
            "median_J": float(np.median(valid_dets)) if len(valid_dets) > 0 else 0.0,
        })

        if frame <= 10 or frame % 10 == 0:
            print(f"[{label}] f={frame:3d} a={act:.2f} inv={n_inv:4d}/{n_tets} "
                  f"J=[{dets.min():.4f}, {dets.max():.4f}] "
                  f"med={results[-1]['median_J']:.4f}")

    # Summary
    max_inv = max(r["n_inv"] for r in results)
    final_inv = results[-1]["n_inv"]
    worst_J = min(r["min_J"] for r in results)
    print(f"\n[{label}] SUMMARY: max_inv={max_inv}/{n_tets} final_inv={final_inv} worst_J={worst_J:.4f}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data/muscle/config/bicep_xpbd_millard.json")
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--label", default="default")
    args = parser.parse_args()
    run_test(args.config, args.frames, args.label)
