"""Sweep post_smooth_iters and SVD repair params to find optimal mesh quality.

Usage:
    uv run python scripts/sweep_post_smooth.py
"""
import sys
import os
import json
import copy
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
os.environ.setdefault("WARP_CACHE_PATH", os.path.join(PROJECT_ROOT, ".cache", "warp"))

import warp as wp
wp.init()
wp.set_device("cpu")

from VMuscle.config import load_config
from VMuscle.controllability import build_coupling_config
from VMuscle.muscle_warp import MuscleSim
from VMuscle.solver_muscle_bone_coupled_warp import SolverMuscleBoneCoupled

sys.path.insert(0, os.path.join(PROJECT_ROOT, "examples"))
from example_couple3 import (
    build_elbow_model, ELBOW_PIVOT, ELBOW_AXIS, _activation_schedule,
)

BASE_CONFIG = os.path.join(PROJECT_ROOT,
                           "data/muscle/config/bicep_fibermillard_coupled.json")
N_STEPS = 300
DT = 1.0 / 60.0
USD_SOURCE = "data/muscle/model/bicep.usd"


def run_one(overrides: dict, label: str):
    """Run simulation with config overrides, return summary metrics."""
    cfg = load_config(BASE_CONFIG)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    cfg.gui = False
    cfg.render_mode = None
    cfg.geo_path = USD_SOURCE
    cfg.bone_geo_path = USD_SOURCE
    cfg.muscle_prim_path = "/character/muscle/bicep"
    cfg.bone_prim_paths = {
        "L_scapula": "/character/bone/L_scapula/L_scapulaShape",
        "L_radius": "/character/bone/L_radius/L_radiusShape",
        "L_humerus": "/character/bone/L_humerus/L_humerusShape",
    }
    if cfg.constraints:
        for c in cfg.constraints:
            if "target_path" in c:
                c["target_path"] = USD_SOURCE

    sim = MuscleSim(cfg)
    jf = float(getattr(cfg, "joint_friction", 0.05))
    model, state, radius_link, joint, sel = build_elbow_model(sim, joint_friction=jf)
    control_config = build_coupling_config("smooth_nonlinear")
    solver = SolverMuscleBoneCoupled(model, sim, control_config=control_config)
    if radius_link is not None and sel.size > 0:
        solver.configure_coupling(
            bone_body_id=radius_link,
            bone_rest_verts=sim.bone_pos[sel].astype(np.float32),
            bone_vertex_indices=sel,
            joint_index=joint,
            joint_pivot=ELBOW_PIVOT,
            joint_axis=ELBOW_AXIS,
        )

    tet_idx = sim.tet_np[:, [3, 0, 1, 2]]
    tet_poses = sim.rest_matrix.numpy()

    import time as _time

    peak_inv = 0
    worst_det = 999.0
    peak_torque = 0.0
    inv_at_end = 0
    inv_history = []

    t0 = _time.perf_counter()
    for step in range(1, N_STEPS + 1):
        cfg.activation = _activation_schedule(step, N_STEPS)
        solver.step(state, state, dt=DT)

        pos_np = solver.core.pos.numpy()
        p = pos_np[tet_idx]
        Ds = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]], axis=-1)
        F = np.einsum("mij,mjk->mik", Ds, tet_poses)
        det_F = np.linalg.det(F)
        n_inv = int(np.sum(det_F <= 0))
        w_det = float(np.min(det_F))

        peak_inv = max(peak_inv, n_inv)
        worst_det = min(worst_det, w_det)
        peak_torque = max(peak_torque, abs(float(solver._axis_torque)))
        inv_history.append(n_inv)

        if step % 50 == 0:
            print(f"    [{label}] step {step}: inv={n_inv}  det={w_det:.3f}  tau={solver._axis_torque:.3f}")

    elapsed = _time.perf_counter() - t0
    ms_per_step = elapsed / N_STEPS * 1000.0

    # Recovery: average inverted in last 20 steps
    inv_at_end = int(np.mean(inv_history[-20:]))

    return {
        "label": label,
        "peak_inv": peak_inv,
        "worst_det": worst_det,
        "peak_torque": peak_torque,
        "inv_at_end": inv_at_end,
        "inv_history": inv_history,
        "elapsed_s": elapsed,
        "ms_per_step": ms_per_step,
    }


def main():
    experiments = [
        # Reference (current best)
        ({"post_smooth_iters": 5, "repair_alpha": 0.1, "repair_iters": 1, "repair_sigma_min": 0.05},
         "a=0.1 σ=0.05 i=1 (ref)"),
        # New tests
        ({"post_smooth_iters": 5, "repair_alpha": 0.2, "repair_iters": 1, "repair_sigma_min": 0.05},
         "a=0.2 σ=0.05 i=1"),
        ({"post_smooth_iters": 5, "repair_alpha": 0.2, "repair_iters": 1, "repair_sigma_min": 0.01},
         "a=0.2 σ=0.01 i=1"),
    ]

    results = []
    for overrides, label in experiments:
        print(f"\n{'='*60}")
        print(f"  Running: {label}")
        print(f"  Overrides: {overrides}")
        print(f"{'='*60}")
        r = run_one(overrides, label)
        results.append(r)

    # Summary table with timing
    base_ms = results[0]["ms_per_step"]  # "no fix" as reference
    hdr = f"{'Config':<25} {'Peak':>5} {'W.det':>7} {'τ':>5} {'End':>4} {'ms/st':>7} {'overhead':>8}"
    print(f"\n{'='*len(hdr)}")
    print(hdr)
    print(f"{'-'*len(hdr)}")
    for r in results:
        overhead = (r["ms_per_step"] / base_ms - 1.0) * 100.0
        print(f"{r['label']:<25} {r['peak_inv']:>5} {r['worst_det']:>7.2f} "
              f"{r['peak_torque']:>5.1f} {r['inv_at_end']:>4} "
              f"{r['ms_per_step']:>6.1f}  {overhead:>+6.1f}%")
    print(f"{'='*len(hdr)}")

    # Plot comparison
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    t = np.arange(1, N_STEPS + 1) * DT
    for r in results:
        ax1.plot(t, r["inv_history"], linewidth=1.2, label=r["label"])
    ax1.set_ylabel("Inverted Tets")
    ax1.set_title("SVD Repair Parameter Sweep (smooth=5 fixed)")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Bar chart: peak_inv vs ms_per_step
    labels = [r["label"] for r in results]
    peaks = [r["peak_inv"] for r in results]
    times = [r["ms_per_step"] for r in results]
    x = np.arange(len(labels))
    w = 0.35
    ax2b = ax2.twinx()
    bars1 = ax2.bar(x - w/2, peaks, w, label="Peak Inverted", color="tab:red", alpha=0.7)
    bars2 = ax2b.bar(x + w/2, times, w, label="ms/step", color="tab:blue", alpha=0.7)
    ax2.set_ylabel("Peak Inverted Tets", color="tab:red")
    ax2b.set_ylabel("ms / step", color="tab:blue")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.legend(loc="upper left", fontsize=8)
    ax2b.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(PROJECT_ROOT, "output", "sweep_post_smooth.png")
    fig.savefig(path, dpi=150)
    print(f"\nPlot saved: {path}")
    plt.close()


if __name__ == "__main__":
    main()
