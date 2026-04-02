"""Generate Stage 5 validation plots for documentation.

Usage:
    uv run python scripts/plot_stage5_results.py
"""
import sys, os
_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, _root)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "docs/imgs/tmp/stage5"
os.makedirs(OUT_DIR, exist_ok=True)


def plot_force_field_verification():
    """Plot 1: Force field at rest config (sliding ball) — direction and magnitude."""
    import warp as wp
    from examples.example_xpbd_millard_sliding_ball import load_config, _build_muscle_sim
    from VMuscle.muscle_warp import fill_float_kernel

    cfg = load_config("data/slidingBall/config_xpbd_millard.json")
    sim, bottom_ids, axis_idx, tets, vertices, fiber_dirs_per_tet, mc = _build_muscle_sim(cfg)
    n_tet = len(tets)
    wp.launch(fill_float_kernel, dim=n_tet, inputs=[sim.activation, wp.float32(1.0)])

    sim.clear_forces()
    sim.accumulate_active_fiber_force()
    force_np = sim.force.numpy()

    # Per-vertex force Z component
    z_forces = force_np[:, axis_idx]
    z_pos = vertices[:, axis_idx]
    stopped_np = sim.stopped.numpy()

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    mask_free = stopped_np == 0
    mask_stop = stopped_np == 1
    ax.scatter(z_pos[mask_free], z_forces[mask_free], c="tab:blue", s=20, label="Free vertices", zorder=3)
    ax.scatter(z_pos[mask_stop], z_forces[mask_stop], c="tab:red", s=20, label="Stopped vertices", zorder=3)
    ax.axhline(0, color="gray", linewidth=0.5)

    total_bottom = sum(z_forces[i] for i in bottom_ids)
    top_ids = [i for i in range(len(vertices)) if stopped_np[i] == 1]
    total_top = sum(z_forces[i] for i in top_ids)
    F_grav = cfg["ball_mass"] * cfg["gravity"]

    ax.set_xlabel("Vertex Z position (m)")
    ax.set_ylabel("Active fiber force Z (N)")
    ax.set_title(f"Explicit active force at rest (a=1.0, lm=1.0)\n"
                 f"Bottom: {total_bottom:.1f}N ↑  Top: {total_top:.1f}N ↓  "
                 f"Gravity: {F_grav:.1f}N ↓")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/force_field_rest.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUT_DIR}/force_field_rest.png")


def plot_simple_arm_trajectory():
    """Plot 2: Simple arm elbow angle trajectory from STO file."""
    sto_path = "output/SimpleArm_XPBD_Millard_Coupled_states.sto"
    if not os.path.exists(sto_path):
        print(f"  Skipped (no {sto_path})")
        return

    # Parse STO file
    with open(sto_path) as f:
        lines = f.readlines()
    header_end = 0
    for i, line in enumerate(lines):
        if line.strip() == "endheader":
            header_end = i + 1
            break
    cols = lines[header_end].strip().split('\t')
    data = []
    for line in lines[header_end + 1:]:
        vals = line.strip().split('\t')
        if len(vals) >= 2:
            data.append([float(v) for v in vals])
    data = np.array(data)
    t = data[:, 0]
    theta = np.degrees(data[:, 1])

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # Elbow angle
    axes[0].plot(t, theta, "tab:blue", linewidth=1.5)
    axes[0].axhline(88.25, color="tab:red", linestyle="--", linewidth=1, label="Reference: 88.25°")
    axes[0].set_ylabel("Elbow angle (deg)")
    axes[0].set_title("Simple Arm with Explicit Active Fiber Force (Stage 5)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Activation (col 4 if available) and force (col 3)
    if data.shape[1] >= 5:
        axes[1].plot(t, data[:, 4], "tab:green", linewidth=1.5, label="Activation")
        ax2 = axes[1].twinx()
        ax2.plot(t, data[:, 3], "tab:orange", linewidth=1, alpha=0.7, label="Force (N)")
        ax2.set_ylabel("Force (N)", color="tab:orange")
        axes[1].set_ylabel("Activation", color="tab:green")
        axes[1].legend(loc="upper left")
        ax2.legend(loc="upper right")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/simple_arm_trajectory.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUT_DIR}/simple_arm_trajectory.png")


if __name__ == "__main__":
    print("Generating Stage 5 plots...")
    plot_force_field_verification()
    plot_simple_arm_trajectory()
    print("Done.")
