"""Plot VBD sliding-ball results, optionally overlay OpenSim reference.

Reads:
  output/vbd_muscle_sliding_ball.csv           (from example_muscle_sliding_ball)
  output/opensim_muscle_comparison.npz         (optional, from OpenSimExample)

Usage:
    uv run scripts/plot_sliding_ball.py
"""

import csv
import os

import numpy as np


def load_vbd_csv(path="output/vbd_muscle_sliding_ball.csv"):
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return {
        "times": np.array([float(r["time"]) for r in rows]),
        "positions": np.array([float(r["ball_y"]) for r in rows]),
        "norm_fiber_lengths": np.array([float(r["norm_fiber_length"]) for r in rows]),
    }


def load_opensim_npz(path="output/opensim_muscle_comparison.npz"):
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vbd = load_vbd_csv()
    osim = load_opensim_npz()
    has_osim = osim is not None and len(osim.get("times", [])) > 0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # -- Position vs time --
    ax = axes[0]
    ax.plot(vbd["times"], vbd["positions"], "b-", lw=1.5, label="VBD")
    if has_osim:
        ax.plot(osim["times"], osim["positions"], "r--", lw=1.5, label="OpenSim")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Ball position (m)")
    ax.set_title("Position")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # -- Fiber length vs time --
    ax = axes[1]
    ax.plot(vbd["times"], vbd["norm_fiber_lengths"], "b-", lw=1.5, label="VBD")
    if has_osim:
        ax.plot(osim["times"], osim["norm_fiber_lengths"], "r--", lw=1.5, label="OpenSim")
    ax.axhline(1.0, color="gray", ls=":", alpha=0.5, label="optimal")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized fiber length")
    ax.set_title("Fiber Length")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    path = "output/vbd_muscle_sliding_ball.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {path}")


if __name__ == "__main__":
    main()
