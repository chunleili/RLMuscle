"""SimpleArm comparison: multiple solvers vs OpenSim reference.

Available modes (solver-hillmodel):
    osim                 OpenSim DGF vs Millard (reference only)
    mujoco-dgf           MuJoCo rigid-body + DGF force injection (no FEM)
    vbd-dgf              VBD FEM + MuJoCo, kinematic boundary (mass=0), DGF
    vbd-coupled-dgf      VBD FEM + Newton, spring ATTACH coupling, DGF
    xpbd-dgf             XPBD FEM + MuJoCo, elastic ATTACH coupling, DGF
    xpbd-millard         XPBD FEM + MuJoCo, elastic ATTACH coupling, Millard  [default]
    all                  Run all modes

Usage:
    uv run python scripts/run_simple_arm_comparison.py
    uv run python scripts/run_simple_arm_comparison.py --mode xpbd-dgf
    uv run python scripts/run_simple_arm_comparison.py --mode all
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_osim_dgf(cfg):
    from scripts.osim_simple_arm import osim_simple_arm_dgf
    return osim_simple_arm_dgf(cfg)


def run_osim_millard(cfg):
    from scripts.osim_simple_arm import osim_simple_arm_millard
    return osim_simple_arm_millard(cfg)


def run_mujoco_dgf(cfg):
    from examples.example_mujoco_simple_arm import mujoco_simple_arm
    return mujoco_simple_arm(cfg)


def run_vbd_dgf(cfg):
    from examples.example_vbd_mujoco_simple_arm import vbd_mujoco_simple_arm
    return vbd_mujoco_simple_arm(cfg)


def run_vbd_coupled_dgf(cfg):
    from examples.example_vbd_coupled_simple_arm import vbd_coupled_simple_arm
    return vbd_coupled_simple_arm(cfg)


def run_xpbd_dgf(cfg):
    cfg_copy = dict(cfg)
    cfg_copy.setdefault("xpbd", {})["hill_model_type"] = "dgf"
    from examples.example_xpbd_coupled_simple_arm import xpbd_coupled_simple_arm
    return xpbd_coupled_simple_arm(cfg_copy)


def run_xpbd_millard(cfg):
    cfg_copy = dict(cfg)
    cfg_copy.setdefault("xpbd", {})["hill_model_type"] = "millard"
    from examples.example_xpbd_coupled_simple_arm import xpbd_coupled_simple_arm
    return xpbd_coupled_simple_arm(cfg_copy)


def plot_comparison(datasets, title, out_path, colors=None):
    """Generate 2x2 comparison plot for any number of datasets.

    Args:
        datasets: list of (label, data_dict) tuples.
        title: Figure title.
        out_path: Output image path.
        colors: Optional dict label -> color.
    """
    if colors is None:
        palette = ["b", "r", "g", "m", "c"]
        colors = {label: palette[i % len(palette)] for i, (label, _) in enumerate(datasets)}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    linestyles = ["-", "--", "-.", ":"]

    for i, (label, data) in enumerate(datasets):
        if data is None:
            continue
        t = data["times"]
        c = colors.get(label, "k")
        ls = linestyles[i % len(linestyles)]

        axes[0, 0].plot(t, np.degrees(data["elbow_angles"]), f"{c}{ls}", lw=1.5, label=label)
        axes[0, 1].plot(t, data["activations"], f"{c}{ls}", lw=1.5, label=label)
        axes[1, 0].plot(t, data["norm_fiber_lengths"], f"{c}{ls}", lw=1.5, label=label)
        axes[1, 1].plot(t, data["forces"], f"{c}{ls}", lw=1.5, label=label)

    titles = ["Elbow Angle", "Activation", "Normalized Fiber Length", "Muscle Force"]
    ylabels = ["Angle (deg)", "Activation", "Norm. fiber length", "Force (N)"]
    for i, ax in enumerate(axes.flat):
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabels[i])
        ax.set_title(titles[i])
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Compute pairwise error metrics for first two datasets
    rmse = None
    max_err = None
    if len(datasets) >= 2 and datasets[0][1] is not None and datasets[1][1] is not None:
        d0, d1 = datasets[0][1], datasets[1][1]
        t_common = np.linspace(
            max(d0["times"][0], d1["times"][0]),
            min(d0["times"][-1], d1["times"][-1]),
            500)
        a0 = np.interp(t_common, d0["times"], np.degrees(d0["elbow_angles"]))
        a1 = np.interp(t_common, d1["times"], np.degrees(d1["elbow_angles"]))
        if not (np.any(np.isnan(a0)) or np.any(np.isnan(a1))):
            rmse = np.sqrt(np.mean((a0 - a1) ** 2))
            max_err = np.max(np.abs(a0 - a1))
            fig.text(0.5, 0.01,
                     f"{datasets[0][0]} vs {datasets[1][0]}: RMSE={rmse:.2f} deg, Max error={max_err:.2f} deg",
                     ha="center", fontsize=11, style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    os.makedirs("output", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")
    plt.close()
    return rmse, max_err


def main():
    parser = argparse.ArgumentParser(description="SimpleArm comparison")
    parser.add_argument("--config", default="data/simpleArm/config.json")
    parser.add_argument("--mode", choices=[
                            "osim", "mujoco-dgf", "vbd-dgf", "vbd-coupled-dgf",
                            "xpbd-dgf", "xpbd-millard", "all"],
                        default="xpbd-millard",
                        help="Solver mode (solver-hillmodel-coupling)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    if args.mode in ("osim", "all"):
        print("=" * 60)
        print("OpenSim DGF vs Millard")
        print("=" * 60)
        dgf = run_osim_dgf(cfg)
        millard = run_osim_millard(cfg)
        plot_comparison(
            [("DGF", dgf), ("Millard", millard)],
            "SimpleArm: DGF vs Millard (OpenSim)",
            "output/simple_arm_osim_dgf_vs_millard.png",
            {"DGF": "b", "Millard": "r"},
        )

    if args.mode in ("mujoco-dgf", "all"):
        print("\n" + "=" * 60)
        print("MuJoCo DGF vs OpenSim DGF")
        print("=" * 60)
        dgf = run_osim_dgf(cfg) if args.mode == "mujoco-dgf" else dgf
        mujoco = run_mujoco_dgf(cfg)
        rmse, max_err = plot_comparison(
            [("OpenSim DGF", dgf), ("MuJoCo DGF", mujoco)],
            "SimpleArm: MuJoCo vs OpenSim (DGF)",
            "output/simple_arm_mujoco_vs_osim.png",
            {"OpenSim DGF": "b", "MuJoCo DGF": "r"},
        )
        if rmse is not None:
            print(f"\nMuJoCo vs OpenSim DGF: RMSE={rmse:.2f} deg, Max error={max_err:.2f} deg")

    if args.mode in ("vbd-dgf", "all"):
        print("\n" + "=" * 60)
        print("VBD+MuJoCo vs OpenSim DGF")
        print("=" * 60)
        dgf = run_osim_dgf(cfg) if args.mode == "vbd-dgf" else dgf
        vbd = run_vbd_dgf(cfg)
        rmse, max_err = plot_comparison(
            [("OpenSim DGF", dgf), ("VBD+MuJoCo", vbd)],
            "SimpleArm: VBD+MuJoCo vs OpenSim (DGF)",
            "output/simple_arm_vbd_vs_osim.png",
            {"OpenSim DGF": "b", "VBD+MuJoCo": "g"},
        )
        if rmse is not None:
            print(f"\nVBD+MuJoCo vs OpenSim DGF: RMSE={rmse:.2f} deg, Max error={max_err:.2f} deg")

    if args.mode in ("vbd-coupled-dgf", "all"):
        print("\n" + "=" * 60)
        print("VBD Coupled vs OpenSim DGF")
        print("=" * 60)
        dgf = run_osim_dgf(cfg) if args.mode == "vbd-coupled-dgf" else dgf
        coupled = run_vbd_coupled_dgf(cfg)
        rmse, max_err = plot_comparison(
            [("OpenSim DGF", dgf), ("VBD Coupled", coupled)],
            "SimpleArm: VBD Coupled vs OpenSim (DGF)",
            "output/simple_arm_coupled_vs_osim.png",
            {"OpenSim DGF": "b", "VBD Coupled": "m"},
        )
        if rmse is not None:
            print(f"\nVBD Coupled vs OpenSim DGF: RMSE={rmse:.2f} deg, Max error={max_err:.2f} deg")

    if args.mode in ("xpbd-dgf", "all"):
        print("\n" + "=" * 60)
        print("XPBD-DGF vs OpenSim DGF")
        print("=" * 60)
        dgf = run_osim_dgf(cfg) if args.mode == "xpbd-dgf" else dgf
        xpbd = run_xpbd_dgf(cfg)
        rmse, max_err = plot_comparison(
            [("OpenSim DGF", dgf), ("XPBD-DGF", xpbd)],
            "SimpleArm: XPBD-DGF vs OpenSim (DGF)",
            "output/simple_arm_xpbd_dgf_vs_osim.png",
            {"OpenSim DGF": "b", "XPBD-DGF": "c"},
        )
        if rmse is not None:
            print(f"\nXPBD-DGF vs OpenSim DGF: RMSE={rmse:.2f} deg, Max error={max_err:.2f} deg")

    if args.mode in ("xpbd-millard", "all"):
        print("\n" + "=" * 60)
        print("XPBD-Millard vs OpenSim Millard")
        print("=" * 60)
        millard_osim = run_osim_millard(cfg)
        xpbd_mill = run_xpbd_millard(cfg)
        rmse, max_err = plot_comparison(
            [("OpenSim Millard", millard_osim), ("XPBD Millard", xpbd_mill)],
            "SimpleArm: XPBD Millard vs OpenSim Millard",
            "output/simple_arm_xpbd_millard_vs_osim.png",
            {"OpenSim Millard": "r", "XPBD Millard": "b"},
        )
        if rmse is not None:
            print(f"\nXPBD Millard vs OpenSim Millard: RMSE={rmse:.2f} deg, Max error={max_err:.2f} deg")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
