"""Sliding-ball comparison: VBD-DGF / XPBD-DGF / XPBD-Millard vs OpenSim.

Usage:
    uv run python scripts/run_sliding_ball_comparison.py --mode vbd-dgf         # VBD-DGF vs OpenSim-DGF
    uv run python scripts/run_sliding_ball_comparison.py --mode xpbd-dgf        # XPBD-DGF vs OpenSim-DGF
    uv run python scripts/run_sliding_ball_comparison.py --mode xpbd-millard    # XPBD-Millard vs OpenSim-Millard
    uv run python scripts/run_sliding_ball_comparison.py --mode xpbd-millard --skip-opensim
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Runners: each returns an NPZ path (XPBD/VBD) or a result dict (OpenSim)
# ---------------------------------------------------------------------------

def run_vbd_dgf(config_path, label):
    print("=" * 60)
    print("VBD-DGF Simulation")
    print("=" * 60)
    from examples.example_muscle_sliding_ball import load_config, run_sim
    cfg = load_config(config_path)
    return run_sim(cfg, label=label)


def run_xpbd_dgf(config_path, label):
    print("=" * 60)
    print("XPBD-DGF Simulation")
    print("=" * 60)
    from examples.example_xpbd_dgf_sliding_ball import load_config, run_sim
    cfg = load_config(config_path)
    return run_sim(cfg, label=label)


def run_xpbd_millard(config_path, label):
    print("=" * 60)
    print("XPBD-Millard Simulation")
    print("=" * 60)
    from examples.example_xpbd_millard_sliding_ball import load_config, run_sim
    cfg = load_config(config_path)
    return run_sim(cfg, label=label)


def run_opensim_dgf(config_path):
    print("\n" + "=" * 60)
    print("OpenSim DGF Reference")
    print("=" * 60)
    with open(config_path) as f:
        raw = json.load(f)
    geo, phys, mus, sol = raw["geometry"], raw["physics"], raw["muscle"], raw["solver"]
    t_end = sol["n_steps"] * sol["dt"]
    try:
        from scripts.osim_sliding_ball import osim_sliding_ball
        os.makedirs("output", exist_ok=True)
        return osim_sliding_ball(
            muscle_length=geo["muscle_length"], ball_mass=phys["ball_mass"],
            sigma0=mus["sigma0"], muscle_radius=geo["muscle_radius"],
            excitation_func=lambda t: min(t / 0.05, 1.0), t_end=t_end, dt=0.001)
    except Exception as e:
        print(f"OpenSim DGF failed: {e}")
        return None


def run_opensim_millard(config_path):
    print("\n" + "=" * 60)
    print("OpenSim Millard Reference")
    print("=" * 60)
    with open(config_path) as f:
        raw = json.load(f)
    geo, phys, mus, sol = raw["geometry"], raw["physics"], raw["muscle"], raw["solver"]
    t_end = sol["n_steps"] * sol["dt"]
    try:
        from scripts.osim_sliding_ball import osim_sliding_ball_millard
        os.makedirs("output", exist_ok=True)
        return osim_sliding_ball_millard(
            muscle_length=geo["muscle_length"], ball_mass=phys["ball_mass"],
            sigma0=mus["sigma0"], muscle_radius=geo["muscle_radius"],
            excitation_func=lambda t: min(t / 0.05, 1.0), t_end=t_end, dt=0.001)
    except Exception as e:
        print(f"OpenSim Millard failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(xpbd_npz, osim_result, label, curve_module="dgf"):
    """Generate 2x2 comparison plot."""
    print("\n" + "=" * 60)
    print("Plotting")
    print("=" * 60)

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dx = np.load(xpbd_npz)
    xpbd_t = dx['times']
    xpbd_y = dx['positions']
    xpbd_nfl = dx['norm_fiber_lengths']
    xpbd_fa = dx['f_active']
    xpbd_a = dx['activations']
    sigma0 = float(dx['sigma0'])
    radius = float(dx['radius'])
    ball_mass = float(dx['ball_mass'])
    fmax = sigma0 * np.pi * radius ** 2

    has_osim = osim_result is not None

    # Get f_L curve for equilibrium plot
    l_norm = np.linspace(0.3, 1.9, 300)
    if curve_module == "millard":
        from VMuscle.millard_curves import MillardCurves
        mc = MillardCurves()
        fl_vals = mc.fl.eval(l_norm)
        curve_label = "Millard $f_L$"
        sim_label = "XPBD-Millard"
        osim_label = "OpenSim-Millard"
    else:
        from VMuscle.dgf_curves import active_force_length
        fl_vals = active_force_length(l_norm)
        curve_label = "DGF $f_L$"
        sim_label = "XPBD-DGF" if "xpbd" in label else "VBD-DGF"
        osim_label = "OpenSim-DGF"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(xpbd_t, xpbd_y, "b-", lw=1.5, label=sim_label)
    if has_osim:
        ax.plot(osim_result['times'], osim_result['positions'], "r--", lw=1.5, label=osim_label)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Ball position (m)")
    ax.set_title("Ball Trajectory"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(xpbd_t, xpbd_nfl, "b-", lw=1.5, label=sim_label)
    if has_osim:
        ax.plot(osim_result['times'], osim_result['norm_fiber_lengths'], "r--", lw=1.5, label=osim_label)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Normalized fiber length")
    ax.set_title("Fiber Length Over Time"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(l_norm, fl_vals, "b-", lw=1.5, label=curve_label)
    eq_force = ball_mass * 9.81 / fmax
    ax.axhline(y=eq_force, color='gray', ls=':', alpha=0.4,
               label=f"Weight/F$_{{max}}$={eq_force:.4f}")
    xpbd_l_eq = float(xpbd_nfl[-1])
    ax.axvline(x=xpbd_l_eq, color='b', ls=':', alpha=0.4)
    ax.plot(xpbd_l_eq, eq_force, 'bo', ms=8, zorder=6,
            label=f"{sim_label} eq. $\\tilde{{l}}$={xpbd_l_eq:.4f}")
    if has_osim:
        osim_l_eq = float(osim_result['norm_fiber_lengths'][-1])
        ax.axvline(x=osim_l_eq, color='r', ls=':', alpha=0.4)
        ax.plot(osim_l_eq, eq_force, 'rs', ms=8, zorder=6,
                label=f"{osim_label} eq. $\\tilde{{l}}$={osim_l_eq:.4f}")
    ax.set_xlabel("Normalized fiber length ($\\tilde{l}$)")
    ax.set_ylabel("Normalized force")
    ax.set_title(f"{curve_label.split('$')[0].strip()} Force-Length Curve")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlim(0.3, 1.9); ax.set_ylim(-0.05, 1.2)

    ax = axes[1, 1]
    ax.plot(xpbd_t, xpbd_a, "k-", lw=1.0, alpha=0.5, label="Activation")
    ax.plot(xpbd_t, xpbd_fa, "b-", lw=1.5, label=f"{sim_label} active")
    if has_osim:
        osim_fmax = osim_result.get('max_iso_force', fmax)
        ax.plot(osim_result['times'], osim_result['active_forces'] / osim_fmax,
                "r-", lw=1.5, label=f"{osim_label} active")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Normalized force")
    ax.set_title("Active Fiber Force"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    out_path = f"output/sliding_ball_comparison_{label}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")
    plt.close()

    # Summary
    print(f"\n--- Summary ---")
    print(f"  {sim_label} final NFL: {xpbd_l_eq:.4f}")
    print(f"  {sim_label} final pos: {xpbd_y[-1]:.6f} m")
    if has_osim:
        osim_l_eq = float(osim_result['norm_fiber_lengths'][-1])
        osim_pos = float(osim_result['positions'][-1])
        print(f"  {osim_label} NFL:      {osim_l_eq:.4f}")
        print(f"  {osim_label} pos:      {osim_pos:.6f} m")
        nfl_err = abs(xpbd_l_eq - osim_l_eq) / osim_l_eq * 100
        pos_err = abs(xpbd_y[-1] - osim_pos) / osim_pos * 100
        print(f"  NFL error: {nfl_err:.1f}%")
        print(f"  Pos error: {pos_err:.1f}%")

    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODE_CONFIGS = {
    "vbd-dgf":       "data/slidingBall/config.json",
    "xpbd-dgf":      "data/slidingBall/config_xpbd_dgf.json",
    "xpbd-millard":   "data/slidingBall/config_xpbd_millard.json",
}


def main():
    parser = argparse.ArgumentParser(description="Sliding-ball comparison")
    parser.add_argument("--mode",
                        choices=["vbd-dgf", "xpbd-dgf", "xpbd-millard"],
                        default="xpbd-millard",
                        help="vbd-dgf / xpbd-dgf / xpbd-millard")
    parser.add_argument("--config", default=None,
                        help="Override config JSON path")
    parser.add_argument("--label", default=None)
    parser.add_argument("--skip-opensim", action="store_true")
    args = parser.parse_args()

    config_path = args.config or MODE_CONFIGS[args.mode]
    label = args.label or args.mode.replace("-", "_")

    # Run simulation
    if args.mode == "vbd-dgf":
        npz = run_vbd_dgf(config_path, label)
    elif args.mode == "xpbd-dgf":
        npz = run_xpbd_dgf(config_path, label)
    elif args.mode == "xpbd-millard":
        npz = run_xpbd_millard(config_path, label)

    # Run OpenSim reference
    osim_result = None
    if not args.skip_opensim:
        if args.mode == "xpbd-millard":
            osim_result = run_opensim_millard(config_path)
        else:
            osim_result = run_opensim_dgf(config_path)

    # Plot
    curve = "millard" if args.mode == "xpbd-millard" else "dgf"
    plot_comparison(npz, osim_result, label, curve_module=curve)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
