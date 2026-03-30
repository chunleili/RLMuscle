"""One-click XPBD-DGF vs OpenSim sliding-ball comparison.

Runs XPBD-DGF simulation, optionally OpenSim reference, and generates comparison plot.

Usage:
    uv run python scripts/run_xpbd_dgf_sliding_ball_comparison.py
    uv run python scripts/run_xpbd_dgf_sliding_ball_comparison.py --skip-opensim
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def run_xpbd_dgf(config_path, label):
    """Run XPBD-DGF simulation."""
    print("=" * 60)
    print("Step 1: XPBD-DGF Simulation")
    print("=" * 60)
    from examples.example_xpbd_dgf_sliding_ball import load_config, run_sim
    cfg = load_config(config_path)
    return run_sim(cfg, label=label)


def run_opensim(config_path):
    import  json
    """Run OpenSim reference simulation matching config parameters."""
    print("\n" + "=" * 60)
    print("Step 2: OpenSim Reference")
    print("=" * 60)

    with open(config_path) as f:
        raw = json.load(f)
    geo, phys, mus, sol = raw["geometry"], raw["physics"], raw["muscle"], raw["solver"]
    t_end = sol["n_steps"] * sol["dt"]

    try:
        from scripts.osim_sliding_ball import osim_sliding_ball
        os.makedirs("output", exist_ok=True)
        return osim_sliding_ball(
            muscle_length=geo["muscle_length"],
            ball_mass=phys["ball_mass"],
            sigma0=mus["sigma0"],
            muscle_radius=geo["muscle_radius"],
            excitation_func=lambda t: min(t / 0.05, 1.0),
            t_end=t_end,
            dt=0.001,
        )
    except Exception as e:
        print(f"OpenSim failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_comparison(xpbd_npz, osim_result, label):
    """Generate XPBD-DGF vs OpenSim comparison plot."""
    print("\n" + "=" * 60)
    print("Step 3: Plotting")
    print("=" * 60)

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from VMuscle.dgf_curves import active_force_length as afl, passive_force_length as pfl

    # Load XPBD-DGF data
    dx = np.load(xpbd_npz)
    xpbd_t = dx['times']
    xpbd_y = dx['positions']
    xpbd_nfl = dx['norm_fiber_lengths']
    xpbd_fa = dx['f_active']
    xpbd_fp = dx['f_passive']
    xpbd_ft = dx['f_total']
    xpbd_a = dx['activations']
    sigma0 = float(dx['sigma0'])
    radius = float(dx['radius'])
    ball_mass = float(dx['ball_mass'])
    fmax = sigma0 * np.pi * radius ** 2

    has_osim = osim_result is not None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) Ball trajectory
    ax = axes[0, 0]
    ax.plot(xpbd_t, xpbd_y, "b-", lw=1.5, label="XPBD-DGF")
    if has_osim:
        ax.plot(osim_result['times'], osim_result['positions'], "r--", lw=1.5, label="OpenSim")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Ball position (m)")
    ax.set_title("Ball Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (0,1) Fiber length
    ax = axes[0, 1]
    ax.plot(xpbd_t, xpbd_nfl, "b-", lw=1.5, label="XPBD-DGF")
    if has_osim:
        ax.plot(osim_result['times'], osim_result['norm_fiber_lengths'], "r--", lw=1.5, label="OpenSim")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized fiber length")
    ax.set_title("Fiber Length Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,0) DGF F-L curves + equilibrium points
    ax = axes[1, 0]
    l_norm = np.linspace(0.2, 1.8, 200)
    fa, fp = afl(l_norm), pfl(l_norm)
    ax.plot(l_norm, fa, "b-", lw=1.5, label="Active $f_L$")
    ax.plot(l_norm, fp, "g-", lw=1.5, label="Passive $f_{PE}$")
    ax.plot(l_norm, fa + fp, "k--", lw=1.2, alpha=0.6, label="Total (a=1)")
    eq_force = ball_mass * 9.81 / fmax
    ax.axhline(y=eq_force, color='gray', ls=':', alpha=0.4,
               label=f"Weight/F$_{{max}}$={eq_force:.4f}")

    xpbd_l_eq = float(xpbd_nfl[-1])
    ax.axvline(x=xpbd_l_eq, color='b', ls=':', alpha=0.4)
    ax.plot(xpbd_l_eq, eq_force, 'bo', ms=8, zorder=6,
            label=f"XPBD eq. $\\tilde{{l}}$={xpbd_l_eq:.3f}")
    if has_osim:
        osim_l_eq = float(osim_result['norm_fiber_lengths'][-1])
        ax.axvline(x=osim_l_eq, color='r', ls=':', alpha=0.4)
        ax.plot(osim_l_eq, eq_force, 'rs', ms=8, zorder=6,
                label=f"OpenSim eq. $\\tilde{{l}}$={osim_l_eq:.3f}")

    ax.set_xlabel("Normalized fiber length ($\\tilde{l}$)")
    ax.set_ylabel("Normalized force")
    ax.set_title("DGF Force-Length Curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.2, 1.8)
    ax.set_ylim(-0.05, 1.2)

    # (1,1) Forces comparison
    ax = axes[1, 1]
    ax.plot(xpbd_t, xpbd_a, "k-", lw=1.0, alpha=0.5, label="Activation")
    ax.plot(xpbd_t, xpbd_fa, "b-", lw=1.5, label="XPBD active")
    ax.plot(xpbd_t, xpbd_fp, "b--", lw=1.0, label="XPBD passive")
    ax.plot(xpbd_t, xpbd_ft, "b:", lw=1.5, label="XPBD total")
    if has_osim:
        osim_fmax = osim_result.get('max_iso_force', fmax)
        ax.plot(osim_result['times'], osim_result['active_forces'] / osim_fmax,
                "r-", lw=1.5, label="OpenSim active")
        ax.plot(osim_result['times'], osim_result['passive_forces'] / osim_fmax,
                "r--", lw=1.0, label="OpenSim passive")
        ax.plot(osim_result['times'], osim_result['forces'] / osim_fmax,
                "r:", lw=1.5, label="OpenSim total")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized force")
    ax.set_title("Fiber Forces (DGF curves)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    out_path = f"output/xpbd_dgf_sliding_ball_comparison_{label}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="XPBD-DGF vs OpenSim sliding-ball comparison")
    parser.add_argument("--xpbd-config",
                        default="data/slidingBall/config_xpbd_dgf.json",
                        help="XPBD-DGF config JSON")
    parser.add_argument("--label", default="default")
    parser.add_argument("--skip-opensim", action="store_true",
                        help="Skip OpenSim, plot XPBD-DGF only")
    args = parser.parse_args()

    # Step 1: XPBD-DGF
    xpbd_npz = run_xpbd_dgf(args.xpbd_config, args.label)

    # Step 2: OpenSim (optional)
    osim_result = None
    if not args.skip_opensim:
        osim_result = run_opensim(args.xpbd_config)

    # Step 3: Plot
    plot_comparison(xpbd_npz, osim_result, args.label)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
