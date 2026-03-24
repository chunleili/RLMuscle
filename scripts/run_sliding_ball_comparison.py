"""One-click VBD vs OpenSim sliding-ball comparison.

Runs VBD simulation, OpenSim reference, and generates comparison plot.

Usage:
    uv run python scripts/run_sliding_ball_comparison.py
    uv run python scripts/run_sliding_ball_comparison.py --config data/slidingBall/config.json
    uv run python scripts/run_sliding_ball_comparison.py --skip-opensim  # VBD only
"""

import argparse
import json
import os
import sys

# Ensure project root is on sys.path for 'examples' import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def run_vbd(config_path, label):
    """Run VBD simulation via example module."""
    print("=" * 60)
    print("Step 1: VBD Simulation")
    print("=" * 60)
    from examples.example_muscle_sliding_ball import load_config, run_sim
    cfg = load_config(config_path)
    return run_sim(cfg, label=label)


def run_opensim(config_path):
    """Run OpenSim reference simulation matching config parameters."""
    print("\n" + "=" * 60)
    print("Step 2: OpenSim Reference")
    print("=" * 60)

    with open(config_path) as f:
        raw = json.load(f)
    geo = raw["geometry"]
    phys = raw["physics"]
    mus = raw["muscle"]
    sol = raw["solver"]

    length = geo["muscle_length"]
    radius = geo["muscle_radius"]
    sigma0 = mus["sigma0"]
    ball_mass = phys["ball_mass"]
    t_end = sol["n_steps"] * sol["dt"]

    try:
        import importlib.util
        # Find osim_sliding_ball.py in OpenSimExample
        osim_paths = [
            os.path.join(os.path.dirname(__file__), "..", "..",
                         "OpenSimExample", "OpenSimExample",
                         "vbd_muscle", "osim_sliding_ball.py"),
            "D:/Dev/OpenSimExample/OpenSimExample/vbd_muscle/osim_sliding_ball.py",
        ]
        osim_path = None
        for p in osim_paths:
            p = os.path.normpath(p)
            if os.path.exists(p):
                osim_path = p
                break

        if osim_path is None:
            print("OpenSim script not found, skipping.")
            return None

        spec = importlib.util.spec_from_file_location("osim_sliding_ball", osim_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        os.makedirs("output", exist_ok=True)
        result = mod.osim_sliding_ball(
            muscle_length=length,
            ball_mass=ball_mass,
            sigma0=sigma0,
            muscle_radius=radius,
            excitation_func=lambda t: min(t / 0.05, 1.0),
            t_end=t_end,
            dt=0.001,
        )

        # Copy .sto to local output
        sto_src = "output/opensim_sliding_ball.sto"
        if os.path.exists(sto_src):
            return sto_src
        return None

    except Exception as e:
        print(f"OpenSim failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_comparison(vbd_npz, sto_path, label):
    """Generate comparison plot."""
    print("\n" + "=" * 60)
    print("Step 3: Plotting")
    print("=" * 60)

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- DGF curves (matching kernel / OpenSim) ---
    b11, b21, b31, b41 = 0.815, 1.055, 0.162, 0.063
    b12, b22, b32, b42 = 0.433, 0.717, -0.030, 0.200
    b13, b23, b33, b43 = 0.1, 1.0, 0.354, 0.0
    KPE = 4.0

    def _gauss(x, b1, b2, b3, b4):
        s = b3 + b4 * x
        return b1 * np.exp(-0.5 * (x - b2) ** 2 / np.maximum(s, 1e-6) ** 2)

    def afl(x):
        x = np.asarray(x, dtype=float)
        return _gauss(x, b11, b21, b31, b41) + _gauss(x, b12, b22, b32, b42) + _gauss(x, b13, b23, b33, b43)

    def pfl(x):
        x = np.asarray(x, dtype=float)
        offset = np.exp(KPE * (0.2 - 1.0) / 0.6)
        denom = np.exp(KPE) - offset
        return (np.exp(np.clip(KPE * (x - 1.0) / 0.6, -50, 50)) - offset) / denom

    # --- Load data ---
    d = np.load(vbd_npz)
    vbd_t, vbd_y = d['times'], d['positions']
    vbd_vz = d['velocities'] if 'velocities' in d else None
    vbd_nfl = d['norm_fiber_lengths']
    sigma0 = float(d['sigma0'])
    radius = float(d['radius'])
    ball_mass = float(d['ball_mass'])
    fmax = sigma0 * np.pi * radius ** 2

    osim = None
    if sto_path and os.path.exists(sto_path):
        with open(sto_path) as f:
            for line in f:
                if line.strip() == "endheader":
                    break
            cols = f.readline().strip().split('\t')
            rows = [[float(v) for v in l.strip().split('\t')] for l in f]
        data = np.array(rows)
        osim = {n: data[:, i] for i, n in enumerate(cols)}

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) Ball trajectory
    ax = axes[0, 0]
    ax.plot(vbd_t, vbd_y, "b-", lw=1.5, label="VBD")
    if osim:
        ax.plot(osim['time'], osim['position'], "r--", lw=1.5, label="OpenSim")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Ball position (m)")
    ax.set_title("Ball Trajectory"); ax.legend(); ax.grid(True, alpha=0.3)

    # (0,1) Fiber length
    ax = axes[0, 1]
    ax.plot(vbd_t, vbd_nfl, "b-", lw=1.5, label="VBD (mean)")
    if osim:
        ax.plot(osim['time'], osim['norm_fiber_length'], "r--", lw=1.5, label="OpenSim")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Normalized fiber length")
    ax.set_title("Fiber Length Over Time"); ax.legend(); ax.grid(True, alpha=0.3)

    # (1,0) F-L curves + equilibrium
    ax = axes[1, 0]
    l_norm = np.linspace(0.2, 1.8, 200)
    fa, fp = afl(l_norm), pfl(l_norm)
    ax.plot(l_norm, fa, "b-", lw=1.5, label="Active $f_L$")
    ax.plot(l_norm, fp, "g-", lw=1.5, label="Passive $f_{PE}$")
    ax.plot(l_norm, fa + fp, "k--", lw=1.2, alpha=0.6, label="Total (a=1)")
    eq_force = ball_mass * 9.81 / fmax
    ax.axhline(y=eq_force, color='gray', ls=':', alpha=0.4,
               label=f"Weight/F$_{{max}}$={eq_force:.2f}")
    vbd_l_eq = float(vbd_nfl[-1])
    ax.axvline(x=vbd_l_eq, color='b', ls=':', alpha=0.4)
    ax.plot(vbd_l_eq, eq_force, 'bo', ms=8, zorder=6,
            label=f"VBD eq. $\\tilde{{l}}$={vbd_l_eq:.2f}")
    if osim:
        osim_l_eq = float(osim['norm_fiber_length'][-1])
        ax.axvline(x=osim_l_eq, color='r', ls=':', alpha=0.4)
        ax.plot(osim_l_eq, eq_force, 'rs', ms=8, zorder=6,
                label=f"OpenSim eq. $\\tilde{{l}}$={osim_l_eq:.2f}")
    ax.set_xlabel("Normalized fiber length ($\\tilde{l}$)")
    ax.set_ylabel("Normalized force")
    ax.set_title("DGF Force-Length Curves")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlim(0.2, 1.8); ax.set_ylim(-0.05, 1.2)

    # (1,1) Force on ball
    ax = axes[1, 1]
    if vbd_vz is not None:
        dt_vbd = np.diff(vbd_t)
        acc = np.diff(vbd_vz) / dt_vbd
        vbd_force = ball_mass * (9.81 + acc)
        ax.plot(vbd_t[:-1], vbd_force / fmax, "b-", lw=1.5, label="VBD")
    ax.axhline(y=eq_force, color='gray', ls=':', alpha=0.5,
               label=f"Weight/F_max={eq_force:.3f}")
    if osim:
        ax.plot(osim['time'], osim['force'] / fmax, "r--", lw=1.5, label="OpenSim")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Force / F_max")
    ax.set_title("Force on Ball"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    out_path = f"output/sliding_ball_comparison_{label}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description="VBD vs OpenSim sliding-ball comparison")
    parser.add_argument("--config", default="data/slidingBall/config.json")
    parser.add_argument("--label", default="default")
    parser.add_argument("--skip-opensim", action="store_true",
                        help="Skip OpenSim, plot VBD only")
    args = parser.parse_args()

    # Step 1: VBD
    vbd_npz = run_vbd(args.config, args.label)

    # Step 2: OpenSim
    sto_path = None
    if not args.skip_opensim:
        sto_path = run_opensim(args.config)

    # Step 3: Plot
    plot_comparison(vbd_npz, sto_path, args.label)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
