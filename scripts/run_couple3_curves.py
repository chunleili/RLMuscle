"""Plot couple3 joint angle / activation / torque / inverted tets vs time.

Usage:
    uv run python scripts/run_couple3_curves.py
"""

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("WARP_CACHE_PATH", os.path.join(PROJECT_ROOT, ".cache", "warp"))

from examples.example_couple3 import setup_couple3, _activation_schedule

N_STEPS = 300
DT = 1.0 / 60.0


def collect_data(solver, sim, state, cfg, n_steps):
    """Run simulation and collect per-step metrics."""
    tet_idx = sim.tet_np[:, [3, 0, 1, 2]]
    tet_poses = sim.rest_matrix.numpy()

    steps, excitations, activations = [], [], []
    torques, angles, inverted_tets = [], [], []

    for step in range(1, n_steps + 1):
        cfg.activation = _activation_schedule(step, n_steps)
        solver.step(state, state, dt=DT)

        pos_np = solver.core.pos.numpy()
        p = pos_np[tet_idx]
        Ds = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]], axis=-1)
        F = np.einsum("mij,mjk->mik", Ds, tet_poses)
        n_inv = int(np.sum(np.linalg.det(F) <= 0))

        joint_q = state.joint_q.numpy()
        joint_angle = float(joint_q[0]) if len(joint_q) > 0 else 0.0

        steps.append(step)
        excitations.append(float(cfg.activation))
        activations.append(float(solver._effective_activation))
        torques.append(float(solver._axis_torque))
        angles.append(np.degrees(joint_angle))
        inverted_tets.append(n_inv)

        if step % 50 == 0:
            print(f"  step {step}/{n_steps}  angle={joint_angle:.3f} rad  "
                  f"torque={solver._axis_torque:.3f}  inverted={n_inv}")

    print(f"  done: peak |torque|={max(abs(t) for t in torques):.2f} N·m, "
          f"max_angle={min(angles):.1f}°, peak_inv_tets={max(inverted_tets)}")

    return {
        "time": np.array(steps) * DT,
        "excitations": np.array(excitations),
        "activations": np.array(activations),
        "torques": np.array(torques),
        "angles": np.array(angles),
        "inverted_tets": np.array(inverted_tets),
    }


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    solver, sim, state, cfg, _ = setup_couple3()

    print("Running couple3 simulation...")
    r = collect_data(solver, sim, state, cfg, N_STEPS)

    t = r["time"]
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(t, r["angles"], "b-", linewidth=1.5)
    axes[0].set_ylabel("Joint Angle (deg)")
    axes[0].set_title("example_couple3: TETFIBERMILLARD + SVD repair")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, r["excitations"], "r--", linewidth=1.2, label="excitation (u)")
    axes[1].plot(t, r["activations"], "r-", linewidth=1.5, label="activation (a)")
    axes[1].set_ylabel("Activation")
    axes[1].set_ylim(-0.05, 1.15)
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, r["torques"], "g-", linewidth=1.5)
    axes[2].set_ylabel("Axis Torque (N·m)")
    axes[2].grid(True, alpha=0.3)

    axes[3].fill_between(t, r["inverted_tets"], alpha=0.3, color="k")
    axes[3].plot(t, r["inverted_tets"], "k-", linewidth=1.2)
    axes[3].set_ylabel("Inverted Tets")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "couple3_curves.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    main()
