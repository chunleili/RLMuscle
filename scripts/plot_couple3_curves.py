"""Run couple3 simulation and plot joint angle / activation / torque vs time.

Usage:
    uv run python scripts/plot_couple3_curves.py
"""

import sys
import os
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


def run_experiment(config_path, n_steps=300, label=""):
    """Run one simulation and return per-step data."""
    cfg = load_config(config_path)
    cfg.gui = False
    cfg.render_mode = None

    usd_source = "data/muscle/model/bicep.usd"
    cfg.geo_path = usd_source
    cfg.bone_geo_path = usd_source
    cfg.muscle_prim_path = "/character/muscle/bicep"
    cfg.bone_prim_paths = {
        "L_scapula": "/character/bone/L_scapula/L_scapulaShape",
        "L_radius": "/character/bone/L_radius/L_radiusShape",
        "L_humerus": "/character/bone/L_humerus/L_humerusShape",
    }
    if cfg.constraints:
        for c in cfg.constraints:
            if "target_path" in c:
                c["target_path"] = usd_source

    sim = MuscleSim(cfg)
    dt = 1.0 / 60.0
    joint_friction = float(getattr(cfg, "joint_friction", 0.05))
    model, state, radius_link, joint, selected_indices = build_elbow_model(
        sim, joint_friction=joint_friction)

    control_config = build_coupling_config("smooth_nonlinear")
    solver = SolverMuscleBoneCoupled(model, sim, control_config=control_config)
    if radius_link is not None and selected_indices.size > 0:
        solver.configure_coupling(
            bone_body_id=radius_link,
            bone_rest_verts=sim.bone_pos[selected_indices].astype(np.float32),
            bone_vertex_indices=selected_indices,
            joint_index=joint,
            joint_pivot=ELBOW_PIVOT,
            joint_axis=ELBOW_AXIS,
        )

    tet_idx = sim.tet_np[:, [3, 0, 1, 2]]
    tet_poses = sim.rest_matrix.numpy()

    steps = []
    excitations = []
    activations = []
    torques = []
    angles = []
    inverted_tets = []

    for step in range(1, n_steps + 1):
        cfg.activation = _activation_schedule(step, n_steps)
        solver.step(state, state, dt=dt)

        pos_np = solver.core.pos.numpy()
        # Vectorized det(F) computation
        p = pos_np[tet_idx]  # (M, 4, 3)
        Ds = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]], axis=-1)  # (M, 3, 3)
        F = np.einsum("mij,mjk->mik", Ds, tet_poses)
        det_F = np.linalg.det(F)
        n_inv = int(np.sum(det_F <= 0))

        joint_q = state.joint_q.numpy()
        joint_angle = float(joint_q[0]) if len(joint_q) > 0 else 0.0

        steps.append(step)
        excitations.append(float(cfg.activation))
        activations.append(float(solver._effective_activation))
        torques.append(float(solver._axis_torque))
        angles.append(np.degrees(joint_angle))
        inverted_tets.append(n_inv)

        if step % 50 == 0:
            print(f"  [{label}] step {step}/{n_steps}  "
                  f"angle={joint_angle:.3f} rad  "
                  f"torque={solver._axis_torque:.3f}  "
                  f"inverted={n_inv}")

    print(f"  [{label}] done: peak |torque|={max(abs(t) for t in torques):.2f} N·m, "
          f"max_angle={min(angles):.1f}°, peak_inv_tets={max(inverted_tets)}")

    return {
        "label": label,
        "steps": np.array(steps),
        "time": np.array(steps) * dt,
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

    out_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(out_dir, exist_ok=True)

    config = os.path.join(PROJECT_ROOT,
                          "data/muscle/config/bicep_fibermillard_coupled.json")
    print("Running couple3 (post_smooth_iters=3)...")
    r = run_experiment(config, n_steps=300, label="post_smooth=3")

    t = r["time"]

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    # 1) Joint Angle
    ax = axes[0]
    ax.plot(t, r["angles"], "b-", linewidth=1.5)
    ax.set_ylabel("Joint Angle (deg)")
    ax.set_title("example_couple3: TETFIBERMILLARD + post_smooth_iters=3")
    ax.grid(True, alpha=0.3)

    # 2) Activation
    ax = axes[1]
    ax.plot(t, r["excitations"], "r--", linewidth=1.2, label="excitation (u)")
    ax.plot(t, r["activations"], "r-", linewidth=1.5, label="activation (a)")
    ax.set_ylabel("Activation")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3) Torque
    ax = axes[2]
    ax.plot(t, r["torques"], "g-", linewidth=1.5)
    ax.set_ylabel("Axis Torque (N·m)")
    ax.grid(True, alpha=0.3)

    # 4) Inverted Tets
    ax = axes[3]
    ax.fill_between(t, r["inverted_tets"], alpha=0.3, color="k")
    ax.plot(t, r["inverted_tets"], "k-", linewidth=1.2)
    ax.set_ylabel("Inverted Tets")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "couple3_curves.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    main()
