"""Run couple3 simulation and generate comparison curve plots."""

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
from VMuscle.mesh_utils import MeshDistortionError, check_mesh_quality
from VMuscle.controllability import build_coupling_config, config_to_dict
from VMuscle.muscle_warp import MuscleSim
from VMuscle.solver_muscle_bone_coupled_warp import SolverMuscleBoneCoupled

# Import from example_couple3
sys.path.insert(0, os.path.join(PROJECT_ROOT, "examples"))
from example_couple3 import (
    build_elbow_model, ELBOW_PIVOT, ELBOW_AXIS, _activation_schedule,
)


def run_experiment(config_path, k_coupling, max_torque, n_steps=300, label=""):
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
    model, state, radius_link, joint, selected_indices = build_elbow_model(sim)

    control_config = build_coupling_config(
        "smooth_nonlinear",
        k_coupling=k_coupling,
        max_torque=max_torque,
    )

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

    # Mesh quality data
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
        # Compute det(F) directly to count inverted tets without exceptions
        n_tet = len(tet_idx)
        n_inv = 0
        for e in range(n_tet):
            i, j, k, l = tet_idx[e]
            Ds = np.column_stack([
                pos_np[j] - pos_np[i],
                pos_np[k] - pos_np[i],
                pos_np[l] - pos_np[i],
            ])
            d = np.linalg.det(Ds @ tet_poses[e])
            if d <= 0.0:
                n_inv += 1

        joint_q = state.joint_q.numpy()
        joint_angle = float(joint_q[0]) if len(joint_q) > 0 else 0.0

        steps.append(step)
        excitations.append(float(cfg.activation))
        activations.append(float(solver._effective_activation))
        torques.append(float(solver._axis_torque))
        angles.append(np.degrees(joint_angle))
        inverted_tets.append(n_inv)

    print(f"  [{label}] done: max_torque={max(abs(t) for t in torques):.2f} N·m, "
          f"max_angle={min(angles):.1f}°, max_inv_tets={max(inverted_tets)}")

    return {
        "label": label,
        "steps": np.array(steps),
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

    n_steps = 300
    out_dir = os.path.join(PROJECT_ROOT, "docs", "imgs", "couple3")
    os.makedirs(out_dir, exist_ok=True)

    experiments = [
        {
            "config": "data/muscle/config/bicep_xpbd_millard.json",
            "k_coupling": None, "max_torque": None,
            "label": "Explicit (max_accel=20)",
        },
        {
            "config": "data/muscle/config/bicep_fibermillard_coupled.json",
            "k_coupling": None, "max_torque": None,
            "label": "TETFIBERMILLARD (default coupling)",
        },
        {
            "config": "data/muscle/config/bicep_fibermillard_coupled.json",
            "k_coupling": 100000.0, "max_torque": 20.0,
            "label": "TETFIBERMILLARD (k=100k, τ_max=20)",
        },
    ]

    results = []
    for exp in experiments:
        print(f"Running: {exp['label']}...")
        r = run_experiment(
            os.path.join(PROJECT_ROOT, exp["config"]),
            k_coupling=exp["k_coupling"],
            max_torque=exp["max_torque"],
            n_steps=n_steps,
            label=exp["label"],
        )
        results.append(r)

    time = results[0]["steps"] / 60.0  # seconds

    # --- Plot 1: Joint Angle ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    ax = axes[0]
    for r in results:
        ax.plot(time, r["angles"], label=r["label"], linewidth=1.5)
    ax.set_ylabel("Joint Angle (°)")
    ax.set_title("couple3: TETFIBERMILLARD vs Explicit Force — Bicep Flexion")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.5)

    # --- Plot 2: Axis Torque ---
    ax = axes[1]
    for r in results:
        ax.plot(time, np.abs(r["torques"]), label=r["label"], linewidth=1.5)
    ax.set_ylabel("|Axis Torque| (N·m)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Plot 3: Inverted Tets ---
    ax = axes[2]
    for r in results:
        ax.plot(time, r["inverted_tets"], label=r["label"], linewidth=1.5)
    ax.set_ylabel("Inverted Tets")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add activation overlay on top axes
    ax_act = axes[0].twinx()
    ax_act.fill_between(time, results[0]["excitations"], alpha=0.1, color="gray")
    ax_act.set_ylabel("Excitation", color="gray", fontsize=9)
    ax_act.set_ylim(0, 1.5)
    ax_act.tick_params(axis='y', labelcolor='gray')

    plt.tight_layout()
    path = os.path.join(out_dir, "approach_comparison.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()

    # --- Plot 4: Best config detail ---
    best = results[-1]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax = axes[0]
    ax.plot(time, best["excitations"], "--", color="gray", label="Excitation", linewidth=1)
    ax.plot(time, best["activations"], color="tab:orange", label="Activation", linewidth=1.5)
    ax.plot(time, best["angles"] / min(best["angles"]) if min(best["angles"]) != 0 else best["angles"],
            color="tab:blue", label="Angle (normalized)", linewidth=1.5)
    ax.set_ylabel("Normalized")
    ax.set_title("TETFIBERMILLARD (k_coupling=100k): Activation → Angle Response")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(time, best["angles"], color="tab:blue", label="Joint Angle (°)", linewidth=1.5)
    ax2 = ax.twinx()
    ax2.plot(time, np.abs(best["torques"]), color="tab:red", label="|Torque| (N·m)", linewidth=1.5)
    ax.set_ylabel("Joint Angle (°)", color="tab:blue")
    ax2.set_ylabel("|Torque| (N·m)", color="tab:red")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "best_config_detail.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    main()
