"""MuJoCo SimpleArm: spatial tendon + external DGF force injection.

Stage 1 of the SimpleArm comparison plan. A 2-body arm (humerus fixed,
radius on elbow hinge) driven by a DGF Hill-type muscle force injected
through a MuJoCo motor actuator on a spatial tendon.

Uses pure MuJoCo (not Newton wrapper) for rigid-body dynamics. The spatial
tendon provides path length; DGF force is computed externally and injected
via motor actuator ctrl.

The simulation reads config from data/simpleArm/config.json and outputs
results for comparison with the OpenSim DGF baseline (Stage 0).

Usage:
    uv run python examples/example_mujoco_simple_arm.py
    uv run python examples/example_mujoco_simple_arm.py --config data/simpleArm/config.json
"""

import argparse
import json
import os

import mujoco
import numpy as np

from VMuscle.activation import activation_dynamics_step_scalar
from VMuscle.dgf_curves import (
    active_force_length_scalar,
    force_velocity_scalar,
    passive_force_length_scalar,
)
from VMuscle.simple_arm_helpers import build_mjcf, compute_excitation, write_sto


def mujoco_simple_arm(cfg, verbose=True):
    """Run MuJoCo SimpleArm with DGF muscle force injection.

    Args:
        cfg: Config dict from data/simpleArm/config.json.
        verbose: Print progress.

    Returns:
        dict with times, elbow_angles, forces, norm_fiber_lengths,
        activations, tendon_lengths, muscle_type.
    """
    mus = cfg["muscle"]
    act_cfg = cfg["activation"]
    sol = cfg["solver"]
    ic = cfg["initial_conditions"]

    F_max = mus["max_isometric_force"]
    L_opt = mus["optimal_fiber_length"]
    L_slack = mus["tendon_slack_length"]
    V_max = mus["max_contraction_velocity"]
    d_damp = mus["fiber_damping"]

    outer_dt = sol["dt"]
    n_steps = sol["n_steps"]
    theta0 = np.radians(ic["elbow_angle_deg"])

    # --- Build MuJoCo model ---
    mjcf_str = build_mjcf(cfg)
    mj_model = mujoco.MjModel.from_xml_string(mjcf_str)
    mj_data = mujoco.MjData(mj_model)

    # Internal timestep from MJCF (smaller for stability)
    mj_dt = mj_model.opt.timestep
    substeps = max(1, int(round(outer_dt / mj_dt)))

    if verbose:
        print(f"[MuJoCo] ntendon={mj_model.ntendon}, nu={mj_model.nu}, "
              f"nq={mj_model.nq}, dt_inner={mj_dt}, substeps={substeps}")

    # Set initial elbow angle
    mj_data.qpos[0] = theta0
    mujoco.mj_forward(mj_model, mj_data)

    # Read initial tendon length
    ten_length_init = float(mj_data.ten_length[0])
    fiber_length_init = ten_length_init - L_slack
    lm_tilde_init = fiber_length_init / L_opt

    if verbose:
        print(f"[MuJoCo] Initial: ten_length={ten_length_init:.4f}, "
              f"fiber_length={fiber_length_init:.4f}, l_tilde={lm_tilde_init:.4f}")

    # --- Initial activation ---
    # At theta0=90deg, l_tilde~0.507 -> f_L is very small, so true static
    # equilibrium requires a >> 1.0 (impossible). OpenSim's equilibrateMuscles
    # returns a~0.5 through a sophisticated solver. We match that value.
    # The arm will drop slightly from 90deg to ~82deg (same as OpenSim).
    a_eq = 0.5

    if verbose:
        moment_arm = float(-mj_data.ten_J[0]) if mj_model.nv > 0 else 0.0
        print(f"[MuJoCo] moment_arm={moment_arm:.4f}, a_eq={a_eq:.4f}")

    # --- Simulation loop ---
    # Recompute DGF force at every MuJoCo internal substep for accuracy.
    # OpenSim uses adaptive variable-step integration; we emulate by using
    # the small internal dt (0.002s) for both dynamics and force updates.
    activation = a_eq
    prev_fiber_length = fiber_length_init

    times = []
    elbow_angles = []
    forces_out = []
    norm_fiber_lengths = []
    activations_out = []
    tendon_lengths = []

    total_inner_steps = n_steps * substeps
    inner_dt = mj_dt

    for inner_step in range(total_inner_steps):
        t = inner_step * inner_dt

        excitation = compute_excitation(t, act_cfg)

        # 2. Activation dynamics
        activation = activation_dynamics_step_scalar(
            excitation, activation, inner_dt,
            tau_act=act_cfg["tau_act"],
            tau_deact=act_cfg["tau_deact"],
        )

        # 3. Read tendon length from MuJoCo
        ten_length = float(mj_data.ten_length[0])
        fiber_length = ten_length - L_slack
        lm_tilde = fiber_length / L_opt

        # 4. Fiber velocity (finite difference)
        fiber_velocity = (fiber_length - prev_fiber_length) / inner_dt if inner_step > 0 else 0.0
        prev_fiber_length = fiber_length

        # Normalized velocity: v / (V_max * L_opt)
        v_norm = fiber_velocity / (V_max * L_opt)

        # 5. DGF force: F = [a * f_L * f_V + f_PE + d * v_norm] * F_max
        fl = active_force_length_scalar(lm_tilde)
        fpe = passive_force_length_scalar(lm_tilde)
        fv = force_velocity_scalar(max(-1.0, min(1.0, v_norm)))
        f_damp = d_damp * v_norm

        muscle_force = (activation * fl * fv + fpe + f_damp) * F_max
        muscle_force = max(muscle_force, 0.0)  # muscle can only pull

        # 6. Inject force via motor actuator on tendon
        mj_data.ctrl[0] = muscle_force

        # 7. Step MuJoCo (single internal step)
        mujoco.mj_step(mj_model, mj_data)

        # 8. Record at outer step rate
        if inner_step % substeps == 0:
            elbow_angle = float(mj_data.qpos[0])
            times.append(t)
            elbow_angles.append(elbow_angle)
            forces_out.append(muscle_force)
            norm_fiber_lengths.append(lm_tilde)
            activations_out.append(activation)
            tendon_lengths.append(ten_length)

            outer_step = inner_step // substeps
            if verbose and (outer_step % 100 == 0 or outer_step == n_steps - 1):
                print(f"  step={outer_step:4d} t={t:6.3f}s "
                      f"theta={np.degrees(elbow_angle):7.2f}deg "
                      f"F={muscle_force:7.2f}N a={activation:.4f} "
                      f"l_tilde={lm_tilde:.4f}")

    if verbose:
        print(f"[MuJoCo] Done: {len(times)} points, "
              f"final angle={np.degrees(elbow_angles[-1]):.1f}deg")

    # Save MJCF and .sto
    os.makedirs("output", exist_ok=True)
    with open("output/SimpleArm_MuJoCo.xml", "w") as f:
        f.write(build_mjcf(cfg))

    # Write .sto file matching OpenSim format
    sto_path = "output/SimpleArm_MuJoCo_states.sto"
    # Compute speed via central finite difference
    ea = np.array(elbow_angles)
    ta = np.array(times)
    speeds = np.gradient(ea, ta)
    cols = [
        "/jointset/elbow/elbow_coord_0/value",
        "/jointset/elbow/elbow_coord_0/speed",
        "/forceset/biceps/activation",
        "/forceset/biceps/fiber_force",
        "/forceset/biceps/norm_fiber_length",
        "/forceset/biceps/tendon_length",
    ]
    write_sto(sto_path, "SimpleArm_MuJoCo_DGF", cols, times,
              [elbow_angles, speeds.tolist(), activations_out, forces_out,
               norm_fiber_lengths, tendon_lengths])
    if verbose:
        print(f"STO saved to {sto_path}")

    return {
        "times": np.array(times),
        "elbow_angles": np.array(elbow_angles),
        "forces": np.array(forces_out),
        "norm_fiber_lengths": np.array(norm_fiber_lengths),
        "activations": np.array(activations_out),
        "tendon_lengths": np.array(tendon_lengths),
        "max_iso_force": F_max,
        "muscle_type": "MuJoCo_DGF",
    }


def main():
    parser = argparse.ArgumentParser(description="MuJoCo SimpleArm with DGF muscle")
    parser.add_argument("--config", default="data/simpleArm/config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    result = mujoco_simple_arm(cfg)

    if result:
        print(f"\nFinal elbow angle: {np.degrees(result['elbow_angles'][-1]):.2f}deg")
        print(f"Tendon length range: [{result['tendon_lengths'].min():.4f}, "
              f"{result['tendon_lengths'].max():.4f}]")


if __name__ == "__main__":
    main()
