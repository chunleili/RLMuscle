"""VBD volume muscle + MuJoCo skeleton for SimpleArm.

Stage 2 of the SimpleArm comparison plan. A cylinder tet mesh (Newton VBD)
represents the bicep muscle volume with active contraction (sigma0 > 0).
MuJoCo handles rigid-body dynamics.

Coupling strategy (reference: sliding ball + SolverMuscleBoneCoupled):
  - VBD vmuscle with sigma0 > 0: the DGF Hill-type force-length and passive
    force curves are evaluated inside the VBD kernel
    (accumulate_fiber_force_and_hessian). VBD solves the coupled 3D elastic
    + muscle equilibrium.
  - Origin (top) and insertion (bottom) vertices are kinematic, positioned
    from MuJoCo tendon geometry each step.
  - After VBD solve, per-tet fiber stretches are extracted from the 3D
    deformation gradient. DGF force is computed from these volumetric
    stretches.
  - For straight cylinder with uniform fibers, VBD stretch = 1D stretch
    (physics-correct). VBD value emerges with complex geometry (wrapping,
    variable cross-section, non-uniform fiber).

Usage:
    uv run python examples/example_vbd_mujoco_simple_arm.py
    uv run python examples/example_vbd_mujoco_simple_arm.py --config data/simpleArm/config.json
"""

import argparse
import json
import os

import mujoco
import numpy as np
import warp as wp

from VMuscle.log import setup_logging
setup_logging()

import newton
from newton.solvers import SolverVBD

from VMuscle.activation import activation_dynamics_step_scalar
from VMuscle.dgf_curves import active_force_length, force_velocity, passive_force_length
from VMuscle.mesh_io import MeshExporter
from VMuscle.mesh_utils import (
    assign_fiber_directions,
    check_mesh_quality,
    create_cylinder_tet_mesh,
    set_vmuscle_properties,
)
from VMuscle.simple_arm_helpers import build_mjcf, compute_excitation, write_sto


def _get_default_warp_device(cfg=None):
    """Prefer CUDA for VBD because CPU execution is significantly slower here."""
    wp.init()
    if cfg is not None:
        return cfg.get("solver", {}).get("arch", "cuda:0" if wp.is_cuda_available() else "cpu")
    return "cuda:0" if wp.is_cuda_available() else "cpu"


def _get_vbd_options(cfg):
    """Extract VBD-specific options aligned with SolverVBD defaults."""
    coupling = cfg.get("coupling", {})
    quasi_static = bool(coupling.get("vbd_quasi_static", True))
    return {
        "quasi_static": quasi_static,
        "iterations": int(coupling.get("vbd_iterations", 20)),
        "substeps": int(coupling.get("vbd_substeps", 20)),
        "scale_guard_min": float(coupling.get("vbd_scale_guard_min", 0.35 if quasi_static else 0.5)),
        "scale_guard_max": float(coupling.get("vbd_scale_guard_max", 1.6 if quasi_static else 1.2)),
    }


def build_vbd_muscle(cfg, mesh_length, device=None):
    """Create VBD muscle with active contraction (sigma0 > 0).

    sigma0 > 0 enables the vmuscle kernel inside VBD: the DGF force-length
    and passive force curves are evaluated per-tet during the VBD solve.
    The kernel accumulates fiber force and Hessian alongside Neo-Hookean
    elasticity, solving the coupled 3D equilibrium.

    Both ends are kinematic (displacement-controlled from MuJoCo). VBD
    solves for the internal vertex equilibrium under combined elastic +
    muscle forces.
    """
    if device is None:
        device = _get_default_warp_device(cfg)

    geo = cfg["geometry"]
    mus = cfg["muscle"]
    vbd_opts = _get_vbd_options(cfg)

    r = geo["muscle_radius"]
    n_circ = geo["n_circumferential"]
    n_axial = geo["n_axial"]

    # sigma0 = F_max / cross-section area [Pa]
    sigma0 = mus["max_isometric_force"] / (np.pi * r ** 2)

    vertices, tets = create_cylinder_tet_mesh(mesh_length, r, n_circ, n_axial)
    fiber_dirs = assign_fiber_directions(vertices, tets, axis=2)

    builder = newton.ModelBuilder(up_axis="Z", gravity=0.0)
    tet_offset = len(builder.tet_indices)

    # Moderate elastic stiffness: enough to resist radial bulging from muscle
    # contraction, but not so high that VBD corrections are suppressed.
    builder.add_soft_mesh(
        pos=(0, 0, 0), rot=wp.quat_identity(), scale=1.0, vel=(0, 0, 0),
        vertices=vertices.tolist(), indices=tets.flatten().tolist(),
        k_mu=1000.0, k_lambda=10000.0, k_damp=1.0, density=1060.0,
    )
    # sigma0 > 0: VBD vmuscle kernel computes quasi-static DGF forces internally
    set_vmuscle_properties(
        builder, tet_offset, fiber_dirs, sigma0=sigma0,
        fiber_damping=mus["fiber_damping"],
    )

    top_ids = []
    bottom_ids = []
    for i in range(builder.particle_count):
        z = builder.particle_q[i][2]
        if z > mesh_length - 1e-4:
            builder.particle_mass[i] = 0.0  # kinematic (origin)
            top_ids.append(i)
        elif z < 1e-4:
            builder.particle_mass[i] = 0.0  # kinematic (insertion)
            bottom_ids.append(i)

    builder.color()
    model = builder.finalize(device=device)

    s0 = model.state()
    s1 = model.state()
    ctrl = model.control()
    solver = SolverVBD(
        model,
        iterations=vbd_opts["iterations"],
        vmuscle_quasi_static=vbd_opts["quasi_static"],
    )

    tet_idx = np.array(builder.tet_indices, dtype=int).reshape(-1, 4)
    tet_poses = np.array(builder.tet_poses).reshape(-1, 3, 3)
    fib_np = model.vmuscle.fiber_dirs.numpy()

    meta = {
        "top_ids": np.array(top_ids, dtype=int),
        "bottom_ids": np.array(bottom_ids, dtype=int),
        "mesh_length": mesh_length,
        "tet_idx": tet_idx,
        "tet_poses": tet_poses,
        "fiber_dirs_np": fib_np,
        "n_tets": model.tet_count,
        "n_particles": model.particle_count,
        "sigma0": sigma0,
        "vbd_options": vbd_opts,
    }
    return model, s0, s1, ctrl, solver, meta


def compute_fiber_stretches(pos, tet_idx, tet_poses, fiber_dirs):
    """Compute per-tet fiber stretch via deformation gradient F = Ds @ Dm_inv."""
    n = len(tet_idx)
    stretches = np.empty(n)
    for e in range(n):
        i, j, k, l = tet_idx[e]
        Ds = np.column_stack([pos[j] - pos[i], pos[k] - pos[i], pos[l] - pos[i]])
        Fd = (Ds @ tet_poses[e]) @ fiber_dirs[e]
        stretches[e] = max(np.linalg.norm(Fd), 1e-8)
    return stretches


def vbd_mujoco_simple_arm(cfg, verbose=True, device=None):
    """Run VBD + MuJoCo SimpleArm simulation.

    Two-rate coupling:
      Slow (per outer step, 0.0167s):
        - Reset VBD mesh to MuJoCo fiber length (displacement-controlled)
        - Set tet_activations for VBD vmuscle kernel
        - VBD step: sigma0 > 0 drives DGF-based active contraction inside
          the solver, solving 3D elastic + muscle equilibrium
        - Extract per-tet fiber stretch from VBD deformation gradient
        - Compute DGF force from VBD's volumetric stretches (averaged)
      Fast (per MuJoCo substep, 0.002s):
        - VBD-derived fl/fpe (position-dependent) + real-time fv (velocity)
        - MuJoCo step
    """
    mus = cfg["muscle"]
    act_cfg = cfg["activation"]
    sol = cfg["solver"]
    ic = cfg["initial_conditions"]
    vbd_opts = _get_vbd_options(cfg)

    F_max = mus["max_isometric_force"]
    L_opt = mus["optimal_fiber_length"]
    L_slack = mus["tendon_slack_length"]
    V_max = mus["max_contraction_velocity"]
    d_damp = mus["fiber_damping"]

    outer_dt = sol["dt"]
    n_steps = sol["n_steps"]
    theta0 = np.radians(ic["elbow_angle_deg"])
    if device is None:
        device = _get_default_warp_device(cfg)
    else:
        wp.init()

    # --- Build MuJoCo ---
    mjcf_str = build_mjcf(cfg)
    mj_model = mujoco.MjModel.from_xml_string(mjcf_str)
    mj_data = mujoco.MjData(mj_model)
    mj_dt = mj_model.opt.timestep
    substeps = max(1, int(round(outer_dt / mj_dt)))

    mj_data.qpos[0] = theta0
    mujoco.mj_forward(mj_model, mj_data)

    ten_length_init = float(mj_data.ten_length[0])
    fiber_length_init = ten_length_init - L_slack

    if verbose:
        print(
            f"[VBD+MuJoCo] Initial: ten_length={ten_length_init:.4f}, "
            f"fiber_length={fiber_length_init:.4f}, "
            f"l_tilde={fiber_length_init / L_opt:.4f}"
        )

    # --- Build VBD muscle ---
    vbd_model, s0, s1, ctrl, vbd_solver, meta = build_vbd_muscle(
        cfg, mesh_length=fiber_length_init, device=device
    )

    top_ids = meta["top_ids"]
    bottom_ids = meta["bottom_ids"]
    mesh_L = meta["mesh_length"]
    tet_idx = meta["tet_idx"]
    tet_poses = meta["tet_poses"]
    fib_np = meta["fiber_dirs_np"]
    n_tets = meta["n_tets"]
    sigma0 = meta["sigma0"]
    # Rest vertices for position reset each step
    rest_verts = s0.particle_q.numpy().copy()

    # Conversion: VBD stretch (relative to rest mesh) -> l_tilde (relative to L_opt)
    stretch_to_ltilde = mesh_L / L_opt

    if verbose:
        print(
            f"[VBD+MuJoCo] sigma0={sigma0:.0f}Pa, n_tets={n_tets}, "
            f"mesh_L={mesh_L:.4f}m, "
            f"top={len(top_ids)} bot={len(bottom_ids)} kinematic, "
            f"device={device}, "
            f"mode={'quasi-static' if vbd_opts['quasi_static'] else 'dynamic'} "
            f"(iters={vbd_opts['iterations']}, substeps={vbd_opts['substeps']})"
        )

    # --- Mesh exporter ---
    fps = int(round(1.0 / (substeps * mj_dt)))
    exporter = MeshExporter(
        path="output/anim",
        format="ply",
        tet_indices=tet_idx,
        positions=rest_verts,
    )

    # --- Simulation ---
    activation = 0.5
    prev_fiber_length = fiber_length_init

    times = []
    elbow_angles = []
    forces_out = []
    norm_fiber_lengths = []
    activations_out = []

    t_end = n_steps * outer_dt

    if verbose:
        print(f"[VBD+MuJoCo] Simulating {t_end:.1f}s, F_max={F_max:.0f}N")

    # Track actual physics time (substeps * mj_dt per outer step, not config dt)
    physics_time = 0.0

    for step in range(n_steps):
        t = physics_time

        # 0. Record state at start of outer step (before VBD + substeps)
        elbow_angle = float(mj_data.qpos[0])
        times.append(t)
        elbow_angles.append(elbow_angle)
        activations_out.append(activation)

        # 1. VBD step: reset mesh, set activation, solve 3D equilibrium
        ten_length = float(mj_data.ten_length[0])
        fiber_length = ten_length - L_slack
        mj_ltilde = fiber_length / L_opt

        # Scale all vertices from rest to match MuJoCo fiber length
        # Top (z=mesh_L) fixed, bottom (z=0) moves to fiber_length position
        scale = fiber_length / mesh_L

        # VBD with sigma0 > 0 is stable only inside a guarded stretch range.
        # Quasi-static mode widens the usable range by removing inertial
        # history from the VBD particle solve.
        run_vbd = vbd_opts["scale_guard_min"] < scale < vbd_opts["scale_guard_max"]

        pos_np = rest_verts.copy()
        pos_np[:, 2] = mesh_L + (rest_verts[:, 2] - mesh_L) * scale

        if run_vbd:
            s0.particle_q.assign(
                wp.array(pos_np.astype(np.float32), dtype=wp.vec3, device=device)
            )
            s0.particle_qd.zero_()

            # Set activation for VBD vmuscle kernel
            act_arr = np.full(n_tets, activation, dtype=np.float32)
            ctrl.tet_activations.assign(
                wp.array(act_arr, dtype=wp.float32, device=device)
            )

            # VBD substeps: smaller dt improves convergence
            vbd_substeps = vbd_opts["substeps"]
            vbd_sub_dt = outer_dt / vbd_substeps
            for _ in range(vbd_substeps):
                vbd_solver.step(s0, s1, ctrl, contacts=None, dt=vbd_sub_dt)
                s0, s1 = s1, s0

        # Save mesh frame (VBD or kinematic fallback)
        vbd_pos = s0.particle_q.numpy() if run_vbd else pos_np
        exporter.save_frame(vbd_pos.astype(np.float32), step)

        # Mesh quality check
        if run_vbd:
            check_mesh_quality(vbd_pos, tet_idx, tet_poses, step=step)

        # Extract per-tet fiber stretch from VBD's 3D deformation gradient
        vbd_valid = not np.any(np.isnan(vbd_pos))
        if vbd_valid:
            stretches = compute_fiber_stretches(
                vbd_pos, tet_idx, tet_poses, fib_np
            )
            ltilde_per_tet = stretches * stretch_to_ltilde
            l_tilde_vbd = float(ltilde_per_tet.mean())
            # Sanity check: VBD stretch should be close to 1D stretch.
            # Transient radial instability (sigma0 vs elastic) can produce
            # extreme stretches in internal vertices — filter these out.
            if abs(l_tilde_vbd - mj_ltilde) > 0.5 or l_tilde_vbd < 0.1 or l_tilde_vbd > 3.0:
                vbd_valid = False

        if vbd_valid:
            # VBD volumetric correction: difference between 3D and 1D stretch
            delta_ltilde = l_tilde_vbd - mj_ltilde
        else:
            delta_ltilde = 0.0
            l_tilde_vbd = mj_ltilde

        # 2. MuJoCo substeps with two-rate coupling:
        #    - Activation dynamics: updated every substep (tau_act=15ms < outer_dt)
        #    - Position-dependent force (fl, fpe): recomputed each substep
        #      from real-time 1D l_tilde + VBD volumetric correction delta
        #    - Velocity-dependent force (fv, damping): real-time from MuJoCo
        for sub in range(substeps):
            t_sub = t + sub * mj_dt

            excitation = compute_excitation(t_sub, act_cfg)

            # Activation dynamics at substep rate (matching Stage 1)
            activation = activation_dynamics_step_scalar(
                excitation, activation, mj_dt,
                tau_act=act_cfg["tau_act"],
                tau_deact=act_cfg["tau_deact"],
            )

            ten_len = float(mj_data.ten_length[0])
            fib_len = ten_len - L_slack

            # VBD-corrected l_tilde: 1D kinematic + 3D volumetric delta
            l_tilde_now = fib_len / L_opt + delta_ltilde

            # Real-time fiber velocity from MuJoCo
            fib_vel = (fib_len - prev_fiber_length) / mj_dt if (step > 0 or sub > 0) else 0.0
            prev_fiber_length = fib_len
            v_norm = fib_vel / (V_max * L_opt)

            # DGF force from VBD-corrected stretch + real-time velocity
            fl = float(active_force_length(l_tilde_now))
            fpe = float(passive_force_length(l_tilde_now))
            fv = float(force_velocity(np.clip(v_norm, -1.0, 1.0)))
            muscle_force = (activation * fl * fv + fpe + d_damp * v_norm) * F_max
            muscle_force = np.clip(muscle_force, 0.0, F_max * 2.0)

            mj_data.ctrl[0] = muscle_force
            mujoco.mj_step(mj_model, mj_data)
            physics_time += mj_dt

        # 3. Update record arrays (force, l_tilde from this step)
        forces_out.append(float(muscle_force))
        norm_fiber_lengths.append(l_tilde_vbd)

        if verbose and (step % 100 == 0 or step == n_steps - 1):
            final_ltilde = (float(mj_data.ten_length[0]) - L_slack) / L_opt
            print(
                f"  step={step:4d} t={t:6.3f}s "
                f"theta={np.degrees(elbow_angle):7.2f}deg "
                f"F={muscle_force:7.2f}N a={activation:.4f} "
                f"l_mj={final_ltilde:.4f} l_vbd={l_tilde_vbd:.4f} "
                f"delta={delta_ltilde:.6f}"
            )

    if verbose:
        print(
            f"[VBD+MuJoCo] Done: {len(times)} points, "
            f"final angle={np.degrees(elbow_angles[-1]):.1f}deg"
        )

    # --- Finalize mesh export ---
    exporter.finalize()

    # --- Save outputs ---
    os.makedirs("output", exist_ok=True)
    sto_path = "output/SimpleArm_VBD_MuJoCo_states.sto"
    cols = [
        "/jointset/elbow/elbow_coord_0/value",
        "/forceset/biceps/activation",
        "/forceset/biceps/fiber_force",
        "/forceset/biceps/norm_fiber_length",
    ]
    write_sto(sto_path, "SimpleArm_VBD_MuJoCo", cols, times,
              [elbow_angles, activations_out, forces_out, norm_fiber_lengths])
    if verbose:
        print(f"STO saved to {sto_path}")

    return {
        "times": np.array(times),
        "elbow_angles": np.array(elbow_angles),
        "forces": np.array(forces_out),
        "norm_fiber_lengths": np.array(norm_fiber_lengths),
        "activations": np.array(activations_out),
        "max_iso_force": F_max,
        "muscle_type": "VBD_MuJoCo",
        "vbd_mode": "quasi-static" if vbd_opts["quasi_static"] else "dynamic",
    }


def main():
    parser = argparse.ArgumentParser(
        description="VBD volume muscle + MuJoCo skeleton SimpleArm (Stage 2)"
    )
    parser.add_argument("--config", default="data/simpleArm/config.json")
    parser.add_argument(
        "--device",
        default=None,
        help="Warp device override (default: cuda:0 if available, else cpu).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    result = vbd_mujoco_simple_arm(cfg, device=args.device)
    if result:
        print(f"\nFinal elbow angle: {np.degrees(result['elbow_angles'][-1]):.2f}deg")


if __name__ == "__main__":
    main()
