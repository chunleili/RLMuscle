"""Fully physical VBD bone-muscle coupling for SimpleArm (Stage 3).

Removes the kinematic full-mesh reset of Stage 2. VBD muscle mesh interior
vertices evolve autonomously with state continuity. Only boundary (kinematic)
vertices are updated from MuJoCo site positions each step.

Coupling strategy:
  - Origin vertices: kinematic (mass=0), fixed to world (humerus is fixed)
  - Insertion vertices: kinematic (mass=0), position updated each step from
    MuJoCo insertion site world position
  - Interior vertices: dynamic (mass>0), evolve freely under VBD elastic +
    vmuscle (sigma0>0) forces
  - Force extraction: fiber stretch from deformation gradient -> DGF curves
  - VBD interior carries state forward between steps (no per-step reset)

Key difference from Stage 2:
  Stage 2: ALL vertices reset to kinematic scale each step (no memory)
  Stage 3: Only boundary vertices updated, interior carries state (has memory)
  This allows VBD to capture history-dependent volumetric effects.

Usage:
    uv run python examples/example_vbd_coupled_simple_arm.py
    uv run python examples/example_vbd_coupled_simple_arm.py --config data/simpleArm/config.json
"""

import argparse
import json
import os
import sys

import mujoco
import numpy as np
import warp as wp

import newton
from newton.solvers import SolverVBD

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from VMuscle.activation import activation_dynamics_step_np
from VMuscle.dgf_curves import active_force_length, force_velocity, passive_force_length
from VMuscle.mesh_io import UsdTetExporter
from VMuscle.mesh_utils import (
    create_cylinder_tet_mesh,
    set_vmuscle_properties,
)


def load_config(path="data/simpleArm/config.json"):
    with open(path) as f:
        return json.load(f)


def _get_vbd_options(cfg):
    """Extract VBD coupling options aligned with SolverVBD defaults."""
    coupling = cfg.get("coupling", {})
    return {
        "quasi_static": bool(coupling.get("vbd_quasi_static", True)),
        "iterations": int(coupling.get("vbd_iterations", 20)),
        "substeps": int(coupling.get("vbd_substeps", 20)),
        "warmup_steps": int(coupling.get("vbd_warmup_steps", 100)),
    }


def _rotation_matrix_from_z_to(target_dir):
    """Compute rotation matrix that rotates Z-axis to target_dir."""
    z = np.array([0, 0, 1], dtype=np.float64)
    t = np.asarray(target_dir, dtype=np.float64)
    t = t / np.linalg.norm(t)

    cos_theta = np.dot(z, t)
    if cos_theta > 1.0 - 1e-8:
        return np.eye(3, dtype=np.float32)
    if cos_theta < -1.0 + 1e-8:
        return np.diag([1, -1, -1]).astype(np.float32)

    axis = np.cross(z, t)
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    R = np.eye(3) + np.sin(np.arccos(cos_theta)) * K + (1 - cos_theta) * (K @ K)
    return R.astype(np.float32)


def build_vbd_muscle_coupled(cfg, origin_pos, insertion_pos, fiber_length, device="cpu"):
    """Create VBD muscle in MuJoCo world frame.

    The mesh represents the muscle fiber, placed in world coordinates along
    the tendon direction from insertion_pos. Both ends are kinematic (mass=0).
    Interior vertices are dynamic and will evolve freely under VBD.

    Returns:
        model, s0, s1, ctrl, solver, meta
    """
    geo = cfg["geometry"]
    mus = cfg["muscle"]
    vbd_opts = _get_vbd_options(cfg)

    r = geo["muscle_radius"]
    n_circ = geo["n_circumferential"]
    n_axial = geo["n_axial"]

    origin_pos = np.asarray(origin_pos, dtype=np.float64)
    insertion_pos = np.asarray(insertion_pos, dtype=np.float64)
    tendon_vec = origin_pos - insertion_pos
    tendon_path_len = float(np.linalg.norm(tendon_vec))
    tendon_dir = (tendon_vec / tendon_path_len).astype(np.float32)

    mesh_length = fiber_length
    sigma0 = mus["max_isometric_force"] / (np.pi * r ** 2)

    # Create cylinder along Z, z=0 -> insertion end, z=mesh_length -> origin end
    vertices, tets = create_cylinder_tet_mesh(mesh_length, r, n_circ, n_axial)

    # Rotate+translate to world frame
    R = _rotation_matrix_from_z_to(tendon_dir)
    vertices = (R @ vertices.T).T + insertion_pos.astype(np.float32)
    fiber_origin_pos = insertion_pos + tendon_dir.astype(np.float64) * mesh_length

    # Fiber directions along tendon
    n_tets = len(tets)
    fiber_dirs = np.tile(tendon_dir, (n_tets, 1))

    builder = newton.ModelBuilder(up_axis="Y", gravity=0.0)
    tet_offset = len(builder.tet_indices)

    mesh = newton.TetMesh(
        vertices=vertices, tet_indices=tets.flatten(),
        k_mu=1000.0, k_lambda=10000.0, k_damp=1.0, density=1060.0,
    )
    builder.add_soft_mesh(
        mesh=mesh, pos=(0, 0, 0), rot=wp.quat_identity(), scale=1.0, vel=(0, 0, 0)
    )
    set_vmuscle_properties(
        builder, tet_offset, fiber_dirs, sigma0=sigma0,
        fiber_damping=mus["fiber_damping"],
    )

    # Identify boundary vertices (kinematic) and interior (dynamic)
    origin_ids = []
    insertion_ids = []
    for i in range(builder.particle_count):
        p = np.array(builder.particle_q[i], dtype=np.float64)
        dist_origin = np.linalg.norm(p - fiber_origin_pos)
        dist_insertion = np.linalg.norm(p - insertion_pos)
        if dist_origin < r * 1.5:
            builder.particle_mass[i] = 0.0  # kinematic origin
            origin_ids.append(i)
        elif dist_insertion < r * 1.5:
            builder.particle_mass[i] = 0.0  # kinematic insertion
            insertion_ids.append(i)

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

    # Rest positions for boundary vertex displacement computation
    rest_positions = s0.particle_q.numpy().copy()

    meta = {
        "origin_ids": np.array(origin_ids, dtype=int),
        "insertion_ids": np.array(insertion_ids, dtype=int),
        "tendon_dir": tendon_dir,
        "R": R,  # rotation matrix Z->world
        "mesh_length": mesh_length,
        "tet_idx": tet_idx,
        "tet_poses": tet_poses,
        "fiber_dirs_np": fib_np,
        "n_tets": model.tet_count,
        "n_particles": model.particle_count,
        "sigma0": sigma0,
        "rest_positions": rest_positions,
        "insertion_pos_init": insertion_pos.copy(),
        "fiber_origin_pos": fiber_origin_pos.copy(),
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


def _update_boundary_vertices(s0, meta, new_insertion_pos, device="cpu"):
    """Update kinematic boundary vertex positions.

    Insertion vertices are displaced to match the new MuJoCo insertion site
    position. Origin vertices stay fixed (humerus doesn't move).
    """
    pos_np = s0.particle_q.numpy().copy()
    rest = meta["rest_positions"]
    insertion_ids = meta["insertion_ids"]
    ins_init = meta["insertion_pos_init"]

    # Displacement of insertion site from initial position
    delta = np.asarray(new_insertion_pos, dtype=np.float32) - ins_init.astype(np.float32)

    # Apply displacement to all insertion vertices
    for vid in insertion_ids:
        pos_np[vid] = rest[vid] + delta

    s0.particle_q.assign(
        wp.array(pos_np.astype(np.float32), dtype=wp.vec3, device=device)
    )
    return pos_np


def vbd_coupled_simple_arm(cfg, verbose=True):
    """Run fully physical VBD + MuJoCo coupled SimpleArm simulation.

    VBD interior vertices evolve autonomously (state continuity). Only boundary
    vertices are updated from MuJoCo site positions. Muscle force is extracted
    from VBD's 3D fiber stretch via DGF curves.

    Returns:
        dict with times, elbow_angles, forces, norm_fiber_lengths, activations.
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
    device = sol.get("arch", "cuda:0" if wp.is_cuda_available() else "cpu")

    vbd_substeps = vbd_opts["substeps"]
    vbd_warmup = vbd_opts["warmup_steps"]
    vbd_sub_dt = outer_dt / vbd_substeps

    # --- Build MuJoCo ---
    _examples_dir = os.path.dirname(os.path.abspath(__file__))
    if _examples_dir not in sys.path:
        sys.path.insert(0, _examples_dir)
    from example_mujoco_simple_arm import build_mjcf

    mjcf_str = build_mjcf(cfg)
    mj_model = mujoco.MjModel.from_xml_string(mjcf_str)
    mj_data = mujoco.MjData(mj_model)
    mj_dt = mj_model.opt.timestep
    substeps = max(1, int(round(outer_dt / mj_dt)))

    mj_data.qpos[0] = theta0
    mujoco.mj_forward(mj_model, mj_data)

    # Read site positions
    origin_site_id = mj_model.site("muscle_origin").id
    insertion_site_id = mj_model.site("muscle_insertion").id
    origin_pos = mj_data.site_xpos[origin_site_id].copy()
    insertion_pos = mj_data.site_xpos[insertion_site_id].copy()

    ten_length_init = float(mj_data.ten_length[0])
    fiber_length_init = ten_length_init - L_slack

    if verbose:
        print(
            f"[Coupled] origin={origin_pos}, insertion={insertion_pos}\n"
            f"[Coupled] ten_length={ten_length_init:.4f}, "
            f"fiber_length={fiber_length_init:.4f}, "
            f"l_tilde={fiber_length_init / L_opt:.4f}"
        )

    # --- Build VBD muscle ---
    wp.init()
    vbd_model, s0, s1, ctrl, vbd_solver, meta = build_vbd_muscle_coupled(
        cfg, origin_pos, insertion_pos,
        fiber_length=fiber_length_init, device=device
    )

    n_tets = meta["n_tets"]
    tendon_dir = meta["tendon_dir"]
    tet_idx = meta["tet_idx"]
    tet_poses = meta["tet_poses"]
    fib_np = meta["fiber_dirs_np"]
    mesh_L = meta["mesh_length"]
    stretch_to_ltilde = mesh_L / L_opt

    if verbose:
        print(
            f"[Coupled] sigma0={meta['sigma0']:.0f}Pa, n_tets={n_tets}, "
            f"mesh_L={mesh_L:.4f}m, "
            f"origin={len(meta['origin_ids'])} kinematic, "
            f"insertion={len(meta['insertion_ids'])} kinematic, "
            f"interior={meta['n_particles'] - len(meta['origin_ids']) - len(meta['insertion_ids'])} dynamic, "
            f"mode={'quasi-static' if vbd_opts['quasi_static'] else 'dynamic'} "
            f"(iters={vbd_opts['iterations']}, substeps={vbd_opts['substeps']})"
        )

    # --- USD exporter ---
    anim_dir = "output/anim"
    os.makedirs(anim_dir, exist_ok=True)
    usd_path = os.path.join(anim_dir, "simple_arm_coupled.usda")
    fps = int(round(1.0 / (substeps * mj_dt)))
    exporter = UsdTetExporter(
        tet_idx, usd_path=usd_path, prim_path="/muscle", fps=fps
    )

    # --- Warm-up: let VBD interior find equilibrium ---
    initial_activation = 0.5
    act_arr = np.full(n_tets, initial_activation, dtype=np.float32)
    ctrl.tet_activations.assign(
        wp.array(act_arr, dtype=wp.float32, device=device)
    )
    if verbose:
        print(f"[Coupled] Warming up VBD ({vbd_warmup} substeps, a={initial_activation})...")
    for _ in range(vbd_warmup):
        vbd_solver.step(s0, s1, ctrl, contacts=None, dt=vbd_sub_dt)
        s0, s1 = s1, s0

    # Check warm-up result
    pos_np = s0.particle_q.numpy()
    warmup_stretches = compute_fiber_stretches(pos_np, tet_idx, tet_poses, fib_np)
    warmup_ltilde = float(warmup_stretches.mean()) * stretch_to_ltilde
    if verbose:
        print(
            f"[Coupled] Warm-up done: mean_stretch={warmup_stretches.mean():.4f}, "
            f"l_tilde={warmup_ltilde:.4f}, NaN={np.any(np.isnan(pos_np))}"
        )

    # --- Simulation ---
    activation = initial_activation
    prev_fiber_length = fiber_length_init

    times = []
    elbow_angles = []
    forces_out = []
    norm_fiber_lengths = []
    activations_out = []

    physics_time = 0.0

    if verbose:
        print(f"[Coupled] Simulating {n_steps * outer_dt:.1f}s, F_max={F_max:.0f}N")

    for step in range(n_steps):
        t = physics_time

        # 0. Record state
        elbow_angle = float(mj_data.qpos[0])
        times.append(t)
        elbow_angles.append(elbow_angle)
        activations_out.append(activation)

        # 1. Update insertion boundary vertices from MuJoCo
        mujoco.mj_forward(mj_model, mj_data)
        new_insertion = mj_data.site_xpos[insertion_site_id].copy()
        _update_boundary_vertices(s0, meta, new_insertion, device=device)

        # 2. Set activation for VBD vmuscle kernel
        act_arr = np.full(n_tets, activation, dtype=np.float32)
        ctrl.tet_activations.assign(
            wp.array(act_arr, dtype=wp.float32, device=device)
        )

        # 3. VBD step (state continuity for interior vertices)
        for _ in range(vbd_substeps):
            vbd_solver.step(s0, s1, ctrl, contacts=None, dt=vbd_sub_dt)
            s0, s1 = s1, s0

        # 4. Extract fiber stretch from VBD deformation gradient
        vbd_pos = s0.particle_q.numpy()
        vbd_valid = not np.any(np.isnan(vbd_pos))
        if vbd_valid:
            stretches = compute_fiber_stretches(vbd_pos, tet_idx, tet_poses, fib_np)
            l_tilde_vbd = float(stretches.mean()) * stretch_to_ltilde
            # Sanity check
            if l_tilde_vbd < 0.1 or l_tilde_vbd > 3.0:
                vbd_valid = False

        # Fallback: use MuJoCo 1D stretch
        ten_length = float(mj_data.ten_length[0])
        fiber_length = ten_length - L_slack
        mj_ltilde = fiber_length / L_opt

        if not vbd_valid:
            l_tilde_vbd = mj_ltilde

        # Save USD frame
        exporter.save_frame(vbd_pos.astype(np.float32), step)

        # 5. MuJoCo substeps with VBD-derived force
        for sub in range(substeps):
            t_sub = t + sub * mj_dt

            # Excitation schedule
            t_start = act_cfg["excitation_start_time"]
            t_end_exc = act_cfg["excitation_end_time"]
            e_off = act_cfg["excitation_off"]
            e_on = act_cfg["excitation_on"]
            if t_sub < t_start:
                excitation = e_off
            elif t_sub >= t_end_exc:
                excitation = e_on
            else:
                frac = (t_sub - t_start) / (t_end_exc - t_start)
                frac = frac * frac * (3.0 - 2.0 * frac)
                excitation = e_off + (e_on - e_off) * frac

            # Activation dynamics at substep rate
            activation = float(activation_dynamics_step_np(
                np.array([excitation], dtype=np.float32),
                np.array([activation], dtype=np.float32),
                mj_dt,
                tau_act=act_cfg["tau_act"],
                tau_deact=act_cfg["tau_deact"],
            )[0])

            # Real-time fiber length and velocity from MuJoCo
            ten_len = float(mj_data.ten_length[0])
            fib_len = ten_len - L_slack
            fib_vel = (fib_len - prev_fiber_length) / mj_dt if (step > 0 or sub > 0) else 0.0
            prev_fiber_length = fib_len
            v_norm = fib_vel / (V_max * L_opt)

            # VBD-corrected l_tilde for position-dependent terms
            # Real-time 1D l_tilde + VBD volumetric delta
            l_tilde_1d = fib_len / L_opt
            delta_ltilde = l_tilde_vbd - (fiber_length / L_opt)  # VBD correction
            l_tilde_now = l_tilde_1d + delta_ltilde

            # DGF force
            fl = float(active_force_length(l_tilde_now))
            fpe = float(passive_force_length(l_tilde_now))
            fv = float(force_velocity(np.clip(v_norm, -1.0, 1.0)))
            muscle_force = (activation * fl * fv + fpe + d_damp * v_norm) * F_max
            muscle_force = np.clip(muscle_force, 0.0, F_max * 2.0)

            mj_data.ctrl[0] = muscle_force
            mujoco.mj_step(mj_model, mj_data)
            physics_time += mj_dt

        # Record
        forces_out.append(float(muscle_force))
        norm_fiber_lengths.append(l_tilde_vbd)

        if verbose and (step % 100 == 0 or step == n_steps - 1):
            print(
                f"  step={step:4d} t={t:6.3f}s "
                f"theta={np.degrees(elbow_angle):7.2f}deg "
                f"F={muscle_force:7.2f}N a={activation:.4f} "
                f"l_vbd={l_tilde_vbd:.4f} l_1d={mj_ltilde:.4f} "
                f"delta={delta_ltilde:.6f}"
            )

    if verbose:
        print(
            f"[Coupled] Done: {len(times)} points, "
            f"final angle={np.degrees(elbow_angles[-1]):.1f}deg"
        )

    # --- Finalize USD ---
    exporter.finalize()
    if verbose:
        print(f"USD animation saved to {usd_path}")

    # --- Save .sto ---
    os.makedirs("output", exist_ok=True)
    sto_path = "output/SimpleArm_Coupled_states.sto"
    n_rows = len(times)
    cols = [
        "/jointset/elbow/elbow_coord_0/value",
        "/forceset/biceps/activation",
        "/forceset/biceps/fiber_force",
        "/forceset/biceps/norm_fiber_length",
    ]
    with open(sto_path, "w") as f:
        f.write("SimpleArm_VBD_Coupled\n")
        f.write("inDegrees=no\n")
        f.write(f"nColumns={len(cols) + 1}\n")
        f.write(f"nRows={n_rows}\n")
        f.write("DataType=double\n")
        f.write("version=3\n")
        f.write("endheader\n")
        f.write("time\t" + "\t".join(cols) + "\n")
        for i in range(n_rows):
            f.write(
                f"{times[i]}\t{elbow_angles[i]}\t{activations_out[i]}\t"
                f"{forces_out[i]}\t{norm_fiber_lengths[i]}\n"
            )
    if verbose:
        print(f"STO saved to {sto_path}")

    return {
        "times": np.array(times),
        "elbow_angles": np.array(elbow_angles),
        "forces": np.array(forces_out),
        "norm_fiber_lengths": np.array(norm_fiber_lengths),
        "activations": np.array(activations_out),
        "max_iso_force": F_max,
        "muscle_type": "VBD_Coupled",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fully physical VBD + MuJoCo coupled SimpleArm (Stage 3)"
    )
    parser.add_argument("--config", default="data/simpleArm/config.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = vbd_coupled_simple_arm(cfg)
    if result:
        print(f"\nFinal elbow angle: {np.degrees(result['elbow_angles'][-1]):.2f}deg")


if __name__ == "__main__":
    main()
