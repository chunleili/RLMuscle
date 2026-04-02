"""XPBD coupled SimpleArm example (Stage 3x).

Builds a MuscleSim programmatically from a cylinder mesh with
TETVOLUME + TETARAP + ATTACH constraints, coupled to MuJoCo rigid-body sim.

Usage:
    RUN=example_xpbd_coupled_simple_arm uv run main.py
    uv run python examples/example_xpbd_coupled_simple_arm.py
    uv run python examples/example_xpbd_coupled_simple_arm.py --config data/simpleArm/config.json
"""

import argparse
import json
import os

import mujoco
import numpy as np
import warp as wp

from VMuscle.activation import activation_dynamics_step_np
from VMuscle.dgf_curves import active_force_length, force_velocity, passive_force_length
from VMuscle.mesh_io import MeshExporter, save_ply
from VMuscle.mesh_utils import (
    create_capsule_mesh,
    create_cylinder_tet_mesh,
    MeshDistortionError,
    rotation_matrix_align,
    transform_by_quat,
)
from VMuscle.muscle_common import compute_fiber_stretches
from VMuscle.muscle_warp import MuscleSim
from VMuscle.simple_arm_helpers import (
    build_attach_constraints,
    build_mjcf,
    compute_excitation,
    write_sto,
)


# ---------------------------------------------------------------------------
# Coupled simulation
# ---------------------------------------------------------------------------

def xpbd_coupled_simple_arm(cfg, verbose=True):
    """Run XPBD + MuJoCo coupled SimpleArm.

    Args:
        cfg: Config dict from data/simpleArm/config.json.
        verbose: Print progress.

    Returns:
        dict with times, elbow_angles, forces, norm_fiber_lengths,
        activations, max_iso_force, muscle_type.
    """
    mus = cfg["muscle"]
    act_cfg = cfg["activation"]
    sol = cfg["solver"]
    ic = cfg["initial_conditions"]
    geo = cfg["geometry"]
    xpbd_cfg = cfg.get("xpbd", {})

    F_max = mus["max_isometric_force"]
    L_opt = mus["optimal_fiber_length"]
    L_slack = mus["tendon_slack_length"]
    V_max = mus["max_contraction_velocity"]
    d_damp = mus["fiber_damping"]

    dt = sol["dt"]
    n_steps = sol["n_steps"]
    device = sol.get("arch", "cpu")
    theta0 = np.radians(ic["elbow_angle_deg"])

    num_substeps = xpbd_cfg.get("num_substeps", 30)
    volume_stiffness = xpbd_cfg.get("volume_stiffness", 1e6)
    arap_stiffness = xpbd_cfg.get("arap_stiffness", 1e6)
    attach_origin_k = xpbd_cfg.get("attach_origin_stiffness", 1e37)
    attach_insertion_k = xpbd_cfg.get("attach_insertion_stiffness", 1e10)
    warmup_steps = xpbd_cfg.get("warmup_steps", 10)
    mj_per_xpbd = xpbd_cfg.get("mj_per_xpbd", 10)
    density = xpbd_cfg.get("density", 1060.0)
    veldamping = xpbd_cfg.get("veldamping", 0.02)

    r = geo["muscle_radius"]
    n_circ = geo["n_circumferential"]
    n_axial = geo["n_axial"]

    # Timestep hierarchy
    dts = dt / num_substeps
    dtmj = dts / mj_per_xpbd

    if verbose:
        print(f"[XPBD] dt={dt} dts={dts:.6f} dtmj={dtmj:.8f} "
              f"substeps={num_substeps} mj_per_xpbd={mj_per_xpbd}")

    # --- MuJoCo ---
    mj_model = mujoco.MjModel.from_xml_string(build_mjcf(cfg))
    mj_model.opt.timestep = dtmj
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[0] = theta0
    mujoco.mj_forward(mj_model, mj_data)

    origin_sid = mj_model.site("muscle_origin").id
    insertion_sid = mj_model.site("muscle_insertion").id
    origin_pos = mj_data.site_xpos[origin_sid].copy()
    insertion_pos = mj_data.site_xpos[insertion_sid].copy()
    fiber_length_init = float(mj_data.ten_length[0]) - L_slack

    if verbose:
        print(f"[XPBD] fiber_length_init={fiber_length_init:.4f} "
              f"l_tilde={fiber_length_init / L_opt:.4f}")

    # --- Cylinder mesh for fiber portion, centered between tendons ---
    tendon_dir = insertion_pos - origin_pos
    tendon_path_length = float(np.linalg.norm(tendon_dir))
    tdu = (tendon_dir / tendon_path_length).astype(np.float32)
    mesh_length = fiber_length_init
    tendon_each = (tendon_path_length - mesh_length) / 2.0

    mesh_origin = origin_pos + tdu * tendon_each
    vertices, tets = create_cylinder_tet_mesh(mesh_length, r, n_axial, n_circ)
    R = rotation_matrix_align(np.array([0, 0, 1]), tdu)
    vertices = (R @ vertices.T).T + mesh_origin.astype(np.float32)

    n_tets = len(tets)
    fiber_dirs = np.tile(tdu, (n_tets, 1))

    # Boundary vertices (endcap layers)
    mesh_end = mesh_origin + tdu * mesh_length
    mask_dist = mesh_length * 0.05 + r
    origin_ids = [i for i in range(len(vertices))
                  if np.linalg.norm(vertices[i] - mesh_origin) < mask_dist]
    insertion_ids = [i for i in range(len(vertices))
                     if np.linalg.norm(vertices[i] - mesh_end) < mask_dist]

    if verbose:
        print(f"[XPBD] mesh: {len(vertices)} verts, {n_tets} tets, "
              f"origin={len(origin_ids)}, insertion={len(insertion_ids)}")

    # Bone targets
    origin_targets_init = vertices[origin_ids].copy()
    insertion_targets_init = vertices[insertion_ids].copy()
    all_attach_init = np.vstack([origin_targets_init, insertion_targets_init]).astype(np.float32)
    mesh_center_init = (mesh_origin + mesh_end).astype(np.float32) / 2.0
    tdu_init = tdu.copy()
    bone_targets_np = all_attach_init.copy()

    # --- Build XPBD sim using generic API ---
    wp.init()
    sim = MuscleSim.from_procedural(
        vertices, tets, fiber_dirs,
        bone_targets=bone_targets_np,
        constraint_configs=[
            {"type": "volume",  "name": "vol",  "stiffness": volume_stiffness, "dampingratio": 0.1},
            {"type": "tetarap", "name": "arap", "stiffness": arap_stiffness,   "dampingratio": 0.01},
        ],
        dts=dts, device=device, density=density, veldamping=veldamping,
    )

    # Add ATTACH constraints for origin + insertion
    attach_cons = build_attach_constraints(
        vertices, origin_ids, insertion_ids, bone_targets_np, tets,
        attach_origin_k, attach_insertion_k)
    sim.rebuild_constraints(extra_constraints=attach_cons)
    sim.attach_constraints = attach_cons
    sim.distanceline_constraints = []

    print(f"Built MuscleSim: {len(vertices)} verts, {n_tets} tets, {sim.n_cons} cons "
          f"(origin_attach={len(origin_ids)}, insertion_attach={len(insertion_ids)})")

    tet_idx = tets.astype(np.int32)
    rest_matrices = np.zeros((n_tets, 3, 3), dtype=np.float32)
    for e in range(n_tets):
        i0, i1, i2, i3 = tet_idx[e]
        M = np.column_stack([vertices[i0] - vertices[i3],
                             vertices[i1] - vertices[i3],
                             vertices[i2] - vertices[i3]])
        if abs(np.linalg.det(M)) > 1e-30:
            rest_matrices[e] = np.linalg.inv(M)
    stretch_to_ltilde = mesh_length / L_opt

    # Mesh exporter
    os.makedirs("output", exist_ok=True)
    exporter = MeshExporter(path="output/anim_xpbd", format="ply",
                            tet_indices=tet_idx, positions=vertices)

    # Bone capsule meshes (rest-pose, body-local coords)
    L_h = geo["humerus_length"]
    L_r = geo["radius_length"]
    humerus_verts_local, humerus_faces = create_capsule_mesh(
        [0, 0, 0], [0, -L_h, 0], radius=0.04, n_circ=12, n_axial=6)
    radius_verts_local, radius_faces = create_capsule_mesh(
        [0, 0, 0], [0, -L_r, 0], radius=0.03, n_circ=12, n_axial=6)

    bone_anim_dir = "output/anim_xpbd_bones"
    os.makedirs(bone_anim_dir, exist_ok=True)

    # Warm-up
    if verbose:
        print(f"[XPBD] Warm-up: {warmup_steps} steps")
    for _ in range(warmup_steps):
        sim.update_attach_targets()
        sim.integrate(); sim.clear(); sim.clear_reaction()
        sim.solve_constraints(); sim.update_velocities()
    if verbose:
        print("[XPBD] Warm-up done.")

    # --- Main loop ---
    activation = act_cfg["excitation_off"]
    prev_fiber_length = fiber_length_init
    physics_time = 0.0
    n_origin = len(origin_ids)

    times, elbow_angles, forces_out = [], [], []
    norm_fiber_lengths, activations_out = [], []

    if verbose:
        print(f"[XPBD] Simulating {n_steps} x {num_substeps} substeps "
              f"({n_steps * dt:.1f}s)")

    for step in range(n_steps):
        elbow_angle = float(mj_data.qpos[0])
        times.append(physics_time)
        elbow_angles.append(elbow_angle)
        activations_out.append(activation)

        for _sub in range(num_substeps):
            t_now = physics_time

            # a) Update attach targets: rotate to current path direction
            mujoco.mj_forward(mj_model, mj_data)
            cur_origin = mj_data.site_xpos[origin_sid].copy()
            cur_insertion = mj_data.site_xpos[insertion_sid].copy()
            cur_dir = cur_insertion - cur_origin
            cur_path_len = float(np.linalg.norm(cur_dir))
            cur_tdu = (cur_dir / cur_path_len).astype(np.float32)
            cur_tendon_each = (cur_path_len - mesh_length) / 2.0
            cur_mesh_origin = cur_origin + cur_tdu * cur_tendon_each
            cur_mesh_center = cur_mesh_origin + cur_tdu * (mesh_length / 2.0)

            R_path = rotation_matrix_align(tdu_init, cur_tdu)
            rotated = (R_path @ (all_attach_init - mesh_center_init).T).T
            bone_arr = (rotated + cur_mesh_center.astype(np.float32)).astype(np.float32)
            sim.bone_pos_field = wp.from_numpy(bone_arr, dtype=wp.vec3)

            # b) One XPBD step
            sim.update_attach_targets()
            sim.integrate(); sim.clear(); sim.clear_reaction()
            sim.solve_constraints(); sim.update_velocities()

            # c) MuJoCo substeps
            for _ in range(mj_per_xpbd):
                exc = compute_excitation(t_now, act_cfg)

                activation = float(activation_dynamics_step_np(
                    np.array([exc], dtype=np.float32),
                    np.array([activation], dtype=np.float32),
                    dtmj, tau_act=act_cfg["tau_act"],
                    tau_deact=act_cfg["tau_deact"])[0])

                fib_len = float(mj_data.ten_length[0]) - L_slack
                fib_vel = (fib_len - prev_fiber_length) / dtmj
                prev_fiber_length = fib_len
                v_norm = fib_vel / (V_max * L_opt)

                fl = float(active_force_length(fib_len / L_opt))
                fpe = float(passive_force_length(fib_len / L_opt))
                fv = float(force_velocity(np.clip(v_norm, -1.0, 1.0)))
                muscle_force = (activation * fl * fv + fpe + d_damp * v_norm) * F_max
                muscle_force = float(np.clip(muscle_force, 0.0, F_max * 2.0))

                mj_data.ctrl[0] = muscle_force
                mujoco.mj_step(mj_model, mj_data)
                t_now += dtmj

            physics_time += dts

        # --- Record + mesh quality ---
        pos_np = sim.pos.numpy()
        stretches = compute_fiber_stretches(pos_np, tet_idx, rest_matrices, fiber_dirs)
        l_tilde_xpbd = float(np.mean(stretches) * stretch_to_ltilde)

        n_inv = 0
        for e in range(n_tets):
            i0, i1, i2, i3 = tet_idx[e]
            Ds = np.column_stack([pos_np[i0] - pos_np[i3],
                                  pos_np[i1] - pos_np[i3],
                                  pos_np[i2] - pos_np[i3]])
            if np.linalg.det(Ds @ rest_matrices[e]) <= 0:
                n_inv += 1
        if n_inv > 0:
            raise MeshDistortionError(
                f"step={step}: {n_inv}/{n_tets} tets inverted")

        exporter.save_frame(pos_np.astype(np.float32), step)

        # Export bone capsules at current MuJoCo pose
        h_verts = humerus_verts_local
        save_ply(os.path.join(bone_anim_dir, f"humerus_{step:04d}.ply"),
                 h_verts, humerus_faces)
        r_pos = mj_data.xpos[2]
        r_quat = mj_data.xquat[2]
        r_verts = transform_by_quat(radius_verts_local, r_pos, r_quat)
        save_ply(os.path.join(bone_anim_dir, f"radius_{step:04d}.ply"),
                 r_verts.astype(np.float32), radius_faces)
        cur_origin = mj_data.site_xpos[origin_sid].copy()
        cur_insertion = mj_data.site_xpos[insertion_sid].copy()
        mesh_top_center = pos_np[origin_ids].mean(axis=0)
        mesh_bot_center = pos_np[insertion_ids].mean(axis=0)
        tp_v, tp_f = create_capsule_mesh(
            cur_origin, mesh_top_center, radius=0.005,
            n_circ=6, n_axial=4, n_cap=2)
        save_ply(os.path.join(bone_anim_dir, f"tendon_prox_{step:04d}.ply"),
                 tp_v, tp_f)
        td_v, td_f = create_capsule_mesh(
            mesh_bot_center, cur_insertion, radius=0.005,
            n_circ=6, n_axial=4, n_cap=2)
        save_ply(os.path.join(bone_anim_dir, f"tendon_dist_{step:04d}.ply"),
                 td_v, td_f)

        nfl_1d = (float(mj_data.ten_length[0]) - L_slack) / L_opt
        forces_out.append(muscle_force)
        norm_fiber_lengths.append(nfl_1d)

        if verbose and (step % 100 == 0 or step == n_steps - 1):
            print(f"  step={step:4d} t={physics_time:6.3f}s "
                  f"theta={np.degrees(elbow_angle):7.2f}deg "
                  f"F={muscle_force:7.2f}N a={activation:.4f} "
                  f"l_xpbd={l_tilde_xpbd:.4f}")

    if verbose:
        print(f"[XPBD] Done: {len(times)} pts, "
              f"final={np.degrees(elbow_angles[-1]):.1f}deg")

    exporter.finalize()

    # Save .sto
    sto_path = "output/SimpleArm_XPBD_Coupled_states.sto"
    cols = ["/jointset/elbow/elbow_coord_0/value",
            "/forceset/biceps/activation",
            "/forceset/biceps/fiber_force",
            "/forceset/biceps/norm_fiber_length"]
    write_sto(sto_path, "SimpleArm_XPBD_Coupled", cols, times,
              [elbow_angles, activations_out, forces_out, norm_fiber_lengths])
    if verbose:
        print(f"STO saved to {sto_path}")

    return {"times": np.array(times), "elbow_angles": np.array(elbow_angles),
            "forces": np.array(forces_out),
            "norm_fiber_lengths": np.array(norm_fiber_lengths),
            "activations": np.array(activations_out),
            "max_iso_force": F_max, "muscle_type": "XPBD_Coupled"}


def main():
    parser = argparse.ArgumentParser(description="XPBD coupled SimpleArm")
    parser.add_argument("--config", default="data/simpleArm/config.json")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    result = xpbd_coupled_simple_arm(cfg)
    if result:
        print(f"\nFinal elbow angle: {np.degrees(result['elbow_angles'][-1]):.2f}deg")


if __name__ == "__main__":
    main()
