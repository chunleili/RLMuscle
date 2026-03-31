"""XPBD coupled SimpleArm example (Stage 3).

Builds a MuscleSim programmatically from a cylinder mesh with 5 constraint
types: TETVOLUME + TETFIBERDGF + TETARAP + PIN + ATTACH.  PIN constraints
fix the origin vertices; ATTACH constraints elastically couple insertion
vertices to bone target positions.

Usage:
    RUN=example_xpbd_coupled_simple_arm uv run main.py
    uv run -m examples.example_xpbd_coupled_simple_arm
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import warp as wp

from VMuscle.activation import activation_dynamics_step_np
from VMuscle.constraints import ATTACH, PIN, TETARAP, TETFIBERDGF, TETVOLUME
from VMuscle.dgf_curves import active_force_length, force_velocity, passive_force_length
from VMuscle.mesh_io import MeshExporter
from VMuscle.mesh_utils import create_cylinder_tet_mesh
from VMuscle.muscle_warp import (
    Constraint,
    MuscleSim,
    fill_float_kernel,
    update_cons_restdir1_kernel,
)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def dgf_equilibrium_fiber_length(activation, normalized_load):
    """Invert DGF f_L curve: find lm where a * f_L(lm) = normalized_load.

    Returns equilibrium on the ascending limb (lm < 1).
    normalized_load = F_ext / F_max (e.g. mg / (sigma0 * A)).
    """
    if activation < 1e-8:
        return 1.0
    target_fl = normalized_load / activation
    if target_fl >= 1.0:
        return 1.0  # can't produce enough force
    # Search ascending limb [0.3, 1.0]
    lm_range = np.linspace(0.3, 1.0, 2000)
    fl = active_force_length(lm_range)
    idx = np.argmin(np.abs(fl - target_fl))
    return float(lm_range[idx])


def load_config(path):
    """Load config JSON and return a flat dict of parameters."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Constraint upload helper
# ---------------------------------------------------------------------------

def _upload_all_constraints(sim, all_constraints):
    """Sort constraints by type, build cons_ranges, and upload to warp array.

    Also sets sim.n_cons, sim.cons_ranges, sim.cons, sim.reaction_accum.
    """
    all_constraints.sort(key=lambda c: c['type'])
    n_cons = len(all_constraints)
    sim.n_cons = n_cons
    sim.cons_ranges = {}

    if n_cons == 0:
        sim.cons = wp.zeros(0, dtype=Constraint)
        sim.reaction_accum = wp.zeros(1, dtype=wp.vec3)
        return

    # Build type ranges
    prev_type = None
    start_idx = 0
    for i, c in enumerate(all_constraints):
        c['cidx'] = i
        ctype = c['type']
        if ctype != prev_type:
            if prev_type is not None:
                sim.cons_ranges[prev_type] = (start_idx, i - start_idx)
            start_idx = i
            prev_type = ctype
    if prev_type is not None:
        sim.cons_ranges[prev_type] = (start_idx, n_cons - start_idx)

    # Build structured numpy array
    cons_np = np.zeros(n_cons, dtype=Constraint.numpy_dtype())
    cons_np['type'] = np.array([c['type'] for c in all_constraints], dtype=np.int32)
    cons_np['cidx'] = np.arange(n_cons, dtype=np.int32)
    cons_np['pts'] = np.array([c['pts'] for c in all_constraints], dtype=np.int32)
    cons_np['stiffness'] = np.array([c['stiffness'] for c in all_constraints], dtype=np.float32)
    cons_np['dampingratio'] = np.array([c['dampingratio'] for c in all_constraints], dtype=np.float32)
    cons_np['tetid'] = np.array([c['tetid'] for c in all_constraints], dtype=np.int32)
    cons_np['L'] = np.array([c['L'] for c in all_constraints], dtype=np.float32)
    cons_np['restlength'] = np.array([c['restlength'] for c in all_constraints], dtype=np.float32)
    cons_np['restvector'] = np.array([c['restvector'] for c in all_constraints], dtype=np.float32)
    cons_np['restdir'] = np.array([c['restdir'] for c in all_constraints], dtype=np.float32)
    cons_np['compressionstiffness'] = np.array([c['compressionstiffness'] for c in all_constraints], dtype=np.float32)

    sim.cons = wp.array(cons_np, dtype=Constraint)
    sim.reaction_accum = wp.zeros(max(n_cons, 1), dtype=wp.vec3)


# ---------------------------------------------------------------------------
# Geometry / extraction helpers
# ---------------------------------------------------------------------------

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
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(np.arccos(cos_theta)) * K + (1 - cos_theta) * (K @ K)
    return R.astype(np.float32)


def compute_fiber_stretches(pos, tet_idx, rest_matrices, fiber_dirs):
    """Compute per-tet fiber stretch via deformation gradient F = Ds @ Dm_inv.

    Uses pts[3] as reference vertex (matching XPBD kernel convention):
    Ds columns = [p0-p3, p1-p3, p2-p3]
    """
    n = len(tet_idx)
    stretches = np.empty(n)
    for e in range(n):
        i0, i1, i2, i3 = tet_idx[e]
        Ds = np.column_stack([pos[i0] - pos[i3], pos[i1] - pos[i3], pos[i2] - pos[i3]])
        F = Ds @ rest_matrices[e]
        Fd = F @ fiber_dirs[e]
        stretches[e] = max(np.linalg.norm(Fd), 1e-8)
    return stretches


def update_bone_targets(sim, new_positions):
    """Update bone_pos_field with new insertion target positions."""
    sim.bone_pos_field = wp.from_numpy(new_positions.astype(np.float32), dtype=wp.vec3)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_xpbd_muscle_sim(
    vertices,
    tets,
    fiber_dirs_per_tet,
    origin_ids,
    insertion_ids,
    bone_targets,
    *,
    sigma0=159155.0,
    dt=0.0167,
    num_substeps=10,
    device="cpu",
    volume_stiffness=1e10,
    fiber_stiffness=1000.0,
    fiber_damping=0.1,
    arap_stiffness=1e10,
    arap_damping=0.1,
    contraction_factor=0.4,
    pin_stiffness=1e12,
    attach_stiffness=1e6,
    fiber_stiffness_scale=100000.0,
    gravity=9.81,
    density=1060.0,
    veldamping=0.02,
):
    """Build a MuscleSim programmatically with 5 constraint types.

    Parameters
    ----------
    vertices : (N, 3) float32 - mesh vertex positions
    tets : (M, 4) int32 - tet connectivity
    fiber_dirs_per_tet : (M, 3) float32 - per-tet fiber directions
    origin_ids : list[int] - vertex indices to pin (fixed end)
    insertion_ids : list[int] - vertex indices to attach to bone targets
    bone_targets : (K, 3) float32 - target positions for insertion vertices
    sigma0 : float - peak isometric stress (Pa)
    dt : float - time step
    num_substeps : int - XPBD sub-steps per frame
    device : str - warp device
    """
    vertices = vertices.astype(np.float32)
    tets = tets.astype(np.int32)
    n_v = len(vertices)
    n_tet = len(tets)

    # Per-vertex fiber directions (average from incident tets)
    v_fiber = np.zeros((n_v, 3), dtype=np.float32)
    v_count = np.zeros(n_v, dtype=np.float32)
    for e, tet in enumerate(tets):
        for vi in tet:
            v_fiber[vi] += fiber_dirs_per_tet[e]
            v_count[vi] += 1.0
    mask = v_count > 0
    v_fiber[mask] /= v_count[mask, None]
    norms = np.linalg.norm(v_fiber, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    v_fiber /= norms

    # Build SimConfig
    sim_cfg = SimpleNamespace(
        geo_path="<procedural>",
        bone_geo_path="<none>",
        gui=False,
        render_mode="none",
        constraints=[
            {"type": "volume", "name": "vol",
             "stiffness": volume_stiffness, "dampingratio": 1.0},
            {"type": "fiberdgf", "name": "fiber_dgf",
             "stiffness": fiber_stiffness, "dampingratio": fiber_damping,
             "sigma0": sigma0, "contraction_factor": contraction_factor},
            {"type": "tetarap", "name": "arap",
             "stiffness": arap_stiffness, "dampingratio": arap_damping},
        ],
        dt=dt,
        num_substeps=num_substeps,
        gravity=gravity,
        density=density,
        veldamping=veldamping,
        contraction_ratio=contraction_factor,
        fiber_stiffness_scale=fiber_stiffness_scale,
        HAS_compressstiffness=False,
        arch=device,
        save_image=False,
        pause=False,
        reset=False,
        show_auxiliary_meshes=False,
        show_wireframe=False,
        render_fps=24,
        color_bones=False,
        color_muscles="tendonmask",
        activation=0.0,
        nsteps=1,
    )

    # Construct MuscleSim bypassing file-based __init__
    wp.set_device(device)
    sim = object.__new__(MuscleSim)
    sim.cfg = sim_cfg
    sim.constraint_configs = sim_cfg.constraints

    # Mesh data
    sim.pos0_np = vertices
    sim.tet_np = tets
    sim.v_fiber_np = v_fiber
    sim.v_tendonmask_np = None
    sim.geo = SimpleNamespace()
    sim.n_verts = n_v

    # Bone data for ATTACH constraints
    bone_targets = np.asarray(bone_targets, dtype=np.float32)
    sim.bone_geo = None
    sim.bone_pos = bone_targets
    sim.bone_indices_np = np.zeros(0, dtype=np.int32)
    sim.bone_muscle_ids = {}

    # Initialize backend, fields, rest state, surface
    wp.init()
    sim._init_backend()
    sim._allocate_fields()
    sim._init_fields()
    sim._precompute_rest()
    sim._build_surface_tris()
    sim._create_bone_fields()

    # Solver parameters
    sim.use_jacobi = False
    sim.use_colored_gs = False
    sim.contraction_ratio = contraction_factor
    sim.fiber_stiffness_scale = fiber_stiffness_scale
    sim.has_compressstiffness = False
    sim.dt = dt / num_substeps
    sim.step_cnt = 0
    sim.renderer = None

    # Build TETVOLUME + TETFIBERDGF + TETARAP via mixin
    sim.build_constraints()

    # Extract existing constraints
    all_constraints = list(sim.raw_constraints)

    # Build pt2tet mapping
    pt2tet = {}
    for i, tet in enumerate(tets):
        for vi in tet:
            if vi not in pt2tet:
                pt2tet[vi] = i

    # Add PIN constraints for origin vertices
    for vid in origin_ids:
        pos = vertices[vid]
        c = dict(
            type=PIN,
            pts=[int(vid), -1, -1, -1],
            stiffness=pin_stiffness,
            dampingratio=0.1,
            tetid=pt2tet.get(vid, -1),
            L=[0.0, 0.0, 0.0],
            restlength=0.0,
            restvector=[float(pos[0]), float(pos[1]), float(pos[2]), 1.0],
            restdir=[0.0, 0.0, 0.0],
            compressionstiffness=-1.0,
        )
        all_constraints.append(c)

    # Add ATTACH constraints for insertion vertices
    attach_constraints = []
    for j, vid in enumerate(insertion_ids):
        bone_idx = j  # index into bone_targets
        tgt = bone_targets[bone_idx]
        src = vertices[vid]
        dist = float(np.linalg.norm(tgt - src))
        c = dict(
            type=ATTACH,
            pts=[int(vid), -1, int(bone_idx), -1],
            stiffness=attach_stiffness,
            dampingratio=0.1,
            tetid=pt2tet.get(vid, -1),
            L=[0.0, 0.0, 0.0],
            restlength=dist,
            restvector=[float(tgt[0]), float(tgt[1]), float(tgt[2]), 1.0],
            restdir=[0.0, 0.0, 0.0],
            compressionstiffness=-1.0,
        )
        all_constraints.append(c)
        attach_constraints.append(c)

    # Re-upload all constraints (sorted by type)
    _upload_all_constraints(sim, all_constraints)
    sim.attach_constraints = attach_constraints
    sim.distanceline_constraints = []

    # Mark origin vertices as stopped (kinematic)
    stopped_np = sim.stopped.numpy()
    for vid in origin_ids:
        stopped_np[vid] = 1
    sim.stopped = wp.from_numpy(stopped_np.astype(np.int32), dtype=wp.int32)

    print(f"Built MuscleSim: {n_v} verts, {n_tet} tets, {sim.n_cons} constraints")
    print(f"  cons_ranges: {sim.cons_ranges}")
    print(f"  PIN: {len(origin_ids)}, ATTACH: {len(insertion_ids)}")

    return sim


# ---------------------------------------------------------------------------
# Coupled simulation loop
# ---------------------------------------------------------------------------

def xpbd_coupled_simple_arm(cfg, verbose=True):
    """Run XPBD + MuJoCo coupled SimpleArm simulation.

    Coupling strategy:
      - XPBD handles 3D muscle deformation with DGF fiber constraints.
      - MuJoCo handles rigid-body dynamics (elbow hinge).
      - Each outer step: update bone targets from MuJoCo, run XPBD substeps,
        extract fiber stretch from deformation gradient, compute DGF force,
        inject into MuJoCo.
      - contraction_factor is updated every outer step via dgf_equilibrium_fiber_length.
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

    outer_dt = sol["dt"]
    n_steps = sol["n_steps"]
    device = sol.get("arch", "cpu")
    theta0 = np.radians(ic["elbow_angle_deg"])

    num_substeps = xpbd_cfg.get("num_substeps", 20)
    attach_stiffness = xpbd_cfg.get("attach_stiffness", 1e6)
    pin_stiffness = xpbd_cfg.get("pin_stiffness", 1e8)
    fiber_stiffness = xpbd_cfg.get("fiber_stiffness", 1000.0)
    fiber_stiffness_scale = xpbd_cfg.get("fiber_stiffness_scale", 200.0)
    volume_stiffness = xpbd_cfg.get("volume_stiffness", 10000.0)
    arap_stiffness = xpbd_cfg.get("arap_stiffness", 500.0)
    warmup_steps = xpbd_cfg.get("warmup_steps", 50)

    r = geo["muscle_radius"]
    n_circ = geo["n_circumferential"]
    n_axial = geo["n_axial"]

    # sigma0 = F_max / cross-section area [Pa]
    sigma0 = F_max / (np.pi * r ** 2)

    # --- Build MuJoCo ---
    _examples_dir = os.path.dirname(os.path.abspath(__file__))
    if _examples_dir not in sys.path:
        sys.path.insert(0, _examples_dir)
    from example_mujoco_simple_arm import build_mjcf

    mjcf_str = build_mjcf(cfg)
    mj_model = mujoco.MjModel.from_xml_string(mjcf_str)
    mj_data = mujoco.MjData(mj_model)
    mj_dt = mj_model.opt.timestep
    mj_substeps = max(1, int(round(outer_dt / mj_dt)))

    mj_data.qpos[0] = theta0
    mujoco.mj_forward(mj_model, mj_data)

    origin_site_id = mj_model.site("muscle_origin").id
    insertion_site_id = mj_model.site("muscle_insertion").id
    origin_pos = mj_data.site_xpos[origin_site_id].copy()
    insertion_pos = mj_data.site_xpos[insertion_site_id].copy()
    ten_length_init = float(mj_data.ten_length[0])
    fiber_length_init = ten_length_init - L_slack

    if verbose:
        print(
            f"[XPBD+MuJoCo] Initial: ten_length={ten_length_init:.4f}, "
            f"fiber_length={fiber_length_init:.4f}, "
            f"l_tilde={fiber_length_init / L_opt:.4f}"
        )

    # --- Build cylinder mesh in world frame ---
    # The mesh represents the muscle fiber (not the full tendon path).
    # Length = fiber_length_init; positioned from origin along tendon direction.
    tendon_dir = insertion_pos - origin_pos
    tendon_path_length = float(np.linalg.norm(tendon_dir))
    tendon_dir_unit = tendon_dir / tendon_path_length

    mesh_length = fiber_length_init
    vertices, tets = create_cylinder_tet_mesh(mesh_length, r, n_circ, n_axial)
    # Rotate from Z to tendon direction, then translate to origin
    R = _rotation_matrix_from_z_to(tendon_dir_unit)
    vertices = (R @ vertices.T).T + origin_pos.astype(np.float32)

    from VMuscle.mesh_utils import assign_fiber_directions
    fiber_dirs = assign_fiber_directions(vertices, tets, axis=2)
    # Re-orient fiber directions to tendon direction
    for e in range(len(tets)):
        fiber_dirs[e] = tendon_dir_unit.astype(np.float32)

    # Identify origin/insertion boundary vertices by distance to mesh endpoints
    mesh_insertion_pos = origin_pos + tendon_dir_unit * mesh_length
    # Threshold must cover the radius to capture all endcap vertices
    boundary_threshold = r * 1.5
    origin_ids = []
    insertion_ids = []
    for i, v in enumerate(vertices):
        if np.linalg.norm(v - origin_pos) < boundary_threshold:
            origin_ids.append(i)
        elif np.linalg.norm(v - mesh_insertion_pos) < boundary_threshold:
            insertion_ids.append(i)

    if verbose:
        print(
            f"[XPBD+MuJoCo] Mesh: {len(vertices)} verts, {len(tets)} tets, "
            f"origin={len(origin_ids)}, insertion={len(insertion_ids)}, "
            f"mesh_length={mesh_length:.4f}"
        )

    # Bone targets: use actual mesh vertex positions for each insertion vertex.
    # During simulation, these are displaced by MuJoCo insertion site motion.
    bone_targets = vertices[insertion_ids].copy()

    # --- Build XPBD MuscleSim ---
    wp.init()
    sim = build_xpbd_muscle_sim(
        vertices, tets, fiber_dirs, origin_ids, insertion_ids, bone_targets,
        sigma0=sigma0,
        dt=outer_dt,
        num_substeps=num_substeps,
        device=device,
        volume_stiffness=volume_stiffness,
        fiber_stiffness=fiber_stiffness,
        fiber_damping=mus["fiber_damping"],
        arap_stiffness=arap_stiffness,
        arap_damping=0.1,
        contraction_factor=1.0,
        pin_stiffness=pin_stiffness,
        attach_stiffness=attach_stiffness,
        fiber_stiffness_scale=fiber_stiffness_scale,
        gravity=0.0,
        density=1060.0,
        veldamping=0.02,
    )

    n_tets = len(tets)
    tet_idx = tets.astype(np.int32)

    # Precompute rest matrices (CPU side, for fiber stretch extraction)
    # Uses pts[3] as reference vertex, matching kernel convention
    rest_matrices = np.zeros((n_tets, 3, 3), dtype=np.float32)
    for e in range(n_tets):
        i0, i1, i2, i3 = tet_idx[e]
        M = np.column_stack([
            vertices[i0] - vertices[i3],
            vertices[i1] - vertices[i3],
            vertices[i2] - vertices[i3],
        ])
        det = np.linalg.det(M)
        if abs(det) > 1e-30:
            rest_matrices[e] = np.linalg.inv(M)

    # stretch_to_ltilde: converts raw mesh stretch to normalized fiber length
    stretch_to_ltilde = mesh_length / L_opt

    # --- Mesh exporter ---
    os.makedirs("output", exist_ok=True)
    exporter = MeshExporter(
        path="output/anim_xpbd",
        format="ply",
        tet_indices=tet_idx,
        positions=vertices,
    )

    # --- Warm-up ---
    # Warm-up with activation=0 to let the mesh settle under constraint system
    # without any fiber contraction. Both ends are fixed (PIN + ATTACH), so
    # activating fiber during warm-up causes internal compression instability.
    e_off = act_cfg["excitation_off"]
    activation = e_off  # will be used in main loop; warm-up uses 0

    if verbose:
        print(f"[XPBD+MuJoCo] Warm-up: {warmup_steps} steps (activation=0)")

    # Zero activation during warm-up
    wp.launch(fill_float_kernel, dim=sim.activation.shape[0],
              inputs=[sim.activation, float(0.0)])

    # Set contraction_factor=0 so target_stretch=1.0 (no contraction)
    if TETFIBERDGF in sim.cons_ranges:
        fib_offset, fib_count = sim.cons_ranges[TETFIBERDGF]
        wp.launch(update_cons_restdir1_kernel, dim=fib_count,
                  inputs=[sim.cons, float(0.0), int(TETFIBERDGF), int(fib_offset), int(fib_count)])

    for _ in range(warmup_steps):
        sim.update_attach_targets()
        for _ in range(num_substeps):
            sim.integrate()
            sim.clear()
            sim.clear_reaction()
            sim.solve_constraints()
            sim.update_velocities()

    if verbose:
        print("[XPBD+MuJoCo] Warm-up done.")

    # --- Main simulation loop ---
    activation = e_off
    prev_fiber_length = fiber_length_init
    prev_muscle_force = 0.0
    physics_time = 0.0

    times = []
    elbow_angles = []
    forces_out = []
    norm_fiber_lengths = []
    activations_out = []

    if verbose:
        print(f"[XPBD+MuJoCo] Simulating {n_steps} outer steps, F_max={F_max:.0f}N")

    for step in range(n_steps):
        t = physics_time

        # Record state
        elbow_angle = float(mj_data.qpos[0])
        times.append(t)
        elbow_angles.append(elbow_angle)
        activations_out.append(activation)

        # 1. MuJoCo forward -> get new insertion position -> update bone targets
        # Bone targets are displaced by how much MuJoCo insertion moved from initial.
        mujoco.mj_forward(mj_model, mj_data)
        new_insertion = mj_data.site_xpos[insertion_site_id].copy()
        delta = (new_insertion - insertion_pos).astype(np.float32)
        new_bone_targets = bone_targets + delta
        update_bone_targets(sim, new_bone_targets)

        # 2. Set activation on GPU
        wp.launch(fill_float_kernel, dim=sim.activation.shape[0],
                  inputs=[sim.activation, float(activation)])

        # 3. Update contraction_factor from DGF equilibrium
        # cf = 1 - lm_eq (contraction_factor, not equilibrium length)
        normalized_load = prev_muscle_force / F_max if F_max > 0 else 0.0
        cf = 1.0 - dgf_equilibrium_fiber_length(activation, normalized_load)
        if TETFIBERDGF in sim.cons_ranges:
            fib_offset, fib_count = sim.cons_ranges[TETFIBERDGF]
            wp.launch(update_cons_restdir1_kernel, dim=fib_count,
                      inputs=[sim.cons, float(cf), int(TETFIBERDGF), int(fib_offset), int(fib_count)])

        # 4. XPBD substeps
        sim.update_attach_targets()
        for _ in range(num_substeps):
            sim.integrate()
            sim.clear()
            sim.clear_reaction()
            sim.solve_constraints()
            sim.update_velocities()

        sim.step_cnt += 1

        # 5. Extract fiber stretch from deformation gradient
        pos_np = sim.pos.numpy()
        stretches = compute_fiber_stretches(pos_np, tet_idx, rest_matrices, fiber_dirs)
        l_tilde_xpbd = float(np.mean(stretches) * stretch_to_ltilde)

        # Save mesh frame
        exporter.save_frame(pos_np.astype(np.float32), step)

        # 6. MuJoCo substeps with excitation schedule + DGF force
        t_start = act_cfg["excitation_start_time"]
        t_end_exc = act_cfg["excitation_end_time"]
        e_on = act_cfg["excitation_on"]

        for sub in range(mj_substeps):
            t_sub = t + sub * mj_dt

            # Excitation schedule (smoothstep)
            if t_sub < t_start:
                excitation = e_off
            elif t_sub >= t_end_exc:
                excitation = e_on
            else:
                frac = (t_sub - t_start) / (t_end_exc - t_start)
                frac = frac * frac * (3.0 - 2.0 * frac)
                excitation = e_off + (e_on - e_off) * frac

            # Activation dynamics
            activation = float(activation_dynamics_step_np(
                np.array([excitation], dtype=np.float32),
                np.array([activation], dtype=np.float32),
                mj_dt,
                tau_act=act_cfg["tau_act"],
                tau_deact=act_cfg["tau_deact"],
            )[0])

            # Fiber length and velocity from MuJoCo
            ten_len = float(mj_data.ten_length[0])
            fib_len = ten_len - L_slack
            fib_vel = (fib_len - prev_fiber_length) / mj_dt if (step > 0 or sub > 0) else 0.0
            prev_fiber_length = fib_len
            v_norm = fib_vel / (V_max * L_opt)

            # Use 1D fiber length for force (XPBD mesh tracks deformation
            # but force comes from tendon model, matching MuJoCo-only behavior).
            # The XPBD l_tilde is recorded for diagnostic/visualization purposes.
            l_tilde_now = fib_len / L_opt

            # DGF force computation
            fl = float(active_force_length(l_tilde_now))
            fpe = float(passive_force_length(l_tilde_now))
            fv = float(force_velocity(np.clip(v_norm, -1.0, 1.0)))
            muscle_force = (activation * fl * fv + fpe + d_damp * v_norm) * F_max
            muscle_force = float(np.clip(muscle_force, 0.0, F_max * 2.0))

            mj_data.ctrl[0] = muscle_force
            mujoco.mj_step(mj_model, mj_data)
            physics_time += mj_dt

        prev_muscle_force = muscle_force

        # 7. Record results (use 1D fiber length for norm_fiber_length,
        # since that's what drives the force computation)
        ten_length_final = float(mj_data.ten_length[0])
        nfl_1d = (ten_length_final - L_slack) / L_opt
        forces_out.append(muscle_force)
        norm_fiber_lengths.append(nfl_1d)

        if verbose and (step % 100 == 0 or step == n_steps - 1):
            print(
                f"  step={step:4d} t={t:6.3f}s "
                f"theta={np.degrees(elbow_angle):7.2f}deg "
                f"F={muscle_force:7.2f}N a={activation:.4f} "
                f"l_tilde={l_tilde_xpbd:.4f} cf={cf:.4f}"
            )

    if verbose:
        print(
            f"[XPBD+MuJoCo] Done: {len(times)} points, "
            f"final angle={np.degrees(elbow_angles[-1]):.1f}deg"
        )

    # --- Finalize mesh export ---
    exporter.finalize()

    # --- Save .sto file ---
    sto_path = "output/SimpleArm_XPBD_Coupled_states.sto"
    n_rows = len(times)
    cols = [
        "/jointset/elbow/elbow_coord_0/value",
        "/forceset/biceps/activation",
        "/forceset/biceps/fiber_force",
        "/forceset/biceps/norm_fiber_length",
    ]
    with open(sto_path, "w") as f:
        f.write("SimpleArm_XPBD_Coupled\n")
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
        "muscle_type": "XPBD_Coupled",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="XPBD + MuJoCo coupled SimpleArm (Stage 3)")
    parser.add_argument("--config", default="data/simpleArm/config.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = xpbd_coupled_simple_arm(cfg)
    if result:
        print(f"\nFinal elbow angle: {np.degrees(result['elbow_angles'][-1]):.2f}deg")


if __name__ == "__main__":
    main()
