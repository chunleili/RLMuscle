"""XPBD coupled SimpleArm example (Stage 3).

Builds a MuscleSim programmatically from a cylinder mesh with
TETVOLUME + TETARAP + ATTACH constraints.  Both origin and insertion
ends use ATTACH springs (no PIN, no stopped/kinematic vertices).

Coupling: flat loop at dts = dt / num_substeps.
Each dts: 1 XPBD solve + 10 MuJoCo substeps.

Usage:
    RUN=example_xpbd_coupled_simple_arm uv run main.py
    uv run python examples/example_xpbd_coupled_simple_arm.py
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
from VMuscle.constraints import ATTACH, TETARAP, TETVOLUME
from VMuscle.dgf_curves import active_force_length, force_velocity, passive_force_length
from VMuscle.mesh_io import MeshExporter
from VMuscle.mesh_utils import create_cylinder_tet_mesh
from VMuscle.muscle_warp import Constraint, MuscleSim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path="data/simpleArm/config.json"):
    with open(path) as f:
        return json.load(f)


def _rotation_matrix_from_z_to(target_dir):
    """Rotation matrix that takes Z-axis to *target_dir*."""
    z = np.array([0, 0, 1], dtype=np.float64)
    t = np.asarray(target_dir, dtype=np.float64)
    t /= np.linalg.norm(t)
    c = np.dot(z, t)
    if c > 1.0 - 1e-8:
        return np.eye(3, dtype=np.float32)
    if c < -1.0 + 1e-8:
        return np.diag([1, -1, -1]).astype(np.float32)
    v = np.cross(z, t)
    v /= np.linalg.norm(v)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + np.sin(np.arccos(c)) * K + (1 - c) * (K @ K)
    return R.astype(np.float32)


def compute_fiber_stretches(pos, tet_idx, rest_matrices, fiber_dirs):
    """Per-tet fiber stretch.  pts[3] is reference vertex."""
    n = len(tet_idx)
    out = np.empty(n)
    for e in range(n):
        i0, i1, i2, i3 = tet_idx[e]
        Ds = np.column_stack([pos[i0] - pos[i3],
                              pos[i1] - pos[i3],
                              pos[i2] - pos[i3]])
        Fd = (Ds @ rest_matrices[e]) @ fiber_dirs[e]
        out[e] = max(np.linalg.norm(Fd), 1e-8)
    return out


def create_capsule_mesh(p0, p1, radius, n_circ=12, n_axial=8, n_cap=4):
    """Create a capsule surface mesh (cylinder + two hemispheres).

    Args:
        p0, p1: (3,) endpoints of the capsule axis.
        radius: capsule radius.
        n_circ: circumferential segments.
        n_axial: axial segments along the cylinder portion.
        n_cap: latitude rings per hemisphere.

    Returns:
        verts: (N, 3) float32, faces: (M, 3) int32
    """
    p0, p1 = np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)
    axis = p1 - p0
    length = np.linalg.norm(axis)
    if length < 1e-12:
        axis = np.array([0, 1, 0], dtype=np.float64)
        length = 1e-6
    az = axis / length

    # Build local frame
    up = np.array([0, 0, 1], dtype=np.float64)
    if abs(np.dot(az, up)) > 0.99:
        up = np.array([1, 0, 0], dtype=np.float64)
    ax = np.cross(az, up); ax /= np.linalg.norm(ax)
    ay = np.cross(az, ax)

    verts, faces = [], []

    def add_ring(center, r_ring, z_offset=0.0):
        """Add a ring of n_circ vertices; return start index."""
        idx0 = len(verts)
        for i in range(n_circ):
            theta = 2.0 * np.pi * i / n_circ
            p = center + r_ring * (np.cos(theta) * ax + np.sin(theta) * ay)
            verts.append(p)
        return idx0

    # Bottom hemisphere (at p0, pointing toward -az)
    verts.append(p0 - az * radius)  # south pole
    pole_bottom = 0
    prev_ring = None
    for j in range(1, n_cap + 1):
        phi = np.pi / 2 * j / n_cap  # 0 → π/2
        r_ring = radius * np.sin(phi)
        z_off = -radius * np.cos(phi)
        center = p0 + az * z_off
        ring_start = add_ring(center, r_ring)
        if prev_ring is None:
            # Connect pole to first ring
            for i in range(n_circ):
                i_next = (i + 1) % n_circ
                faces.append([pole_bottom, ring_start + i_next, ring_start + i])
        else:
            for i in range(n_circ):
                i_next = (i + 1) % n_circ
                faces.append([prev_ring + i, ring_start + i, ring_start + i_next])
                faces.append([prev_ring + i, ring_start + i_next, prev_ring + i_next])
        prev_ring = ring_start

    # Cylinder body (from p0 to p1)
    bottom_ring = prev_ring
    for j in range(1, n_axial + 1):
        t = j / n_axial
        center = p0 + axis * t
        ring_start = add_ring(center, radius)
        for i in range(n_circ):
            i_next = (i + 1) % n_circ
            faces.append([prev_ring + i, ring_start + i, ring_start + i_next])
            faces.append([prev_ring + i, ring_start + i_next, prev_ring + i_next])
        prev_ring = ring_start

    # Top hemisphere (at p1, pointing toward +az)
    for j in range(1, n_cap + 1):
        phi = np.pi / 2 * j / n_cap
        r_ring = radius * np.cos(phi)
        z_off = radius * np.sin(phi)
        center = p1 + az * z_off
        if r_ring < 1e-10:
            # Top pole
            pole_top = len(verts)
            verts.append(p1 + az * radius)
            for i in range(n_circ):
                i_next = (i + 1) % n_circ
                faces.append([prev_ring + i, pole_top, prev_ring + i_next])
        else:
            ring_start = add_ring(center, r_ring)
            for i in range(n_circ):
                i_next = (i + 1) % n_circ
                faces.append([prev_ring + i, ring_start + i, ring_start + i_next])
                faces.append([prev_ring + i, ring_start + i_next, prev_ring + i_next])
            prev_ring = ring_start

    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


def transform_capsule(verts, body_pos, body_quat):
    """Transform capsule vertices by MuJoCo body pose (pos + quaternion)."""
    # MuJoCo quaternion: [w, x, y, z]
    w, x, y, z = body_quat
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return (R @ verts.astype(np.float64).T).T + body_pos


def _upload_all_constraints(sim, all_constraints):
    """Sort → assign cidx → upload to warp array.  Sets cons / cons_ranges."""
    all_constraints.sort(key=lambda c: c["type"])
    n = len(all_constraints)
    sim.n_cons = n
    sim.cons_ranges = {}
    if n == 0:
        sim.cons = wp.zeros(0, dtype=Constraint)
        sim.reaction_accum = wp.zeros(1, dtype=wp.vec3)
        return
    prev, start = None, 0
    for i, c in enumerate(all_constraints):
        c["cidx"] = i
        t = c["type"]
        if t != prev:
            if prev is not None:
                sim.cons_ranges[prev] = (start, i - start)
            start, prev = i, t
    sim.cons_ranges[prev] = (start, n - start)

    dt = Constraint.numpy_dtype()
    arr = np.zeros(n, dtype=dt)
    for key in ("type", "pts", "stiffness", "dampingratio", "tetid",
                "L", "restlength", "restvector", "restdir",
                "compressionstiffness"):
        arr[key] = np.array([c[key] for c in all_constraints])
    arr["cidx"] = np.arange(n, dtype=np.int32)
    sim.cons = wp.array(arr, dtype=Constraint)
    sim.reaction_accum = wp.zeros(max(n, 1), dtype=wp.vec3)


# ---------------------------------------------------------------------------
# Build MuscleSim (object.__new__ pattern, no file loading)
# ---------------------------------------------------------------------------

def build_xpbd_muscle_sim(
    vertices, tets, fiber_dirs_per_tet,
    origin_ids, insertion_ids,
    bone_targets_np,          # (n_origin + n_insertion, 3) float32
    *,
    dts,                      # XPBD timestep (= dt / num_substeps)
    device="cpu",
    volume_stiffness=1e6,
    arap_stiffness=1e6,
    arap_damping=0.01,
    attach_origin_stiffness=1e37,
    attach_insertion_stiffness=1e10,
    density=1060.0,
    veldamping=0.02,
):
    """Build MuscleSim with TETVOLUME + TETARAP + ATTACH (origin+insertion).

    bone_targets_np layout: first len(origin_ids) rows = origin targets,
    next len(insertion_ids) rows = insertion targets.
    """
    vertices = vertices.astype(np.float32)
    tets = tets.astype(np.int32)
    n_v, n_tet = len(vertices), len(tets)

    # Per-vertex fiber directions (averaged from per-tet)
    vf = np.zeros((n_v, 3), dtype=np.float32)
    vc = np.zeros(n_v, dtype=np.float32)
    for e, t in enumerate(tets):
        for vi in t:
            vf[vi] += fiber_dirs_per_tet[e]; vc[vi] += 1.0
    vc = np.maximum(vc, 1.0)
    vf /= vc[:, None]
    vf /= np.maximum(np.linalg.norm(vf, axis=1, keepdims=True), 1e-8)

    # Config — only TETVOLUME + TETARAP via mixin; ATTACH added manually below
    sim_cfg = SimpleNamespace(
        geo_path="<procedural>", bone_geo_path="<none>",
        gui=False, render_mode="none",
        constraints=[
            {"type": "volume",  "name": "vol",  "stiffness": volume_stiffness, "dampingratio": 0.1},
            {"type": "tetarap", "name": "arap", "stiffness": arap_stiffness,   "dampingratio": arap_damping},
        ],
        dt=dts, num_substeps=1,
        gravity=0.0, density=density, veldamping=veldamping,
        contraction_ratio=0.0, fiber_stiffness_scale=1.0,
        HAS_compressstiffness=False, arch=device,
        save_image=False, pause=False, reset=False,
        show_auxiliary_meshes=False, show_wireframe=False,
        render_fps=24, color_bones=False, color_muscles="tendonmask",
        activation=0.0, nsteps=1,
    )

    wp.set_device(device)
    sim = object.__new__(MuscleSim)
    sim.cfg = sim_cfg
    sim.constraint_configs = sim_cfg.constraints

    sim.pos0_np = vertices
    sim.tet_np = tets
    sim.v_fiber_np = vf
    sim.v_tendonmask_np = None
    sim.geo = SimpleNamespace()
    sim.n_verts = n_v

    # bone_pos stores ALL attachment targets (origin first, then insertion)
    sim.bone_geo = None
    sim.bone_pos = bone_targets_np.astype(np.float32)
    sim.bone_indices_np = np.zeros(0, dtype=np.int32)
    sim.bone_muscle_ids = {}

    wp.init()
    sim._init_backend()
    sim._allocate_fields()
    sim._init_fields()
    sim._precompute_rest()
    sim._build_surface_tris()
    sim._create_bone_fields()

    sim.use_jacobi = False
    sim.use_colored_gs = False
    sim.contraction_ratio = 0.0
    sim.fiber_stiffness_scale = 1.0
    sim.has_compressstiffness = False
    sim.dt = dts
    sim.step_cnt = 0
    sim.renderer = None

    # Build TETVOLUME + TETARAP via mixin
    sim.build_constraints()
    all_cons = list(sim.raw_constraints)

    # pt → tet mapping
    pt2tet = {}
    for i, t in enumerate(tets):
        for vi in t:
            pt2tet.setdefault(int(vi), int(i))

    n_origin = len(origin_ids)
    n_insertion = len(insertion_ids)

    # ATTACH for origin vertices (ultra-high stiffness, fixed targets)
    attach_cons = []
    for j, vid in enumerate(origin_ids):
        bone_idx = j  # first n_origin slots in bone_targets
        tgt = bone_targets_np[bone_idx]
        dist = float(np.linalg.norm(vertices[vid] - tgt))
        c = dict(type=ATTACH, pts=[int(vid), -1, int(bone_idx), -1],
                 stiffness=attach_origin_stiffness, dampingratio=0.0,
                 tetid=pt2tet.get(int(vid), -1), L=[0.0, 0.0, 0.0],
                 restlength=dist,
                 restvector=[float(tgt[0]), float(tgt[1]), float(tgt[2]), 1.0],
                 restdir=[0.0, 0.0, 0.0], compressionstiffness=-1.0)
        all_cons.append(c); attach_cons.append(c)

    # ATTACH for insertion vertices (high stiffness, targets updated per step)
    for j, vid in enumerate(insertion_ids):
        bone_idx = n_origin + j  # after origin slots
        tgt = bone_targets_np[bone_idx]
        dist = float(np.linalg.norm(vertices[vid] - tgt))
        c = dict(type=ATTACH, pts=[int(vid), -1, int(bone_idx), -1],
                 stiffness=attach_insertion_stiffness, dampingratio=0.0,
                 tetid=pt2tet.get(int(vid), -1), L=[0.0, 0.0, 0.0],
                 restlength=dist,
                 restvector=[float(tgt[0]), float(tgt[1]), float(tgt[2]), 1.0],
                 restdir=[0.0, 0.0, 0.0], compressionstiffness=-1.0)
        all_cons.append(c); attach_cons.append(c)

    _upload_all_constraints(sim, all_cons)
    sim.attach_constraints = attach_cons
    sim.distanceline_constraints = []

    # NO stopped array — all vertices are dynamic
    print(f"Built MuscleSim: {n_v} verts, {n_tet} tets, {sim.n_cons} cons "
          f"(origin_attach={n_origin}, insertion_attach={n_insertion})")
    return sim


# ---------------------------------------------------------------------------
# Coupled simulation
# ---------------------------------------------------------------------------

def xpbd_coupled_simple_arm(cfg, verbose=True):
    """Run XPBD + MuJoCo coupled SimpleArm.

    Flat loop at dts = dt / num_substeps.
    Each dts: 1 XPBD solve + mj_per_xpbd MuJoCo substeps.
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

    r = geo["muscle_radius"]
    n_circ = geo["n_circumferential"]
    n_axial = geo["n_axial"]

    # Timestep hierarchy
    dts = dt / num_substeps
    mj_per_xpbd = 10
    dtmj = dts / mj_per_xpbd

    if verbose:
        print(f"[XPBD] dt={dt} dts={dts:.6f} dtmj={dtmj:.8f} "
              f"substeps={num_substeps} mj_per_xpbd={mj_per_xpbd}")

    # --- MuJoCo ---
    _ed = os.path.dirname(os.path.abspath(__file__))
    if _ed not in sys.path:
        sys.path.insert(0, _ed)
    from example_mujoco_simple_arm import build_mjcf

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

    # --- Cylinder mesh for fiber portion only ---
    # Mesh covers fiber_length from origin along tendon direction.
    # Tendon gap (origin_end → insertion) is visualized separately.
    tendon_dir = insertion_pos - origin_pos
    tendon_path_length = float(np.linalg.norm(tendon_dir))
    tdu = (tendon_dir / tendon_path_length).astype(np.float32)
    mesh_length = fiber_length_init

    vertices, tets = create_cylinder_tet_mesh(mesh_length, r, n_axial, n_circ)
    R = _rotation_matrix_from_z_to(tdu)
    vertices = (R @ vertices.T).T + origin_pos.astype(np.float32)

    n_tets = len(tets)
    fiber_dirs = np.tile(tdu, (n_tets, 1))

    # Boundary vertices (endcap layers, mask by 5% distance along axis)
    mesh_end = origin_pos + tdu * mesh_length
    mask_dist = mesh_length * 0.05 + r  # 5% axial + radius
    origin_ids = [i for i in range(len(vertices))
                  if np.linalg.norm(vertices[i] - origin_pos) < mask_dist]
    insertion_ids = [i for i in range(len(vertices))
                     if np.linalg.norm(vertices[i] - mesh_end) < mask_dist]

    if verbose:
        print(f"[XPBD] mesh: {len(vertices)} verts, {n_tets} tets, "
              f"origin={len(origin_ids)}, insertion={len(insertion_ids)}")

    # Bone targets: origin targets are FIXED (rest positions),
    # insertion targets follow MuJoCo (initially = rest positions).
    origin_targets = vertices[origin_ids].copy()
    insertion_targets = vertices[insertion_ids].copy()
    bone_targets_np = np.vstack([origin_targets, insertion_targets]).astype(np.float32)

    # --- Build XPBD sim ---
    wp.init()
    sim = build_xpbd_muscle_sim(
        vertices, tets, fiber_dirs,
        origin_ids, insertion_ids, bone_targets_np,
        dts=dts, device=device,
        volume_stiffness=volume_stiffness, arap_stiffness=arap_stiffness,
        attach_origin_stiffness=attach_origin_k,
        attach_insertion_stiffness=attach_insertion_k,
        density=1060.0, veldamping=0.02,
    )

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

    # Mesh exporter (muscle)
    os.makedirs("output", exist_ok=True)
    exporter = MeshExporter(path="output/anim_xpbd", format="ply",
                            tet_indices=tet_idx, positions=vertices)

    # Bone capsule meshes (rest-pose, in body-local coords)
    # Humerus: fromto (0,0,0)→(0,-L_h,0) in world (body is at origin)
    L_h = geo["humerus_length"]
    L_r = geo["radius_length"]
    humerus_verts_local, humerus_faces = create_capsule_mesh(
        [0, 0, 0], [0, -L_h, 0], radius=0.04, n_circ=12, n_axial=6)
    # Radius: fromto (0,0,0)→(0,-L_r,0) in body-local coords
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

            # a) Update insertion bone targets from MuJoCo
            mujoco.mj_forward(mj_model, mj_data)
            new_ins = mj_data.site_xpos[insertion_sid].copy()
            delta = (new_ins - insertion_pos).astype(np.float32)
            new_ins_targets = insertion_targets + delta
            # Rebuild full bone array (origin fixed + insertion updated)
            bone_arr = np.vstack([origin_targets, new_ins_targets]).astype(np.float32)
            sim.bone_pos_field = wp.from_numpy(bone_arr, dtype=wp.vec3)

            # b) One XPBD step
            sim.update_attach_targets()
            sim.integrate(); sim.clear(); sim.clear_reaction()
            sim.solve_constraints(); sim.update_velocities()

            # c) MuJoCo substeps
            for _ in range(mj_per_xpbd):
                t_start = act_cfg["excitation_start_time"]
                t_end = act_cfg["excitation_end_time"]
                e_off = act_cfg["excitation_off"]
                if t_now < t_start:
                    exc = e_off
                elif t_now >= t_end:
                    exc = act_cfg["excitation_on"]
                else:
                    f_ = (t_now - t_start) / (t_end - t_start)
                    f_ = f_ * f_ * (3.0 - 2.0 * f_)
                    exc = e_off + (act_cfg["excitation_on"] - e_off) * f_

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
            from VMuscle.mesh_utils import MeshDistortionError
            raise MeshDistortionError(
                f"step={step}: {n_inv}/{n_tets} tets inverted")

        exporter.save_frame(pos_np.astype(np.float32), step)

        # Export bone capsules at current MuJoCo pose
        from VMuscle.mesh_io import save_ply
        # Humerus is fixed (body 1, identity transform)
        h_verts = humerus_verts_local  # no transform needed (fixed body)
        save_ply(os.path.join(bone_anim_dir, f"humerus_{step:04d}.ply"),
                 h_verts, humerus_faces)
        # Radius rotates with elbow (body 2)
        r_pos = mj_data.xpos[2]
        r_quat = mj_data.xquat[2]
        r_verts = transform_capsule(radius_verts_local, r_pos, r_quat)
        save_ply(os.path.join(bone_anim_dir, f"radius_{step:04d}.ply"),
                 r_verts.astype(np.float32), radius_faces)
        # Tendon: thin capsule from muscle mesh bottom to MuJoCo insertion site.
        # The muscle mesh covers the fiber portion (origin → mesh_end).
        # The tendon bridges mesh_end → insertion.
        mesh_bottom_center = pos_np[insertion_ids].mean(axis=0)
        cur_insertion = mj_data.site_xpos[insertion_sid].copy()
        t_verts, t_faces = create_capsule_mesh(
            mesh_bottom_center, cur_insertion, radius=0.005,
            n_circ=6, n_axial=4, n_cap=2)
        save_ply(os.path.join(bone_anim_dir, f"tendon_{step:04d}.ply"),
                 t_verts, t_faces)

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
    with open(sto_path, "w") as f:
        f.write("SimpleArm_XPBD_Coupled\ninDegrees=no\n"
                f"nColumns={len(cols)+1}\nnRows={len(times)}\n"
                "DataType=double\nversion=3\nendheader\n"
                "time\t" + "\t".join(cols) + "\n")
        for i in range(len(times)):
            f.write(f"{times[i]}\t{elbow_angles[i]}\t{activations_out[i]}\t"
                    f"{forces_out[i]}\t{norm_fiber_lengths[i]}\n")
    if verbose:
        print(f"STO saved to {sto_path}")

    return {"times": np.array(times), "elbow_angles": np.array(elbow_angles),
            "forces": np.array(forces_out),
            "norm_fiber_lengths": np.array(norm_fiber_lengths),
            "activations": np.array(activations_out),
            "max_iso_force": F_max, "muscle_type": "XPBD_Coupled"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data/simpleArm/config.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = xpbd_coupled_simple_arm(cfg)
    if result:
        print(f"\nFinal elbow angle: {np.degrees(result['elbow_angles'][-1]):.2f}deg")


if __name__ == "__main__":
    main()
