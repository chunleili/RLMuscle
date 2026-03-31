"""XPBD-DGF sliding-ball example (quasi-static).

Generates a cylinder tet mesh with a concentrated ball mass at the bottom,
fixes the top vertices, applies activation ramp, and simulates contraction
with the XPBD solver using DGF (DeGroote-Fregly) constitutive model.

Outputs NPZ + prints for comparison with the VBD version.

Usage:
    RUN=example_xpbd_dgf_sliding_ball uv run main.py
    uv run -m examples.example_xpbd_dgf_sliding_ball
    uv run -m examples.example_xpbd_dgf_sliding_ball --config data/slidingBall/config_xpbd_dgf.json
"""

import json
import os
import argparse
from types import SimpleNamespace

import numpy as np
import warp as wp

from VMuscle.activation import activation_dynamics_step_np
from VMuscle.dgf_curves import active_force_length, compute_fiber_forces
from VMuscle.mesh_io import build_surface_tris
from VMuscle.mesh_utils import create_cylinder_tet_mesh, assign_fiber_directions
from VMuscle.muscle_warp import (MuscleSim, fill_float_kernel,
                                   update_cons_restdir1_kernel)


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
    # Find closest match
    idx = np.argmin(np.abs(fl - target_fl))
    return float(lm_range[idx])


def load_config(path):
    """Load config JSON and return a flat dict of parameters."""
    with open(path) as f:
        raw = json.load(f)
    geo = raw["geometry"]
    phys = raw["physics"]
    mus = raw["muscle"]
    act = raw["activation"]
    sol = raw["solver"]
    xpbd = raw.get("xpbd", {})
    return {
        "length": geo["muscle_length"],
        "radius": geo["muscle_radius"],
        "n_circ": geo["n_circumferential"],
        "n_axial": geo["n_axial"],
        "up_axis": geo["up_axis"],
        "density": phys["density"],
        "gravity": phys.get("gravity", 9.81),
        "ball_mass": phys["ball_mass"],
        "sigma0": mus["sigma0"],
        "excitation": act["excitation"],
        "act_substep_dt": act["substep_dt"],
        "dt": sol["dt"],
        "n_steps": sol["n_steps"],
        "num_substeps": xpbd.get("num_substeps", 10),
        "veldamping": xpbd.get("veldamping", 0.02),
        "fiber_stiffness_scale": xpbd.get("fiber_stiffness_scale", 100000.0),
        "volume_stiffness": xpbd.get("volume_stiffness", 1e10),
        "fiber_stiffness": xpbd.get("fiber_stiffness", 1000.0),
        "arap_stiffness": xpbd.get("arap_stiffness", 1e10),
        "dampingratio": xpbd.get("dampingratio", 0.1),
        "contraction_factor": xpbd.get("contraction_factor", 0.4),
        "optimal_fiber_length": xpbd.get("optimal_fiber_length", 1.0),
        "snh_mu": xpbd.get("snh_mu", 0.0),
        "snh_lam": xpbd.get("snh_lam", 10000.0),
        "snh_dampingratio": xpbd.get("snh_dampingratio", 0.01),
        "n_constraint_iters": xpbd.get("n_constraint_iters", 1),
    }


def _build_muscle_sim(cfg):
    """Build a MuscleSim from procedural cylinder mesh (bypass USD loading)."""
    length = cfg["length"]
    radius = cfg["radius"]
    axis_idx = {"X": 0, "Y": 1, "Z": 2}[cfg["up_axis"]]

    vertices, tets = create_cylinder_tet_mesh(
        length, radius, cfg["n_circ"], cfg["n_axial"])
    # Fix tet winding for XPBD convention: det([p0-p3, p1-p3, p2-p3]) > 0
    tet_pos = vertices[tets]
    cols = tet_pos[:, :3, :] - tet_pos[:, 3:4, :]
    M = np.transpose(cols, (0, 2, 1))
    dets = np.linalg.det(M)
    inverted = dets < 0
    if np.any(inverted):
        tets[inverted, 2], tets[inverted, 3] = (
            tets[inverted, 3].copy(), tets[inverted, 2].copy())
    fiber_dirs_per_tet = assign_fiber_directions(vertices, tets, axis=axis_idx)
    n_v = len(vertices)
    n_tet = len(tets)
    print(f"Mesh: {n_v} verts, {n_tet} tets (fixed {inverted.sum()} inverted tets)")

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

    # Build a minimal SimConfig
    sim_cfg = SimpleNamespace(
        geo_path="<procedural>",
        bone_geo_path="<none>",
        gui=False,
        render_mode="none",
        constraints=[
            {"type": "fiberdgf", "name": "fiber_dgf",
             "stiffness": cfg["fiber_stiffness"],
             "dampingratio": cfg["dampingratio"],
             "sigma0": cfg["sigma0"],
             "contraction_factor": cfg["contraction_factor"]},
        ] + ([{"type": "snh", "name": "snh",
               "mu": cfg["snh_mu"],
               "lam": cfg["snh_lam"],
               "dampingratio": cfg["snh_dampingratio"]}]
             if cfg.get("snh_mu", 0.0) > 0.0 else []),
        dt=cfg["dt"],
        num_substeps=cfg["num_substeps"],
        gravity=cfg["gravity"],
        density=cfg["density"],
        veldamping=cfg["veldamping"],
        contraction_ratio=cfg["contraction_factor"],
        fiber_stiffness_scale=cfg["fiber_stiffness_scale"],
        HAS_compressstiffness=False,
        arch="cpu",
        save_image=False,
        pause=False,
        reset=False,
        show_auxiliary_meshes=False,
        show_wireframe=False,
        render_fps=24,
        color_bones=False,
        color_muscles="tendonmask",
        activation=0.0,
        nsteps=cfg["n_steps"],
    )

    # Manually construct MuscleSim bypassing file-based __init__
    wp.set_device("cpu")
    sim = object.__new__(MuscleSim)
    sim.cfg = sim_cfg
    sim.constraint_configs = sim_cfg.constraints

    # Set mesh data directly
    sim.pos0_np = vertices.astype(np.float32)
    sim.tet_np = tets.astype(np.int32)
    sim.v_fiber_np = v_fiber.astype(np.float32)
    sim.v_tendonmask_np = None  # no tendon for cylinder
    sim.geo = SimpleNamespace()  # empty geo stub
    sim.n_verts = n_v

    # Bone data (none for this example)
    sim.bone_geo = None
    sim.bone_pos = np.zeros((0, 3), dtype=np.float32)
    sim.bone_indices_np = np.zeros(0, dtype=np.int32)
    sim.bone_muscle_ids = {}

    # Initialize backend
    wp.init()
    sim._init_backend()
    sim._allocate_fields()
    sim._init_fields()
    sim._precompute_rest()
    sim._build_surface_tris()
    sim._create_bone_fields()

    sim.use_jacobi = False
    sim.use_colored_gs = False
    sim.contraction_ratio = cfg["contraction_factor"]
    sim.fiber_stiffness_scale = cfg["fiber_stiffness_scale"]
    sim.has_compressstiffness = False
    sim.dt = cfg["dt"] / cfg["num_substeps"]
    sim.step_cnt = 0
    sim.renderer = None

    sim.build_constraints()

    # Fix top vertices (z ≈ length) and add ball mass to bottom
    mass_np = sim.mass.numpy()
    stopped_np = sim.stopped.numpy()
    bottom_ids = []
    for i in range(n_v):
        z = vertices[i, axis_idx]
        if z > length - 1e-4:
            stopped_np[i] = 1  # kinematic: fixed
        elif z < 1e-4:
            bottom_ids.append(i)

    ball_mass = cfg["ball_mass"]
    mass_per_bottom = ball_mass / max(len(bottom_ids), 1)
    for bid in bottom_ids:
        mass_np[bid] += mass_per_bottom
    print(f"Ball mass: {ball_mass} kg distributed to {len(bottom_ids)} bottom verts")

    sim.mass = wp.from_numpy(mass_np.astype(np.float32), dtype=wp.float32)
    sim.stopped = wp.from_numpy(stopped_np.astype(np.int32), dtype=wp.int32)

    # Compute DGF equilibrium for initial contraction_factor
    A_cross = np.pi * radius ** 2
    F_max = cfg["sigma0"] * A_cross
    norm_load = ball_mass * cfg["gravity"] / F_max
    lm_eq = dgf_equilibrium_fiber_length(1.0, norm_load)
    cf_dgf = 1.0 - lm_eq
    print(f"DGF equilibrium: lm_eq={lm_eq:.4f}, cf={cf_dgf:.4f}, "
          f"F_max={F_max:.1f}N, norm_load={norm_load:.4f}")

    return sim, bottom_ids, axis_idx, tets, vertices, fiber_dirs_per_tet


def compute_fiber_data(pos, tet_idx, rest_matrices, fiber_dirs,
                       activation, dt, l_prev_mean=None):
    """Compute per-tet fiber stretches and mean normalized forces."""
    n = len(tet_idx)
    stretches = np.empty(n)
    for e in range(n):
        i0, i1, i2, i3 = tet_idx[e]
        Ds = np.column_stack([pos[i0] - pos[i3], pos[i1] - pos[i3], pos[i2] - pos[i3]])
        F = Ds @ rest_matrices[e]
        Fd = F @ fiber_dirs[e]
        stretches[e] = max(np.linalg.norm(Fd), 1e-8)

    l_mean = float(stretches.mean())
    v_norm = 0.0  # quasi-static
    fd = compute_fiber_forces(stretches, activation, v_norm, include_passive=False)
    fd['l_mean'] = l_mean
    return fd


def run_sim(cfg, label="default"):
    """Run one XPBD-DGF sliding-ball simulation."""
    sim, bottom_ids, axis_idx, tets, vertices, fiber_dirs_per_tet = _build_muscle_sim(cfg)

    dt = cfg["dt"]
    n_steps = cfg["n_steps"]
    n_tet = len(tets)

    # Pre-compute rest matrices on CPU for fiber analysis
    rest_matrices = np.zeros((n_tet, 3, 3), dtype=np.float32)
    for e in range(n_tet):
        i0, i1, i2, i3 = tets[e]
        M = np.column_stack([
            vertices[i0] - vertices[i3],
            vertices[i1] - vertices[i3],
            vertices[i2] - vertices[i3]])
        det = np.linalg.det(M)
        if abs(det) > 1e-30:
            rest_matrices[e] = np.linalg.inv(M)

    # Activation dynamics state
    act_arr = np.full(n_tet, 0.01, dtype=np.float32)
    act_sub_dt = cfg["act_substep_dt"]
    excitation_val = cfg["excitation"]

    # Normalized load for DGF equilibrium computation
    A_cross = np.pi * cfg["radius"] ** 2
    F_max = cfg["sigma0"] * A_cross
    norm_load = cfg["ball_mass"] * cfg["gravity"] / F_max

    # Recording
    rec_t, rec_z, rec_a = [], [], []
    rec_fiber = []
    l_prev_mean = None

    print(f"\nSimulating {n_steps * dt:.1f}s (top fixed, "
          f"{len(bottom_ids)} bottom verts, ball={cfg['ball_mass']}kg)...")

    for step in range(n_steps):
        t = step * dt

        # Excitation → activation dynamics
        exc = np.full(n_tet, excitation_val, dtype=np.float32)
        n_sub = max(1, int(np.ceil(dt / act_sub_dt)))
        sub_dt = dt / n_sub
        for _ in range(n_sub):
            act_arr = activation_dynamics_step_np(exc, act_arr, sub_dt)
        a = float(act_arr.mean())

        # Set per-tet activation in XPBD sim
        wp.launch(fill_float_kernel, dim=n_tet,
                  inputs=[sim.activation, wp.float32(a)])

        # Compute DGF equilibrium for current activation → contraction_factor
        lm_eq_a = dgf_equilibrium_fiber_length(a, norm_load)
        cf_a = 1.0 - lm_eq_a
        # Update contraction_factor (restdir[1]) for all DGF constraints
        from VMuscle.constraints import TETFIBERDGF
        if hasattr(sim, 'cons_ranges') and TETFIBERDGF in sim.cons_ranges:
            off, cnt = sim.cons_ranges[TETFIBERDGF]
            wp.launch(update_cons_restdir1_kernel, dim=cnt,
                      inputs=[sim.cons, cf_a, TETFIBERDGF, off, cnt])

        # Step XPBD
        n_iters = cfg.get("n_constraint_iters", 1)
        sim.update_attach_targets()
        for _ in range(sim.cfg.num_substeps):
            sim.integrate()
            sim.clear()
            for _ in range(n_iters):
                sim.solve_constraints()
            sim.update_velocities()

        # Read back positions
        pos = sim.pos.numpy()
        bottom_z = float(np.mean([pos[bid][axis_idx] for bid in bottom_ids]))

        fd = compute_fiber_data(pos, tets, rest_matrices, fiber_dirs_per_tet,
                                a, dt, l_prev_mean)
        l_prev_mean = fd['l_mean']

        rec_t.append(t + dt)
        rec_z.append(bottom_z)
        rec_a.append(a)
        rec_fiber.append(fd)

        if step % 50 == 0 or step == n_steps - 1:
            print(f"  step={step:4d}  t={t:.3f}s  a={a:.2f}  "
                  f"bottom_z={bottom_z:.6f}  l~={fd['l_mean']:.4f}")

    print("Done.")

    # Save NPZ
    os.makedirs("output", exist_ok=True)
    out = f"output/xpbd_dgf_sliding_ball_{label}.npz"
    np.savez(out,
             times=np.array(rec_t),
             positions=np.array(rec_z),
             activations=np.array(rec_a),
             norm_fiber_lengths=np.array([d['l_mean'] for d in rec_fiber]),
             f_active=np.array([d['f_active'] for d in rec_fiber]),
             f_passive=np.array([d['f_passive'] for d in rec_fiber]),
             f_total=np.array([d['f_total'] for d in rec_fiber]),
             sigma0=cfg["sigma0"], radius=cfg["radius"],
             muscle_length=cfg["length"],
             ball_mass=cfg["ball_mass"], dt=dt)
    print(f"NPZ saved to {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description="XPBD-DGF sliding-ball muscle example")
    parser.add_argument("--config", default="data/slidingBall/config_xpbd_dgf.json",
                        help="Path to config JSON")
    parser.add_argument("--label", default="default")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_sim(cfg, label=args.label)


if __name__ == "__main__":
    main()
