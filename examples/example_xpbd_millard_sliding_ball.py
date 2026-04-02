"""XPBD-Millard sliding-ball example (quasi-static).

Same setup as the DGF sliding ball but uses Millard 2012 quintic Bezier
force-length curves instead of DGF 3-Gaussian curves.

Usage:
    RUN=example_xpbd_millard_sliding_ball uv run main.py
    uv run -m examples.example_xpbd_millard_sliding_ball
"""

import json
import os
import argparse
from types import SimpleNamespace

import numpy as np
import warp as wp

from VMuscle.activation import activation_dynamics_step_np
from VMuscle.mesh_utils import create_cylinder_tet_mesh, assign_fiber_directions
from VMuscle.millard_curves import MillardCurves
from VMuscle.muscle_warp import MuscleSim, fill_float_kernel


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
        "lambda_opt": mus.get("lambda_opt", 1.0),
        "excitation": act["excitation"],
        "act_substep_dt": act["substep_dt"],
        "dt": sol["dt"],
        "n_steps": sol["n_steps"],
        "num_substeps": xpbd.get("num_substeps", 10),
        "veldamping": xpbd.get("veldamping", 0.02),
        "fiber_stiffness_scale": xpbd.get("fiber_stiffness_scale", 100000.0),
        "dampingratio": xpbd.get("dampingratio", 0.1),
        "n_constraint_iters": xpbd.get("n_constraint_iters", 1),
        # Constraints read from JSON "constraints" array
        "constraints": raw.get("constraints", []),
    }


def _build_muscle_sim(cfg):
    """Build a MuscleSim with Millard fiber constraints from procedural cylinder mesh."""
    length = cfg["length"]
    radius = cfg["radius"]
    axis_idx = {"X": 0, "Y": 1, "Z": 2}[cfg["up_axis"]]

    vertices, tets = create_cylinder_tet_mesh(
        length, radius, cfg["n_circ"], cfg["n_axial"])
    # Fix tet winding
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
    print(f"Mesh: {n_v} verts, {n_tet} tets")

    # Per-vertex fiber directions
    v_fiber = np.zeros((n_v, 3), dtype=np.float32)
    v_count = np.zeros(n_v, dtype=np.float32)
    for e, tet in enumerate(tets):
        for vi in tet:
            v_fiber[vi] += fiber_dirs_per_tet[e]
            v_count[vi] += 1.0
    mask = v_count > 0
    v_fiber[mask] /= v_count[mask, None]
    norms = np.linalg.norm(v_fiber, axis=1, keepdims=True)
    v_fiber /= np.maximum(norms, 1e-8)

    # Build SimConfig — constraints from JSON config
    constraint_list = []
    for cc in cfg.get("constraints", []):
        entry = dict(cc)  # copy
        entry.setdefault("name", cc["type"])
        constraint_list.append(entry)
    if not constraint_list:
        raise ValueError("No constraints defined in config JSON")

    sim_cfg = SimpleNamespace(
        geo_path="<procedural>",
        bone_geo_path="<none>",
        gui=False,
        render_mode="none",
        constraints=constraint_list,
        dt=cfg["dt"],
        num_substeps=cfg["num_substeps"],
        gravity=cfg["gravity"],
        density=cfg["density"],
        veldamping=cfg["veldamping"],
        contraction_ratio=0.0,
        fiber_stiffness_scale=cfg["fiber_stiffness_scale"],
        sigma0=cfg["sigma0"],
        lambda_opt=cfg["lambda_opt"],
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

    # Manually construct MuscleSim
    wp.set_device("cpu")
    sim = object.__new__(MuscleSim)
    sim.cfg = sim_cfg
    sim.constraint_configs = sim_cfg.constraints
    sim.pos0_np = vertices.astype(np.float32)
    sim.tet_np = tets.astype(np.int32)
    sim.v_fiber_np = v_fiber.astype(np.float32)
    sim.v_tendonmask_np = None
    sim.geo = SimpleNamespace()
    sim.n_verts = n_v
    sim.bone_geo = None
    sim.bone_pos = np.zeros((0, 3), dtype=np.float32)
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
    sim.fiber_stiffness_scale = cfg["fiber_stiffness_scale"]
    sim.has_compressstiffness = False
    sim.dt = cfg["dt"] / cfg["num_substeps"]
    sim.step_cnt = 0
    sim.renderer = None

    sim.build_constraints()

    # Set gravity vector based on up axis
    g = cfg["gravity"]
    if axis_idx == 0:
        sim._gravity_vec = wp.vec3(-g, 0.0, 0.0)
    elif axis_idx == 1:
        sim._gravity_vec = wp.vec3(0.0, -g, 0.0)
    else:
        sim._gravity_vec = wp.vec3(0.0, 0.0, -g)

    # Fix top vertices and add ball mass to bottom
    mass_np = sim.mass.numpy()
    stopped_np = sim.stopped.numpy()
    bottom_ids = []
    for i in range(n_v):
        z = vertices[i, axis_idx]
        if z > length - 1e-4:
            stopped_np[i] = 1
        elif z < 1e-4:
            bottom_ids.append(i)

    ball_mass = cfg["ball_mass"]
    mass_per_bottom = ball_mass / max(len(bottom_ids), 1)
    for bid in bottom_ids:
        mass_np[bid] += mass_per_bottom
    print(f"Ball mass: {ball_mass} kg -> {len(bottom_ids)} bottom verts")

    sim.mass = wp.from_numpy(mass_np.astype(np.float32), dtype=wp.float32)
    sim.stopped = wp.from_numpy(stopped_np.astype(np.int32), dtype=wp.int32)

    # Explicit active force mode — log expected equilibrium values
    mc = MillardCurves()
    A_cross = np.pi * radius ** 2
    F_max = cfg["sigma0"] * A_cross
    norm_load = ball_mass * cfg["gravity"] / F_max
    print(f"Explicit active force mode: F_max={F_max:.1f}N, norm_load={norm_load:.4f}")
    print(f"  lambda_opt={cfg['lambda_opt']:.3f}")
    print(f"  Expected: f_L(lm_eq)={norm_load:.4f} → lm_eq≈0.546")

    return sim, bottom_ids, axis_idx, tets, vertices, fiber_dirs_per_tet, mc


def compute_fiber_data(pos, tet_idx, rest_matrices, fiber_dirs,
                       activation, mc):
    """Compute per-tet fiber stretches and mean normalized forces using Millard curves."""
    n = len(tet_idx)
    stretches = np.empty(n)
    for e in range(n):
        i0, i1, i2, i3 = tet_idx[e]
        Ds = np.column_stack([pos[i0] - pos[i3], pos[i1] - pos[i3], pos[i2] - pos[i3]])
        F = Ds @ rest_matrices[e]
        Fd = F @ fiber_dirs[e]
        stretches[e] = max(np.linalg.norm(Fd), 1e-8)

    l_mean = float(stretches.mean())
    fl_vals = mc.fl.eval(stretches)
    fpe_vals = mc.fpe.eval(stretches)
    f_active = float(np.mean(activation * fl_vals))
    f_passive = float(np.mean(fpe_vals))
    return {
        'f_active': f_active,
        'f_passive': f_passive,
        'f_total': f_active + f_passive,
        'l_mean': l_mean,
    }


def run_sim(cfg, label="default"):
    """Run one XPBD-Millard sliding-ball simulation."""
    sim, bottom_ids, axis_idx, tets, vertices, fiber_dirs_per_tet, mc = \
        _build_muscle_sim(cfg)

    dt = cfg["dt"]
    n_steps = cfg["n_steps"]
    n_tet = len(tets)

    # Pre-compute rest matrices
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

    rec_t, rec_z, rec_a, rec_fiber = [], [], [], []

    print(f"\nSimulating {n_steps * dt:.1f}s (Millard, "
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

        wp.launch(fill_float_kernel, dim=n_tet,
                  inputs=[sim.activation, wp.float32(a)])

        # Explicit active force: no CPU equilibrium solve needed

        # Step XPBD (active force explicit + passive constraints)
        n_iters = cfg.get("n_constraint_iters", 1)
        sim.update_attach_targets()
        for _ in range(sim.cfg.num_substeps):
            sim.clear_forces()
            sim.accumulate_active_fiber_force()
            sim.integrate()
            sim.clear()
            for _ in range(n_iters):
                sim.solve_constraints()
            sim.update_velocities()

        pos = sim.pos.numpy()
        bottom_z = float(np.mean([pos[bid][axis_idx] for bid in bottom_ids]))
        fd = compute_fiber_data(pos, tets, rest_matrices, fiber_dirs_per_tet, a, mc)

        rec_t.append(t + dt)
        rec_z.append(bottom_z)
        rec_a.append(a)
        rec_fiber.append(fd)

        if step % 50 == 0 or step == n_steps - 1:
            print(f"  step={step:4d}  t={t:.3f}s  a={a:.2f}  "
                  f"bottom_z={bottom_z:.6f}  l~={fd['l_mean']:.4f}")

    print("Done.")

    os.makedirs("output", exist_ok=True)
    out = f"output/xpbd_millard_sliding_ball_{label}.npz"
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
    parser = argparse.ArgumentParser(description="XPBD-Millard sliding-ball muscle example")
    parser.add_argument("--config", default="data/slidingBall/config_xpbd_millard.json")
    parser.add_argument("--label", default="default")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_sim(cfg, label=args.label)


if __name__ == "__main__":
    main()
