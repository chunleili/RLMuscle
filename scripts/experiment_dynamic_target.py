"""Experiment: Dynamic target force injection with mass ratio mitigations.

Tests three mitigations for the XPBD mass imbalance issue:
1. Higher mesh density (increases internal vertex mass)
2. Ball mass distributed to ALL vertices (not just bottom layer)
3. More constraint iterations per substep

Reference (Stage 4, equilibrium-based):
  lambda_eq = 0.5498, ball_z = 0.0449

Usage:
    uv run python scripts/experiment_dynamic_target.py
"""

import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import warp as wp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from VMuscle.activation import activation_dynamics_step_np
from VMuscle.mesh_utils import create_cylinder_tet_mesh, assign_fiber_directions
from VMuscle.millard_curves import MillardCurves
from VMuscle.muscle_warp import MuscleSim, fill_float_kernel


def run_experiment(
    density=1060.0,
    ball_mass_mode="bottom_only",  # "bottom_only", "all_verts", "bottom_layers"
    n_constraint_iters=1,
    n_steps=300,
    label="default",
):
    """Run sliding ball with dynamic target kernel variant.

    Instead of modifying the kernel, we use the existing equilibrium-based kernel
    but set contraction_factor dynamically to simulate the dynamic target effect.
    This avoids Warp recompilation issues.

    Actually, we DO need the dynamic target kernel. Let's modify restdir[1] each step
    to be a "dynamic contraction factor" computed from the current mesh state.
    """
    cfg_path = "data/slidingBall/config_xpbd_millard.json"
    with open(cfg_path) as f:
        raw = json.load(f)

    length = raw["geometry"]["muscle_length"]
    radius = raw["geometry"]["muscle_radius"]
    n_circ = raw["geometry"]["n_circumferential"]
    n_axial = raw["geometry"]["n_axial"]
    axis_idx = {"X": 0, "Y": 1, "Z": 2}[raw["geometry"]["up_axis"]]
    sigma0 = raw["muscle"]["sigma0"]
    ball_mass = raw["physics"]["ball_mass"]
    gravity = raw["physics"].get("gravity", 9.81)
    dt = raw["solver"]["dt"]
    num_substeps = raw["xpbd"].get("num_substeps", 40)
    fiber_stiffness_scale = raw["xpbd"].get("fiber_stiffness_scale", 100000.0)
    fiber_stiffness = raw["xpbd"].get("fiber_stiffness", 1000.0)
    dampingratio = raw["xpbd"].get("dampingratio", 0.018)
    contraction_factor = raw["xpbd"].get("contraction_factor", 0.41)
    veldamping = raw["xpbd"].get("veldamping", 0.003)

    # Build mesh
    vertices, tets = create_cylinder_tet_mesh(length, radius, n_circ, n_axial)
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

    # Build sim
    sim_cfg = SimpleNamespace(
        geo_path="<procedural>",
        bone_geo_path="<none>",
        gui=False, render_mode="none",
        constraints=[
            {"type": "fibermillard", "name": "fiber_millard",
             "stiffness": fiber_stiffness, "dampingratio": dampingratio,
             "sigma0": sigma0, "contraction_factor": contraction_factor},
        ],
        dt=dt, num_substeps=num_substeps,
        gravity=gravity, density=density,  # <-- variable density
        veldamping=veldamping,
        contraction_ratio=contraction_factor,
        fiber_stiffness_scale=fiber_stiffness_scale,
        HAS_compressstiffness=False, arch="cpu",
        save_image=False, pause=False, reset=False,
        show_auxiliary_meshes=False, show_wireframe=False,
        render_fps=24, color_bones=False, color_muscles="tendonmask",
        activation=0.0, nsteps=n_steps,
    )

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
    sim.contraction_ratio = contraction_factor
    sim.fiber_stiffness_scale = fiber_stiffness_scale
    sim.has_compressstiffness = False
    sim.dt = dt / num_substeps
    sim.step_cnt = 0
    sim.renderer = None

    sim.build_constraints()

    # --- Mass setup ---
    mass_np = sim.mass.numpy()
    stopped_np = sim.stopped.numpy()

    # Identify vertex layers
    top_ids = []
    bottom_ids = []
    mid_ids = []
    for i in range(n_v):
        z = vertices[i, axis_idx]
        if z > length - 1e-4:
            stopped_np[i] = 1
            top_ids.append(i)
        elif z < 1e-4:
            bottom_ids.append(i)
        else:
            mid_ids.append(i)

    # Ball mass distribution
    if ball_mass_mode == "bottom_only":
        # Original: only bottom layer
        mass_per = ball_mass / max(len(bottom_ids), 1)
        for bid in bottom_ids:
            mass_np[bid] += mass_per
    elif ball_mass_mode == "all_verts":
        # Distribute to ALL non-stopped vertices
        free_ids = [i for i in range(n_v) if stopped_np[i] == 0]
        mass_per = ball_mass / max(len(free_ids), 1)
        for fid in free_ids:
            mass_np[fid] += mass_per
    elif ball_mass_mode == "bottom_layers":
        # Distribute to bottom half of mesh
        half_z = length / 2.0
        bottom_half = [i for i in range(n_v) if vertices[i, axis_idx] < half_z and stopped_np[i] == 0]
        mass_per = ball_mass / max(len(bottom_half), 1)
        for bid in bottom_half:
            mass_np[bid] += mass_per

    sim.mass = wp.from_numpy(mass_np.astype(np.float32), dtype=wp.float32)
    sim.stopped = wp.from_numpy(stopped_np.astype(np.int32), dtype=wp.int32)

    # Print mass stats
    free_mass = mass_np[stopped_np == 0]
    print(f"  Mass stats: min={free_mass.min():.6f} max={free_mass.max():.6f} "
          f"ratio={free_mass.max()/free_mass.min():.1f}x "
          f"mean={free_mass.mean():.6f}")

    # Millard curves for fiber data computation
    mc = MillardCurves()

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

    # Dynamic target: instead of CPU equilibrium, we set contraction_factor
    # to a value that makes target = lm_tilde - sigma0*f_total/k_base
    # But since we can't change the kernel, we simulate this by computing
    # the "equivalent contraction_factor" each step:
    #   target = 1 - a * cf  (kernel formula)
    #   target = lm_tilde - sigma0 * f_total / k_base  (desired)
    # So: cf = (1 - target) / a = (1 - lm_tilde + sigma0*f_total/k_base) / a
    #
    # But this requires knowing lm_tilde (current fiber stretch).
    # We can get it from the mesh after each step.
    #
    # This simulates the dynamic target WITHOUT modifying the kernel.

    from VMuscle.constraints import TETFIBERMILLARD
    from VMuscle.muscle_warp import update_cons_restdir1_kernel

    # Activation
    act_arr = np.full(n_tet, 0.01, dtype=np.float32)
    act_sub_dt = raw["activation"]["substep_dt"]
    excitation_val = raw["activation"]["excitation"]

    # For dynamic target, k_base controls the balance
    # We try to match the equilibrium-based approach's steady state
    k_base = sigma0  # = 300000, makes C = f_total (dimensionless, ~1.0 max)

    rec_t, rec_z, rec_lm = [], [], []
    t0 = time.time()

    for step in range(n_steps):
        t = step * dt

        # Activation dynamics
        exc = np.full(n_tet, excitation_val, dtype=np.float32)
        n_sub = max(1, int(np.ceil(dt / act_sub_dt)))
        sub_dt = dt / n_sub
        for _ in range(n_sub):
            act_arr = activation_dynamics_step_np(exc, act_arr, sub_dt)
        a = float(act_arr.mean())

        wp.launch(fill_float_kernel, dim=n_tet,
                  inputs=[sim.activation, wp.float32(a)])

        # Dynamic target: compute cf from current mesh state
        pos_np = sim.pos.numpy()
        stretches = np.empty(n_tet)
        for e in range(n_tet):
            i0, i1, i2, i3 = tets[e]
            Ds = np.column_stack([pos_np[i0] - pos_np[i3],
                                  pos_np[i1] - pos_np[i3],
                                  pos_np[i2] - pos_np[i3]])
            Fd = (Ds @ rest_matrices[e]) @ fiber_dirs_per_tet[e]
            stretches[e] = max(np.linalg.norm(Fd), 1e-8)
        lm_mean = float(stretches.mean())

        # Evaluate Millard at current stretch
        fl_val = float(mc.fl.eval(lm_mean))
        fpe_val = float(mc.fpe.eval(lm_mean))
        f_total = a * fl_val + fpe_val

        # Equivalent contraction factor for dynamic target
        # target = lm_mean - sigma0 * f_total / k_base
        # From kernel: target = 1 - a * cf
        # So: cf = (1 - lm_mean + sigma0 * f_total / k_base) / max(a, 0.01)
        target_dynamic = lm_mean - sigma0 * f_total / k_base
        target_dynamic = np.clip(target_dynamic, 0.1, 3.0)
        cf_dynamic = (1.0 - target_dynamic) / max(a, 0.01)
        cf_dynamic = float(np.clip(cf_dynamic, -2.0, 2.0))

        if hasattr(sim, 'cons_ranges') and TETFIBERMILLARD in sim.cons_ranges:
            off, cnt = sim.cons_ranges[TETFIBERMILLARD]
            wp.launch(update_cons_restdir1_kernel, dim=cnt,
                      inputs=[sim.cons, cf_dynamic, TETFIBERMILLARD, off, cnt])

        # XPBD step
        sim.update_attach_targets()
        for _ in range(sim.cfg.num_substeps):
            sim.integrate()
            sim.clear()
            for _ in range(n_constraint_iters):
                sim.solve_constraints()
            sim.update_velocities()

        pos = sim.pos.numpy()
        bottom_z = float(np.mean([pos[bid][axis_idx] for bid in bottom_ids]))

        rec_t.append(t + dt)
        rec_z.append(bottom_z)
        rec_lm.append(lm_mean)

        if step % 50 == 0 or step == n_steps - 1:
            print(f"  step={step:4d} t={t:.3f}s a={a:.2f} "
                  f"z={bottom_z:.6f} l~={lm_mean:.4f} cf={cf_dynamic:.4f}")

    elapsed = time.time() - t0
    final_lm = rec_lm[-1]
    final_z = rec_z[-1]
    print(f"  => l~={final_lm:.4f} z={final_z:.6f} ({elapsed:.1f}s)")
    return final_lm, final_z


def main():
    print("=" * 70)
    print("Dynamic Target Experiment: Mass Ratio Mitigations")
    print("Reference: l~=0.5498, z=0.0449 (equilibrium-based)")
    print("=" * 70)

    results = {}

    # Baseline: original settings (should fail like before)
    print("\n--- Baseline: density=1060, bottom_only, iters=1 ---")
    results["baseline"] = run_experiment(
        density=1060, ball_mass_mode="bottom_only", n_constraint_iters=1,
        n_steps=200, label="baseline")

    # Mitigation 1: Higher density (10x)
    print("\n--- High density: density=10600, bottom_only, iters=1 ---")
    results["high_density"] = run_experiment(
        density=10600, ball_mass_mode="bottom_only", n_constraint_iters=1,
        n_steps=200, label="high_density")

    # Mitigation 2: Ball mass to all vertices
    print("\n--- All-vert mass: density=1060, all_verts, iters=1 ---")
    results["all_verts"] = run_experiment(
        density=1060, ball_mass_mode="all_verts", n_constraint_iters=1,
        n_steps=200, label="all_verts")

    # Mitigation 3: More iterations
    print("\n--- More iters: density=1060, bottom_only, iters=5 ---")
    results["more_iters"] = run_experiment(
        density=1060, ball_mass_mode="bottom_only", n_constraint_iters=5,
        n_steps=200, label="more_iters")

    # Combined: all mitigations
    print("\n--- Combined: density=10600, all_verts, iters=5 ---")
    results["combined"] = run_experiment(
        density=10600, ball_mass_mode="all_verts", n_constraint_iters=5,
        n_steps=200, label="combined")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"{'Config':<20} {'l~':>8} {'z':>10} {'l~ err':>8} {'z err':>8}")
    print("-" * 60)
    ref_lm, ref_z = 0.5498, 0.0449
    for name, (lm, z) in results.items():
        lm_err = abs(lm - ref_lm) / ref_lm * 100
        z_err = abs(z - ref_z) / ref_z * 100
        print(f"{name:<20} {lm:8.4f} {z:10.6f} {lm_err:7.1f}% {z_err:7.1f}%")
    print(f"{'reference':<20} {ref_lm:8.4f} {ref_z:10.6f} {'0.0':>7}% {'0.0':>7}%")


if __name__ == "__main__":
    main()
