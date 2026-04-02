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

import os
import argparse

import numpy as np
import warp as wp

from VMuscle.activation import activation_dynamics_step_np
from VMuscle.dgf_curves import active_force_length, compute_fiber_forces
from VMuscle.muscle_warp import fill_float_kernel, update_cons_restdir1_kernel
from VMuscle.sliding_ball_helpers import (
    build_xpbd_sliding_ball_sim,
    compute_rest_matrices,
    load_sliding_ball_config_base,
)


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
    cfg, raw = load_sliding_ball_config_base(path)
    xpbd = raw.get("xpbd", {})
    cfg.update({
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
    })
    return cfg


def _build_constraint_configs(cfg):
    """Build DGF-specific constraint configs from flat config dict."""
    configs = [
        {"type": "fiberdgf", "name": "fiber_dgf",
         "stiffness": cfg["fiber_stiffness"],
         "dampingratio": cfg["dampingratio"],
         "sigma0": cfg["sigma0"],
         "contraction_factor": cfg["contraction_factor"]},
    ]
    if cfg.get("snh_mu", 0.0) > 0.0:
        configs.append({"type": "snh", "name": "snh",
                        "mu": cfg["snh_mu"],
                        "lam": cfg["snh_lam"],
                        "dampingratio": cfg["snh_dampingratio"]})
    return configs


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
    constraint_configs = _build_constraint_configs(cfg)
    sim, bottom_ids, axis_idx, tets, vertices, fiber_dirs_per_tet = \
        build_xpbd_sliding_ball_sim(
            cfg, constraint_configs,
            contraction_ratio=cfg["contraction_factor"])

    # Log DGF equilibrium for initial contraction_factor
    A_cross = np.pi * cfg["radius"] ** 2
    F_max = cfg["sigma0"] * A_cross
    norm_load = cfg["ball_mass"] * cfg["gravity"] / F_max
    lm_eq = dgf_equilibrium_fiber_length(1.0, norm_load)
    cf_dgf = 1.0 - lm_eq
    print(f"DGF equilibrium: lm_eq={lm_eq:.4f}, cf={cf_dgf:.4f}, "
          f"F_max={F_max:.1f}N, norm_load={norm_load:.4f}")

    dt = cfg["dt"]
    n_steps = cfg["n_steps"]
    n_tet = len(tets)
    rest_matrices = compute_rest_matrices(vertices, tets)

    # Activation dynamics state
    act_arr = np.full(n_tet, 0.01, dtype=np.float32)
    act_sub_dt = cfg["act_substep_dt"]
    excitation_val = cfg["excitation"]

    rec_t, rec_z, rec_a, rec_fiber = [], [], [], []
    l_prev_mean = None

    print(f"\nSimulating {n_steps * dt:.1f}s (top fixed, "
          f"{len(bottom_ids)} bottom verts, ball={cfg['ball_mass']}kg)...")

    for step in range(n_steps):
        t = step * dt

        # Excitation -> activation dynamics
        exc = np.full(n_tet, excitation_val, dtype=np.float32)
        n_sub = max(1, int(np.ceil(dt / act_sub_dt)))
        sub_dt = dt / n_sub
        for _ in range(n_sub):
            act_arr = activation_dynamics_step_np(exc, act_arr, sub_dt)
        a = float(act_arr.mean())

        wp.launch(fill_float_kernel, dim=n_tet,
                  inputs=[sim.activation, wp.float32(a)])

        # Update DGF equilibrium contraction_factor for current activation
        lm_eq_a = dgf_equilibrium_fiber_length(a, norm_load)
        cf_a = 1.0 - lm_eq_a
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
