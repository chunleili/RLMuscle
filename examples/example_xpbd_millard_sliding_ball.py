"""XPBD-Millard sliding-ball example (quasi-static).

Same setup as the DGF sliding ball but uses Millard 2012 quintic Bezier
force-length curves instead of DGF 3-Gaussian curves.

Usage:
    RUN=example_xpbd_millard_sliding_ball uv run main.py
    uv run -m examples.example_xpbd_millard_sliding_ball
"""

import os
import argparse

import numpy as np
import warp as wp

from VMuscle.activation import activation_dynamics_step_np
from VMuscle.millard_curves import MillardCurves
from VMuscle.muscle_warp import fill_float_kernel
from VMuscle.sliding_ball_helpers import (
    build_xpbd_sliding_ball_sim,
    compute_rest_matrices,
    load_sliding_ball_config_base,
)


def load_config(path):
    """Load config JSON and return a flat dict of parameters."""
    cfg, raw = load_sliding_ball_config_base(path)
    mus = raw["muscle"]
    xpbd = raw.get("xpbd", {})
    cfg.update({
        "lambda_opt": mus.get("lambda_opt", 1.0),
        "num_substeps": xpbd.get("num_substeps", 10),
        "veldamping": xpbd.get("veldamping", 0.02),
        "fiber_stiffness_scale": xpbd.get("fiber_stiffness_scale", 100000.0),
        "dampingratio": xpbd.get("dampingratio", 0.1),
        "n_constraint_iters": xpbd.get("n_constraint_iters", 1),
        "constraints": raw.get("constraints", []),
    })
    return cfg


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
    # Build constraint configs from JSON
    constraint_list = []
    for cc in cfg.get("constraints", []):
        entry = dict(cc)
        entry.setdefault("name", cc["type"])
        constraint_list.append(entry)
    if not constraint_list:
        raise ValueError("No constraints defined in config JSON")

    mc = MillardCurves()

    sim, bottom_ids, axis_idx, tets, vertices, fiber_dirs_per_tet = \
        build_xpbd_sliding_ball_sim(
            cfg, constraint_list,
            contraction_ratio=0.0,
            sigma0=cfg["sigma0"],
            lambda_opt=cfg.get("lambda_opt", 1.0))

    # Log expected equilibrium values
    A_cross = np.pi * cfg["radius"] ** 2
    F_max = cfg["sigma0"] * A_cross
    norm_load = cfg["ball_mass"] * cfg["gravity"] / F_max
    print(f"Explicit active force mode: F_max={F_max:.1f}N, norm_load={norm_load:.4f}")
    print(f"  lambda_opt={cfg.get('lambda_opt', 1.0):.3f}")
    print(f"  Expected: f_L(lm_eq)={norm_load:.4f}")

    dt = cfg["dt"]
    n_steps = cfg["n_steps"]
    n_tet = len(tets)
    rest_matrices = compute_rest_matrices(vertices, tets)

    # Activation dynamics state
    act_arr = np.full(n_tet, 0.01, dtype=np.float32)
    act_sub_dt = cfg["act_substep_dt"]
    excitation_val = cfg["excitation"]

    rec_t, rec_z, rec_a, rec_fiber = [], [], [], []

    print(f"\nSimulating {n_steps * dt:.1f}s (Millard, "
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
