"""VBD volumetric muscle sliding-ball example.

Generates a cylinder tet mesh with a concentrated ball mass at the bottom,
applies activation ramp, simulates contraction with VBD, and saves results
to NPZ for comparison with OpenSim (done externally in OpenSimExample).

Usage:
    uv run -m examples.example_muscle_sliding_ball
    uv run -m examples.example_muscle_sliding_ball --config data/slidingBall/config.json

Plotting (in OpenSimExample repo):
    python scripts/plot_sliding_ball.py --vbd output/vbd_muscle_sliding_ball_default.npz
"""

import json
import os
import argparse

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverVBD

from VMuscle.activation import activation_dynamics_step_np
from VMuscle.dgf_curves import compute_fiber_forces
from VMuscle.mesh_utils import (
    assign_fiber_directions,
    create_cylinder_tet_mesh,
    set_vmuscle_properties,
)


def load_config(path):
    """Load config JSON and return a flat dict of parameters."""
    with open(path) as f:
        raw = json.load(f)

    geo = raw["geometry"]
    phys = raw["physics"]
    mus = raw["muscle"]
    mat = raw["material"]
    act = raw["activation"]
    sol = raw["solver"]

    return {
        "length": geo["muscle_length"],
        "radius": geo["muscle_radius"],
        "n_circ": geo["n_circumferential"],
        "n_axial": geo["n_axial"],
        "up_axis": geo["up_axis"],
        "density": phys["density"],
        "ball_mass": phys["ball_mass"],
        "sigma0": mus["sigma0"],
        "v_max": mus["max_contraction_velocity"],
        "fiber_damping": mus["fiber_damping"],
        "k_mu": mat["k_mu"],
        "k_lambda": mat["k_lambda"],
        "k_damp": mat["k_damp"],
        "excitation": act["excitation"],
        "act_substep_dt": act["substep_dt"],
        "dt": sol["dt"],
        "n_steps": sol["n_steps"],
        "iterations": sol["iterations"],
    }


def compute_fiber_data(pos, tet_idx, tet_poses, fiber_dirs,
                       activation, dt, v_max, l_prev_mean=None):
    """Compute per-tet fiber stretches and mean normalized forces.

    Returns dict with l_mean, f_active, f_passive, f_total, f_velocity.
    """
    n = len(tet_idx)
    stretches = np.empty(n)
    for e in range(n):
        i, j, k, l = tet_idx[e]
        Ds = np.column_stack([pos[j] - pos[i], pos[k] - pos[i], pos[l] - pos[i]])
        Fd = (Ds @ tet_poses[e]) @ fiber_dirs[e]
        stretches[e] = max(np.linalg.norm(Fd), 1e-8)

    l_mean = float(stretches.mean())
    if l_prev_mean is not None and dt > 0:
        v_norm = (l_mean - l_prev_mean) / (dt * v_max)
    else:
        v_norm = 0.0  # isometric for first sample

    fd = compute_fiber_forces(stretches, activation, v_norm)
    fd['l_mean'] = l_mean
    return fd


def run_sim(cfg, label="default"):
    """Run one VBD sliding-ball simulation from config dict."""
    length = cfg["length"]
    radius = cfg["radius"]
    sigma0 = cfg["sigma0"]
    density = cfg["density"]
    ball_mass = cfg["ball_mass"]
    dt = cfg["dt"]
    n_steps = cfg["n_steps"]
    iterations = cfg["iterations"]

    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    print(f"Device: {device}")

    # Mesh
    vertices, tets = create_cylinder_tet_mesh(
        length, radius, cfg["n_circ"], cfg["n_axial"])
    axis = {"X": 0, "Y": 1, "Z": 2}[cfg["up_axis"]]
    fiber_dirs = assign_fiber_directions(vertices, tets, axis=axis)
    print(f"Mesh: {len(vertices)} verts, {len(tets)} tets")

    # Build model
    builder = newton.ModelBuilder(up_axis=cfg["up_axis"])
    tet_offset = len(builder.tet_indices)
    mesh = newton.TetMesh(vertices=vertices, tet_indices=tets.flatten(),
                          k_mu=cfg["k_mu"], k_lambda=cfg["k_lambda"],
                          k_damp=cfg["k_damp"], density=density)
    builder.add_soft_mesh(mesh=mesh, pos=(0, 0, 0), rot=wp.quat_identity(),
                          scale=1.0, vel=(0, 0, 0))
    set_vmuscle_properties(builder, tet_offset, fiber_dirs, sigma0)
    builder.vmuscle_max_contraction_velocity = cfg["v_max"]
    builder.vmuscle_fiber_damping = cfg["fiber_damping"]

    # Fix TOP (z ≈ length) — hanging configuration matching OpenSim ceiling.
    bottom_ids = []
    for i in range(builder.particle_count):
        z = builder.particle_q[i][axis]
        if z > length - 1e-4:
            builder.particle_mass[i] = 0.0  # kinematic: fixed ceiling
        elif z < 1e-4:
            bottom_ids.append(i)

    # Add concentrated ball mass to bottom vertices
    mass_per_bottom = ball_mass / len(bottom_ids)
    for bid in bottom_ids:
        builder.particle_mass[bid] += mass_per_bottom
    print(f"Ball mass: {ball_mass} kg distributed to {len(bottom_ids)} bottom verts")

    builder.color()
    model = builder.finalize(device=device)
    F_max = sigma0 * np.pi * radius ** 2
    print(f"vmuscle: {model.vmuscle_count}, tets: {model.tet_count}")
    print(f"F_max={F_max:.1f}N, weight={ball_mass*9.81:.1f}N, "
          f"ratio={F_max/(ball_mass*9.81):.2f}")

    # CPU-side tet data for fiber extraction
    tet_idx = np.array(builder.tet_indices, dtype=int)
    tet_poses = np.array(builder.tet_poses)
    fib_np = np.array(builder.vmuscle_tet_fiber_dirs, dtype=np.float32)

    # Solver
    s0, s1, ctrl = model.state(), model.state(), model.control()
    solver = SolverVBD(model, iterations=iterations)

    # Simulate
    v_max = cfg["v_max"]
    t_end = n_steps * dt
    print(f"\nSimulating {t_end:.1f}s (top fixed, {len(bottom_ids)} bottom verts, "
          f"ball={ball_mass}kg)...")
    rec_t, rec_z, rec_vz, rec_a = [], [], [], []
    rec_fiber = []
    l_prev_mean = None
    n_tets = model.vmuscle_tet_activations.shape[0]
    act_arr = np.full(n_tets, 0.01, dtype=np.float32)
    act_sub_dt = cfg["act_substep_dt"]
    excitation_val = cfg["excitation"]

    for step in range(n_steps):
        t = step * dt
        # Excitation → activation dynamics with sub-stepping
        exc = np.full(n_tets, excitation_val, dtype=np.float32)
        n_sub = max(1, int(np.ceil(dt / act_sub_dt)))
        sub_dt = dt / n_sub
        for _ in range(n_sub):
            act_arr = activation_dynamics_step_np(exc, act_arr, sub_dt)
        model.vmuscle_tet_activations.assign(
            wp.array(act_arr, dtype=wp.float32, device=device))
        a = float(act_arr.mean())

        solver.step(s0, s1, ctrl, contacts=None, dt=dt)
        s0, s1 = s1, s0

        pos = s0.particle_q.numpy()
        vel = s0.particle_qd.numpy()
        bottom_z = float(np.mean([pos[bid][axis] for bid in bottom_ids]))
        bottom_vz = float(np.mean([vel[bid][axis] for bid in bottom_ids]))

        fd = compute_fiber_data(pos, tet_idx, tet_poses, fib_np,
                                a, dt, v_max, l_prev_mean)
        l_prev_mean = fd['l_mean']

        rec_t.append(t + dt)
        rec_z.append(bottom_z)
        rec_vz.append(bottom_vz)
        rec_a.append(a)
        rec_fiber.append(fd)

        if step % 50 == 0 or step == n_steps - 1:
            print(f"  step={step:4d}  t={t:.3f}s  a={a:.2f}  "
                  f"bottom_z={bottom_z:.6f}  l~={fd['l_mean']:.4f}")

    print("Done.")

    # Save NPZ
    os.makedirs("output", exist_ok=True)
    out = f"output/vbd_muscle_sliding_ball_{label}.npz"
    np.savez(out,
             times=np.array(rec_t),
             positions=np.array(rec_z),
             velocities=np.array(rec_vz),
             activations=np.array(rec_a),
             norm_fiber_lengths=np.array([d['l_mean'] for d in rec_fiber]),
             f_active=np.array([d['f_active'] for d in rec_fiber]),
             f_passive=np.array([d['f_passive'] for d in rec_fiber]),
             f_total=np.array([d['f_total'] for d in rec_fiber]),
             f_velocity=np.array([d['f_velocity'] for d in rec_fiber]),
             sigma0=sigma0, radius=radius, muscle_length=length,
             ball_mass=ball_mass, dt=dt)
    print(f"NPZ saved to {out}")

    # Save .sto with OpenSim state-path column names so the file can be
    # loaded as a motion in OpenSim GUI alongside vbd_muscle_comparison.osim.
    sto_path = f"output/vbd_sliding_ball_{label}.sto"
    columns = ["/jointset/slider/height/value",
               "/jointset/slider/height/speed",
               "/forceset/muscle/activation"]
    n_rows = len(rec_t)
    with open(sto_path, "w") as f:
        f.write("vbd_states\n")
        f.write(f"nRows={n_rows}\n")
        f.write(f"nColumns={len(columns) + 1}\n")
        f.write("inDegrees=no\n")
        f.write("DataType=double\n")
        f.write("version=3\n")
        f.write("endheader\n")
        f.write("time\t" + "\t".join(columns) + "\n")
        for i in range(n_rows):
            f.write(f"{rec_t[i]}\t{rec_z[i]}\t{rec_vz[i]}\t{rec_a[i]}\n")
    print(f"Wrote {sto_path}")
    return out


def main():
    parser = argparse.ArgumentParser(description="VBD sliding-ball muscle example")
    parser.add_argument("--config", default="data/slidingBall/config.json",
                        help="Path to config JSON")
    parser.add_argument("--label", default="default")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_sim(cfg, label=args.label)


if __name__ == "__main__":
    main()
