"""VBD volumetric muscle sliding-ball example.

Generates a cylinder tet mesh with a concentrated ball mass at the bottom,
applies activation ramp, simulates contraction with VBD, and saves results
to CSV for comparison with OpenSim (done externally in OpenSimExample).

Usage:
    uv run -m examples.example_muscle_sliding_ball
"""

import csv
import os

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverVBD

from VMuscle.mesh_utils import (
    assign_fiber_directions,
    create_cylinder_tet_mesh,
    set_vmuscle_properties,
)


def compute_mean_fiber_stretch(pos, tet_idx, tet_poses, fiber_dirs):
    """Compute mean normalized fiber length (|F d|) over all tets."""
    n = len(tet_idx)
    stretch = np.empty(n)
    for e in range(n):
        i, j, k, l = tet_idx[e]
        Ds = np.column_stack([pos[j] - pos[i], pos[k] - pos[i], pos[l] - pos[i]])
        Fd = (Ds @ tet_poses[e]) @ fiber_dirs[e]
        stretch[e] = max(np.linalg.norm(Fd), 1e-8)
    return float(stretch.mean())


def main():
    length, radius = 0.10, 0.02
    # Passive tissue stiffness — low so DGF curves dominate (real muscle ~1-10 kPa)
    k_mu, k_lambda = 1.0e3, 1.0e4
    sigma0 = 3.0e5          # Pa, peak isometric stress
    density = 1060.0         # kg/m^3
    dt = 1.0 / 60.0
    n_steps = 300            # 5 seconds
    iterations = 30
    ramp_steps = 3           # very fast activation ramp (~0.05s)

    # Concentrated ball mass: choose so equilibrium l~ is in physiological range.
    # F_max = sigma0 * pi * r^2 ≈ 377 N.
    # ball_mass=10 → weight=98N, active eq. f_L(l~)=0.26 → l~≈0.63
    ball_mass = 10.0  # kg

    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    print(f"Device: {device}")

    # Mesh
    vertices, tets = create_cylinder_tet_mesh(length, radius, 8, 6)
    fiber_dirs = assign_fiber_directions(vertices, tets, axis=2)
    print(f"Mesh: {len(vertices)} verts, {len(tets)} tets")

    # Build model
    builder = newton.ModelBuilder(up_axis="Z")
    tet_offset = len(builder.tet_indices)
    mesh = newton.TetMesh(vertices=vertices, tet_indices=tets.flatten(),
                          k_mu=k_mu, k_lambda=k_lambda, density=density)
    builder.add_soft_mesh(mesh=mesh, pos=(0,0,0), rot=wp.quat_identity(),
                          scale=1.0, vel=(0,0,0))
    set_vmuscle_properties(builder, tet_offset, fiber_dirs, sigma0)
    # Disable F-V (large v_max → v_norm≈0 → fV≈1) to avoid lagged-velocity
    # oscillation in VBD.  TODO: fix F-V with implicit/smoothed velocity.
    builder.vmuscle_max_contraction_velocity = 1.0e6

    # Fix TOP (z ≈ length) — hanging configuration matching OpenSim ceiling.
    # Gravity pulls bottom down (stretching), muscle activation contracts upward.
    bottom_ids = []
    for i in range(builder.particle_count):
        z = builder.particle_q[i][2]
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
    t_end = n_steps * dt
    print(f"\nSimulating {t_end:.1f}s (top fixed, {len(bottom_ids)} bottom verts, "
          f"ball={ball_mass}kg)...")
    rec_t, rec_z, rec_a, rec_l = [], [], [], []

    for step in range(n_steps):
        t = step * dt
        a = min(step / ramp_steps, 1.0)
        act_arr = np.full(model.vmuscle_tet_activations.shape[0], a, dtype=np.float32)
        model.vmuscle_tet_activations.assign(
            wp.array(act_arr, dtype=wp.float32, device=device))

        solver.step(s0, s1, ctrl, contacts=None, dt=dt)
        s0, s1 = s1, s0

        pos = s0.particle_q.numpy()
        bottom_z = float(np.mean([pos[bid][2] for bid in bottom_ids]))
        lm = (compute_mean_fiber_stretch(pos, tet_idx, tet_poses, fib_np)
              if step % 5 == 0 or step == n_steps - 1 else rec_l[-1])

        rec_t.append(t + dt)
        rec_z.append(bottom_z)
        rec_a.append(a)
        rec_l.append(lm)

        if step % 50 == 0 or step == n_steps - 1:
            print(f"  step={step:4d}  t={t:.3f}s  a={a:.2f}  "
                  f"bottom_z={bottom_z:.6f}  l~={lm:.4f}")

    print("Done.")

    # Save CSV (compatible with OpenSimExample/vbd_muscle/demo.py format)
    os.makedirs("output", exist_ok=True)
    out_csv = "output/vbd_muscle_sliding_ball.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "ball_y", "norm_fiber_length"])
        for i in range(len(rec_t)):
            w.writerow([rec_t[i], rec_z[i], rec_l[i]])
    print(f"CSV saved to {out_csv}")


if __name__ == "__main__":
    main()
