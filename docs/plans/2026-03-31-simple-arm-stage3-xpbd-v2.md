# SimpleArm Stage 3: XPBD Coupled (Improved Plan v2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace VBD Stage 3 (which explodes at step 17) with XPBD MuscleSim using ATTACH elastic boundaries, DGF force-length matching via dynamic contraction_factor, and ARAP mesh quality protection.

**Architecture:** Build MuscleSim programmatically (via `object.__new__` pattern, proven in sliding ball). Constraints: TETVOLUME (volume) + TETFIBERDGF (fiber contraction) + TETARAP (shape preservation) + PIN (origin fixed) + ATTACH (insertion → MuJoCo bone). Each outer step: MuJoCo forward → compute normalized load → update contraction_factor via `update_cons_restdir1_kernel` → XPBD substeps → extract fiber stretch from deformation gradient → DGF curves → inject force into MuJoCo.

**Tech Stack:** Warp (XPBD MuscleSim), MuJoCo, NumPy, existing `src/VMuscle/` infrastructure

---

## Key Improvements Over Original Plan (v1)

| Issue in v1 | Fix in v2 |
|---|---|
| Missing `contraction_factor` dynamic update | Add `dgf_equilibrium_fiber_length` + `update_cons_restdir1_kernel` each step |
| Missing `tendonmask` field → kernel crash | Use parent `_init_fields()` which allocates it |
| No mesh shape protection → distortion risk | Add TETARAP constraints (low stiffness) |
| Fragile subclass bypassing `__init__` | Use `object.__new__(MuscleSim)` + manual init (proven pattern) |
| Deformation gradient vertex order mismatch | Use pts[3] as reference (matching kernel convention) |
| Warm-up with wrong activation (0.5) | Warm-up with `excitation_off` initial activation |
| Uses deprecated `UsdTetExporter` | Use `MeshExporter` |

---

## File Structure

| File | Operation | Responsibility |
|------|-----------|----------------|
| `examples/example_xpbd_coupled_simple_arm.py` | Create | XPBD coupled SimpleArm: build sim + coupling loop + force extraction |
| `scripts/run_simple_arm_comparison.py` | Modify | Add `--mode xpbd` option |
| `data/simpleArm/config.json` | Modify | Add `xpbd` config section |
| `tests/test_xpbd_simple_arm.py` | Create | Smoke tests |

---

## Task 1: Build MuscleSim Programmatically + Constraint Setup

Create `examples/example_xpbd_coupled_simple_arm.py` with the function that builds a MuscleSim from a cylinder mesh, with all 5 constraint types.

**Files:**
- Create: `examples/example_xpbd_coupled_simple_arm.py`

- [ ] **Step 1: Create file with imports and `build_xpbd_muscle_sim` function**

```python
"""XPBD bone-muscle coupling for SimpleArm (Stage 3).

Uses XPBD MuscleSim with ATTACH constraints for stable boundary coupling.
Force extracted from deformation gradient via DGF curves.

Usage:
    uv run python examples/example_xpbd_coupled_simple_arm.py
    uv run python examples/example_xpbd_coupled_simple_arm.py --config data/simpleArm/config.json
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import warp as wp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from VMuscle.activation import activation_dynamics_step_np
from VMuscle.constraints import ATTACH, PIN, TETARAP, TETFIBERDGF, TETVOLUME
from VMuscle.dgf_curves import active_force_length, force_velocity, passive_force_length
from VMuscle.mesh_io import MeshExporter
from VMuscle.mesh_utils import create_cylinder_tet_mesh
from VMuscle.muscle_warp import (MuscleSim, fill_float_kernel,
                                  update_cons_restdir1_kernel)


def dgf_equilibrium_fiber_length(activation, normalized_load):
    """Invert DGF f_L curve: find lm where a * f_L(lm) + f_PE(lm) = normalized_load.

    Returns equilibrium on the ascending limb (lm < 1).
    normalized_load = F_ext / F_max.
    """
    if activation < 1e-8:
        return 1.0
    target = normalized_load
    lm_range = np.linspace(0.3, 1.8, 3000)
    fl = active_force_length(lm_range)
    fpe = passive_force_length(lm_range)
    f_total = activation * fl + fpe
    idx = np.argmin(np.abs(f_total - target))
    return float(lm_range[idx])


def build_xpbd_muscle_sim(
    vertices, tets, fiber_dirs,
    origin_ids, insertion_ids, bone_targets,
    sigma0, dt, num_substeps,
    attach_stiffness=1e6,
    pin_stiffness=1e8,
    fiber_stiffness=1000.0,
    fiber_stiffness_scale=200.0,
    fiber_damping=0.5,
    volume_stiffness=10000.0,
    arap_stiffness=500.0,
    arap_damping=0.01,
    contraction_factor=0.4,
    density=1060.0,
    veldamping=0.003,
    device="cpu",
):
    """Build a MuscleSim programmatically for a cylinder muscle mesh.

    Uses the object.__new__ pattern (proven in example_xpbd_dgf_sliding_ball.py).
    Constraints: TETVOLUME + TETFIBERDGF + TETARAP + PIN + ATTACH.

    Args:
        vertices: (N, 3) float32 vertex positions.
        tets: (M, 4) int32 tet connectivity.
        fiber_dirs: (M, 3) float32 per-tet fiber directions.
        origin_ids: list of vertex indices pinned to world (origin end).
        insertion_ids: list of vertex indices attached to bone (insertion end).
        bone_targets: (len(insertion_ids), 3) float32 initial bone target positions.
        sigma0: Peak isometric stress [Pa].
        dt: Outer timestep [s].
        num_substeps: XPBD substeps per outer step.
        attach_stiffness: ATTACH constraint stiffness.
        pin_stiffness: PIN constraint stiffness.
        fiber_stiffness: TETFIBERDGF base stiffness.
        fiber_stiffness_scale: Global fiber stiffness multiplier.
        fiber_damping: TETFIBERDGF damping ratio.
        volume_stiffness: TETVOLUME stiffness.
        arap_stiffness: TETARAP stiffness (shape preservation).
        arap_damping: TETARAP damping ratio.
        contraction_factor: Initial contraction factor (updated per step).
        density: Tissue density [kg/m^3].
        veldamping: Velocity damping.
        device: Warp device string.

    Returns:
        sim: MuscleSim instance ready for stepping.
    """
    n_v = len(vertices)
    n_tet = len(tets)

    # Per-vertex fiber directions (averaged from per-tet, for v_fiber_dir field)
    v_fiber_np = np.zeros((n_v, 3), dtype=np.float32)
    v_count = np.zeros(n_v, dtype=np.float32)
    for t in range(n_tet):
        for vi in tets[t]:
            v_fiber_np[vi] += fiber_dirs[t]
            v_count[vi] += 1.0
    v_count = np.maximum(v_count, 1.0)
    v_fiber_np /= v_count[:, None]
    norms = np.linalg.norm(v_fiber_np, axis=1, keepdims=True)
    v_fiber_np /= np.maximum(norms, 1e-8)

    # Build constraint config list (matching ConstraintBuilderMixin format)
    constraint_configs = [
        {"type": "volume", "name": "vol",
         "stiffness": volume_stiffness, "dampingratio": 1.0},
        {"type": "fiberdgf", "name": "fiber_dgf",
         "stiffness": fiber_stiffness, "dampingratio": fiber_damping,
         "sigma0": sigma0, "contraction_factor": contraction_factor},
        {"type": "arap", "name": "arap",
         "stiffness": arap_stiffness, "dampingratio": arap_damping},
    ]

    # Build MuscleSim via object.__new__ (bypass file loading)
    sim = object.__new__(MuscleSim)

    sim.cfg = SimpleNamespace(
        dt=dt,
        num_substeps=num_substeps,
        gravity=0.0,
        density=density,
        veldamping=veldamping,
        constraints=constraint_configs,
        arch=device,
        gui=False,
        render_mode=None,
        activation=0.0,
        contraction_ratio=contraction_factor,
        fiber_stiffness_scale=fiber_stiffness_scale,
        HAS_compressstiffness=False,
        k_mu=fiber_stiffness,
        k_lambda=volume_stiffness,
        k_damp=1.0,
    )
    sim.constraint_configs = constraint_configs

    # Set mesh data (normally loaded from file)
    sim.pos0_np = vertices.astype(np.float32)
    sim.tet_np = tets.astype(np.int32)
    sim.v_fiber_np = v_fiber_np.astype(np.float32)
    sim.v_tendonmask_np = None  # no tendon for cylinder
    sim.geo = SimpleNamespace()  # empty geo stub
    sim.n_verts = n_v

    # Bone data: one bone vertex per insertion vertex
    sim.bone_geo = None
    sim.bone_pos = bone_targets.astype(np.float32)
    sim.bone_indices_np = np.zeros(0, dtype=np.int32)
    sim.bone_muscle_ids = {}

    # Standard init sequence
    wp.init()
    sim._init_backend()
    sim._allocate_fields()
    sim._init_fields()
    sim._precompute_rest()
    sim._build_surface_tris()
    sim._create_bone_fields()

    sim.use_jacobi = False
    sim.use_colored_gs = False  # CPU safe; enable for CUDA after testing
    sim.contraction_ratio = contraction_factor
    sim.fiber_stiffness_scale = fiber_stiffness_scale
    sim.has_compressstiffness = False
    sim.dt = dt / num_substeps  # substep dt
    sim.step_cnt = 0
    sim.renderer = None

    # Build standard constraints (TETVOLUME + TETFIBERDGF + TETARAP)
    # via ConstraintBuilderMixin._collect_raw_constraints()
    sim.build_constraints()

    # Now manually add PIN + ATTACH constraints
    # We need to rebuild the full constraint array with these appended.
    # First get existing raw constraints from build_constraints
    existing_cons = []
    cons_np_existing = sim.cons.numpy()
    for i in range(sim.n_cons):
        c = {}
        for field in ['type', 'cidx', 'pts', 'stiffness', 'dampingratio',
                       'tetid', 'L', 'restlength', 'restvector', 'restdir',
                       'compressionstiffness']:
            c[field] = cons_np_existing[field][i]
            # Convert numpy arrays to lists for dict format
            if hasattr(c[field], 'tolist'):
                c[field] = c[field].tolist() if hasattr(c[field], '__len__') else c[field].item()
        existing_cons.append(c)

    # Helper: map vertex to containing tet
    pt2tet = {}
    for t in range(n_tet):
        for v in tets[t]:
            if int(v) not in pt2tet:
                pt2tet[int(v)] = t

    # PIN constraints for origin vertices
    for vid in origin_ids:
        pos = vertices[vid]
        c = dict(
            type=PIN,
            pts=[int(vid), -1, -1, -1],
            stiffness=pin_stiffness,
            dampingratio=0.1,
            tetid=pt2tet.get(int(vid), -1),
            L=[0.0, 0.0, 0.0],
            restlength=0.0,
            restvector=[float(pos[0]), float(pos[1]), float(pos[2]), 1.0],
            restdir=[0.0, 0.0, 0.0],
            compressionstiffness=-1.0,
        )
        existing_cons.append(c)

    # ATTACH constraints for insertion vertices → bone targets
    for idx, vid in enumerate(insertion_ids):
        tgt_pos = bone_targets[idx]
        dist = float(np.linalg.norm(vertices[vid] - tgt_pos))
        c = dict(
            type=ATTACH,
            pts=[int(vid), -1, int(idx), -1],  # pts[2] = bone vertex index
            stiffness=attach_stiffness,
            dampingratio=0.1,
            tetid=pt2tet.get(int(vid), -1),
            L=[0.0, 0.0, 0.0],
            restlength=dist,
            restvector=[float(tgt_pos[0]), float(tgt_pos[1]), float(tgt_pos[2]), 1.0],
            restdir=[0.0, 0.0, 0.0],
            compressionstiffness=-1.0,
        )
        existing_cons.append(c)

    # Re-upload all constraints (sort by type, assign cidx, build ranges)
    _upload_all_constraints(sim, existing_cons)

    # Cache attach constraints for update_attach_targets
    sim.attach_constraints = [c for c in existing_cons if c['type'] == ATTACH]
    sim.distanceline_constraints = []

    print(f"[build_xpbd] {n_v} verts, {n_tet} tets, {sim.n_cons} constraints: "
          + ", ".join(f"{k}={v[1]}" for k, v in sorted(sim.cons_ranges.items())))

    return sim


def _upload_all_constraints(sim, all_constraints):
    """Sort, assign cidx, upload constraint list to Warp array, build cons_ranges."""
    from VMuscle.muscle_warp import Constraint

    all_constraints.sort(key=lambda c: c['type'])
    n_cons = len(all_constraints)
    sim.n_cons = n_cons
    sim.cons_ranges = {}

    if n_cons > 0:
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
        cons_np['compressionstiffness'] = np.array(
            [c['compressionstiffness'] for c in all_constraints], dtype=np.float32)
        sim.cons = wp.array(cons_np, dtype=Constraint)
    else:
        sim.cons = wp.zeros(0, dtype=Constraint)

    sim.reaction_accum = wp.zeros(max(n_cons, 1), dtype=wp.vec3)
```

- [ ] **Step 2: Verify syntax compiles**

Run: `uv run python -m py_compile examples/example_xpbd_coupled_simple_arm.py`
Expected: No output (success)

- [ ] **Step 3: Verify MuscleSim builds successfully**

Run:
```bash
uv run python -c "
import sys, os; sys.path.insert(0, '.')
import numpy as np, warp as wp
from VMuscle.mesh_utils import create_cylinder_tet_mesh
wp.init()
verts, tets = create_cylinder_tet_mesh(0.3, 0.02, 8, 10)
n_tets = len(tets)
fiber_dirs = np.tile([0,0,1], (n_tets,1)).astype(np.float32)
origin_ids = [i for i in range(len(verts)) if verts[i,2] > 0.28]
insertion_ids = [i for i in range(len(verts)) if verts[i,2] < 0.02]
bone_targets = verts[insertion_ids].copy()
sys.path.insert(0, 'examples')
from example_xpbd_coupled_simple_arm import build_xpbd_muscle_sim
sim = build_xpbd_muscle_sim(verts, tets, fiber_dirs,
    origin_ids, insertion_ids, bone_targets,
    sigma0=159155.0, dt=0.0167, num_substeps=10, device='cpu')
print(f'OK: {sim.n_verts} verts, {len(tets)} tets, {sim.n_cons} constraints')
print(f'cons_ranges: {sim.cons_ranges}')
# Quick step test
sim.integrate()
sim.clear()
sim.clear_reaction()
sim.solve_constraints()
sim.update_velocities()
pos = sim.pos.numpy()
print(f'Step OK: NaN={np.any(np.isnan(pos))}, mean_pos={pos.mean(axis=0)}')
"
```
Expected: Prints constraint stats, step completes without NaN.

- [ ] **Step 4: Commit**

```bash
git add examples/example_xpbd_coupled_simple_arm.py
git commit -m "feat: add build_xpbd_muscle_sim with TETVOLUME+TETFIBERDGF+TETARAP+PIN+ATTACH"
```

---

## Task 2: XPBD Coupled Simulation Loop

Add the main simulation loop with dynamic contraction_factor update, MuJoCo coupling, and deformation gradient force extraction.

**Files:**
- Modify: `examples/example_xpbd_coupled_simple_arm.py`

- [ ] **Step 1: Add helper functions after `_upload_all_constraints`**

```python
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

    Uses pts[3] as reference vertex (matching XPBD kernel convention).
    """
    n = len(tet_idx)
    stretches = np.empty(n)
    for e in range(n):
        i0, i1, i2, i3 = tet_idx[e]
        # Ds columns: p0-p3, p1-p3, p2-p3 (same as kernel)
        Ds = np.column_stack([pos[i0] - pos[i3], pos[i1] - pos[i3], pos[i2] - pos[i3]])
        F = Ds @ rest_matrices[e]
        Fd = F @ fiber_dirs[e]
        stretches[e] = max(np.linalg.norm(Fd), 1e-8)
    return stretches


def update_bone_targets(sim, new_positions):
    """Update bone_pos_field with new insertion target positions."""
    bone_np = new_positions.astype(np.float32)
    sim.bone_pos_field = wp.from_numpy(bone_np, dtype=wp.vec3)
```

- [ ] **Step 2: Add `xpbd_coupled_simple_arm` main function**

```python
def xpbd_coupled_simple_arm(cfg, verbose=True):
    """Run XPBD + MuJoCo coupled SimpleArm simulation.

    Returns:
        dict with times, elbow_angles, forces, norm_fiber_lengths, activations.
    """
    mus = cfg["muscle"]
    act_cfg = cfg["activation"]
    sol = cfg["solver"]
    ic = cfg["initial_conditions"]
    xpbd_cfg = cfg.get("xpbd", {})

    F_max = mus["max_isometric_force"]
    L_opt = mus["optimal_fiber_length"]
    L_slack = mus["tendon_slack_length"]
    V_max = mus["max_contraction_velocity"]
    d_damp = mus["fiber_damping"]

    outer_dt = sol["dt"]
    n_steps = sol["n_steps"]
    theta0 = np.radians(ic["elbow_angle_deg"])
    device = sol.get("arch", "cuda:0" if wp.is_cuda_available() else "cpu")

    num_substeps = xpbd_cfg.get("num_substeps", 20)
    attach_stiffness = xpbd_cfg.get("attach_stiffness", 1e6)
    pin_stiffness = xpbd_cfg.get("pin_stiffness", 1e8)
    fiber_stiffness = xpbd_cfg.get("fiber_stiffness", 1000.0)
    fiber_stiffness_scale = xpbd_cfg.get("fiber_stiffness_scale", 200.0)
    volume_stiffness = xpbd_cfg.get("volume_stiffness", 10000.0)
    arap_stiffness = xpbd_cfg.get("arap_stiffness", 500.0)
    warmup_steps = xpbd_cfg.get("warmup_steps", 50)

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
        print(f"[XPBD] origin={origin_pos}, insertion={insertion_pos}")
        print(f"[XPBD] ten_length={ten_length_init:.4f}, "
              f"fiber_length={fiber_length_init:.4f}, "
              f"l_tilde={fiber_length_init / L_opt:.4f}")

    # --- Build cylinder mesh in world frame ---
    geo = cfg["geometry"]
    r = geo["muscle_radius"]
    n_circ = geo["n_circumferential"]
    n_axial = geo["n_axial"]
    mesh_length = fiber_length_init

    tendon_vec = origin_pos - insertion_pos
    tendon_dir = (tendon_vec / np.linalg.norm(tendon_vec)).astype(np.float32)

    vertices, tets = create_cylinder_tet_mesh(mesh_length, r, n_circ, n_axial)
    R = _rotation_matrix_from_z_to(tendon_dir)
    vertices = (R @ vertices.T).T + insertion_pos.astype(np.float32)
    fiber_origin_pos = insertion_pos + tendon_dir.astype(np.float64) * mesh_length

    n_tets = len(tets)
    fiber_dirs = np.tile(tendon_dir, (n_tets, 1))
    sigma0 = F_max / (np.pi * r ** 2)

    # Identify boundary vertices
    origin_ids = []
    insertion_ids = []
    for i in range(len(vertices)):
        p = vertices[i].astype(np.float64)
        if np.linalg.norm(p - fiber_origin_pos) < r * 1.5:
            origin_ids.append(i)
        elif np.linalg.norm(p - insertion_pos) < r * 1.5:
            insertion_ids.append(i)

    bone_targets = vertices[insertion_ids].copy()

    if verbose:
        print(f"[XPBD] mesh: {len(vertices)} verts, {n_tets} tets, "
              f"origin={len(origin_ids)}, insertion={len(insertion_ids)}")

    # --- Build XPBD MuscleSim ---
    wp.init()
    sim = build_xpbd_muscle_sim(
        vertices, tets, fiber_dirs,
        origin_ids, insertion_ids, bone_targets,
        sigma0=sigma0,
        dt=outer_dt, num_substeps=num_substeps,
        attach_stiffness=attach_stiffness,
        pin_stiffness=pin_stiffness,
        fiber_stiffness=fiber_stiffness,
        fiber_stiffness_scale=fiber_stiffness_scale,
        fiber_damping=d_damp,
        volume_stiffness=volume_stiffness,
        arap_stiffness=arap_stiffness,
        density=1060.0,
        device=device,
    )

    # Precompute for fiber stretch extraction (CPU)
    tet_idx = tets.astype(np.int32)
    # Rest matrices with pts[3] as reference (matching kernel convention)
    rest_matrices = np.zeros((n_tets, 3, 3), dtype=np.float32)
    for e in range(n_tets):
        i0, i1, i2, i3 = tet_idx[e]
        M = np.column_stack([
            vertices[i0] - vertices[i3],
            vertices[i1] - vertices[i3],
            vertices[i2] - vertices[i3]])
        det = np.linalg.det(M)
        if abs(det) > 1e-30:
            rest_matrices[e] = np.linalg.inv(M)
    stretch_to_ltilde = mesh_length / L_opt

    # --- Mesh exporter ---
    anim_dir = "output/anim"
    os.makedirs(anim_dir, exist_ok=True)
    exporter = MeshExporter(
        path=anim_dir, format="ply",
        tet_indices=tet_idx, positions=vertices)

    # --- Warm-up with initial activation ---
    initial_activation = act_cfg["excitation_off"]
    n_tet = len(tets)
    wp.launch(fill_float_kernel, dim=n_tet,
              inputs=[sim.activation, wp.float32(initial_activation)])

    # Compute initial contraction_factor from DGF equilibrium
    # At rest, external load ~ 0 (gravity off), so lm_eq ~ 1 for low activation
    cf_init = 1.0 - dgf_equilibrium_fiber_length(initial_activation, 0.0)
    if TETFIBERDGF in sim.cons_ranges:
        off, cnt = sim.cons_ranges[TETFIBERDGF]
        wp.launch(update_cons_restdir1_kernel, dim=cnt,
                  inputs=[sim.cons, cf_init, TETFIBERDGF, off, cnt])

    if verbose:
        print(f"[XPBD] Warming up ({warmup_steps} steps, a={initial_activation:.2f}, "
              f"cf={cf_init:.4f})...")

    for _ in range(warmup_steps):
        sim.integrate()
        sim.clear()
        sim.update_attach_targets()
        sim.clear_reaction()
        sim.solve_constraints()
        if sim.use_jacobi:
            sim.apply_dP()
        sim.update_velocities()

    pos_np = sim.pos.numpy()
    if verbose:
        stretches = compute_fiber_stretches(pos_np, tet_idx, rest_matrices, fiber_dirs)
        print(f"[XPBD] Warm-up done: mean_stretch={stretches.mean():.4f}, "
              f"NaN={np.any(np.isnan(pos_np))}")

    # --- Simulation ---
    activation = initial_activation
    prev_fiber_length = fiber_length_init
    prev_muscle_force = 0.0

    times, elbow_angles, forces_out = [], [], []
    norm_fiber_lengths, activations_out = [], []
    physics_time = 0.0

    if verbose:
        print(f"[XPBD] Simulating {n_steps * outer_dt:.1f}s, F_max={F_max:.0f}N")

    for step in range(n_steps):
        t = physics_time
        elbow_angle = float(mj_data.qpos[0])
        times.append(t)
        elbow_angles.append(elbow_angle)
        activations_out.append(activation)

        # 1. Update ATTACH targets from MuJoCo insertion site
        mujoco.mj_forward(mj_model, mj_data)
        new_insertion = mj_data.site_xpos[insertion_site_id].copy()
        delta = new_insertion.astype(np.float32) - insertion_pos.astype(np.float32)
        new_bone_targets = bone_targets + delta
        update_bone_targets(sim, new_bone_targets)

        # 2. Set activation on GPU
        wp.launch(fill_float_kernel, dim=n_tet,
                  inputs=[sim.activation, wp.float32(activation)])

        # 3. Update contraction_factor from DGF equilibrium
        # Estimate normalized load from previous step's muscle force
        norm_load = max(prev_muscle_force, 0.0) / F_max
        lm_eq = dgf_equilibrium_fiber_length(activation, norm_load)
        cf = 1.0 - lm_eq
        if TETFIBERDGF in sim.cons_ranges:
            off, cnt = sim.cons_ranges[TETFIBERDGF]
            wp.launch(update_cons_restdir1_kernel, dim=cnt,
                      inputs=[sim.cons, cf, TETFIBERDGF, off, cnt])

        # 4. XPBD substeps
        for _ in range(num_substeps):
            sim.integrate()
            sim.clear()
            sim.update_attach_targets()
            sim.clear_reaction()
            sim.solve_constraints()
            if sim.use_jacobi:
                sim.apply_dP()
            sim.update_velocities()

        # 5. Extract fiber stretch from deformation gradient
        xpbd_pos = sim.pos.numpy()
        xpbd_valid = not np.any(np.isnan(xpbd_pos))
        if xpbd_valid:
            stretches = compute_fiber_stretches(xpbd_pos, tet_idx, rest_matrices, fiber_dirs)
            l_tilde_xpbd = float(stretches.mean()) * stretch_to_ltilde
            if l_tilde_xpbd < 0.1 or l_tilde_xpbd > 3.0:
                xpbd_valid = False

        ten_length = float(mj_data.ten_length[0])
        fiber_length = ten_length - L_slack
        mj_ltilde = fiber_length / L_opt

        if not xpbd_valid:
            l_tilde_xpbd = mj_ltilde

        # Save mesh frame
        exporter.save_frame(xpbd_pos.astype(np.float32), step)

        # 6. MuJoCo substeps with DGF force
        for sub in range(mj_substeps):
            t_sub = t + sub * mj_dt

            # Excitation schedule
            t_start = act_cfg["excitation_start_time"]
            t_end_exc = act_cfg["excitation_end_time"]
            e_off = act_cfg["excitation_off"]
            e_on = act_cfg["excitation_on"]
            if t_sub < t_start:
                excitation = e_off
            elif t_sub >= t_end_exc:
                excitation = e_on
            else:
                frac = (t_sub - t_start) / (t_end_exc - t_start)
                frac = frac * frac * (3.0 - 2.0 * frac)
                excitation = e_off + (e_on - e_off) * frac

            activation = float(activation_dynamics_step_np(
                np.array([excitation], dtype=np.float32),
                np.array([activation], dtype=np.float32),
                mj_dt,
                tau_act=act_cfg["tau_act"],
                tau_deact=act_cfg["tau_deact"],
            )[0])

            ten_len = float(mj_data.ten_length[0])
            fib_len = ten_len - L_slack
            fib_vel = (fib_len - prev_fiber_length) / mj_dt if (step > 0 or sub > 0) else 0.0
            prev_fiber_length = fib_len
            v_norm = fib_vel / (V_max * L_opt)

            # Use XPBD-corrected l_tilde for DGF force evaluation
            l_tilde_1d = fib_len / L_opt
            delta_ltilde = l_tilde_xpbd - mj_ltilde
            l_tilde_now = l_tilde_1d + delta_ltilde

            fl = float(active_force_length(l_tilde_now))
            fpe = float(passive_force_length(l_tilde_now))
            fv = float(force_velocity(np.clip(v_norm, -1.0, 1.0)))
            muscle_force = (activation * fl * fv + fpe + d_damp * v_norm) * F_max
            muscle_force = np.clip(muscle_force, 0.0, F_max * 2.0)

            mj_data.ctrl[0] = muscle_force
            mujoco.mj_step(mj_model, mj_data)
            physics_time += mj_dt

        prev_muscle_force = float(muscle_force)
        forces_out.append(float(muscle_force))
        norm_fiber_lengths.append(l_tilde_xpbd)

        if verbose and (step % 100 == 0 or step == n_steps - 1):
            print(f"  step={step:4d} t={t:6.3f}s "
                  f"theta={np.degrees(elbow_angle):7.2f}deg "
                  f"F={muscle_force:7.2f}N a={activation:.4f} "
                  f"l_xpbd={l_tilde_xpbd:.4f} l_1d={mj_ltilde:.4f} "
                  f"cf={cf:.4f}")

    if verbose:
        print(f"[XPBD] Done: {len(times)} points, "
              f"final angle={np.degrees(elbow_angles[-1]):.1f}deg")

    exporter.finalize()

    # Save .sto
    os.makedirs("output", exist_ok=True)
    sto_path = "output/SimpleArm_XPBD_Coupled_states.sto"
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
        f.write(f"nRows={len(times)}\n")
        f.write("DataType=double\nversion=3\nendheader\n")
        f.write("time\t" + "\t".join(cols) + "\n")
        for i in range(len(times)):
            f.write(f"{times[i]}\t{elbow_angles[i]}\t{activations_out[i]}\t"
                    f"{forces_out[i]}\t{norm_fiber_lengths[i]}\n")
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


def load_config(path="data/simpleArm/config.json"):
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="XPBD + MuJoCo coupled SimpleArm (Stage 3)")
    parser.add_argument("--config", default="data/simpleArm/config.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = xpbd_coupled_simple_arm(cfg)
    if result:
        print(f"\nFinal elbow angle: {np.degrees(result['elbow_angles'][-1]):.2f}deg")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify syntax compiles**

Run: `uv run python -m py_compile examples/example_xpbd_coupled_simple_arm.py`
Expected: No output (success)

- [ ] **Step 4: Run smoke test (10 steps)**

Run:
```bash
uv run python -c "
import sys; sys.path.insert(0, '.')
from examples.example_xpbd_coupled_simple_arm import xpbd_coupled_simple_arm, load_config
cfg = load_config()
cfg['solver']['n_steps'] = 10
cfg['solver']['arch'] = 'cpu'
result = xpbd_coupled_simple_arm(cfg)
import numpy as np
print(f'OK: {len(result[\"times\"])} steps')
print(f'NaN angles: {np.any(np.isnan(result[\"elbow_angles\"]))}')
print(f'NaN forces: {np.any(np.isnan(result[\"forces\"]))}')
print(f'Force range: [{result[\"forces\"].min():.1f}, {result[\"forces\"].max():.1f}]')
print(f'l_tilde range: [{result[\"norm_fiber_lengths\"].min():.4f}, {result[\"norm_fiber_lengths\"].max():.4f}]')
"
```
Expected: 10 steps complete, no NaN, forces in [0, 400], l_tilde in [0.3, 2.0].

- [ ] **Step 5: Commit**

```bash
git add examples/example_xpbd_coupled_simple_arm.py
git commit -m "feat: add XPBD coupled simulation loop with dynamic contraction_factor"
```

---

## Task 3: Config and Comparison Integration

**Files:**
- Modify: `data/simpleArm/config.json`
- Modify: `scripts/run_simple_arm_comparison.py`

- [ ] **Step 1: Add xpbd config section to config.json**

Add after the `"coupling"` section (before the closing `}`):

```json
  "xpbd": {
    "num_substeps": 20,
    "attach_stiffness": 1e6,
    "pin_stiffness": 1e8,
    "fiber_stiffness": 1000.0,
    "fiber_stiffness_scale": 200.0,
    "volume_stiffness": 10000.0,
    "arap_stiffness": 500.0,
    "warmup_steps": 50
  }
```

- [ ] **Step 2: Add xpbd mode to run_simple_arm_comparison.py**

Add `run_xpbd` function after `run_coupled`:

```python
def run_xpbd(cfg):
    from examples.example_xpbd_coupled_simple_arm import xpbd_coupled_simple_arm
    return xpbd_coupled_simple_arm(cfg)
```

Add `"xpbd"` to the `choices` list in `argparse`:

```python
    parser.add_argument("--mode", choices=["osim", "mujoco", "vbd", "coupled", "xpbd", "all"],
                        default="coupled",
```

Add xpbd block before the final `print("Done!")`:

```python
    if args.mode in ("xpbd", "all"):
        print("\n" + "=" * 60)
        print("Stage 3 XPBD: XPBD Coupled vs OpenSim DGF")
        print("=" * 60)
        if 'dgf' not in dir():
            dgf = run_dgf(cfg)
        xpbd = run_xpbd(cfg)
        rmse, max_err = plot_comparison(
            [("OpenSim DGF", dgf), ("XPBD Coupled", xpbd)],
            "SimpleArm Stage 3: XPBD Coupled vs OpenSim (DGF)",
            "output/simple_arm_xpbd_vs_osim.png",
            {"OpenSim DGF": "b", "XPBD Coupled": "c"},
        )
        if rmse is not None:
            print(f"\nXPBD Coupled vs OpenSim DGF: RMSE={rmse:.2f} deg, Max error={max_err:.2f} deg")
```

- [ ] **Step 3: Verify syntax**

Run:
```bash
uv run python -m py_compile scripts/run_simple_arm_comparison.py
uv run python -c "import json; json.load(open('data/simpleArm/config.json'))"
```
Expected: Both succeed silently.

- [ ] **Step 4: Commit**

```bash
git add data/simpleArm/config.json scripts/run_simple_arm_comparison.py
git commit -m "feat: integrate XPBD Stage 3 into comparison pipeline"
```

---

## Task 4: Smoke Tests

**Files:**
- Create: `tests/test_xpbd_simple_arm.py`

- [ ] **Step 1: Write test file**

```python
"""Smoke tests for XPBD coupled SimpleArm Stage 3."""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="module")
def xpbd_result():
    """Run XPBD simulation once for all tests (30 steps, CPU)."""
    from examples.example_xpbd_coupled_simple_arm import (
        xpbd_coupled_simple_arm, load_config)
    cfg = load_config()
    cfg["solver"]["n_steps"] = 30
    cfg["solver"]["arch"] = "cpu"
    return xpbd_coupled_simple_arm(cfg, verbose=False)


def test_xpbd_completes(xpbd_result):
    """XPBD simulation completes 30 steps without crash."""
    assert xpbd_result is not None
    assert len(xpbd_result["times"]) == 30


def test_xpbd_no_nan(xpbd_result):
    """No NaN in any output array."""
    assert not np.any(np.isnan(xpbd_result["elbow_angles"]))
    assert not np.any(np.isnan(xpbd_result["forces"]))
    assert not np.any(np.isnan(xpbd_result["norm_fiber_lengths"]))


def test_xpbd_force_reasonable(xpbd_result):
    """Forces stay in physically reasonable range [0, 2*F_max]."""
    F_max = xpbd_result["max_iso_force"]
    assert np.all(xpbd_result["forces"] >= 0)
    assert np.all(xpbd_result["forces"] <= F_max * 2.0)


def test_xpbd_fiber_stretch_physiological(xpbd_result):
    """Normalized fiber lengths are in physiological range [0.3, 2.0]."""
    nfl = xpbd_result["norm_fiber_lengths"]
    assert np.all(nfl > 0.3), f"min nfl={nfl.min():.4f}"
    assert np.all(nfl < 2.0), f"max nfl={nfl.max():.4f}"
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_xpbd_simple_arm.py -v`
Expected: 4 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_xpbd_simple_arm.py
git commit -m "test: add XPBD SimpleArm smoke tests"
```

---

## Task 5: Full Run, Validation, and Tuning

- [ ] **Step 1: Run 600-step simulation**

Run: `uv run python scripts/run_simple_arm_comparison.py --mode xpbd`
Expected: Completes without crash, produces `output/simple_arm_xpbd_vs_osim.png`.

- [ ] **Step 2: Inspect results**

Read and display `output/simple_arm_xpbd_vs_osim.png`. Check:
- Elbow angle reaches steady state near 82 deg (matching OpenSim DGF)
- RMSE < 5 deg vs OpenSim DGF
- Force curve follows DGF profile (bell-shaped rise, stable plateau)
- No sudden jumps or divergence

- [ ] **Step 3: If unstable, tune parameters in this priority order**

1. `num_substeps`: 20 → 30 → 40 (more substeps = more stable but slower)
2. `attach_stiffness`: 1e6 → 1e5 → 1e7 (too high oscillates, too low decouples)
3. `pin_stiffness`: 1e8 → 1e7 → 1e9
4. `arap_stiffness`: 500 → 200 → 1000 (too high fights fiber contraction)
5. `fiber_stiffness_scale`: 200 → 100 → 500
6. `fiber_stiffness`: 1000 → 500 → 2000

After each parameter change, re-run 30-step test to verify no crash, then 600-step for quality.

- [ ] **Step 4: Mesh quality check**

Add a quick det(F) check at the end of the simulation to verify no inverted tets:

```bash
uv run python -c "
import sys; sys.path.insert(0, '.')
from examples.example_xpbd_coupled_simple_arm import *
import numpy as np
cfg = load_config()
cfg['solver']['n_steps'] = 600
result = xpbd_coupled_simple_arm(cfg, verbose=True)
# result includes final positions implicitly; check mesh quality via stretches
print(f'Final l_tilde range: [{result[\"norm_fiber_lengths\"].min():.4f}, {result[\"norm_fiber_lengths\"].max():.4f}]')
print(f'Final force: {result[\"forces\"][-1]:.2f}N')
print(f'Final angle: {np.degrees(result[\"elbow_angles\"][-1]):.2f}deg')
"
```

- [ ] **Step 5: Update progress doc**

Write `docs/progress/2026-03-31-simple-arm-stage3-xpbd.md` documenting:
- XPBD vs VBD comparison (VBD exploded, XPBD stable)
- XPBD vs OpenSim DGF RMSE
- Parameter values that worked
- Key findings (ARAP effect, contraction_factor dynamics, etc.)

- [ ] **Step 6: Commit**

```bash
git add docs/progress/2026-03-31-simple-arm-stage3-xpbd.md
git add data/simpleArm/config.json  # if parameters were tuned
git commit -m "docs: add XPBD Stage 3 results and tuned parameters"
```

---

## Debugging Guide

### If mesh explodes (NaN positions)
1. Reduce `num_substeps` first? No — **increase** it (20→40). More substeps = smaller dt = more stable.
2. Reduce `attach_stiffness` (1e6→1e5). High stiffness with few substeps causes oscillation.
3. Check `pin_stiffness` isn't too high (1e8→1e7).

### If forces don't match DGF curve
1. Check `contraction_factor` is being updated (print `cf` each step).
2. Check `fiber_stiffness_scale` — too low means weak fiber response.
3. Check `l_tilde_xpbd` is reasonable (should be 0.5-1.5 for physiological range).

### If mesh distorts but doesn't explode
1. Increase `arap_stiffness` (500→2000). ARAP prevents shape distortion.
2. Check individual tet stretches (not just mean) — look for outliers.
3. Increase mesh density (`n_axial`: 10→15, `n_circumferential`: 8→12).

### If elbow angle diverges from OpenSim
1. The coupling is one-way (XPBD→force→MuJoCo), so check force magnitude.
2. If force too high: reduce `fiber_stiffness_scale`.
3. If force too low: increase `fiber_stiffness_scale` or check contraction_factor.
