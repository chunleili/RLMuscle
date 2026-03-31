"""XPBD coupled SimpleArm example (Stage 3).

Builds a MuscleSim programmatically from a cylinder mesh with 5 constraint
types: TETVOLUME + TETFIBERDGF + TETARAP + PIN + ATTACH.  PIN constraints
fix the origin vertices; ATTACH constraints elastically couple insertion
vertices to bone target positions.

Usage:
    RUN=example_xpbd_coupled_simple_arm uv run main.py
    uv run -m examples.example_xpbd_coupled_simple_arm
"""

import json
from types import SimpleNamespace

import numpy as np
import warp as wp

from VMuscle.activation import activation_dynamics_step_np  # noqa: F401
from VMuscle.constraints import ATTACH, PIN, TETARAP, TETFIBERDGF, TETVOLUME
from VMuscle.dgf_curves import active_force_length  # noqa: F401
from VMuscle.mesh_io import MeshExporter  # noqa: F401
from VMuscle.mesh_utils import create_cylinder_tet_mesh  # noqa: F401
from VMuscle.muscle_warp import (
    Constraint,
    MuscleSim,
    fill_float_kernel,  # noqa: F401
    update_cons_restdir1_kernel,  # noqa: F401
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
# Entry point (stub for Task 2)
# ---------------------------------------------------------------------------

def main():
    pass


if __name__ == "__main__":
    main()
