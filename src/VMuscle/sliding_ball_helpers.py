"""Shared helpers for sliding-ball XPBD examples.

Extracts duplicated code from example_xpbd_dgf_sliding_ball.py and
example_xpbd_millard_sliding_ball.py into reusable functions:
- build_xpbd_sliding_ball_sim: procedural cylinder mesh + MuscleSim construction
- compute_rest_matrices: inverse rest-pose matrices for fiber analysis
- load_sliding_ball_config_base: common config flattening (geometry/physics/muscle/activation/solver)
"""

import json
from types import SimpleNamespace

import numpy as np
import warp as wp

from VMuscle.mesh_utils import create_cylinder_tet_mesh, assign_fiber_directions
from VMuscle.muscle_warp import MuscleSim


def load_sliding_ball_config_base(path):
    """Load sliding-ball JSON and flatten common sections into a flat dict.

    Returns a dict with standard keys from geometry, physics, muscle,
    activation, and solver sections. Callers extend with backend-specific keys.
    Also returns the raw JSON dict for further extraction.
    """
    with open(path) as f:
        raw = json.load(f)
    geo = raw["geometry"]
    phys = raw["physics"]
    mus = raw["muscle"]
    act = raw["activation"]
    sol = raw["solver"]
    cfg = {
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
    }
    return cfg, raw


def build_xpbd_sliding_ball_sim(cfg, constraint_configs,
                                contraction_ratio=0.0,
                                sigma0=0.0, lambda_opt=1.0):
    """Build XPBD MuscleSim from procedural cylinder mesh for sliding-ball.

    Handles: mesh creation, tet winding fix, per-vertex fiber averaging,
    manual MuscleSim construction, top-vertex fixing, bottom ball-mass.

    Args:
        cfg: Flat config dict with keys: length, radius, n_circ, n_axial,
             up_axis, density, gravity, ball_mass, dt, num_substeps,
             veldamping, fiber_stiffness_scale.
        constraint_configs: List of constraint dicts for MuscleSim.
        contraction_ratio: Initial contraction ratio (DGF uses dynamic value, Millard uses 0).
        sigma0: Peak isometric stress for MuscleSim config.
        lambda_opt: Optimal fiber stretch ratio for MuscleSim config.

    Returns:
        (sim, bottom_ids, axis_idx, tets, vertices, fiber_dirs_per_tet)
    """
    length = cfg["length"]
    radius = cfg["radius"]
    axis_idx = {"X": 0, "Y": 1, "Z": 2}[cfg["up_axis"]]

    # Create cylinder tet mesh
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
    v_fiber /= np.maximum(norms, 1e-8)

    # Build SimConfig
    sim_cfg = SimpleNamespace(
        geo_path="<procedural>",
        bone_geo_path="<none>",
        gui=False,
        render_mode="none",
        constraints=constraint_configs,
        dt=cfg["dt"],
        num_substeps=cfg["num_substeps"],
        gravity=cfg["gravity"],
        density=cfg["density"],
        veldamping=cfg["veldamping"],
        contraction_ratio=contraction_ratio,
        fiber_stiffness_scale=cfg["fiber_stiffness_scale"],
        sigma0=sigma0,
        lambda_opt=lambda_opt,
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
    sim.contraction_ratio = contraction_ratio
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
    print(f"Ball mass: {ball_mass} kg distributed to {len(bottom_ids)} bottom verts")

    sim.mass = wp.from_numpy(mass_np.astype(np.float32), dtype=wp.float32)
    sim.stopped = wp.from_numpy(stopped_np.astype(np.int32), dtype=wp.int32)

    return sim, bottom_ids, axis_idx, tets, vertices, fiber_dirs_per_tet


def compute_rest_matrices(vertices, tets):
    """Compute inverse rest-pose matrices for tet mesh.

    Uses XPBD convention: columns are [p0-p3, p1-p3, p2-p3].

    Returns:
        (M, 3, 3) float32 array of inverse rest matrices.
    """
    n_tet = len(tets)
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
    return rest_matrices
