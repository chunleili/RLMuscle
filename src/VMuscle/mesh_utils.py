"""Mesh utilities for volumetric muscle (vmuscle) in VBD solver.

Provides helper functions to load muscle TetMesh data into Newton's
ModelBuilder with vmuscle fiber properties.
"""

import numpy as np


def set_vmuscle_properties(builder, tet_offset, fiber_dirs, sigma0,
                           fiber_damping=None):
    """Batch-set vmuscle properties for a range of tets already in the builder.

    Automatically registers vmuscle custom attributes (idempotent) via
    ``SolverVBD.register_custom_attributes``, so the caller does not need
    to do it manually.

    Args:
        builder: Newton ModelBuilder that already contains the tets.
        tet_offset: Index of the first tet in the batch (from before add_soft_mesh).
        fiber_dirs: Per-tet fiber directions, shape (n_tets, 3).
        sigma0: Scalar or per-tet array of peak isometric stress [Pa].
        fiber_damping: Fiber viscous damping coefficient. If None, keeps default (0.0).
    """
    from newton.solvers import SolverVBD
    SolverVBD.register_custom_attributes(builder)

    fiber_dirs = np.asarray(fiber_dirs, dtype=np.float32)
    n_tets = len(fiber_dirs)
    if np.ndim(sigma0) == 0:
        sigma0 = np.full(n_tets, float(sigma0), dtype=np.float32)
    else:
        sigma0 = np.asarray(sigma0, dtype=np.float32)

    fd_attr = builder.custom_attributes["vmuscle:fiber_dirs"]
    s0_attr = builder.custom_attributes["vmuscle:sigma0"]
    for t in range(n_tets):
        tet_id = tet_offset + t
        fd_attr.values[tet_id] = tuple(fiber_dirs[t])
        s0_attr.values[tet_id] = float(sigma0[t])

    if fiber_damping is not None:
        builder.custom_attributes["vmuscle:fiber_damping"].values[0] = float(fiber_damping)



def add_soft_vmuscle_mesh(
    builder,
    mesh,
    cfg,
    pos=(0.0, 0.0, 0.0),
    rot=(0.0, 0.0, 0.0, 1.0),
    scale=1.0,
    vel=(0.0, 0.0, 0.0),
):
    """Import a TetMesh with volumetric muscle properties into ModelBuilder.

    Reads 'materialW', 'muscle_id', 'tendonmask' from mesh.custom_attributes.
    Converts per-vertex attributes to per-tet and populates vmuscle custom
    attributes via the Newton custom attribute system.

    Args:
        builder: Newton ModelBuilder instance.
        mesh: TetMesh loaded via newton.usd.get_tetmesh(), must contain
              custom_attributes 'materialW', 'muscle_id', 'tendonmask'.
        cfg: SimConfig with fields: tendon_threshold, muscles, etc.
        pos: Translation vector.
        rot: Rotation quaternion (x, y, z, w).
        scale: Uniform scale factor.
        vel: Initial velocity.
    """
    from newton.solvers import SolverVBD
    SolverVBD.register_custom_attributes(builder)

    # Record offsets before adding mesh
    tet_offset = len(builder.tet_indices)
    particle_offset = builder.particle_count

    # Add the standard soft mesh (particles + tets + Neo-Hookean)
    builder.add_soft_mesh(mesh, pos=pos, rot=rot, scale=scale, vel=vel)

    n_new_tets = len(builder.tet_indices) - tet_offset

    # Extract custom attributes from mesh
    attrs = mesh.custom_attributes
    fiber_dir_data = attrs.get("materialW")  # (array, frequency) tuple
    muscle_id_data = attrs.get("muscle_id")
    tendonmask_data = attrs.get("tendonmask")

    fd_attr = builder.custom_attributes["vmuscle:fiber_dirs"]
    s0_attr = builder.custom_attributes["vmuscle:sigma0"]

    if fiber_dir_data is None:
        # No muscle data — defaults (0-vec, 0.0) are already set by custom attribute
        return

    # Unpack arrays (drop frequency info)
    fiber_dirs_vert = np.asarray(fiber_dir_data[0], dtype=np.float32)
    muscle_ids_vert = (
        np.asarray(muscle_id_data[0], dtype=np.int32)
        if muscle_id_data is not None
        else np.zeros(len(fiber_dirs_vert), dtype=np.int32)
    )
    tendonmask_vert = (
        np.asarray(tendonmask_data[0], dtype=np.float32)
        if tendonmask_data is not None
        else np.zeros(len(fiber_dirs_vert), dtype=np.float32)
    )

    tendon_threshold = getattr(cfg, "tendon_threshold", 0.7)
    muscles_cfg = getattr(cfg, "muscles", {})
    default_sigma0 = getattr(cfg, "default_sigma0", 3e5)

    # Tet indices from builder (4 per tet, as list of tuples)
    tet_indices_flat = mesh.tet_indices.reshape(-1, 4)

    for t in range(n_new_tets):
        tet_id_global = tet_offset + t
        # Get vertex indices of this tet (relative to mesh, not global)
        v_ids = tet_indices_flat[t]

        # Per-vertex -> per-tet: fiber direction (average + normalize)
        fd = fiber_dirs_vert[v_ids].mean(axis=0)
        fd_norm = np.linalg.norm(fd)
        if fd_norm > 1e-8:
            fd = fd / fd_norm
        else:
            fd = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Per-vertex -> per-tet: muscle_id (take first vertex)
        mid = int(muscle_ids_vert[v_ids[0]])

        # Per-vertex -> per-tet: tendonmask (average)
        tm = float(tendonmask_vert[v_ids].mean())

        # Determine sigma0 from config
        mid_str = str(mid)
        if mid_str in muscles_cfg:
            sigma0 = muscles_cfg[mid_str].get("sigma0", default_sigma0)
        elif isinstance(muscles_cfg, dict) and mid_str in muscles_cfg:
            sigma0 = muscles_cfg[mid_str]["sigma0"]
        else:
            sigma0 = default_sigma0

        # Tendon tets have sigma0 = 0 (no active contraction)
        if tm >= tendon_threshold:
            sigma0 = 0.0

        fd_attr.values[tet_id_global] = tuple(fd)
        s0_attr.values[tet_id_global] = float(sigma0)



def create_cylinder_tet_mesh(length=0.1, radius=0.02, n_length=8, n_radial=6):
    """Generate a cylinder tetrahedral mesh for testing.

    The cylinder extends along the Z axis from 0 to length.

    Args:
        length: Cylinder length [m].
        radius: Cylinder radius [m].
        n_length: Number of segments along length.
        n_radial: Number of radial segments.

    Returns:
        (vertices, tet_indices): NumPy arrays, vertices (N,3), tet_indices (M,4).
    """
    vertices = []
    # Create vertices: center axis + radial ring at each cross-section
    for iz in range(n_length + 1):
        z = length * iz / n_length
        # Center vertex
        vertices.append([0.0, 0.0, z])
        # Radial vertices
        for ir in range(n_radial):
            theta = 2.0 * np.pi * ir / n_radial
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            vertices.append([x, y, z])

    vertices = np.array(vertices, dtype=np.float32)
    n_per_section = 1 + n_radial  # 1 center + n_radial ring

    tets = []
    for iz in range(n_length):
        base0 = iz * n_per_section
        base1 = (iz + 1) * n_per_section
        c0 = base0  # center of lower section
        c1 = base1  # center of upper section

        for ir in range(n_radial):
            ir_next = (ir + 1) % n_radial
            # Lower ring vertex indices
            r0a = base0 + 1 + ir
            r0b = base0 + 1 + ir_next
            # Upper ring vertex indices
            r1a = base1 + 1 + ir
            r1b = base1 + 1 + ir_next

            # Create tets connecting the two sections
            # Tet 1: c0, r0a, r0b, r1a
            tets.append([c0, r0a, r0b, r1a])
            # Tet 2: c1, r1a, r1b, r0b
            tets.append([c1, r1a, r1b, r0b])
            # Tet 3: c0, c1, r0b, r1a
            tets.append([c0, c1, r0b, r1a])

    tets = np.array(tets, dtype=np.int32)
    # Fix winding: ensure positive volume for all tets
    tets = fix_tet_winding(vertices, tets)
    return vertices, tets


def fix_tet_winding(vertices, tets):
    """Fix tet winding so all tetrahedra have positive volume.

    Swaps vertex 0 and 1 for any inverted tet (negative signed volume).

    Args:
        vertices: Vertex positions (N, 3).
        tets: Tet indices (M, 4).

    Returns:
        Fixed tet indices (M, 4).
    """
    tets = tets.copy()
    p0 = vertices[tets[:, 0]]
    p1 = vertices[tets[:, 1]]
    p2 = vertices[tets[:, 2]]
    p3 = vertices[tets[:, 3]]
    signed6v = np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), p3 - p0)
    inverted = signed6v < 0.0
    if np.any(inverted):
        t0 = tets[inverted, 0].copy()
        tets[inverted, 0] = tets[inverted, 1]
        tets[inverted, 1] = t0
    return tets


def assign_fiber_directions(vertices, tets, axis=2):
    """Assign fiber directions along a specified axis for each tet.

    Args:
        vertices: Vertex positions (N, 3).
        tets: Tet indices (M, 4).
        axis: 0=X, 1=Y, 2=Z.

    Returns:
        fiber_dirs: Per-tet fiber directions (M, 3), unit vectors.
    """
    fiber_dirs = np.zeros((len(tets), 3), dtype=np.float32)
    fiber_dirs[:, axis] = 1.0
    return fiber_dirs
