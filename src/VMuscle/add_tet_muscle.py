"""Utilities for loading TetMesh muscles from USD into Newton.

Shared by example_tetmesh_import and example_human_import2.
"""

from __future__ import annotations

import logging
from zlib import crc32

import numpy as np
import warp as wp
import newton


def muscle_name_to_id(name: str) -> np.int32:
    """Deterministic string -> int32 hash for muscle identification."""
    return np.int32(crc32(name.encode()) & 0x7FFFFFFF)


def register_muscle_attributes(builder: newton.ModelBuilder) -> None:
    """Register standard muscle custom attributes on a ModelBuilder."""
    muscle_attrs = [
        ("materialW", wp.vec3),
        ("muscleendmask", wp.float32),
        ("muscletobonemask", wp.float32),
        ("muscletomuscleglue", wp.float32),
        ("tendonmask", wp.float32),
        ("tensionendmask", wp.float32),
    ]
    for attr_name, dtype in muscle_attrs:
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name=attr_name,
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                dtype=dtype,
                default=0.0,
            )
        )
    # Integer muscle_id (USD stores as string, we convert to int hash)
    builder.add_custom_attribute(
        newton.ModelBuilder.CustomAttribute(
            name="muscle_id",
            frequency=newton.Model.AttributeFrequency.PARTICLE,
            dtype=wp.int32,
            default=-1,
        )
    )


def add_tet_muscles(
    stage,
    builder: newton.ModelBuilder,
    muscle_prim_paths: list[str],
) -> tuple[dict[int, str], dict[str, int]]:
    """Add TetMesh muscles from *stage* and add them as soft bodies to *builder*.

    Returns:
        (hash_to_name, name_to_hash) mapping dicts for muscle identification.
    """
    from pxr import UsdGeom

    hash_to_name: dict[int, str] = {}
    name_to_hash: dict[str, int] = {}

    for prim_path in muscle_prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            logging.getLogger(__name__).warning("SKIP: %s not found", prim_path)
            continue

        tetmesh = newton.usd.get_tetmesh(prim)
        name = prim_path.split("/")[-1]
        n_verts = len(tetmesh.vertices)
        n_tets = len(tetmesh.tet_indices) // 4
        n_surf = len(tetmesh.surface_tri_indices) // 3

        attr_names = list(tetmesh.custom_attributes.keys()) if tetmesh.custom_attributes else []
        logging.getLogger(__name__).info(
            "  %s: %d verts, %d tets, %d surf tris, attrs=%s",
            name, n_verts, n_tets, n_surf, attr_names)

        # Replace string muscle_id with per-particle int32 hash.
        # get_tetmesh() returns the global name list, so we resolve
        # the actual muscle name via USD indexed primvar.
        pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("muscle_id")
        if pv and tetmesh.custom_attributes and "muscle_id" in tetmesh.custom_attributes:
            muscle_name = str(pv.Get()[pv.GetIndices()[0]])
            mhash = muscle_name_to_id(muscle_name)
            hash_to_name[int(mhash)] = muscle_name
            name_to_hash[muscle_name] = int(mhash)
            tetmesh.custom_attributes["muscle_id"] = (
                np.full(n_verts, mhash, dtype=np.int32),
                newton.Model.AttributeFrequency.PARTICLE,
            )

        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            mesh=tetmesh,
        )

    return hash_to_name, name_to_hash
