# NOTE: This module has a broken import — `read_auxiliary_meshes` was removed
# from VMuscle.mesh_io during the 2026-03-10 SimConfig refactor.
# Calling `add_aux_meshes` will raise ImportError at runtime.
# Used by: examples/example_human_import2.py

import newton

def add_aux_meshes(builder:newton.ModelBuilder):
    from VMuscle.mesh_io import read_auxiliary_meshes  # noqa: F811 — broken, see module note
    ground_v,ground_ind, coord_v, coord_ind = read_auxiliary_meshes()
    ground = newton.Mesh(
            vertices=ground_v,
            indices=ground_ind.reshape(-1),
        )
    coord = newton.Mesh(
            vertices=coord_v,
            indices=coord_ind.reshape(-1),
        )
    builder.add_shape_mesh(-1,None,ground)
    builder.add_shape_mesh(-1,None,coord)
    return