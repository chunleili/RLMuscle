"""Load TetMesh muscles from Human.usd and output animated USD for Houdini.

Demonstrates:
  1. Loading TetMesh muscles via newton.usd.get_tetmesh() (new API)
  2. Adding muscles as soft bodies via builder.add_soft_mesh(mesh=...)
  3. Simple per-step motion, exported to USD for Houdini inspection

Usage:
  uv run python -m examples.example_tetmesh_import
"""

from pathlib import Path

import numpy as np
import warp as wp
import newton
import newton.usd

try:
    import newton_usd_schemas
except Exception:
    pass

from pxr import Usd

INPUT_USD = Path(__file__).resolve().parents[1] / "data" / "Human" / "Human.usd"
OUTPUT_USD = Path(__file__).resolve().parents[1] / "output" / "tetmesh_test.usd"
OUTPUT_USD.parent.mkdir(parents=True, exist_ok=True)

# -- 1) Find all TetMesh prims ----------------------------------------------
stage = Usd.Stage.Open(str(INPUT_USD))
tet_prims = newton.usd.find_tetmesh_prims(stage)
print(f"Found {len(tet_prims)} TetMesh prims (muscles)")

# -- 2) Load a subset of muscles --------------------------------------------
TEST_MUSCLES = [
    "/Human/muscle/arms/R_bicepsBrachialis/R_bicepsBrachialis_Tet",
    "/Human/muscle/arms/L_bicepsBrachialis/L_bicepsBrachialis_Tet",
    "/Human/muscle/legs/R_rectusFemoris/R_rectusFemoris_Tet",
    "/Human/muscle/legs/L_rectusFemoris/L_rectusFemoris_Tet",
]

builder = newton.ModelBuilder()

for prim_path in TEST_MUSCLES:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"  SKIP: {prim_path} not found")
        continue

    tetmesh = newton.usd.get_tetmesh(prim)
    name = prim_path.split("/")[-1]
    n_verts = len(tetmesh.vertices)
    n_tets = len(tetmesh.tet_indices) // 4
    n_surf = len(tetmesh.surface_tri_indices) // 3

    attr_names = list(tetmesh.custom_attributes.keys()) if tetmesh.custom_attributes else []
    print(f"  {name}: {n_verts} verts, {n_tets} tets, {n_surf} surf tris, attrs={attr_names}")

    # Clear custom attributes for now (need registration for full pipeline)
    tetmesh.custom_attributes = None

    builder.add_soft_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        mesh=tetmesh,
        density=1000.0,
        k_mu=5000.0,
        k_lambda=5000.0,
        k_damp=10.0,
    )

# -- 3) Build model ----------------------------------------------------------
model = builder.finalize()
state = model.state()

n_particles = model.particle_count
n_tets_total = model.tet_count
n_tris_total = model.tri_count
print(f"\nModel: {n_particles} particles, {n_tets_total} tets, {n_tris_total} tris")

# -- 4) Simple motion + USD output -------------------------------------------
fps = 24
num_frames = 60  # 2.5 seconds
frame_dt = 1.0 / fps

viewer = newton.viewer.ViewerUSD(output_path=str(OUTPUT_USD), fps=fps, up_axis="Y")
viewer.set_model(model)

t = 0.0
for frame in range(num_frames):
    # Shift all particles slightly each frame to verify animation
    q = state.particle_q.numpy()
    q[:, 1] += 0.01  # move up Y by 0.01 per frame
    state.particle_q = wp.array(q, dtype=wp.vec3)

    viewer.begin_frame(t)
    viewer.log_state(state)
    viewer.end_frame()
    t += frame_dt

    if frame % 10 == 0:
        print(f"  frame {frame}/{num_frames}, mean_y = {q[:, 1].mean():.4f}")

viewer.close()
print(f"\nSaved: {OUTPUT_USD.resolve()}")
print("Open in Houdini to inspect the animated TetMesh muscles.")
