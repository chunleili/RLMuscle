"""Load TetMesh muscles from Human.usd and output animated USD for Houdini.

Demonstrates:
  1. Loading TetMesh muscles via newton.usd.get_tetmesh() (new API)
  2. Adding muscles as soft bodies via builder.add_soft_mesh(mesh=...)
  3. Simple per-step motion, exported to USD for Houdini inspection

Usage:
  RUN=example_tetmesh_import uv run main.py
  uv run python -m examples.example_tetmesh_import
"""

from pathlib import Path
from zlib import crc32

import numpy as np
import warp as wp
import newton
import newton.usd

try:
    import newton_usd_schemas
except Exception:
    pass

from pxr import Usd, UsdGeom

INPUT_USD = Path(__file__).resolve().parents[1] / "data" / "Human" / "Human.usd"
OUTPUT_USD = Path(__file__).resolve().parents[1] / "output" / "tetmesh_test.usd"

TEST_MUSCLES = [
    "/Human/muscle/arms/R_bicepsBrachialis/R_bicepsBrachialis_Tet",
    "/Human/muscle/arms/L_bicepsBrachialis/L_bicepsBrachialis_Tet",
    "/Human/muscle/legs/R_rectusFemoris/R_rectusFemoris_Tet",
    "/Human/muscle/legs/L_rectusFemoris/L_rectusFemoris_Tet",
]

MUSCLE_ATTRS = [
    ("materialW", wp.vec3),
    ("muscleendmask", wp.float32),
    ("muscletobonemask", wp.float32),
    ("tendonmask", wp.float32),
    ("tensionendmask", wp.float32),
]


def _muscle_name_to_id(name: str) -> np.int32:
    """Deterministic string -> int32 hash for muscle identification."""
    return np.int32(crc32(name.encode()) & 0x7FFFFFFF)


def main():
    OUTPUT_USD.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.Open(str(INPUT_USD))
    tet_prims = newton.usd.find_tetmesh_prims(stage)
    print(f"Found {len(tet_prims)} TetMesh prims (muscles)")

    builder = newton.ModelBuilder()

    # Register custom attributes carried by TetMesh muscles
    for attr_name, dtype in MUSCLE_ATTRS:
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name=attr_name,
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                dtype=dtype,
                default=0.0,
            )
        )

    builder.add_custom_attribute(
        newton.ModelBuilder.CustomAttribute(
            name="muscle_id",
            frequency=newton.Model.AttributeFrequency.PARTICLE,
            dtype=wp.int32,
            default=-1,
        )
    )

    muscle_hash_to_name = {}
    muscle_name_to_hash = {}

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

        # Replace string muscle_id with per-particle int32 hash
        pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("muscle_id")
        if pv and tetmesh.custom_attributes and "muscle_id" in tetmesh.custom_attributes:
            muscle_name = str(pv.Get()[pv.GetIndices()[0]])
            mhash = _muscle_name_to_id(muscle_name)
            muscle_hash_to_name[int(mhash)] = muscle_name
            muscle_name_to_hash[muscle_name] = int(mhash)
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

    # Build model
    model = builder.finalize()
    state = model.state()

    print(f"\nModel: {model.particle_count} particles, {model.tet_count} tets, {model.tri_count} tris")

    # Simple motion + USD output
    fps = 24
    num_frames = 60
    frame_dt = 1.0 / fps

    viewer = newton.viewer.ViewerUSD(output_path=str(OUTPUT_USD), fps=fps, up_axis="Y")
    viewer.set_model(model)

    t = 0.0
    for frame in range(num_frames):
        q = state.particle_q.numpy()
        q[:, 1] += 0.01
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


if __name__ == "__main__":
    main()
