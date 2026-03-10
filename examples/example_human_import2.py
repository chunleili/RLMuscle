###########################################################################
# Example Human Import 2 — Filtered USD Loading Test
#
# Loads 3 bones (L_upperarm, L_radius, L_clavicleScap) from Human.usd
# using ignore_paths filter. Each step translates all bodies upward
# to verify USD I/O pipeline.
#
# Command: RUN=example_human_import2 uv run main.py
#   or:    uv run python -m examples.example_human_import2
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from VMuscle.log import setup_logging
from VMuscle.usd_io import UsdIO
from VMuscle.add_tet_muscle import register_muscle_attributes, add_tet_muscles
from VMuscle.util import add_aux_meshes


# Bones to keep in the Newton model
KEEP_BONES = {"L_radius", "L_upperarm", "L_clavicleScap"}
KEEP_MUSCLES = {"L_bicepsBrachialis"}


def _build_ignore_paths(stage, keep_bones):
    """Build ignore_paths list: exclude all rigid bodies and joints not in keep_bones."""
    from pxr import UsdPhysics

    kept_body_paths = set()
    ignore = []

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "/Ragdoll/Bones/" in path and UsdPhysics.RigidBodyAPI(prim):
            if path.split("/")[-1] in keep_bones:
                kept_body_paths.add(path)
            else:
                ignore.append(path)

    # Ignore joints that reference any excluded body
    for prim in stage.Traverse():
        joint = UsdPhysics.Joint(prim)
        if not joint:
            continue
        targets = [
            str(t)
            for rel in (joint.GetBody0Rel(), joint.GetBody1Rel())
            for t in rel.GetTargets()
        ]
        if any(t not in kept_body_paths for t in targets if t):
            ignore.append(str(prim.GetPath()))

    return ignore


def _resolve_muscle_paths(stage, keep_muscles):
    """Resolve short muscle names to full TetMesh prim paths."""
    tet_prims = newton.usd.find_tetmesh_prims(stage)
    resolved = []
    for prim in tet_prims:
        path = str(prim.GetPath())
        name = path.split("/")[-1].replace("_Tet", "")
        if name in keep_muscles:
            resolved.append(path)
    return resolved


class Example():
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer
        self.log = setup_logging()

        # -- 1) Load filtered Newton model from Human.usd --------------------
        self.usd = UsdIO(
            source_usd_path=str(args.usd_path),
        )
        stage = self.usd._stage
        ignore_paths = _build_ignore_paths(stage, KEEP_BONES)
        self.log.info("Filtering USD: keep %d bones, ignore %d paths", len(KEEP_BONES), len(ignore_paths))

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
        try:
            newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        except Exception:
            pass

        register_muscle_attributes(builder)

        builder.add_usd(
            source=stage,
            root_path="/Human",
            ignore_paths=ignore_paths,
        )
        add_aux_meshes(builder)

        # Resolve muscle short names to full TetMesh prim paths
        muscle_paths = _resolve_muscle_paths(stage, KEEP_MUSCLES)
        self.log.info("Resolved muscle paths: %s", muscle_paths)
        (hash_to_name, name_to_hash) = add_tet_muscles(stage, builder, muscle_paths)

        # Identify muscle and bone meshes by prim path keywords
        self.bone_meshes = [m for m in self.usd.meshes if any(k in m.mesh_path for k in KEEP_BONES)]
        self.muscle_meshes = [m for m in self.usd.meshes if any(k in m.mesh_path for k in KEEP_MUSCLES)]

        self.model = builder.finalize()
        self.log.info(
            "Model finalized: %d bodies, %d joints",
            self.model.body_count, self.model.joint_count,
        )

        for i, label in enumerate(self.model.body_label):
            self.log.info("  body[%d] = %s", i, label)

        # -- Loading summary -------------------------------------------------
        self.log.info("Loaded: %d bone meshes, %d muscle meshes", len(self.bone_meshes), len(self.muscle_meshes))
        for bm in self.bone_meshes:
            self.log.info("  bone: %s (%d verts)", bm.mesh_path, len(bm.vertices))
        for mm in self.muscle_meshes:
            self.log.info("  muscle: %s (%d verts, %d tets)", mm.mesh_path, len(mm.vertices),
                          mm.tets.shape[0] if mm.tets is not None else 0)

        # -- 2) States -------------------------------------------------------
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        try:
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
            self.log.info("eval_fk OK")
        except Exception as e:
            self.log.warning("eval_fk skipped: %s", e)

        self.step_count = 0

        # Read local-space rest vertices from USD prim (not world-space bm.vertices)
        from pxr import UsdGeom
        self._bone_local_rest = []
        self._bone_body_idx = []
        for bm in self.bone_meshes:
            prim = stage.GetPrimAtPath(bm.mesh_path)
            pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            self._bone_local_rest.append(np.asarray(pts, dtype=np.float32))
            # Map to body index
            idx = -1
            for i, label in enumerate(self.model.body_label):
                if bm.mesh_path.startswith(label + "/"):
                    idx = i
                    break
            self._bone_body_idx.append(idx)

        # Current positions (mutable, updated each step)
        self._bone_points = [v.copy() for v in self._bone_local_rest]
        self._muscle_points = [mm.vertices.copy() for mm in self.muscle_meshes]
        self._initial_body_q = self.state_0.body_q.numpy().copy()

        self.viewer.set_model(self.model)

    # -----------------------------------------------------------------
    # Simulation loop
    # -----------------------------------------------------------------

    def step(self):
        self.step_count += 1
        self.sim_time += self.frame_dt

        # Translate all bodies upward by 0.001 per step
        body_q = self.state_0.body_q.numpy()
        body_q[:, 1] += 0.001  # Y-up
        self.state_0.body_q.assign(wp.array(body_q, dtype=wp.transform))

        # Update bone vertices from body_q delta
        body_q = self.state_0.body_q.numpy()
        for i, body_idx in enumerate(self._bone_body_idx):
            if body_idx < 0:
                continue
            delta = (body_q[body_idx, :3] - self._initial_body_q[body_idx, :3]).astype(np.float32)
            self._bone_points[i] = self._bone_local_rest[i] + delta

        # Offset muscle mesh vertices each step
        dy = np.array([0.0, 0.0005, 0.0], dtype=np.float32)
        for i in range(len(self._muscle_points)):
            self._muscle_points[i] += dy

        if self.step_count % 25 == 0 or self.step_count == 1:
            bq = self.state_0.body_q.numpy()
            self.log.info(
                "frame=%d t=%.4f | body_y=[%s]",
                self.step_count, self.sim_time,
                ", ".join(f"{bq[i, 1]:.4f}" for i in range(self.model.body_count)),
            )

    def render(self):
        if not hasattr(self, '_usd_started'):
            return
        frame = self.step_count

        # Write bone vertices
        for bm, pts in zip(self.bone_meshes, self._bone_points):
            self.usd.set_points(bm.mesh_path, pts, frame=frame)

        # Write muscle vertices
        for mm, pts in zip(self.muscle_meshes, self._muscle_points):
            self.usd.set_points(mm.mesh_path, pts, frame=frame)

        self.usd.set_runtime("sim_frame", frame, frame=frame)

    def run(self, args):
        num_frames = int(args.num_frames)
        # Initialize layered USD output
        self.usd.start()
        self._usd_started = True
        self.log.info("USD output started: %s", self.usd.output_path)

        for frame in range(num_frames):
            self.step()
            self.render()

    def close(self):
        self.usd.close()
        self.log.info("Layered USD saved: %s", self.usd.output_path)


def main():
    from VMuscle.config import load_conifg
    parser = newton.examples.create_parser()
    parser.set_defaults(viewer="null")
    parser.set_defaults(num_frames="100")
    viewer, args = newton.examples.init(parser)
    args = load_conifg(args, "data/Human/human_import2.json")
    example = Example(viewer, args)
    try:
        example.run(args)
    finally:
        example.close()


if __name__ == "__main__":
    main()
