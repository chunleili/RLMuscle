"""Layered USD export for rigid-flexible coupling (IO-only demo).

Reads muscle TetMesh + bone Meshes from a single USD via UsdIO,
applies a trivial per-frame translation (placeholder for real simulation),
and writes deformed vertices + bone transforms into a layered USD file.

The source USD is never modified; all edits go into the output layer.

Usage:
    RUN=example_bicep_usd_io uv run main.py --viewer usd
    RUN=example_bicep_usd_io uv run main.py --viewer null --headless --num-frames 60
"""

import logging

import numpy as np
import warp as wp

import newton
import newton.examples
from VMuscle.usd_io import UsdIO, usd_args
from VMuscle.visualization import ViewerVisualization

log = logging.getLogger(__name__)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_frame = 0

        # --- 1. Read all meshes from USD (preserves prim paths) ---
        self.usd = UsdIO(
            source_usd_path=str(args.usd_path),
            root_path=str(args.usd_root_path),
            y_up_to_z_up=False,
        ).read()
        
        # Identify muscle and bone meshes by prim path
        self.muscle_mesh = self.usd.find_mesh("muscle")
        self.bone_meshes = self.usd.find_meshes("bone")

        log.info("Muscle: %s  verts=%d  tets=%d",
                 self.muscle_mesh.mesh_path,
                 self.muscle_mesh.vertices.shape[0],
                 self.muscle_mesh.tets.shape[0] if self.muscle_mesh.tets is not None else 0)
        if self.muscle_mesh.primvars:
            log.info("  primvars: %s", list(self.muscle_mesh.primvars.keys()))
        for bm in self.bone_meshes:
            log.info("Bone: %s  verts=%d", bm.mesh_path, bm.vertices.shape[0])

        # Warp arrays for viewer rendering
        self.warp_data = self.usd.warp_mesh_data()

        # Minimal Newton model (empty, just for viewer camera)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.viewer.set_model(self.model)
        self.vis = ViewerVisualization(self.viewer, self.usd.focus_points)

        # --- 2. Start layered USD output ---
        if args.use_layered_usd:
            self.usd.start(args.output_path, copy_usd=args.copy_usd)
            self.usd.set_runtime("fps", self.fps)
            log.info("Layered USD output: %s", self.usd.output_path)

        self._use_layered_usd = args.use_layered_usd

    def step(self):
        # --- Placeholder simulation: translate all muscle verts upward ---
        for wd in self.warp_data:
            pos_np = wd.pos.numpy()
            pos_np[:, 1] += 0.0005  # small Y-up shift per frame
            wd.pos = wp.array(pos_np, dtype=wp.vec3)

        self.sim_frame += 1
        self.sim_time += self.frame_dt
        if self.sim_frame % 10 == 0:
            log.info("step %d  t=%.3f", self.sim_frame, self.sim_time)

    def render(self):
        # self.vis.update_focus_hotkey()
        # self.viewer.begin_frame(self.sim_time)
        # for wd in self.warp_data:
        #     self.viewer.log_mesh(wd.name, wd.pos, wd.tri_indices)
        # self.vis.log_debug_visuals()
        # self.viewer.end_frame()

        # --- Write to layered USD ---
        if self._use_layered_usd:
            # Deformed muscle vertices
            if self.muscle_mesh:
                muscle_wd = next(
                    (w for w in self.warp_data if w.name == self.muscle_mesh.mesh_path), None
                )
                if muscle_wd:
                    self.usd.set_points(
                        self.muscle_mesh.mesh_path, muscle_wd.pos, frame=self.sim_frame
                    )

            # Bone transforms (write xform on parent Xform prim, not Shape)
            dy = self.sim_frame * 0.0005
            identity_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # w,x,y,z
            for bm in self.bone_meshes:
                xform_path = str(bm.mesh_path).rsplit("/", 1)[0]  # e.g. /character/bone/L_radius
                self.usd.set_xform(
                    xform_path,
                    pos=np.array([0.0, dy, 0.0]),
                    quat_wxyz=identity_quat,
                    frame=self.sim_frame,
                )

            # Custom runtime data
            self.usd.set_runtime("sim_frame", self.sim_frame, frame=self.sim_frame)

    def close(self):
        self.usd.close()
        log.info("Layered USD saved: %s", self.usd.output_path)

    def run(self,args):
        for frame in range(args.num_frames):
            self.step()
            self.render()




def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")

    parser = usd_args(
        "data/muscle/model/bicep.usd",
        "output/example_couple_usd_export.anim.usd",
    )
    parser.set_defaults(viewer="null")
    parser.set_defaults(num_frames="100")

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    try:
        example.run(args)
    finally:
        example.close()


if __name__ == "__main__":
    main()
