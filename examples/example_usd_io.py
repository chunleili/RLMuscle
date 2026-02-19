"""Minimal USD IO example: load USD meshes, render, and write layered runtime values."""

import newton
import newton.examples
import warp as wp
from VMuscle.usd_io import UsdIO, usd_args
from VMuscle.visualization import ViewerVisualization


class Example:
    def __init__(self, viewer, args):
        self.args = args
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_frame = 0

        self._usd = UsdIO(
            source_usd_path=str(args.usd_path),
            root_path=str(args.usd_root_path),
            y_up_to_z_up=True,
            center_model=True,
            up_axis=int(newton.Axis.Z),
        ).read()

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        for mesh in self._usd.meshes:
            builder.add_shape_mesh(
                body=-1,
                xform=wp.transform(),
                mesh=newton.Mesh(
                    vertices=mesh.vertices,
                    indices=mesh.faces.reshape(-1),
                    compute_inertia=False,
                    is_solid=True,
                    color=mesh.color,
                ),
            )
        builder.add_ground_plane()

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.viewer.set_model(self.model)
        self.vis = ViewerVisualization(self.viewer, self._usd.focus_points)

        if self.args.use_layered_usd:
            self._usd.start(self.args.output_path, copy_usd=self.args.copy_usd)
            self._usd.set_runtime("fps", self.fps)

    def step(self):
        self.sim_frame += 1
        self.sim_time += self.frame_dt

    def render(self):
        self.vis.update_focus_hotkey()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.vis.log_debug_visuals()
        self.viewer.end_frame()

        if self.args.use_layered_usd:
            self._usd.set_runtime("frame", self.sim_frame, frame=self.sim_frame)
            self._usd.set_runtime("sim_time", self.sim_time, frame=self.sim_frame)

    def close(self):
        self._usd.close()


def main():
    parser = usd_args("data/muscle/model/bicep.usd", "output/example_usd_io.anim.usda")
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    try:
        newton.examples.run(example, args)
    finally:
        example.close()


if __name__ == "__main__":
    main()
