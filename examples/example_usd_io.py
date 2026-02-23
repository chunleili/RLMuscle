"""Minimal USD IO example: load USD meshes, render, and write layered runtime values.

Demonstrates per-vertex deformation using viewer.log_mesh() instead of rigid-body
add_shape_mesh(). Each frame the vertices are modified directly and re-uploaded to
the GL vertex buffer.
"""

import newton
import newton.examples
import warp as wp
from VMuscle.usd_io import UsdIO, usd_args
from VMuscle.visualization import ViewerVisualization


@wp.kernel
def _single_vertex_deform_kernel(
    src: wp.array(dtype=wp.vec3),
    dst: wp.array(dtype=wp.vec3),
    target_vertex: int,
    time: float,
    amplitude: float,
):
    """Move only one vertex along Z; all others stay at rest pose."""
    i = wp.tid()
    p = src[i]
    if i == target_vertex:
        dz = amplitude * wp.sin(time * 4.0)
        dst[i] = wp.vec3(p[0], p[1], p[2] + dz)
    else:
        dst[i] = p


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
        ).read()

        self.mesh_data = self._usd.warp_mesh_data()

        # Still need a Newton model for viewer camera / focus handling
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.viewer.set_model(self.model)
        self.vis = ViewerVisualization(self.viewer, self._usd.focus_points)

        if self.args.use_layered_usd:
            self._usd.start(self.args.output_path, copy_usd=self.args.copy_usd)
            self._usd.set_runtime("fps", self.fps)

    def step(self):
        # Move a single vertex to demonstrate per-vertex modification
        for md in self.mesh_data:
            wp.launch(
                _single_vertex_deform_kernel,
                dim=md.rest_pos.shape[0],
                inputs=[md.rest_pos, md.pos, 0, self.sim_time, 0.5],
            )
        self.sim_frame += 1
        self.sim_time += self.frame_dt

    def render(self):
        self.vis.update_focus_hotkey()
        self.viewer.begin_frame(self.sim_time)
        # Render deformed meshes directly (bypasses rigid-body state)
        for md in self.mesh_data:
            self.viewer.log_mesh(md.name, md.pos, md.tri_indices)
        self.vis.log_debug_visuals()
        self.viewer.end_frame()

        if self.args.use_layered_usd:
            for md in self.mesh_data:
                self._usd.set_points(md.name, md.pos, frame=self.sim_frame)

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
