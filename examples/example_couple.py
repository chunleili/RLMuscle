"""Couple MuscleSim (Taichi PBD) with Newton rigid-body skeleton.

Scapula and humerus are static geometry; radius is a kinematically-driven
rigid body connected to the world via a revolute joint at the elbow.
Each frame the joint angle is set via a sine wave, FK computes body_q,
bone vertex positions are synced to MuscleSim's bone_pos_field, and
MuscleSim attach constraints pull the muscle to follow the bone.
"""

import math
import numpy as np
import newton
import newton.examples
import warp as wp

from RLVometricMuscle.usd_io import UsdIO, usd_args
from RLVometricMuscle.visualization import ViewerVisualization
from RLVometricMuscle.solver_volumetric_muscle import SolverVolumetricMuscle
from RLVometricMuscle.solver_muscle_bone_coupled import _quat_rotate_batch

# Elbow pivot in Z-up centered space (from USD skeleton L_lowerarm bind transform)
ELBOW_PIVOT = wp.vec3(0.06, 0.02, 0.26)
# Elbow flexion axis (roughly along Y in Z-up space)
ELBOW_AXIS = wp.vec3(0.0, 1.0, 0.0)


class Example:
    def __init__(self, viewer, args):
        self.args = args
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.sim_frame = 0

        self._usd = UsdIO(
            source_usd_path=str(args.usd_path),
            root_path=str(args.usd_root_path),
            y_up_to_z_up=True,
            center_model=True,
            up_axis=int(newton.Axis.Z),
        ).read()

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)

        self._radius_link = None

        for mesh in self._usd.meshes:
            if mesh.tets is not None and mesh.tets.size > 0:
                # Muscle — volumetric soft body (for Newton viewer particle rendering)
                builder.add_soft_mesh(
                    pos=wp.vec3(0.0, 0.0, 0.0),
                    rot=wp.quat_identity(),
                    scale=1.0,
                    vel=wp.vec3(0.0, 0.0, 0.0),
                    vertices=mesh.vertices.tolist(),
                    indices=mesh.tets.reshape(-1).tolist(),
                    density=1000.0,
                    k_mu=2000.0,
                    k_lambda=2000.0,
                    k_damp=2.0,
                )
            elif "radius" in mesh.mesh_path.lower():
                # Radius — rigid body with revolute joint (driven kinematically)
                self._radius_link = builder.add_link(
                    xform=wp.transform(),
                )
                builder.add_shape_mesh(
                    body=self._radius_link,
                    xform=wp.transform(),
                    mesh=newton.Mesh(
                        vertices=mesh.vertices,
                        indices=mesh.faces.reshape(-1),
                        compute_inertia=True,
                        is_solid=True,
                        color=mesh.color,
                    ),
                )
                joint = builder.add_joint_revolute(
                    parent=-1,
                    child=self._radius_link,
                    axis=ELBOW_AXIS,
                    parent_xform=wp.transform(p=ELBOW_PIVOT),
                    child_xform=wp.transform(p=ELBOW_PIVOT),
                    limit_lower=-2.5,
                    limit_upper=0.1,
                )
                builder.add_articulation([joint], key="elbow")

            else:
                # Scapula / humerus — static geometry
                builder.add_shape_mesh(
                    body=-1,
                    xform=wp.transform(),
                    mesh=newton.Mesh(
                        vertices=mesh.vertices,
                        indices=mesh.faces.reshape(-1),
                        compute_inertia=True,
                        is_solid=True,
                        color=mesh.color,
                    ),
                )

        builder.add_ground_plane()

        self.model = builder.finalize()
        self.muscle_solver = SolverVolumetricMuscle(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Initialize body poses from joint configuration
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Tell muscle solver how to convert Y-up ↔ Z-up centered
        self.muscle_solver.set_coord_transform(self._usd._cached.center_shift)

        # Configure bone→muscle coupling
        self._coupling_configured = False
        if self._radius_link is not None:
            self._configure_coupling()

        self.viewer.set_model(self.model)
        self.vis = ViewerVisualization(self.viewer, self._usd.focus_points)

        if self.args.use_layered_usd:
            self._usd.start(self.args.output_path, copy_usd=self.args.copy_usd)
            self._usd.set_runtime("fps", self.fps)

    def _configure_coupling(self):
        """Set up the mapping from Newton's radius body to MuscleSim bone_pos_field."""
        core = self.muscle_solver.core

        if not hasattr(core, "bone_muscle_ids") or "L_radius" not in core.bone_muscle_ids:
            print("Warning: bone_muscle_ids['L_radius'] not found, coupling disabled.")
            return

        self._bone_radius_indices = core.bone_muscle_ids["L_radius"]
        self._center_shift = self._usd._cached.center_shift

        # Rest-pose bone positions in Z-up centered space (for FK transform)
        bone_pos_yup = core.bone_pos[self._bone_radius_indices]
        rest_zup = np.empty_like(bone_pos_yup)
        rest_zup[:, 0] = bone_pos_yup[:, 0]
        rest_zup[:, 1] = -bone_pos_yup[:, 2]
        rest_zup[:, 2] = bone_pos_yup[:, 1]
        self._bone_rest_verts_zup = (rest_zup - self._center_shift).astype(np.float32)

        self._coupling_configured = True
        print(f"Coupling configured: body={self._radius_link}, {len(self._bone_radius_indices)} radius verts")

    def _sync_bone_to_muscle(self, state):
        """Read body_q for radius, transform rest verts to world, convert to Y-up, update MuscleSim."""
        if not self._coupling_configured:
            return
        body_q = state.body_q.numpy()
        xform = body_q[self._radius_link]
        p = xform[:3]
        q = xform[3:]

        world_verts = _quat_rotate_batch(q, self._bone_rest_verts_zup) + p
        uncenter = world_verts + self._center_shift
        yup = np.empty_like(uncenter)
        yup[:, 0] = uncenter[:, 0]
        yup[:, 1] = uncenter[:, 2]
        yup[:, 2] = -uncenter[:, 1]

        self.muscle_solver.update_bone_positions(self._bone_radius_indices, yup)

    def simulate(self):
        # Kinematic drive: slow sine wave for elbow flexion
        # Range: 0 to -1.5 rad (0° to ~86° flexion)
        target_angle = -0.75 * (1.0 - math.cos(self.sim_time * 2.0))

        # Set joint angle and compute FK → body_q
        joint_q = self.model.joint_q.numpy()
        joint_q[0] = target_angle
        joint_q_wp = wp.from_numpy(joint_q, dtype=wp.float32, device=self.model.device)
        newton.eval_fk(self.model, joint_q_wp, self.model.joint_qd, self.state_0)

        # Sync bone positions to MuscleSim
        self._sync_bone_to_muscle(self.state_0)

        # Run muscle PBD
        self.muscle_solver.step(self.state_0, self.state_0)

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        self.sim_frame += 1

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
    parser = usd_args("data/muscle/model/bicep.usd", "output/example_couple.anim.usda")
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    try:
        newton.examples.run(example, args)
    finally:
        example.close()


if __name__ == "__main__":
    main()
