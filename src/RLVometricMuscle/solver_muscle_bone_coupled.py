import inspect

import numpy as np
import warp as wp

from newton.solvers import SolverFeatherstone
from .solver_volumetric_muscle import SolverVolumetricMuscle


class SolverMuscleBoneCoupled:
    def __init__(self, model, **solver_kwargs):
        self.model = model
        bone_sig = inspect.signature(SolverFeatherstone.__init__)
        bone_kwargs = {
            k: v for k, v in solver_kwargs.items() if k in bone_sig.parameters and k != "model"
        }
        self.bone_solver = SolverFeatherstone(model, **bone_kwargs)
        self.muscle_solver = SolverVolumetricMuscle(model, **solver_kwargs)
        self._coupling_configured = False

    def configure_coupling(
        self,
        bone_body_id: int,
        bone_rest_verts_zup: np.ndarray,
        bone_vertex_indices: np.ndarray,
        center_shift: np.ndarray,
    ):
        """Set up bone-to-muscle position sync.

        Args:
            bone_body_id: Newton body index for the dynamic bone (radius).
            bone_rest_verts_zup: Rest-pose vertices in Newton Z-up centered space, shape (N, 3).
            bone_vertex_indices: Indices into MuscleSim bone_pos_field for the dynamic bone vertices.
            center_shift: The center_shift applied by UsdIO (used for coordinate conversion).
        """
        self._bone_body_id = bone_body_id
        self._bone_rest_verts_zup = bone_rest_verts_zup.astype(np.float32)
        self._bone_vertex_indices = bone_vertex_indices.astype(np.int32)
        self._center_shift = center_shift.astype(np.float32)
        self._coupling_configured = True

    def _sync_bone_positions(self, state):
        """Read body_q for the dynamic bone, transform rest verts to world, convert to Y-up, update MuscleSim."""
        body_q_wp = state.body_q.numpy()
        xform = body_q_wp[self._bone_body_id]  # (7,) — px,py,pz, qx,qy,qz,qw
        p = xform[:3]
        q = xform[3:]  # (qx, qy, qz, qw)

        # Rotate rest vertices by quaternion and translate
        world_verts = _quat_rotate_batch(q, self._bone_rest_verts_zup) + p

        # Convert Z-up centered → original Y-up (.geo coordinate system)
        # Undo center: add back center_shift
        uncenter = world_verts + self._center_shift
        # Z-up (x, y, z) → Y-up (x, z, -y)
        yup = np.empty_like(uncenter)
        yup[:, 0] = uncenter[:, 0]
        yup[:, 1] = uncenter[:, 2]
        yup[:, 2] = -uncenter[:, 1]

        self.muscle_solver.update_bone_positions(self._bone_vertex_indices, yup)

    def step(self, state_in, state_out, control, contacts, dt):
        if control is None:
            control = self.model.control(clone_variables=False)

        # 1. Bone dynamics — pass contacts=None to avoid particle-body collision
        #    forces that would push the radius body (muscle particles overlap the
        #    radius mesh in Newton's coordinate space).
        self.bone_solver.step(state_in, state_out, control, None, dt)

        # 2. Sync bone positions to MuscleSim
        if self._coupling_configured:
            self._sync_bone_positions(state_out)

        # 3. Muscle PBD
        self.muscle_solver.step(state_in, state_out, control, contacts, dt)


def _quat_rotate_batch(q, points):
    """Rotate an array of points by a quaternion (qx, qy, qz, qw).

    Args:
        q: quaternion as (4,) array — (qx, qy, qz, qw)
        points: (N, 3) array

    Returns:
        Rotated points (N, 3).
    """
    qx, qy, qz, qw = q
    # Rotation matrix from quaternion
    R = np.array([
        [1 - 2*(qy*qy + qz*qz),  2*(qx*qy - qz*qw),      2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),      1 - 2*(qx*qx + qz*qz),  2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),      2*(qy*qz + qx*qw),      1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float32)
    return points @ R.T
