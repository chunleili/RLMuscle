"""Bidirectional muscle-bone coupling solver.

Couples Newton rigid-body dynamics with Taichi MuscleSim PBD.
Spring force model: F = -k_coupling * sum(C*n) / N_substeps.
"""

import logging

import numpy as np
import taichi as ti
import warp as wp

from newton.solvers import SolverMuJoCo, SolverFeatherstone
from .muscle import MuscleSim

log = logging.getLogger("couple")


@ti.data_oriented
class SolverMuscleBoneCoupled:
    """Couples Newton rigid-body bone dynamics with Taichi MuscleSim PBD.

    Both systems operate in Y-up coordinate space — no conversion needed.

    Per-step coupling flow:
      1. Apply previous frame's muscle torque to Newton joint_f
      2. Newton bone substeps (rigid-body dynamics)
      3. Sync bone positions to MuscleSim bone_pos_field
      4. Muscle PBD substeps (bilateral attach accumulates C*n)
      5. Extract muscle torque for next frame (EMA smoothed)
    """

    def __init__(self, model, core: MuscleSim, bone_substeps: int = 5,
                 k_coupling: float = 5000.0,
                 max_torque: float = 50.0, torque_smoothing: float = 0.3):
        self.model = model
        self.core = core
        # self.bone_solver = SolverMuJoCo(model, solver="cg")
        self.bone_solver = SolverFeatherstone(model, angular_damping=0.15,friction_smoothing=2.0,use_tile_gemm=False) 
        self.bone_substeps = bone_substeps
        self.k_coupling = k_coupling
        self.max_torque = max_torque
        self.torque_smoothing = torque_smoothing  # EMA alpha: 0=instant, 1=frozen
        self._coupling_configured = False
        self._muscle_torque = np.zeros(3, dtype=np.float32)

    # -- Configuration ------------------------------------------------------

    def configure_coupling(
        self,
        bone_body_id: int,
        bone_rest_verts: np.ndarray,
        bone_vertex_indices: np.ndarray,
        joint_index: int,
        joint_pivot: np.ndarray,
        joint_axis: np.ndarray = None,
    ):
        """Set up bidirectional bone-muscle coupling.

        Args:
            bone_body_id: Newton body index for the dynamic bone.
            bone_rest_verts: Rest-pose vertices, shape (N, 3).
            bone_vertex_indices: Indices into MuscleSim bone_pos_field.
            joint_index: Newton joint index for the driven joint.
            joint_pivot: Joint pivot position in world space, shape (3,).
            joint_axis: Joint rotation axis (for revolute projection).
        """
        self._bone_body_id = bone_body_id

        # Taichi fields for bone sync kernel
        n_bone = len(bone_vertex_indices)
        self._bone_rest_verts_field = ti.Vector.field(3, dtype=ti.f32, shape=n_bone)
        self._bone_rest_verts_field.from_numpy(bone_rest_verts.astype(np.float32))
        self._bone_idx_field = ti.field(dtype=ti.i32, shape=n_bone)
        self._bone_idx_field.from_numpy(bone_vertex_indices.astype(np.int32))
        self._n_bone_verts = n_bone
        self._bone_xform = ti.field(dtype=ti.f32, shape=7)  # px,py,pz, qx,qy,qz,qw

        # Joint DOF info
        qd_starts = self.model.joint_qd_start.numpy()
        self._joint_dof_index = int(qd_starts[joint_index])
        if joint_index + 1 < len(qd_starts):
            self._joint_n_dofs = int(qd_starts[joint_index + 1]) - self._joint_dof_index
        else:
            self._joint_n_dofs = int(self.model.joint_dof_count) - self._joint_dof_index
        self._joint_pivot = np.array(joint_pivot, dtype=np.float32)
        self._joint_axis = None
        if joint_axis is not None:
            self._joint_axis = np.array(joint_axis, dtype=np.float32)
            self._joint_axis /= np.linalg.norm(self._joint_axis)

        # Collect attach constraints targeting this bone
        radius_set = set(bone_vertex_indices.tolist())
        attach_cidx = []
        for c in self.core.attach_constraints:
            if int(c['pts'][2]) in radius_set:
                attach_cidx.append(int(c['cidx']))

        self._n_attach = len(attach_cidx)
        self._attach_cidx = ti.field(dtype=ti.i32, shape=max(self._n_attach, 1))
        if self._n_attach > 0:
            self._attach_cidx.from_numpy(np.array(attach_cidx, dtype=np.int32))

        self._torque_accum = ti.Vector.field(3, dtype=ti.f32, shape=())
        self._coupling_configured = True
        log.info(f"Coupling: {self._n_attach} constraints, "
                 f"DOF={self._joint_dof_index}, n_dofs={self._joint_n_dofs}, "
                 f"k={self.k_coupling}")

    # -- Bone sync ----------------------------------------------------------

    def _sync_bone_positions(self, state):
        """Read body_q from Newton, write transformed verts to bone_pos_field."""
        body_q = state.body_q.numpy()
        self._bone_xform.from_numpy(body_q[self._bone_body_id].astype(np.float32))
        self._sync_bone_kernel()

    @ti.kernel
    def _sync_bone_kernel(self):
        px, py, pz = self._bone_xform[0], self._bone_xform[1], self._bone_xform[2]
        qx, qy, qz, qw = (self._bone_xform[3], self._bone_xform[4],
                           self._bone_xform[5], self._bone_xform[6])
        for i in range(self._n_bone_verts):
            v = self._bone_rest_verts_field[i]
            # Quaternion rotation: v' = v + 2w(q x v) + 2(q x (q x v))
            tx = 2.0 * (qy * v[2] - qz * v[1])
            ty = 2.0 * (qz * v[0] - qx * v[2])
            tz = 2.0 * (qx * v[1] - qy * v[0])
            rx = v[0] + qw * tx + (qy * tz - qz * ty) + px
            ry = v[1] + qw * ty + (qz * tx - qx * tz) + py
            rz = v[2] + qw * tz + (qx * ty - qy * tx) + pz
            self.core.bone_pos_field[self._bone_idx_field[i]] = ti.Vector([rx, ry, rz])

    # -- Torque extraction --------------------------------------------------

    @ti.kernel
    def _compute_torque_kernel(
        self, k: ti.f32, inv_N: ti.f32,
        pivot_x: ti.f32, pivot_y: ti.f32, pivot_z: ti.f32,
    ):
        """Sum torque = sum( (target - pivot) x (-k * C_n / N) ) over attach constraints."""
        self._torque_accum[None] = ti.Vector([0.0, 0.0, 0.0])
        pivot = ti.Vector([pivot_x, pivot_y, pivot_z])
        ti.loop_config(serialize=True)
        for i in range(self._n_attach):
            cidx = self._attach_cidx[i]
            target = ti.Vector([
                self.core.cons[cidx].restvector[0],
                self.core.cons[cidx].restvector[1],
                self.core.cons[cidx].restvector[2],
            ])
            force = -k * self.core.reaction_accum[cidx] * inv_N
            self._torque_accum[None] += (target - pivot).cross(force)

    def _compute_muscle_torque(self, inv_N) -> np.ndarray:
        """Extract 3D joint torque from bilateral attach constraint reactions."""
        if self._n_attach == 0:
            return np.zeros(3, dtype=np.float32)

        self._compute_torque_kernel(
            self.k_coupling, inv_N,
            self._joint_pivot[0], self._joint_pivot[1], self._joint_pivot[2],
        )
        torque = self._torque_accum[None].to_numpy().copy()

        mag = float(np.linalg.norm(torque))
        if mag > self.max_torque:
            torque *= self.max_torque / mag
        return torque

    # -- Debug diagnostics (uncomment when needed) --------------------------
    # def _log_torque_debug(self, torque, inv_N):
    #     C_total = np.zeros(3, dtype=np.float32)
    #     for i in range(self._n_attach):
    #         cidx = int(self._attach_cidx[i])
    #         C_total += self.core.reaction_accum[cidx].to_numpy()
    #     C_mag = float(np.linalg.norm(C_total))
    #     mag = float(np.linalg.norm(torque))
    #     axis_tau = ""
    #     if self._joint_axis is not None:
    #         axis_tau = f" axis_t={float(np.dot(torque, self._joint_axis)):.4f}"
    #     log.info(f"  |C|={C_mag:.4f} F={self.k_coupling * C_mag * inv_N:.2f}N "
    #              f"|t|={mag:.4f}{axis_tau} act={self.core.cfg.activation:.2f}")

    # -- Main step ----------------------------------------------------------

    def step(self, state_in, state_out, control=None, contacts=None, dt=None):
        """One coupled simulation step.

        bone dynamics -> muscle PBD -> torque extraction.
        """
        if dt is None:
            dt = 1.0 / 60.0
        if control is None:
            control = self.model.control(clone_variables=False)

        # 1. Apply muscle torque from previous frame
        if self._coupling_configured and np.any(self._muscle_torque != 0.0):
            joint_f = control.joint_f.numpy()
            dof = self._joint_dof_index
            n = self._joint_n_dofs
            if n == 1 and self._joint_axis is not None:
                # Revolute: project 3D torque onto joint axis
                joint_f[dof] = float(np.dot(self._muscle_torque, self._joint_axis))
            else:
                # Ball (3 DOF) or other multi-DOF joints
                m = min(n, 3)
                joint_f[dof:dof + m] = self._muscle_torque[:m]
            control.joint_f = wp.array(
                joint_f, dtype=wp.float32, device=control.joint_f.device)

        # 2. Bone rigid-body dynamics
        bone_dt = dt / self.bone_substeps
        for _ in range(self.bone_substeps):
            self.bone_solver.step(state_in, state_out, control, None, bone_dt)

        # 3. Sync bone positions to muscle sim
        if self._coupling_configured:
            self._sync_bone_positions(state_out)

        # 4. Muscle PBD substeps
        dt_sub = dt / self.core.cfg.num_substeps
        self.core.dt = dt_sub
        self.core.activation.fill(self.core.cfg.activation)
        self.core.update_attach_targets()

        if self._coupling_configured:
            self.core.clear_reaction()

        for _ in range(self.core.cfg.num_substeps):
            self.core.integrate()
            self.core.clear()
            self.core.solve_constraints()
            if self.core.use_jacobi:
                self.core.apply_dP()
        self.core.update_velocities()

        # 5. Extract muscle torque for next frame (EMA smoothed)
        if self._coupling_configured:
            inv_N = 1.0 / self.core.cfg.num_substeps
            raw_torque = self._compute_muscle_torque(inv_N)
            alpha = self.torque_smoothing
            self._muscle_torque = alpha * self._muscle_torque + (1.0 - alpha) * raw_torque
