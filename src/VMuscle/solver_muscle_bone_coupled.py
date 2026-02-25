"""Bidirectional muscle-bone coupling solver.

Couples Newton MuJoCo rigid-body dynamics with Taichi MuscleSim PBD.
Spring force model: F = -k_coupling * sum(C*n) / N_substeps.
"""

import logging

import numpy as np
import taichi as ti
import warp as wp

import newton
from newton.solvers import SolverMuJoCo
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

    def __init__(self, model, core: MuscleSim,
                 bone_substeps: int = 5,
                 k_coupling: float = 5000.0,
                 max_torque: float = 50.0):
        self.model = model
        self.core = core
        self.bone_solver = SolverMuJoCo(model, solver="cg", use_mujoco_cpu=True)
        self.bone_substeps = bone_substeps
        self.k_coupling = k_coupling
        self.max_torque = max_torque
        self._coupling_configured = False
        self._muscle_torque = np.zeros(3, dtype=np.float32)  # last-substep torque, for external inspection
        self._step_count = 0

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

    # -- Main step ----------------------------------------------------------

    def step(self, state_in, state_out, control=None, contacts=None, dt=None):
        """One coupled simulation step — interleaved substeps (no one-frame delay).

        Each substep:
          1. muscle.integrate + solve_constraints  -> reaction_accum (this substep)
          2. reaction -> torque -> control.joint_f (applied immediately)
          3. bone_solver.step                      (uses current muscle torque)
          4. sync new bone position -> muscle       (for next substep's attach targets)
        """
        if dt is None:
            dt = 1.0 / 60.0
        if control is None:
            control = self.model.control(clone_variables=False)

        N = self.core.cfg.num_substeps
        dt_sub = dt / N
        self.core.dt = dt_sub
        self.core.activation.fill(self.core.cfg.activation)

        # Sync initial bone position to muscle before first substep
        if self._coupling_configured:
            self._sync_bone_positions(state_in)

        joint_f_device = control.joint_f.device
        joint_f_np = np.zeros(control.joint_f.shape[0], dtype=np.float32)

        for _ in range(N):
            # 1. Muscle: predict positions
            self.core.integrate()
            self.core.clear()

            # 2. Update attach targets from current bone position, then solve constraints
            if self._coupling_configured:
                self.core.update_attach_targets()
                self.core.clear_reaction()  # per-substep: fresh reaction

            self.core.solve_constraints()
            if self.core.use_jacobi:
                self.core.apply_dP()
            self.core.update_velocities()

            # 3. Immediately convert this substep's reaction to torque and apply to bone
            if self._coupling_configured:
                torque = self._compute_muscle_torque(inv_N=1.0)
                self._muscle_torque = torque  # expose for external inspection
                dof = self._joint_dof_index
                n_dofs = self._joint_n_dofs
                joint_f_np[:] = 0.0
                if n_dofs == 1 and self._joint_axis is not None:
                    joint_f_np[dof] = float(np.dot(torque, self._joint_axis))
                else:
                    m = min(n_dofs, 3)
                    joint_f_np[dof:dof + m] = torque[:m]
                control.joint_f = wp.array(joint_f_np, dtype=wp.float32, device=joint_f_device)

            # 4. Bone substep
            self.bone_solver.step(state_in, state_out, control, None, dt_sub)

            # 5. Sync new bone position to muscle for next substep
            if self._coupling_configured:
                self._sync_bone_positions(state_out)

        self._step_count += 1
        if self._coupling_configured and (self._step_count % 5 == 1):
            mag = float(np.linalg.norm(self._muscle_torque))
            axis_info = ""
            if self._joint_axis is not None:
                axis_info = f" axis_tau={float(np.dot(self._muscle_torque, self._joint_axis)):.4f}"
            
            # Joint angle and bone orientation
            joint_q = state_out.joint_q.numpy()
            joint_angle = float(joint_q[self._joint_dof_index]) if len(joint_q) > self._joint_dof_index else 0.0
            body_q = state_out.body_q.numpy()[self._bone_body_id]
            qx, qy, qz, qw = float(body_q[3]), float(body_q[4]), float(body_q[5]), float(body_q[6])
            
            log.info(f"step={self._step_count} act={self.core.cfg.activation:.2f} |tau|={mag:.4f}{axis_info}")
            log.info(f"  q_joint={joint_angle:.4f} q_bone=({qx:.3f},{qy:.3f},{qz:.3f},{qw:.3f})")

    # -- Reset ---------------------------------------------------------------

    def reset_bone(self, state):
        """Reset bone rigid-body state to initial pose and sync to muscle."""
        # Restore initial joint angles and zero velocities
        state.joint_q = wp.clone(self.model.joint_q)
        state.joint_qd.zero_()
        # Recompute body transforms from joint state
        newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
        # Sync bone vertex positions to muscle PBD
        if self._coupling_configured:
            self._sync_bone_positions(state)
        self._muscle_torque = np.zeros(3, dtype=np.float32)
        self._step_count = 0
        log.info("Bone reset to initial pose")

