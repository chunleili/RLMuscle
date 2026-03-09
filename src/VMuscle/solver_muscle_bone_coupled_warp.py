"""Bidirectional muscle-bone coupling solver.

Couples Newton MuJoCo rigid-body dynamics with MuscleSim PBD.
Spring force model: F = -k_coupling * sum(C*n) / N_substeps.
All Taichi fields are accessed via numpy bridging.
"""

import logging

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo
from .muscle import MuscleSim

log = logging.getLogger("couple")


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

        # Numpy arrays for bone sync
        n_bone = len(bone_vertex_indices)
        self._bone_rest_verts = bone_rest_verts.astype(np.float32)  # (N, 3)
        self._bone_idx = bone_vertex_indices.astype(np.int32)       # (N,)
        self._n_bone_verts = n_bone

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
        self._attach_cidx = np.array(attach_cidx, dtype=np.int32) if attach_cidx else np.empty(0, dtype=np.int32)
        self._coupling_configured = True
        log.info(f"Coupling: {self._n_attach} constraints, "
                 f"DOF={self._joint_dof_index}, n_dofs={self._joint_n_dofs}, "
                 f"k={self.k_coupling}")

    # -- Bone sync ----------------------------------------------------------

    def _sync_bone_positions(self, state):
        """Read body_q from Newton, transform rest verts, write to bone_pos_field via numpy."""
        body_q = state.body_q.numpy()
        xf = body_q[self._bone_body_id].astype(np.float32)
        pos = xf[:3]                       # translation
        q_xyz, qw = xf[3:6], xf[6]        # quaternion (x,y,z,w)

        # Vectorised quaternion rotation: v' = v + 2w(q x v) + 2(q x (q x v))
        v = self._bone_rest_verts           # (N, 3)
        t = 2.0 * np.cross(q_xyz, v)       # (N, 3)
        rotated = v + qw * t + np.cross(q_xyz, t) + pos  # (N, 3)

        # Write back to taichi bone_pos_field through numpy
        bone_np = self.core.bone_pos_field.to_numpy()
        bone_np[self._bone_idx] = rotated
        self.core.bone_pos_field.from_numpy(bone_np)

    # -- Torque extraction --------------------------------------------------

    def _compute_muscle_torque(self, inv_N) -> np.ndarray:
        """Extract 3D joint torque from bilateral attach constraint reactions via numpy."""
        if self._n_attach == 0:
            return np.zeros(3, dtype=np.float32)

        # Read taichi fields through numpy
        restvec_np = self.core.cons.restvector.to_numpy()   # (n_cons, 4)
        reaction_np = self.core.reaction_accum.to_numpy()   # (n_cons, 3)

        cidxs = self._attach_cidx
        targets = restvec_np[cidxs, :3]                     # (n_attach, 3)
        reactions = reaction_np[cidxs]                       # (n_attach, 3)

        forces = -self.k_coupling * reactions * inv_N        # (n_attach, 3)
        arms = targets - self._joint_pivot                   # (n_attach, 3)
        torque = np.cross(arms, forces).sum(axis=0).astype(np.float32)

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

