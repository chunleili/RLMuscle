"""Warp-only bidirectional-ish muscle-bone coupling solver.

This keeps the original `solver_muscle_bone_coupled.py` untouched and provides
an alternative solver that works with `VMuscle.muscle_warp.MuscleSim`.

Notes:
- Bone -> muscle coupling is preserved by syncing rigid bone vertices into
  `core.bone_pos_field` before each muscle step.
- Muscle -> bone torque is approximated from attach-constraint displacement
  vectors (no Taichi reaction accumulator is available in the warp path).
"""

from __future__ import annotations

import logging

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherstone, SolverMuJoCo

log = logging.getLogger("couple")


class SolverMuscleBoneCoupledWarp:
    """Couple Newton rigid body with Warp MuscleSim.

    The Warp `MuscleSim` implementation does not expose per-constraint reaction
    accumulators (unlike the Taichi version), so reverse torque is estimated by
    spring-like displacement on attach pairs.
    """

    def __init__(
        self,
        model,
        core,
        bone_substeps: int = 5,
        k_coupling: float = 5000.0,
        max_torque: float = 50.0,
        torque_ema: float = 0.2,
    ):
        self.model = model
        self.core = core
        # Prefer MuJoCo for closer behavior to the original solver.
        # Fallback to Featherstone when MuJoCo backend isn't installed.
        try:
            self.bone_solver = SolverMuJoCo(model, solver="cg", use_mujoco_cpu=True)
            self._bone_solver_name = "mujoco"
        except Exception as exc:  # noqa: BLE001 - backend availability is runtime-dependent
            self.bone_solver = SolverFeatherstone(model)
            self._bone_solver_name = "featherstone"
            log.warning("MuJoCo backend unavailable, fallback to Featherstone: %s", exc)

        self.bone_substeps = int(max(1, bone_substeps))
        self.k_coupling = float(k_coupling)
        self.max_torque = float(max_torque)
        self.torque_ema = float(np.clip(torque_ema, 0.0, 1.0))

        self._coupling_configured = False
        self._muscle_torque = np.zeros(3, dtype=np.float32)
        self._step_count = 0

    def configure_coupling(
        self,
        bone_body_id: int,
        bone_rest_verts: np.ndarray,
        bone_vertex_indices: np.ndarray,
        joint_index: int,
        joint_pivot: np.ndarray,
        joint_axis: np.ndarray | None = None,
    ):
        self._bone_body_id = int(bone_body_id)
        self._bone_rest_verts = np.asarray(bone_rest_verts, dtype=np.float32)
        self._bone_vertex_indices = np.asarray(bone_vertex_indices, dtype=np.int32)

        qd_starts = self.model.joint_qd_start.numpy()
        self._joint_dof_index = int(qd_starts[joint_index])
        if joint_index + 1 < len(qd_starts):
            self._joint_n_dofs = int(qd_starts[joint_index + 1]) - self._joint_dof_index
        else:
            self._joint_n_dofs = int(self.model.joint_dof_count) - self._joint_dof_index

        self._joint_pivot = np.asarray(joint_pivot, dtype=np.float32)
        self._joint_axis = None
        if joint_axis is not None:
            axis = np.asarray(joint_axis, dtype=np.float32)
            norm = np.linalg.norm(axis)
            if norm > 1e-8:
                self._joint_axis = axis / norm

        selected = set(self._bone_vertex_indices.tolist())
        self._attach_pairs: list[tuple[int, int]] = []
        for c in getattr(self.core, "attach_constraints", []):
            src = int(c["pts"][0])
            tgt = int(c["pts"][2])
            if tgt in selected:
                self._attach_pairs.append((src, tgt))

        self._coupling_configured = True
        log.info(
            "Warp coupling: attach_pairs=%d dof=%d n_dofs=%d k=%.1f bone_solver=%s",
            len(self._attach_pairs),
            self._joint_dof_index,
            self._joint_n_dofs,
            self.k_coupling,
            self._bone_solver_name,
        )

    @staticmethod
    def _quat_rotate(points: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
        qx, qy, qz, qw = quat_xyzw
        q = np.array([qx, qy, qz], dtype=np.float32)
        q = np.broadcast_to(q, points.shape)
        t = 2.0 * np.cross(q, points)
        return points + qw * t + np.cross(q, t)

    def _sync_bone_positions(self, state):
        body_q = state.body_q.numpy()[self._bone_body_id]
        p = np.asarray(body_q[:3], dtype=np.float32)
        q = np.asarray(body_q[3:], dtype=np.float32)

        deformed = self._quat_rotate(self._bone_rest_verts, q) + p

        bone_np = self.core.bone_pos_field.numpy()
        bone_np[self._bone_vertex_indices] = deformed
        self.core.bone_pos_field = wp.array(
            bone_np.astype(np.float32),
            dtype=wp.vec3,
            device=self.core.bone_pos_field.device,
        )

    def _compute_muscle_torque(self) -> np.ndarray:
        if not self._attach_pairs:
            return np.zeros(3, dtype=np.float32)

        pos_np = self.core.pos.numpy()
        bone_np = self.core.bone_pos_field.numpy()
        torque = np.zeros(3, dtype=np.float32)

        inv_n = 1.0 / max(1, len(self._attach_pairs))
        for src, tgt in self._attach_pairs:
            # Approximate reaction on bone from attach displacement.
            disp = pos_np[src] - bone_np[tgt]
            force_on_bone = self.k_coupling * disp
            arm = bone_np[tgt] - self._joint_pivot
            torque += np.cross(arm, force_on_bone).astype(np.float32)

        torque *= inv_n
        mag = float(np.linalg.norm(torque))
        if mag > self.max_torque and mag > 1e-12:
            torque *= self.max_torque / mag

        if self.torque_ema > 0.0:
            self._muscle_torque = (
                (1.0 - self.torque_ema) * self._muscle_torque + self.torque_ema * torque
            ).astype(np.float32)
        else:
            self._muscle_torque = torque.astype(np.float32)
        return self._muscle_torque

    def step(self, state_in, state_out, control=None, contacts=None, dt=None):
        del contacts

        if dt is None:
            dt = 1.0 / 60.0
        if control is None:
            control = self.model.control(clone_variables=False)

        if self._coupling_configured:
            self._sync_bone_positions(state_in)

        self.core.activation.fill_(self.core.cfg.activation)
        self.core.step()

        joint_f_np = np.zeros(control.joint_f.shape[0], dtype=np.float32)
        if self._coupling_configured:
            torque = self._compute_muscle_torque()
            dof = self._joint_dof_index
            n_dofs = self._joint_n_dofs
            if n_dofs == 1 and self._joint_axis is not None:
                joint_f_np[dof] = float(np.dot(torque, self._joint_axis))
            else:
                m = min(n_dofs, 3)
                joint_f_np[dof : dof + m] = torque[:m]

        joint_f_device = control.joint_f.device
        control.joint_f = wp.array(joint_f_np, dtype=wp.float32, device=joint_f_device)

        dt_sub = float(dt) / float(self.bone_substeps)
        for _ in range(self.bone_substeps):
            self.bone_solver.step(state_in, state_out, control, None, dt_sub)

        if self._coupling_configured:
            self._sync_bone_positions(state_out)

        self._step_count += 1
        if self._coupling_configured and (self._step_count % 10 == 1):
            mag = float(np.linalg.norm(self._muscle_torque))
            axis_info = ""
            if self._joint_axis is not None:
                axis_info = f" axis_tau={float(np.dot(self._muscle_torque, self._joint_axis)):.4f}"
            log.info(
                "step=%d act=%.2f |tau|=%.4f%s",
                self._step_count,
                float(self.core.cfg.activation),
                mag,
                axis_info,
            )

    def reset_bone(self, state):
        state.joint_q = wp.clone(self.model.joint_q)
        state.joint_qd.zero_()
        newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
        if self._coupling_configured:
            self._sync_bone_positions(state)
        self._muscle_torque = np.zeros(3, dtype=np.float32)
        self._step_count = 0
        log.info("Bone reset to initial pose")
