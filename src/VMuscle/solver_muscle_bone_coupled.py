import numpy as np
import taichi as ti
import warp as wp

from newton.solvers import SolverMuJoCo
from .muscle import MuscleSim


@ti.data_oriented
class SolverMuscleBoneCoupled:
    """Couples Newton rigid-body bone dynamics with Taichi MuscleSim PBD.

    Both systems operate in the same Y-up coordinate space — no coordinate
    conversion is needed at the sync point.

    Bidirectional coupling:
      bone → muscle: attach constraints pull muscle vertices to bone positions
      muscle → bone: reaction forces from attach constraints drive joint torques
    """

    def __init__(self, model, core: MuscleSim, bone_substeps: int = 16,
                 coupling_stiffness: float = 50.0, max_torque: float = 20.0):
        self.model = model
        self.core = core
        self.bone_solver = SolverMuJoCo(model, solver="cg")
        self.bone_substeps = bone_substeps
        self.coupling_stiffness = coupling_stiffness
        self.max_torque = max_torque
        self._coupling_configured = False
        self._muscle_torque = 0.0
        self._step_count = 0

    def configure_coupling(
        self,
        bone_body_id: int,
        bone_rest_verts: np.ndarray,
        bone_vertex_indices: np.ndarray,
        joint_index: int,
        joint_pivot: np.ndarray,
        joint_axis: np.ndarray,
    ):
        """Set up bidirectional bone-muscle coupling.

        Args:
            bone_body_id: Newton body index for the dynamic bone (radius).
            bone_rest_verts: Rest-pose vertices in Y-up space, shape (N, 3).
            bone_vertex_indices: Indices into MuscleSim bone_pos_field.
            joint_index: Newton joint index for the driven joint.
            joint_pivot: Joint pivot position in world space, shape (3,).
            joint_axis: Joint rotation axis in world space, shape (3,).
        """
        self._bone_body_id = bone_body_id
        self._bone_vertex_indices = bone_vertex_indices.astype(np.int32)

        # Taichi fields for bone sync kernel
        n_bone = len(bone_vertex_indices)
        self._bone_rest_verts_field = ti.Vector.field(3, dtype=ti.f32, shape=n_bone)
        self._bone_rest_verts_field.from_numpy(bone_rest_verts.astype(np.float32))
        self._bone_idx_field = ti.field(dtype=ti.i32, shape=n_bone)
        self._bone_idx_field.from_numpy(bone_vertex_indices.astype(np.int32))
        self._n_bone_verts = n_bone
        # Rotation + translation written from Python (7 floats: px,py,pz, qx,qy,qz,qw)
        self._bone_xform = ti.field(dtype=ti.f32, shape=7)

        # Joint info for torque feedback
        self._joint_dof_index = int(self.model.joint_qd_start.numpy()[joint_index])
        self._joint_pivot = np.array(joint_pivot, dtype=np.float32)
        self._joint_axis = np.array(joint_axis, dtype=np.float32)
        self._joint_axis /= np.linalg.norm(self._joint_axis)

        # Reuse attach constraint data already collected by MuscleSim
        attach = self.core.attach_constraints  # list of dicts with 'cidx' and 'pts'
        n_attach = len(attach)
        cidx_arr = np.array([c['cidx'] for c in attach], dtype=np.int32)
        src_arr = np.array([int(c['pts'][0]) for c in attach], dtype=np.int32)
        self._attach_cidx = ti.field(dtype=ti.i32, shape=n_attach)
        self._attach_src = ti.field(dtype=ti.i32, shape=n_attach)
        self._attach_cidx.from_numpy(cidx_arr)
        self._attach_src.from_numpy(src_arr)
        self._n_attach = n_attach

        # Taichi fields for torque accumulation
        self._torque_accum = ti.Vector.field(3, dtype=ti.f32, shape=())

        self._coupling_configured = True
        print(f"Coupling: {n_attach} attach constraints, "
              f"DOF={self._joint_dof_index}, "
              f"pivot={self._joint_pivot}, axis={self._joint_axis}")

    def _sync_bone_positions(self, state):
        """Read body_q from Newton, write transformed verts to bone_pos_field via kernel."""
        body_q = state.body_q.numpy()
        xform = body_q[self._bone_body_id]  # (7,) — px,py,pz, qx,qy,qz,qw
        self._bone_xform.from_numpy(xform.astype(np.float32))
        self._sync_bone_kernel()

    @ti.kernel
    def _sync_bone_kernel(self):
        px = self._bone_xform[0]
        py = self._bone_xform[1]
        pz = self._bone_xform[2]
        qx = self._bone_xform[3]
        qy = self._bone_xform[4]
        qz = self._bone_xform[5]
        qw = self._bone_xform[6]
        for i in range(self._n_bone_verts):
            v = self._bone_rest_verts_field[i]
            # Quaternion rotation: R * v
            # t = 2 * cross(q_xyz, v)
            tx = 2.0 * (qy * v[2] - qz * v[1])
            ty = 2.0 * (qz * v[0] - qx * v[2])
            tz = 2.0 * (qx * v[1] - qy * v[0])
            # rotated = v + qw * t + cross(q_xyz, t)
            rx = v[0] + qw * tx + (qy * tz - qz * ty) + px
            ry = v[1] + qw * ty + (qz * tx - qx * tz) + py
            rz = v[2] + qw * tz + (qx * ty - qy * tx) + pz
            idx = self._bone_idx_field[i]
            self.core.bone_pos_field[idx] = ti.Vector([rx, ry, rz])

    @ti.kernel
    def _compute_torque_kernel(
        self,
        attach_cidx: ti.template(),
        attach_src: ti.template(),
        n_attach: ti.i32,
        coupling_k: ti.f32,
        pivot_x: ti.f32, pivot_y: ti.f32, pivot_z: ti.f32,
    ):
        self._torque_accum[None] = ti.Vector([0.0, 0.0, 0.0])
        ti.loop_config(serialize=True)
        for i in range(n_attach):
            cidx = attach_cidx[i]
            src_idx = attach_src[i]
            target = self.core.cons[cidx].restvector.xyz
            muscle_pos = self.core.pos[src_idx]
            # Force on bone: muscle pulls bone toward its contracted position
            force = coupling_k * (muscle_pos - target)
            # Torque = r × F, where r = attachment point - pivot
            pivot = ti.Vector([pivot_x, pivot_y, pivot_z])
            r = target - pivot
            torque = r.cross(force)
            self._torque_accum[None] += torque

    def _compute_muscle_torque(self, debug=False) -> float:
        """Compute joint torque from attach constraint reaction forces."""
        self._compute_torque_kernel(
            self._attach_cidx,
            self._attach_src,
            self._n_attach,
            self.coupling_stiffness,
            self._joint_pivot[0], self._joint_pivot[1], self._joint_pivot[2],
        )
        torque_vec = self._torque_accum[None].to_numpy()
        raw = -float(np.dot(torque_vec, self._joint_axis))
        clamped = float(np.clip(raw, -self.max_torque, self.max_torque))
        if debug:
            # Sample one attach constraint to show force direction
            if self._n_attach > 0:
                cidx = int(self._attach_cidx[0])
                src_idx = int(self._attach_src[0])
                target = np.array(self.core.cons[cidx].restvector.xyz)
                muscle_pos = np.array(self.core.pos[src_idx])
                disp = muscle_pos - target
                print(f"  [sample] target={target}, muscle={muscle_pos}, disp={disp}, |disp|={np.linalg.norm(disp):.6f}")
            print(f"  torque_vec={torque_vec}, axis={self._joint_axis}, "
                  f"raw={raw:.4f}, clamped={clamped:.4f}, act={self.core.cfg.activation:.2f}")
        return clamped

    def step(self, state_in, state_out, control=None, contacts=None, dt=None):
        if dt is None:
            dt = 1.0 / 60.0
        if control is None:
            control = self.model.control(clone_variables=False)

        # 1. Apply muscle torque from previous frame
        if self._coupling_configured and self._muscle_torque != 0.0:
            control.joint_f.numpy()[self._joint_dof_index] = self._muscle_torque

        # 2. Newton bone dynamics (gravity + muscle-derived torque)
        bone_dt = dt / self.bone_substeps
        for _ in range(self.bone_substeps):
            self.bone_solver.step(state_in, state_out, control, None, bone_dt)

        # 3. Sync bone positions to MuscleSim (same Y-up space, no conversion)
        if self._coupling_configured:
            self._sync_bone_positions(state_out)

        # 4. Muscle PBD
        self.core.activation.fill(self.core.cfg.activation)
        self.core.update_attach_targets()
        for _ in range(self.core.cfg.num_substeps):
            self.core.integrate()
            self.core.clear()
            self.core.solve_constraints()
            if self.core.use_jacobi:
                self.core.apply_dP()
        self.core.update_velocities()

        # 5. Extract muscle torque for next frame
        if self._coupling_configured:
            self._step_count += 1
            debug = (self._step_count % 60 == 1)  # print every ~1 second
            self._muscle_torque = self._compute_muscle_torque(debug=debug)

    def _timer(self, name: str):
        return wp.ScopedTimer(
            name,
        )