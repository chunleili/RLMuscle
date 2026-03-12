import time

import numpy as np
import taichi as ti

from VMuscle.config import SimConfig, load_config  # noqa: F401
from VMuscle.constraints import (
    PIN, ATTACH, TETVOLUME, TETFIBERNORM, DISTANCELINE,
    TETARAP, TETARAPNORM,
    LINEARENERGY, NORMSTIFFNESS,
)
from VMuscle.mesh_io import build_surface_tris
from VMuscle.muscle_common import MuscleSimBase, get_bbox  # noqa: F401
from VMuscle.vis_taichi import Visualizer


def pick_arch(name: str):
    name = name.lower()
    if name == "vulkan":
        return ti.vulkan
    if name == "cpu":
        return ti.cpu
    if name == "cuda":
        return ti.cuda
    return ti.cpu


# ---------------------------------------------------------------------------
# Taichi math helpers (@ti.func)
# ---------------------------------------------------------------------------

@ti.func
def updatedP(dP, dPw, dp: ti.types.vector(3, ti.f32), pt: ti.i32):
    for j in ti.static(range(3)):
        ti.atomic_add(dP[pt][j], dp[j])
    ti.atomic_add(dPw[pt], 1.0)

@ti.func
def fem_flags(ctype: ti.i32) -> ti.i32:
    flags = 0
    if ctype == TETARAP or ctype == TETARAPNORM:
        flags |= LINEARENERGY
    if ctype == TETARAPNORM or ctype == TETFIBERNORM:
        flags |= NORMSTIFFNESS
    return flags

@ti.func
def project_to_line(p: ti.types.vector(3, ti.f32),
                    orig: ti.types.vector(3, ti.f32),
                    direction: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32):
    return orig + direction * (p - orig).dot(direction)

@ti.func
def ssvd(F):
    U, sig, V = ti.svd(F)
    if U.determinant() < 0:
        for i in ti.static(range(3)):
            U[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    if V.determinant() < 0:
        for i in ti.static(range(3)):
            V[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    return U, sig, V

@ti.func
def polar_decomposition(F):
    U, sig, V = ssvd(F)
    R = U @ V.transpose()
    S = V @ sig @ V.transpose()
    return S, R

@ti.func
def squared_norm3(a: ti.types.matrix(3, 3, ti.f32)) -> ti.f32:
    return a.norm_sqr()

@ti.func
def mat3_to_quat(m: ti.types.matrix(3, 3, ti.f32)) -> ti.types.vector(4, ti.f32):
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    qw, qx, qy, qz = 0.0, 0.0, 0.0, 0.0
    if trace > 0.0:
        s = ti.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = ti.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = ti.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = ti.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    return ti.Vector([qx, qy, qz, qw])

@ti.func
def get_inv_mass(idx: ti.i32, mass: ti.template(), stopped: ti.template()) -> ti.f32:
    res = 0.0
    if (stopped[idx]):
        res = 0.0
    else:
        m = mass[idx]
        res = 1.0 / m if m > 0.0 else 0.0
    return res


# ---------------------------------------------------------------------------
# MuscleSim (Taichi backend)
# ---------------------------------------------------------------------------

@ti.data_oriented
class MuscleSim(MuscleSimBase):

    def _init_backend(self):
        ti.init(arch=pick_arch(self.cfg.arch))

    def _build_surface_tris(self):
        self.surface_tris = build_surface_tris(self.tet_np)
        self.n_tris = self.surface_tris.shape[0]
        self.surface_tris_field = ti.field(dtype=ti.i32, shape=(self.n_tris * 3,))
        self.surface_tris_field.from_numpy(self.surface_tris.reshape(-1))

    def _allocate_fields(self):
        n_v = self.n_verts
        n_tet = self.tet_np.shape[0]

        self.pos = ti.Vector.field(3, dtype=ti.f32, shape=n_v)
        self.pprev = ti.Vector.field(3, dtype=ti.f32, shape=n_v)
        self.pos0 = ti.Vector.field(3, dtype=ti.f32, shape=n_v)
        self.vel = ti.Vector.field(3, dtype=ti.f32, shape=n_v)
        self.force = ti.Vector.field(3, dtype=ti.f32, shape=n_v)
        self.mass = ti.field(dtype=ti.f32, shape=n_v)
        self.stopped = ti.field(dtype=ti.i32, shape=n_v)
        self.v_fiber_dir = ti.Vector.field(3, dtype=ti.f32, shape=n_v)
        self.dP = ti.Vector.field(3, dtype=ti.f32, shape=n_v)
        self.dPw = ti.field(dtype=ti.f32, shape=n_v)
        self.tet_indices = ti.Vector.field(4, dtype=ti.i32, shape=n_tet)

        self.rest_volume = ti.field(dtype=ti.f32, shape=n_tet)
        self.rest_matrix = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_tet)
        self.activation = ti.field(dtype=ti.f32, shape=n_tet)

    def _create_bone_fields(self):
        if self.bone_pos.shape[0] > 0:
            self.bone_pos_field = ti.Vector.field(3, dtype=ti.f32, shape=self.bone_pos.shape[0])
            self.bone_pos_field.from_numpy(self.bone_pos)
            if self.bone_indices_np.shape[0] > 0:
                self.bone_indices_field = ti.field(dtype=ti.i32, shape=self.bone_indices_np.shape[0])
                self.bone_indices_field.from_numpy(self.bone_indices_np)

    def _init_fields(self):
        self.pos0.from_numpy(self.pos0_np)
        self.pos.from_numpy(self.pos0_np)
        self.vel.fill(0)
        self.force.fill(0)
        self.mass.fill(0)
        self.stopped.fill(0)
        self.tet_indices.from_numpy(self.tet_np)

        if self.v_fiber_np is None:
            self.v_fiber_np = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (self.n_verts, 1))
        self.v_fiber_dir.from_numpy(self.v_fiber_np)

        self.tendonmask = ti.field(dtype=ti.f32, shape=(self.tet_np.shape[0],))
        self._compute_cell_tendon_mask(self.tet_indices, self.v_tendonmask_np, self.tendonmask)

        self.activation.fill(0.0)
        self.total_rest_volume = ti.field(dtype=ti.f32, shape=())

        print("Initialized fields done.")

    def build_constraints(self):
        """Pack raw constraints into taichi struct field."""
        print("Building constraints...")
        constraint_struct = ti.types.struct(
            type=ti.i32,
            cidx=ti.i32,
            pts=ti.types.vector(4, ti.i32),
            stiffness=ti.f32,
            dampingratio=ti.f32,
            tetid=ti.i32,
            L=ti.types.vector(3, ti.f32),
            restlength=ti.f32,
            restvector=ti.types.vector(4, ti.f32),
            restdir=ti.types.vector(3, ti.f32),
            compressionstiffness=ti.f32,
        )

        all_constraints, _dt_collect = self._collect_raw_constraints()

        # Sort by type for per-type kernel dispatch
        all_constraints.sort(key=lambda c: c['type'])

        n_cons = len(all_constraints)
        self.cons_ranges = {}
        if n_cons > 0:
            prev_type = None
            start_idx = 0
            for i, c in enumerate(all_constraints):
                c['cidx'] = i
                ctype = c['type']
                if ctype != prev_type:
                    if prev_type is not None:
                        self.cons_ranges[prev_type] = (start_idx, i - start_idx)
                    start_idx = i
                    prev_type = ctype
            if prev_type is not None:
                self.cons_ranges[prev_type] = (start_idx, n_cons - start_idx)

            type_arr = np.array([c['type'] for c in all_constraints], dtype=np.int32)
            cidx_arr = np.arange(n_cons, dtype=np.int32)
            pts_arr = np.array([[c['pts'][j] for j in range(4)] for c in all_constraints], dtype=np.int32)
            stiffness_arr = np.array([c['stiffness'] for c in all_constraints], dtype=np.float32)
            dampingratio_arr = np.array([c['dampingratio'] for c in all_constraints], dtype=np.float32)
            tetid_arr = np.array([c['tetid'] for c in all_constraints], dtype=np.int32)
            L_arr = np.array([[c['L'][j] for j in range(3)] for c in all_constraints], dtype=np.float32)
            restlength_arr = np.array([c['restlength'] for c in all_constraints], dtype=np.float32)
            restvector_arr = np.array([[c['restvector'][j] for j in range(4)] for c in all_constraints], dtype=np.float32)
            restdir_arr = np.array([[c['restdir'][j] for j in range(3)] for c in all_constraints], dtype=np.float32)
            compressionstiffness_arr = np.array([c['compressionstiffness'] for c in all_constraints], dtype=np.float32)

            self.cons = constraint_struct.field(shape=n_cons)
            self.cons.type.from_numpy(type_arr)
            self.cons.cidx.from_numpy(cidx_arr)
            self.cons.pts.from_numpy(pts_arr)
            self.cons.stiffness.from_numpy(stiffness_arr)
            self.cons.dampingratio.from_numpy(dampingratio_arr)
            self.cons.tetid.from_numpy(tetid_arr)
            self.cons.L.from_numpy(L_arr)
            self.cons.restlength.from_numpy(restlength_arr)
            self.cons.restvector.from_numpy(restvector_arr)
            self.cons.restdir.from_numpy(restdir_arr)
            self.cons.compressionstiffness.from_numpy(compressionstiffness_arr)
        else:
            self.cons = constraint_struct.field(shape=0)
            self.raw_constraints = []

        print(f"Built {n_cons} constraints total. [{_dt_collect*1000:.0f}ms]")

        # Reaction accumulator for bilateral attach coupling
        self.reaction_accum = ti.Vector.field(3, dtype=ti.f32, shape=max(n_cons, 1))

        # Build colored constraint groups for parallel Gauss-Seidel
        if self.use_colored_gs and n_cons > 0:
            self._build_colored_gs(all_constraints, constraint_struct)

    def _build_colored_gs(self, all_constraints, constraint_struct):
        """Reorder constraints by (color, type) and build per-color dispatch ranges."""
        from .constraints import build_constraint_color_groups
        import time

        t0 = time.perf_counter()
        color_groups = build_constraint_color_groups(all_constraints)
        n_colors = len(color_groups)
        print(f"  Graph coloring: {n_colors} colors, "
              f"sizes: {[len(g) for g in color_groups]} "
              f"[{(time.perf_counter()-t0)*1000:.0f}ms]")

        # Reorder constraints: (color, type) sorted, build new array
        reordered = []
        # color_type_ranges[color_idx] = {type: (offset, count)}
        self.color_type_ranges = []
        global_offset = 0
        for color_idx, group in enumerate(color_groups):
            # get constraints in this color, sort by type
            color_cons = [all_constraints[i] for i in group]
            color_cons.sort(key=lambda c: c['type'])

            type_ranges = {}
            prev_type = None
            start = 0
            for i, c in enumerate(color_cons):
                ctype = c['type']
                if ctype != prev_type:
                    if prev_type is not None:
                        type_ranges[prev_type] = (global_offset + start, i - start)
                    start = i
                    prev_type = ctype
            if prev_type is not None:
                type_ranges[prev_type] = (global_offset + start, len(color_cons) - start)

            self.color_type_ranges.append(type_ranges)
            reordered.extend(color_cons)
            global_offset += len(color_cons)

        # Rebuild constraint arrays with new order
        n_cons = len(reordered)
        for i, c in enumerate(reordered):
            c['cidx'] = i

        type_arr = np.array([c['type'] for c in reordered], dtype=np.int32)
        pts_arr = np.array([c['pts'] for c in reordered], dtype=np.int32)
        stiffness_arr = np.array([c['stiffness'] for c in reordered], dtype=np.float32)
        dampingratio_arr = np.array([c['dampingratio'] for c in reordered], dtype=np.float32)
        tetid_arr = np.array([c['tetid'] for c in reordered], dtype=np.int32)
        L_arr = np.array([c['L'] for c in reordered], dtype=np.float32)
        restlength_arr = np.array([c['restlength'] for c in reordered], dtype=np.float32)
        restvector_arr = np.array([c['restvector'] for c in reordered], dtype=np.float32)
        restdir_arr = np.array([[c['restdir'][j] for j in range(3)] for c in reordered], dtype=np.float32)
        compressionstiffness_arr = np.array([c['compressionstiffness'] for c in reordered], dtype=np.float32)

        self.cons = constraint_struct.field(shape=n_cons)
        self.cons.type.from_numpy(type_arr)
        self.cons.cidx.from_numpy(np.arange(n_cons, dtype=np.int32))
        self.cons.pts.from_numpy(pts_arr)
        self.cons.stiffness.from_numpy(stiffness_arr)
        self.cons.dampingratio.from_numpy(dampingratio_arr)
        self.cons.tetid.from_numpy(tetid_arr)
        self.cons.L.from_numpy(L_arr)
        self.cons.restlength.from_numpy(restlength_arr)
        self.cons.restvector.from_numpy(restvector_arr)
        self.cons.restdir.from_numpy(restdir_arr)
        self.cons.compressionstiffness.from_numpy(compressionstiffness_arr)

    def _init_renderer(self):
        print("Initializing visualization...")
        print("Renderer mode:", self.cfg.render_mode)
        self.vis = Visualizer(self.cfg, self)
        self.vis._init_attach_vis(self.attach_constraints)

    def reset(self):
        self.pos.from_numpy(self.pos0_np)
        self.pprev.from_numpy(self.pos0_np)
        self.vel.fill(0)
        self.force.fill(0)
        self.activation.fill(0.0)
        self.clear()
        self.step_cnt = 1

    @ti.kernel
    def _update_attach_targets_kernel(self):
        for c in range(self.cons.shape[0]):
            ctype = self.cons[c].type
            if ctype == ATTACH:
                tgt_idx = self.cons[c].pts[2]
                if tgt_idx >= 0:
                    target_pos = self.bone_pos_field[tgt_idx]
                    self.cons[c].restvector.xyz = target_pos
            elif ctype == DISTANCELINE:
                tgt_idx = self.cons[c].pts[1]
                if tgt_idx >= 0:
                    target_pos = self.bone_pos_field[tgt_idx]
                    self.cons[c].restvector.xyz = target_pos

    def update_attach_targets(self):
        if (
            hasattr(self, "bone_pos_field")
            and self.bone_pos_field.shape[0] > 0
            and hasattr(self, "attach_constraints")
            and len(self.attach_constraints) > 0
        ):
            self._update_attach_targets_kernel()

    @ti.kernel
    def _compute_cell_tendon_mask(self, tet_indices: ti.template(), v_tendonmask_np: ti.types.ndarray(), tendon_mask: ti.template()):
        for c in tet_indices:
            sum_mask = 0.0
            for i in ti.static(range(4)):
                v_idx = tet_indices[c][i]
                sum_mask += v_tendonmask_np[v_idx]
            tendon_mask[c] = sum_mask / 4.0

    @ti.func
    def Ds_rest(self, pts):
        return ti.Matrix.cols([self.pos0[pts[i]] - self.pos0[pts[3]] for i in range(3)])

    @ti.func
    def Ds(self, pts):
        return ti.Matrix.cols([self.pos[pts[i]] - self.pos[pts[3]] for i in range(3)])

    @ti.kernel
    def _precompute_rest(self):
        n_tet = self.rest_volume.shape[0]
        self.total_rest_volume[None] = 0.0
        for c in range(n_tet):
            pts = self.tet_indices[c]
            Dm = self.Ds_rest(pts)
            self.rest_volume[c] = ti.abs(Dm.determinant()) / 6.0
            self.total_rest_volume[None] += self.rest_volume[c]
            for i in range(4):
                ti.atomic_add(self.mass[pts[i]], self.rest_volume[c] * self.cfg.density / 4.0)
            self.rest_matrix[c] = Dm.inverse()

    @ti.func
    def get_compression_stiffness(self, c: int) -> float:
        kstiff = self.cons[c].stiffness
        kstiffcompress = self.cons[c].compressionstiffness
        kstiffcompress = ti.select(kstiffcompress >= 0.0, kstiffcompress, kstiff)
        return kstiffcompress

    @ti.func
    def transfer_tension(self, muscletension, tendonmask, minfiberscale=0.0, maxfiberscale=10.0):
        fiberscale = minfiberscale + (1.0 - tendonmask) * muscletension * (maxfiberscale - minfiberscale)
        return fiberscale

    # -- per-type constraint solver kernels -----------------------------------

    @ti.kernel
    def solve_tetvolume(self, offset: int, count: int):
        for idx in range(count):
            c = offset + idx
            kstiffcompress = self.get_compression_stiffness(c)
            if self.cons[c].stiffness <= 0.0:
                continue
            pts = self.cons[c].pts
            tetid = self.cons[c].tetid
            self.tet_volume_update_xpbd(
                self.use_jacobi, c, self.cons, self.pos, self.pprev,
                self.cons[c].restlength, self.dP, self.dPw,
                tetid, pts, self.dt, self.cons[c].stiffness,
                kstiffcompress, self.cons[c].dampingratio,
                self.mass, self.stopped, 0
            )

    @ti.kernel
    def solve_tetfibernorm(self, offset: int, count: int):
        for idx in range(count):
            c = offset + idx
            if self.cons[c].stiffness <= 0.0:
                continue
            pts = self.cons[c].pts
            fiber_dir = (self.v_fiber_dir[pts[0]] + self.v_fiber_dir[pts[1]] + self.v_fiber_dir[pts[2]] + self.v_fiber_dir[pts[3]]) / 4.0
            tetid = self.cons[c].tetid
            Dminv = self.rest_matrix[tetid]
            acti = self.activation[tetid]
            _tendonmask = self.tendonmask[tetid]
            belly_factor = 1.0 - _tendonmask

            fiberscale = self.transfer_tension(acti, _tendonmask)
            stiffness = self.cons[c].stiffness * fiberscale * self.fiber_stiffness_scale
            if stiffness <= 0.0:
                continue

            target_stretch = 1.0 - belly_factor * acti * self.contraction_ratio

            self.tet_fiber_update_xpbd(
                self.use_jacobi, self.pos, self.pprev,
                self.dP, self.dPw, c, self.cons, pts,
                self.dt, fiber_dir, stiffness, Dminv,
                self.cons[c].dampingratio, self.cons[c].restlength,
                self.cons[c].restvector, acti, self.mass,
                self.stopped, target_stretch,
            )

    @ti.kernel
    def solve_attach(self, offset: int, count: int):
        for idx in range(count):
            c = offset + idx
            kstiffcompress = self.get_compression_stiffness(c)
            if self.cons[c].stiffness <= 0.0:
                continue
            pts = self.cons[c].pts
            pt_src = pts[0]
            self.attach_bilateral_update(
                self.use_jacobi, c, self.cons, pt_src,
                self.cons[c].restvector.xyz, self.pos[pt_src],
                self.pos, self.pprev, self.dP, self.dPw,
                self.mass, self.stopped, self.cons[c].restlength,
                self.cons[c].stiffness, self.cons[c].dampingratio,
                kstiffcompress,
            )

    @ti.kernel
    def solve_pin(self, offset: int, count: int):
        for idx in range(count):
            c = offset + idx
            if self.cons[c].stiffness <= 0.0:
                continue
            kstiffcompress = self.get_compression_stiffness(c)
            pts = self.cons[c].pts
            pt_src = pts[0]
            self.distance_pos_update_xpbd(
                self.use_jacobi, c, self.cons, -1, pt_src,
                self.cons[c].restvector.xyz, self.pos[pt_src],
                self.pos, self.pprev, self.dP, self.dPw,
                self.mass, self.stopped, self.cons[c].restlength,
                self.cons[c].stiffness, self.cons[c].dampingratio,
                kstiffcompress,
            )

    @ti.kernel
    def solve_distanceline(self, offset: int, count: int):
        for idx in range(count):
            c = offset + idx
            kstiffcompress = self.get_compression_stiffness(c)
            if self.cons[c].stiffness <= 0.0:
                continue
            pts = self.cons[c].pts
            pt_src = pts[0]
            p_src = self.pos[pt_src]
            line_origin = self.cons[c].restvector.xyz
            line_dir = self.cons[c].restdir

            p_projected = project_to_line(p_src, line_origin, line_dir)

            self.distance_pos_update_xpbd(
                self.use_jacobi, c, self.cons,
                -1, pt_src,
                p_projected, p_src,
                self.pos, self.pprev,
                self.dP, self.dPw,
                self.mass, self.stopped,
                self.cons[c].restlength, self.cons[c].stiffness,
                self.cons[c].dampingratio, kstiffcompress,
            )

    @ti.kernel
    def solve_tetarap(self, offset: int, count: int):
        for idx in range(count):
            c = offset + idx
            if self.cons[c].stiffness <= 0.0:
                continue
            pts = self.cons[c].pts
            tetid = self.cons[c].tetid
            self.tet_arap_update_xpbd(
                self.use_jacobi, c, self.cons, pts,
                self.dt, self.pos, self.pprev,
                self.dP, self.dPw, self.mass, self.stopped,
                self.cons[c].restlength, self.cons[c].restvector,
                self.rest_matrix[tetid], self.cons[c].stiffness,
                self.cons[c].dampingratio, fem_flags(self.cons[c].type),
            )

    def _dispatch_constraints(self, type_ranges):
        """Dispatch constraint solver kernels for a given {type: (offset, count)} mapping."""
        for ctype, (offset, count) in type_ranges.items():
            if ctype == TETVOLUME:
                self.solve_tetvolume(offset, count)
            elif ctype == TETFIBERNORM:
                self.solve_tetfibernorm(offset, count)
            elif ctype == ATTACH:
                self.solve_attach(offset, count)
            elif ctype == PIN:
                self.solve_pin(offset, count)
            elif ctype == DISTANCELINE:
                self.solve_distanceline(offset, count)
            elif ctype == TETARAP:
                self.solve_tetarap(offset, count)

    def solve_constraints(self):
        if self.use_colored_gs and hasattr(self, 'color_type_ranges'):
            for type_ranges in self.color_type_ranges:
                self._dispatch_constraints(type_ranges)
        else:
            self._dispatch_constraints(self.cons_ranges)

    @ti.kernel
    def integrate(self):
        for i in range(self.pos.shape[0]):
            extacc = ti.Vector([0.0, self.cfg.gravity, 0.0])
            self.pprev[i] = self.pos[i]
            self.vel[i] = (1.0 - self.cfg.veldamping) * self.vel[i]
            self.vel[i] += self.dt * extacc
            self.pos[i] += self.dt * self.vel[i]

    @ti.kernel
    def update_velocities(self):
        for i in range(self.pos.shape[0]):
            self.vel[i] = (self.pos[i] - self.pprev[i]) / self.dt

    @ti.kernel
    def calc_vol_error(self) -> ti.f32:
        total_vol = 0.0
        for c in range(self.rest_volume.shape[0]):
            pts = self.tet_indices[c]
            p0, p1, p2, p3 = self.pos[pts[0]], self.pos[pts[1]], self.pos[pts[2]], self.pos[pts[3]]
            d1 = p1 - p0
            d2 = p2 - p0
            d3 = p3 - p0
            volume = ((d2).cross(d1)).dot(d3) / 6.0
            total_vol += volume

        vol_err = (total_vol - self.total_rest_volume[None]) / self.total_rest_volume[None]
        return vol_err

    @ti.kernel
    def clear(self):
        for i in self.dP:
            self.dP[i] = ti.Vector([0.0, 0.0, 0.0])
            self.dPw[i] = 0.0
        for i in self.cons:
            self.cons[i].L = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def clear_reaction(self):
        """Clear reaction accumulator (call once per frame, NOT per substep)."""
        for i in range(self.cons.shape[0]):
            self.reaction_accum[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def apply_dP(self):
        for idx in self.pos:
            w = self.dPw[idx]
            if w > 1e-9:
                self.pos[idx] += self.dP[idx] / w

    @ti.func
    def inCompressBand(self, curlen: ti.f32, restlen: ti.f32) -> ti.i32:
        res = 0
        if self.has_compressstiffness:
            if curlen < restlen:
                res = 1
        return res


    # Reference: pbd_constraints.cl:L956 tetVolumeUpdateXPBD
    @ti.func
    def tet_volume_update_xpbd(
        self,
        use_jacobi: ti.template(),
        cidx: ti.i32,
        cons: ti.template(),
        pos: ti.template(),
        pprev: ti.template(),
        restlength: ti.template(),
        dP: ti.template(),
        dPw: ti.template(),
        tetid: ti.i32,
        pts: ti.types.vector(4, ti.i32),
        dt: ti.f32,
        stiffness: ti.f32,
        kstiffcompress: ti.f32,
        kdampratio: ti.f32,
        mass: ti.template(),
        stopped: ti.template(),
        loff: ti.i32
    ):
        loff = 0
        inv_masses = ti.Vector([0.0, 0.0, 0.0, 0.0])
        for i in ti.static(range(4)):
            inv_masses[i] = get_inv_mass(pts[i], mass, stopped)

        p0, p1, p2, p3 = pos[pts[0]], pos[pts[1]], pos[pts[2]], pos[pts[3]]

        d1 = p1 - p0
        d2 = p2 - p0
        d3 = p3 - p0
        grad1 = (d3.cross(d2)) / 6.0
        grad2 = (d1.cross(d3)) / 6.0
        grad3 = (d2.cross(d1)) / 6.0
        grad0 = -(grad1 + grad2 + grad3)

        w_sum = (inv_masses[0] * grad0.norm_sqr() +
                inv_masses[1] * grad1.norm_sqr() +
                inv_masses[2] * grad2.norm_sqr() +
                inv_masses[3] * grad3.norm_sqr())

        if w_sum > 1e-9:
            volume = ((d2).cross(d1)).dot(d3) / 6.0

            comp = self.inCompressBand(volume, restlength)
            kstiff = ti.select(comp, kstiffcompress, stiffness)
            loff += comp
            loff = ti.min(loff, 2)

            if kstiff != 0.0:
                l = cons[cidx].L[loff]

                alpha = 1.0 / kstiff
                alpha /= dt * dt

                C = volume - restlength

                dsum = 0.0
                gamma = 1.0
                if kdampratio > 0.0:
                    prev0 = pprev[pts[0]]
                    prev1 = pprev[pts[1]]
                    prev2 = pprev[pts[2]]
                    prev3 = pprev[pts[3]]
                    beta = kstiff * kdampratio * dt * dt
                    gamma = alpha * beta / dt
                    dsum = grad0.dot(pos[pts[0]] - prev0) + grad1.dot(pos[pts[1]] - prev1) + grad2.dot(pos[pts[2]] - prev2) + grad3.dot(pos[pts[3]] - prev3)
                    dsum *= gamma
                    gamma += 1.0

                dlambda = (-C - alpha * l - dsum) / (gamma * w_sum + alpha)

                if use_jacobi:
                    updatedP(dP, dPw, dlambda * inv_masses[0] * grad0, pts[0])
                    updatedP(dP, dPw, dlambda * inv_masses[1] * grad1, pts[1])
                    updatedP(dP, dPw, dlambda * inv_masses[2] * grad2, pts[2])
                    updatedP(dP, dPw, dlambda * inv_masses[3] * grad3, pts[3])
                else:
                    pos[pts[0]] += dlambda * inv_masses[0] * grad0
                    pos[pts[1]] += dlambda * inv_masses[1] * grad1
                    pos[pts[2]] += dlambda * inv_masses[2] * grad2
                    pos[pts[3]] += dlambda * inv_masses[3] * grad3
                    cons[cidx].L[loff] = cons[cidx].L[loff] + dlambda


    # Reference: pbd_constraints.cl:L1093 tetFiberUpdateXPBD
    @ti.func
    def tet_fiber_update_xpbd(
        self,
        use_jacobi: ti.template(),
        pos: ti.template(),
        pprev: ti.template(),
        dP: ti.template(),
        dPw: ti.template(),
        cidx: ti.i32,
        cons: ti.template(),
        pts: ti.types.vector(4, ti.i32),
        dt: ti.f32,
        fiber: ti.types.vector(3, ti.f32),
        stiffness: ti.f32,
        Dminv: ti.types.matrix(3, 3, ti.f32),
        kdampratio: ti.f32,
        restlength: ti.f32,
        restvector: ti.types.vector(4, ti.f32),
        acti: ti.f32,
        mass: ti.template(),
        stopped: ti.template(),
        target_stretch: ti.f32 = 1.0,
    ):
        inv_masses = ti.Vector([0.0, 0.0, 0.0, 0.0])
        for i in ti.static(range(4)):
            inv_masses[i] = get_inv_mass(pts[i], mass, stopped)
        l = cons[cidx].L[0]
        alpha = 1.0 / stiffness
        alpha /= restlength  # NORMSTIFFNESS
        alpha /= dt * dt
        grad_scale = 1.0
        psi = 0.0

        _Ds = ti.Matrix.cols([pos[pts[i]] - pos[pts[3]] for i in range(3)])

        # C = psi = 0.5 * ||Fw||^2
        # gradC = F w w^T Dm^-T
        wTDminvT = restvector.xyz
        FwT = wTDminvT @ _Ds.transpose()
        psi = 0.5 * FwT.norm_sqr()

        if psi > 1e-9:
            psi_sqrt = ti.sqrt(2.0 * psi)
            grad_scale = 1.0 / psi_sqrt
            psi = psi_sqrt

        Ht = ti.Matrix.outer_product(wTDminvT, FwT)

        grad0 = grad_scale * ti.Vector([Ht[0, 0], Ht[0, 1], Ht[0, 2]])
        grad1 = grad_scale * ti.Vector([Ht[1, 0], Ht[1, 1], Ht[1, 2]])
        grad2 = grad_scale * ti.Vector([Ht[2, 0], Ht[2, 1], Ht[2, 2]])
        grad3 = -grad0 - grad1 - grad2

        w_sum = (inv_masses[0] * grad0.norm_sqr() +
                 inv_masses[1] * grad1.norm_sqr() +
                 inv_masses[2] * grad2.norm_sqr() +
                 inv_masses[3] * grad3.norm_sqr())

        if w_sum > 1e-9:
            dsum = 0.0
            gamma = 1.0
            if kdampratio > 0:
                beta = stiffness * kdampratio * dt * dt
                beta *= restlength  # NORMSTIFFNESS
                gamma = alpha * beta / dt

                dsum = (grad0.dot(pos[pts[0]] - pprev[pts[0]]) +
                        grad1.dot(pos[pts[1]] - pprev[pts[1]]) +
                        grad2.dot(pos[pts[2]] - pprev[pts[2]]) +
                        grad3.dot(pos[pts[3]] - pprev[pts[3]]))
                dsum *= gamma
                gamma += 1.0

            C = psi - target_stretch
            dL = (-C - alpha * l - dsum) / (gamma * w_sum + alpha)
            if use_jacobi:
                updatedP(dP, dPw, dL * inv_masses[0] * grad0, pts[0])
                updatedP(dP, dPw, dL * inv_masses[1] * grad1, pts[1])
                updatedP(dP, dPw, dL * inv_masses[2] * grad2, pts[2])
                updatedP(dP, dPw, dL * inv_masses[3] * grad3, pts[3])
            else:
                pos[pts[0]] += dL * inv_masses[0] * grad0
                pos[pts[1]] += dL * inv_masses[1] * grad1
                pos[pts[2]] += dL * inv_masses[2] * grad2
                pos[pts[3]] += dL * inv_masses[3] * grad3

                cons[cidx].L[0] += dL

    @ti.func
    def attach_bilateral_update(
        self,
        use_jacobi: ti.template(),
        cidx: ti.i32,
        cons: ti.template(),
        pt1: ti.i32,
        p0: ti.types.vector(3, ti.f32),
        p1: ti.types.vector(3, ti.f32),
        pos: ti.template(),
        pprev: ti.template(),
        dP: ti.template(),
        dPw: ti.template(),
        mass: ti.template(),
        stopped: ti.template(),
        restlength: ti.f32,
        kstiff: ti.f32,
        kdampratio: ti.f32,
        kstiffcompress: ti.f32,
    ):
        """Bilateral attach: position correction plus reaction accumulation."""
        invmass1 = get_inv_mass(pt1, mass, stopped)
        wsum = invmass1

        if wsum != 0.0:
            p1_current = pos[pt1]
            p0_current = p0

            n = p1_current - p0_current
            d = n.norm()
            if d >= 1e-6:
                loff = self.inCompressBand(d, restlength)
                kstiff_val = kstiffcompress if loff else kstiff
                if kstiff_val != 0.0:
                    l = cons[cidx].L[loff]

                    alpha = 1.0 / kstiff_val
                    alpha /= self.dt * self.dt

                    C = d - restlength
                    n = n / d

                    dsum = 0.0
                    gamma = 1.0
                    if kdampratio > 0.0:
                        prev1 = pprev[pt1]
                        beta = kstiff_val * kdampratio * self.dt * self.dt
                        gamma = alpha * beta / self.dt
                        dsum = gamma * n.dot(p1_current - prev1)
                        gamma += 1.0

                    dL = (-C - alpha * l - dsum) / (gamma * wsum + alpha)
                    dp = n * (-dL)

                    if use_jacobi:
                        updatedP(dP, dPw, -invmass1 * dp, pt1)
                    else:
                        pos[pt1] -= invmass1 * dp
                        cons[cidx].L[loff] = cons[cidx].L[loff] + dL

                    self.reaction_accum[cidx] += C * n

    # FIXME: the pt0 is target and pt1 is source, which is opposite to intuitive
    @ti.func
    def distance_pos_update_xpbd(
        self,
        use_jacobi: ti.template(),
        cidx: ti.i32,
        cons: ti.template(),
        pt0: ti.i32,
        pt1: ti.i32,
        p0: ti.types.vector(3, ti.f32),
        p1: ti.types.vector(3, ti.f32),
        pos: ti.template(),
        pprev: ti.template(),
        dP: ti.template(),
        dPw: ti.template(),
        mass: ti.template(),
        stopped: ti.template(),
        restlength: ti.f32,
        kstiff: ti.f32,
        kdampratio: ti.f32,
        kstiffcompress: ti.f32,
    ):
        invmass0 = 0.0
        invmass1 = get_inv_mass(pt1, mass, stopped)

        if pt0 >= 0:
            invmass0 = get_inv_mass(pt0, mass, stopped)

        wsum = invmass0 + invmass1

        if wsum != 0.0:
            p1_current = pos[pt1]
            p0_current = p0
            if pt0 >= 0:
                p0_current = pos[pt0]

            n = p1_current - p0_current
            d = n.norm()
            if d >= 1e-6:
                loff = self.inCompressBand(d, restlength)
                kstiff_val = kstiffcompress if loff else kstiff
                if kstiff_val != 0.0:
                    l = cons[cidx].L[loff]

                    alpha = 1.0 / kstiff_val
                    alpha /= self.dt * self.dt

                    C = d - restlength
                    n = n / d
                    gradC = n

                    dsum = 0.0
                    gamma = 1.0
                    if kdampratio > 0.0:
                        if pt0 >= 0:
                            prev0 = pprev[pt0]
                            prev1 = pprev[pt1]
                            beta = kstiff_val * kdampratio * self.dt * self.dt
                            gamma = alpha * beta / self.dt
                            dsum = gamma * (-gradC.dot(p0_current - prev0) + gradC.dot(p1_current - prev1))
                        else:
                            prev1 = pprev[pt1]
                            beta = kstiff_val * kdampratio * self.dt * self.dt
                            gamma = alpha * beta / self.dt
                            dsum = gamma * gradC.dot(p1_current - prev1)
                        gamma += 1.0

                    dL = (-C - alpha * l - dsum) / (gamma * wsum + alpha)
                    dp = n * (-dL)

                    if use_jacobi:
                        if pt0 >= 0:
                            updatedP(dP, dPw, invmass0 * dp, pt0)
                        updatedP(dP, dPw, -invmass1 * dp, pt1)
                    else:
                        if pt0 >= 0:
                            pos[pt0] += invmass0 * dp
                        pos[pt1] -= invmass1 * dp
                        cons[cidx].L[loff] = cons[cidx].L[loff] + dL

    @ti.func
    def tet_arap_update_xpbd(
        self,
        use_jacobi: ti.template(),
        cidx: ti.i32,
        cons: ti.template(),
        pts: ti.types.vector(4, ti.i32),
        dt: ti.f32,
        pos: ti.template(),
        pprev: ti.template(),
        dP: ti.template(),
        dPw: ti.template(),
        mass: ti.template(),
        stopped: ti.template(),
        restlength: ti.f32,
        restvector: ti.types.vector(4, ti.f32),
        restmatrix: ti.types.matrix(3, 3, ti.f32),
        kstiff: ti.f32,
        kdampratio: ti.f32,
        flags: ti.i32,
    ):
        pt0 = pts[0]
        pt1 = pts[1]
        pt2 = pts[2]
        pt3 = pts[3]
        p0 = pos[pt0]
        p1 = pos[pt1]
        p2 = pos[pt2]
        p3 = pos[pt3]

        invmass0 = get_inv_mass(pt0, mass, stopped)
        invmass1 = get_inv_mass(pt1, mass, stopped)
        invmass2 = get_inv_mass(pt2, mass, stopped)
        invmass3 = get_inv_mass(pt3, mass, stopped)

        Ds = ti.Matrix.cols([p0 - p3, p1 - p3, p2 - p3])
        F = Ds @ restmatrix

        _, R = polar_decomposition(F)
        d = F - R
        cons[cidx].restvector = mat3_to_quat(R)

        psi = squared_norm3(d)
        gradscale = 2.0
        if (flags & (LINEARENERGY | NORMSTIFFNESS)) != 0:
            psi = ti.sqrt(psi)
            gradscale = 1.0 / psi

        if psi >= 1e-6:
            Ht = restmatrix @ d.transpose()
            grad0 = gradscale * ti.Vector([Ht[0, 0], Ht[0, 1], Ht[0, 2]])
            grad1 = gradscale * ti.Vector([Ht[1, 0], Ht[1, 1], Ht[1, 2]])
            grad2 = gradscale * ti.Vector([Ht[2, 0], Ht[2, 1], Ht[2, 2]])
            grad3 = -grad0 - grad1 - grad2

            wsum = (invmass0 * grad0.dot(grad0) +
                    invmass1 * grad1.dot(grad1) +
                    invmass2 * grad2.dot(grad2) +
                    invmass3 * grad3.dot(grad3))

            if wsum != 0.0:
                alpha = 1.0 / kstiff
                if (flags & NORMSTIFFNESS) != 0:
                    alpha /= restlength
                alpha /= dt * dt

                dsum = 0.0
                gamma = 1.0
                if kdampratio > 0.0:
                    prev0 = pprev[pt0]
                    prev1 = pprev[pt1]
                    prev2 = pprev[pt2]
                    prev3 = pprev[pt3]
                    beta = kstiff * kdampratio * dt * dt
                    if (flags & NORMSTIFFNESS) != 0:
                        beta *= restlength
                    gamma = alpha * beta / dt
                    dsum = (grad0.dot(p0 - prev0) +
                            grad1.dot(p1 - prev1) +
                            grad2.dot(p2 - prev2) +
                            grad3.dot(p3 - prev3))
                    dsum *= gamma
                    gamma += 1.0

                C = psi
                dL = (-C - alpha * cons[cidx].L[0] - dsum) / (gamma * wsum + alpha)
                if use_jacobi:
                    updatedP(dP, dPw, dL * invmass0 * grad0, pt0)
                    updatedP(dP, dPw, dL * invmass1 * grad1, pt1)
                    updatedP(dP, dPw, dL * invmass2 * grad2, pt2)
                    updatedP(dP, dPw, dL * invmass3 * grad3, pt3)
                else:
                    pos[pt0] += dL * invmass0 * grad0
                    pos[pt1] += dL * invmass1 * grad1
                    pos[pt2] += dL * invmass2 * grad2
                    pos[pt3] += dL * invmass3 * grad3
                    cons[cidx].L[0] += dL

    def run(self):
        self.step_cnt = 1
        while self.step_cnt <= self.cfg.nsteps:
            self.vis._render_control()
            if self.cfg.reset:
                self.reset()
                self.step_cnt = 1
                self.cfg.reset = False
            if not self.cfg.pause:
                self.step_start_time = time.perf_counter()
                self.step()
                self.step_cnt += 1
                self.step_end_time = time.perf_counter()
            if self.cfg.gui or getattr(self.cfg, 'save_image', False):
                self.vis._render_frame(self.step_cnt, getattr(self.cfg, 'save_image', False))
        while self.cfg.gui and hasattr(self.vis, 'window') and self.vis.window.running:
            self.vis._render_frame(self.step_cnt, getattr(self.cfg, 'save_image', False))


def main():
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "data/muscle/config/bicep.json"
    print("Using config:", config_path)
    cfg = load_config(config_path)
    sim = MuscleSim(cfg)
    print("Running for", cfg.nsteps, "steps.")
    sim.run()


if __name__ == "__main__":
    main()
