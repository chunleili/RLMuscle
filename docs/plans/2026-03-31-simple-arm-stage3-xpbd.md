# SimpleArm Stage 3: XPBD + ATTACH + Deformation Gradient 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 XPBD MuscleSim (ATTACH 弹性边界) 替代 VBD (kinematic 边界)，解决网格爆炸问题，同时保留 deformation gradient 力提取路径。

**Architecture:** XPBD solver 提供弹性约束（TETVOLUME + TETFIBERDGF + ATTACH + PIN），insertion 端通过 ATTACH 弹簧耦合到 MuJoCo 骨骼位置，origin 端用 PIN 固定。力提取仍走 deformation gradient → DGF curves → MuJoCo motor actuator。MuJoCo 负责刚体动力学和 activation dynamics。

**Tech Stack:** Warp (XPBD MuscleSim), MuJoCo, NumPy, existing `src/VMuscle/` 基础设施

---

## 背景

### 为什么从 VBD 切换到 XPBD

Stage 3 的 VBD 方案在 step 17 (t≈0.27s) 发生网格爆炸：kinematic 边界直接位移导致边界 tet 极度拉伸 (det(F) > 3.5)，VBD 迭代无法恢复，连锁翻转。

XPBD 的 ATTACH 约束通过弹簧隐式参与求解，可以处理大位移而不会突变。且 XPBD 基础设施 (`MuscleSim`, `SolverMuscleBoneCoupled`) 已经成熟。

### 架构概览

```
每个 outer step (dt=0.0167s):
  ┌─────────────────────────────────────────────┐
  │ 1. 更新 bone_pos_field ← MuJoCo insertion   │
  │ 2. 设置 activation → TETFIBERDGF            │
  │ 3. XPBD substeps:                           │
  │    integrate → clear → update_attach_targets │
  │    → clear_reaction → solve_constraints      │
  │    → update_velocities                       │
  │    (PIN 固定 origin, ATTACH 拉 insertion,    │
  │     TETVOLUME + TETFIBERDGF 驱动 interior)  │
  │ 4. 读 vertex positions                      │
  │    → compute_fiber_stretches (F = Ds@Dm⁻¹)  │
  │    → DGF curves → muscle force              │
  │ 5. MuJoCo substeps (activation dynamics +   │
  │    force injection + rigid body dynamics)    │
  └─────────────────────────────────────────────┘
```

### 关键设计决策

1. **不使用 `SolverMuscleBoneCoupled`**：它需要 Newton 刚体模型，而 SimpleArm 用的是 MuJoCo。直接操作 MuscleSim 更简单。
2. **不使用 config JSON 驱动 MuscleSim**：MuscleSim 的标准初始化需要 USD 文件和 mask-based ATTACH。我们写一个轻量子类 `SimpleArmMuscleSim`，用编程方式创建 mesh 和 constraints。
3. **力提取走 deformation gradient**：不走 ATTACH reaction force → torque 路径（那是 `SolverMuscleBoneCoupled` 的方式）。保持与 Stage 2/3 一致的 `compute_fiber_stretches` → DGF curves 路径。
4. **PIN vs kinematic**：origin 端用 PIN 约束（高刚度弹簧拉向固定位置），而非 mass=0 kinematic。这避免了 kinematic/dynamic 边界的突变问题。

---

## 文件结构

| 文件 | 操作 | 职责 |
|------|------|------|
| `examples/example_xpbd_coupled_simple_arm.py` | 新建 | Stage 3 XPBD 主程序：`SimpleArmMuscleSim` 子类 + 耦合循环 |
| `scripts/run_simple_arm_comparison.py` | 修改 | 添加 `--mode xpbd` 选项调用新 example |
| `data/simpleArm/config.json` | 修改 | 添加 `xpbd` 配置段 |
| `tests/test_xpbd_simple_arm.py` | 新建 | 冒烟测试：mesh 不爆炸 + 力提取合理 |

---

## Task 1: 创建 SimpleArmMuscleSim 子类

`MuscleSim` 要求从 USD 加载 mesh。我们子类化绕过文件加载，直接注入编程创建的 cylinder mesh + 手动 constraints。

**Files:**
- Create: `examples/example_xpbd_coupled_simple_arm.py`

- [ ] **Step 1: 创建文件骨架和 SimpleArmMuscleSim 类**

```python
"""XPBD bone-muscle coupling for SimpleArm (Stage 3).

Uses XPBD MuscleSim with ATTACH constraints for stable boundary coupling.
Force extracted from deformation gradient via DGF curves (same as VBD Stage 3).

Usage:
    uv run python examples/example_xpbd_coupled_simple_arm.py
    uv run python examples/example_xpbd_coupled_simple_arm.py --config data/simpleArm/config.json
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import warp as wp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from VMuscle.activation import activation_dynamics_step_np
from VMuscle.constraints import ATTACH, PIN, TETFIBERDGF, TETVOLUME
from VMuscle.dgf_curves import active_force_length, force_velocity, passive_force_length
from VMuscle.mesh_io import UsdTetExporter
from VMuscle.mesh_utils import create_cylinder_tet_mesh
from VMuscle.muscle_warp import MuscleSim


def load_config(path="data/simpleArm/config.json"):
    with open(path) as f:
        return json.load(f)


class SimpleArmMuscleSim(MuscleSim):
    """Lightweight MuscleSim subclass for programmatic cylinder mesh.

    Bypasses MuscleSimBase.__init__ file-loading. Injects:
      - Cylinder tet mesh (pos0_np, tet_np)
      - Fiber directions along tendon axis
      - ATTACH constraints on insertion vertices → bone target
      - PIN constraints on origin vertices → fixed world position
      - TETVOLUME + TETFIBERDGF for all tets
    """

    def __init__(self, vertices, tets, fiber_dirs,
                 origin_ids, insertion_ids,
                 bone_target_positions,
                 sigma0, dt, num_substeps,
                 attach_stiffness=1e8,
                 pin_stiffness=1e10,
                 fiber_stiffness_scale=200.0,
                 contraction_ratio=0.4,
                 fiber_damping=0.5,
                 density=1060.0,
                 k_mu=1000.0, k_lambda=10000.0, k_damp=1.0,
                 device="cpu"):
        # Skip MuscleSimBase.__init__ entirely — set up manually
        self._init_backend()

        # Store mesh data (normally loaded from file)
        self.pos0_np = vertices.astype(np.float32)
        self.tet_np = tets.astype(np.int32)
        self.n_verts = len(vertices)
        self.v_fiber_np = fiber_dirs.astype(np.float32)
        self.v_tendonmask_np = np.zeros(self.n_verts, dtype=np.float32)
        self.geo = None

        # Bone data: one "bone vertex" per insertion vertex
        n_bone = len(bone_target_positions)
        self.bone_pos = bone_target_positions.astype(np.float32)
        self.bone_indices_np = np.empty(0, dtype=np.int32)  # no bone triangles
        self.bone_muscle_ids = {}

        # Minimal config
        self.cfg = SimpleNamespace(
            dt=dt,
            num_substeps=num_substeps,
            gravity=0.0,
            density=density,
            veldamping=0.02,
            constraints=[],
            arch=device,
            gui=False,
            render_mode=None,
            activation=0.0,
            contraction_ratio=contraction_ratio,
            fiber_stiffness_scale=fiber_stiffness_scale,
            HAS_compressstiffness=False,
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=k_damp,
        )

        # Store construction parameters for constraint building
        self._origin_ids = np.asarray(origin_ids, dtype=np.int32)
        self._insertion_ids = np.asarray(insertion_ids, dtype=np.int32)
        self._bone_targets = bone_target_positions.astype(np.float32)
        self._attach_stiffness = attach_stiffness
        self._pin_stiffness = pin_stiffness
        self._sigma0 = sigma0
        self._fiber_damping = fiber_damping

        # Standard init sequence (matching MuscleSimBase.__init__ post-load)
        self._allocate_fields()
        self._init_fields()
        self._precompute_rest()
        self._build_surface_tris()
        self.use_jacobi = False
        arch = device.lower()
        self.use_colored_gs = arch != 'cpu'
        self.build_constraints()
        self.contraction_ratio = contraction_ratio
        self.fiber_stiffness_scale = fiber_stiffness_scale
        self.has_compressstiffness = False
        self.dt = dt / num_substeps
        self.step_cnt = 0
        self.renderer = None

    def _init_fields(self):
        """Upload numpy data to Warp fields."""
        self.pos.assign(wp.from_numpy(self.pos0_np, dtype=wp.vec3))
        self.pprev.assign(wp.from_numpy(self.pos0_np, dtype=wp.vec3))
        self.pos0.assign(wp.from_numpy(self.pos0_np, dtype=wp.vec3))

        # Mass from density and rest volume
        # (simplified: distribute tet volume equally to 4 vertices)
        mass_np = np.full(self.n_verts, 1e-6, dtype=np.float32)  # small default
        rest_vol = self.rest_volume.numpy()
        tet_idx = self.tet_np
        for t in range(len(tet_idx)):
            m_tet = self.cfg.density * abs(float(rest_vol[t])) / 4.0
            for v in tet_idx[t]:
                mass_np[v] += m_tet
        self.mass.assign(wp.from_numpy(mass_np, dtype=wp.float32))

        # Per-vertex fiber direction (average from incident tets)
        v_fiber = np.zeros((self.n_verts, 3), dtype=np.float32)
        v_count = np.zeros(self.n_verts, dtype=np.float32)
        for t in range(len(tet_idx)):
            for v in tet_idx[t]:
                v_fiber[v] += self.v_fiber_np[t]
                v_count[v] += 1.0
        v_count = np.maximum(v_count, 1.0)
        v_fiber /= v_count[:, None]
        norms = np.linalg.norm(v_fiber, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        v_fiber /= norms
        self.v_fiber_dir.assign(wp.from_numpy(v_fiber, dtype=wp.vec3))

        # Tet indices
        self.tet_indices.assign(
            wp.from_numpy(self.tet_np.astype(np.int32), dtype=wp.vec4i)
        )

        # Bone positions
        self._create_bone_fields()

    def _precompute_rest(self):
        """Compute rest volumes and inverse rest matrices."""
        n_tet = len(self.tet_np)
        rest_vol_np = np.zeros(n_tet, dtype=np.float32)
        rest_mat_np = np.zeros((n_tet, 3, 3), dtype=np.float32)
        for t in range(n_tet):
            i, j, k, l = self.tet_np[t]
            p = self.pos0_np
            Ds = np.column_stack([p[j] - p[i], p[k] - p[i], p[l] - p[i]])
            vol = np.linalg.det(Ds) / 6.0
            rest_vol_np[t] = vol
            rest_mat_np[t] = np.linalg.inv(Ds)
        self.rest_volume.assign(wp.from_numpy(rest_vol_np, dtype=wp.float32))
        self.rest_matrix.assign(wp.from_numpy(rest_mat_np, dtype=wp.mat33))

    def build_constraints(self):
        """Create XPBD constraints programmatically."""
        all_constraints = []

        # Helper: map vertex to containing tet
        pt2tet = {}
        for t in range(len(self.tet_np)):
            for v in self.tet_np[t]:
                if v not in pt2tet:
                    pt2tet[v] = t

        # 1. TETVOLUME for all tets
        rest_vol_np = self.rest_volume.numpy()
        rest_mat_np = self.rest_matrix.numpy()
        for t in range(len(self.tet_np)):
            i, j, k, l = self.tet_np[t]
            Dminv = rest_mat_np[t]
            c = dict(
                type=TETVOLUME,
                pts=[int(i), int(j), int(k), int(l)],
                stiffness=float(self.cfg.k_lambda),
                dampingratio=float(self.cfg.k_damp),
                tetid=t,
                L=[0.0, 0.0, 0.0],
                restlength=float(rest_vol_np[t]),
                restvector=[float(Dminv[0, 0]), float(Dminv[1, 0]), float(Dminv[2, 0]), 0.0],
                restdir=[0.0, 0.0, 0.0],
                compressionstiffness=-1.0,
            )
            all_constraints.append(c)

        # 2. TETFIBERDGF for all tets
        for t in range(len(self.tet_np)):
            i, j, k, l = self.tet_np[t]
            fiber_dir = self.v_fiber_np[t]
            Dminv = rest_mat_np[t]
            # restvector stores Dm_inv^T @ fiber_dir (precomputed for kernel)
            wTDminvT = Dminv.T @ fiber_dir
            c = dict(
                type=TETFIBERDGF,
                pts=[int(i), int(j), int(k), int(l)],
                stiffness=float(self.cfg.k_mu),
                dampingratio=float(self._fiber_damping),
                tetid=t,
                L=[0.0, 0.0, 0.0],
                restlength=1.0,  # rest fiber stretch = 1
                restvector=[float(wTDminvT[0]), float(wTDminvT[1]), float(wTDminvT[2]), 0.0],
                restdir=[float(self._sigma0), float(self.contraction_ratio), 0.0],
                compressionstiffness=-1.0,
            )
            all_constraints.append(c)

        # 3. PIN for origin vertices (fixed to world)
        for vid in self._origin_ids:
            pos = self.pos0_np[vid]
            c = dict(
                type=PIN,
                pts=[int(vid), -1, -1, -1],
                stiffness=self._pin_stiffness,
                dampingratio=0.1,
                tetid=pt2tet.get(int(vid), -1),
                L=[0.0, 0.0, 0.0],
                restlength=0.0,
                restvector=[float(pos[0]), float(pos[1]), float(pos[2]), 1.0],
                restdir=[0.0, 0.0, 0.0],
                compressionstiffness=-1.0,
            )
            all_constraints.append(c)

        # 4. ATTACH for insertion vertices → bone targets
        for idx, vid in enumerate(self._insertion_ids):
            tgt_pos = self._bone_targets[idx]
            dist = float(np.linalg.norm(self.pos0_np[vid] - tgt_pos))
            c = dict(
                type=ATTACH,
                pts=[int(vid), -1, int(idx), -1],  # pts[2] = bone vertex index
                stiffness=self._attach_stiffness,
                dampingratio=0.1,
                tetid=pt2tet.get(int(vid), -1),
                L=[0.0, 0.0, 0.0],
                restlength=dist,
                restvector=[float(tgt_pos[0]), float(tgt_pos[1]), float(tgt_pos[2]), 1.0],
                restdir=[0.0, 0.0, 0.0],
                compressionstiffness=-1.0,
            )
            all_constraints.append(c)

        # Store raw constraints for inspection
        self.raw_constraints = all_constraints
        self.attach_constraints = [c for c in all_constraints if c['type'] == ATTACH]

        # Call parent's constraint upload (sort + convert to Warp array)
        # Reuse MuscleSim.build_constraints() logic but with our raw_constraints
        self.constraint_configs = []  # prevent _collect_raw_constraints from running
        # Directly set raw_constraints and process
        self._upload_constraints(all_constraints)

    def _upload_constraints(self, all_constraints):
        """Upload constraint list to Warp arrays (extracted from MuscleSim.build_constraints)."""
        from VMuscle.muscle_warp import Constraint

        all_constraints.sort(key=lambda c: c['type'])
        n_cons = len(all_constraints)
        self.n_cons = n_cons
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

            cons_np = np.zeros(n_cons, dtype=Constraint.numpy_dtype())
            cons_np['type'] = np.array([c['type'] for c in all_constraints], dtype=np.int32)
            cons_np['cidx'] = np.arange(n_cons, dtype=np.int32)
            cons_np['pts'] = np.array([c['pts'] for c in all_constraints], dtype=np.int32)
            cons_np['stiffness'] = np.array([c['stiffness'] for c in all_constraints], dtype=np.float32)
            cons_np['dampingratio'] = np.array([c['dampingratio'] for c in all_constraints], dtype=np.float32)
            cons_np['tetid'] = np.array([c['tetid'] for c in all_constraints], dtype=np.int32)
            cons_np['L'] = np.array([c['L'] for c in all_constraints], dtype=np.float32)
            cons_np['restlength'] = np.array([c['restlength'] for c in all_constraints], dtype=np.float32)
            cons_np['restvector'] = np.array([c['restvector'] for c in all_constraints], dtype=np.float32)
            cons_np['restdir'] = np.array([c['restdir'] for c in all_constraints], dtype=np.float32)
            cons_np['compressionstiffness'] = np.array(
                [c['compressionstiffness'] for c in all_constraints], dtype=np.float32)

            self.cons = wp.array(cons_np, dtype=Constraint)
        else:
            self.cons = wp.zeros(0, dtype=Constraint)

        self.reaction_accum = wp.zeros(max(n_cons, 1), dtype=wp.vec3)
        print(f"Built {n_cons} constraints: "
              + ", ".join(f"{k}={v[1]}" for k, v in sorted(self.cons_ranges.items())))

    def update_bone_targets(self, new_positions):
        """Update bone_pos_field with new insertion target positions.

        Call this before each muscle step to set where ATTACH constraints pull toward.
        """
        bone_np = self.bone_pos_field.numpy()
        bone_np[:len(new_positions)] = new_positions.astype(np.float32)
        self.bone_pos_field = wp.from_numpy(bone_np, dtype=wp.vec3)
```

- [ ] **Step 2: 验证 SimpleArmMuscleSim 能实例化**

Run:
```bash
uv run python -c "
import sys; sys.path.insert(0, '.')
import numpy as np, warp as wp
from VMuscle.mesh_utils import create_cylinder_tet_mesh
from examples.example_xpbd_coupled_simple_arm import SimpleArmMuscleSim
wp.init()
verts, tets = create_cylinder_tet_mesh(0.3, 0.02, 8, 10)
n_tets = len(tets)
fiber_dirs = np.tile([0,0,1], (n_tets,1)).astype(np.float32)
origin_ids = [i for i in range(len(verts)) if verts[i,2] > 0.28]
insertion_ids = [i for i in range(len(verts)) if verts[i,2] < 0.02]
bone_targets = verts[insertion_ids].copy()
sim = SimpleArmMuscleSim(verts, tets, fiber_dirs,
    origin_ids, insertion_ids, bone_targets,
    sigma0=159155.0, dt=0.0167, num_substeps=10, device='cpu')
print(f'OK: {sim.n_verts} verts, {len(tets)} tets, {sim.n_cons} constraints')
"
```
Expected: 成功打印 constraint 统计

- [ ] **Step 3: Commit**

```bash
git add examples/example_xpbd_coupled_simple_arm.py
git commit -m "feat: add SimpleArmMuscleSim XPBD subclass for Stage 3"
```

---

## Task 2: 实现 XPBD 耦合主循环

在 `example_xpbd_coupled_simple_arm.py` 中添加 `xpbd_coupled_simple_arm()` 主函数，结构与现有 `vbd_coupled_simple_arm()` 对应但使用 XPBD。

**Files:**
- Modify: `examples/example_xpbd_coupled_simple_arm.py`

- [ ] **Step 1: 添加辅助函数和主循环**

在 `SimpleArmMuscleSim` 类之后添加：

```python
def _rotation_matrix_from_z_to(target_dir):
    """Compute rotation matrix that rotates Z-axis to target_dir."""
    z = np.array([0, 0, 1], dtype=np.float64)
    t = np.asarray(target_dir, dtype=np.float64)
    t = t / np.linalg.norm(t)
    cos_theta = np.dot(z, t)
    if cos_theta > 1.0 - 1e-8:
        return np.eye(3, dtype=np.float32)
    if cos_theta < -1.0 + 1e-8:
        return np.diag([1, -1, -1]).astype(np.float32)
    axis = np.cross(z, t)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(np.arccos(cos_theta)) * K + (1 - cos_theta) * (K @ K)
    return R.astype(np.float32)


def compute_fiber_stretches(pos, tet_idx, tet_poses, fiber_dirs):
    """Compute per-tet fiber stretch via deformation gradient F = Ds @ Dm_inv."""
    n = len(tet_idx)
    stretches = np.empty(n)
    for e in range(n):
        i, j, k, l = tet_idx[e]
        Ds = np.column_stack([pos[j] - pos[i], pos[k] - pos[i], pos[l] - pos[i]])
        Fd = (Ds @ tet_poses[e]) @ fiber_dirs[e]
        stretches[e] = max(np.linalg.norm(Fd), 1e-8)
    return stretches


def xpbd_coupled_simple_arm(cfg, verbose=True):
    """Run XPBD + MuJoCo coupled SimpleArm simulation.

    XPBD MuscleSim with ATTACH constraints for stable boundary coupling.
    Force extracted from deformation gradient via DGF curves.

    Returns:
        dict with times, elbow_angles, forces, norm_fiber_lengths, activations.
    """
    mus = cfg["muscle"]
    act_cfg = cfg["activation"]
    sol = cfg["solver"]
    ic = cfg["initial_conditions"]
    xpbd_cfg = cfg.get("xpbd", {})

    F_max = mus["max_isometric_force"]
    L_opt = mus["optimal_fiber_length"]
    L_slack = mus["tendon_slack_length"]
    V_max = mus["max_contraction_velocity"]
    d_damp = mus["fiber_damping"]

    outer_dt = sol["dt"]
    n_steps = sol["n_steps"]
    theta0 = np.radians(ic["elbow_angle_deg"])
    device = sol.get("arch", "cuda:0" if wp.is_cuda_available() else "cpu")

    num_substeps = xpbd_cfg.get("num_substeps", 10)
    attach_stiffness = xpbd_cfg.get("attach_stiffness", 1e8)
    pin_stiffness = xpbd_cfg.get("pin_stiffness", 1e10)
    fiber_stiffness_scale = xpbd_cfg.get("fiber_stiffness_scale", 200.0)
    warmup_steps = xpbd_cfg.get("warmup_steps", 50)

    # --- Build MuJoCo ---
    _examples_dir = os.path.dirname(os.path.abspath(__file__))
    if _examples_dir not in sys.path:
        sys.path.insert(0, _examples_dir)
    from example_mujoco_simple_arm import build_mjcf

    mjcf_str = build_mjcf(cfg)
    mj_model = mujoco.MjModel.from_xml_string(mjcf_str)
    mj_data = mujoco.MjData(mj_model)
    mj_dt = mj_model.opt.timestep
    mj_substeps = max(1, int(round(outer_dt / mj_dt)))

    mj_data.qpos[0] = theta0
    mujoco.mj_forward(mj_model, mj_data)

    origin_site_id = mj_model.site("muscle_origin").id
    insertion_site_id = mj_model.site("muscle_insertion").id
    origin_pos = mj_data.site_xpos[origin_site_id].copy()
    insertion_pos = mj_data.site_xpos[insertion_site_id].copy()

    ten_length_init = float(mj_data.ten_length[0])
    fiber_length_init = ten_length_init - L_slack

    if verbose:
        print(f"[XPBD] origin={origin_pos}, insertion={insertion_pos}")
        print(f"[XPBD] ten_length={ten_length_init:.4f}, "
              f"fiber_length={fiber_length_init:.4f}, "
              f"l_tilde={fiber_length_init / L_opt:.4f}")

    # --- Build cylinder mesh in world frame ---
    geo = cfg["geometry"]
    r = geo["muscle_radius"]
    n_circ = geo["n_circumferential"]
    n_axial = geo["n_axial"]
    mesh_length = fiber_length_init

    tendon_vec = origin_pos - insertion_pos
    tendon_dir = (tendon_vec / np.linalg.norm(tendon_vec)).astype(np.float32)

    vertices, tets = create_cylinder_tet_mesh(mesh_length, r, n_circ, n_axial)
    R = _rotation_matrix_from_z_to(tendon_dir)
    vertices = (R @ vertices.T).T + insertion_pos.astype(np.float32)
    fiber_origin_pos = insertion_pos + tendon_dir.astype(np.float64) * mesh_length

    n_tets = len(tets)
    fiber_dirs = np.tile(tendon_dir, (n_tets, 1))
    sigma0 = F_max / (np.pi * r ** 2)

    # Identify boundary vertices
    origin_ids = []
    insertion_ids = []
    for i in range(len(vertices)):
        p = vertices[i].astype(np.float64)
        if np.linalg.norm(p - fiber_origin_pos) < r * 1.5:
            origin_ids.append(i)
        elif np.linalg.norm(p - insertion_pos) < r * 1.5:
            insertion_ids.append(i)

    # Bone targets = initial insertion vertex positions (updated each step)
    bone_targets = vertices[insertion_ids].copy()

    if verbose:
        print(f"[XPBD] mesh: {len(vertices)} verts, {n_tets} tets, "
              f"origin={len(origin_ids)}, insertion={len(insertion_ids)}")

    # --- Build XPBD MuscleSim ---
    wp.init()
    sim = SimpleArmMuscleSim(
        vertices, tets, fiber_dirs,
        origin_ids, insertion_ids, bone_targets,
        sigma0=sigma0,
        dt=outer_dt, num_substeps=num_substeps,
        attach_stiffness=attach_stiffness,
        pin_stiffness=pin_stiffness,
        fiber_stiffness_scale=fiber_stiffness_scale,
        contraction_ratio=mus.get("contraction_ratio", 0.4),
        fiber_damping=d_damp,
        device=device,
    )

    # Precompute for fiber stretch extraction
    tet_idx = tets.astype(np.int32)
    rest_mat = sim.rest_matrix.numpy()
    stretch_to_ltilde = mesh_length / L_opt

    # --- USD exporter ---
    anim_dir = "output/anim"
    os.makedirs(anim_dir, exist_ok=True)
    usd_path = os.path.join(anim_dir, "simple_arm_xpbd.usda")
    fps = int(round(1.0 / (mj_substeps * mj_dt)))
    exporter = UsdTetExporter(tet_idx, usd_path=usd_path, prim_path="/muscle", fps=fps)

    # --- Warm-up ---
    initial_activation = 0.5
    sim.activation.fill_(initial_activation)
    if verbose:
        print(f"[XPBD] Warming up ({warmup_steps} steps, a={initial_activation})...")
    for _ in range(warmup_steps):
        sim.integrate()
        sim.clear()
        sim.update_attach_targets()
        sim.clear_reaction()
        sim.solve_constraints()
        if sim.use_jacobi:
            sim.apply_dP()
        sim.update_velocities()

    pos_np = sim.pos.numpy()
    if verbose:
        stretches = compute_fiber_stretches(pos_np, tet_idx, rest_mat, fiber_dirs)
        print(f"[XPBD] Warm-up done: mean_stretch={stretches.mean():.4f}, "
              f"NaN={np.any(np.isnan(pos_np))}")

    # --- Simulation ---
    activation = initial_activation
    prev_fiber_length = fiber_length_init

    times, elbow_angles, forces_out = [], [], []
    norm_fiber_lengths, activations_out = [], []
    physics_time = 0.0

    if verbose:
        print(f"[XPBD] Simulating {n_steps * outer_dt:.1f}s, F_max={F_max:.0f}N")

    for step in range(n_steps):
        t = physics_time
        elbow_angle = float(mj_data.qpos[0])
        times.append(t)
        elbow_angles.append(elbow_angle)
        activations_out.append(activation)

        # 1. Update ATTACH targets from MuJoCo insertion site
        mujoco.mj_forward(mj_model, mj_data)
        new_insertion = mj_data.site_xpos[insertion_site_id].copy()
        # Compute displaced insertion vertex positions
        delta = new_insertion.astype(np.float32) - insertion_pos.astype(np.float32)
        new_bone_targets = bone_targets + delta  # broadcast displacement
        sim.update_bone_targets(new_bone_targets)

        # 2. Set activation
        sim.activation.fill_(activation)

        # 3. XPBD substeps
        for _ in range(num_substeps):
            sim.integrate()
            sim.clear()
            sim.update_attach_targets()
            sim.clear_reaction()
            sim.solve_constraints()
            if sim.use_jacobi:
                sim.apply_dP()
            sim.update_velocities()

        # 4. Extract fiber stretch from deformation gradient
        xpbd_pos = sim.pos.numpy()
        xpbd_valid = not np.any(np.isnan(xpbd_pos))
        if xpbd_valid:
            stretches = compute_fiber_stretches(xpbd_pos, tet_idx, rest_mat, fiber_dirs)
            l_tilde_xpbd = float(stretches.mean()) * stretch_to_ltilde
            if l_tilde_xpbd < 0.1 or l_tilde_xpbd > 3.0:
                xpbd_valid = False

        ten_length = float(mj_data.ten_length[0])
        fiber_length = ten_length - L_slack
        mj_ltilde = fiber_length / L_opt

        if not xpbd_valid:
            l_tilde_xpbd = mj_ltilde

        # Save USD frame
        exporter.save_frame(xpbd_pos.astype(np.float32), step)

        # 5. MuJoCo substeps with DGF force
        for sub in range(mj_substeps):
            t_sub = t + sub * mj_dt

            # Excitation schedule
            t_start = act_cfg["excitation_start_time"]
            t_end_exc = act_cfg["excitation_end_time"]
            e_off = act_cfg["excitation_off"]
            e_on = act_cfg["excitation_on"]
            if t_sub < t_start:
                excitation = e_off
            elif t_sub >= t_end_exc:
                excitation = e_on
            else:
                frac = (t_sub - t_start) / (t_end_exc - t_start)
                frac = frac * frac * (3.0 - 2.0 * frac)
                excitation = e_off + (e_on - e_off) * frac

            activation = float(activation_dynamics_step_np(
                np.array([excitation], dtype=np.float32),
                np.array([activation], dtype=np.float32),
                mj_dt,
                tau_act=act_cfg["tau_act"],
                tau_deact=act_cfg["tau_deact"],
            )[0])

            ten_len = float(mj_data.ten_length[0])
            fib_len = ten_len - L_slack
            fib_vel = (fib_len - prev_fiber_length) / mj_dt if (step > 0 or sub > 0) else 0.0
            prev_fiber_length = fib_len
            v_norm = fib_vel / (V_max * L_opt)

            l_tilde_1d = fib_len / L_opt
            delta_ltilde = l_tilde_xpbd - (fiber_length / L_opt)
            l_tilde_now = l_tilde_1d + delta_ltilde

            fl = float(active_force_length(l_tilde_now))
            fpe = float(passive_force_length(l_tilde_now))
            fv = float(force_velocity(np.clip(v_norm, -1.0, 1.0)))
            muscle_force = (activation * fl * fv + fpe + d_damp * v_norm) * F_max
            muscle_force = np.clip(muscle_force, 0.0, F_max * 2.0)

            mj_data.ctrl[0] = muscle_force
            mujoco.mj_step(mj_model, mj_data)
            physics_time += mj_dt

        forces_out.append(float(muscle_force))
        norm_fiber_lengths.append(l_tilde_xpbd)

        if verbose and (step % 100 == 0 or step == n_steps - 1):
            print(f"  step={step:4d} t={t:6.3f}s "
                  f"theta={np.degrees(elbow_angle):7.2f}deg "
                  f"F={muscle_force:7.2f}N a={activation:.4f} "
                  f"l_xpbd={l_tilde_xpbd:.4f} l_1d={mj_ltilde:.4f} "
                  f"delta={delta_ltilde:.6f}")

    if verbose:
        print(f"[XPBD] Done: {len(times)} points, "
              f"final angle={np.degrees(elbow_angles[-1]):.1f}deg")

    exporter.finalize()
    if verbose:
        print(f"USD animation saved to {usd_path}")

    # Save .sto
    os.makedirs("output", exist_ok=True)
    sto_path = "output/SimpleArm_XPBD_Coupled_states.sto"
    cols = [
        "/jointset/elbow/elbow_coord_0/value",
        "/forceset/biceps/activation",
        "/forceset/biceps/fiber_force",
        "/forceset/biceps/norm_fiber_length",
    ]
    with open(sto_path, "w") as f:
        f.write("SimpleArm_XPBD_Coupled\n")
        f.write("inDegrees=no\n")
        f.write(f"nColumns={len(cols) + 1}\n")
        f.write(f"nRows={len(times)}\n")
        f.write("DataType=double\nversion=3\nendheader\n")
        f.write("time\t" + "\t".join(cols) + "\n")
        for i in range(len(times)):
            f.write(f"{times[i]}\t{elbow_angles[i]}\t{activations_out[i]}\t"
                    f"{forces_out[i]}\t{norm_fiber_lengths[i]}\n")
    if verbose:
        print(f"STO saved to {sto_path}")

    return {
        "times": np.array(times),
        "elbow_angles": np.array(elbow_angles),
        "forces": np.array(forces_out),
        "norm_fiber_lengths": np.array(norm_fiber_lengths),
        "activations": np.array(activations_out),
        "max_iso_force": F_max,
        "muscle_type": "XPBD_Coupled",
    }


def main():
    parser = argparse.ArgumentParser(
        description="XPBD + MuJoCo coupled SimpleArm (Stage 3)")
    parser.add_argument("--config", default="data/simpleArm/config.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = xpbd_coupled_simple_arm(cfg)
    if result:
        print(f"\nFinal elbow angle: {np.degrees(result['elbow_angles'][-1]):.2f}deg")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证语法正确**

Run: `uv run python -m py_compile examples/example_xpbd_coupled_simple_arm.py`
Expected: 无输出（成功）

- [ ] **Step 3: 运行冒烟测试（短 10 步）**

Run:
```bash
uv run python -c "
import sys; sys.path.insert(0, '.')
from examples.example_xpbd_coupled_simple_arm import xpbd_coupled_simple_arm, load_config
cfg = load_config()
cfg['solver']['n_steps'] = 10
cfg['solver']['arch'] = 'cpu'
result = xpbd_coupled_simple_arm(cfg)
print(f'OK: {len(result[\"times\"])} steps, final={result[\"elbow_angles\"][-1]:.4f}')
"
```
Expected: 10 步完成，无 NaN，无爆炸

- [ ] **Step 4: Commit**

```bash
git add examples/example_xpbd_coupled_simple_arm.py
git commit -m "feat: add XPBD coupled simulation loop for SimpleArm Stage 3"
```

---

## Task 3: 添加 config 和 comparison 集成

**Files:**
- Modify: `data/simpleArm/config.json`
- Modify: `scripts/run_simple_arm_comparison.py`

- [ ] **Step 1: 添加 xpbd 配置段到 config.json**

在 `coupling` 段之后添加：

```json
{
  "xpbd": {
    "num_substeps": 10,
    "attach_stiffness": 1e8,
    "pin_stiffness": 1e10,
    "fiber_stiffness_scale": 200.0,
    "warmup_steps": 50
  }
}
```

- [ ] **Step 2: 在 run_simple_arm_comparison.py 添加 xpbd mode**

添加 `run_xpbd()` 函数和 `--mode xpbd` 选项：

```python
def run_xpbd(cfg):
    from examples.example_xpbd_coupled_simple_arm import xpbd_coupled_simple_arm
    return xpbd_coupled_simple_arm(cfg)
```

在 `main()` 的 choices 中添加 `"xpbd"`，在 plotting section 添加对应分支。

- [ ] **Step 3: 验证运行**

Run: `uv run python scripts/run_simple_arm_comparison.py --mode xpbd`
Expected: Stage 3 XPBD 完成，输出对比图到 `output/simple_arm_xpbd_vs_osim.png`

- [ ] **Step 4: Commit**

```bash
git add data/simpleArm/config.json scripts/run_simple_arm_comparison.py
git commit -m "feat: integrate XPBD Stage 3 into comparison pipeline"
```

---

## Task 4: 冒烟测试

**Files:**
- Create: `tests/test_xpbd_simple_arm.py`

- [ ] **Step 1: 写测试**

```python
"""Smoke test for XPBD coupled SimpleArm Stage 3."""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def cfg():
    from examples.example_xpbd_coupled_simple_arm import load_config
    cfg = load_config()
    cfg["solver"]["n_steps"] = 30
    cfg["solver"]["arch"] = "cpu"
    return cfg


def test_xpbd_no_explosion(cfg):
    """XPBD simulation completes 30 steps without NaN or crash."""
    from examples.example_xpbd_coupled_simple_arm import xpbd_coupled_simple_arm
    result = xpbd_coupled_simple_arm(cfg, verbose=False)
    assert result is not None
    assert len(result["times"]) == 30
    assert not np.any(np.isnan(result["elbow_angles"]))
    assert not np.any(np.isnan(result["forces"]))


def test_xpbd_force_reasonable(cfg):
    """Forces stay in physically reasonable range."""
    from examples.example_xpbd_coupled_simple_arm import xpbd_coupled_simple_arm
    result = xpbd_coupled_simple_arm(cfg, verbose=False)
    F_max = result["max_iso_force"]
    assert np.all(result["forces"] >= 0)
    assert np.all(result["forces"] <= F_max * 2.0)


def test_xpbd_fiber_stretch_nonzero(cfg):
    """Fiber lengths are in physiological range."""
    from examples.example_xpbd_coupled_simple_arm import xpbd_coupled_simple_arm
    result = xpbd_coupled_simple_arm(cfg, verbose=False)
    assert np.all(result["norm_fiber_lengths"] > 0.1)
    assert np.all(result["norm_fiber_lengths"] < 3.0)
```

- [ ] **Step 2: 运行测试**

Run: `uv run pytest tests/test_xpbd_simple_arm.py -v`
Expected: 3 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_xpbd_simple_arm.py
git commit -m "test: add XPBD SimpleArm smoke tests"
```

---

## Task 5: 完整运行验证 + 对比图

- [ ] **Step 1: 600 步完整运行**

Run: `uv run python scripts/run_simple_arm_comparison.py --mode xpbd`
Expected: 完成 600 步，无爆炸，输出 PNG + STO

- [ ] **Step 2: 检查结果**

验证：
- 稳态角度接近 82°（与 Stage 1/2 一致）
- delta ≠ 0（如果 XPBD 的弹性约束产生了 3D 效应）或 delta ≈ 0（直圆柱预期）
- 无 NaN，mesh 不爆炸

- [ ] **Step 3: 更新 progress doc**

写 `docs/progress/2026-03-31-simple-arm-stage3-xpbd.md`，记录：
- XPBD vs VBD 的结果对比
- ATTACH 边界是否解决了爆炸问题
- delta 值的变化（是否有 3D 效应）
- 下一步方向

---

## 注意事项

### SimpleArmMuscleSim 的关键假设

1. **v_fiber_np 是 per-tet**（不是 per-vertex）：`create_cylinder_tet_mesh` 返回的是 per-tet fiber directions
2. **bone_pos_field 的索引**：ATTACH constraints 的 `pts[2]` 指向 bone_pos_field 中的索引。我们创建 `len(insertion_ids)` 个 bone 顶点，每个对应一个 insertion vertex
3. **TETVOLUME restvector**：存储 `Dm_inv` 的第一列（用于 kernel 计算）。需要验证这是否符合 `solve_tetvolume_kernel` 的期望格式
4. **TETFIBERDGF restvector**：存储 `Dm_inv^T @ fiber_dir`（预计算给 kernel）。对比 `solve_tetfiberdgf_kernel:1000` 确认
5. **constraint upload**：`_upload_constraints` 复制了 `MuscleSim.build_constraints` 的逻辑但不调用 `_collect_raw_constraints`。如果 MuscleSim 的 `build_constraints` 有额外初始化（如 colored GS），需要检查并补全

### 调参优先级

如果第一次运行不稳定：
1. `attach_stiffness`: 1e8 → 1e6 → 1e10（过高可能振荡，过低耦合太松）
2. `num_substeps`: 10 → 20 → 30
3. `pin_stiffness`: 1e10 → 1e8
4. `fiber_stiffness_scale`: 200 → 100 → 500
