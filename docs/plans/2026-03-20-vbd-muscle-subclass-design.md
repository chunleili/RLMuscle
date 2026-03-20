# VBD Muscle 子类化设计方案

> **目标**：在不修改 Newton 源码的前提下，通过子类化 `ModelBuilder` 和 `SolverVBD`，实现基于 DeGroote-Fregly 2016 (DGF) Hill-type 肌肉模型的各向异性纤维能量。

> **前置文档**：
> - `docs/plans/2026-03-18-vbd-muscle-newton.md` — 原始侵入式方案（数学公式仍有效）
> - `D:\Dev\OpenSimExample\OpenSimExample\docs\plan_VBD_muscle.md` — 完整数学推导

---

## 1. 架构总览

```
用户代码
  │
  ├── MuscleModelBuilder(newton.ModelBuilder)    ← 子类化 Builder
  │     ├── add_muscle_tetrahedron()             ← 新方法：添加带纤维属性的四面体
  │     ├── add_tetrahedron() override           ← 保持 muscle lists 同步
  │     └── finalize() override                  ← 注入 MuscleData 到 model.muscle
  │
  ├── MuscleData                                 ← 独立数据容器
  │     ├── tet_fiber_dirs: wp.array(vec3)
  │     ├── tet_muscle_sigma0: wp.array(float)
  │     ├── tet_activations: wp.array(float)
  │     └── tet_fiber_lengths_prev: wp.array(float)   ← F-V lagged velocity
  │
  ├── SolverVBDMuscle(SolverVBD)                 ← 子类化 Solver
  │     ├── __init__()                           ← 初始化 muscle kernel 所需资源
  │     ├── _solve_particle_iteration() override ← 在颜色循环中插入 muscle kernel
  │     └── step() override                      ← 更新 fiber_lengths_prev
  │
  └── Warp Kernels (src/muscle_vbd/kernels.py)
        ├── @wp.func dgf_active_force_length()
        ├── @wp.func dgf_passive_force()
        ├── @wp.func dgf_force_velocity()
        ├── @wp.func fiber_energy_derivatives()
        ├── @wp.func evaluate_fiber_pk1_and_hessian()
        ├── @wp.func evaluate_fiber_force_and_hessian()
        └── @wp.kernel accumulate_fiber_force_and_hessian()
```

### 数据流

```
每个时间步:
  ┌─────────────────────────────────────────────────────────┐
  │  SolverVBDMuscle.step()                                 │
  │    ├── 计算 lagged fiber velocity (from prev stretch)   │
  │    ├── _solve_particle_iteration() × N iterations       │
  │    │     ├── zero particle_forces / particle_hessians     │
  │    │     └── for each color:                            │
  │    │     │     ├── accumulate contact forces  (Newton)  │
  │    │     │     ├── accumulate spring forces   (Newton)  │
  │    │     │     ├── accumulate self-contact    (Newton)  │
  │    │     │     ├── ★ accumulate_fiber_force   (ours)    │
  │    │     │     └── solve_elasticity           (Newton)  │
  │    │     │           reads particle_forces/hessians     │
  │    │     │           adds Neo-Hookean internally        │
  │    │     │           solves dx = H⁻¹f                   │
  │    │     └── _penetration_free_truncation     (Newton)  │
  │    └── 更新 fiber_lengths_prev                          │
  └─────────────────────────────────────────────────────────┘
```

---

## 2. 核心数学公式

### 2.1 纤维应变能

给定形变梯度 `F`，纤维方向 `d₀`（rest config 单位向量）：

```
I₄ = d₀ᵀ Fᵀ F d₀ = |F d₀|²     (纤维拉伸平方)
λ = √I₄                          (纤维拉伸比)
l̃ = λ                            (归一化纤维长度，网格在 l_opt 处建模)
```

### 2.2 纤维应力公式（含 F-V）

完整的纤维应力密度：

```
σ_fiber = σ₀ · [a · f_L(λ) · f_V(ṽ) + f_PE(λ)]
```

其中 `ṽ = (λ - λ_prev) / (dt · V_max)` 为归一化纤维速度（lagged，取上一时间步）。

对应的应变能密度一阶导：

```
Ψ'(I₄) = σ₀ / (2λ) · [a · f_L(λ) · f_V(ṽ) + f_PE(λ)]
```

二阶导（Hessian 用，f_V 作为常数处理，不对 I₄ 求导）：

```
Ψ''(I₄) = σ₀ / (4λ³) · [a · f_V(ṽ) · (f_L'·λ - f_L) + (f_PE'·λ - f_PE)]
```

> **注**：此公式与原始参考文档（`2026-03-18-vbd-muscle-newton.md` 第 35 行）的区别在于 `f_V` 因子。
> 原文档不含 F-V，此处扩展为含 lagged f_V 的版本。f_V(ṽ) 在对 I₄ 求导时视为常数
>（因为 ṽ 来自上一时间步），所以它作为标量乘子出现在主动项中。

### 2.3 DGF 曲线

**f_L** (Active Force-Length, 三高斯和):

```
f_L(l̃) = Σ(i=1..3) b1i · exp(-0.5 · ((l̃ - b2i) / (b3i + b4i·l̃))²)
Constants: b11=0.815, b21=1.055, b31=0.162, b41=0.063
           b12=0.433, b22=0.717, b32=-0.030, b42=0.200
           b13=0.1,   b23=1.0,   b33=0.354,  b43=0.0
```

**f_PE** (Passive Force, 指数):

```
offset = exp(kPE·(l_min - 1)/e0)
f_PE(l̃) = (exp(kPE·(l̃ - 1)/e0) - offset) / (exp(kPE) - offset)
kPE=4.0, e0=0.6, l_min=0.2
```

**f_V** (Force-Velocity, DGF sinh⁻¹ 形式):

```
f_V(ṽ) = d₁ · ln(d₂·ṽ + d₃ + √[(d₂·ṽ + d₃)² + 1]) + d₄
d₁ = -0.3211346127989808
d₂ = -8.149
d₃ = -0.374
d₄ = 0.8825327733249912
```

### 2.4 PK1 应力与 9×9 Hessian

```
A = d₀ ⊗ d₀                              (3×3 结构张量)
P_fiber = V_rest · 2·Ψ'(I₄)·F·A          (PK1 vec9, 含 rest volume V_rest = |det(Dm)|/6)
∂²W/∂F² = V_rest · [2·Ψ'(I₄)·kron(A, I₃) + 4·Ψ''(I₄)·outer(vec(FA), vec(FA))]  (9×9 Hessian)
```

> **注**：`V_rest` 在 `evaluate_fiber_pk1_and_hessian` 内从 `Dm_inv` 计算并乘入，与 Newton Neo-Hookean 的 `rest_volume` 处理方式一致。

### 2.5 逐顶点投影

对四面体的顶点 `v_order`（0-3），通过 `Dm_inv` 行向量 `m` 和 Newton 已有的 `assemble_tet_vertex_force_and_hessian(P_vec9, H99, m1, m2, m3)` 函数投影到 `(vec3 force, mat33 hessian)`。

### 2.6 F-V 半隐式策略

- **每个时间步开始时**：从上一步的最终位置计算所有肌肉四面体的纤维拉伸 `λ_prev`
- **在 VBD 迭代中**：`f_V` 使用 lagged velocity `ṽ = (λ_current_step_start - λ_prev_step) / (dt · V_max)`
- **每个时间步结束后**：更新 `λ_prev ← λ_current`
- f_V 在整个 VBD 迭代过程中保持**常数**（不随 GS 迭代更新），这保证了能量的一致性和收敛性

---

## 3. 依赖的 Newton 内部 API

以下 API 需要从 Newton 的 `_src` 私有模块获取。由于 `newton._src` 属于内部 API（AGENTS.md
明确禁止外部导入），我们采用**复制策略**：将以下约 30 行代码复制到 `kernels.py` 中，
消除运行时对 `_src` 的依赖。这些都是底层数据结构和纯函数，变化概率极低。

**复制到 `kernels.py` 的符号**（约 30 行）：

| 符号 | 类型 | 原始位置 | 行数 |
|------|------|----------|------|
| `vec9` | Warp vector type | `particle_vbd_kernels.py:74` | 2 行 |
| `mat99` | Warp matrix type | `particle_vbd_kernels.py:62` | 2 行 |
| `ParticleForceElementAdjacencyInfo` | `@wp.struct` | `particle_vbd_kernels.py:78-120` | ~15 行（仅 tet 相关字段） |
| `get_vertex_num_adjacent_tets` | `@wp.func` | `particle_vbd_kernels.py:156-157` | 3 行 |
| `get_vertex_adjacent_tet_id_order` | `@wp.func` | `particle_vbd_kernels.py:161-163` | 4 行 |
| `assemble_tet_vertex_force_and_hessian` | `@wp.func` | `particle_vbd_kernels.py:166-235` | ~70 行 |

**从 Newton 运行时导入的符号**（仅 Python 类，非 Warp JIT）：

| 符号 | 类型 | 用途 |
|------|------|------|
| `SolverVBD` | class (`solver_vbd.py`) | 基类，子类化 |
| `self.particle_adjacency` | `ParticleForceElementAdjacencyInfo` 实例 | 从 SolverVBD 继承，传给 kernel |

> **注意**：`self.particle_adjacency` 由 `SolverVBD.__init__` 构建并赋值。我们的子类通过 `self.particle_adjacency` 访问，
> 无需直接 import struct 定义——Warp kernel 通过复制的 struct 类型接收它。
> 需要验证 Warp 是否允许两个同结构但不同 Python 对象的 `@wp.struct` 互传；
> 如果不行，则改为从 `_src` 直接 import struct 定义（接受私有 API 依赖）。

从 `newton._src.solvers.vbd.solver_vbd` 导入：

| 符号 | 类型 | 用途 |
|------|------|------|
| `SolverVBD` | class | 基类，子类化 |

从 `newton` 公共 API 导入：

| 符号 | 用途 |
|------|------|
| `newton.ModelBuilder` | Builder 基类 |
| `newton.Model` | Model 类型 |

### 耦合风险评估

- **低风险**：`vec9`、`mat99`、`ParticleForceElementAdjacencyInfo`、`assemble_tet_vertex_force_and_hessian` — 基础数据结构，Newton 更新时几乎不会变
- **中风险**：`_solve_particle_iteration` 方法签名和内部结构 — 可能因新功能（如新的 contact 类型）而增加代码，但整体模式不变
- **缓解策略**：在 `SolverVBDMuscle.__init__` 中检查 Newton 版本，不兼容时抛出明确错误信息

---

## 4. 文件结构与职责

```
src/muscle_vbd/
  __init__.py              # 公共 API 导出
  muscle_data.py           # MuscleData dataclass
  muscle_builder.py        # MuscleModelBuilder(newton.ModelBuilder)
  kernels.py               # 所有 Warp @wp.func 和 @wp.kernel
  solver.py                # SolverVBDMuscle(SolverVBD)
  activation.py            # DGF 激活动力学 ODE (excitation → activation)
  mesh_utils.py            # 圆柱 tet mesh 生成 + 纤维方向赋值工具
```

### 4.1 muscle_data.py

```python
@dataclass
class MuscleData:
    tet_fiber_dirs: wp.array       # (n_tets,) vec3, rest config unit vectors
    tet_muscle_sigma0: wp.array    # (n_tets,) float, peak isometric stress [Pa]
    tet_activations: wp.array      # (n_tets,) float, activation level [0,1]
    tet_fiber_lengths_prev: wp.array  # (n_tets,) float, λ from previous timestep
    max_contraction_velocity: float = 10.0  # V_max [l_opt/s], default from OpenSim
```

### 4.2 muscle_builder.py

```python
class MuscleModelBuilder(newton.ModelBuilder):
    """Builder that tracks per-tet muscle fiber properties alongside Newton's standard tet data."""

    def __init__(self):
        super().__init__()
        self._tet_fiber_dirs = []
        self._tet_muscle_sigma0 = []

    def add_tetrahedron(self, i, j, k, l, **kwargs) -> float:
        """Override: keep muscle lists in sync. Non-muscle tets get zero fiber."""
        prev_count = len(self.tet_indices)
        vol = super().add_tetrahedron(i, j, k, l, **kwargs)
        # Only append if tet was actually added (volume > 0)
        if len(self.tet_indices) > prev_count:
            if len(self._tet_fiber_dirs) < len(self.tet_indices):
                self._tet_fiber_dirs.append((0.0, 0.0, 0.0))
                self._tet_muscle_sigma0.append(0.0)
        return vol

    def add_muscle_tetrahedron(self, i, j, k, l,
                                k_mu, k_lambda, k_damp=0.0,
                                fiber_dir=(0,0,0), muscle_sigma0=0.0) -> float:
        """Add a tet with muscle fiber properties.
        Calls Newton's base add_tetrahedron directly (bypassing our override)
        to avoid double-appending to muscle lists.
        """
        prev_count = len(self.tet_indices)
        vol = newton.ModelBuilder.add_tetrahedron(
            self, i, j, k, l, k_mu=k_mu, k_lambda=k_lambda, k_damp=k_damp
        )
        if len(self.tet_indices) > prev_count:
            self._tet_fiber_dirs.append(tuple(fiber_dir))
            self._tet_muscle_sigma0.append(float(muscle_sigma0))
        return vol

    def finalize(self, **kwargs):
        model = super().finalize(**kwargs)
        device = model.tet_indices.device if model.tet_indices is not None else "cpu"
        n_tets = len(self._tet_fiber_dirs)
        model.muscle = MuscleData(
            tet_fiber_dirs=wp.array(self._tet_fiber_dirs, dtype=wp.vec3, device=device),
            tet_muscle_sigma0=wp.array(self._tet_muscle_sigma0, dtype=wp.float32, device=device),
            tet_activations=wp.zeros(n_tets, dtype=wp.float32, device=device),
            tet_fiber_lengths_prev=wp.ones(n_tets, dtype=wp.float32, device=device),  # λ=1 at rest
        )
        return model
```

### 4.3 kernels.py

包含所有 `@wp.func` 和 `@wp.kernel`：

**@wp.func 函数**（约 200 行）：
- `dgf_active_force_length(lm_tilde) → float`
- `dgf_active_force_length_deriv(lm_tilde) → float`
- `dgf_passive_force(lm_tilde) → float`
- `dgf_passive_force_deriv(lm_tilde) → float`
- `dgf_force_velocity(v_norm) → float` — DGF sinh⁻¹ 公式
- `fiber_energy_derivatives(I4, activation, sigma0, fv_factor) → vec2` — 返回 (Ψ', Ψ'')
- `evaluate_fiber_pk1_and_hessian(F, fiber_dir, activation, sigma0, rest_volume, fv_factor) → (vec9, mat99)`
- `evaluate_fiber_force_and_hessian(tet_id, v_order, pos_prev, pos, tet_indices, Dm_inv, fiber_dir, sigma0, activation, fv_factor, damping, dt) → (vec3, mat33)`

**@wp.kernel**（约 60 行）：

```python
@wp.kernel
def accumulate_fiber_force_and_hessian(
    dt: float,
    particle_ids_in_color: wp.array(dtype=wp.int32),
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    tet_indices: wp.array(dtype=wp.int32, ndim=2),
    tet_poses: wp.array(dtype=wp.mat33),
    tet_materials: wp.array(dtype=float, ndim=2),  # for damping column
    tet_fiber_dirs: wp.array(dtype=wp.vec3),
    tet_muscle_sigma0: wp.array(dtype=float),
    tet_activations: wp.array(dtype=float),
    tet_fv_factors: wp.array(dtype=float),          # precomputed f_V per tet
    particle_adjacency: ParticleForceElementAdjacencyInfo,
    # output (accumulated, not overwritten)
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
):
    tid = wp.tid()
    particle_index = particle_ids_in_color[tid]

    f = wp.vec3(0.0, 0.0, 0.0)
    h = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    num_adj_tets = get_vertex_num_adjacent_tets(particle_adjacency, particle_index)
    for adj in range(num_adj_tets):
        tet_id, v_order = get_vertex_adjacent_tet_id_order(
            particle_adjacency, particle_index, adj
        )
        if tet_muscle_sigma0[tet_id] > 0.0:
            f_fiber, h_fiber = evaluate_fiber_force_and_hessian(
                tet_id, v_order,
                pos_prev, pos,
                tet_indices, tet_poses[tet_id],
                tet_fiber_dirs[tet_id],
                tet_muscle_sigma0[tet_id],
                tet_activations[tet_id],
                tet_fv_factors[tet_id],
                tet_materials[tet_id, 2],  # damping
                dt,
            )
            f = f + f_fiber
            h = h + h_fiber

    particle_forces[particle_index] = particle_forces[particle_index] + f
    particle_hessians[particle_index] = particle_hessians[particle_index] + h
```

**@wp.kernel 辅助**：

```python
@wp.kernel
def compute_fiber_lengths(
    pos: wp.array(dtype=wp.vec3),
    tet_indices: wp.array(dtype=wp.int32, ndim=2),
    tet_poses: wp.array(dtype=wp.mat33),
    tet_fiber_dirs: wp.array(dtype=wp.vec3),
    tet_muscle_sigma0: wp.array(dtype=float),
    # output
    tet_fiber_lengths: wp.array(dtype=float),
):
    """Compute current fiber stretch λ = |F d₀| for each muscle tet."""
    tid = wp.tid()
    if tet_muscle_sigma0[tid] <= 0.0:
        return
    v0 = pos[tet_indices[tid, 0]]
    v1 = pos[tet_indices[tid, 1]]
    v2 = pos[tet_indices[tid, 2]]
    v3 = pos[tet_indices[tid, 3]]
    Ds = wp.matrix_from_cols(v1 - v0, v2 - v0, v3 - v0)
    F = Ds * tet_poses[tid]
    Fd0 = F * tet_fiber_dirs[tid]
    tet_fiber_lengths[tid] = wp.sqrt(wp.dot(Fd0, Fd0))


@wp.kernel
def compute_fv_factors(
    tet_fiber_lengths_current: wp.array(dtype=float),
    tet_fiber_lengths_prev: wp.array(dtype=float),
    tet_muscle_sigma0: wp.array(dtype=float),
    dt: float,
    v_max: float,
    # output
    tet_fv_factors: wp.array(dtype=float),
):
    """Compute f_V factor from lagged fiber velocity for each muscle tet."""
    tid = wp.tid()
    if tet_muscle_sigma0[tid] <= 0.0:
        tet_fv_factors[tid] = 1.0
        return
    lam = tet_fiber_lengths_current[tid]
    lam_prev = tet_fiber_lengths_prev[tid]
    v_norm = (lam - lam_prev) / (dt * v_max)
    v_norm = wp.clamp(v_norm, -1.0, 1.0)
    tet_fv_factors[tid] = dgf_force_velocity(v_norm)
```

### 4.4 solver.py

```python
class SolverVBDMuscle(SolverVBD):
    """VBD solver with DeGroote-Fregly Hill-type muscle fiber energy."""

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        muscle = model.muscle
        n_tets = muscle.tet_fiber_dirs.shape[0]
        self._tet_fv_factors = wp.ones(n_tets, dtype=float, device=self.device)
        self._tet_fiber_lengths_current = wp.ones(n_tets, dtype=float, device=self.device)

    def step(self, state_in, state_out, control=None, contacts=None, dt=1.0/60.0):
        muscle = self.model.muscle

        # 1. Compute current fiber lengths from state_in positions
        wp.launch(compute_fiber_lengths, dim=muscle.tet_fiber_dirs.shape[0],
                  inputs=[state_in.particle_q, self.model.tet_indices,
                          self.model.tet_poses, muscle.tet_fiber_dirs,
                          muscle.tet_muscle_sigma0],
                  outputs=[self._tet_fiber_lengths_current],
                  device=self.device)

        # 2. Compute f_V factors from lagged velocity
        wp.launch(compute_fv_factors, dim=muscle.tet_fiber_dirs.shape[0],
                  inputs=[self._tet_fiber_lengths_current,
                          muscle.tet_fiber_lengths_prev,
                          muscle.tet_muscle_sigma0,
                          dt, muscle.max_contraction_velocity],
                  outputs=[self._tet_fv_factors],
                  device=self.device)

        # 3. Resolve activations from control if available
        if control is not None and hasattr(control, 'tet_activations') and control.tet_activations is not None:
            wp.copy(muscle.tet_activations, control.tet_activations)

        # 4. Call parent step (which calls our overridden _solve_particle_iteration)
        super().step(state_in, state_out, control, contacts, dt)

        # 5. Update prev fiber lengths for next timestep
        wp.copy(muscle.tet_fiber_lengths_prev, self._tet_fiber_lengths_current)

    def _solve_particle_iteration(self, state_in, state_out, contacts, dt, iter_num):
        """Override: insert muscle force accumulation kernel before solve_elasticity.

        复制自 SolverVBD._solve_particle_iteration (~150 行)，
        唯一修改：在每个 color 的 solve_elasticity launch 之前插入 muscle kernel。

        ⚠ 重要：必须在 BOTH 分支（tiled 和 non-tiled）之前都插入 muscle kernel。
        """
        # ... (collision detection, zero forces/hessians — 与父类相同) ...

        for color in range(len(self.model.particle_color_groups)):
            # ... (contact / spring / self-contact accumulation — 与父类相同) ...

            # ★ 新增: launch accumulate_fiber_force_and_hessian (在两个分支之前)
            if hasattr(self.model, 'muscle') and self.model.muscle is not None:
                wp.launch(
                    kernel=accumulate_fiber_force_and_hessian,
                    dim=self.model.particle_color_groups[color].size,
                    inputs=[dt, self.model.particle_color_groups[color],
                            self.particle_q_prev, state_in.particle_q,
                            self.model.tet_indices, self.model.tet_poses,
                            self.model.tet_materials,
                            self.model.muscle.tet_fiber_dirs,
                            self.model.muscle.tet_muscle_sigma0,
                            self.model.muscle.tet_activations,
                            self._tet_fv_factors,
                            self.particle_adjacency],
                    outputs=[self.particle_forces, self.particle_hessians],
                    device=self.device)

            # tiled / non-tiled 分支 — 与父类完全相同
            if self.use_particle_tile_solve:
                wp.launch(kernel=solve_elasticity_tile, ...)  # 与父类相同
            else:
                wp.launch(kernel=solve_elasticity, ...)       # 与父类相同

            self._penetration_free_truncation(state_in.particle_q)

        wp.copy(state_out.particle_q, state_in.particle_q)
```

### 4.5 activation.py

DGF 激活动力学（excitation → activation 的一阶 ODE）：

```python
def activation_dynamics_step(excitation: np.ndarray, activation: np.ndarray,
                              dt: float, tau_act=0.015, tau_deact=0.060) -> np.ndarray:
    """First-order activation dynamics: da/dt = (u - a) / tau(u, a)
    tau = tau_act * (0.5 + 1.5*a) when u > a, else tau_deact / (0.5 + 1.5*a)
    """
    ...
```

此模块在 CPU 上运行（激活值通常由 RL policy 提供），每步调用一次后写入 `muscle.tet_activations`。

### 4.6 mesh_utils.py

- `create_cylinder_tet_mesh(length, radius, n_length, n_radial)` — 生成圆柱体四面体网格
- `assign_fiber_directions(vertices, tets, axis)` — 沿指定轴为每个四面体赋纤维方向
- `add_muscle_to_builder(builder, vertices, tets, fiber_dirs, mu, lam, sigma0)` — 批量添加肌肉四面体到 builder

---

## 5. SPD 策略

使用 **clamp Ψ' ≥ 0** 策略（方案 A）：

```python
dPsi = wp.max(dPsi, 0.0)  # ensure PSD Hessian
```

在正常拉伸区（λ ∈ [0.5, 2.0]），kron(A, I₃) 和 outer 项均半正定。极端压缩区（λ < 0.5）时，Newton 已有的 `det(h) > 1e-8` 检查会自动跳过该顶点更新。

如果后续遇到收敛问题，可升级到 eigenvalue clamping（方案 B）。

---

## 6. 参数映射

| Hill 参数 | FEM 参数 | 映射关系 |
|---|---|---|
| `max_isometric_force` (F₀) | `sigma0` | `σ₀ = F₀ / PCSA` |
| `optimal_fiber_length` (l_opt) | 网格几何 | 网格在 l_opt 处建模 |
| `pennation_angle` (α₀) | `fiber_dir` | 每单元纤维方向向量 |
| `max_contraction_velocity` (V_max) | `MuscleData.max_contraction_velocity` | 默认 10 l_opt/s |
| `fiber_damping` (β) | `tet_materials[:, 2]` | 复用 Newton 的 k_damp |

---

## 7. 测试计划

### 7.1 单元测试

- **DGF 曲线验证**：f_L(1.0) ≈ 1.0, f_PE(1.0) ≈ 0, f_V(0) ≈ 1.0
- **导数有限差分**：f_L', f_PE', f_V' 与 FD 误差 < 1e-4
- **纤维梯度 FD**：单四面体纤维力 vs 能量有限差分，误差 < 1e-4
- **f_V 乘子正确性**：f_V=1 时退化为纯 F-L 模型

### 7.2 集成测试

- **无肌肉回归**：所有 sigma0=0 时，行为与原 SolverVBD 完全一致
- **单肌肉收缩**：activation=1.0 → 自由端产生位移
- **F-L 配准**：固定两端扫描 λ ∈ [0.5, 1.5]，比较稳态力 vs OpenSim DGF 曲线，RMSE < 5%
- **F-V 配准**：固定 λ=1.0，施加不同速度，比较力 vs DGF f_V 曲线，|ṽ| < 0.5 误差 < 10%

### 7.3 可视化测试

- 圆柱肌肉收缩动画 → 截图到 `output/` 对比

---

## 8. 风险与缓解

| 风险 | 缓解 |
|------|------|
| Newton 更新改变 `_solve_particle_iteration` 结构 | 在 `__init__` 中版本检查；override 方法仅复制 ~150 行，diff 明确 |
| 私有 API (`_src`) 变更 | 复制策略：关键 `@wp.func` 复制到本地 kernels.py，消除运行时 `_src` 依赖 |
| `model.muscle = MuscleData(...)` 动态属性注入 | Python `Model` 类使用 `__dict__`（无 `__slots__`），动态属性可行。实现时需验证 |
| F-V lagged velocity 在大 dt 时不准 | clamp v_norm 到 [-1, 1]；dt > 0.01s 时可能需要子步 |
| 纤维 Hessian 不定导致不收敛 | Ψ' clamp ≥ 0 + Newton 的 det 检查双重保护 |
| F 计算两次（muscle kernel + elasticity kernel） | GPU 上 3×3 矩阵乘法开销可忽略 |
| `tet_materials` ndim 假设 | 实现前验证 `model.tet_materials.ndim == 2`，如为 1D 需调整索引 |

---

## 9. 未来扩展

- **requires_grad 支持**：当前 `MuscleData` 数组不启用梯度。未来 RL 需要通过肌肉能量反传梯度时，`finalize()` 中需传入 `requires_grad` 参数并透传到 `wp.array` 创建。
- **Eigenvalue clamping (SPD 方案 B)**：如果收敛问题严重，实现 3×3 Jacobi SVD + 特征值截断。
- **非线性 F-V Hessian**：当前 f_V 在 Hessian 中作为常数。如需更高精度，可在 VBD 迭代内更新 f_V（代价：每次迭代多一次 kernel launch）。
