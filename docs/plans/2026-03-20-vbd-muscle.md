# VBD Muscle 设计方案（直接修改 Newton）

> **目标**：在 Newton 的 `feat/vbd-muscle` 分支上，直接修改 VBD 求解器和 ModelBuilder，实现基于 DeGroote-Fregly 2016 (DGF) Hill-type 肌肉模型的各向异性纤维能量。

> **前置知识文档**：
> - `D:\Dev\OpenSimExample\OpenSimExample\docs\plan_VBD_muscle.md` — 完整数学推导
> - `D:\Dev\OpenSimExample\OpenSimExample\learn_opensim.md:236-310` — DGF 曲线公式常数
> - `D:\Dev\OpenSimExample\OpenSimExample\vbd_energy.cl` — Houdini VBD 纤维实现参考
> - `D:\dev\RLMuscle-dev\docs\experiments\vbd_code_analysis.md` — Newton VBD 代码分析
> - `D:\Dev\OpenSimExample\OpenSimExample\vbd_muscle\vbd_muscle.py` — 现有基于 warp 的 VBD 肌肉实现（不直接复用，但设计思路借鉴）

参考文献：
> - Smith et al., "Stable Neo-Hookean Flesh Simulation."
> - Kim et al., "Anisotropic Elasticity for Inversion-Safety and Element Rehabilitation."

---

## 1. 架构总览

### 修改策略

1. **在 Newton 子模块创建 `feat/vbd-muscle` 分支**，主项目追踪该分支，主项目使用同名分支。
2. **直接修改 `vbd/` 目录**，不新建 `vbd_muscle/`
3. **尽量新增文件**，必要时才在源文件插入最小改动
4. **kernel launch 与 warp kernel/func 分离**，保持 Newton 现有风格
5. **直接修改 ModelBuilder**，新增字段以 `vmuscle_` 为前缀（与现有 `muscle_*` 区分）
6. 代码符合 Newton 风格（Google docstring、前缀命名、SI 单位）

### 设计原则：Example / Model(State) / Solver 三层分离

| 层 | 职责 | 不应包含 |
|---|---|---|
| **Example** | 算例参数、网格生成、边界条件、程序入口 | 求解算法、数据结构定义 |
| **Model / State** | 数据容器（vmuscle_* 字段）、Builder（构建数据） | 求解逻辑、算例特定参数 |
| **Solver** | VBD 迭代、kernel launch、时间积分 | 网格生成、算例配置 |

### 架构图

```
examples/muscle_sliding_ball.py                ← Example 层（算例入口）
  │  设置网格、材料参数、边界条件、activation 序列
  │  调用 ModelBuilder 构建 Model，创建 SolverVBD，驱动仿真循环
  │
  ▼
external/newton/newton/_src/                  ← Model + Solver 层（Newton 内部修改）
  │
  ├── sim/builder.py [修改]                   ← Model 层：ModelBuilder 直接添加字段
  │     ├── __init__(): 新增 vmuscle_* 列表
  │     ├── (vmuscle_* 列表由外部 set_vmuscle_properties() 填充)
  │     └── finalize(): 新增 vmuscle_* 数组写入 Model
  │
  ├── solvers/vbd/                            ← Solver 层（VBD 目录，尽量新增文件）
  │     ├── solver_vbd.py [最小修改]
  │     │     └── _solve_particle_iteration(): 插入 muscle kernel launch
  │     │
  │     ├── vmuscle_kernels.py [新增]         ← Warp @wp.func 和 @wp.kernel
  │     │     ├── @wp.func dgf_active_force_length()
  │     │     ├── @wp.func dgf_passive_force()
  │     │     ├── @wp.func dgf_force_velocity()
  │     │     ├── @wp.func fiber_energy_derivatives()
  │     │     ├── @wp.func evaluate_fiber_pk1_and_hessian()
  │     │     ├── @wp.func evaluate_fiber_force_and_hessian()
  │     │     └── @wp.kernel accumulate_fiber_force_and_hessian()
  │     │
  │     └── vmuscle_launch.py [新增]          ← kernel launch 封装函数
  │           └── launch_accumulate_fiber_force_and_hessian()
  │
  └── (其他 Newton 文件不修改)

src/VMuscle/                                  ← Example 层辅助（主项目，不在 Newton 内）
  ├── config.py          # load_config() — 已有，复用
  └── mesh_utils.py      # add_soft_vmuscle_mesh() — 批量导入肌肉网格

examples/                                     ← Example 层（算例入口）
  ├── muscle_sliding_ball.py  # 算例入口
  └── (mesh_utils.py 已移至 src/VMuscle/)
```

### 数据流

```
每个时间步:
  ┌─────────────────────────────────────────────────────────┐
  │  SolverVBD.step()                                       │
  │    ├── particle_q_prev2 ← particle_q_prev (wp.copy)    │
  │    ├── _solve_particle_iteration() × N iterations       │
  │    │     ├── zero particle_forces / particle_hessians   │
  │    │     └── for each color:                            │
  │    │     │     ├── accumulate contact forces  (Newton)  │
  │    │     │     ├── accumulate spring forces   (Newton)  │
  │    │     │     ├── accumulate self-contact    (Newton)  │
  │    │     │     ├── ★ accumulate_fiber_force   (ours)    │
  │    │     │     │     从 particle_q_prev / prev2         │
  │    │     │     │     内联算 l̃, ṽ, f_V, force, hessian   │
  │    │     │     └── solve_elasticity           (Newton)  │
  │    │     │           reads particle_forces/hessians     │
  │    │     │           adds Neo-Hookean internally        │
  │    │     │           solves dx = H⁻¹f                   │
  │    │     └── _penetration_free_truncation     (Newton)  │
  └─────────────────────────────────────────────────────────┘
```

### USD 网格加载数据流

```
┌─ USD 文件 ──────────────────────────────────────────┐
│  TetMesh prim                                        │
│    points, tetVertexIndices, physics material         │
│    primvars: materialW, muscle_id, tendonmask         │
└──────────────────────────────────────────────────────┘
          │
          ▼
   newton.usd.get_tetmesh(prim)
   → TetMesh(vertices, tet_indices, custom_attributes={
       "materialW":   (array, PARTICLE),
       "muscle_id":   (array, PARTICLE),
       "tendonmask":  (array, PARTICLE),
     })
          │                    ┌─ JSON config ──────────┐
          │                    │  tendon_threshold: 0.7  │
          │                    │  muscles: {0: {sigma0}} │
          ▼                    └────────────┬───────────┘
   add_soft_vmuscle_mesh(builder, mesh, cfg)
     ├── builder.add_soft_mesh(mesh)    ← particles + tets + Neo-Hookean
     ├── materialW → fiber_dir 映射
     ├── per-vertex → per-tet 转换
     ├── tendonmask → sigma0=0 (tendon)
     ├── muscle_configs[id] → sigma0 (belly)
     └── 写入 builder.vmuscle_* 列表
          │
          ▼
   builder.finalize() → Model (含 vmuscle_* warp arrays)
```

---

## 2. 核心数学公式

### 2.1 纤维应变能

给定形变梯度 `F`，纤维参考方向 `d`（rest config 单位向量）：

```
I₅ = dᵀ Fᵀ F d = |F d|²         (纤维拉伸平方，Houdini/VBD 约定)
l̃ = √I₅                          (归一化纤维长度 = 纤维拉伸比，网格在 l_opt 处建模)

注：本文采用 Houdini VBD / Kim et al. 的不变量编号约定：
  I₅ = dᵀCd = dᵀFᵀFd（纤维拉伸平方，能量的主不变量）
  I₄ = dᵀSd（极分解有符号拉伸，仅用于翻转检测，本文不使用）
  参考：Smith et al., "Stable Neo-Hookean Flesh Simulation"
        Kim et al., "Anisotropic Elasticity for Inversion-Safety and Element Rehabilitation"
```

### 2.2 纤维应力公式（含 F-V）

Hill 力公式（标量，= dΨ/dl̃，即能量对拉伸比的导数，参见 Dexterous 公式 7）：

```
f_hill(l̃, ṽ, a) = σ₀ · [a · f_L(l̃) · f_V(ṽ) + f_PE(l̃)]
```

其中 `ṽ = (l̃ - l̃_prev) / (dt · V_max)` 为归一化纤维速度（lagged，取上一时间步）。

对应的应变能密度对 I₅ 的一阶导（链式法则 I₅ = l̃² → Ψ'(I₅) = f_hill / (2l̃)）：

```
Ψ'(I₅) = f_hill / (2l̃) = σ₀ / (2l̃) · [a · f_L(l̃) · f_V(ṽ) + f_PE(l̃)]
```

二阶导（Hessian 用，f_V 作为常数处理，不对 I₅ 求导）：

```
Ψ''(I₅) = σ₀ / (4l̃³) · [a · f_V(ṽ) · (f_L'·l̃ - f_L) + (f_PE'·l̃ - f_PE)]
```

> **注**：f_V(ṽ) 在对 I₅ 求导时视为常数（因为 ṽ 来自上一时间步），所以它作为标量乘子出现在主动项中。

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
A = d ⊗ d                              (3×3 结构张量)
P_fiber = V_rest · 2·Ψ'(I₅)·F·A          (PK1 vec9, 含 rest volume V_rest = |det(Dm)|/6)
∂²W/∂F² = V_rest · [2·Ψ'(I₅)·kron(A, I₃) + 4·Ψ''(I₅)·outer(vec(FA), vec(FA))]  (9×9 Hessian)
```

> **注**：`V_rest` 在 `evaluate_fiber_pk1_and_hessian` 内从 `Dm_inv` 计算并乘入，与 Newton Neo-Hookean 的 `rest_volume` 处理方式一致。

### 2.5 逐顶点力与 Hessian 组装

对四面体的顶点 `v_order`（0-3），通过 `Dm_inv` 行向量 `m` 和 Newton 已有的 `assemble_tet_vertex_force_and_hessian(P_vec9, H99, m1, m2, m3)` 函数，从单元 PK1 (vec9) 和 9×9 Hessian 组装出该顶点的 `(vec3 force, mat33 hessian)`。

> **注**：由于直接在 Newton 内部修改，可以直接 import `particle_vbd_kernels.py` 中的 `assemble_tet_vertex_force_and_hessian`、`ParticleForceElementAdjacencyInfo`、`vec9`、`mat99` 等符号，无需复制。

### 2.6 F-V lagged velocity 策略

使用 `particle_q_prev2`（两步前的位置）直接在 accumulate kernel 内计算 lagged fiber velocity，无需额外的预计算步骤或存储。

- **Solver 维护 `particle_q_prev2`**：每个时间步开始时，`particle_q_prev2 ← particle_q_prev`（一次 `wp.copy`）
- **在 accumulate kernel 内**：从 `particle_q_prev`（上一步位置）和 `particle_q_prev2`（两步前位置）分别计算纤维长度，内联算出 ṽ 和 f_V
- **f_V 在 VBD 迭代中保持常数**：因为 `particle_q_prev` 和 `particle_q_prev2` 在当前时间步内都不变，保证能量一致性和收敛性

```
在 accumulate kernel 内（per tet, 内联）:
  F_prev  = Ds(particle_q_prev) * Dm_inv   →  l̃_current = |F_prev · d|
  F_prev2 = Ds(particle_q_prev2) * Dm_inv  →  l̃_prev    = |F_prev2 · d|
  ṽ = (l̃_current - l̃_prev) / (dt · V_max)
  fv = dgf_force_velocity(ṽ)
```

> **优势**：不需要 `vmuscle_tet_fiber_lengths_prev` 数组、不需要 `compute_fiber_lengths` / `compute_fv_factors` 预计算 kernel，数据流更简洁。代价是每个 tet 多算一次 F（从 prev2 位置），但 GPU 上 3×3 矩阵乘法开销可忽略。

---

## 3. Newton 修改清单

### 3.1 分支管理

```bash
cd external/newton
git checkout -b feat/vbd-muscle
# 主项目 .gitmodules 追踪此分支
```

### 3.2 修改文件列表

| 文件 | 修改类型 | 修改内容 |
|------|---------|---------|
| `_src/sim/builder.py` | 插入 | `__init__` 新增 `vmuscle_*` 列表；`finalize()` 写入数组（padding 到 tet_count） |
| `_src/solvers/vbd/solver_vbd.py` | 插入 | `__init__` 初始化 `particle_q_prev2`；`step()` 中更新 `prev2 ← prev`；`_solve_particle_iteration()` 中插入 muscle kernel launch |
| `_src/solvers/vbd/__init__.py` | 插入 | 导出新模块（如需要） |

### 3.3 新增文件列表

| 文件 | 职责 |
|------|------|
| `_src/solvers/vbd/vmuscle_kernels.py` | DGF 曲线 `@wp.func` + accumulate/compute `@wp.kernel` |
| `_src/solvers/vbd/vmuscle_launch.py` | kernel launch 封装函数（供 solver_vbd.py 调用） |

---

## 4. 各文件详细设计

### 4.1 builder.py 修改（ModelBuilder）

在 `__init__` 中新增：

```python
# volumetric muscle (vmuscle) — per-tet fiber properties for VBD Hill-type muscle
self.vmuscle_tet_ids = []             # indices into tet arrays for vmuscle tets
self.vmuscle_tet_fiber_dirs = []      # vec3, rest config unit fiber direction
self.vmuscle_tet_sigma0 = []          # float, peak isometric stress [Pa]
self.vmuscle_max_contraction_velocity = 10.0  # V_max [l_opt/s], DGF default
```
> 直接采用newton中自带的tet_activation而不是自定义vmuscle_tet_activations

> **不新增 `add_vmuscle_tetrahedron()` 方法**。实际工作流始终是先调用 `add_soft_mesh()`
> 添加完整网格（粒子、质量、四面体、表面三角形），然后通过外部 helper 函数
> `set_vmuscle_properties()` 批量填充 vmuscle 数据。`add_vmuscle_tetrahedron()` 内部
> 调用 `add_tetrahedron()` 会导致重复添加四面体，与 `add_soft_mesh()` 工作流冲突，
> 因此不提供此方法。

在 `finalize()` 末尾新增（padding 到 `tet_count`，非肌肉四面体填 zero）：

```python
# volumetric muscle arrays (padded to tet_count, non-muscle tets get zero)
tet_count = len(self.tet_indices)
if self.vmuscle_tet_ids:
    fiber_dirs = [(0.0, 0.0, 0.0)] * tet_count
    sigma0_arr = [0.0] * tet_count
    activations_arr = [0.0] * tet_count
    for idx, tet_id in enumerate(self.vmuscle_tet_ids):
        fiber_dirs[tet_id] = self.vmuscle_tet_fiber_dirs[idx]
        sigma0_arr[tet_id] = self.vmuscle_tet_sigma0[idx]
        activations_arr[tet_id] = self.vmuscle_tet_activations[idx]
    m.vmuscle_tet_fiber_dirs = wp.array(fiber_dirs, dtype=wp.vec3, ...)
    m.vmuscle_tet_sigma0 = wp.array(sigma0_arr, dtype=wp.float32, ...)
    m.vmuscle_tet_activations = wp.array(activations_arr, dtype=wp.float32, ...)
    m.vmuscle_max_contraction_velocity = self.vmuscle_max_contraction_velocity
    m.vmuscle_count = len(self.vmuscle_tet_ids)
else:
    m.vmuscle_count = 0
```

### 4.2 vmuscle_kernels.py（新增文件）

所有 `@wp.func` 和 `@wp.kernel`，集中在一个文件中便于修改。

**@wp.func 函数**（约 200 行）：

```python
from .particle_vbd_kernels import (
    vec9, mat99,
    ParticleForceElementAdjacencyInfo,
    get_vertex_num_adjacent_tets,
    get_vertex_adjacent_tet_id_order,
    assemble_tet_vertex_force_and_hessian,
)

@wp.func
def dgf_active_force_length(lm_tilde: float) -> float:
    """DGF 2016 active force-length curve (sum of 3 Gaussians)."""
    # Gaussian 1: b1=0.815, b2=1.055, b3=0.162, b4=0.063
    denom1 = 0.162 + 0.063 * lm_tilde
    t1 = (lm_tilde - 1.055) / denom1
    g1 = 0.815 * wp.exp(-0.5 * t1 * t1)
    # Gaussian 2: b1=0.433, b2=0.717, b3=-0.030, b4=0.200
    denom2 = -0.030 + 0.200 * lm_tilde
    t2 = (lm_tilde - 0.717) / denom2
    g2 = 0.433 * wp.exp(-0.5 * t2 * t2)
    # Gaussian 3: b1=0.1, b2=1.0, b3=0.354, b4=0.0
    t3 = (lm_tilde - 1.0) / 0.354
    g3 = 0.1 * wp.exp(-0.5 * t3 * t3)
    return g1 + g2 + g3

@wp.func
def dgf_active_force_length_deriv(lm_tilde: float) -> float:
    """Derivative of f_L w.r.t. l̃.
    d/dl [b1*exp(-0.5*t^2)] where t = (l-b2)/(b3+b4*l)
    dt/dl = [b3 + b4*b2] / (b3+b4*l)^2
    d/dl = b1 * exp(-0.5*t^2) * (-t) * dt/dl
    """
    result = float(0.0)
    # Gaussian 1
    d1 = 0.162 + 0.063 * lm_tilde
    t1 = (lm_tilde - 1.055) / d1
    dtdl1 = (0.162 + 0.063 * 1.055) / (d1 * d1)
    result += 0.815 * wp.exp(-0.5 * t1 * t1) * (-t1) * dtdl1
    # Gaussian 2
    d2 = -0.030 + 0.200 * lm_tilde
    t2 = (lm_tilde - 0.717) / d2
    dtdl2 = (-0.030 + 0.200 * 0.717) / (d2 * d2)
    result += 0.433 * wp.exp(-0.5 * t2 * t2) * (-t2) * dtdl2
    # Gaussian 3
    d3 = 0.354
    t3 = (lm_tilde - 1.0) / d3
    dtdl3 = 1.0 / d3
    result += 0.1 * wp.exp(-0.5 * t3 * t3) * (-t3) * dtdl3
    return result

@wp.func
def dgf_passive_force(lm_tilde: float) -> float:
    """DGF 2016 passive force-length curve (exponential)."""
    kPE = 4.0
    e0 = 0.6
    lm_min = 0.2
    offset = wp.exp(kPE * (lm_min - 1.0) / e0)
    denom = wp.exp(kPE) - offset
    result = (wp.exp(kPE * (lm_tilde - 1.0) / e0) - offset) / denom
    return wp.max(result, 0.0)

@wp.func
def dgf_passive_force_deriv(lm_tilde: float) -> float:
    """Derivative of f_PE w.r.t. l̃."""
    kPE = 4.0
    e0 = 0.6
    lm_min = 0.2
    offset = wp.exp(kPE * (lm_min - 1.0) / e0)
    denom = wp.exp(kPE) - offset
    return (kPE / e0) * wp.exp(kPE * (lm_tilde - 1.0) / e0) / denom

@wp.func
def dgf_force_velocity(v_norm: float) -> float:
    """DGF 2016 force-velocity curve (sinh⁻¹ form)."""
    d1 = -0.3211346127989808
    d2 = -8.149
    d3 = -0.374
    d4 = 0.8825327733249912
    x = d2 * v_norm + d3
    return d1 * wp.log(x + wp.sqrt(x * x + 1.0)) + d4

@wp.func
def fiber_energy_derivatives(I5: float, activation: float, sigma0: float, fv_factor: float):
    """Compute Ψ'(I₅) and Ψ''(I₅) for Hill-type fiber energy.
    Returns: (dPsi_dI5, d2Psi_dI5_2)
    """
    lam = wp.sqrt(wp.max(I5, 1.0e-8))
    lm_tilde = lam

    fL = dgf_active_force_length(lm_tilde)
    fPE = dgf_passive_force(lm_tilde)
    dfL = dgf_active_force_length_deriv(lm_tilde)
    dfPE = dgf_passive_force_deriv(lm_tilde)

    # Ψ'(I₅) = σ₀/(2λ) · [a·f_L(λ)·f_V(ṽ) + f_PE(λ)]
    total_force = activation * fL * fv_factor + fPE
    dPsi = sigma0 / (2.0 * lam) * total_force
    dPsi = wp.max(dPsi, 0.0)  # clamp to ensure PSD Hessian

    # Ψ''(I₅) = σ₀/(4λ³) · [a·f_V(ṽ)·(f_L'·λ - f_L) + (f_PE'·λ - f_PE)]
    d2_term = activation * fv_factor * (dfL * lam - fL) + (dfPE * lam - fPE)
    d2Psi = sigma0 / (4.0 * lam * lam * lam) * d2_term

    return wp.vec2(dPsi, d2Psi)

@wp.func
def evaluate_fiber_pk1_and_hessian(
    F: wp.mat33,
    fiber_dir: wp.vec3,
    activation: float,
    sigma0: float,
    rest_volume: float,
    fv_factor: float,
) -> tuple[vec9, mat99]:
    """Compute fiber PK1 stress (vec9) and 9x9 Hessian, scaled by rest_volume.

    Fiber energy: W = Ψ(I₅) where I₅ = |F d|²
    PK1: P = 2Ψ'·F·A, A = d⊗d
    Hessian: H = 2Ψ'·kron(A,I₃) + 4Ψ''·outer(vec(FA), vec(FA))
    """
    d0 = fiber_dir
    A = wp.outer(d0, d0)

    Fd0 = F * d0
    I5 = wp.dot(Fd0, Fd0)

    derivs = fiber_energy_derivatives(I5, activation, sigma0, fv_factor)
    dPsi = derivs[0]
    d2Psi = derivs[1]

    # PK1 stress: P = 2·Ψ'·F·A
    FA = F * A
    P = FA * (2.0 * dPsi)
    P_vec = vec9(
        P[0, 0], P[1, 0], P[2, 0],
        P[0, 1], P[1, 1], P[2, 1],
        P[0, 2], P[1, 2], P[2, 2],
    )
    P_vec = P_vec * rest_volume

    # 9×9 Hessian
    FA_vec = vec9(
        FA[0, 0], FA[1, 0], FA[2, 0],
        FA[0, 1], FA[1, 1], FA[2, 1],
        FA[0, 2], FA[1, 2], FA[2, 2],
    )

    # Term 1: 2Ψ'·kron(A, I₃)
    H = mat99()
    for ci in range(3):
        for cj in range(3):
            a_val = A[ci, cj] * 2.0 * dPsi
            for k in range(3):
                H[ci * 3 + k, cj * 3 + k] = a_val

    # Term 2: 4Ψ''·outer(vec(FA), vec(FA))
    for i in range(9):
        for j in range(9):
            H[i, j] = H[i, j] + 4.0 * d2Psi * FA_vec[i] * FA_vec[j]

    H = H * rest_volume
    return P_vec, H

@wp.func
def evaluate_fiber_force_and_hessian(tet_id: int, v_order: int, ...):
    """Compute per-vertex (vec3 force, mat33 hessian) for one adjacent tet."""
    ...
```

**辅助 @wp.func**（内联 F-V 计算）：

```python
@wp.func
def compute_fv_factor_inline(
    tet_id: int,
    tet_indices: wp.array(dtype=wp.int32, ndim=2),
    Dm_inv: wp.mat33,
    fiber_dir: wp.vec3,
    q_prev: wp.array(dtype=wp.vec3),
    q_prev2: wp.array(dtype=wp.vec3),
    dt: float,
    v_max: float,
) -> float:
    """Compute f_V factor from two consecutive timestep positions (inline)."""
    i0 = tet_indices[tet_id, 0]
    i1 = tet_indices[tet_id, 1]
    i2 = tet_indices[tet_id, 2]
    i3 = tet_indices[tet_id, 3]

    # l̃_current from prev positions x(n-1)
    Ds1 = wp.matrix_from_cols(q_prev[i1] - q_prev[i0], q_prev[i2] - q_prev[i0], q_prev[i3] - q_prev[i0])
    Fd1 = (Ds1 * Dm_inv) * fiber_dir
    l_current = wp.sqrt(wp.dot(Fd1, Fd1))

    # l̃_prev from prev2 positions x(n-2)
    Ds2 = wp.matrix_from_cols(q_prev2[i1] - q_prev2[i0], q_prev2[i2] - q_prev2[i0], q_prev2[i3] - q_prev2[i0])
    Fd2 = (Ds2 * Dm_inv) * fiber_dir
    l_prev = wp.sqrt(wp.dot(Fd2, Fd2))

    v_norm = wp.clamp((l_current - l_prev) / (dt * v_max), -1.0, 1.0)
    return dgf_force_velocity(v_norm)
```

**@wp.kernel**（约 60 行）：

```python
@wp.kernel
def accumulate_fiber_force_and_hessian(
    dt: float,
    v_max: float,
    particle_ids_in_color: wp.array(dtype=wp.int32),
    particle_q_prev: wp.array(dtype=wp.vec3),     # x(n-1), 上一步位置
    particle_q_prev2: wp.array(dtype=wp.vec3),    # x(n-2), 两步前位置
    pos: wp.array(dtype=wp.vec3),                  # 当前迭代位置
    tet_indices: wp.array(dtype=wp.int32, ndim=2),
    tet_poses: wp.array(dtype=wp.mat33),
    tet_materials: wp.array(dtype=float, ndim=2),
    vmuscle_tet_fiber_dirs: wp.array(dtype=wp.vec3),
    vmuscle_tet_sigma0: wp.array(dtype=float),
    vmuscle_tet_activations: wp.array(dtype=float),
    particle_adjacency: ParticleForceElementAdjacencyInfo,
    # output (accumulated)
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
):
    """Accumulate Hill-type fiber force and Hessian for each particle in color group.

    F-V factor is computed inline from particle_q_prev and particle_q_prev2,
    no precomputation needed.
    """
    tid = wp.tid()
    particle_index = particle_ids_in_color[tid]
    f = wp.vec3(0.0, 0.0, 0.0)
    h = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    num_adj_tets = get_vertex_num_adjacent_tets(particle_adjacency, particle_index)
    for adj in range(num_adj_tets):
        tet_id, v_order = get_vertex_adjacent_tet_id_order(
            particle_adjacency, particle_index, adj)
        if vmuscle_tet_sigma0[tet_id] > 0.0:
            # inline F-V: compute l̃ from prev and prev2 positions
            fiber_dir = vmuscle_tet_fiber_dirs[tet_id]
            Dm_inv = tet_poses[tet_id]
            fv_factor = compute_fv_factor_inline(
                tet_id, tet_indices, Dm_inv, fiber_dir,
                particle_q_prev, particle_q_prev2, dt, v_max)

            f_fiber, h_fiber = evaluate_fiber_force_and_hessian(
                tet_id, v_order, particle_q_prev, pos,
                tet_indices, Dm_inv,
                fiber_dir,
                vmuscle_tet_sigma0[tet_id],
                vmuscle_tet_activations[tet_id],
                fv_factor,
                tet_materials[tet_id, 2], dt)
            f = f + f_fiber
            h = h + h_fiber

    particle_forces[particle_index] = particle_forces[particle_index] + f
    particle_hessians[particle_index] = particle_hessians[particle_index] + h
```

### 4.3 vmuscle_launch.py（新增文件）

封装 kernel launch 调用，供 `solver_vbd.py` 调用。保持 solver 代码简洁。

```python
"""Kernel launch wrappers for volumetric muscle (vmuscle) in VBD solver."""

import warp as wp
from .vmuscle_kernels import accumulate_fiber_force_and_hessian


def launch_accumulate_fiber_force_and_hessian(
    model, dt, color_group,
    particle_q_prev, particle_q_prev2, pos,
    particle_adjacency,
    particle_forces, particle_hessians, device,
):
    """Accumulate fiber force and Hessian for one color group."""
    wp.launch(
        kernel=accumulate_fiber_force_and_hessian,
        dim=color_group.size,
        inputs=[dt, model.vmuscle_max_contraction_velocity,
                color_group,
                particle_q_prev, particle_q_prev2, pos,
                model.tet_indices, model.tet_poses, model.tet_materials,
                model.vmuscle_tet_fiber_dirs, model.vmuscle_tet_sigma0,
                model.vmuscle_tet_activations,
                particle_adjacency],
        outputs=[particle_forces, particle_hessians],
        device=device,
    )
```

### 4.4 solver_vbd.py 修改

**修改点 1**：`__init__` 中初始化 `particle_q_prev2`

```python
# --- vmuscle: allocate prev2 position buffer (inserted) ---
if hasattr(self.model, 'vmuscle_count') and self.model.vmuscle_count > 0:
    n_particles = self.model.particle_count
    self.particle_q_prev2 = wp.zeros(n_particles, dtype=wp.vec3, device=self.device)
    wp.copy(self.particle_q_prev2, self.particle_q_prev)  # init to same as prev
```

**修改点 2**：`step()` 开头，更新 prev2（在任何求解之前）

```python
# --- vmuscle: shift prev positions (inserted) ---
if hasattr(self.model, 'vmuscle_count') and self.model.vmuscle_count > 0:
    wp.copy(self.particle_q_prev2, self.particle_q_prev)
```

**修改点 3**：`_solve_particle_iteration()` 中，在每个 color 的 `solve_elasticity` launch 之前插入

```python
# --- vmuscle: accumulate fiber force (inserted) ---
if hasattr(self.model, 'vmuscle_count') and self.model.vmuscle_count > 0:
    from .vmuscle_launch import launch_accumulate_fiber_force_and_hessian
    launch_accumulate_fiber_force_and_hessian(
        self.model, dt,
        self.model.particle_color_groups[color],
        self.particle_q_prev, self.particle_q_prev2,
        state_in.particle_q,
        self.particle_adjacency,
        self.particle_forces, self.particle_hessians,
        self.device)
```

> **注**：必须在 tiled 和 non-tiled 两个分支**之前**都插入。
>
> solver_vbd.py 总共只有 3 处插入，无需修改任何现有逻辑。

### 4.5 activation.py（主项目 `src/` 或 `examples/`）

DGF 激活动力学（excitation → activation 的一阶 ODE），可在 CPU 或 GPU 上运行：

```python
@wp.func
def dgf_activation_dynamics(excitation: float, activation: float, dt: float) -> float:
    """One step of DGF activation dynamics (implicit Euler).
    da/dt = [(1/(tau_a*(0.5+1.5*a))) * (0.5+0.5*tanh(b*(e-a)))
           + (0.5+1.5*a)/tau_d * (0.5-0.5*tanh(b*(e-a)))] * (e - a)
    """
    tau_a = 0.015  # activation time constant [s]
    tau_d = 0.060  # deactivation time constant [s]
    b = 10.0       # smoothing factor

    t = wp.tanh(b * (excitation - activation))
    f_act = (0.5 + 0.5 * t) / (tau_a * (0.5 + 1.5 * activation))
    f_deact = (0.5 - 0.5 * t) * (0.5 + 1.5 * activation) / tau_d
    da_dt = (f_act + f_deact) * (excitation - activation)

    a_new = activation + dt * da_dt
    return wp.clamp(a_new, 0.01, 1.0)


@wp.kernel
def update_activations(
    excitations: wp.array(dtype=float),
    activations: wp.array(dtype=float),
    dt: float,
):
    """Update all tet activations from excitation signals."""
    tid = wp.tid()
    activations[tid] = dgf_activation_dynamics(excitations[tid], activations[tid], dt)
```

CPU 版本（NumPy）也可用：

```python
def activation_dynamics_step(excitation: np.ndarray, activation: np.ndarray,
                              dt: float, tau_act=0.015, tau_deact=0.060) -> np.ndarray:
    """First-order activation dynamics: da/dt = (u - a) / tau(u, a)"""
    ...
```

此模块**不在 Newton 内部**，放在主项目 `examples/` 或 `src/`。每步调用后写入 `model.vmuscle_tet_activations`。

### 4.6 src/VMuscle/mesh_utils.py — vmuscle 辅助函数

放在 Example 层（`src/VMuscle/`），不侵入 Newton。提供两层 API：

**核心 API（程序化网格 / 通用）**：

```python
def set_vmuscle_properties(builder, tet_offset, fiber_dirs, sigma0):
    """Batch-set vmuscle properties for a range of tets already in the builder.

    Call after builder.add_soft_mesh() to tag newly added tets with fiber data.
    """
```

这是最底层的填充接口。`fiber_dirs` 为 (n_tets, 3) 数组，`sigma0` 可以是标量或 per-tet 数组。

**USD 网格导入 API**：

```python
def add_soft_vmuscle_mesh(builder, mesh, cfg, ...):
    """Import a TetMesh with volumetric muscle properties from USD.

    Reads 'materialW', 'muscle_id', 'tendonmask' from mesh.custom_attributes.
    内部先 add_soft_mesh()，再 per-vertex→per-tet 转换后填充 vmuscle 数据。
    """
```

内部流程：add_soft_mesh → 提取 custom_attributes → per-vertex→per-tet 转换 → 填充 builder.vmuscle_* 列表。

**网格生成工具**：

- `create_cylinder_tet_mesh(length, radius, n_length, n_radial)` — 生成圆柱体四面体网格
- `fix_tet_winding(vertices, tets)` — 修复反转四面体的顶点顺序
- `assign_fiber_directions(vertices, tets, axis)` — 沿指定轴为每个四面体赋纤维方向

### 4.7 examples/muscle_sliding_ball.py（Example 层算例入口）

算例入口，职责：
1. 调用 `mesh_utils` 生成圆柱网格 或 从 USD 加载肌肉网格
2. 使用 `newton.ModelBuilder` 构建 Model（设置材料参数、边界条件）
3. 创建 `newton.SolverVBD` 实例（标准 VBD，内部自动检测 vmuscle 数据）
4. 驱动仿真循环：设置 activation 序列 → 调用 `solver.step()` → 收集输出数据
5. 输出可视化结果到 `output/`

---

## 5. USD Primvar 约定与网格加载

### 5.1 属性命名约定

所有肌肉属性在 USD 中以 **per-vertex** primvar 存储（interpolation = `"vertex"`），与 Houdini 导出约定一致。

| USD primvar 名 | 内部变量名 | dtype | 说明 |
|---|---|---|---|
| `materialW` | `fiber_dir` | vec3 | 纤维方向（rest config 单位向量） |
| `muscle_id` | `muscle_id` | int | 肌肉组 ID（0, 1, 2, ...） |
| `tendonmask` | `tendonmask` | float | belly/tendon 权重（0.0=belly, 1.0=tendon） |

- `materialW` 是 Houdini 的默认名字，读取后在内部映射为 `fiber_dir`。
- `muscle_id` 和 `tendonmask` 与 Houdini 约定一致，直接使用。
- Newton 的 `newton.usd.get_tetmesh()` 自动读取 primvar 并存入 `TetMesh.custom_attributes`，无需额外加载代码。

### 5.2 JSON Config

在现有 config JSON（如 `data/Human/xxx.json`）中追加以下字段，通过 `load_config()` 加载为 `SimConfig`：

```json
{
  "...existing fields...": "...",

  "tendon_threshold": 0.7,
  "tendon_k_mu": 5e4,
  "tendon_k_lambda": 5e4,
  "vmuscle_max_contraction_velocity": 10.0,
  "muscles": {
    "0": {"sigma0": 3e5, "label": "bicep"},
    "1": {"sigma0": 2e5, "label": "tricep"}
  }
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `tendon_threshold` | float | tendonmask 阈值（>= 此值视为 tendon） |
| `tendon_k_mu` | float | tendon 区域覆盖的剪切模量 [Pa] |
| `tendon_k_lambda` | float | tendon 区域覆盖的体积模量 [Pa] |
| `vmuscle_max_contraction_velocity` | float | V_max [l_opt/s]，DGF 默认 10.0 |
| `muscles` | dict | key=muscle_id(str), value=per-muscle 参数 |
| `muscles[id].sigma0` | float | 峰值等长应力 [Pa] |
| `muscles[id].label` | str | 可选，肌肉名称（便于调试） |

### 5.3 Per-vertex → Per-tet 转换规则

| 属性 | 转换方式 | 说明 |
|---|---|---|
| `fiber_dir` | 4 顶点平均后归一化 | `normalize(mean(fiber_dirs[v0..v3]))` |
| `muscle_id` | 取第一个顶点的值 | 快速原型简化，后续可改为多数投票 |
| `tendonmask` | 4 顶点平均 | 与阈值比较决定是否为 tendon tet |

### 5.4 不需要新增的东西

- 不需要新数据类（`VMuscleMesh` 等）
- 不需要修改 Newton 的 `TetMesh` 类或 USD 加载代码
- 不需要修改 Newton 的 `add_soft_mesh()` 方法
- **不需要 `add_vmuscle_tetrahedron()`**（与 `add_soft_mesh()` 工作流冲突，会重复添加 tet）。vmuscle 数据通过外部 `set_vmuscle_properties()` 在 `add_soft_mesh()` 之后批量填充。

---

## 6. SPD 策略

使用 **clamp Ψ' ≥ 0** 策略（方案 A）：

```python
dPsi = wp.max(dPsi, 0.0)  # ensure PSD Hessian
```

在正常拉伸区（l̃ ∈ [0.5, 2.0]），kron(A, I₃) 和 outer 项均半正定。极端压缩区（l̃ < 0.5）时，Newton 已有的 `det(h) > 1e-8` 检查会自动跳过该顶点更新。

如果后续遇到收敛问题，可升级到 eigenvalue clamping（方案 B）。

> **实施备注 (2026-03-23)**：实际实现中发现仅 clamp Ψ' ≥ 0 不够，Ψ'' 也需要 clamp ≥ 0。
> 原因：active force 的导数项 (f_L'·λ - f_L) 在 λ ≈ 1 附近为负（f_L 在峰值处导数 ≈ 0），
> 导致 4·Ψ''·outer(FA,FA) 项使 Hessian 负定。已在 vmuscle_kernels.py 中增加 `d2Psi = wp.max(d2Psi, 0.0)`。
>
> 此外，DGF active force-length 曲线的 Gaussian 2 分母 `(-0.030 + 0.200·l̃)` 在 `l̃ ≈ 0.15` 时过零，
> 已增加 `wp.max(wp.abs(...), 1e-6)` 保护以防除零。

---

## 7. 参数映射

| Hill 参数 | FEM 参数 | 映射关系 |
|---|---|---|
| `max_isometric_force` (F₀) | `sigma0` | `σ₀ = F₀ / PCSA` |
| `optimal_fiber_length` (l_opt) | 网格几何 | 网格在 l_opt 处建模 |
| `pennation_angle` (α₀) | `fiber_dir` | 每单元纤维方向向量 |
| `max_contraction_velocity` (V_max) | `vmuscle_max_contraction_velocity` | 默认 10 l_opt/s |
| `fiber_damping` (β) | `tet_materials[:, 2]` | 复用 Newton 的 k_damp |

---

## 8. 测试计划

### 8.1 单元测试

- **DGF 曲线验证**：f_L(1.0) ≈ 1.0, f_PE(1.0) ≈ 0, f_V(0) ≈ 1.0
- **导数有限差分**：f_L', f_PE', f_V' 与 FD 误差 < 1e-4
- **纤维梯度 FD**：单四面体纤维力 vs 能量有限差分，误差 < 1e-4
- **f_V 乘子正确性**：f_V=1 时退化为纯 F-L 模型

### 8.2 集成测试

- **无肌肉回归**：所有 sigma0=0 时，行为与原 SolverVBD 完全一致
- **单肌肉收缩**：activation=1.0 → 自由端产生位移
- **F-L 配准**：固定两端扫描 l̃ ∈ [0.5, 1.5]，比较稳态力 vs OpenSim DGF 曲线，RMSE < 5%
- **F-V 配准**：固定 l̃=1.0，施加不同速度，比较力 vs DGF f_V 曲线，|ṽ| < 0.5 误差 < 10%

### 8.3 线图对比测试

所有线图保存到 `output/` 目录，用于定量对比。

- **DGF 参考曲线**：
  - f_L(l̃) 曲线，l̃ ∈ [0.4, 1.6]（三高斯解析曲线）
  - f_V(ṽ) 曲线，ṽ ∈ [-1, 1]（sinh⁻¹ 解析曲线）
  - f_PE(l̃) 曲线，l̃ ∈ [0.4, 1.6]（指数解析曲线）

- **VBD 实测 vs DGF 参考叠加图**：
  - **F-L 对比图**：VBD 仿真中实测的稳态纤维力（纵轴）vs 实测纤维长度 l̃（横轴），叠加 DGF f_L 解析曲线。数据来源：固定两端扫描不同 l̃，读取收敛后的实际纤维应力
  - **F-V 对比图**：VBD 仿真中实测的纤维力（纵轴）vs 实测纤维速度 ṽ（横轴），叠加 DGF f_V 解析曲线。数据来源：固定 l̃≈1.0，施加不同速度，读取实际力输出

- **轨迹图**：
  - 自由端小球高度随时间变化曲线（单肌肉收缩场景），验证收缩动力学的合理性

### 8.4 可视化测试

- 圆柱肌肉收缩动画 → 截图到 `output/` 对比

### 8.5 激活动力学测试

- **阶跃响应**：50% rise time ~11ms, 90% rise time ~38ms（DGF spec）

---

## 9. 风险与缓解

| 风险 | 缓解 |
|------|------|
| vmuscle 数组长度与 tet_count 不匹配 | `finalize()` 中 padding vmuscle 数组到 tet_count 长度，非肌肉 tet 的 sigma0=0 |
| 第一个时间步 prev2 == prev，ṽ=0 | f_V(0)≈1.0，退化为纯 F-L 模型，物理上合理 |
| F-V lagged velocity 在大 dt 时不准 | clamp v_norm 到 [-1, 1]；dt > 0.01s 时可能需要子步 |
| 每个 tet 重复计算 F（prev 和 prev2） | GPU 上 3×3 矩阵乘法开销可忽略，换取更简洁的数据流 |
| 纤维 Hessian 不定导致不收敛 | Ψ' clamp ≥ 0 + Newton 的 det 检查双重保护 |
| F 计算两次（muscle kernel + elasticity kernel） | GPU 上 3×3 矩阵乘法开销可忽略 |
| `tet_materials` ndim 假设 | 实现前验证 `model.tet_materials.ndim == 2`，如为 1D 需调整索引 |
| Newton 上游更新冲突 | `feat/vbd-muscle` 分支隔离，合并前 rebase；新增文件无冲突，源文件插入点有限 |

---

## 10. 未来扩展

- **requires_grad 支持**：当前 vmuscle 数组不启用梯度。未来 RL 需要通过肌肉能量反传梯度时，`finalize()` 中需传入 `requires_grad` 参数并透传到 `wp.array` 创建。
- **Eigenvalue clamping (SPD 方案 B)**：如果收敛问题严重，实现 3×3 Jacobi SVD + 特征值截断。
- **非线性 F-V Hessian**：当前 f_V 在 Hessian 中作为常数。如需更高精度，可在 VBD 迭代内更新 f_V（代价：每次迭代多一次 kernel launch）。
