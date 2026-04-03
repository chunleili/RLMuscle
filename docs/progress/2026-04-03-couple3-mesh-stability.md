# couple3 mesh 稳定性调查

## 问题

example_couple3 在 frame 3（activation 开始时）mesh 完全崩溃，所有 tet 反转。

## 根因分析

### 1. 退化四面体导致极端质量比

Bicep mesh（来自 Houdini）存在极端质量比：
- min_mass = 3.74e-7 kg, max_mass = 4.52e-3 kg → **比值 12095:1**
- 117 个顶点（9.4%）质量不到均值的 1%
- min tet volume = 3.74e-10 m³ vs max = 1.01e-6 m³

### 2. Explicit force + 退化 tets → 不稳定

对于退化 tets，`Dm_inv` 的条目极大（条件数最高 2689），放大了 force：
- `F_j = -V₀ · τ_act · w[j] · n`，其中 `w = Dm_inv @ d`
- max_force = 23.2 N（activation=1.0, sigma0=300kPa）
- min_mass 顶点的 acceleration = **368,174 m/s²**
- 单个 substep 的 max_dx = **0.71 m**

对比：在 simple_arm 的均匀圆柱 mesh 上，相同方法完全稳定。

### 3. 对比：couple2 基线

couple2（TETFIBERNORM 约束，无 explicit force）在 activation=1.0 时也有 **~70 个反转 tets（1.8%）**。这是网格本身的基线问题，不是 explicit force 独有的。

## 修复

### Mass floor（`_precompute_rest` 中添加）

当 sigma0 > 0 时，自动将 vertex mass 下限设为 median mass 的 1%。影响 24 个顶点。

### Acceleration clamping（`accumulate_active_fiber_force` 后添加）

新增 `_clamp_force_by_accel_kernel`：将每个顶点的 |force|/mass 限制到 `max_accel`（默认 2000 m/s²，config 可覆盖）。

### 被动 fiber 约束

在 config 中恢复 TETFIBERNORM 约束（stiffness=1000），但 `contraction_ratio=0` 使其不驱动收缩，仅提供纤维方向的被动弹性抵抗。

### num_substeps 增加

从 10 增加到 30，减小每个 substep 的 dts。

## 反转 tets 对比

| 配置 | activation=1.0 时反转 tets |
|------|---------------------------|
| couple2（TETFIBERNORM） | ~70 (1.8%) |
| couple3 max_accel=20 | ~66 (1.7%) |
| couple3 max_accel=10 | ~56 (1.4%) |
| couple3 max_accel=100 | ~300 (7.6%) |
| couple3 无 clamping | ~1950 (49.5%) — 爆炸 |

## 当前限制

max_accel 越低越稳定，但 explicit force 贡献越小：
- max_accel=20: torque ~0.14 N·m（很弱）
- 原始（无 clamping）: torque ~6.5 N·m（mesh 爆炸）

## 下一步

1. 调优 max_accel 和 k_coupling 的平衡点
2. 考虑 mesh 质量改进（remesh 去除退化 tets）
3. 考虑混合方案：TETFIBERNORM 驱动主要收缩 + 小量 explicit force 调制
4. 或使用 TETFIBERMILLARD 能量约束（Route A）替代 explicit force（Route C）
