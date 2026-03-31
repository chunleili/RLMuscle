# XPBD-DGF: 基于 f-L 曲线的约束求解

## 背景

原始 `TETFIBERDGF` 约束使用**常数刚度** (`10 * activation`)，DGF 曲线仅用于后处理分析。
本次改动将 DGF active force-length 曲线集成到约束求解内核中，使 XPBD 力-长度行为与 OpenSim `DeGrooteFregly2016Muscle` 一致。

## 改动摘要

### 1. 约束求解内核 `solve_tetfiberdgf_kernel`（`src/VMuscle/muscle_warp.py`）

**之前：**
```python
fiberscale = 10.0 * acti                          # 常数
stiffness_val = cstiffness * fiberscale * scale
target_stretch = 1.0 - acti * contraction_factor   # 固定 cf
```

**现在：**
```python
f_L_val = dgf_active_force_length_wp(lm_tilde)
f_PE_val = dgf_passive_force_length_wp(lm_tilde)
f_total = acti * f_L_val + f_PE_val
fiberscale = max(f_total, 0.01)                    # DGF 曲线调制
stiffness_val = cstiffness * fiberscale * scale
target_stretch = 1.0 - acti * contraction_factor   # cf 由 DGF 反求
```

- 刚度随当前纤维长度 `lm_tilde` 变化，遵循 DGF 钟形 f-L 曲线
- `sigma0` 通过 `restdir[0]` 传入（预留，未直接用于刚度公式）
- `contraction_factor` 通过 `restdir[1]` 每步动态更新

### 2. 约束构建 `create_tet_fiber_dgf_constraint`（`src/VMuscle/constraints.py`）

- `restdir[0]` = `sigma0`（峰值等距应力，Pa）
- `restdir[1]` = `contraction_factor`（从 DGF 曲线反求，运行时更新）
- 新增参数 `sigma0`、`contraction_factor` 传入

### 3. DGF 平衡点反求（`examples/example_xpbd_dgf_sliding_ball.py`）

新增 `dgf_equilibrium_fiber_length(activation, normalized_load)`:
- 输入：激活水平 `a`，归一化载荷 `F_ext / F_max`
- 输出：ascending limb 上满足 `a * f_L(lm) = normalized_load` 的 `lm_eq`
- 每步计算 `contraction_factor = 1 - lm_eq`，通过 GPU kernel 更新约束

### 4. GPU 约束更新 kernel（`src/VMuscle/muscle_warp.py`）

新增 `update_cons_restdir1_kernel`：每步将新的 `contraction_factor` 写入所有 `TETFIBERDGF` 约束的 `restdir[1]`。

### 5. 示例代码变更（`examples/example_xpbd_dgf_sliding_ball.py`）

- `run_sim()` 中每步计算 DGF 平衡点并更新 contraction_factor
- 新增 `norm_load = ball_mass * g / F_max` 计算
- 导入 `update_cons_restdir1_kernel`、`active_force_length`

## 验证结果

sliding ball, excitation=1.0, 300 steps (5s), ball_mass=10kg, sigma0=300kPa:

| 指标 | XPBD-DGF | OpenSim | 误差 |
|------|----------|---------|------|
| 归一化纤维长度 | 0.5945 | 0.5899 | 0.8% |
| 球位置 (m) | 0.0405 | 0.0400 | 1.2% |
| 稳态主动力 (normalized) | ~0.26 | ~0.26 | <1% |

## 关键发现

1. **`target=0` 方案不可行**：所有 tet 同时收缩到零长度，内部顶点质量太小（~0.002kg vs 球 1.4kg/vert），无法通过 XPBD compliance 自然平衡。即使增加迭代次数、提高密度、或加入 SNH 约束，都无法得到正确平衡点。

2. **正确方案**：DGF 曲线反求目标拉伸 + f_L 调制刚度。目标保持在物理合理范围（~0.59），避免 mesh collapse；DGF 调制刚度保证力-长度 profile 正确。

3. **XPBD 收敛特性**：L (Lagrange multiplier) 每 substep 重置，因此单次 solve 的力取决于 C（约束偏差）和 w_sum（质量加权梯度范数）。太小的 C 导致力不足，太大的 C 导致内部不稳定。DGF 反求目标使 C 保持适中。

## 下一步

- 测试变激活（excitation ramp / step）下的瞬态响应
- 与 VBD 版 sliding ball 对比
- 考虑加入 force-velocity 曲线用于动态场景
