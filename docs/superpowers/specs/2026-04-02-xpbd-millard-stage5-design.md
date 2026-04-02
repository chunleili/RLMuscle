# XPBD-Millard Stage 5: 能量本构 C=√(2Ψ) + DG 力提取

> 日期: 2026-04-02
> 分支: xpbd-millard-stage5
> 前置: Stage 1-4 完成（Millard 曲线模块、TETFIBERMILLARD 约束、sliding ball + simple arm 验证）

## 问题

1. **Mesh 与力分离**：XPBD mesh 仅用于可视化，力输出走独立的 1D Hill 模型（从 MuJoCo tendon_length 计算）
2. **CPU-GPU 通信瓶颈**：每步需 CPU 端反求 Millard 平衡点 → contraction_factor → GPU kernel 更新
3. **两端约束 tet 翻转**：target_stretch 驱动在 isometric 场景下产生过大 C 值 → mesh collapse
4. **不可泛化**：1D 模型假设均匀纤维方向和截面积，无法扩展到复杂几何（人体全身）

## 设计约束

- **稳定性最优先**：必须容忍任意复杂 mesh、任意激活输入不 crash（后续用于 RL）
- **可泛化**：设计必须适用于非圆柱几何、多肌肉系统
- **性能**：消除 CPU-GPU 通信，GPU-only 求解

## 失败方案记录

### 方案 A: 动态目标力注入（Dynamic Target Force Injection）— 已失败

**思路**：`target = λ - σ₀·f_total/k_base`，`stiffness = k_base`，使约束力精确等于 Millard 力。

**失败原因**：
- XPBD 质量加权更新中，内部顶点质量极小（~0.0002kg），constraint correction 被放大 → mesh 过度收缩
- k_base 在力表达式中 cancel 掉（力 = stiffness × C = k_base × σ₀·f_total/k_base = σ₀·f_total），无法通过调节 k_base 控制位移量
- 即使改善质量比（增加密度、减少球质量等），方案仍然不成功
- 与 Stage 3 发现的"target=0 方案不可行"同源：XPBD compliance-based 求解器中，力的大小由 C/w_sum 决定，当 C 和 stiffness 之间存在耦合时无法独立控制

## 方案选择

### 方案 C: 能量本构约束 C = √(2Ψ_Millard)

**核心思想**：利用 XPBD 约束的能量等价性。XPBD 约束 C 在一次 solve 中产生的能量为：

```
E_constraint ≈ ½ · C² / α ≈ ½ · C² · stiffness
```

如果我们令：

```
C = √(2 · Ψ_Millard(λ))
stiffness = 1.0（或适当缩放）
```

则约束产生的能量密度 ≈ Ψ_Millard(λ)，与 Millard 本构模型精确一致。

**关键优势**：
1. **Ψ 已有闭式解**：Millard quintic Bezier 的能量积分是 10 次多项式，已在 Stage 1 实现
2. **GPU-only**：C 从当前 λ 直接计算，无需 CPU 端平衡反求
3. **物理一致性**：约束力 = ∂E/∂x ∝ dΨ/dλ = f_hill(λ)，自动恢复 Hill 力-长度关系
4. **有界性**：Ψ ≥ 0（被动能量天然非负；主动能量需偏移处理），C ≥ 0
5. **无需 target_stretch**：约束的 rest state 对应能量最小值的位置（不一定是 Ψ=0，但一定是 Ψ 的极小值点）

### 数学推导

#### 主动力 a·f_L（Stage 5 重点）

主动能量密度：
```
Ψ_active(λ, a) = σ₀ · a · ∫_{λ_opt}^λ f_L(s) ds
```

**问题**：f_L 在 λ=λ_opt=1.0 处取最大值 1.0，在两侧下降。积分 ∫₁^λ f_L(s) ds：
- λ > 1: 正值（沿下降段积分）
- λ < 1: 负值（f_L 在 ascending 段单调递增，从1反向积分）

**偏移处理**：定义 Ψ_active 的参考点为 f_L 的左肩 λ_min（Millard: 0.4441）：
```
Ψ_active(λ, a) = σ₀ · a · ∫_{λ_min}^{λ} f_L(λ') dλ'
```

由于 f_L(λ_min) = y_low = 0.1 > 0（Millard 肩部非零），且 f_L 在 [λ_min, λ_max] 上恒正，所以 Ψ_active ≥ 0 对所有 λ ≥ λ_min。

**约束的物理含义**：C_active = √(2·Ψ_active) 编码了"纤维偏离最短长度的能量代价"。求解器会驱动纤维向 λ_min 方向收缩（能量最小化），但被 volume/arap/attach 约束平衡在物理合理的长度。

**注意**：被动力 f_PE 暂不处理（保持现有 stiffness-modulated 方案或忽略）。Stage 5 聚焦验证主动力能量约束的正确性。

---

## 架构设计

### 1. 修改 solve_tetfibermillard_kernel

**文件**: `src/VMuscle/muscle_warp.py`

当前流程：
```
CPU: millard_equilibrium_fiber_length(a, load) → cf
CPU→GPU: update_cons_restdir1_kernel(cf)
GPU: target = 1 - a * cf, stiffness = k * f_total
```

新流程（全 GPU，仅主动力）：
```
GPU: λ = ||F·d||                           (已有)
GPU: Ψ_L  = millard_energy_eval_wp(λ)     (新增：10次多项式求值)
GPU: C_act = √(2·σ₀·a·Ψ_L)              (新)
GPU: XPBD solve with C_act                (新)
```

**具体修改**：

```python
# 新增 GPU 函数：Millard 能量解析求值（闭式多项式，非数值积分）
@wp.func
def millard_energy_eval_wp(
    lm: float,
    x_coeffs: wp.array(dtype=float),
    e_coeffs: wp.array(dtype=float),  # 闭式能量多项式系数（10次，预计算）
    seg_bounds: wp.array(dtype=float),
    n_seg: int,
    x_lo: float, x_hi: float,
) -> float:
    """Evaluate Millard energy Ψ_L(λ) via closed-form 10th-degree polynomial.

    Energy coefficients are pre-computed analytically from Bezier control points
    (y(u)·x'(u) convolution + term-by-term antiderivative), NOT numerical quadrature.
    """
    # 域外：返回边界值
    # 域内：查找段 → Newton x→u → Horner 求值 e_coeffs（10次多项式直接求值）
    ...

# 约束内核修改
sigma0 = cons[c].restdir[0]

# 主动能量约束
psi_act = sigma0 * acti * millard_energy_eval_wp(lm_tilde, fl_energy_data...)
if psi_act > 1.0e-12:
    C_act = wp.sqrt(2.0 * psi_act)
    # dC/dlambda 需要 f_L(lm_tilde)，已有 millard_eval_wp
    f_L_val = millard_eval_wp(lm_tilde, fl_data...)
    grad_C_wrt_lambda = sigma0 * acti * f_L_val / C_act
    # XPBD position update with C_act and grad_C_wrt_lambda
    ...
```

### 2. 扩展 MillardCurves 的能量数据上传

**文件**: `src/VMuscle/millard_curves.py`

能量积分系数（10次多项式）已在 `_build_energy_integral()` 中计算。需要：
- 将 f_L 的能量系数上传为 Warp 数组
- f_L 的能量参考点设为 λ_min = 0.4441（确保 Ψ_L ≥ 0）

### 3. GPU 纤维拉伸提取 kernel（不变）

与之前方案相同：`compute_fiber_stretch_kernel` 提取每 tet 的 ||F·d||，用于耦合力计算。

### 4. 修改耦合 Simple Arm 示例

**文件**: `examples/example_xpbd_coupled_simple_arm_millard.py`

与之前方案相同：删除 1D Hill 模型，用 mesh DG 提取的 fiber stretch + Millard 曲线评估 force。

### 5. 修改 Sliding Ball 示例

**文件**: `examples/example_xpbd_millard_sliding_ball.py`

删除 CPU 端平衡反求。能量约束自动达到正确平衡点，不需要 contraction_factor。

---

## 稳定性分析

### C 值有界性

Ψ_L(λ) = ∫_{λ_min}^λ f_L(s)ds。f_L 有界（最大值 1.0），所以 Ψ_L(λ_max) = ∫_{0.44}^{1.81} f_L ds ≈ 0.7（有限值）。C_act = √(2σ₀·a·Ψ_L) 有界。

### 奇异性处理

当 Ψ → 0 时，C → 0 但 dC/dλ → ∞（梯度爆炸）。处理方式：
- 设置阈值 ε（如 1e-10），当 Ψ < ε 时跳过约束（C=0 即无偏差，无需修正）
- 或使用正则化：C = √(2Ψ + ε²) - ε

### RL 鲁棒性

- activation ∈ [0, 1]：C_act ∝ √a，有界且平滑
- activation = 0：C_act = 0，不施加任何力（正确）
- activation 突变：C 平滑变化（√ 函数），不会产生脉冲

---

## 力提取数学

（同之前方案，不变）

对每个 tet i：
```
F_i = Ds_i · Dm_i⁻¹
λ_i = ||F_i · d_i||
```

平均纤维拉伸 → 归一化 → Millard 曲线求值 → 力输出。

---

## 验证计划

### Sliding Ball 验证
- 对比 Stage 4 结果：λ_eq、球位置误差应 <2%
- 对比 OpenSim-Millard 基准
- 确认已无 CPU 端平衡反求调用

### Simple Arm 验证
- 对比 Stage 4 结果：稳态肘角误差 <1°
- 确认力来自 mesh DG 而非 1D 模型
- 确认无 NaN、无 inverted tet
- 1000 步 (10s) 稳定性

---

## 关键文件清单

| 操作 | 文件 | 改动量 |
|------|------|--------|
| 修改 | `src/VMuscle/muscle_warp.py` | ~100 行（能量求值函数 + kernel 重写） |
| 修改 | `src/VMuscle/millard_curves.py` | ~30 行（能量系数上传） |
| 修改 | `src/VMuscle/constraints.py` | ~10 行（移除 contraction_factor） |
| 修改 | `examples/example_xpbd_millard_sliding_ball.py` | ~30 行（删除 CPU 平衡反求） |
| 修改 | `examples/example_xpbd_coupled_simple_arm_millard.py` | ~40 行（替换 1D Hill 为 DG 提取） |
| 不变 | `src/VMuscle/millard_curves.py` (CPU 端) | 能量积分已实现 |
