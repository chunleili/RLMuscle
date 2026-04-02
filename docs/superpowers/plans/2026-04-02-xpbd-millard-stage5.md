# XPBD-Millard Stage 5: 能量本构 C=√(2Ψ_L) 主动力约束

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用能量本构约束 C=√(2·Ψ_L) 替代当前的 stiffness-modulated + target_stretch 方案，使 XPBD 约束力自动匹配 Millard 主动力-长度曲线，消除 CPU 端平衡反求。

**Architecture:** 在 GPU kernel 中直接从 Millard f_L 的能量积分（已有闭式10次多项式）计算 C 值。约束力 ∝ dΨ/dλ = σ₀·a·f_L(λ)，自动恢复 Hill 力-长度关系。

**Tech Stack:** Python, Warp (GPU kernels), NumPy, MuJoCo (coupling)

**Spec:** `docs/superpowers/specs/2026-04-02-xpbd-millard-stage5-design.md`

**失败方案记录:** 动态目标力注入 `target = λ - σ₀·f_total/k_base` 已证明不可行（k_base cancel、质量比问题无法解决）。

---

### Task 1: 实现 GPU 端 Millard 能量求值函数

**Files:**
- Modify: `src/VMuscle/muscle_warp.py` (新增 `millard_energy_eval_wp` 函数)
- Modify: `src/VMuscle/millard_curves.py` (上传 f_L 能量积分系数到 GPU)

**Context:** `MillardCurves` 已在 CPU 端通过解析方法预计算了 f_L 的能量多项式系数（`_build_energy_integral()` → 闭式10次多项式 `F_coeffs[11]`，由 Bezier 控制点的 y(u)·x'(u) 卷积 + 逐项求原函数得到，**非数值积分**）。需要：
1. 将 f_L 的能量多项式系数上传为 Warp 数组
2. 实现 GPU 函数 `millard_energy_eval_wp`，输入 λ，直接求值闭式多项式输出 Ψ_L(λ)

能量参考点：Ψ_L(λ_min=0.4441) = 0，确保 Ψ_L ≥ 0 对所有 λ ≥ λ_min。

- [ ] **Step 1: 检查现有能量多项式实现**

阅读 `src/VMuscle/millard_curves.py` 中的 `_build_energy_integral()` 方法，理解闭式能量多项式系数的存储格式和求值方式。确认 CPU 端 `energy_eval(lm)` 的正确性。

- [ ] **Step 2: 在 MillardCurves 中添加能量多项式系数上传到 GPU**

在 `_init_millard_curves()`（`muscle_warp.py`）或 MillardCurves 中，将 f_L 的闭式能量多项式系数（每段11个系数 × n_seg段）和段边界上传为 `wp.array`。

- [ ] **Step 3: 实现 `millard_energy_eval_wp` GPU 函数**

逻辑与 `millard_eval_wp`（已有的力求值）类似，直接求值预计算的闭式多项式（非数值积分）：
1. 域外检查（λ < x_lo 或 λ > x_hi）→ 边界能量值（需预计算）
2. 线性扫描找段
3. Newton 迭代 x → u
4. Horner 求值 e_coeffs（闭式10次多项式，而非力的5次）

关键：能量多项式在参数空间 u 中求值，需要加上该段起始点的累积能量偏移。

- [ ] **Step 4: 单元测试——GPU 能量求值 vs CPU 能量求值**

扩展 `tests/test_millard_gpu.py`，对比 `millard_energy_eval_wp(λ)` 和 CPU 端 `mc.fl.energy_eval(λ)` 的结果，确认精度 < 1e-5。

- [ ] **Step 5: Commit**

---

### Task 2: CPU 端验证——能量约束公式正确性

**原则：总是先用 CPU 实验验证数学正确性，再集成到 GPU kernel。**

**Files:**
- Create: `scripts/experiment_energy_constraint.py`（临时 CPU 验证脚本）

**Context:** 在修改 GPU kernel 之前，先用纯 CPU/NumPy 验证 C=√(2Ψ_L) 的行为。构建一个简化的 1D XPBD 求解循环（单 tet、单约束），验证：
1. 能量约束能否驱动纤维收缩（activation > 0 时 λ 从 1.0 下降）
2. 在外力平衡下是否收敛到正确的 λ_eq（对比 Stage 4 的 0.5498）
3. C 值和梯度在整个 λ 范围内的数值稳定性

- [ ] **Step 1: 编写 CPU 验证脚本**

使用 `MillardCurves` 的 CPU 端 `energy_eval(lm)` 和 `eval(lm)` 方法。模拟 XPBD 单约束求解：
```python
for iter in range(n_iters):
    psi_L = mc.fl.energy_eval(lambda_current)  # 能量积分
    f_L = mc.fl.eval(lambda_current)            # 力
    C = sqrt(2 * sigma0 * a * psi_L)
    if C < eps: break
    dC_dlambda = sigma0 * a * f_L / C
    # XPBD update: delta_lambda = -C / (w * dC^2 + alpha)
    ...
```

验证结果 vs Stage 4 sliding ball 平衡点。

- [ ] **Step 2: 运行 CPU 验证，确认数值正确**
- [ ] **Step 3: Commit 验证脚本（放 scripts/）**

---

### Task 3: 修改 TETFIBERMILLARD Kernel 使用能量约束

**Files:**
- Modify: `src/VMuscle/muscle_warp.py` (`solve_tetfibermillard_kernel`)

**Context:** 当前 kernel 使用 `stiffness = k * max(f_total, 0.01) * scale` + `target = 1 - a * cf`。替换为能量约束 `C = √(2·σ₀·a·Ψ_L(λ))`。

约束梯度推导：
```
C = √(2·σ₀·a·Ψ_L)
dC/dλ = σ₀·a·f_L(λ) / C    (when C > ε)
```

XPBD 更新公式不变：`Δλ_n = -C / (Σw_i|∇C_i|² + α)`，只是 C 和 ∇C 的计算方式改变。

- [ ] **Step 1: 修改 kernel**

替换 stiffness + target_stretch 逻辑为能量约束计算。注意：
- 当 Ψ_L < ε（如 1e-12）时跳过（C=0，无偏差）
- `fiber_stiffness_scale` 继续作为能量的缩放因子
- 需要同时传入 f_L 力数据（已有）和 f_L 能量数据（新增）

- [ ] **Step 2: 更新 `_init_millard_curves` 和 `_dispatch_constraints`**

确保能量数据数组在 kernel launch 时传入。

- [ ] **Step 3: 编译测试**

`uv run python -c "import warp as wp; wp.init(); from VMuscle.muscle_warp import solve_tetfibermillard_kernel; print('OK')"`

- [ ] **Step 4: Commit**

---

### Task 4: Sliding Ball 验证（CPU 先行 → GPU）

**Files:**
- Modify: `examples/example_xpbd_millard_sliding_ball.py`
- Modify: `data/slidingBall/config_xpbd_millard.json`

**Context:** 删除 CPU 端 `millard_equilibrium_fiber_length` 反求和 `update_cons_restdir1_kernel` 调用。能量约束自动达到正确平衡点。

- [ ] **Step 1: CPU 模式运行 sliding ball，验证能量约束行为**

先用 CPU device 运行，便于调试和打印中间值。确认：
- λ_eq 收敛到 ~0.5498
- 球位置收敛到 ~0.0449m
- 无 NaN、无发散

- [ ] **Step 2: 删除 CPU 平衡反求代码，清理 imports**
- [ ] **Step 3: GPU 模式运行，对比 CPU 结果**

`RUN=example_xpbd_millard_sliding_ball uv run main.py`

对比 Stage 4 结果：λ_eq ≈ 0.5498, 球位置 ≈ 0.0449m（误差 <5%）

- [ ] **Step 4: Commit**

---

### Task 5: Simple Arm 验证（CPU 先行 → GPU）

**Files:**
- Modify: `src/VMuscle/muscle_warp.py` (新增 `compute_fiber_stretch_kernel` + `extract_fiber_stretch()`)
- Modify: `examples/example_xpbd_coupled_simple_arm_millard.py`

**Context:**
1. 新增 GPU kernel 提取每 tet 的 fiber stretch
2. 新增 `MuscleSim.extract_fiber_stretch()` 返回均值
3. 在 coupled simple arm 中，用 mesh DG 提取的 stretch 替代 1D Hill 模型计算力
4. 如果使用 TETFIBERMILLARD 约束，需要添加到 sim builder

- [ ] **Step 1: 添加 GPU fiber stretch extraction kernel + method**
- [ ] **Step 2: 修改 coupled simple arm 使用 mesh 力提取**
- [ ] **Step 3: CPU 模式运行 simple arm，验证正确性**

先用 CPU device 运行，确认：
- 稳态肘角 ≈ 88.25°
- 力范围合理（~4-55N）
- 无 NaN、无 inverted tet

- [ ] **Step 4: GPU 模式运行，对比 CPU 结果**

`RUN=example_xpbd_coupled_simple_arm_millard uv run main.py`

对比 Stage 4：稳态肘角 ≈ 88.25°（误差 <2°）

- [ ] **Step 5: Commit**

---

### Task 6: 验证与文档

- [ ] **Step 1: 汇总两个示例的 CPU/GPU 对比结果**
- [ ] **Step 2: 创建 `docs/progress/2026-04-02-xpbd-millard-stage5.md`**
- [ ] **Step 3: Commit**
