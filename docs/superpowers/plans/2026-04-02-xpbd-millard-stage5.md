# XPBD-Millard Stage 5: 显式主动力 + 被动 XPBD 约束

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用显式主动力（direct force）替代 XPBD 能量约束，使肌肉主动力直接作为 FEM 节点力施加，消除 CPU 端平衡反求。

**Architecture:** 
- Active force: 直接在 GPU kernel 中计算 τ_act = σ₀·a·f_FL(r)，散射到节点力
- Passive: SNH 体积约束保留在 XPBD 中
- 关键对齐: r = λ_f / λ_opt（fiber stretch ratio 分母对齐）

**Tech Stack:** Python, Warp (GPU kernels), NumPy

**Spec:** `docs/superpowers/specs/2026-04-02-xpbd-millard-stage5-design.md`
**理论参考:** `docs/notes/xpbd_fem_fiber_lm.md`

**失败方案记录:**
1. 动态目标力注入 `target = lm - σ₀·f_total/k_base` — k_base cancel、质量比问题
2. 能量约束 C=√(2·σ₀·a·Ψ_L) — 可行但不必要，active/passive 混合，V0 藏在 compliance 中不直观

---

### Task 1: CPU 端验证——能量约束公式正确性 ✅ 已完成

**Files:** `scripts/experiment_energy_constraint.py`
7 项 CPU 测试全部通过，验证了 Millard 能量积分正确性和静态平衡 lm_eq ≈ 0.5498。
虽然新方案不再使用能量约束，但数学验证（dΨ/dlm = f_L、静态平衡点）仍是有效基础。

---

### Task 2: 实现 GPU 端 Millard 能量求值函数 ✅ 已完成

**Files:** `src/VMuscle/muscle_warp.py` (`millard_energy_eval_wp`), `tests/test_millard_gpu.py`
GPU 能量求值和力求值均已实现并通过测试（max_err < 1e-3）。
新方案主要使用 `millard_eval_wp`（力求值），能量求值可供后续被动纤维约束使用。

---

### Task 3: 实现显式主动纤维力 kernel（核心变更）

**Files:**
- Modify: `src/VMuscle/muscle_warp.py`（新增 kernel + 修改 integrate）
- Modify: `src/VMuscle/muscle_common.py`（修改 step 循环）

**Context:** 基于 `docs/notes/xpbd_fem_fiber_lm.md` 的分析，将主动力从 XPBD constraint 改为显式 FEM 节点力。三个关键设计要点：
1. Active 力不走 constraint，直接作为附加内力
2. V0 作为力的直接缩放因子（应力→力）
3. fiber stretch ratio 分母对齐：r = λ_f / λ_opt

- [ ] **Step 1: 新增 clear_force_kernel**

清零 force field，在每个 substep 开始时调用。

```python
@wp.kernel
def clear_force_kernel(force: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    force[i] = wp.vec3(0.0, 0.0, 0.0)
```

- [ ] **Step 2: 新增 accumulate_active_fiber_force_kernel**

Per-tet kernel。对每个 tet：
1. 取 activation、Dminv、V0、顶点纤维方向
2. 计算 d（平均顶点纤维方向）、w = Dminv @ d
3. 计算 Fd = Ds · w、λ_f = ||Fd||
4. 对齐: r = λ_f / λ_opt
5. 评估 f_FL(r) via `millard_eval_wp`
6. τ_act = σ₀ · a · f_FL(r)
7. n = Fd / λ_f
8. 散射节点力: f_j = -V0 · τ_act · w[j] · n（atomic add）

注意：
- w[j] 是 vec3 w 的第 j 个分量（标量），不是 vec3
- f_3 = -(f_0 + f_1 + f_2)
- 固定顶点（stopped=1）的力可以累加但不影响（integrate 中 stopped check）

- [ ] **Step 3: 修改 integrate_kernel 加入 force/mass**

```python
extacc = gravity + force[i] / mass[i]  # 加入主动力加速度
```

注意：固定顶点（stopped=1）的处理保持不变。

- [ ] **Step 4: 修改 step() 循环**

在 `muscle_common.py` 的 `step()` 中，在 integrate 前调用：
```python
self.clear_forces()
self.accumulate_active_fiber_force()
```

在 `muscle_warp.py` 中实现这两个方法，负责 launch 对应 kernel。

- [ ] **Step 5: 新增 lambda_opt 参数**

在 config JSON 中加入 `"lambda_opt": 1.0`（默认 rest=optimal）。
传入 kernel 作为参数。

- [ ] **Step 6: 从 constraint dispatch 中移除 fibermillard**

更新 `config_xpbd_millard.json`：删除 fibermillard 约束，仅保留 snh。
注意：`solve_tetfibermillard_kernel` 代码保留，仅不使用。

- [ ] **Step 7: 编译测试**

`uv run python -c "import warp as wp; wp.init(); from VMuscle.muscle_warp import accumulate_active_fiber_force_kernel; print('OK')"`

- [ ] **Step 8: Commit**

---

### Task 4: Sliding Ball 验证

**Files:**
- Modify: `examples/example_xpbd_millard_sliding_ball.py`
- Modify: `data/slidingBall/config_xpbd_millard.json`

**Context:** 删除不再需要的 CPU 平衡反求代码。显式主动力自动达到正确平衡点。

- [ ] **Step 1: 更新 sliding ball config**

移除 `fibermillard` 约束，加入 `lambda_opt`。确认 snh 配置。

- [ ] **Step 2: CPU 模式运行 sliding ball**

验证：
- lm_eq 收敛到 ~0.5498
- 球位置收敛到合理值（~0.055m × lm_eq）
- 无 NaN、无发散
- 力平衡：σ₀·a·f_FL(lm_eq)·A = m·g

- [ ] **Step 3: 删除冗余代码，清理 imports**
- [ ] **Step 4: Commit**

---

### Task 5: Simple Arm 验证

（同前，待 Task 4 通过后进行）

- [ ] **Step 1: 修改 coupled simple arm 使用新方案**
- [ ] **Step 2: CPU 模式运行，验证肘角 ≈ 88.25°**
- [ ] **Step 3: GPU 模式运行，对比 CPU**
- [ ] **Step 4: Commit**

---

### Task 6: 验证与文档

- [ ] **Step 1: 汇总 sliding ball + simple arm 结果**
- [ ] **Step 2: 创建 `docs/progress/2026-04-02-xpbd-millard-stage5.md`**
- [ ] **Step 3: Final Commit**
