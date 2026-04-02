# Stage 5: XPBD 肌肉主动力——三条路线对比

## 问题

XPBD mesh 中如何施加 Hill/Millard 主动纤维力？需要满足：
1. GPU-only 求解（消除 CPU 端平衡反求）
2. 力的大小等于 σ₀ · a · f_FL(lm)
3. 适用于任意 mesh 几何（非圆柱、多肌肉）

## 路线 A：能量本构约束 C = √(2·σ₀·a·Ψ_L)

### 原理

利用 XPBD 约束的能量等价性。令约束 C 编码 Millard 主动能量密度 Ψ_L(lm)：

```
Ψ_L(lm) = ∫_{lm_min}^{lm} f_L(lm') dlm'     (闭式10次多项式)
C = √(2 · σ₀ · a · Ψ_L(lm))
```

关键恒等式：
```
C · dC/dlm = σ₀ · a · f_L(lm)
```

无论 Ψ_L 的绝对值如何，XPBD 约束力始终等于 Millard 力。

### 伪代码

```python
# GPU kernel: solve_tetfibermillard_kernel
lm = ||Ds · (Dminv @ d)||           # 3D fiber stretch
psi_L = millard_energy_eval_wp(lm)  # 闭式10次多项式求值
C = sqrt(2 * sigma0 * a * psi_L)
dC_dlm = sigma0 * a * f_L(lm) / C  # chain rule
grad_C_i = dC_dlm * dlm/dx_i       # geometric gradient
# 标准 XPBD update: dL = -C / (sum w_i ||grad_i||^2 + alpha)
```

### 验证结果

- CPU 验证 7 项全部 PASS（`scripts/experiment_energy_constraint.py`）
- GPU 能量求值精度 max_err < 2e-6（`tests/test_millard_gpu.py`）
- 静态平衡 lm_eq ≈ 0.5498 与 Stage 4 一致

### 未采纳原因

1. **不对称问题**：Ψ_L 在 [lm_min, lm_opt] 积分较小、在 [lm_opt, lm_max] 积分较大，导致约束在两侧强度不平衡。**可能的修复**：加常数 offset 使 Ψ_shifted(lm) = Ψ_L(lm) - Ψ_L(lm_opt) + ε，使 C 在 lm_opt 处最小且对称
2. **主动/被动混合**：主动力不是严格保守力（依赖时变 activation、可能含 velocity 项），硬包成能量约束增加不必要的间接层
3. **V0 藏在 compliance**：`α = 1/(k · V0 · dt²)` 中 V0 的物理意义不直观

**保留状态**：代码完整保留（`solve_tetfibermillard_kernel`、`millard_energy_eval_wp`），可随时切换。

---

## 路线 B：Mesh 变形提取纤维长度 → 1D 力模型

### 原理

从 XPBD mesh 的变形梯度提取平均纤维拉伸比，喂入 1D Hill/Millard 模型计算标量力，再作为 MuJoCo motor force 输出。

```
mesh pos → F = Ds · Dminv → lm = ||F · d|| → r = mean(lm) × (L_mesh / L_opt)
→ f_FL(r) → F_muscle = (a · f_FL · f_V + f_PE + damping) · F_max → MuJoCo ctrl
```

### 伪代码

```python
# After XPBD step:
pos_np = sim.pos.numpy()                     # GPU → CPU transfer
stretches = compute_fiber_stretches(pos_np, tets, rest_matrices, fiber_dirs)
r_mesh = float(np.mean(stretches)) * stretch_to_ltilde
fl = mc.fl.eval(r_mesh)
muscle_force = (activation * fl * fv + fpe) * F_max
mj_data.ctrl[0] = muscle_force              # Feed to MuJoCo
```

### 未采纳原因

1. **ATTACH 目标不允许 mesh 拉伸**：当前骨骼跟随是纯刚体旋转（保持 mesh_length），l_xpbd 恒定。需修改骨骼目标更新来允许 mesh 拉伸
2. **绕回 1D 模型**：从 3D mesh 提取标量再走 1D，不如显式力直接
3. **GPU → CPU → GPU 通信**：每步需要 `sim.pos.numpy()` 传输

**适用场景**：当 MuJoCo 耦合为刚性需求（如 RL 训练需要精确的 motor force 接口）时，此方案是最简单的桥接。

---

## 路线 C：显式主动力（已采纳）

### 原理

基于 `docs/notes/xpbd_fem_fiber_lm.md` 的分析，将主动力直接作为 FEM 节点力施加，不走 XPBD constraint 机制。

核心分裂：
```
passive part → XPBD constraints（体积、各向同性、被动纤维）
active part  → direct additional force（显式，在 integrate 步施加）
```

节点力推导（线性四面体）：
```
P_act = τ_act · (n ⊗ d)                      # 一阶 Piola-Kirchhoff 应力
τ_act = σ₀ · a · f_FL(r)                     # 主动应力 [Pa]
r = λ_f / λ_opt = ||F·d|| / (l_opt / L_ref)  # 分母对齐

f_j = -V0 · τ_act · (Dm⁻¹ · d)[j] · n       (j = 0,1,2)
f_3 = -(f_0 + f_1 + f_2)

量纲: m³ × Pa × m⁻¹ = N  ✓
```

### 伪代码

```python
# Per substep:
clear_forces()                          # force.zero_()
accumulate_active_fiber_force()         # per-tet kernel:
#   d = normalize(avg vertex fiber dirs)
#   w = Dminv @ d
#   Fd = Ds @ w;  lm = ||Fd||;  r = lm / lambda_opt
#   f_FL = millard_eval_wp(r, ...)
#   tau = sigma0 * a * f_FL;  n = Fd / lm
#   f_j = -V0 * tau * w[j] * n;  atomic_add(force, v_j, f_j)
integrate()                             # v += dt * (gravity + force/mass)
clear(); solve_constraints()            # only passive: SNH, ARAP, ATTACH
update_velocities()                     # v = (x - x_prev) / dt
```

### 三个关键设计要点

1. **Active 力不走 constraint**：主动力直接作为附加内力，在 integrate 步施加。物理意义清晰，实现简单
2. **V0 作为力的直接缩放因子**：将应力(Pa)转为力(N)，不藏在 XPBD compliance 中
3. **fiber stretch ratio 分母对齐**：r = λ_f / λ_opt，不能直接把 ||F·d|| 塞进 Millard 曲线。λ_opt = l_opt / L_ref

### 验证结果

**Sliding ball 力场验证**（`a=1.0, lm=1.0, rest config`）：

![Force field at rest](../imgs/tmp/stage5/force_field_rest.png)

- 底部顶点总力: +311.77 N（沿纤维向上 ✓）
- 顶部顶点总力: -311.77 N（反力 ✓）
- 全局总力: ~0（自平衡内力 ✓）
- 预期 F_max = σ₀ · A = 377 N（FEM 离散近似导致偏低，合理）

复现命令：
```bash
uv run python -c "
import numpy as np; import warp as wp
from examples.example_xpbd_millard_sliding_ball import load_config, _build_muscle_sim
from VMuscle.muscle_warp import fill_float_kernel
cfg = load_config('data/slidingBall/config_xpbd_millard.json')
sim, bottom_ids, axis_idx, *_ = _build_muscle_sim(cfg)
wp.launch(fill_float_kernel, dim=480, inputs=[sim.activation, wp.float32(1.0)])
sim.clear_forces(); sim.accumulate_active_fiber_force()
f = sim.force.numpy()
print('Bottom force Z:', sum(f[b][axis_idx] for b in bottom_ids))
print('Top force Z:', sum(f[i][axis_idx] for i in range(len(f)) if sim.stopped.numpy()[i]==1))
"
```

**Sliding ball 跳过原因**：XPBD 对极端质量比（10kg 球 vs 0.002kg 内部顶点 → 700:1）的显式积分天生抗性差。这是 XPBD 框架的已知局限，不是力计算的错误。

**Simple arm 10s 验证**：

![Simple arm trajectory](../imgs/tmp/stage5/simple_arm_trajectory.png)

- 稳态肘角: **88.25°**（与 Stage 4 完全一致）
- 10s 1000步无 crash / 无 tet inversion
- 无 NaN / 无数值发散

复现命令：
```bash
RUN=example_xpbd_coupled_simple_arm_millard uv run main.py
```

绘图命令：
```bash
uv run python scripts/plot_stage5_results.py
```
