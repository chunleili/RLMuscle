# XPBD-Millard Stage 5: 显式主动力 + 被动 XPBD 约束

> 日期: 2026-04-02
> 分支: xpbd-millard-stage5
> 前置: Stage 1-4 完成（Millard 曲线模块、TETFIBERMILLARD 约束、sliding ball + simple arm 验证）
> **重要参考**: `docs/notes/xpbd_fem_fiber_lm.md`

## 问题

1. **Mesh 与力分离**：XPBD mesh 仅用于可视化，力输出走独立的 1D Hill 模型
2. **CPU-GPU 通信瓶颈**：每步需 CPU 端反求 Millard 平衡点 → contraction_factor → GPU kernel 更新
3. **两端约束 tet 翻转**：target_stretch 驱动在 isometric 场景下产生过大 C 值 → mesh collapse
4. **不可泛化**：1D 模型假设均匀纤维方向和截面积

## 设计约束

- **稳定性最优先**：必须容忍任意复杂 mesh、任意激活输入不 crash（后续用于 RL）
- **可泛化**：设计必须适用于非圆柱几何、多肌肉系统
- **性能**：消除 CPU-GPU 通信，GPU-only 求解

## 失败方案记录

### 方案 A: 动态目标力注入 — 已失败

**思路**：`target = lm - σ₀·f_total/k_base`
**失败原因**：k_base cancel、质量比问题无法解决

### 方案 B: 能量约束 C = √(2·σ₀·a·Ψ_L) — 已弃用

**思路**：将主动力包装为能量约束，通过 XPBD constraint 机制求解
**弃用原因**（基于 `docs/notes/xpbd_fem_fiber_lm.md` 分析）：
1. 主动力不是严格保守力（依赖 activation 时变、可能含 velocity 项），硬包成能量约束增加不必要的间接层
2. V0 藏在 XPBD compliance α 中，物理意义不直观
3. 不区分 active/passive，混在一个 constraint 里难以调参
4. **直接作为附加力更自然、更简单**

## 方案选择

### 方案 C: 显式主动力 + 被动 XPBD 约束（Active/Passive Split）

**核心原则**（来自 `docs/notes/xpbd_fem_fiber_lm.md`）：
```
passive part → XPBD constraints (体积、基质、被动纤维)
active part  → direct additional force (显式/滞后显式)
```

**三个关键设计要点**：

1. **Active 力不走 constraint**：直接作为附加内力施加在 integrate 步
2. **V0 作为力的直接缩放因子**：f = V0 · τ · scalar · n，量纲 m³ × Pa × m⁻¹ = N
3. **fiber stretch ratio 分母对齐**：`r = λ_f / λ_opt`，不能直接把 `||F·d||` 塞进 Millard 曲线

---

## 数学推导

### 主动力的 FEM 离散化

对每个四面体单元 e，主动纤维应力（一阶 Piola-Kirchhoff）：
```
P_act = τ_act · (n ⊗ d)
τ_act = σ₀ · a · f_FL(r)
```

其中：
- `σ₀`：峰值等距应力 [Pa]
- `a`：激活水平 [0, 1]
- `f_FL(r)`：Millard 主动力-长度曲线（无量纲）
- `n = F·d / ||F·d||`：当前纤维单位方向
- `d`：材料空间纤维方向（单位向量）

### Stretch ratio 对齐

连续体 fiber stretch：
```
λ_f = ||F·d|| = l_f / L_ref
```

Millard 曲线输入（归一化到最优长度）：
```
r = l_f / l_opt = λ_f / λ_opt
λ_opt = l_opt / L_ref
```

**如果 rest config = optimal length**（如 sliding ball）：λ_opt = 1.0，r = λ_f
**一般情况必须显式传入 λ_opt**。

### 节点力公式

线性四面体形函数梯度：
```
∇_X N_j = column j of Dm⁻ᵀ = row j of Dm⁻¹    (j = 0, 1, 2)
∇_X N_3 = -(∇_X N_0 + ∇_X N_1 + ∇_X N_2)
```

节点内力：
```
f_j = -V0 · P_act · ∇_X N_j = -V0 · τ_act · n · (d · ∇_X N_j)
```

利用预计算向量 `w = Dm⁻¹ · d`（已存在于 restvector[0:3]）：
```
d · ∇_X N_j = (Dm⁻¹ · d)[j] = w[j]
```

因此：
```
f_j = -V0 · τ_act · w[j] · n        (j = 0, 1, 2)
f_3 = -(f_0 + f_1 + f_2)

量纲验证: m³ × Pa × m⁻¹ × 1 = N  ✓
```

### 实现中的简化

`F·d` 可直接用已有的 `Ds · w` 计算（w = Dm⁻¹·d）：
```
Fd = Ds · (Dm⁻¹ · d) = Ds · w = (Ds @ Dminv) @ d = F @ d
λ_f = ||Fd||
n = Fd / λ_f
```

---

## 架构设计

### 1. 新增 accumulate_active_fiber_force_kernel（per-tet）

**文件**: `src/VMuscle/muscle_warp.py`

```python
@wp.kernel
def accumulate_active_fiber_force_kernel(
    tet_indices: wp.array(dtype=wp.vec4i),
    pos: wp.array(dtype=wp.vec3),
    force: wp.array(dtype=wp.vec3),         # output: atomic add
    rest_matrix: wp.array(dtype=wp.mat33),  # Dminv per tet
    rest_volume: wp.array(dtype=wp.float32),# V0 per tet
    v_fiber_dir: wp.array(dtype=wp.vec3),   # per-vertex fiber dir
    activation: wp.array(dtype=wp.float32), # per-tet activation
    stopped: wp.array(dtype=wp.int32),      # fixed vertex flags
    sigma0: float,
    lambda_opt: float,
    # Millard f_L curve data (force only, not energy)
    fl_x_coeffs, fl_y_coeffs, fl_seg_bounds, fl_n_seg,
    fl_x_lo, fl_x_hi, fl_y_lo, fl_y_hi, fl_dydx_lo, fl_dydx_hi,
):
    tid = wp.tid()
    pts = tet_indices[tid]
    acti = activation[tid]
    if acti < 1e-6:
        return

    # Ds = [p0-p3, p1-p3, p2-p3]
    p0 = pos[pts[0]]; p1 = pos[pts[1]]; p2 = pos[pts[2]]; p3 = pos[pts[3]]
    c0 = p0 - p3; c1 = p1 - p3; c2 = p2 - p3
    _Ds = mat33(c0, c1, c2)  # column-major

    Dminv = rest_matrix[tid]
    V0 = rest_volume[tid]

    # Per-tet fiber direction (average vertex dirs)
    d = normalize(v_fiber_dir[pts[0]] + v_fiber_dir[pts[1]]
                + v_fiber_dir[pts[2]] + v_fiber_dir[pts[3]])

    # w = Dminv @ d (precomputable, but compute on-the-fly for simplicity)
    w = Dminv * d

    # F @ d = Ds @ w
    Fd = _Ds * w
    lm_raw = length(Fd)
    if lm_raw < 1e-8:
        return

    # Stretch ratio alignment
    r = lm_raw / lambda_opt

    # Evaluate Millard f_FL(r)
    f_FL_val = millard_eval_wp(r, ...)

    # Active stress
    tau_act = sigma0 * acti * f_FL_val

    # Current fiber direction
    n = Fd / lm_raw

    # Scatter forces to vertices
    s = -V0 * tau_act
    f0 = s * w[0] * n
    f1 = s * w[1] * n
    f2 = s * w[2] * n
    f3 = -(f0 + f1 + f2)

    # Skip fixed vertices (stopped[i] == 1)
    wp.atomic_add(force, pts[0], f0)
    wp.atomic_add(force, pts[1], f1)
    wp.atomic_add(force, pts[2], f2)
    wp.atomic_add(force, pts[3], f3)
```

### 2. 修改 integrate_kernel

**文件**: `src/VMuscle/muscle_warp.py`

```python
@wp.kernel
def integrate_kernel(
    pos, pprev, vel, force, mass, stopped,
    gravity, veldamping, dt,
):
    i = wp.tid()
    if stopped[i] != 0:
        pprev[i] = pos[i]
        vel[i] = wp.vec3(0.0, 0.0, 0.0)
        return
    extacc = wp.vec3(0.0, gravity, 0.0)
    # Add active fiber force / mass
    m = mass[i]
    if m > 1e-12:
        extacc = extacc + force[i] / m
    pprev[i] = pos[i]
    v = (1.0 - veldamping) * vel[i] + dt * extacc
    vel[i] = v
    pos[i] = pos[i] + dt * v
```

### 3. 修改仿真循环

**文件**: `src/VMuscle/muscle_common.py`

```python
def step(self):
    self.update_attach_targets()
    for _ in range(self.cfg.num_substeps):
        self.clear_forces()                    # NEW: zero force field
        self.accumulate_active_fiber_force()   # NEW: explicit active force
        self.integrate()                       # MODIFIED
        self.clear()                           # existing
        self.solve_constraints()               # Only passive: SNH
        if self.use_jacobi:
            self.apply_dP()
        self.update_velocities()
```

### 4. Constraint 变更

- **TETFIBERMILLARD**: 从 constraint dispatch 中移除（不再走 XPBD）
- **SNH**: 保留，处理体积约束
- **可选**: 后续可添加被动纤维 XPBD 约束 `C_fib_passive`

### 5. 配置变更

**文件**: `data/slidingBall/config_xpbd_millard.json`

```json
{
  "muscle": {
    "sigma0": 300000.0,
    "lambda_opt": 1.0
  },
  "constraints": [
    {
      "type": "snh",
      "mu": 0.0,
      "lam": 10000.0,
      "dampingratio": 0.01
    }
  ]
}
```

注意：移除了 `fibermillard` 约束，主动力改由 `accumulate_active_fiber_force` 处理。

---

## 稳定性分析

### 显式积分稳定性

主动力显式处理，每个 substep 的 dt_sub = dt / num_substeps ≈ 0.0167/40 = 4.17e-4s。
对于 σ₀=300kPa、f_FL ≤ 1.0 的力范围，显式积分在此 dt 下应稳定。

如不稳定，可选方案：
1. 增加 substeps
2. 对主动力做半隐式处理（冻结 activation 后，主动力仅依赖 λ_f，可用 Newton 隐式化）

### 被动约束

SNH (λ=10000) 提供体积保持。如需额外纤维方向刚度，可后续添加被动纤维约束。

### RL 鲁棒性

- activation = 0 → f_act = 0，不施加任何力（正确）
- activation ∈ [0, 1] → τ_act ∝ a，有界且连续
- 力方向始终沿 n（当前纤维方向），物理合理

---

## 关键文件清单

| 操作 | 文件 | 改动量 |
|------|------|--------|
| 修改 | `src/VMuscle/muscle_warp.py` | ~80 行（新 kernel + 修改 integrate） |
| 修改 | `src/VMuscle/muscle_common.py` | ~10 行（修改 step 循环） |
| 修改 | `data/slidingBall/config_xpbd_millard.json` | 移除 fibermillard 约束 |
| 修改 | `examples/example_xpbd_millard_sliding_ball.py` | ~20 行（清理） |
| 不变 | `src/VMuscle/millard_curves.py` | f_L 力求值已实现 |
| 不变 | `src/VMuscle/constraints.py` | fibermillard 保留代码但不使用 |
