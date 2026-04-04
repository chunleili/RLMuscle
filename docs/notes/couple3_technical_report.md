# XPBD-Millard 肌骨耦合仿真技术报告

> 日期: 2026-04-04
> Git tag: `couple3-bicep`

---

## 1. 项目背景与演进

本项目的目标是在真实解剖几何上实现 XPBD 肌肉仿真与刚体骨骼动力学的双向耦合。以下为近期探索路线：

| 日期 | 里程碑 | 关键结果 | 参考 |
|------|--------|---------|------|
| 03-30 | XPBD-DGF 本构模型 | 引入 TETFIBERDGF 约束，sliding ball 误差 0.8% | [note](../progress/2026-03-30-xpbd-dgf-constitutive.md) |
| 03-30 | 耦合可控性层 | `smooth_nonlinear` preset，activation dynamics | [note](../progress/2026-03-30-coupling-controllability.md) |
| 03-31 | f-L 曲线集成 | DGF force-length 驱动 XPBD stiffness | [note](../progress/2026-03-31-xpbd-dgf-fl-curve.md) |
| 03-31 | SimpleArm XPBD 耦合 | ATTACH 约束替代运动学边界，flat loop 架构 | [note](../progress/2026-03-31-simple-arm-stage3-xpbd.md) |
| 04-02 | **Millard 曲线 + TETFIBERMILLARD** | quintic Bezier → 10 次多项式闭式能量积分，SimpleArm 误差 0.10° | [note](../progress/2026-04-02-xpbd-millard-stage1-3.md) |
| 04-02 | Explicit active fiber force | 显式 Millard force 路线（Route C）验证 | [note](../progress/2026-04-02-xpbd-millard-stage5.md) |
| 04-03 | 3D 力反馈修复 | Hybrid: mesh deformation → 1D Hill → force，RMSE 1.91° | [note](../progress/2026-04-03-3d-force-feedback.md) |
| 04-03 | couple3 方案对比 | Route C 在退化 mesh 上失败，**Route A (TETFIBERMILLARD) 胜出** | [note](../progress/2026-04-03-couple3-approach-comparison.md) |
| 04-03 | 网格稳定性根因 | 退化 tet 质量比 12095:1，条件数 2689 | [note](../progress/2026-04-03-couple3-mesh-stability.md) |
| 04-03~04 | **Post-smooth + SVD repair** | 反转 tets: 480→9 (98%↓)，完全恢复 | [note](../progress/2026-04-03-couple3-post-smooth.md), [experiment](../experiments/2026-04-04-svd-repair-sweep.md) |

**当前路线**: TETFIBERMILLARD 能量约束 (Route A) + Post-smooth + SVD clamp，在 bicep 几何上经过完整验证。

## 2. 系统架构

### 2.1 总体耦合流程

```
SolverMuscleBoneCoupled.step (1 frame = 1/60 s)
├── sync bone → muscle (初始位置)
└── for substep in 1..30:
    ├── activation dynamics        # excitation → activation (一阶低通)
    ├── muscle PBD substep         # §3 详述
    ├── extract τ from attach      # ATTACH 反力 → torque
    ├── shape torque               # EMA + slew rate + clamp
    ├── bone MuJoCo substep        # joint_f = τ, 刚体积分
    └── sync bone → muscle         # 更新 attach target
```

### 2.2 力矩提取

```
τ_raw = Σ_i (r_i × F_i)
  F_i = -k_coupling · reaction_i    # attach 约束的 position correction
  r_i = target_i - pivot            # 力臂 (到肘关节)
τ_axis = τ_raw · joint_axis         # 投影到 revolute 轴
τ = shape_torque(τ_axis, ...)       # EMA + slew rate + max_torque clamp
```

### 2.3 Controllability (`smooth_nonlinear` preset)

- 一阶 activation dynamics：excitation → activation 低通滤波
- 非线性 gamma 映射
- Torque EMA 平滑 + slew rate 限制
- max_torque clamp (6.5 N·m)

## 3. Muscle PBD Substep Pipeline

```
① clear_forces()
② accumulate_active_fiber_force()     # explicit force (σ₀ > 0 时)
③ integrate()                         # x* = x + v·dt + f/m·dt²
④ clear()                             # dP = 0, dPw = 0
⑤ solve_constraints()                 # XPBD Jacobi: 11941 constraints
⑥ apply_dP()                          # x += dP / dPw
⑦ post_smooth()     ─── 5 轮 ───┐
                                 │   clear → solve(volume+ARAP+attach) → apply
                                 └── 排除 fiber 类约束
⑧ repair_inverted_tets()              # 1 轮 SVD clamp (α=0.1, σ_min=0.05)
⑨ update_velocities()                 # v = (x - x_prev) / dt
```

### 3.1 约束列表

| 约束 | 类型 | 数量 | Stiffness | 作用 |
|------|------|:---:|:---:|------|
| volume | VOLUME | 3938 | 1e10 | $C = \det(F) - 1$ |
| fiber_millard_active | TETFIBERMILLARD | 3938 | 10000 | 主动纤维收缩 (Millard f_L) |
| tetarap | TETARAP | 3938 | 1e10 | As-Rigid-As-Possible |
| muscleToBonesAttach | ATTACH | 114 | 1e10 | 表面附着 |
| muscleEndAttach × 2 | ATTACH | 13 | 1e37 | 端点刚性附着 |
| **合计** | | **11941** | | |

### 3.2 TETFIBERMILLARD 约束

基于 Millard 2012 f-L 曲线的 XPBD 能量约束：

$$C = \sqrt{2 \sigma_0 \cdot a \cdot \Psi_L(lm)}, \quad \Psi_L = \int_{lm_{min}}^{lm} f_L(lm') \, dlm'$$

- $lm = \|F \cdot d\|$：变形后纤维拉伸比
- $f_L$：5 段 quintic Bezier 样条 → 10 次多项式闭式积分
- $\sigma_0 = 300$ kPa, $a \in [0,1]$, $d$ = 材料空间纤维方向

### 3.3 Post-Smooth (⑦)

核心思想：fiber 约束的大变形由 volume/ARAP/attach 的额外迭代来修复。

每轮解 volume(3938) + ARAP(3938) + attach(127) = 8003 约束，5 轮 = **40015 constraint solves/substep**。

### 3.4 SVD Clamp Repair (⑧)

$$F = U \Sigma V^T \rightarrow \sigma_i' = \max(\sigma_i, 0.05) \rightarrow F_{target} = U\,\text{diag}(\sigma')\,V^T$$

Centroid-preserving 修正 + Jacobi 累加，$\alpha = 0.1$，仅 1 轮。

## 4. 参数配置

### Config: `data/muscle/config/bicep_fibermillard_coupled.json`

**仿真全局**

| 参数 | 值 | 说明 |
|------|:---:|------|
| dt | 1/60 s | frame 时间步 |
| num_substeps | 30 | substep 数 |
| density | 1000 kg/m³ | 肌肉密度 |
| veldamping | 0.02 | 速度阻尼 |
| gravity | 0 | 无重力 |
| fiber_stiffness_scale | 10000 | fiber 约束 stiffness 缩放 |

**TETFIBERMILLARD**

| 参数 | 值 |
|------|:---:|
| stiffness | 10000 |
| sigma0 | 300 kPa |
| contraction_factor | 0.4 |

**网格质量**

| 参数 | 值 | 说明 |
|------|:---:|------|
| post_smooth_iters | **5** | non-fiber 约束平滑轮数 |
| repair_alpha | **0.1** | SVD 混合系数 |
| repair_iters | **1** | SVD 迭代数 |
| repair_sigma_min | **0.05** | singular value 下限 |

**骨骼耦合**

| 参数 | 值 |
|------|:---:|
| k_coupling | 24000 (default preset) |
| max_torque | 6.5 (default preset) |
| joint_friction | 0.05 |
| preset | smooth_nonlinear |

## 5. 方案演进与否决

| 方案 | Peak 反转 | 力/角度 | 状态 | 原因 |
|------|:---:|:---:|:---:|------|
| Explicit force (Route C) | 爆炸/66 | 0.14 N·m | **否决** | 退化 mesh 上 force clamping 导致力太弱 |
| Hybrid (TETFIBERNORM + explicit) | 130 | 0.21 N·m | **否决** | 两种力互相竞争 |
| Remesh | — | — | **否决** | 拓扑变化破坏 mask/attach 配置 |
| Centroid repair | 360 | — | **否决** | trap state，无法恢复 |
| SVD repair alone | 260 | — | **否决** | trap state |
| Fiber kernel skip (det(F)<thr) | 59 | — | **否决** | 问题在邻居挤压而非自身 |
| Freeze near-inverted | — | — | **否决** | 阻碍恢复 |
| **TETFIBERMILLARD + smooth=5 + SVD i=1** | **9** | **6.5 N·m / -53°** | **采用** | 98%↓反转，完全恢复 |

## 6. 性能

| 配置 | ms/step | Overhead | Peak 反转 |
|------|:---:|:---:|:---:|
| 无修复 | 156 | 基准 | 480 (12.2%) |
| + smooth=5 | 506 | +224% | 10 (0.25%) |
| **+ smooth=5 + SVD i=1** | **506** | **+224%** | **9 (0.23%)** |

SVD i=1 零额外开销。i>1 性价比极差（详见 [参数扫描](../experiments/2026-04-04-svd-repair-sweep.md)）。

## 7. 最终结果

| 指标 | Baseline | 最终 | 改善 |
|------|:---:|:---:|:---:|
| Peak 反转 tets | 480 (12.2%) | **9 (0.23%)** | 98%↓ |
| Worst det(F) | -927 | **-0.54** | 99.9%↓ |
| 去激活恢复 | 19 永久 | **0-1** | 完全恢复 |
| 峰值 torque | 6.5 N·m | **6.5 N·m** | 保持 |
| 峰值关节角度 | -53° | **-53°** | 保持 |

![SVD 参数扫描](../imgs/couple3/sweep_svd_params.png)

## 8. 复现

```bash
# 运行 couple3 仿真
uv run python examples/example_couple3.py --auto --steps 300
uv run python examples/example_couple3.py --auto --steps 300 --no-usd

# 绘制曲线图
uv run python scripts/run_couple3_curves.py

# SVD 参数扫描
uv run python scripts/experiments/sweep_post_smooth.py

# SimpleArm Millard 对比 (验证 Millard 曲线精度)
uv run python scripts/run_simple_arm_comparison.py --mode xpbd-millard
```

### 输出文件

| 文件 | 说明 |
|------|------|
| `output/example_couple3.anim.usd` | USD 动画 |
| `output/couple3_curves.png` | 曲线图 |
| `output/sweep_post_smooth.png` | 参数扫描图 |

## 9. 代码索引

| 文件 | 关键内容 |
|------|---------|
| `data/muscle/config/bicep_fibermillard_coupled.json` | couple3 全部参数 |
| `src/VMuscle/muscle_warp.py` | MuscleSim: 约束 kernels, post_smooth, SVD repair |
| `src/VMuscle/muscle_common.py` | substep 循环 |
| `src/VMuscle/solver_muscle_bone_coupled_warp.py` | 肌骨耦合 solver, torque 提取 |
| `src/VMuscle/controllability.py` | activation dynamics, torque shaping, presets |
| `src/VMuscle/millard_curves.py` | Millard 2012 f_L/f_PE quintic Bezier 实现 |
| `examples/example_couple3.py` | example 入口 |
| `scripts/run_couple3_curves.py` | 曲线绘图 |
| `scripts/experiments/sweep_post_smooth.py` | 参数扫描 |
