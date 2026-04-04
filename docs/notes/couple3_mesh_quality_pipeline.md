# couple3 网格质量 Pipeline 技术报告

> 日期: 2026-04-04
> Git tag: `couple3-bicep`
> 关联文档:
> - [Post-smooth 进展](../progress/2026-04-03-couple3-post-smooth.md) — 开发过程与方法对比
> - [SVD repair 参数扫描](../experiments/2026-04-04-svd-repair-sweep.md) — 完整参数扫描数据
> - [Mesh 稳定性根因分析](../progress/2026-04-03-couple3-mesh-stability.md) — 问题根因
> - [couple3 初始实现](../progress/2026-04-03-example-couple3.md) — couple3 设计
> - [XPBD fiber 公式推导](xpbd_fem_fiber_lm.md) — XPBD 约束公式

## 1. 问题背景

Bicep mesh（来自 Houdini，1251 vertices，3938 tets）在 TETFIBERMILLARD 约束驱动收缩时，由于退化四面体（质量比 12095:1、条件数最高 2689）导致大量网格反转。

无修复 baseline: **480 反转 tets (12.2%)**，worst det(F) = -927，去激活后 19 个永久反转。

## 2. Pipeline 算法

每个 frame 包含 `num_substeps=30` 个子步。每个子步的完整 pipeline：

```
for each substep:
    ① clear_forces()
    ② accumulate_active_fiber_force()       # explicit fiber force (σ₀ > 0 时)
    ③ integrate()                            # x* = x + v·dt + f/m·dt²
    ④ clear()                                # dP = 0, dPw = 0
    ⑤ solve_constraints()                    # XPBD Jacobi: volume + fiber + ARAP + attach
    ⑥ apply_dP()                             # x = x + dP / dPw
    ⑦ post_smooth()                          # 5 轮 non-fiber 约束 Jacobi 平滑
    ⑧ freeze_near_inverted()                 # (disabled, threshold=0)
    ⑨ repair_inverted_tets()                 # 1 轮 SVD clamp + Jacobi apply
    ⑩ update_velocities()                    # v = (x - x_prev) / dt
```

### ⑤ solve_constraints — XPBD 约束求解

约束类型及参数：

| 约束 | 类型 | 数量 | Stiffness | 作用 |
|------|------|:---:|:---:|------|
| volume | VOLUME | 3938 | 1e10 | 体积保持 |
| fiber_millard_active | TETFIBERMILLARD | 3938 | 10000 | 主动纤维收缩（σ₀=300kPa, Millard f_L 曲线） |
| tetarap | TETARAP | 3938 | 1e10 | As-Rigid-As-Possible 形状保持 |
| muscleToBonesAttach | ATTACH | 114 | 1e10 | 肌肉-骨骼附着 |
| muscleEndAttachProximal | ATTACH | 8 | 1e37 | 近端刚性附着 |
| muscleEndAttachDistal | ATTACH | 5 | 1e37 | 远端刚性附着 |
| **合计** | | **11941** | | |

XPBD 更新规则（Jacobi 模式）：

$$\Delta \lambda = \frac{-C - \tilde{\alpha} \lambda}{\sum_i w_i \|\nabla_{x_i} C\|^2 + \tilde{\alpha}}$$

$$\Delta x_i = w_i \nabla_{x_i} C \cdot \Delta\lambda$$

其中 $\tilde{\alpha} = \alpha / (dt)^2$，$\alpha = 1/\text{stiffness}$。

### ⑦ post_smooth — 非 fiber 约束后平滑

**核心思想**：fiber 约束驱动收缩导致局部大变形，volume/ARAP/attach 负责维持网格质量。Post-smooth 给予这些约束额外迭代来修复被 fiber 力扰乱的网格。

```python
_FIBER_TYPES = {TETFIBERNORM, TETFIBERDGF, TETFIBERMILLARD}

def post_smooth(self):
    for _ in range(post_smooth_iters):  # 5 次
        clear dP, dPw
        solve volume + ARAP + attach constraints (Jacobi)
        x += dP / dPw
```

参与 smooth 的约束：volume (3938) + tetarap (3938) + attach (127) = **8003 constraints × 5 iters = 40015 constraint solves/substep**。

### ⑨ repair_inverted_tets — SVD Clamp

对每个 tet 计算变形梯度 $F = D_s \cdot D_m^{-1}$，进行 signed SVD 分解 $F = U \Sigma V^T$：

1. 若所有 $\sigma_i \geq \sigma_{min}$ (0.05)，跳过
2. Clamp: $\sigma_i' = \max(\sigma_i, \sigma_{min})$
3. 重建目标: $F_{target} = U \cdot \text{diag}(\sigma') \cdot V^T$
4. 计算目标边矩阵: $D_{s,target} = F_{target} \cdot D_m$
5. 算每个顶点修正 $\delta_i$，centroid-preserving 约束使 $\sum \delta_i = 0$
6. Jacobi 累加: $dP_i \mathrel{+}= \delta_i$，$dPw_i \mathrel{+}= 1$
7. Apply: $x_i \mathrel{+}= \alpha \cdot dP_i / dPw_i$

仅 1 轮，$\alpha = 0.1$，温和修正避免 trap state。

## 3. 参数配置

### 全局参数

| 参数 | 值 | 说明 |
|------|:---:|------|
| dt | 1/60 s | frame 时间步 |
| num_substeps | 30 | 每 frame 子步数 |
| density | 1000 kg/m³ | 肌肉密度 |
| veldamping | 0.02 | 速度阻尼 |
| gravity | 0 | 无重力 |

### 网格质量参数

| 参数 | 值 | 说明 |
|------|:---:|------|
| post_smooth_iters | **5** | 非 fiber 约束后平滑迭代数 |
| repair_alpha | **0.1** | SVD clamp 混合系数 |
| repair_iters | **1** | SVD clamp 迭代数 |
| repair_sigma_min | **0.05** | SVD singular value 下限 |

### 无效方案的参数（默认 disabled）

| 参数 | 默认值 | 说明 |
|------|:---:|------|
| fiber_skip_detF | 0 | fiber kernel 内跳过 det(F)<threshold 的 tet（无效） |
| freeze_detF_threshold | 0 | 冻结近反转 tet 顶点到 pprev（无效） |

## 4. 性能分析

| 方案 | ms/step | Overhead | Peak 反转 |
|------|:---:|:---:|:---:|
| 无修复 | 156 | 基准 | 480 (12.2%) |
| +smooth=5 | 506 | +224% | 10 (0.25%) |
| +smooth=5 +SVD(i=1) | 506 | +224% | **9 (0.23%)** |
| +smooth=5 +SVD(i=5) | 692 | +344% | 17 (0.43%) |
| +smooth=5 +SVD(i=10) | 926 | +493% | 18 (0.46%) |

- Overhead 全部来自 post_smooth（5 轮 × 8003 约束）
- SVD i=1 在 smooth=5 基础上增加 <1ms（零额外开销）
- SVD i>1 性价比极差：开销线性增长但效果反而更差（多轮 SVD 扰动邻居形成振荡）

## 5. 最终结果

![SVD 参数扫描](../imgs/couple3/sweep_svd_params.png)

| 指标 | Baseline | 最终方案 | 改善 |
|------|:---:|:---:|:---:|
| Peak 反转 tets | 480 (12.2%) | **9 (0.23%)** | **98%↓** |
| Worst det(F) | -927 | **-0.54** | **99.9%↓** |
| 去激活恢复 | 19 永久 | **0-1** | 完全恢复 |
| 峰值 torque | 6.2 N·m | **6.5 N·m** | 保持 |
| 峰值关节角度 | — | **-0.92 rad (-53°)** | — |

## 6. 复现

```bash
# 运行仿真（300 步，带 USD 输出）
uv run python examples/example_couple3.py --auto --steps 300

# 运行仿真（无 USD，仅终端输出）
uv run python examples/example_couple3.py --auto --steps 300 --no-usd

# 绘制曲线图（角度/激活/力矩/反转 vs 时间）
uv run python scripts/plot_couple3_curves.py

# 参数扫描
uv run python scripts/sweep_post_smooth.py
```

### 输出文件

| 文件 | 说明 |
|------|------|
| `output/example_couple3.anim.usd` | USD 动画 |
| `output/couple3_curves.png` | 曲线图 |
| `output/sweep_post_smooth.png` | 参数扫描图 |

## 7. 代码文件索引

| 文件 | 关键内容 |
|------|---------|
| `data/muscle/config/bicep_fibermillard_coupled.json` | 全部参数配置 |
| `src/VMuscle/muscle_warp.py` | `post_smooth()`, `freeze_near_inverted()`, `repair_inverted_tets()`, SVD kernels |
| `src/VMuscle/muscle_common.py` | substep 循环（步骤 ①-⑩） |
| `src/VMuscle/solver_muscle_bone_coupled_warp.py` | coupled 版 substep 循环 |
| `examples/example_couple3.py` | 完整 example 入口 |
| `scripts/plot_couple3_curves.py` | 曲线绘图 |
| `scripts/sweep_post_smooth.py` | 参数扫描 |
