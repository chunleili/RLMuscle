# Mesh Quality Improvement Experiments

> 相关文档:
> - [初始实现](2026-04-03-example-couple3.md) — couple3 设计与初始验证
> - [Mesh 稳定性调查](2026-04-03-couple3-mesh-stability.md) — 退化 tet 根因分析
> - [方法对比实验](2026-04-03-couple3-approach-comparison.md) — 最终采用 TETFIBERMILLARD + k_coupling=100k

## 目的

调查并改善 XPBD-Millard bicep 仿真中的网格质量。基线：~75 个反转 tets（1.9%），worst det(F)=-14.2。

## 方案对比

| 方案 | max_inv | worst_J | 评价 |
|------|---------|---------|------|
| **Baseline** (damping=0.1, 无 SNH) | 75 | -14.20 | 基线 |
| A: 高 damping (0.3) | 155 | -60.76 | **更差** — damping 抑制约束校正 |
| A: 高 damping (0.5) | 207 | -64.11 | **大幅恶化** |
| A: 高 damping (0.8) | 259 | -73.05 | **最差** |
| B1: TETSNH 替换 ARAP | 302 | -8035 | **爆炸** — TETARAP 不可替代 |
| B2: TETSNH + ARAP | 91 | -22.10 | 无改善 |
| B3: TETSNH vol-only | 963 | -1863 | **严重恶化** |
| C: Laplacian α=0.1 1pass | 1708 | -1474 | **全局 smooth 破坏约束** |
| C: Laplacian α=0.1 3pass | 1997 | -408 | 更多 pass 更差 |
| **Targeted repair α=0.1 2iter** | **68** | **-1.34** | **最佳！worst_J 改善 10x** |
| Targeted repair α=0.2 3iter | 125 | -1.68 | 稍激进，后期退化 |
| Targeted repair α=0.4 5iter | 516 | -2.62 | 过于激进 |

## 关键发现

### 1. 反转 tets 是确定性的
Frames 15-25: always=94, ever=94。永远是同一批 tets 反转。

### 2. 反转 tets 与退化网格的关系
- 反转 tet median rest volume = 3.0e-9 m³
- 正常 tet median rest volume = 6.4e-8 m³（21x 更大）
- 最小 5% 体积的 tets 中 12% 会反转
- 但条件数不极端（median=10.2, max=64.3）

### 3. 各方案分析
- **Damping**: XPBD damping 增大 → gamma 项增大 → 约束校正被速度惩罚减弱 → 更差
- **TETSNH**: SNH 的耦合 vol+dev 约束与 TETARAP 重叠，不如 TETARAP 对此网格有效
- **Laplacian smoothing**: 破坏约束求解器的精心布局，全局 smooth 不可行
- **Targeted repair**: 只修复反转/近反转 tets，不影响其余网格，效果最佳

## 实现

### `_repair_inverted_tets_kernel`（Warp GPU kernel）
- 遍历每个 tet，计算 det(F)
- 若 J < 0.01（反转或接近反转），将 4 个顶点向重心移动 α
- 在 `solve_constraints()` 之后、`update_velocities()` 之前调用
- Config 参数：`repair_alpha`（默认 0），`repair_iters`（默认 0）

### 推荐配置
```json
{
  "repair_alpha": 0.1,
  "repair_iters": 2
}
```

## 耦合仿真验证（300 steps, auto activation）

使用 repair + max_accel=20 + passive fiber + substeps=30：

| Step | Activation | Joint Angle | Torque | Inverted Tets |
|------|-----------|-------------|--------|---------------|
| 100 | 1.00 | -0.0003 rad | 0.81 N·m | 528 |
| 150 | 1.00 | -0.062 rad (-3.5°) | 3.34 N·m | 525 |
| 200 | 0.70 | -0.157 rad (-9.0°) | 0.41 N·m | 537 |
| 300 | 0.00 | -0.155 rad (-8.9°) | 0.00 N·m | 17 |

- 仿真全程稳定，无 crash
- 产生有意义的 torque（最高 3.3 N·m）和屈曲（~9°）
- Activation 期间 ~525 反转 tets（13%），去激活后恢复到 17（0.4%）
- USD 输出正常

## 下一步

1. mesh 预处理：remesh 去除退化 tets 是根本解决方案
2. 参数调优：max_accel、repair_alpha 的最优平衡
3. 考虑 TETFIBERMILLARD 约束（Route A）作为更稳定的替代方案
