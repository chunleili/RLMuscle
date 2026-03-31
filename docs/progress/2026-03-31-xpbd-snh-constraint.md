# XPBD Stable Neo-Hookean 约束实现

日期：2026-03-31

## 概述

在 XPBD 求解器中实现了 Stable Neo-Hookean (SNH) 超弹性约束，作为肌肉组织的被动弹性模型。

## 实现

### 公式 (Macklin XPBD SNH)

将 SNH 能量分解为两个 XPBD 约束：

- **Deviatoric (形变)**: C_D = ||F||_F - √3, compliance = 1/(μ·V·dt²)
- **Volumetric (体积)**: C_H = J - α, compliance = 1/(λ·V·dt²)
  - α = 1 + μ/(λ+μ) (SNH 稳定性偏移)

使用耦合 2×2 线性求解同时更新两个 Lagrange 乘子（参考 pbd_constraints.cl 的 `tetARAPCoupledUpdateXPBD`）。

### 代码改动

- `src/VMuscle/constraints.py`: 新增 `TETSNH` 类型、`create_tet_snh_constraint()` 方法
- `src/VMuscle/muscle_warp.py`: 新增 `tet_snh_update_xpbd_fn()`、`solve_tetsnh_kernel`
- `data/slidingBall/config_xpbd_dgf.json`: 新增 `snh_mu/snh_lam/snh_dampingratio` 参数
- `examples/example_xpbd_dgf_sliding_ball.py`: 当 `snh_mu > 0` 时自动添加 SNH 约束

### 使用

在配置中设置 `snh_mu > 0` 即可启用：
```json
"xpbd": {
  "snh_mu": 1000.0,
  "snh_lam": 10000.0,
  "snh_dampingratio": 0.01
}
```

## 实验发现

### 收缩场景 (excitation=1.0, sliding ball)

- SNH 在收缩场景中增加了形状恢复力，使纤维不能完全收缩到 DGF 目标长度
- μ=1000 时影响很小（SNH 力 << 肌肉主动力），几乎不改变结果
- 与 DGF fiber 约束共同工作良好

### 拉伸场景 (excitation=0, 重力拉伸)

- SNH 单独使用时不足以防止网格坍塌（需配合 ARAP 或 volume 约束）
- α > 1 的设计导致体积项偏向膨胀态，在拉伸场景下不理想
- α = 1 去掉了反转保护，导致 tet 反转 → NaN

### Clamped SNH (仅抗拉伸)

- 实验了仅在拉伸时激活 SNH 的方案
- 问题：收缩时局部个别 tet 仍处于拉伸态（||F||_F > √3），导致不平衡力和持续漂移
- 需要更精细的 clamp 策略或与其他约束配合使用

### SNH vs DGF Passive 力学特性对比

| 特性 | SNH | DGF Passive |
|------|-----|-------------|
| 方向性 | 各向同性（双向对称） | 各向异性（沿纤维方向） |
| 非线性 | 近似线性 | 指数增长 |
| 压缩抵抗 | 有 | 无 |
| 力量级 (μ=1000) | ~0.03 F_max | ~1.0 F_max (λ=1.6) |

## 当前状态

- SNH 约束代码已实现，默认关闭（`snh_mu=0`）
- 收缩场景的 baseline 对比（XPBD-DGF vs OpenSim）结果良好
- 最佳参数: `num_substeps=40, veldamping=0.003, dampingratio=0.018`

## 下一步

- SNH 需配合 ARAP 或 volume 约束使用，才能在无激活场景下保持稳定
- 考虑在 VBD (Newton) 求解器中直接集成 SNH（不需要 XPBD 的 compliance 限制）
- 探索 SNH 作为肌肉组织被动弹性的最佳参数范围
