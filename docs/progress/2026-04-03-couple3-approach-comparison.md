# couple3 方法对比实验

## 背景

couple3 使用 XPBD-Millard 方法在真实 bicep mesh 上驱动肘关节屈曲。由于 bicep mesh 存在退化四面体（质量比 12095:1），explicit active fiber force（Route C）在 force clamping 后力量严重不足。本实验对比三个替代方案。

## 方案

1. **Approach 1**: 调参 — 提高 explicit force 的 max_accel，寻找稳定性与力的平衡
3. **Approach 3**: TETFIBERMILLARD 约束（Route A）— 使用已实现的 energy-based XPBD 约束
4. **Approach 4**: 混合 — TETFIBERNORM 驱动收缩 + 小量 explicit force

## 结果

| 方案 | Config | Max Torque (N·m) | Max Angle (°) | 反转 tets | 稳定 |
|------|--------|:----------------:|:-------------:|:---------:|:----:|
| 基线: explicit max_accel=20 | bicep_xpbd_millard | 0.14 | ~0 | ~55 (1.4%) | 是（太弱） |
| Approach 1: max_accel=100 | bicep_xpbd_millard_tuned | 0.25 | ~0 | ~130 (3.3%) | 是（太弱） |
| Approach 1: max_accel=500 | — | 不定（振荡） | ~0 | ~430 (11%) | 力不连贯 |
| Approach 3: TETFIBERMILLARD k=1000 | bicep_fibermillard | 1.9 | 3.3 | ~500 (12.7%) | 是 |
| Approach 3: TETFIBERMILLARD k=10000 | bicep_fibermillard_high | 3.3 | 8.9 | ~510 (13%) | 是 |
| Approach 3: TETFIBERMILLARD k=100000 | bicep_fibermillard_vhigh | 4.0 | 12.5 | ~510 (13%) | 是 |
| **Approach 3: TETFIBERMILLARD k=10000 + k_coupling=100k** | **bicep_fibermillard_coupled** | **9.6** | **61** | **~550 (14%)** | **是** |
| Approach 3: σ₀=1MPa, k=10000 | bicep_fibermillard_strong | 2.5 | 9.2 | ~510 (13%) | 是 |
| Approach 4: Hybrid TETFIBERNORM+50kPa | bicep_hybrid | 0.21 | ~0 | ~130 (3.3%) | 是（太弱） |

## 关键发现

### TETFIBERMILLARD（Route A）是最优方案

- **Explicit force（Route C）根本无法工作**: 在退化 mesh 上，acceleration clamping 必须非常激进才能防止爆炸，导致 force 贡献几乎为零。即使提高 max_accel 到 500，force 方向变得不连贯。
- **TETFIBERMILLARD 天然稳定**: 作为 XPBD position-based 约束，不存在 explicit force 的不稳定问题。~14% 反转 tets 是 mesh 质量的固有限制。
- **Hybrid 方案无价值**: TETFIBERNORM 的 position correction + explicit force 的 vertex pushing 互相竞争，结果不如单独使用任何一种方法。

### 关键瓶颈是 k_coupling，不是约束 stiffness

默认 k_coupling=24000 和 max_torque=6.5 限制了 ATTACH 反力到骨骼的传递。将 k_coupling 提高到 100000 并放宽 max_torque=20 后，TETFIBERMILLARD 成功驱动 61° 屈曲。

约束 stiffness 在 10000-100000 范围内饱和（4.0 vs 3.3 N·m），增加 sigma0 到 1MPa 反而降低了 torque。

### 反转 tets 是 mesh 质量的基线问题

所有驱动明显收缩的方案都产生 ~500 反转 tets（13%），这些反转集中在 tet 3637、3807 等退化四面体上。改善需要 remesh。

## 最佳配置

```
config: bicep_fibermillard_coupled.json
  TETFIBERMILLARD: stiffness=10000, sigma0=300kPa, contraction_factor=0.4
  k_coupling: 100000
  max_torque: 20.0
```

运行命令：
```bash
uv run python examples/example_couple3.py --auto --steps 300 --k-coupling 100000 --max-torque 20
```

## TETFIBERMILLARD 行为 (k_coupling=100k, max_τ=20, 300 steps)

| Step | Activation | Joint Angle (°) | Torque (N·m) | 反转 tets |
|------|:----------:|:----------------:|:------------:|:---------:|
| 50 | 0.00 | 0.0 | 0.00 | 0 |
| 75 | 0.50 | -0.07 | 1.33 | ~400 |
| 100 | 1.00 | -3.2 | 3.60 | ~520 |
| 125 | 1.00 | -15.1 | 6.60 | ~540 |
| 150 | 1.00 | -35.6 | 9.60 | ~540 |
| 175 | 0.70 | -56.4 | 6.84 | ~548 |
| 200 | 0.70 | -61.4 | 3.84 | ~557 |
| 225 | 0.30 | -56.5 | 0.84 | ~545 |
| 250 | 0.07 | -41.8 | 0.25 | ~547 |
| 275 | 0.00 | — | 0.00 | ~65 |

## 下一步

1. ~~Approach 1 (explicit force tuning)~~ — 放弃
2. ~~Approach 4 (hybrid)~~ — 放弃
3. **TETFIBERMILLARD + 合适 coupling 参数** — 采用
4. 考虑 remesh bicep 去除退化 tets（减少反转 tets 到 couple2 水平）
5. 精调 coupling 参数（k_coupling, max_torque, EMA smoothing）
6. 考虑 controllability preset 专为 fibermillard 优化
