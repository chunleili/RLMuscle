# example_couple3: XPBD-Millard + MuJoCo Bicep Flexion

> 相关文档:
> - [Mesh 稳定性调查](2026-04-03-couple3-mesh-stability.md) — 退化 tet 根因分析、force clamping/mass floor
> - [方法对比实验](2026-04-03-couple3-approach-comparison.md) — explicit force vs TETFIBERMILLARD vs hybrid
> - [Mesh 质量改善实验](2026-04-03-mesh-quality-experiments.md) — damping/TETSNH/Laplacian/targeted repair

## 概述

实现了 `example_couple3`，在真实 bicep 几何上验证 XPBD-Millard-explicit-force 方法与 MuJoCo 骨骼动力学的耦合。

## 设计

**方法**: 复用 couple2 的 `SolverMuscleBoneCoupled` 框架，在 substep 循环中添加 `clear_forces()` + `accumulate_active_fiber_force()` 支持 explicit force。

**关键变更**:
1. `solver_muscle_bone_coupled_warp.py`: substep 循环添加 2 行 explicit force 调用（向后兼容）
2. `data/muscle/config/bicep_xpbd_millard.json`: 基于 bicep.json，移除 fiber 约束，添加 sigma0=300kPa, lambda_opt=1.0
3. `examples/example_couple3.py`: 基于 couple2 结构，默认使用 xpbd_millard config

## 验证结果

### couple3 vs couple2 对比 (300 steps, auto activation schedule)

| Metric | couple2 (TETFIBERNORM) | couple3 (Explicit Force) |
|--------|----------------------|--------------------------|
| 最大 torque | 0.36 N·m | 6.50 N·m (clamped) |
| 最大屈曲角 | ~0° | ~60° |
| 稳定性 | 稳定 | 稳定 |

### couple3 行为 (500 steps)

| Step | Activation | Joint Angle | Torque |
|------|-----------|-------------|--------|
| 100 | 1.00 | -5.0° | 4.10 |
| 150 | 1.00 | -34.7° | 6.50 |
| 250 | 1.00 | -55.6° | 6.46 |
| 300 | 0.70 | -59.9° | 5.95 |
| 400 | 0.30 | -58.5° | 3.33 |
| 500 | 0.00 | -20.4° | 0.00 |

### 关键发现

1. **Explicit force 产生了显著更大的力**: sigma0=300kPa 直接应用物理肌肉应力，比 TETFIBERNORM (stiffness=1000) 强得多
2. **屈曲行为符合预期**: 激活→屈曲→去激活→回弹的完整周期
3. **向后兼容**: couple2 在修改后的 solver 上运行结果不变（sigma0=0 时 accumulate_active_fiber_force 直接返回）
4. **Torque 被 max_torque=6.5 限制**: 说明肌肉产生的力超过骨骼驱动需求，controllability 系统正常工作

## 运行方式

```bash
# 基本运行
RUN=example_couple3 uv run python examples/example_couple3.py --auto --steps 300

# 带渲染
RUN=example_couple3 uv run python examples/example_couple3.py --auto --render

# Eval sweep
RUN=example_couple3 uv run python examples/example_couple3.py --eval --preset smooth_nonlinear
```

## 下一步

- 参数调优: 调整 sigma0、k_coupling、max_torque 以匹配期望的屈曲行为
- 与 OpenSim 对比: 定量验证角度和力的准确性
- 增加 num_substeps 如果需要更高精度
- 考虑添加 force-velocity (f_V) 曲线用于动态场景
