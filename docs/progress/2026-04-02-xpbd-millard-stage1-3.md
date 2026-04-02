# XPBD-Millard Stage 1-3 进度

## 完成内容

### Stage 1: Millard 曲线模块 (`src/VMuscle/millard_curves.py`)

- 移植 OpenSim quintic Bezier 控制点算法 (`calcQuinticBezierCornerControlPoints`)
- 移植 active f_L 曲线构造（5段 Bezier）和 passive f_PE 曲线构造（2段 Bezier）
- **精确多项式表达**：用 Bernstein→幂基转换得到 x(u), y(u) 的 5 次多项式系数
- **能量积分闭式**：y(u)·x'(u) 的不定积分 F(u) 是 10 次多项式，精确无近似
- CPU 端 Newton 迭代 x→u + Horner 求值
- 25 项单元测试全通过

### Stage 2: XPBD 约束 (`TETFIBERMILLARD`)

- `constraints.py`: 新增类型常量 `TETFIBERMILLARD`，约束构建器 `create_tet_fiber_millard_constraint`
- `muscle_warp.py`:
  - Warp `@wp.func millard_eval_wp`: GPU 上分段 Bezier 求值（线性扫描段 + Newton x→u + Horner y(u)）
  - `@wp.kernel solve_tetfibermillard_kernel`: 与 DGF 内核结构相同，替换曲线求值
  - `_init_millard_curves()`: 构建时自动初始化曲线系数数组
  - `_dispatch_constraints`: 添加 TETFIBERMILLARD 分支
- GPU/CPU 精度对比：最大误差 ~1.6e-6（float32 精度内）

### Stage 3: Sliding Ball 验证

- 配置: `data/slidingBall/config_xpbd_millard.json`
- 示例: `examples/example_xpbd_millard_sliding_ball.py`
- 结果: 300 步 (5s) 仿真成功，收敛到稳态

## 关键结果

### XPBD-Millard vs OpenSim-Millard（Sliding Ball）

| 指标 | XPBD-Millard | OpenSim-Millard | 误差 |
|------|-------------|-----------------|------|
| λ_eq (稳态) | 0.5498 | 0.5461 | **0.7%** |
| 球位置 (m) | 0.0449 | 0.0444 | **1.2%** |

### 与 DGF 对比

| 指标 | XPBD-Millard | XPBD-DGF | OpenSim-DGF |
|------|-------------|----------|-------------|
| λ_eq (稳态) | 0.5498 | 0.5945 | 0.5899 |
| 球位置 (m) | 0.0449 | 0.0405 | 0.0400 |

Millard 和 DGF 结果不同是因为两个本构模型的曲线形状本身不同，非实现误差。两者各自对比 OpenSim 的误差均在 ~1% 以内。

## 发现

1. **XPBD-Millard vs OpenSim-Millard 精度极好**：纤维长度误差 0.7%，球位置误差 1.2%，与 DGF 实现的精度水平 (~0.8%) 相当。
2. **Millard 和 DGF 曲线形状差异**：Millard f_L 的 ascending limb 更陡、肩部有 ylow=0.1 最小值；DGF 在肩部降到 ~0。导致 Millard 平衡纤维长度更短。
3. **被动力差异更大**：DGF f_PE 指数增长无界，Millard 在 strain=0.7 后线性外推。
4. **GPU 性能**：Millard 的 Newton 迭代 + Horner 求值（~70 FLOPs/tet）与 DGF 的 3×exp()（~60-90 FLOPs）相当。
5. **能量积分可用**：F(u) = ∫y·x' dt 作为 10 次多项式已实现，精度 <1e-7。

## Stage 4: Simple Arm 验证

### XPBD-Millard vs OpenSim-Millard（Simple Arm）

| 指标 | XPBD-Millard | OpenSim-Millard | 差异 |
|------|-------------|-----------------|------|
| 稳态肘角 | 88.25° | 88.15° | **0.10°** |
| 稳定性 | 无 NaN/crash/inverted tet | - | ✓ |
| 力范围 | [4.1, 54.3] N | - | 合理 |

- 1D Hill 模型中的 DGF 曲线已替换为 Millard 曲线
- 瞬态振荡模式与 OpenSim 一致
- 1000 步 (10s) CPU 仿真无异常

## 下一步

- [ ] 被动能量约束 C=√(2Ψ_PE) 实验
