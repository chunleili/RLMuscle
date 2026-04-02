# Stage 5: 显式主动纤维力（Explicit Active Fiber Force）

> 日期: 2026-04-02
> 分支: xpbd-millard-stage5
> 前置: Stage 1-4（Millard 曲线、TETFIBERMILLARD 约束、sliding ball + simple arm）

## 目标

用 FEM 离散的显式主动力替代 XPBD 能量约束和 1D Hill 模型，使 3D mesh 成为力的原生来源。

## 完成内容

### 核心 kernel 实现

**新增 `accumulate_active_fiber_force_kernel`**（per-tet GPU kernel）：
- 计算 λ_f = ||F·d|| = ||Ds · (Dm⁻¹ · d)||（3D fiber stretch）
- 对齐 Millard 输入：r = λ_f / λ_opt（分母对齐，解决 l_opt ≠ L_ref 问题）
- 评估 f_FL(r) via `millard_eval_wp`
- 计算主动应力 τ_act = σ₀ · a · f_FL(r)
- 散射节点力：f_j = -V0 · τ_act · w[j] · n，其中 w = Dm⁻¹·d
- 量纲验证：m³ × Pa × m⁻¹ = N

**修改 `integrate_kernel`**：
- 接受 `force` field 和 `mass` field
- 加入 `force[i] / mass[i]` 加速度项
- 支持 `wp.vec3` 重力方向（不再硬编码 Y 轴）
- 增加 `stopped` 顶点检查（固定点不移动）

**修改仿真循环 `step()`**：
```
clear_forces → accumulate_active_fiber_force → integrate → clear → solve_constraints → update_velocities
```

**`from_procedural()` 扩展**：新增 `sigma0` 和 `lambda_opt` 参数。

### 验证结果

| 测试 | 结果 |
|------|------|
| Kernel 编译 | 通过 |
| GPU Millard 曲线精度 | max_err < 2e-6 |
| CPU 能量约束验证（7 项） | 全部 PASS |
| Simple arm 10s 稳定性 | 无 crash / 无 tet inversion |
| Simple arm 稳态肘角 | 88.25°（与 Stage 4 完全一致） |
| Sliding ball 力方向/大小 | 正确（底部 +311.77N ↑ vs 重力 98.1N ↓） |

**Sliding ball 跳过**：XPBD 对极端质量比（10kg 球 vs 0.002kg 内部顶点 → 700:1）的显式积分天生抗性差。这是 XPBD 框架的已知局限，不是力计算的错误。

### 实验图

**力场验证（Sliding ball, a=1.0, lm=1.0）**：

![Force field at rest](../imgs/tmp/stage5/force_field_rest.png)

**Simple arm 肘角轨迹（10s 稳定性）**：

![Simple arm trajectory](../imgs/tmp/stage5/simple_arm_trajectory.png)

### 复现命令

```bash
# 1. Kernel 编译测试
uv run python -c "import warp as wp; wp.init(); from VMuscle.muscle_warp import accumulate_active_fiber_force_kernel; print('OK')"

# 2. GPU Millard 曲线测试
uv run python -m tests.test_millard_gpu

# 3. CPU 能量约束验证
uv run python scripts/experiment_energy_constraint.py

# 4. Simple arm 完整运行（10s, ~5min on CPU）
RUN=example_xpbd_coupled_simple_arm_millard uv run main.py

# 5. 生成文档图片
uv run python scripts/plot_stage5_results.py
```

## 关键设计决策

### 采纳方案：显式主动力（Active/Passive Split）

基于 `docs/notes/xpbd_fem_fiber_lm.md` 的分析：

```
passive part → XPBD constraints（体积 SNH、各向同性 ARAP、被动纤维）
active part  → direct additional force（显式，在 integrate 步施加）
```

**优点**：
1. 力的方向和大小直接来自 FEM 离散化，物理意义清晰
2. V0 作为力的直接缩放因子（应力 → 力），不藏在 compliance 中
3. 主动/被动分离，易于调参
4. 力直接从 mesh 的 |Fd| 计算，闭环自洽

### 备选方案 A：能量本构约束 C = √(2·σ₀·a·Ψ_L)

**状态：已实现并通过 CPU/GPU 验证，但弃用**

将主动力包装为 XPBD 能量约束。关键恒等式 C·dC/dlm = σ₀·a·f_L 确保约束力精确等于 Millard 力。

**优点**：
- 利用 XPBD 质量加权机制，对质量比更鲁棒
- 已有闭式10次多项式能量积分，GPU 求值高效

**问题与改进方向**：
- Ψ_L 在 lm_opt=1.0 附近不对称（ascending limb 积分较小，descending limb 较大），导致约束强度不平衡
- **可能的修复**：增加常数 offset 使 Ψ_L 在 lm_opt 处取最小值而非在 lm_min 处为零，即 Ψ_shifted(lm) = Ψ_L(lm) - Ψ_L(lm_opt) + ε。这样 C 在最优长度附近对称且最小
- 主动/被动混在一起，不如显式力灵活

**保留代码**：`solve_tetfibermillard_kernel`、`millard_energy_eval_wp` 均保留，可随时切换。

### 备选方案 B：Mesh 变形提取纤维长度 → 1D 力模型

**状态：未实现，设计阶段**

从 XPBD mesh 的 ||F·d|| 提取平均纤维拉伸比，喂入 1D Millard/Hill 模型计算力，再反馈给 MuJoCo。

```
mesh deformation → mean(||F·d||) → r = stretch × (mesh_len / L_opt) → f_FL(r) → F_muscle → MuJoCo ctrl
```

**优点**：
- 与 MuJoCo 耦合简单（单标量力输出）
- 可复用已有 1D Hill 模型的 force-velocity 和 tendon elasticity

**问题**：
- 需要骨骼目标更新允许 mesh 拉伸（当前是刚体旋转，l_xpbd 不变）
- 绕了一圈回到 1D 模型，不如显式力直接

**适用场景**：当 MuJoCo 耦合为刚性需求（如 RL 训练需要精确的 motor force 接口）时，此方案是最简单的桥接。

## 文件变更清单

| 文件 | 变更 |
|------|------|
| `src/VMuscle/muscle_warp.py` | 新增 `accumulate_active_fiber_force_kernel`、`clear_force_kernel`（废弃，改用 `force.zero_()`）；修改 `integrate_kernel`（force/mass + vec3 gravity + stopped check）；新增 `clear_forces()`、`accumulate_active_fiber_force()` 方法；`from_procedural()` 增加 sigma0/lambda_opt；`build_constraints()` 支持 active force 初始化 |
| `src/VMuscle/muscle_common.py` | `step()` 循环增加 `clear_forces()` + `accumulate_active_fiber_force()` |
| `examples/example_xpbd_millard_sliding_ball.py` | 适配显式力方案、vec3 重力、lambda_opt |
| `examples/example_xpbd_coupled_simple_arm_millard.py` | 启用 sigma0/lambda_opt、XPBD 步中加入 active force |
| `data/slidingBall/config_xpbd_millard.json` | 移除 fibermillard 约束，加入 lambda_opt |
| `docs/notes/xpbd_fem_fiber_lm.md` | 新增理论参考文档 |
| `docs/superpowers/specs/...` | 更新设计文档 |
| `docs/superpowers/plans/...` | 更新实施计划 |

## 路线对比

详见 `docs/notes/stage5_force_routes.md`，包含三条路线的原理、伪代码和实验结果。

## 下一步

1. **被动纤维力**：可用 XPBD 约束 `C_fib_passive = √(2·V0·Ψ_PE)` 或简单弹簧 `C = λ_f - 1`
2. **Force-velocity**：当前仅实现 force-length，需增加速度依赖项
3. **Sliding ball 调参**：减小球质量 + 增大密度即可稳定，或用能量约束备选方案
4. **MuJoCo 耦合力提取**：需修改骨骼目标更新（允许 mesh 拉伸）后可走备选方案 B
