# XPBD-DGF 本构模型

## 已完成

### 新增 TETFIBERDGF 约束类型
- 在 XPBD 求解器中引入 DGF (DeGroote-Fregly 2016) 力-长度/被动弹性曲线
- 3 条 DGF 曲线移植为 Warp `@wp.func`（active force-length, passive force-length, force-velocity）
- 准静态模式（省略 force-velocity，`f_V = 1.0`）
- DGF 力调制约束刚度 + 目标拉伸：`k_eff = k_base * f_total`, `target = 1 - contraction * f_active`

### 修改文件
- `src/VMuscle/muscle_warp.py`: +DGF warp 函数, +`solve_tetfiberdgf_kernel`, +dispatch
- `src/VMuscle/constraints.py`: +`TETFIBERDGF` 常量, +`create_tet_fiber_dgf_constraint()`, +别名
- `data/muscle/config/bicep_dgf.json`: bicep DGF 配置
- `examples/example_couple2.py`: +`--dgf` 标志

### 新增 XPBD-DGF Sliding Ball 示例
- `examples/example_xpbd_dgf_sliding_ball.py`: 程序化圆柱体 + 集中球质量
- `data/slidingBall/config_xpbd_dgf.json`: XPBD-DGF 配置
- 修正了 `create_cylinder_tet_mesh` 的 tet winding 与 XPBD 约定不一致的问题

### 验证结果（sliding ball, excitation=1.0, 300 steps）
- 初始 fiber length ~1.0（正确）
- 全激活后收缩到 ~0.90（约 10% 收缩）
- 仿真稳定，无爆炸

### DGF f-L 曲线驱动 + OpenSim 验证 (2026-03-31)

**核心改动**：`solve_tetfiberdgf_kernel` 现在使用 DGF active f-L 曲线调制约束刚度：
- `stiffness = cstiffness * f_total(lm_tilde) * fiber_stiffness_scale`
  - `f_total = a * f_L(lm_tilde) + f_PE(lm_tilde)` (active + passive DGF curves)
- `target_stretch = 1 - a * contraction_factor`
  - `contraction_factor` 每步从 DGF 曲线反求: `f_L(lm_eq) = mg / (F_max * a)`
- 添加 `dgf_equilibrium_fiber_length()` 反求 ascending limb 上的平衡点
- 添加 `update_cons_restdir1_kernel` 每步更新 GPU 约束中的 contraction_factor

**验证结果（sliding ball, excitation=1.0, 300 steps vs OpenSim DGF）**：
- XPBD-DGF: lm_eq = 0.5945, bottom_z = 0.0405m
- OpenSim: nfl = 0.5899, y = 0.0400m
- 误差: ~0.8% 归一化纤维长度
- 力曲线稳态吻合良好

**关键发现**：
- `target=0` 方案导致 mesh collapse（内部顶点质量太小，无 lateral resistance）
- 正确方案：DGF 反求目标拉伸 + f_L 调制刚度，既保证 f-L profile 正确，又防止 mesh 不稳定
- `sigma0` 通过 `restdir[0]` 传入 kernel，`contraction_factor` 通过 `restdir[1]` 动态更新

## 下一步
- 与 VBD 版 sliding ball 对比（需修复 VBD 例子的 Newton 兼容性问题）
- 考虑动态版本（加入 force-velocity 曲线）
- 测试变激活（excitation ramp / step）下的瞬态响应
