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

## 下一步
- 调参：`fiber_stiffness_scale` 对力矩和收缩量的影响
- 与 VBD 版 sliding ball 对比（需修复 VBD 例子的 Newton 兼容性问题）
- 考虑动态版本（加入 force-velocity 曲线）
