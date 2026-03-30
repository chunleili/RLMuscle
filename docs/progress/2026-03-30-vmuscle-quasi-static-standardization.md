# 2026-03-30 vmuscle quasi-static 固定化

## 背景

目标是把这轮针对 VBD 体肌肉大拉伸不稳定的临时修补，整理成一条固定、可复现、接口一致的 quasi-static 路线，而不是继续依赖 example 层 hack。

## 代码改动

### 1. 清理动态主动项的遗留路径

- 查看 `external/newton` 里最近的 vmuscle 相关提交后，移除了当前不再保留的动态项：
  - 删除 `SolverVBD` 中的 `particle_q_prev2`
  - 删除 `vmuscle_kernels.py` 中的 `dgf_force_velocity()`、`compute_fv_factor_inline()` 和 `fv_factor` 参数链
  - 删除 vmuscle custom attribute `max_contraction_velocity`
- `set_vmuscle_properties()` 同步简化，只保留当前 quasi-static 路线真正使用的参数。

### 2. 把 quasi-static 下沉到 solver 层

- `particle_vbd_kernels.py` 新增 `forward_step_quasi_static()`
- `SolverVBD.__init__()` 新增 `vmuscle_quasi_static: bool = True`
- `SolverVBD._initialize_particles()` 中，当 `has_vmuscle and vmuscle_quasi_static` 时，默认走 quasi-static forward step
- 这个 quasi-static forward step：
  - 保留重力和外力项
  - 忽略输入速度历史
  - 保留 VBD 质量 Hessian 正则项

### 3. 规范化项目侧调用

- `example_vbd_mujoco_simple_arm.py` 删除手动同步 `particle_q_prev` 的 helper
- `example_vbd_mujoco_simple_arm.py` / `example_vbd_coupled_simple_arm.py` 统一从 `coupling.vbd_*` 读取 VBD 选项
- Stage 2 的默认值改为和 `SolverVBD` 一致，默认启用 quasi-static
- `example_muscle_sliding_ball.py` 显式使用 `vmuscle_quasi_static=True`
- `data/simpleArm/config.json` 固定默认值：
  - `vbd_quasi_static = true`
  - `vbd_scale_guard_min = 0.35`
  - `vbd_scale_guard_max = 1.6`
  - `vbd_substeps = 20`
  - `vbd_iterations = 20`

## 验证

### Scale sweep

直接对 SimpleArm 的 VBD 圆柱体做拉伸扫描，观察 `l_tilde_mean`：

- dynamic：`scale=1.35 -> 1267.24`，`scale=1.50 -> 3987.71`
- quasi-static：`scale=1.35 -> 0.6849`，`scale=1.50 -> 0.7610`

结论：去掉速度历史以后，大拉伸下的 blow-up 明显消失，说明当前不稳定的主因之一确实在于 VBD 惯性/历史项和主动收缩的耦合。

### 静态检查

- `uv run python -m py_compile external/newton/newton/_src/solvers/vbd/vmuscle_kernels.py external/newton/newton/_src/solvers/vbd/vmuscle_launch.py external/newton/newton/_src/solvers/vbd/solver_vbd.py src/VMuscle/mesh_utils.py examples/example_muscle_sliding_ball.py examples/example_vbd_coupled_simple_arm.py examples/example_vbd_mujoco_simple_arm.py`

通过。

### 对比脚本

运行：

- `uv run python scripts/run_simple_arm_comparison.py`

结果：

- 默认 `Stage 3: VBD Coupled vs OpenSim DGF`
- `RMSE = 1.83 deg`
- `Max error = 8.37 deg`
- 最终 `final angle ≈ 82.2 deg`
- VBD Coupled 日志显示 `mode=quasi-static`

## 当前结论

- 这轮改动后，quasi-static 已经是 vmuscle VBD 的默认路径，不再依赖 example 层补丁
- vmuscle 代码路径比之前更短，更接近当前真实实现
- 如果目标是先把大拉伸算稳，这条路线已经可以继续往下用
- 如果后面要回到“真实动态主动本构”，还需要重新设计动态项，而不是把旧的 `f_v + prev2` 逻辑再塞回来
