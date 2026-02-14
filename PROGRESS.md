# 进度

## example_dynamics

### 已完成
- 创建 `data/pendulum/pendulum.usda` — `UsdIO` 所需的最小 USD 源文件。
- 创建 `examples/example_dynamics.py` — 单链摆示例：
  - `SolverFeatherstone` 刚体物理仿真（纯刚体场景，不加载肌肉）。
  - `motor` GUI 滑块（范围 -1000 到 1000），每个子步设置 `control.joint_f[0]`。
  - `UsdIO` 分层输出：每帧将 `joint_angle`、`motor`、`body_pos_xyz` 写入 `output/example_dynamics.anim.usda`。
- 注释掉 `SolverVolumetricMuscle.step()` 中的 `self.core.step()`（保留 TODO）。

### Pipeline 修复
- **solver**: `example_dynamics` 从 `SolverMuscleBoneCoupled` 切换为 `SolverFeatherstone`，避免无用的肌肉加载。
- **joint_f 分配**: 预分配 `_motor_buf`（numpy），避免每子步重建 warp 数组。
- **USD 输出**: 增加 body 世界坐标 (`body_pos_x/y/z`)，支持 DCC 回放。
- **AGENTS.md**: 清理误入的分析文本。

### 已测试
- 无头模式 30 帧运行无报错，不再加载 Taichi / bicep.geo。
- `output/example_dynamics.anim.usda` 包含逐帧 `joint_angle`、`motor`、`body_pos_xyz`。

### 下一步
- 运行 GUI 测试，验证滑块交互和实时摆锤运动：
  ```bash
  .venv/Scripts/python.exe examples/example_dynamics.py --viewer gl
  ```
