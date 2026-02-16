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

### 运行验证（本轮）
- 迭代 1：执行 `examples/example_dynamics.py`（`--viewer null --headless --num-frames 30 --use-layered-usd`）失败，报错：`ModelBuilder` 不存在 `add_link`（当前 `newton==0.1.3` 仅有 `add_body` 路径）。
- 迭代 2：转测上一个 commit 里新增的 `examples/example_couple.py`（30 帧，无头 + layered USD）运行通过，产出 `output/example_couple_30.anim.usda`。
- 迭代 3：将 `example_couple` 扩展到 120 帧回归，运行通过，产出 `output/example_couple_120.anim.usda`。
- 对 USD 文件进行结构检查（`pxr.Usd`）：
  - `example_couple_30.anim.usda` 与 `example_couple_120.anim.usda` 均包含 `/anim/runtime` 下 `joint_angle`、`muscle_activation`、`muscle_centroid_y` 三组 timeSamples。
  - `over /pendulum/joint0` 的 `xformOp:rotateY.timeSamples` 与帧数一致（30/120）。
  - 分层结构正确：`subLayers = [@pendulum.usda@]`，且输出目录存在 `output/pendulum.usda`。

### 可能问题（已记录）
- **API 兼容性回退**：`example_dynamics.py` 仍使用旧的 `newton` 构模 API（`add_link`），在当前环境不可运行。
- **文档/进度与代码不一致**：当前 `example_dynamics.py` 只写出 `joint_angle` 和 `motor`，未实现 `PROGRESS.md` 中声明的 `body_pos_x/y/z` 输出。
- **依赖声明问题**：`uv sync` 失败（`pyproject.toml` 中 `workspace=true` 与 `editable=true` 同时设置冲突），会影响一键复现实验环境。

### 下一步
- 将 `example_dynamics.py` 对齐 `newton==0.1.3` API（参考 `example_couple` 的 `add_body/add_articulation` 用法），先恢复可运行性。
- 补齐或回滚 `body_pos_x/y/z` 的声明，保证“代码行为”和“PROGRESS/README 描述”一致。
- 修复 `pyproject.toml` 的 `tool.uv.sources` 配置冲突，恢复 `uv sync` 可用，避免依赖安装路径分叉。

## example_couple

### 已完成
- 新增 `src/RLVometricMuscle/muscle_core.py`：
  - 将肌肉核心改写为 Warp 版本（去除 Taichi 依赖）。
  - 提供 `SimConfig`、`load_config`、`.geo` 读取、`MuscleCore.step()`。
- 改造 `src/RLVometricMuscle/solver_volumetric_muscle.py`：
  - `SolverVolumetricMuscle` 改为包装 `MuscleCore`。
  - 支持从 Newton `state_in/state_out` 同步粒子状态。
  - 支持 `tet_activations` / `muscle_activations` 输入。
- 新增 `examples/example_couple.py`：
  - 使用 `SolverMuscleBoneCoupled` 同时驱动肌肉和骨骼。
  - 使用激活信号驱动关节 motor，形成简单耦合回路。
  - 使用 `UsdIO` layered 输出 `joint_angle`、`muscle_activation`、`muscle_centroid_y`。

### 下一步
- 在 GUI 模式下观察耦合效果与参数灵敏度（`max_motor`、`activation_hz`）。
- 后续可将肌肉力映射从“标量 motor”升级为更真实的几何力臂模型。

## dependency_setup

### 已完成
- 在 `.venv` 中执行 `ensurepip`，补齐 `pip`。
- 尝试多种路径安装运行 `example_couple` 所需依赖：
  - `uv add numpy warp-lang --python .venv/bin/python`
  - `uv pip install --python .venv/bin/python numpy`
  - `.venv/bin/python -m pip install numpy warp-lang`
  - `uv sync --python .venv/bin/python`
- 在 `pyproject.toml` 中补充显式依赖：`numpy`、`warp-lang`。

### 当前阻塞
- 代理对外网包源返回 `403 Forbidden`（CONNECT tunnel failed），且直连网络不可达（`Network is unreachable`）。
- 因此当前环境无法从 PyPI 下载 `numpy/warp-lang/newton/usd-core`，`example_couple` 仍无法运行。

### 下一步
- 在可访问包源的环境执行：
  - `uv add numpy warp-lang`
  - `uv sync`
  - `.venv/bin/python examples/example_couple.py --viewer null --headless --num-frames 5 --use-layered-usd`


### 运行修复（本轮）
- 已恢复网络后完成依赖安装：`uv add numpy warp-lang typing-extensions --python .venv/bin/python`。
- 已执行 `uv sync --python .venv/bin/python`。
- 为兼容当前 `newton==0.1.3` API，修复 `examples/example_couple.py`：
  - `add_link()` -> `add_body()`
  - `add_articulation([j0], key=...)` -> `add_articulation(key=...)`
  - 移除 `create_collision_pipeline` 依赖，改用 `model.collide(state)`
  - `newton.examples.run(example, args)` -> `newton.examples.run(example)`
- `example_couple` 无头运行 5 帧已通过。
