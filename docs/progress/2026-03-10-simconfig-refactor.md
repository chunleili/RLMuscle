# SimConfig 与 Constraints 重构

> 日期：2026-03-10

## 概述

1. 将 `SimConfig` 从 dataclass 改为 `SimpleNamespace`，去掉所有硬编码默认值，一切从 JSON 导入。
2. 合并 `parse_json_config` 与 `parse_json_args` 为统一`load_conifg(json_path, args=None)`。
3. 将 `constraints.py` 中的 taichi 依赖彻底分离，实现 warp/taichi 两端共用同一套 constraint 构建逻辑。

## 已完成

### config.py 重构
- `SimConfig = SimpleNamespace`，不再是 dataclass，无默认值
- `load_conifg(json_path, args=None)`：统一函数
  - `args=None` → 创建新 SimConfig
  - `args=obj` → 覆盖已有对象属性（原 `parse_json_args` 的功能）
- `load_config` / `parse_json_args` 保留为向后兼容别名
- 删除 `parse_json_args.py`，合并到 `config.py`

### constraints 分离
- **`constraints.py`** — 纯 Python/numpy，无 taichi/warp 依赖
  - 常量（TETVOLUME, TETARAP, ...）、`constraint_alias`、`build_surface_tris`
  - `ConstraintBuilderMixin`：所有 `create_*` 方法（返回 plain list dict）
  - `_collect_raw_constraints()`：调度 constraint_configs → create_* 方法
- **`constraints_taichi.py`** — 新文件，taichi 专用
  - `TaichiConstraintMixin.build_constraints()`：将 raw constraints 打包为 `ti.types.struct` field
- **`muscle.py`** — `MuscleSim(ConstraintBuilderMixin, TaichiConstraintMixin)`
- **`muscle_warp.py`** — `MuscleSim(ConstraintBuilderMixin)`
  - 删除约 330 行重复代码（常量、`constraint_alias`、`build_surface_tris`、所有 `create_*` 方法）
  - `build_constraints()` 调用 `_collect_raw_constraints()` 后打包为 warp 数组

### import 路径更新
- `SimConfig` / `load_config` 统一从 `VMuscle.config` 导入
- 更新了 `examples/example_muscle_warp.py`、`examples/example_couple2.py`、`tests/test_warp_*.py`、`tests/test_visual_comparison.py`、`tests/test_muscle_warp_vs_taichi.py`

## 关键工作流变化

- **新增配置字段**：只需在 JSON 中添加，`SimConfig` 无需修改（SimpleNamespace 动态属性）
- **新增 constraint 类型**：在 `constraints.py` 的 `ConstraintBuilderMixin` 中添加 `create_*` 方法，两端自动可用
- **导入方式**：推荐 `from VMuscle.config import SimConfig, load_conifg`

### muscle_common.py 重构（第二阶段）

将 `muscle.py` 和 `muscle_warp.py` 中 ~90% 重复的纯 Python 代码提取到 `muscle_common.py`：

- **新增 `muscle_common.py`**：
  - `load_mesh_*` 系列函数（JSON, tetgen, geo, USD, one_tet）
  - `get_bbox`, `generate_muscle_id_colors`, `get_config_path` 工具函数
  - `MuscleSimBase` 基类：`__init__`（mesh/bone 加载、字段分配、约束构建）、`load_bone_geo`、`step()`、`get_fps()`
  - 抽象钩子：`_init_backend`, `_allocate_fields`, `_init_fields`, `_precompute_rest`, `_build_surface_tris`, `_create_bone_fields`

- **muscle.py** → 继承 `MuscleSimBase, TaichiConstraintMixin`，仅保留 `@ti.func/@ti.kernel` 和 `Visualizer`
- **muscle_warp.py** → 继承 `MuscleSimBase`，仅保留 `@wp.func/@wp.kernel/@wp.struct` 和 `WarpRenderer`

- **修复**：
  - `contraction_ratio`/`fiber_stiffness_scale`/`HAS_compressstiffness` 用 `getattr` 提供默认值
  - `HAS_compressstiffness` 缓存为 `self.has_compressstiffness`（避免 Taichi AST 警告）
  - `color_bones` 同样用 `getattr` 防止属性缺失
  - `load_mesh_json` 文件句柄泄漏修复

## 下一步

- 如有新的仿真参数或 constraint 类型需求，分别在 JSON 和 `constraints.py` 中添加即可
- 可进一步优化：hasattr 清理、render_mode/color_muscles 枚举化、numpy 冗余副本清理
