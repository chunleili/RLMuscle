# SimConfig 与 Constraints 重构

> 日期：2026-03-10

## 概述

1. 将 `SimConfig` 从 dataclass 改为 `SimpleNamespace`，去掉所有硬编码默认值，一切从 JSON 导入。
2. 合并 `parse_json_config` 与 `parse_json_args` 为统一 `load_config(json_path, args=None)`。
3. 将 `constraints.py` 中的 taichi 依赖彻底分离，实现 warp/taichi 两端共用同一套 constraint 构建逻辑。

## 已完成

### 第一阶段：config.py 重构
- `SimConfig = SimpleNamespace`，不再是 dataclass，无默认值
- `load_config(json_path, args=None)`：统一函数
  - `args=None` → 创建新 SimConfig
  - `args=obj` → 覆盖已有对象属性（原 `parse_json_args` 的功能）
- 删除 `parse_json_args.py`，合并到 `config.py`

### 第二阶段：constraints 分离 + muscle_common.py 提取
- **`constraints.py`** — 纯 Python/numpy，无 taichi/warp 依赖
  - 常量（TETVOLUME, TETARAP, ...）、`constraint_alias`、`build_surface_tris`
  - `ConstraintBuilderMixin`：所有 `create_*` 方法（返回 plain list dict）
  - `_collect_raw_constraints()`：调度 constraint_configs → create_* 方法
- **`muscle_common.py`**：
  - `load_mesh()` 统一入口（支持 usd/geo/json 格式）
  - `get_bbox` 工具函数
  - `MuscleSimBase` 基类：`__init__`（mesh/bone 加载、字段分配、约束构建）、`load_bone_geo`、`step()`、`get_fps()`
  - 抽象钩子：`_init_backend`, `_allocate_fields`, `_init_fields`, `_precompute_rest`, `_build_surface_tris`, `_create_bone_fields`
- **muscle.py** → 继承 `MuscleSimBase`，仅保留 `@ti.func/@ti.kernel` 和 `Visualizer`
- **muscle_warp.py** → 继承 `MuscleSimBase`，仅保留 `@wp.func/@wp.kernel/@wp.struct` 和 `WarpRenderer`

### 第三阶段：全面清理
- **删除 `constraints_taichi.py`** — `TaichiConstraintMixin.build_constraints()` 合并到 `muscle.py` 的 `MuscleSim.build_constraints()` 中
- **muscle.py 清理**（1485 行 → ~740 行）：
  - 合并 `TaichiConstraintMixin`，`MuscleSim(MuscleSimBase)` 单继承
  - 删除：`flatten`、`outer_product`、`invariant4`、`invariant5`、`triangle_xform_and_area`、`squared_norm2`
  - 删除：`read_auxiliary_meshes`、`distance_update_xpbd`、`tri_arap_update_xpbd`、`transfer_shape_and_bulge`、`close`、`_compute_cell_fiber_dir`
  - 删除 solve_constraints 中 DISTANCE/TRIARAP 分支
  - 简化 `fem_flags`：只保留 TETARAP/TETARAPNORM/TETFIBERNORM
  - 简化 `tet_fiber_update_xpbd`：删除死代码 `use_anisotropic_arap` 分支
  - Visualizer 清理：删除 `_generate_muscle_colors`（肌肉/骨骼染色）、辅助网格渲染、`rgb_array` 模式
  - 删除 `bone_vertex_colors` 相关代码
- **muscle_warp.py 清理**（1543 行 → ~1200 行）：
  - 删除：`TriXformResult`、`distance_update_xpbd_fn`、`tri_arap_update_xpbd_fn`、`triangle_xform_and_area_fn`、`squared_norm2_fn`、`invariant4_fn`、`compute_cell_fiber_dir_kernel`
  - 删除 solve_constraints_kernel 中 DISTANCE/TRIARAP 分支
  - 简化 `fem_flags_fn`
  - 删除 `bone_vertex_colors`/`bone_colors_np` 相关代码
- **constraints.py 清理**：
  - 删除未使用常量：DISTANCE, BEND, STRETCHSHEAR, BENDTWIST, PINORIENT, PRESSURE, TRIAREA, TRIANGLEBEND, ANGLE, TETFIBER, PTPRIM, DISTANCEPLANE, TRIARAP, TRIARAPNL, TRIARAPNORM, TETARAPNL, TETARAPVOL, TETARAPNLVOL, TETARAPNORMVOL, SHAPEMATCH
  - 删除：`create_pin_constraints`、`create_tri_arap_constraints`、`compute_tri_rest_matrix`
- **muscle_common.py 清理**：
  - 删除：`generate_muscle_id_colors`、`get_config_path`、独立格式加载函数（合并到 `load_mesh`）
  - 修复 `contraction_ratio`/`fiber_stiffness_scale`/`HAS_compressstiffness` 用 `getattr` 提供默认值

### import 路径更新
- `SimConfig` / `load_config` 统一从 `VMuscle.config` 导入
- 更新了所有 examples 和 tests

### 第四阶段：mesh_io 拆分 + visualizer 提取
- **`mesh_io.py` 统一 mesh I/O**：
  - `load_mesh(path)` — 加载肌肉 TetMesh（.usd/.geo）
  - `load_bone_mesh(path)` — 加载骨骼表面网格（.usd/.geo）
  - `build_surface_tris(tets, positions)` — 提取表面三角面（从 constraints.py 移入，保留带 winding 修正的版本）
  - 内部函数：`_load_mesh_usd`, `_load_mesh_geo`, `_load_bone_usd`, `_load_bone_geo`, `_build_muscle_id_mapping`
  - 删除 JSON 格式支持、`read_auxiliary_meshes`（.obj/trimesh）
- **`muscle_common.py` 精简**：
  - 删除所有 mesh I/O 代码（`load_mesh`, `load_mesh_usd`, `load_bone_geo` 及其辅助方法）
  - 新增简化的 `_load_bone()` 调用 `mesh_io.load_bone_mesh()`
  - `load_bone_geo()` 保留为缓存代理（constraint builders 使用）
  - 从 `mesh_io` 导入 `load_mesh`, `load_bone_mesh`
- **`constraints.py`**：删除 `build_surface_tris`（移至 mesh_io.py）
- **`vis_taichi.py`**：从 muscle.py 提取 Taichi `Visualizer` 类
- **`vis_warp.py`**：从 muscle_warp.py 提取 `WarpRenderer` 类
- **muscle.py / muscle_warp.py**：更新导入，`build_surface_tris` 从 `mesh_io` 导入，Visualizer/WarpRenderer 从 `vis_*.py` 导入

## 当前文件结构

```
src/VMuscle/
  config.py           — SimConfig = SimpleNamespace, load_config()
  mesh_io.py          — load_mesh, load_bone_mesh, build_surface_tris (.geo/.usd only)
  constraints.py      — 纯 Python constraint 构建（ConstraintBuilderMixin）
  muscle_common.py    — MuscleSimBase 基类 + get_bbox
  muscle.py           — Taichi 后端 MuscleSim
  muscle_warp.py      — Warp 后端 MuscleSim
  vis_taichi.py       — Taichi Visualizer
  vis_warp.py         — Warp WarpRenderer
```

## 关键工作流变化

- **新增配置字段**：只需在 JSON 中添加，`SimConfig` 无需修改（SimpleNamespace 动态属性）
- **新增 constraint 类型**：在 `constraints.py` 的 `ConstraintBuilderMixin` 中添加 `create_*` 方法，两端自动可用
- **导入方式**：`from VMuscle.config import SimConfig, load_config`
- **Mesh I/O**：`from VMuscle.mesh_io import load_mesh, load_bone_mesh, build_surface_tris`
- **不再需要** `constraints_taichi.py`（已合并）

## 下一步

- 如有新的仿真参数或 constraint 类型需求，分别在 JSON 和 `constraints.py` 中添加即可
- Taichi 标记为 deprecated，后续开发以 Warp 为主
