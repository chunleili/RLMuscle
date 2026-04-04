# 代码库全面清理与重构

**日期**: 2026-04-04  
**分支**: dev  
**关联计划**: 审查报告（5 个并行 subagent 生成）  
**状态**: 成功

## 概述

对整个代码库进行了全面的代码审查和重构，覆盖 `src/VMuscle/`、`examples/`、`scripts/`、`data/config/`。使用 5 个并行审查 agent 分别检查了源码质量、示例规范性、脚本模式、架构一致性和代码重复。

## Phase 1: 修复 Critical Bug

| Bug | 文件 | 修复 |
|-----|------|------|
| `Warning()` 构造后丢弃（无效警告） | `constraints.py:409,414,462,467` | 改为 `warnings.warn()` |
| `FuncAnimation(frames=120)` 忽略实际帧数 | `usd_to_mp4.py:112` | 改为 `frames=len(frames)` |
| 死 `elif` 分支（永远不执行） | `mesh_utils.py:141-143` | 删除死分支 |
| `bone_pt_index` 用了两次 | `geo.py:271` | 第二个改为 `muscle_pt_index` |
| 导入废弃 Taichi solver（从未使用） | `example_minimal_joint.py:30` | 删除 stale import |

## Phase 2: couple2 与 couple3 架构对齐

### 核心变更
- **创建 `src/VMuscle/bicep_helpers.py`**: 提取 `ELBOW_PIVOT`、`ELBOW_AXIS`、`extract_radius_mesh()`、`build_elbow_model()` 到共享模块
- **扩展 `src/VMuscle/controllability.py`**: 添加 `parse_levels()`（从 3 处重复中提取）和 `run_eval_sweep()`（从 3 处重复中提取）
- **扩展 `src/VMuscle/usd_io.py`**: 添加 `build_bone_prim_map()`（从 2 处重复中提取）

### couple2 重构（432→228 行，-47%）
- 移除 11 个模拟参数 CLI 参数（`--preset`、`--k-coupling`、`--max-torque`、`--dgf` 等）
- 移除 `main()` 中的 cfg 路径覆盖（`geo_path`、`bone_geo_path`、`muscle_prim_path`、`bone_prim_paths`、`target_path`）
- 添加 `setup_couple2()` 工厂函数（镜像 couple3 模式）
- 耦合参数完全从 JSON config 读取（`"coupling"` 段）
- 创建 `data/muscle/config/bicep_coupled.json`（基于 `bicep.json` + coupling 段）

### couple3 更新（477→275 行，-42%）
- 导入共享模块替代本地定义
- `_activation_schedule` 委托给 `muscle_common.activation_ramp`
- `_run_eval_sweep` 委托给 `controllability.run_eval_sweep`

## Phase 3: src/ 废弃代码清理

### geo.py（370→245 行，-34%）
- 移除 `sys.path.append(os.getcwd())`（库代码不应有）
- 移除死函数：`get_args()`、`test_geo_vtk()`、`test_animation()`
- 移除重复的 `_pairListToDict` 定义
- 翻译所有中文注释为英文
- 移除未完成函数的 bare `...`
- 替换 `print()` 为 `logging`

### constraints.py — 消除三重复制
- 提取 `_compute_fiber_rest_data()` 共享方法
- `create_tet_fiber_constraint`、`create_tet_fiber_dgf_constraint`、`create_tet_fiber_millard_constraint` 均调用共享方法

### usd_io.py — 消除内部重复
- `_fix_tet_winding` → 导入 `mesh_utils.fix_tet_winding`
- `_extract_surface_tris` → 导入 `mesh_io.build_surface_tris`
- `_Y_TO_Z` → 导入 `mesh_io._Y_TO_Z`

### 其他
- `muscle_warp.py`: 移除无用的 `load_config`/`get_bbox` re-export
- `config.py`: 清理尾部空行
- `util.py`: 添加 broken import 注释说明

## Phase 4: Config 清理

| 操作 | 文件 | 说明 |
|------|------|------|
| 删除死字段 | `bicep_dgf.json` | 移除 `optimal_fiber_length`、`v_max_scale`（从未被代码读取） |
| 补全缺失字段 | `bicep_xpbd_millard.json` | 添加 `repair_sigma_min: 0.1`、`post_smooth_iters: 5` |
| 归档孤立 config | `bicep_fibermillard.json`、`bicep_hybrid.json` | 移到 `data/muscle/config/archived/`（无代码引用） |

## 统计

- **涉及文件**: 17 个修改 + 3 个新建
- **净删除**: -468 行（313 新增，781 删除）
- **新建共享模块**: `src/VMuscle/bicep_helpers.py`（102 行）
- **新建 config**: `data/muscle/config/bicep_coupled.json`

## 验证

- couple2: `RUN=example_couple2 uv run main.py --auto --steps 50 --no-usd` ✅
- couple3: `RUN=example_couple3 uv run main.py --auto --steps 50 --no-usd` ✅
- standalone: `RUN=example_muscle_warp uv run main.py --steps 5` ✅
- 所有修改文件语法检查: `ast.parse` ✅
- 所有共享模块导入检查 ✅

## 遗留 TODO（未在本次处理）

以下问题已识别但不在本次范围（按优先级排序）：

1. **scripts 壳子模式**: `test_mesh_quality.py`、`osim_*.py` 包含自己的模拟逻辑，应移到 examples
2. **`_import_opensim()` 四重重复**: 4 个 osim scripts 各自定义，应提取到 `src/`
3. **`compute_fiber_stretches` 五重重复**: `example_vbd_*.py` 应导入 `muscle_common` 版本
4. **`example_tetmesh_import*.py` 无 `main()`**: 模拟代码在模块级执行
5. **`example_human_import.py` 全中文注释 + 无 `main()`**
6. **`solver_muscle_bone_coupled` Taichi vs Warp**: ~400 行重叠，可提取 base class
7. **`activation.py` Warp kernel 硬编码 tau**: 忽略 config 中的 `tau_act`/`tau_deact`
8. **`example_couple.py`**: Taichi 版 couple，含 9 个模拟 CLI 参数（将随 Taichi 废弃）
9. **`load_config` 无 schema 验证**: 拼写错误的 JSON 字段被静默忽略
10. **`sliding_ball_helpers.py` `object.__new__` bypass**: 应使用 `MuscleSim.from_procedural()`
