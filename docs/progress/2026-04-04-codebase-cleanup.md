# 代码库全面清理与重构

**日期**: 2026-04-04  
**分支**: dev  
**Commits**: `39ee427`, `c7010ee`  
**关联文档**: [2026-04-03-refactor-dedup.md](2026-04-03-refactor-dedup.md)（前次重构，SimpleArm + Sliding-ball 去重）  
**状态**: 成功

## 概述

在前次重构（SimpleArm/Sliding-ball 去重，净减 ~340 行）基础上，对整个代码库进行了两轮清理：

1. **Commit `39ee427`**: couple3 架构标准化——提取 `setup_couple3()` 工厂函数，移除 12 个模拟 CLI 参数，简化 scripts
2. **Commit `c7010ee`**: 全面审查与修复——5 个 bug 修复、couple2 对齐 couple3 架构、src/ 废弃代码清理、config 规范化

使用 5 个并行审查 agent 分别检查了源码质量、示例规范性、脚本模式、架构一致性和代码重复。

---

## Round 1: couple3 架构标准化 (`39ee427`)

### 核心变更

| 文件 | 变更 |
|------|------|
| `examples/example_couple3.py` | 提取 `setup_couple3()` 工厂函数，返回 `(solver, sim, state, cfg, dt)`；移除 12 个模拟 CLI 参数（`--sigma0`, `--max-accel`, `--k-coupling`, `--max-torque`, `--preset` 等） |
| `data/muscle/config/bicep_fibermillard_coupled.json` | 添加 `"coupling"` 段：`preset`, `k_coupling=100000`, `max_torque=20` |
| `scripts/run_couple3_curves.py` | 177->102 行，导入 `setup_couple3()` 替代重复 setup 代码 |
| `scripts/experiments/sweep_post_smooth.py` | 197->132 行，导入 `setup_couple3()` 替代重复 setup 代码 |
| `tests/test_mesh_quality.py` | 移除硬编码路径覆盖 |
| `CLAUDE.md` | 添加 scripts-as-shells 和 no-CLI-params 规则 |

**统计**: 6 文件，183 新增，305 删除，净减 122 行

### 设计原则

- **参数统一从 config 读取**: 所有模拟参数（sigma0, coupling preset, k_coupling, max_torque 等）从 JSON config 的 `"coupling"` 段读取，CLI 仅保留工作流参数（`--auto`, `--eval`, `--steps`）
- **Scripts-as-shells**: 脚本导入 example 的 `setup_*()` 函数，不自行实现模拟逻辑

---

## Round 2: 全面审查与修复 (`c7010ee`)

### Phase 1: 修复 Critical Bug

| Bug | 文件 | 修复 |
|-----|------|------|
| `Warning()` 构造后丢弃（无效警告） | `constraints.py:409,414,462,467` | 改为 `warnings.warn()` |
| `FuncAnimation(frames=120)` 忽略实际帧数 | `usd_to_mp4.py:112` | 改为 `frames=len(frames)` |
| 死 `elif` 分支（永远不执行） | `mesh_utils.py:141-143` | 删除死分支 |
| `bone_pt_index` 用了两次 | `geo.py:271` | 第二个改为 `muscle_pt_index` |
| 导入废弃 Taichi solver（从未使用） | `example_minimal_joint.py:30` | 删除 stale import |

### Phase 2: couple2 与 couple3 架构对齐

**新建共享模块**:
- **`src/VMuscle/bicep_helpers.py`** (111 行): 提取 `ELBOW_PIVOT`、`ELBOW_AXIS`、`extract_radius_mesh()`、`build_elbow_model()` 到共享模块
- **扩展 `src/VMuscle/controllability.py`** (+73 行): 添加 `parse_levels()` 和 `run_eval_sweep()`
- **扩展 `src/VMuscle/usd_io.py`**: 添加 `build_bone_prim_map()`

**couple2 重写** (`examples/example_couple2.py`, 432->228 行，-47%):
- 移除 11 个模拟参数 CLI 参数
- 移除 `main()` 中的 cfg 路径覆盖
- 添加 `setup_couple2()` 工厂函数（镜像 couple3 模式）
- 创建 `data/muscle/config/bicep_coupled.json`（基于 bicep.json + coupling 段）

**couple3 瘦身** (`examples/example_couple3.py`, 477->275 行，-42%):
- 导入共享模块替代本地定义
- `_activation_schedule` 委托给 `muscle_common.activation_ramp`
- `_run_eval_sweep` 委托给 `controllability.run_eval_sweep`

### Phase 3: src/ 废弃代码清理

**geo.py** (370->245 行，-34%):
- 移除 `sys.path.append(os.getcwd())`
- 移除死函数：`get_args()`、`test_geo_vtk()`、`test_animation()`
- 移除重复的 `_pairListToDict` 定义
- 翻译所有中文注释为英文
- 替换 `print()` 为 `logging`

**constraints.py** — 消除三重复制:
- 提取 `_compute_fiber_rest_data()` 共享方法

**usd_io.py** — 消除内部重复:
- `_fix_tet_winding` -> 导入 `mesh_utils.fix_tet_winding`
- `_extract_surface_tris` -> 导入 `mesh_io.build_surface_tris`
- `_Y_TO_Z` -> 导入 `mesh_io._Y_TO_Z`

**其他**:
- `muscle_warp.py`: 移除无用的 `load_config`/`get_bbox` re-export
- `config.py`: 清理尾部空行
- `util.py`: 添加 broken import 注释说明

### Phase 4: Config 清理

| 操作 | 文件 | 说明 |
|------|------|------|
| 删除死字段 | `bicep_dgf.json` | 移除 `optimal_fiber_length`、`v_max_scale`（从未被代码读取） |
| 补全缺失字段 | `bicep_xpbd_millard.json` | 添加 `repair_sigma_min: 0.1`、`post_smooth_iters: 5` |
| 归档孤立 config | `bicep_fibermillard.json`、`bicep_hybrid.json` | 移到 `data/muscle/config/archived/` |
| 新建 config | `bicep_coupled.json` | couple2 专用，含 coupling 段 |

---

## 总统计

| 指标 | Round 1 | Round 2 | 合计 |
|------|---------|---------|------|
| 修改文件 | 6 | 20 | 26 |
| 新增行 | 183 | 610 | 793 |
| 删除行 | 305 | 615 | 920 |
| 净变化 | -122 | -5 | **-127** |
| 新建共享模块 | 0 | 1 (`bicep_helpers.py`) | 1 |
| 新建 config | 0 | 1 (`bicep_coupled.json`) | 1 |
| Bug 修复 | 0 | 5 | 5 |

## 验证

```bash
# couple2 smoke test
RUN=example_couple2 uv run main.py --auto --steps 50 --no-usd  # OK

# couple3 smoke test
RUN=example_couple3 uv run main.py --auto --steps 50 --no-usd  # OK

# standalone muscle
RUN=example_muscle_warp uv run main.py --steps 5               # OK

# 语法检查: 所有修改文件通过 ast.parse
# 导入检查: 所有共享模块导入正确
```

## 架构变化总结

重构前后的 example 架构对比：

```
重构前:
  example_couple2.py  (432 行, 11 个模拟 CLI 参数, 本地定义 elbow model)
  example_couple3.py  (477 行, 12 个模拟 CLI 参数, 本地定义 elbow model)
  scripts/*.py        (各自重复 setup 代码)

重构后:
  example_couple2.py  (228 行, setup_couple2(), 导入 bicep_helpers)
  example_couple3.py  (275 行, setup_couple3(), 导入 bicep_helpers)
  scripts/*.py        (导入 setup_couple3(), 纯壳子)
  src/VMuscle/
    bicep_helpers.py       (共享 elbow model)
    controllability.py     (+parse_levels, +run_eval_sweep)
    usd_io.py              (+build_bone_prim_map)
```

所有模拟参数统一从 JSON config 读取，CLI 仅用于工作流控制。

## 遗留 TODO（未在本次处理）

以下问题已识别但不在本次范围（按优先级排序）：

1. **`_import_opensim()` 四重重复**: 4 个 osim scripts 各自定义，应提取到 `src/`
2. **`compute_fiber_stretches` 五重重复**: `example_vbd_*.py` 应导入 `muscle_common` 版本
3. **`example_tetmesh_import*.py` 无 `main()`**: 模拟代码在模块级执行
4. **`example_human_import.py` 全中文注释 + 无 `main()`**
5. **`solver_muscle_bone_coupled` Taichi vs Warp**: ~400 行重叠，可提取 base class
6. **`activation.py` Warp kernel 硬编码 tau**: 忽略 config 中的 `tau_act`/`tau_deact`
7. **`example_couple.py`**: Taichi 版 couple，含 9 个模拟 CLI 参数（将随 Taichi 废弃）
8. **`load_config` 无 schema 验证**: 拼写错误的 JSON 字段被静默忽略
9. **`sliding_ball_helpers.py` `object.__new__` bypass**: 应使用 `MuscleSim.from_procedural()`
