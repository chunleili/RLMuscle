# 代码库全面清理与重构

**日期**: 2026-04-04  
**分支**: dev  
**Commits**: `39ee427`, `c7010ee`, `fe13f9f`, `5efb279`, `2f476f0`  
**关联文档**: [2026-04-03-refactor-dedup.md](2026-04-03-refactor-dedup.md)（前次重构，SimpleArm + Sliding-ball 去重）  
**状态**: 成功

## 概述

在前次重构（SimpleArm/Sliding-ball 去重，净减 ~340 行）基础上，对整个代码库进行了五轮清理：

1. **Commit `39ee427`**: couple3 架构标准化——提取 `setup_couple3()` 工厂函数，移除 12 个模拟 CLI 参数，简化 scripts
2. **Commit `c7010ee`**: 全面审查与修复——5 个 bug 修复、couple2 对齐 couple3 架构、src/ 废弃代码清理、config 规范化
3. **Commit `fe13f9f`**: 统一命名——`muscle_type`/`curve_type` 重命名为 `hill_model_type`，默认值统一为 `"millard"`
4. **Commit `5efb279`**: 修复硬编码参数——activation.py tau 参数化、消除 `object.__new__` hack、清理冗余 `sys.path.insert`
5. **Commit `2f476f0`**: 为所有 JSON config 添加 `metainfo` 元信息（description/used_by/created）

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

## Round 3: 统一命名 (`fe13f9f`)

将分散在代码库中的 `muscle_type` / `curve_type` 参数统一重命名为 `hill_model_type`，消除同一概念的多名称混乱。

### 核心变更

| 范围 | 旧名 | 新名 | 影响文件 |
|------|------|------|----------|
| osim scripts 函数参数 | `muscle_type` | `hill_model_type` | `osim_simple_arm.py`, `osim_sliding_ball.py` |
| XPBD config JSON 字段 | `curve_type` | `hill_model_type` | `example_xpbd_coupled_simple_arm.py`, `run_simple_arm_comparison.py` |
| CLI 参数 | `--muscle-type` | `--hill-model-type` | `osim_simple_arm.py`, `osim_sliding_ball.py` |

**默认值统一**: 所有入口默认 `hill_model_type="millard"`（Millard 2012 是当前主力模型）。

---

## Round 4: 修复硬编码参数 (`5efb279`)

### 4.1 activation.py — tau 参数化

Warp kernel `dgf_activation_dynamics` 中 `tau_a=0.015`、`tau_d=0.060`、`b=10.0` 原为函数体内硬编码。改为显式函数参数：

```python
# Before
@wp.func
def dgf_activation_dynamics(excitation, activation, dt, min_activation=0.0):
    tau_a = 0.015; tau_d = 0.060  # hardcoded

# After
@wp.func
def dgf_activation_dynamics(excitation, activation, dt,
                            tau_a, tau_d, b, min_activation):
```

同步更新 `activation_dynamics_step_np()` 接受 `b` 参数。

### 4.2 sliding_ball_helpers.py — 消除 `object.__new__` hack

将 ~60 行手动 `object.__new__(MuscleSim)` + 20+ 属性赋值替换为单次 `MuscleSim.from_procedural()` 调用。

为支持 sliding ball 场景，扩展 `from_procedural()` 新增 3 个参数：
- `num_substeps` — substep 数（影响 dt 计算）
- `contraction_ratio` — 初始收缩比
- `fiber_stiffness_scale` — 纤维约束刚度缩放

### 4.3 清理冗余 sys.path.insert

项目通过 `pyproject.toml` 安装 VMuscle 包，`sys.path.insert(0, "src")` 是冗余的。

| 文件 | 变更 |
|------|------|
| `scripts/run_couple3_curves.py` | 移除 `src/` 路径 |
| `scripts/experiments/sweep_post_smooth.py` | 移除 `src/` 路径 |
| `tests/conftest.py` | 改为添加项目根目录（覆盖 `examples.*` 导入） |
| 4 个 test 文件 | 移除各自的 `sys.path.insert`，由 conftest.py 统一处理 |

### 4.4 其他规范化

- `constraints.py`: `print("Warning: ...")` → `warnings.warn(..., DeprecationWarning)`
- `config.py`: `print()` → `logging.info()`
- `sliding_ball_helpers.py`: `print()` → `logging.info()`

**统计**: 13 文件，55 新增，117 删除

---

## Round 5: JSON config 元信息 (`2f476f0`)

为所有 12 个 JSON config 文件添加 `metainfo` 字段，记录用途、使用者和创建日期：

```json
{
  "metainfo": {
    "description": "Brief English description of purpose",
    "used_by": ["examples/xxx.py", "scripts/yyy.py"],
    "created": "2026-02-10"
  },
  ...
}
```

| Config | description |
|--------|-------------|
| `bicep.json` | Default bicep muscle with generic fiber constraints (VBD solver) |
| `bicep_coupled.json` | VBD muscle-bone coupled bicep with smooth nonlinear coupling |
| `bicep_fibermillard_coupled.json` | Millard 2012 energy constitutive model, muscle-bone coupled |
| `bicep_dgf.json` | DGF constitutive muscle model with attachnormal constraints |
| `bicep_xpbd_millard.json` | XPBD integrator with Millard 2012 energy model and SVD repair |
| `simpleArm/config.json` | 1-DOF elbow joint, multi-solver comparison |
| `slidingBall/config.json` | 1-DOF sliding ball with DGF Hill-type muscle (VBD) |
| `slidingBall/config_xpbd_dgf.json` | XPBD-DGF sliding ball |
| `slidingBall/config_xpbd_millard.json` | XPBD-Millard sliding ball |
| `Human/human_import2.json` | Full human skeleton/muscle import |
| `archived/bicep_fibermillard.json` | [ARCHIVED] Millard without coupling |
| `archived/bicep_hybrid.json` | [ARCHIVED] Hybrid fiber + sigma0 |

3 个 slidingBall config 的旧顶层 `description` 字段迁移至 `metainfo` 内。`load_config()` 无需修改，`metainfo` 作为属性对现有代码透明。

**统计**: 12 文件，87 新增，3 删除

---

## 总统计

| 指标 | R1 | R2 | R3 | R4 | R5 | 合计 |
|------|----|----|----|----|----|----|
| 修改文件 | 6 | 20 | 6 | 13 | 12 | 57 |
| 新增行 | 183 | 610 | 30 | 55 | 87 | 965 |
| 删除行 | 305 | 615 | 30 | 117 | 3 | 1070 |
| 净变化 | -122 | -5 | 0 | -62 | +84 | **-105** |

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

## 遗留 TODO

以下问题已识别但未处理（按优先级排序）：

1. ~~**`_import_opensim()` 四重重复**~~ → P3 合并 osim scripts 时已解决 (`2b05971`)
2. **`compute_fiber_stretches` 五重重复**: `example_vbd_*.py` 应导入 `muscle_common` 版本
3. ~~**`example_tetmesh_import*.py` 无 `main()`**~~ → P2 已补齐 (`2b05971`)
4. ~~**`example_human_import.py` 全中文注释 + 无 `main()`**~~ → P2 已补齐 (`2b05971`)
5. **`solver_muscle_bone_coupled` Taichi vs Warp**: ~400 行重叠，可提取 base class（将随 Taichi 废弃）
6. ~~**`activation.py` Warp kernel 硬编码 tau**~~ → Round 4 已参数化 (`5efb279`)
7. **`example_couple.py`**: Taichi 版 couple，含 9 个模拟 CLI 参数（将随 Taichi 废弃）
8. **`load_config` 无 schema 验证**: 拼写错误的 JSON 字段被静默忽略
9. ~~**`sliding_ball_helpers.py` `object.__new__` bypass**~~ → Round 4 用 `from_procedural()` 替换 (`5efb279`)
10. **`activation.py` Warp kernel `update_activations` 疑似死代码**: 无外部调用方（仅 NumPy 版 `activation_dynamics_step_np` 被使用）
11. **src/ 中 ~25 处 `print()` 未迁移到 `logging`**: `muscle_common.py`, `mesh_io.py`, `muscle_warp.py` 等
