# 代码库文件结构优化

**日期**: 2026-04-04  
**分支**: dev  
**Commit**: `2b05971`  
**关联 Commits**: `fe13f9f`（hill_model_type 统一命名，紧随本次合并）  
**关联文档**: [2026-04-04-codebase-cleanup.md](2026-04-04-codebase-cleanup.md)（前次代码逻辑清理）  
**状态**: 成功

## 概述

在代码逻辑清理（bug 修复、去重、架构对齐）完成后，对文件结构进行优化：删除死代码、补齐 main()、合并近重复文件。

## P1: 删除死代码（3 个文件）

| 文件 | 原因 |
|------|------|
| `src/VMuscle/util.py` (21 行) | 唯一函数 `add_aux_meshes()` 导入不存在的 `read_auxiliary_meshes`（2026-03-10 已删除） |
| `examples/example_warp_mesh.py` (179 行) | 零 VMuscle 导入，Warp SDK PBD cloth demo 拷贝 |
| `examples/example_minimal_joint.py` (170 行) | 零 VMuscle 导入，Newton SDK 双摆 demo 拷贝 |

同步修复：`examples/example_human_import2.py` 移除 `add_aux_meshes` 的 broken import 和调用。

注：`examples/example_muscle_taichi.py`（8 行 stub）按用户要求保留，作为 Taichi 入口一致性。

## P2: 补齐 main()（3 个 example）

以下文件原为模块级执行，无法通过 `RUN=` 正常启动。包裹为 `def main():` + `if __name__ == "__main__": main()`。

| 文件 | 行数 | 附加变更 |
|------|------|----------|
| `examples/example_human_import.py` | 75→82 | 翻译中文注释为英文 |
| `examples/example_tetmesh_import.py` | 154→156 | 无 |
| `examples/example_tetmesh_import_gl.py` | 107→112 | 无 |

## P3: 合并 osim scripts（4→2）

### osim_simple_arm_dgf.py + osim_simple_arm_millard.py → osim_simple_arm.py

- **合并方式**: `osim_simple_arm(cfg, muscle_type="dgf"|"millard")`
- **差异处理**: muscle 创建逻辑按 type 分支（DGF: `DeGrooteFregly2016Muscle`, Millard: `Millard2012EquilibriumMuscle`）
- **向后兼容**: `osim_simple_arm_dgf(cfg)` 和 `osim_simple_arm_millard(cfg)` 作为 thin wrapper 保留
- **统一**: `_import_opensim()` 合并为单一版本，同时 patch DGF 和 Millard 类

### osim_sliding_ball.py + osim_sliding_ball_millard.py → osim_sliding_ball.py

- **合并方式**: `osim_sliding_ball(..., muscle_type="dgf"|"millard")`
- **向后兼容**: `osim_sliding_ball_millard()` 作为 thin wrapper 保留

### 更新的依赖

- `run_simple_arm_comparison.py`: 导入从 `osim_simple_arm_dgf` / `osim_simple_arm_millard` 改为 `osim_simple_arm`
- `run_sliding_ball_comparison.py`: 导入从 `osim_sliding_ball_millard` 改为 `osim_sliding_ball`

## P4: 合并近重复 examples（2→1，部分执行）

### 已合并：XPBD coupled simple arm

`example_xpbd_coupled_simple_arm.py` + `example_xpbd_coupled_simple_arm_millard.py` → `example_xpbd_coupled_simple_arm.py`

- **分发机制**: `cfg["xpbd"]["curve_type"]`（`"dgf"` 默认或 `"millard"`）
- **3 处 curve_type 分支**:
  1. Sim 构造：Millard 模式传入 sigma0/lambda_opt
  2. Substep：Millard 额外调用 `clear_forces()` + `accumulate_active_fiber_force()`
  3. Force 计算：DGF 用 `active_force_length()`，Millard 用 `mc.fl.eval()`
- **更新**: `run_simple_arm_comparison.py` 的 `run_xpbd_millard()` 设置 `curve_type="millard"`

### 未合并：XPBD sliding ball

`example_xpbd_dgf_sliding_ball.py` 和 `example_xpbd_millard_sliding_ball.py` **保持不变**。

**原因**: 两者差异远大于预期（~50% 而非 95%）:
- DGF: 约束驱动收缩（`TETFIBERDGF` + 动态 `contraction_factor`）
- Millard: 显式力注入（`clear_forces()` + `accumulate_active_fiber_force()`）
- substep 循环、config 结构、constraint setup 均不同
- 强行合并会产生大量 if/else 分支，可读性反而下降

## P5: 拆分 muscle_warp.py — 暂不执行

用户决定暂不拆分（2743 行）。Warp kernel 跨模块编译风险高，当前功能稳定。

## 统计

| 指标 | 数值 |
|------|------|
| 删除文件 | 7 (util.py, 2 examples, 1 millard example, 3 osim scripts) |
| 修改文件 | 9 |
| 新增文件 | 2 (osim_simple_arm.py, 本文档) |
| 净删除 | **-1254 行** (483 新增, 1737 删除) |

## 验证

```bash
# Core examples
RUN=example_couple2 uv run main.py --auto --steps 50 --no-usd   # OK
RUN=example_couple3 uv run main.py --auto --steps 50 --no-usd   # OK
python -c "from VMuscle.muscle_warp import MuscleSim; print('OK')"  # OK

# 语法检查: 28 个 .py 文件全部通过 ast.parse
# 残留引用检查: 无残留指向已删除文件的导入
```

## 复现命令

```bash
# 合并后的 osim scripts
uv run python scripts/osim_simple_arm.py --muscle-type dgf
uv run python scripts/osim_simple_arm.py --muscle-type millard

# 合并后的 XPBD simple arm (DGF 默认)
RUN=example_xpbd_coupled_simple_arm uv run main.py

# 合并后的 XPBD simple arm (Millard 通过 config)
# 在 data/simpleArm/config.json 的 xpbd 段中添加 "curve_type": "millard"
```
