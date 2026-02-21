# 下一步

- 在 CUDA 上做完整的 Taichi vs Warp 可视化对比（当前 Taichi 在同进程下有 LLVM 崩溃问题，需分进程运行）
- 骨骼-肌肉耦合（`reaction_accum`）Warp 版尚未移植

# 已完成

## 2026-02-21: Warp 移植审查与修复

### 审查结论
- **CPU Jacobi**: Warp 与 Taichi 数值一致，max_err < 2.3e-4 ✅
- **CPU Gauss-Seidel**: max_err ~0.05，属于并行 GS 固有非确定性 ✅
- **CUDA Gauss-Seidel (修复前)**: 发散爆炸，NaN/Inf ❌
- **CUDA Jacobi**: 稳定 ✅

### 修复内容 (`src/VMuscle/muscle_warp.py`)
1. `use_jacobi` 默认值 `False` → `True`
2. Gauss-Seidel 分支 6 处 `pos[pt] = pos[pt] + delta` → `wp.atomic_add(pos, pt, delta)`

### 代码审查发现
- 核心数学公式（SVD、XPBD、约束求解）移植正确
- 缺失: `reaction_accum` / `clear_reaction`（骨骼耦合功能）
- 缺失: `Visualizer` 类（预期行为，Warp 版不含可视化）
- 小问题: `show_auxiliary_meshes` 默认值不一致（Warp=True, Taichi=False）
# 测试命令

## pytest 自动化（推荐）

```bash
# 一键全跑
uv run pytest -v

# 跳过慢测试
uv run pytest -m "not slow" -v

# 跳过 CUDA 测试
uv run pytest -m "not cuda" -v
```

## 手动运行（仍可用）

```bash
uv run python tests/test_muscle_warp_vs_taichi.py --mode jacobi --steps 100
uv run python tests/test_warp_cpu_vs_cuda.py
uv run python tests/test_warp_cuda_jacobi.py
uv run python tests/test_visual_comparison.py
```
---

*旧记录: 骨骼与肌肉耦合阻碍见 `example_couple.py` 和 `solver_muscle_bone_coupled.py`*
