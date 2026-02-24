# 2026-02-21: 耦合计划与 Warp 移植

## 耦合计划与测试脚手架

- 按要求仅修改 `pyproject.toml`：去除 `newton` 的 `editable=true`。
- 使用 `uv run --no-project` 完成 benchmark 执行与扩展实验（矩阵、质量敏感性、子步对比）。
- 新增 `tests/test_plan_couple_regression.py` 回归测试，覆盖 4 个关键稳定性结论。
- 在 `doc/plan-couple.md` 增补质量敏感性与子步/solver profile 对比表格。

## Warp 移植审查与修复

- CPU Jacobi 与 Taichi 数值一致 (max_err < 2.3e-4)；CUDA GS 发散已修复
- `use_jacobi` 默认改 True；GS 分支 6 处改 `wp.atomic_add`
- 缺失: `reaction_accum`（骨骼耦合）、`Visualizer` 类
