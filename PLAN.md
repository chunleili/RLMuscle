# PLAN

1. 仅修改 `pyproject.toml` 中 `newton` source 的 `editable` 配置（移除 `editable=true`），不改其它项。
2. 优先使用 `uv run` 执行测试；若 workspace 结构限制 `uv sync`，则用 `uv run --no-project` 继续实验。
3. 增加回归测试用例，覆盖耦合稳定性关键关系（torque 发散、惯量增大抑制峰值等）。
4. 继续做参数实验（质量、子步/solver profile），并以表格补充到 `doc/plan-couple.md`。
5. 更新 `PROGRESS.md`，提交并创建 PR。
