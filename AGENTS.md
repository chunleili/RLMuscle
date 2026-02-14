# 规范

- 使用 `.venv` 中的 python 运行和测试代码，不要留下临时测试文件。
- `AGENTS.md` 是简洁的全局计划。`PROGRESS.md` 是详细日志（下一步 + 已完成）。
- 每一步都要谨慎，改动最小化，结构清晰易读。这是快速原型，之后可以重构。
- 可复用代码放 `src/`，示例放 `examples/`（各自独立可运行，简单有教学意义）。
- 代码（包括注释）使用英文，文档 markdown 使用中文。

# 计划

## example_dynamics
单摆 + motor 扭矩控制 demo，基于 Newton 刚体动力学，使用 layered USD 输出。详见 PROGRESS.md。
