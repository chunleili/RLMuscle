# 规范
- 如果你是初次运行，务必首先查看 `README.md` 中的运行说明（如git clone external/newton）。
- 使用 `.venv` 中的 python 运行和测试代码，可以写临时测试文件，但是写完要删除。
- `AGENTS.md` 是简洁的全局计划。`PROGRESS.md` 是详细日志（下一步 + 已完成）。
- 每一步都要谨慎，改动最小化，结构清晰易读。这是快速原型，之后可以重构。
- 可复用代码放 `src/`，示例放 `examples/`（各自独立可运行，简单有教学意义）。
- 代码（包括注释）使用英文，文档 markdown 使用中文，尽量不要更改README。

# 计划

## example_dynamics (done)
单摆 + motor 扭矩控制 demo，基于 Newton 刚体动力学，使用 layered USD 输出。详见 PROGRESS.md。

## example_couple (on-going)
使用muscle_core实现骨骼和肌肉的耦合。其中muscle_core是参考muscle.py(taichi)的warp版本（只实现其中核心step部分）。可视化和IO采用layered_usd + gl流程（见example_usd_io）。muscle_core不采用taichi和geo流程。 example_couple.py调用solver_muscle_bone_coupled作为solver。该solver分别调用solver_volumetric_muscle（作为muscle_core的wrapper）和solverFeatherStone（来自newton），来实现肌肉和骨骼的耦合。