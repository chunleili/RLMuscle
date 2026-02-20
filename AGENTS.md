# INSTRUCTION
- 如果你是初次运行，务必首先查看 `@README.md` 中的运行说明（如git clone external/newton）。
- 使用 `@.venv` 中的 python 运行和测试代码，必要时可以写临时测试文件。仅保留重要的测试文件放到tests/文件夹下。总是测试直到正常运行后才结束任务。
- `@AGENTS.md` 是简洁的全局计划。`@PROGRESS.md` 放日志（下一步 + 已完成）。`@PLAN.md`放最详细的执行方案。`@PROGRESS.md`和`@PLAN.md`的内容不断更新，过期内容需要清理。
- 每一步都要谨慎，改动最小化，结构清晰易读。这是快速原型，之后可以重构。
- 可复用代码放 `@src/`，示例放 `@examples/`（各自独立可运行，简单有教学意义）。
- 代码（包括注释）使用英文，文档 markdown 尽量使用中文，尽量不要更改README。
- 当作出功能修改时，指出关键的工作流的改变（如有）。
- 当进行bugfix时，务必反复迭代。可以写入侵入性的代码，可以采用python logging输出debug代码，输出可以放到临时性的`log.md`。


# Workflow
- use uv to manage the project and run the project if possible.
- Be sure to test it until get expected results after you implemented a new feature or fixed a bug. 
- When you test something, you don't need to write a formal test, a invasive but simple code is allowed. 
