# 2026-03-30 耦合可控性进展

## 已完成

- 新增共享可控性层 `src/VMuscle/controllability.py`，统一配置、映射、力矩整形与 sweep 评估。
- 在 Warp 与 Taichi 两个 coupled solver 中接入：
  - excitation -> effective activation 的一阶 dynamics；
  - joint axis projection；
  - torque EMA 与 slew rate 限制；
  - 共享 preset `legacy`、`linear_tuned`、`smooth_nonlinear`、`rate_limited`。
- 修改 `src/VMuscle/activation.py`，允许 activation 释放到 `0.0`。
- 为 `examples/example_couple.py` 与 `examples/example_couple2.py` 增加：
  - `--preset`
  - `--eval`
  - 参数覆盖入口
  - sweep JSON 输出
- 修正 `example_couple --eval` 为纯 headless，不再走 GUI。
- 为 Warp/Taichi cache 设置项目内默认目录，减少机器环境差异。
- 增加 `tests/test_controllability.py`，覆盖 activation release、力矩限幅、rate limit、activation 映射单调性。

## 关键发现

- `legacy` 在 Warp 侧存在明显饱和与过冲，`activation=1.0` 的稳态力矩远大于中低 activation，且 release 后无法在评估窗口内稳定收敛。
- `smooth_nonlinear` 在 Warp 侧表现最好，稳态力矩与角度对 activation 保持单调，且 release 后 1 step 内满足 settle 判据。
- Taichi 侧三种新方案都优于 `legacy`，但仍存在 `activation=0.7` 左右的力矩符号翻转，说明当前残余问题更偏向后端数值响应，而不是单纯映射函数。

## 工作流变化

- `example_couple2` 默认不再自动导出 USD，需要显式传 `--export-usd`。
- 两个 example 都支持 `--eval` 直接产出可比对的 controllability 报告。
- `main.py` 与 example 脚本会默认使用项目内 `.cache/warp` 与 `.cache/taichi`。

## 下一步

- 若继续优化 Taichi 侧可控性，优先检查 `reaction_accum` 在中高激活时的符号变化，以及 joint axis projection 前后的力矩方向稳定性。
