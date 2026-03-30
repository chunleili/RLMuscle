# 耦合可控性改进计划

## 目标

提升 `example_couple` 与 `example_couple2` 的耦合可控性，使 activation 输入对关节角度与关节力矩产生更连续、可区分、可回落的响应。

## 分阶段执行

### Stage 0: 基线与运行环境

- 状态：已完成
- 初始化 `external/newton`，补齐 `uv` 依赖。
- 将 Warp 与 Taichi cache 重定向到项目内 `.cache/`，避免本机权限与全局缓存干扰评估。
- 为两个 example 增加 headless sweep 入口，输出 JSON 到 `output/`。

### Stage 1: 调整现有耦合参数

- 状态：已完成
- 抽离共享控制配置 `CouplingControlConfig`。
- 将原先分散在示例里的 `k_coupling`、`max_torque`、被动比例、EMA、slew rate 参数统一到 `src/VMuscle/controllability.py`。
- 提供 `legacy` 与 `linear_tuned` 两套可复现实验基线。

### Stage 2: 增加 activation dynamics

- 状态：已完成
- 复用 `src/VMuscle/activation.py` 的一阶 activation dynamics。
- 让 excitation 和 solver 内 effective activation 分离，按 substep 更新。
- 允许 activation 真实衰减到 `0.0`，避免 release 后残留最小激活值。

### Stage 3: 探索新的耦合方案

- 状态：已完成
- 增加 `smooth_nonlinear` 映射，用于改善低中 activation 分辨率。
- 增加 `rate_limited` 模式，用于限制力矩跃迁并抑制尖峰。
- 对单自由度关节在 shaping 前先做 joint axis projection，减少非关节轴方向扭矩污染。

### Stage 4: 自动评估与默认方案

- 状态：已完成
- 添加 activation sweep 评估，统计 steady torque、steady angle、overshoot、release settle。
- Warp 版本默认方案确定为 `smooth_nonlinear`。
- Taichi 版本共用相同基础设施与默认 preset，但仍保留 `0.7` 附近符号翻转残余问题，后续若继续优化应优先针对该点排查。

## 当前结论

- `example_couple2`：`smooth_nonlinear` 已满足“更大 activation 对应更大稳态扭矩，release 后快速归零”的目标。
- `example_couple`：整体较 legacy 明显改善，但所有新方案在 `0.7` 附近仍有非单调现象；默认仍保留 `smooth_nonlinear`，因为它与 Warp 侧一致，且高 activation 输出更强。
