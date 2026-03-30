# 2026-03-30 耦合可控性实验总结

## 实验目的

按顺序验证以下三类改动是否能提升 `example_couple` 与 `example_couple2` 的 controllability：

1. 调整耦合参数
2. 引入 activation dynamics
3. 探索新的耦合映射

## 评估方法

- 激活水平：`0.0, 0.1, 0.3, 0.5, 0.7, 1.0`
- 每个 episode：
  - 先 warmup 零激活
  - 再 hold 当前 activation
  - 最后 release 到 `0.0`
- 记录指标：
  - `steady_axis_torque_abs`
  - `steady_joint_angle_abs`
  - `overshoot_axis_torque`
  - `settle_steps_after_release`
  - `monotonic_steady_torque`
  - `monotonic_steady_angle`

## 主要命令

```powershell
.\.venv\Scripts\python.exe examples\example_couple2.py --eval --preset smooth_nonlinear
.\.venv\Scripts\python.exe examples\example_couple2.py --eval --preset rate_limited
.\.venv\Scripts\python.exe examples\example_couple.py --eval --preset smooth_nonlinear --eval-hold-steps 60 --eval-release-steps 60
.\.venv\Scripts\python.exe examples\example_couple.py --eval --preset linear_tuned --eval-hold-steps 60 --eval-release-steps 60
.\.venv\Scripts\python.exe examples\example_couple.py --eval --preset rate_limited --eval-hold-steps 60 --eval-release-steps 60
```

## `example_couple2` 结果

| preset | torque 单调 | angle 单调 | 1.0 稳态力矩绝对值 | 1.0 settle | 结论 |
| --- | --- | --- | ---: | ---: | --- |
| legacy | 否 | 否 | 22.5446 | 未收敛 | 明显饱和，过冲极大 |
| linear_tuned | 否 | 是 | 0.0572 | 1 | 中低激活可分辨，但 1.0 略低于 0.7 |
| smooth_nonlinear | 是 | 是 | 0.0602 | 1 | 最平衡，作为默认 |
| rate_limited | 否 | 是 | 0.0539 | 1 | 比 legacy 稳，但 1.0 仍低于 0.7 |

### `smooth_nonlinear` 稳态力矩绝对值

| activation | torque_abs |
| ---: | ---: |
| 0.1 | 0.0022 |
| 0.3 | 0.0156 |
| 0.5 | 0.0369 |
| 0.7 | 0.0497 |
| 1.0 | 0.0602 |

结论：

- Warp 侧已经达到目标定义中的“更大 activation -> 更大输出”和“release 后快速归零”。
- `smooth_nonlinear` 同时优于 `linear_tuned` 和 `rate_limited`。

## `example_couple` 结果

| preset | torque 单调 | angle 单调 | 1.0 稳态力矩绝对值 | 1.0 settle | 结论 |
| --- | --- | --- | ---: | ---: | --- |
| smooth_nonlinear | 否 | 是 | 0.3986 | 11 | 默认保留，与 Warp 侧一致，高激活输出更强 |
| linear_tuned | 否 | 是 | 0.3701 | 11 | 中低激活分辨率略高，但 0.7 处翻转更明显 |
| rate_limited | 否 | 是 | 0.3409 | 13 | 更保守，但 settle 更慢 |

### Taichi 侧共同现象

- `0.1 -> 0.5` 区间响应基本可区分，且 activation 降回 `0.0` 后都能最终停下来。
- `0.7` 附近出现符号翻转，导致稳态力矩绝对值低于 `0.5`，单调性仍不满足。
- `1.0` 时三种新方案都显著优于 `legacy`，不会出现 Warp 版旧方案那种瞬时爆炸式饱和。

## 总结

- 已完成三类改动的实现与对比。
- 默认 preset 选择 `smooth_nonlinear`。
- `example_couple2` 已获得明显更好的 controllability。
- `example_couple` 已从“强饱和/弱可控”改善到“中低激活可分辨、release 可回落”，但 Taichi 后端仍有中高激活方向不稳定的残留问题。

## Follow-up: 端点约束简化

### 目标

- 用户指出最下端网格存在明显局部扭曲，希望保留 controllability 的同时收敛端点约束。

### 最终方案

- 默认端点约束统一改为 `attach` only，包括 proximal 与 distal。
- `attachnormal` 仅保留兼容旧配置的 deprecated 通道，不再进入任何默认工作流。
- 在 `src/VMuscle/constraints.py` 中保留 `source_nearest_group` / `target_group`，用于在需要时继续把端点分到不同骨组。

### 最终结果

输出：`output/attach_only_default_scan.json`

| 区域 | `max_rigid_rms` | `max_pair_error` | `max_seed_disp_std` |
| --- | ---: | ---: | ---: |
| proximal | 0.00221 | 0.00762 | 0.01164 |
| distal | 0.00200 | 0.00669 | 0.00276 |

补充：

- `max_joint_abs = 3.06601`
- `example_couple2_eval_smooth_nonlinear.json` 仍保持：
  - `monotonic_steady_torque = true`
  - `monotonic_steady_angle = true`

### 结论

- 当前最核心、最稳定的修改就是默认移除 `attachnormal`，统一使用 `attach`。
- 这既降低了 proximal / distal 两端的局部扭曲，也保留了当前 Warp 侧的 controllability 表现。

## 输出文件

- `output/example_couple2_eval_legacy.json`
- `output/example_couple2_eval_linear_tuned.json`
- `output/example_couple2_eval_smooth_nonlinear.json`
- `output/example_couple2_eval_rate_limited.json`
- `output/example_couple_eval_smooth_nonlinear.json`
- `output/example_couple_eval_linear_tuned.json`
- `output/example_couple_eval_rate_limited.json`
- `output/attach_only_default_scan.json`
