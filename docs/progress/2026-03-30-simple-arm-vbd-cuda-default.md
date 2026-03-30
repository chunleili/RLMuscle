# 2026-03-30 SimpleArm VBD 默认 CUDA

## 背景

`examples/example_vbd_mujoco_simple_arm.py` 默认把 Warp 设备固定成 `cpu`，导致即使机器上有 CUDA，也会走 CPU VBD 路径。对当前 SimpleArm case，这会显著放大 `SolverVBD.step()` 的耗时。

## 改动

- 在 `examples/example_vbd_mujoco_simple_arm.py` 中增加 `_get_default_warp_device()`。
- 默认设备改为：`cuda:0 if wp.is_cuda_available() else cpu`。
- 增加 `--device` 参数，允许手动覆盖设备。
- verbose 日志中打印实际设备，便于排查性能问题。

## 验证

- `uv run python -m py_compile examples/example_vbd_mujoco_simple_arm.py`
- 1-step smoke test 确认默认日志打印 `device=cuda:0`
- 整体运行时间：
  - 旧版默认 CPU：约 `277.3s`
  - 新版默认 CUDA：约 `143.3s`

## XPBD 对比（同网格、同 dt、同 iterations=20、CUDA）

口径：固定两端的同一个 cylinder tet mesh，施加相同初始轴向拉伸后，重复 `400` 次 solver step，对比单步求解成本。

- `SolverXPBD`（纯软体）：`2.176 ms/step`
- `SolverVBD`（纯软体）：`9.459 ms/step`
- `SolverVBD + vmuscle`：`12.292 ms/step`

结论：

- 在这个 case 上，`XPBD` 纯软体求解明显更快，约为 `VBD` 纯软体的 `4.35x`。
- `vmuscle` 会在 VBD 基础上继续增加开销，但不是唯一慢点；VBD 自身的 per-color / per-iteration kernel 结构已经偏重。
