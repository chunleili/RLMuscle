# 2026-02-24: 主动静止长度收缩 — 可控性改进

- **核心修改**：纤维约束从 `C = ||Fw||`（目标零长度）改为 `C = ||Fw|| - target_stretch`（目标受 activation 控制的静止长度）
- `target_stretch = 1.0 - belly_factor × activation × contraction_ratio`
- 新增 SimConfig 参数：`contraction_ratio`（默认 0.4）、`fiber_stiffness_scale`（默认 200.0，替代硬编码 10000.0）
- **可控性显著改善**（MuJoCo solver）：
  - monotonicity: 0.50/0.75 → **1.00**（完美单调）
  - act_angle_corr: 0.70 → **0.90**（接近 A 级阈值）
  - 不同激活水平终于产生明显区分的稳态角度（0→0.37→0.88→0.88→0.90 rad）
- **已知限制**：settling_steps 从 0 退化到 ~120（因为肌肉真正发生了物理收缩/松弛，需要时间回到静止），阻止了 B 级评定（阈值 90）
- **稳定性**：featherstone 和 mujoco 系列无退化，xpbd/semi_implicit 维持原状
- 修改文件：`muscle.py`（SimConfig + MuscleSim.__init__ + solve_constraints + tet_fiber_update_xpbd）、`bicep.json`
