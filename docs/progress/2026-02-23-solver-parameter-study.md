# 2026-02-23: Solver 参数研究与优化

## 默认 solver 切换为 SolverMuJoCo(cpu)

- 基于参数研究实验结论，将 `SolverMuscleBoneCoupled` 默认 bone solver 从 `SolverFeatherstone` 切换为 `SolverMuJoCo(solver="cg", use_mujoco_cpu=True)`。
- 理由：最快（30.9 ms/step）、最稳定（energy_ratio 最低）、收敛最快（settling=0）、超调最低（0.13）。
- 默认参数不变（k_coupling=5000, max_torque=50, torque_smoothing=0.3），已由实验验证为最优。
- `example_couple.py` auto test 验证通过。

## 真实性能测试 + 文档精简

- 新增 `tests/test_perf_couple.py`：基于 `example_couple.py` 的真实耦合场景（Taichi PBD 肌肉 + Newton 骨骼）进行性能测试。
- 测试设计：排除初始化和可视化，仅计时核心 `solver.step()` 循环。200 步 × 3 次取均值，20 步 warmup。
- 解决 Taichi 单进程初始化限制：场景只构建一次，不同 bone solver 通过替换 `coupled.bone_solver` 切换。
- **真实性能数据（CPU）**：Featherstone 最快（30.5 ms/step, 32.7 fps）> semi_implicit（31.3 ms）> XPBD（35.0 ms）> MuJoCo（82.6 ms）。
- VBD 不支持 revolute joint，无法用于此场景。
- 精简 `plan-couple.md`：去除轻量级仿真的 FPS 列（纯 Python 算术无法反映真实性能），性能数据统一使用真实引擎测试结果。
- 基准推荐从 XPBD 更新为 **Featherstone**（真实性能最优）。

## 基准切换 XPBD + 统一参数重跑全部实验

- **基准 solver 从 mujoco 切换为 xpbd**：XPBD 精度最高且 fps 高于 mujoco（482k vs 432k）。
- **统一基准参数**：所有实验共用 `mass=1.2, inertia=0.32, stiffness=7.2, damping=6.8, dt=1/180`，消除了此前横向对比与单参数扫描间的参数不一致。
- 以 XPBD 基准重跑全部实验（横向对比、activation 验证、耦合模式、质量/刚度/dt×damping/stiffness×damping），数据全面更新。
- 评价体系升级为三维：**可控性**（peak_theta + tail_mae）、**稳定性**（mae）、**效率**（fps）。
- 10 个回归测试在新基准下全部通过。
- 完整重写 `docs/plans/plan-couple.md`，所有表格、结论、推荐方案均基于 XPBD 统一基准。

## 扩展对比实验 + FPS 性能指标

- 在 `plan_couple_sim.py` 中增加 `time.perf_counter()` 计时，`run_sim()` 返回 `elapsed_ms` 和 `fps`。
- 新增 4 组实验（C/D/E/F）：
  - **实验 C**：刚度敏感性（9 值 × 3 activation = 27 配置）。发现 tail_mae 非单调，最优 stiffness 依赖 activation。
  - **实验 D**：dt × 阻尼交互（4 dt × 5 damping = 20 配置，固定仿真时间）。修正了早期"dt 异常"——实为仿真时间混淆。修正后敏感度排序：inertia > damping > stiffness >> dt。
  - **实验 E**：FPS 性能对比（5 solver × 5 runs）。XPBD 精度/性能比最优。
  - **实验 F**：刚度 × 阻尼 Pareto（4×4 网格）。发现 (k=7.2, d=14.0) 优于默认 (k=7.2, d=6.8)。
- 回归测试从 4 个扩展到 10 个，全部通过。
