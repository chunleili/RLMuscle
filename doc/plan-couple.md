# 骨骼-肌肉耦合方案迭代（面向 Newton 五类 Solver）

## 目标
目标：在 **activation 可控驱动关节转动** 的前提下，实现耦合稳定可控，并对 Newton 的 solver 方案做可量化比较。

本轮严格对比对象改为 Newton 文档中的五类：
- `SolverFeatherStone`
- `SolverMujoco`
- `SolverSemiImplicit`
- `SolverVBD`
- `SolverXPBD`

> 说明：当前测试脚本是“快速参数研究夹具”，用于先筛参数与耦合策略，再迁移到真实 `example_couple.py` 场景。脚本中的 solver 名称与上面五类一一对应。 

---

## 方法
- 脚本：`tests/plan_couple_sim.py`
- 耦合模式：`torque / distance / weak / hybrid`
- activation 控制：`step / pulse / ramp`
- 关键物理参数：`mass, inertia, damping, dt, substeps(iterations)`
- 评价指标：
  - `peak_theta`：峰值角度（过冲风险）
  - `mae`：全程平均误差
  - `tail_mae`：后 20% 时段误差（稳定收敛质量）

---

## 六轮以上迭代（实际 8 轮）
以下 8 轮均在 `pulse` activation 下做连续改参（每轮基于前一轮）：

| iteration | solver | dt | inertia | damping | peak_theta | mae | tail_mae | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 baseline | mujoco | 0.00417 | 0.24 | 5.8 | 2.6866 | 0.9013 | 0.3456 | 基线可运行，但过冲偏大 |
| 2 inertia_up | mujoco | 0.00417 | 0.32 | 5.8 | 2.1326 | 0.6252 | 0.1891 | 增大惯量明显抑制过冲 |
| 3 damping_up | mujoco | 0.00417 | 0.32 | 6.8 | 2.0495 | 0.6075 | 0.2419 | 阻尼提升继续降峰值，但尾段误差略上升 |
| 4 dt_up | mujoco | 0.00556 | 0.32 | 6.8 | 2.0498 | 0.4903 | 0.1579 | 较大 dt 在该配置下反而获得更低误差 |
| 5 dt_down | mujoco | 0.00333 | 0.32 | 6.8 | 2.0493 | 0.6986 | 1.1961 | 更小 dt 出现尾段误差恶化，需重配阻尼/刚度 |
| 6 xpbd_swap | xpbd | 0.00333 | 0.32 | 6.8 | 2.0436 | 0.6959 | 1.1927 | 在当前参数下比 mujoco 略好 |
| 7 feather_swap | featherstone | 0.00333 | 0.32 | 6.8 | 2.0622 | 0.7045 | 1.2038 | 稳定性接近，但弱于 xpbd |
| 8 semi_swap | semi_implicit | 0.00333 | 0.32 | 6.8 | 2.0777 | 0.7116 | 1.2131 | 该参数下最弱，需要更保守耦合 |

阶段结论：
1. 参数敏感度排序（本夹具中）：**inertia > dt > damping > solver 切换**。
2. `dt` 不是越小越好；与阻尼、刚度、子步有耦合，需要联调。
3. solver 切换有效，但在“参数未对齐”时提升有限。

---

## Newton 五类 Solver 横向对比（统一参数）
统一参数：`mode=hybrid, mass=1.2, inertia=0.24, stiffness=7.2, damping=5.8, dt=1/240`。

| solver | activation | peak_theta | mae | tail_mae |
|---|---|---:|---:|---:|
| semi_implicit | step | 4.2884 | 1.8331 | 0.7094 |
| featherstone | step | 4.2792 | 1.8294 | 0.7333 |
| vbd | step | 4.2780 | 1.8289 | 0.7360 |
| xpbd | step | 4.2673 | 1.8247 | 0.7622 |
| mujoco | step | 4.2711 | 1.8262 | 0.7533 |
| semi_implicit | pulse | 2.7200 | 0.9067 | 0.3157 |
| featherstone | pulse | 2.7018 | 0.9038 | 0.3321 |
| vbd | pulse | 2.6997 | 0.9033 | 0.3338 |
| xpbd | pulse | 2.6798 | 0.9001 | 0.3516 |
| mujoco | pulse | 2.6866 | 0.9013 | 0.3456 |
| semi_implicit | ramp | 4.0947 | 1.5312 | 2.9229 |
| featherstone | ramp | 4.0799 | 1.5221 | 2.9176 |
| vbd | ramp | 4.0782 | 1.5210 | 2.9157 |
| xpbd | ramp | 4.0616 | 1.5110 | 2.9106 |
| mujoco | ramp | 4.0673 | 1.5144 | 2.9129 |

结论（当前参数下）：
- `XPBD` 与 `Mujoco` 在 `peak_theta/mae` 略占优。
- `step/ramp` 比 `pulse` 更容易触发较大误差，说明激励形状对调参影响很大。

---

## 多种 activation 控制方式验证（固定推荐参数）
参数：`solver=mujoco, mode=hybrid, inertia=0.32, damping=6.8, dt=1/180`。

| activation | peak_theta | final_theta | mae | tail_mae | 观察 |
|---|---:|---:|---:|---:|---|
| step | 3.3443 | 0.7073 | 1.0633 | 0.1896 | 有明显上冲，但尾段可收敛 |
| pulse | 2.0498 | 0.3600 | 0.4903 | 0.1579 | 三者最稳，收敛最好 |
| ramp | 3.1571 | 0.6182 | 0.9480 | 0.3035 | 平滑激励仍需抑制累积偏差 |

结论：在多种 activation 下都可实现可控转动，但 `step/ramp` 需要更强的尾段收敛控制。

---

## 耦合模式对比（同一组推荐参数）
参数：`solver=mujoco, activation=pulse, inertia=0.32, damping=6.8, dt=1/180`。

| mode | peak_theta | final_theta | mae | tail_mae | 结论 |
|---|---:|---:|---:|---:|---|
| torque | 30.0997 | 30.0997 | 18.3488 | 28.4041 | 发散，不可用 |
| distance | 3.1275 | 3.1275 | 2.1118 | 2.5638 | 稳定但偏硬，误差大 |
| weak | 0.4866 | 0.0791 | 0.3378 | 0.4669 | 峰值最小，但响应偏弱 |
| hybrid | 2.0498 | 0.3600 | 0.4903 | 0.1579 | 兼顾响应与尾段稳定，当前最佳折中 |

---

## 当前推荐落地方案（精简）
1. 默认先走 **`SolverMujoco + hybrid`**，并以 `inertia=0.32, damping=6.8, dt=1/180` 作为首个候选起点。
2. 若要进一步压峰值，优先尝试 `SolverXPBD`，再细调阻尼和 stiffness。
3. 严禁 `torque-only` 直接上线；至少叠加 weak/hybrid 约束。
4. 每次调整必须同时在 `step/pulse/ramp` 三类 activation 下回归。

---

## 可运行测试命令
- 全矩阵（五 solver × 三 activation）：
  - `python tests/plan_couple_sim.py --matrix`
- 单次场景：
  - `python tests/plan_couple_sim.py --mode hybrid --solver mujoco --activation-profile pulse --inertia 0.32 --damping 6.8 --dt 0.0055555556`

---

## 下一轮迭代
- 把当前推荐参数迁移到 `examples/example_couple.py`（真实几何+关节）并复现三类 activation。
- 在 `src/VMuscle/solver_muscle_bone_coupled.py` 增加 tail error/峰值监控日志，形成自动回归门槛。
- 若真实场景出现震荡，优先调整 `dt × damping × substeps` 联合配置，再讨论 solver 替换。


---

## 增补实验 A：质量敏感性（固定其他参数）
参数：`solver=mujoco, mode=hybrid, activation=pulse, inertia=0.32, damping=6.8, dt=1/180`。

| mass | peak_theta | mae | tail_mae | 观察 |
|---:|---:|---:|---:|---|
| 0.8 | 2.1838 | 0.5354 | 0.1329 | 响应更激进，峰值偏高 |
| 1.0 | 2.1141 | 0.5118 | 0.1459 | 稳定性开始改善 |
| 1.2 | 2.0498 | 0.4903 | 0.1579 | 当前默认基线 |
| 1.6 | 1.9349 | 0.4525 | 0.1793 | 峰值继续下降，但尾段误差上升 |
| 2.0 | 1.8354 | 0.4206 | 0.1978 | 控制更“钝”，尾段收敛变慢 |

结论：质量增大可降低峰值与全程误差，但会牺牲尾段响应速度。

## 增补实验 B：子步/solver profile 对比（固定其他参数）
参数：`mode=hybrid, activation=pulse, mass=1.2, inertia=0.32, damping=6.8, dt=1/180`。

| solver | substeps | peak_theta | mae | tail_mae |
|---|---:|---:|---:|---:|
| semi_implicit | 6 | 2.0781 | 0.4977 | 0.1658 |
| featherstone | 8 | 2.0626 | 0.4937 | 0.1616 |
| vbd | 10 | 2.0610 | 0.4931 | 0.1610 |
| xpbd | 12 | 2.0441 | 0.4887 | 0.1562 |
| mujoco | 14 | 2.0498 | 0.4903 | 0.1579 |

结论：在该参数段，`XPBD/Mujoco` 仍是较优解，低子步配置（semi_implicit）更易出现误差放大。
