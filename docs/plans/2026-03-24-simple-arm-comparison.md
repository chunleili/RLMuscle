# SimpleArm 配准计划

## Context

项目已完成 sliding ball 的 VBD vs OpenSim 对比验证（3.3% 误差）。下一步将 SimpleArm（2-body 肘关节+二头肌）进行配准。

核心区别：sliding ball 是 1-DOF 平移，SimpleArm 是 1-DOF 旋转（肘关节），肌肉需要产生关节力矩驱动刚体运动。

**参数决策**：统一使用刚性肌腱 + 有激活动力学。

每完成一个阶段即进行测试并记录关键发现到progress, 并由此更新本plan.

## 架构决策

**确定路线：MuJoCo 骨骼/Tendon + VBD 肌肉**（不采用 VBD 统一架构）

- **阶段 0**：调试 OpenSim 参数（刚性肌腱、激活动力学），DGF vs Millard 对比，确立基线
- **阶段 1**：MuJoCo spatial tendon + 1D DGF 力注入 vs OpenSim。验证 spatial tendon 路径几何 + 配准 MuJoCo 到 OpenSim
- **阶段 2**：VBD 体积肌肉（DGF 本构）+ MuJoCo 骨骼/Tendon 耦合。VBD 肌肉输出力应与阶段 1 的 1D DGF 力一致

阶段 1→2 的关键：阶段 1 确认 DGF 力和力臂正确后，阶段 2 任何额外误差可归因于体积化效应。

---

## 技术调研结论

### 刚体求解器
- **MuJoCo**：最稳定（settling=0, overshoot=0.13）→ 阶段 1、2 均采用

### MuJoCo Spatial Tendon（阶段 1 重点）

Tendon 是核心组件，要求：**刚性** + **支持 waypoint**。

- **Waypoint**：完全支持。`<spatial>` 内可串联多个 `<site>` + `<geom>` wrapping + `<pulley>`
- **刚性策略**：stiffness=0（纯几何路径），外部计算 DGF 力通过 motor actuator 注入。"刚性"由肌肉模型 `ignore_tendon_compliance` 保证。Newton 未暴露 `equality/tendon`，无法做到真正零拉伸
- **运行时 API**：`solver.mj_data.ten_length[idx]` 读长度，`control.mujoco.ctrl[idx]` 注入力
- **参考**：`external/newton/newton/tests/test_spatial_tendon.py`

### VBD-MuJoCo 耦合（阶段 2 前置）

VBD 肌肉通过附着点连接到 MuJoCo 骨骼。力传递方向：
- VBD 肌肉收缩 → 计算附着点力 → 通过 MuJoCo motor actuator 注入骨骼
- MuJoCo 骨骼运动 → 更新附着点位置 → 驱动 VBD 肌肉边界条件

参考 DexterousManipulation 的交替方向耦合策略。

### vmuscle API
- Activations 通过 `Control.tet_activations` 传入
- 肌肉属性在 `model.vmuscle` 命名空间（fiber_dirs, sigma0, max_contraction_velocity, fiber_damping）

### DexterousManipulation 参考 (SIGGRAPH 2018)
- FEM 体积肌肉 + DART 刚体，肌腱用 `AttachmentCst`（弹簧 k=1E6）+ waypoints
- 交替方向耦合：FEM 隐式求解 → 读弹簧力 → 显式施加到刚体
- 肌肉端的弹簧力通过加入到能量函数隐式施加, 骨骼端的显式施加.  为了保证稳定,骨骼侧dt(1/1000)显著小于肌肉.
- 参考：`D:/dev/DexterousManipulation/notes_architecture.md`

---

## 阶段 0: OpenSim 基线

修改 `scripts/osim_simple_arm_dgf.py`：
- `tendon_slack_length`: 保持 0.55（肌腱几何长度不变，刚性通过 `ignore_tendon_compliance=True` 实现）
- `ignore_activation_dynamics`: True → False
- `fiber_damping`: 0.1 → 0.5（测试 0.1/0.5/1.0，0.5 振荡消除且稳态不变）

简化两版脚本，对比 DGF vs Millard → `output/simple_arm_osim_dgf_vs_millard.png`

增加函数接口 `osim_simple_arm_dgf(cfg) -> dict` / `osim_simple_arm_millard(cfg) -> dict`，支持 config 驱动。

**已完成** ✓ — DGF 82.2° vs Millard 88.2°（稳态差 6°，曲线形状固有差异），RMSE=7.15°。 见`docs\progress\2026-03-27-simple-arm-stage0.md`

---

## 阶段 1: MuJoCo Spatial Tendon + DGF vs OpenSim

**新建** `examples/example_mujoco_simple_arm.py`

使用**纯 MuJoCo**（非 Newton 包装，因 mujoco_warp/mujoco 版本不兼容）。

MuJoCo 模型（MJCF）：
```
humerus (fixed, no joint) → elbow (hinge Z-axis, range [0,π]) → radius
site "muscle_origin" (humerus) → spatial tendon (stiffness=0) → site "muscle_insertion" (radius)
motor actuator (gear=-1) on tendon → 注入 DGF 力
```

仿真循环（每个内部子步 dt=0.002s 执行）：
1. Excitation schedule（匹配 OpenSim StepFunction：t<0.5→0.3, t>3.0→1.0, 中间 Hermite 插值）
2. 激活动力学 step
3. `tendon_length = mj_data.ten_length[0]` → fiber_length → norm_fiber_length
4. DGF 力：`(a * f_L(l̃) * f_V(ṽ) + f_PE(l̃) + d * ṽ) * F_max`
5. `mj_data.ctrl[0] = muscle_force`（gear=-1 确保正 ctrl = flexion）
6. `mujoco.mj_step()`

**已完成** ✓ — RMSE=1.80°（2.2%），稳态 82.2° vs OpenSim 82.2°，Max error=8.28°（瞬态）. 见`docs\progress\2026-03-27-simple-arm-stage1.md`

---

## 阶段 2: 体积化肌肉 Mesh + MuJoCo 骨骼/Tendon 耦合

**新建** `examples/example_vbd_mujoco_simple_arm.py`

实际架构（VBD sigma0>0 + MuJoCo 刚体，两级耦合）：

VBD 层（慢，每 outer step = 0.016s）：
1. MuJoCo tendon path length → fiber_length（刚性肌腱）
2. 重置 mesh 顶点到对应位移（origin + insertion kinematic）
3. 设置 tet_activations，VBD step（sigma0>0, iterations=50）
4. VBD vmuscle kernel 内部计算 DGF 力（force-length, force-velocity, passive），
   与 Neo-Hookean 弹性力一起求解 3D 平衡
5. 从 deformation gradient 提取 per-tet fiber stretch → volumetric delta

MuJoCo 层（快，每 substep = 0.002s）：
1. 实时激活动力学（tau_act=15ms 需要小步长）
2. l_tilde = 1D_l_tilde + VBD delta → DGF 力 → motor actuator
3. MuJoCo step

参数：sigma0 = F_max / (πr²) ≈ 159155 Pa, k_mu=1000, k_lambda=10000,
k_damp=1.0, density=1060, iterations=20, vbd_substeps=20。
直圆柱 delta≈0（物理正确：3D axial stretch = 1D stretch）。
sigma0 驱动的收缩力与 kinematic 边界可产生瞬态径向不稳定，通过 stretch sanity check 过滤。

**已完成** ✓ — RMSE=1.75°，稳态 82.16° vs OpenSim 82.2°。见`docs\progress\2026-03-28-simple-arm-stage2.md`

---

## 阶段 3: 完全物理的 VBD 骨骼肌肉耦合

**新建** `examples/example_vbd_coupled_simple_arm.py`

去除阶段 2 的运动学缩放，改用完全物理的耦合：
- VBD 肌肉 mesh（sigma0>0）自主演化，不再每步重置顶点
- Attachment 顶点通过弹簧/ATTACH 约束跟随 MuJoCo 骨骼位置（参考 SolverMuscleBoneCoupled）
- 肌肉力从 ATTACH 约束的反力提取（reaction_accum），或从 deformation gradient 提取
- MuJoCo 骨骼接收 VBD 计算的力，驱动关节运动

阶段 2 vs 3 的关键区别：
| | 阶段 2 | 阶段 3 |
|---|---|---|
| VBD 状态 | 每步从 rest 重置+缩放 | 自主演化，有状态延续 |
| 边界条件 | 全部 kinematic（位移控制） | origin kinematic + insertion ATTACH |
| 适用几何 | 仅直圆柱 | 任意肌肉几何 |
| scale 限制 | 0.5-1.2 外跳过 VBD | 无（弹簧耦合自适应） |

前置依赖：
- 探索如何提高VBD稳定性
- 理解 DexterousManipulation 中 Rigid-soft 耦合实现细节（ `D:\Dev\DexterousManipulation\notes_architecture.md`）
- 理解 MuscleSim ATTACH 约束如何在 PBD 层面传力
- 或在 SolverVBD 中实现类似的弹簧耦合边界
- 参考 `src/VMuscle/solver_muscle_bone_coupled_warp.py`、`examples/example_couple2.py`

---

## 配置文件 `data/simpleArm/config.json`

```json
{
  "geometry": {
    "humerus_length": 1.0, "radius_length": 1.0,
    "muscle_origin_on_humerus": [0, 0.8, 0],
    "muscle_insertion_on_radius": [0, 0.7, 0],
    "muscle_radius": 0.02, "n_circumferential": 8, "n_axial": 10
  },
  "muscle": {
    "max_isometric_force": 200.0, "optimal_fiber_length": 0.6,
    "tendon_slack_length": 0.55, "max_contraction_velocity": 10.0,
    "fiber_damping": 0.5
  },
  "activation": {
    "excitation_start_time": 0.5, "excitation_end_time": 3.0,
    "excitation_off": 0.3, "excitation_on": 1.0,
    "tau_act": 0.015, "tau_deact": 0.060
  },
  "solver": { "dt": 0.0167, "n_steps": 600 },
  "initial_conditions": { "elbow_angle_deg": 90.0 }
}
```

## 关键文件

| 操作 | 文件 |
|------|------|
| 修改 | `scripts/osim_simple_arm_dgf.py`, `scripts/osim_simple_arm_millard.py` |
| 新建 | `data/simpleArm/config.json` |
| 新建 | `examples/example_mujoco_simple_arm.py` — 阶段 1 |
| 新建 | `examples/example_vbd_mujoco_simple_arm.py` — 阶段 2 |
| 新建 | `scripts/run_simple_arm_comparison.py` |
| 复用 | `src/VMuscle/dgf_curves.py`, `src/VMuscle/activation.py`, `src/VMuscle/mesh_utils.py` |
| 参考 | `external/newton/newton/tests/test_spatial_tendon.py`, `examples/example_couple2.py`, `scripts/run_sliding_ball_comparison.py` |

## 验证

```bash
# 阶段 0: DGF vs Millard 对比（一键运行）
uv run python scripts/run_simple_arm_comparison.py
# 输出: output/simple_arm_osim_dgf_vs_millard.png, output/SimpleArm_DGF.osim, output/SimpleArm_Millard.osim

# 阶段 0: 单独运行
uv run python scripts/osim_simple_arm_dgf.py
uv run python scripts/osim_simple_arm_millard.py
# 均支持 --config data/simpleArm/config.json 指定配置

# 阶段 1: MuJoCo DGF vs OpenSim DGF（已完成 ✓ RMSE=1.80°）
uv run python scripts/run_simple_arm_comparison.py --mode mujoco
# 或单独运行 MuJoCo:
uv run python examples/example_mujoco_simple_arm.py
# 输出: output/simple_arm_mujoco_vs_osim.png, output/SimpleArm_MuJoCo.xml

# 阶段 2: VBD+MuJoCo vs OpenSim DGF（已完成 ✓ RMSE=1.86°）
uv run python scripts/run_simple_arm_comparison.py --mode vbd
# 或单独运行:
uv run python examples/example_vbd_mujoco_simple_arm.py
# 输出: output/simple_arm_vbd_vs_osim.png, output/SimpleArm_VBD_MuJoCo_states.sto

# 全部阶段
uv run python scripts/run_simple_arm_comparison.py --mode all

# 目标：阶段 1 误差 <5% ✓（RMSE=1.80°），阶段 2 误差 <10% ✓（RMSE=1.75°）
```
