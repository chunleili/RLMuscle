# SimpleArm 阶段 1: MuJoCo Spatial Tendon + DGF vs OpenSim

日期：2026-03-28

## 已完成

- 创建 `examples/example_mujoco_simple_arm.py` — 纯 MuJoCo 仿真（spatial tendon + motor actuator + 外部 DGF 力注入）
- 更新 `scripts/run_simple_arm_comparison.py` — 添加 `--mode mujoco` 支持
- 输出对比图 `output/simple_arm_mujoco_vs_osim.png`
- **RMSE=1.80°（2.2%），Max error=8.28°**，稳态完美匹配（82.2° vs 82.2°）

## 关键发现

### 架构决策：纯 MuJoCo 而非 Newton 包装

原计划通过 Newton `SolverMuJoCo` 使用 spatial tendon。但 Newton 的 mujoco_warp 模块（v3.5.0.2）与 mujoco（v3.6.0）存在版本不兼容（缺少 `ten_J_rownnz` 属性）。

改用**纯 MuJoCo Python API**（`import mujoco`），更简洁且直接控制所有参数。

### MJCF 模型结构

```xml
<mujoco model="simple_arm">
  <worldbody>
    <body name="humerus" pos="0 0 0">   <!-- fixed, no joint -->
      <site name="muscle_origin" pos="0 -0.2 0"/>
      <body name="radius" pos="0 -1 0">
        <joint name="elbow" type="hinge" axis="0 0 1"
               limited="true" range="0 pi"/>
        <inertial pos="0 -1 0" mass="1" diaginertia="0.001 0.001 0.001"/>
        <site name="muscle_insertion" pos="0 -0.3 0"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <spatial name="biceps_tendon" stiffness="0" damping="0">
      <site site="muscle_origin"/><site site="muscle_insertion"/>
    </spatial>
  </tendon>
  <actuator>
    <motor name="biceps_motor" tendon="biceps_tendon" gear="-1"/>
  </actuator>
</mujoco>
```

### 三个关键 Bug 修复

1. **Motor actuator 符号**：MuJoCo `motor` on tendon 的 qfrc = `ten_J × ctrl × gear`。对 flexor muscle（`ten_J < 0`），必须设 `gear=-1` 使正 ctrl 产生 flexion 力矩。
2. **Excitation 时间表**：OpenSim `StepFunction(0.5, 3.0, 0.3, 1.0)` 的语义是 t < 0.5 返回 0.3，t > 3.0 返回 **1.0**（不回落！）。我最初错误地在 t > 3.0 时恢复到 0.3，导致稳态差异 16°。
3. **Force 更新频率**：最初每 0.0167s 计算一次力并持续 8 个子步。改为每个内部子步（0.002s）重新计算 DGF 力，提高精度。

### OpenSim-MuJoCo 坐标映射

| 概念 | OpenSim | MuJoCo |
|------|---------|--------|
| Humerus body origin | elbow 端 (0,0,0) | shoulder 端 (0,0,0) |
| Muscle origin | (0, 0.8, 0) on humerus | (0, -0.2, 0) on humerus |
| Muscle insertion | (0, 0.7, 0) on radius | (0, -0.3, 0) on radius |
| Joint in child | (0, L, 0) | body origin (0, 0, 0) |
| COM | body origin = bottom | explicit (0, -L, 0) |

### 数值验证

| 量 | OpenSim DGF | MuJoCo DGF |
|----|-------------|------------|
| 稳态角度 | 82.24° | 82.16° |
| 稳态力 | ~36.6N | 36.5N |
| 稳态 l̃ | 0.570 | 0.570 |
| RMSE | — | 1.80° |
| Max error | — | 8.28° (瞬态) |

### 技术细节

- MuJoCo 内部 dt=0.002s（MJCF `<option timestep>`），outer dt=0.0167s → 8 substeps
- Integrator: `implicit` (MuJoCo native, NOT Newton)
- 初始 activation=0.5（匹配 OpenSim `equilibrateMuscles`）
- DGF 激活动力学每个内部子步更新一次
- 关节限位 `range="0 π"` 防止过度旋转

## 下一步

- 阶段 2：VBD 体积肌肉 + MuJoCo 骨骼/Tendon 耦合
