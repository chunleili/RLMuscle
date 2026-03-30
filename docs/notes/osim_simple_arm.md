# OpenSim Simple Arm 示例解析

文件：`OpenSimExample/vbd_muscle/osim_simple_arm.py`

基于 OpenSim Python API 的简单手臂前向动力学仿真示例。

## 模型结构

- **刚体**：humerus（肱骨）+ radius（桡骨），各 1kg、1m 长
- **关节**：shoulder（肩，PinJoint，锁定）+ elbow（肘，PinJoint，初始 90°）
- **肌肉**：biceps（肱二头肌），Millard2012EquilibriumMuscle，最大等长力 200N，最优纤维长 0.6m，肌腱松弛长 0.55m
- **控制器**：PrescribedController + StepFunction，0.5s~3.0s 激励从 0.3 升到 1.0

## 仿真流程

1. 锁肩、屈肘 90°、肌肉平衡初始化
2. 前向动力学积分 10s
3. 肱二头肌收缩驱动前臂围绕肘关节摆动

## 输出

| 文件 | 内容 |
|------|------|
| `output/SimpleArm.osim` | 模型 XML |
| `output/arm_states.sto` | 状态轨迹（关节角/速度、肌肉激活/纤维长度） |
| 控制台 | 每 1s 打印 fiber_force 和 elbow_angle |

## 关键 API 用法

```python
# 写 STO 文件（最简方式）
states_table = manager.getStatesTable()
osim.STOFileAdapter().write(states_table, "output/arm_states.sto")

# Vec3 数据输出 STO（需 flatten）
table_reporter = osim.TableReporterVec3()
pos_table_flat = table_reporter.getTable().flatten()
osim.STOFileAdapter().write(pos_table_flat, "output/positions.sto")
```
