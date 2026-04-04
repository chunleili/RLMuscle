# RL simpleArm 固定目标姿态细则

日期：2026-04-04
关联总纲：`docs/plans/2026-04-04-rl-onboarding-to-motion-tracking.md`

## 目标

定义第一阶段 RL 任务：在 `simpleArm` 后端上，agent 输出 excitation，经由现有 activation dynamics 和肌肉力学链路，控制肘关节到达目标角度并稳定保持。

这份细则只覆盖第一阶段最小任务，不涉及参考轨迹、不涉及 `couple3`、不涉及动作数据库。

## 任务边界

### 本阶段包含

- 单关节控制
- 单目标角度
- 连续动作空间
- dense reward
- 基础训练与基础评估
- 控制链路日志验证

### 本阶段不包含

- 轨迹跟踪
- 多关节控制
- 多肌冗余分配
- 视频或数据库驱动目标
- imitation learning
- 复杂 curriculum

## backend 选择

第一版使用 MuJoCo `simpleArm` 作为 RL backend。

原因：

- 状态维度低
- 训练成本低
- 当前控制链路最清晰
- 更适合先验证 RL 框架设计

XPBD simpleArm 和 `couple3` 不属于本阶段实现范围，只在后续迁移阶段接入。

## 控制链路

本阶段必须显式保留并观测如下链路：

`excitation -> activation -> muscle force/torque -> elbow angle`

RL policy 只输出 excitation，不允许直接输出 torque，也不绕过 activation dynamics。

## 环境定义

### 动作空间

- 维度：1
- 类型：连续
- 范围：`[0, 1]`
- 物理含义：肌肉 excitation

### 观测空间

第一版观测向量固定包含以下字段：

1. `target_angle`
2. `current_angle`
3. `angle_error`
4. `joint_velocity`
5. `activation`
6. `previous_action`

第一版先不加入更复杂的内部物理量，以保持任务最小可训练。  
如果后续发现训练明显不稳定，再讨论是否扩展观测。

### reset 随机化

每个 episode reset 时至少随机以下量：

- 目标角度
- 初始关节角度
- 小范围初始 activation

随机化目标是防止 policy 仅记忆单一初值，而不是制造困难 domain randomization。

## reward 设计

第一版 reward 使用 dense reward，定义为四项组合：

```text
reward =
  - w_pose   * angle_error^2
  - w_vel    * joint_velocity^2
  - w_act    * activation^2
  - w_smooth * (action_t - action_{t-1})^2
```

各项职责：

- `pose`：主优化目标，决定是否达到目标角度
- `vel`：抑制姿态附近的持续振荡
- `act`：轻度约束用力水平，为后续最小激活正则预留位置
- `smooth`：抑制高频 excitation 抖动

## 激活正则策略

本阶段不把“最小激活”设为强默认约束，但必须在设计上预留。

原因：

- 姿态追踪本身可能不足以约束“怎么发力”
- 如果策略学出长期过度发力，则后续 tracking 任务会更不稳定

因此本阶段要求：

- 默认使用弱激活惩罚
- 训练完成后补一个小规模对比实验

对比项：

1. 无激活正则
2. 弱激活正则
3. 更强激活正则

实验目的是判断该项是否需要成为后续阶段的默认配置。

## episode 规则

### 开始条件

- 环境 reset 后初始化目标角、状态和内部激活状态

### 终止条件

若达到目标角误差阈值，并且连续若干步保持稳定，则可记为成功终止。

### 截断条件

达到固定最大步数时截断。

### 异常终止

出现数值异常、状态越界或环境失效时，允许提前结束 episode，并在日志中显式标记。

## 成功判据

第一阶段的成功不以“奖励看起来上升”为标准，而以以下结果为准：

1. 随机目标角任务可收敛到误差阈值内。
2. 达到目标后不会持续大幅振荡。
3. episode return 明显优于随机策略。
4. `excitation -> activation -> force/torque -> joint angle` 链路在日志中可直接观测。
5. 评估结果在固定 seed 下可重复。

## 日志与可视化要求

第一阶段必须记录以下时间序列：

- `target_angle`
- `current_angle`
- `angle_error`
- `action / excitation`
- `activation`
- `muscle_force` 或等价 torque 指标
- `reward`

其中控制链路相关量是必需的，不能只保留最终姿态误差。

## 训练建议

算法优先级如下：

1. `PPO`
2. `SAC`

采用 `PPO` 的前提是环境 rollout 成本可接受；如果发现并行收益不明显或样本效率不足，再切换到 `SAC` 作为第二方案。

## 评估要求

第一阶段评估必须与训练分开，至少包含：

- 固定 seed rollout
- 多随机目标角测试
- 与随机策略的基线对比
- 不同激活正则配置的对比

建议核心指标：

- success rate
- final angle error
- episode return
- action smoothness
- mean activation

## 输出物

本阶段最终应产出：

- 一个标准 `simpleArm pose` RL 环境设计
- 一套可复现的训练配置
- 一套独立评估流程
- 一份阶段性 progress 文档
- 一组能说明控制链路成立的曲线与数值

## 不应提前做的事

以下内容不应在本细则范围内提前实现：

- 直接开始 `couple3` RL 训练
- 把动作数据库接进第一版训练
- 把视频骨架作为目标输入
- 设计多关节统一控制器
- 在 reward 中堆叠过多先验项
