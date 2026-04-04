# RL simpleArm 短参考轨迹细则

日期：2026-04-04
关联总纲：`docs/plans/2026-04-04-rl-onboarding-to-motion-tracking.md`

## 目标

在 `simpleArm pose` 阶段稳定后，将任务升级为时间相关的目标轨迹跟踪。  
这一阶段的目的不是增加物理复杂度，而是把“动作追踪”这一问题抽象清楚，为后续动作数据库与更复杂 backend 做准备。

## 前置条件

开始本阶段前，应满足以下条件：

1. `simpleArm pose` 任务已经稳定训练。
2. 固定目标姿态控制可复现。
3. 基础 reward 行为已验证。
4. 控制链路日志已可稳定导出。

如果这些条件未满足，则不应进入本阶段。

## 任务边界

### 本阶段包含

- 时间相关目标
- 短时关节轨迹跟踪
- 程序生成 reference trajectory
- tracking 指标定义

### 本阶段不包含

- 动作数据库接入
- 视频骨架接入
- `couple3` 训练
- 多关节轨迹
- imitation policy 先验

## 目标轨迹来源

第二阶段的 reference target 仅来自程序生成轨迹，包含以下四类：

1. 常值目标
2. 阶跃目标
3. 正弦轨迹
4. 2 到 4 段 piecewise cubic 轨迹

这样做的目的，是先把 tracking 任务本身定义清楚，而不是把数据源问题引入进来。

## 环境变化

相比 `simpleArm pose`，本阶段环境只做最小增量修改。

### 动作空间

保持不变：

- 1 维连续 excitation
- 范围仍为 `[0, 1]`

### 观测变化

在阶段 1 的基础上，增加用于 tracking 的目标信息。建议候选为两类：

1. 当前阶段变量，例如 `phase` 或归一化时间
2. 短视野未来目标，例如未来 1 到 3 个参考点

目标是让 policy 知道“接下来要往哪里去”，而不是只看到当前误差。

### 环境不变部分

以下内容保持与阶段 1 一致：

- activation dynamics
- excitation 语义
- 基础控制链路
- 主要日志字段

## reward 设计

tracking 任务的主项切换为跟踪误差：

```text
- w_track * (q - q_ref)^2
```

同时保留以下辅助项：

- 速度惩罚
- 激活惩罚
- 平滑惩罚

这里的关键不是设计尽可能复杂的 reward，而是确保：

1. 跟踪误差是主目标
2. 姿态切换不出现明显抖动
3. 激活水平不会无约束飙升

## episode 设计

每个 episode 对应一段短轨迹。  
reset 时应随机轨迹参数或轨迹类型，但第一版不要引入跨度过大的长时轨迹。

建议每次只覆盖短时间窗，优先验证以下能力：

- 跟随慢变化目标
- 响应阶跃变化
- 保持短时间稳定跟踪

## 验收指标

本阶段成功标准主要看 tracking 表现，而不是单点姿态控制：

- tracking RMSE
- final window error
- steady-state error
- action smoothness
- mean activation
- success rate

其中 RMSE 是最核心指标。

## 与后续阶段的关系

本阶段的价值在于先把以下问题确定下来：

1. target 如何表达
2. observation 中应包含哪些目标信息
3. tracking reward 的主结构如何定义
4. 评估 tracking 应使用哪些指标

这些问题一旦在 `simpleArm` 上稳定，就可以更安全地迁移到 XPBD simpleArm 与 `couple3`。

## 输出物

本阶段应产出：

- `simpleArm track` 任务定义
- 标准 reference trajectory 集合
- tracking 评估指标定义
- 与阶段 1 对齐的训练/评估流程

## 风险点

### 1. 目标提示不足

如果 observation 中没有足够的目标未来信息，policy 可能只能被动纠偏，而难以形成稳定跟踪。

### 2. 轨迹过难

如果一开始就采用频率过高、变化过快或过长的轨迹，会掩盖 tracking 定义本身是否合理。

### 3. 激活正则与 tracking 目标冲突

如果激活惩罚过强，可能牺牲跟踪精度；如果过弱，则可能学出高能耗策略。  
因此本阶段仍应保留激活正则对比实验。
