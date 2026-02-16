## Bicep USD TetMesh + Attachment 力矩耦合改造（`example_couple`）

### 摘要
本轮按你确认的方向落地：
1. `example_couple` 切换到 `bicep.usd` 骨骼场景（不再用 pendulum 场景）。
2. `muscle_core` 移除 `.geo` 默认流程，改为从 USD `TetMesh` 读取体网格与 primvar。
3. 激活由用户输入（GUI 滑条 + CLI 常量），不做骨骼状态到激活反馈。
4. 耦合力矩只来自 attachment 反力矩，不再叠加独立 motor。
5. attachment 使用“初始帧近邻映射 + 强/弱约束两区域”，优先 `muscleendmask/muscletobonemask`，缺失时回退 `tendonmask`。

### 具体实施（决策完成）
1. 在 `/Users/cl/Dev/RLMuscle/src/RLVometricMuscle/muscle_core.py` 增加 USD TetMesh 读取路径（默认主路径），并保留最小 fallback。
涉及改动：
- 扩展 `SimConfig`：新增 `usd_path`、`tet_prim_path`、`strong_mask_name`、`weak_mask_name`、`fallback_mask_name`、`strong_threshold`、`weak_threshold`、`attach_stiffness_strong`、`attach_stiffness_weak`、`attach_damping`。
- 新增 `load_mesh_usd_tet(...)`：读取 `points`、`tetVertexIndices`、`primvars:materialW`、mask primvar。
- 默认加载：`usd_path=data/muscle/model/bicep.usd`，`tet_prim_path=/character/muscle/bicep`。
- 若 USD TetMesh 不可用，才回退到内置最小 tetra（确保示例不崩）。

2. 在 `/Users/cl/Dev/RLMuscle/src/RLVometricMuscle/muscle_core.py` 增加 attachment 约束与反力矩计算。
涉及改动：
- 新增一次性映射接口（初始化时调用）：基于肌肉点与骨点近邻建立映射。
- 强区域点集来源：`muscleendmask`；缺失则 `tendonmask > strong_threshold`。
- 弱区域点集来源：`muscletobonemask`；缺失则 `weak_threshold < tendonmask <= strong_threshold`。
- 强区域：映射到静态骨点+动态骨点的并集最近邻。
- 弱区域：映射到静态骨点最近邻。
- 每子步：计算每个 attachment 弹簧阻尼力，施加到肌肉点，并累计动态骨反作用力矩。
- 输出 `last_attachment_torque`（标量，沿关节轴投影）。

3. 在 `/Users/cl/Dev/RLMuscle/src/RLVometricMuscle/solver_volumetric_muscle.py` 增加“骨骼耦合上下文”。
涉及改动：
- 新增配置接口：传入静态骨点、动态骨点、动态 body index、joint dof index、joint pivot、joint axis。
- 在 `step()` 内每步根据 `state_in.body_q` 更新动态骨目标点世界位姿，再调用 `core.step(dt)`。
- `step()` 返回 attachment torque（float），供耦合 solver 使用。
- 默认仅适配 `newton==0.2.0`，不做 0.1.x 分支。

4. 在 `/Users/cl/Dev/RLMuscle/src/RLVometricMuscle/solver_muscle_bone_coupled.py` 实现“先肌肉后骨骼”的力矩注入链路。
涉及改动：
- 调用肌肉步进得到 `tau_attach`。
- 写入 `control.joint_f[joint_dof_index] = tau_attach`。
- 仅使用 attachment torque，不叠加独立 motor。
- 再调用 `SolverFeatherstone.step(...)`。
- 保持该类仍作为 `example_couple` 的主 solver 入口。

5. 在 `/Users/cl/Dev/RLMuscle/examples/example_couple.py` 切换到 bicep 场景并接入新耦合。
涉及改动：
- CLI 参数：新增/保留 `--activation`（headless 默认常量激活），移除正弦驱动参数依赖。
- GUI：增加 `activation` 滑条（0~1）。
- 骨骼场景：`humerus/scapula` 静态，`radius` 单动态 body + 单 revolute（符合“单动态前臂”）。
- 轴位置信息来源策略：使用 USD skeleton kinematic tree 选定手臂关节链语义；关节枢轴位置用 humerus/radius 近邻接触中点；轴方向从该关节在 skeleton rest 变换中的轴向提取，若异常回退到固定轴。
- 初始化时建立 attachment 近邻映射并注入 solver。
- `newton.examples.run(example, args)` 以兼容当前 0.2.0 API。

6. 在 `/Users/cl/Dev/RLMuscle/examples/example_couple.py` 的 layered USD 输出中更新信号。
涉及改动：
- 输出 runtime 字段：`joint_angle`、`muscle_activation`、`attachment_torque`、`muscle_centroid_y`。
- 输出 prim custom：对 `/character/muscle/bicep` 写 `activation`（便于 DCC/分析）。
- 不再写 pendulum 专用路径 `/pendulum/joint0`。

7. 在 `/Users/cl/Dev/RLMuscle/PROGRESS.md` 记录本轮详细日志（中文）。
涉及改动：
- `example_couple` 新增“已完成/已测试/下一步”。
- 明确标注：已切 USD TetMesh，已移除 `.geo` 主路径依赖，已实现 attachment torque 耦合。

8. 不修改 README；不新增临时文件；全部运行验证使用 `.venv/bin/python`。

### 对外接口/类型变更
- `SimConfig`（`muscle_core.py`）新增 USD 与 attachment 配置字段。
- `SolverVolumetricMuscle.step(...)` 从“仅副作用”变为“副作用 + 返回 attachment torque(float)”。
- `SolverMuscleBoneCoupled` 新增接收/使用 attachment torque 的控制注入逻辑。
- `example_couple.py` CLI 变更：主激活输入改为 `--activation` + GUI slider，去除当前正弦驱动依赖。

### 测试与验收场景
1. 无头主验收：
- 命令：`.venv/bin/python /Users/cl/Dev/RLMuscle/examples/example_couple.py --viewer null --headless --num-frames 5 --use-layered-usd --activation 0.35`
- 期望：退出码 0；生成 `output/example_couple.anim.usda`；包含 runtime 字段 `joint_angle/muscle_activation/attachment_torque/muscle_centroid_y`。

2. 激活边界行为：
- 命令A：`--activation 0.0`，命令B：`--activation 1.0`
- 期望：`attachment_torque` 时序存在可观差异，关节角变化趋势不同。

3. 兼容回归（不破坏已完成示例）：
- 命令：`.venv/bin/python /Users/cl/Dev/RLMuscle/examples/example_dynamics.py --viewer null --headless --num-frames 2 --use-layered-usd`
- 期望：退出码 0。

### 明确假设与默认值
- 仅支持当前环境 `newton==0.2.0`。
- `example_couple` 默认使用 `data/muscle/model/bicep.usd` 的 TetMesh `/character/muscle/bicep`。
- `muscleendmask/muscletobonemask` 缺失时自动回退到 `tendonmask` 分层。
- 骨骼自由度固定为“单动态前臂（radius）”。
- 总关节力矩仅由 attachment 反力矩提供（不叠加独立 motor）。
- 这是快速原型实现，后续再做更高保真几何力臂/多关节扩展。
