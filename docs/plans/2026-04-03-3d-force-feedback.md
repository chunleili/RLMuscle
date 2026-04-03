# Plan: 闭合 3D 力反馈环路（XPBD → MuJoCo）

## 问题

`example_xpbd_coupled_simple_arm_millard.py` 中，XPBD 3D mesh 的 active fiber force
虽然在 `accumulate_active_fiber_force()` 中计算并作用于 mesh 顶点，但
**没有反馈到 MuJoCo 关节**。驱动关节的力由独立的 1D Hill 公式计算：

```python
# 当前的 shortcut（line 286-292）
fib_len = mj_data.ten_length[0] - L_slack        # ← 1D MuJoCo 肌腱长度
fl = _mc.fl.eval(fib_len / L_opt)                 # ← 1D Millard 曲线
muscle_force = (a * fl * fv + fpe + damp) * F_max  # ← 纯 1D 公式
mj_data.ctrl[0] = muscle_force                     # ← 直接喂给 MuJoCo
```

导致 "XPBD vs OpenSim" 对比曲线虚假地完美匹配。

## 解决方案概述

从 XPBD ATTACH 约束的 **Lagrange 乘子** 提取 insertion 端的 3D 反力，
投影到肌腱方向得到标量肌力，用于驱动 MuJoCo。

### 力提取原理

XPBD 框架中，ATTACH 约束求解后，`cons[cidx].L[0]` 存储累积 Lagrange 乘子 λ。
根据 XPBD 理论（Macklin et al. 2016），约束力为：

```
f_constraint = ∇C · λ / dts²
```

对于 ATTACH bilateral 距离约束：
- ∇C = n（从 bone target 指向 muscle vertex 的单位方向）
- f_muscle_vertex = n · λ / dts²

Insertion 端对 bone 的反力（肌力）：
```
F_bone = -Σᵢ nᵢ · λᵢ / dts²     (i ∈ insertion vertices)
```

投影到肌腱方向得标量力：
```
F_scalar = -dot(F_bone, tendon_unit)
```

其中 `tendon_unit` = (insertion - origin) / |insertion - origin|。
当肌肉收缩时 F_scalar > 0，对应 `mj_data.ctrl[0]` 的正值（gear=-1 使正 ctrl 缩短肌腱）。

### 前提条件

- **GS 模式**：已确认 `use_jacobi = False`（`muscle_warp.py:1876`），
  GS 模式下 `cons.L` 在每个 substep 内被正确累积
- **clear() 行为**：`clear()` 调用 `clear_cons_L_kernel` 将所有 L 置零，
  在 `solve_constraints()` 前调用，确保 L 只包含当前 substep 的值
- **Constraint struct**：`L: wp.vec3`，ATTACH 使用 `L[0]`（标量）

## 修改范围

| 文件 | 修改内容 |
|------|---------|
| `src/VMuscle/muscle_warp.py` | 新增 `get_insertion_force()` 方法 |
| `examples/example_xpbd_coupled_simple_arm_millard.py` | 重构主循环：用 3D 力替代 1D Hill |
| `scripts/run_simple_arm_comparison.py` | 微调绘图（可选：增加 3D/1D 力对比子图） |

## 实施阶段

### Stage 1: 力提取基础设施

在 `MuscleSim` 类中添加方法：

```python
def get_insertion_force(self, insertion_cidxs, dts):
    """从 ATTACH 约束 Lagrange 乘子提取 insertion 端的 3D 力。

    Args:
        insertion_cidxs: insertion 侧 ATTACH 约束的 cidx 列表
        dts: XPBD substep 时间步长

    Returns:
        net_force: (3,) ndarray — insertion 端对 bone 的净力（世界坐标）
    """
    cons_np = self.cons.numpy()           # structured array
    pos_np = self.pos.numpy()             # (n_verts, 3)

    force_sum = np.zeros(3, dtype=np.float64)
    dts2 = dts * dts
    for cidx in insertion_cidxs:
        c = cons_np[cidx]
        ia = int(c['pts'][0])             # muscle vertex index
        target = c['restvector'][:3]      # bone target position
        lam = float(c['L'][0])            # accumulated Lagrange multiplier

        dx = pos_np[ia] - target
        d = np.linalg.norm(dx)
        if d > 1e-10:
            n = dx / d
            # 约束力 on muscle = n * λ / dts²
            # 反力 on bone = -n * λ / dts²
            force_sum += -n * lam / dts2

    return force_sum.astype(np.float32)
```

**验证方法**：
- 在激活前（a=0）运行几步，提取力应 ≈ 0（仅重力平衡的残余力）
- 在激活后（a>0），提取力应沿肌腱方向，量级与 F_max·a·f_L 可比
- 与 `reaction_accum` 方法（现有 coupled solver）做交叉验证

### Stage 2: 修改耦合主循环

重构 `example_xpbd_coupled_simple_arm_millard.py` 的 substep 循环：

```python
for _sub in range(num_substeps):
    # a) 更新 attach targets（MuJoCo → XPBD，不变）
    ...

    # b) XPBD step（不变）
    wp.launch(fill_float_kernel, dim=n_tets,
              inputs=[sim.activation, wp.float32(activation)])
    sim.update_attach_targets()
    sim.clear_forces()
    sim.accumulate_active_fiber_force()
    sim.integrate(); sim.clear(); sim.clear_reaction()
    sim.solve_constraints(); sim.update_velocities()

    # b2) ★ 新增：从 XPBD 提取 3D 力
    F_3d = sim.get_insertion_force(insertion_cidxs, dts)
    F_scalar = float(-np.dot(F_3d, cur_tdu))  # 投影到肌腱方向
    F_scalar = max(F_scalar, 0.0)              # 肌肉只能拉不能推

    # c) MuJoCo substeps — 用 3D 力替代 1D 公式
    for _ in range(mj_per_xpbd):
        exc = compute_excitation(t_now, act_cfg)
        activation = float(activation_dynamics_step_np(
            np.array([exc]), np.array([activation]),
            dtmj, tau_act=..., tau_deact=...)[0])

        mj_data.ctrl[0] = F_scalar  # ★ 用 3D 力
        mujoco.mj_step(mj_model, mj_data)
        t_now += dtmj

    physics_time += dts
```

**关键变更**：
1. 删除 1D Hill 力计算（`fl`, `fv`, `fpe` 等）用于 ctrl
2. `mj_data.ctrl[0]` 改用 3D 提取的标量力
3. 激活动力学保留在 MuJoCo 步进中（更新 `activation` 变量）
4. 更新的 `activation` 在下一 substep 通过 `fill_float_kernel` 传入 XPBD

**需要前置准备**：
- 在循环前建立 `insertion_cidxs` 映射（从 ATTACH 约束中筛选 insertion 侧的 cidx）
- cidx 来自 `sim.cons.numpy()['cidx']`，筛选条件：约束类型为 ATTACH 且对应的 vertex id ∈ insertion_ids

### Stage 3: 更新记录

**记录的量**：
| 变量 | 来源 | 用途 |
|------|------|------|
| `forces_out` | 3D 提取的 F_scalar | 主输出，用于对比 |
| `forces_1d` | 1D Hill 公式（仅用于 logging） | 调试参考 |
| `norm_fiber_lengths` | MuJoCo `ten_length` | 与 OpenSim 可比的 1D 量 |
| `l_tilde_xpbd` | mesh stretch（已有） | 3D fiber length 参考 |

对比图保持 2×2 布局，force 子图显示 3D 力（主线）。
可选：添加 1D 力为虚线以突出差异。

### Stage 4: 运行验证

1. 运行 `uv run python scripts/run_simple_arm_comparison.py --mode xpbd-millard`
2. 检查：
   - 仿真稳定（无 tet inversion）
   - 3D 力在合理范围（0 ~ 2·F_max）
   - 关节角度合理变化（应有弯曲运动）
   - 与 OpenSim 存在 **可解释的差异**（不再完美匹配）
3. 记录结果到 `docs/progress/2026-04-03-3d-force-feedback.md`

## 已知局限（本次不修复）

1. **缺少 f_V**：3D kernel `accumulate_active_fiber_force_kernel` 仅使用
   `τ_act = σ₀·a·f_L(r)`，不含 force-velocity 调制。这意味着 3D 力
   在快速收缩/伸展时不会衰减，与 1D Hill 模型有本质差异。
2. **无纤维阻尼**：1D 模型有 `d_damp * v_norm`，3D 模型无此项
   （但 XPBD 的 veldamping 提供了部分等价效果）。
3. **被动力路径不同**：1D 模型用 `fpe(l_tilde)`，3D 模型通过 XPBD volume/ARAP
   约束隐式实现被动弹性，二者不完全等价。
4. **GPU-to-CPU 传输**：`cons.numpy()` 每 substep 一次。对小 mesh 可接受，
   大规模 mesh 需 GPU kernel 来做归约。

## 风险与对策

| 风险 | 对策 |
|------|------|
| 力量级错误（过大/过小） | 先 logging 验证量级，与 1D 力对比 |
| 仿真不稳定 | 检查 F_scalar 范围，必要时 clip |
| 符号错误 | 单元测试验证收缩时 F_scalar > 0 |
| cons.L 未正确累积 | GS 模式已确认；添加 assert 检查 |
