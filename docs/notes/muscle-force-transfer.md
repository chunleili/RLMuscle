# 肌肉力传递过程

本文档记录 XPBD mesh 肌肉力如何产生并传递到 MuJoCo 刚体关节。

## 总体架构

```
XPBD Mesh (3D 连续体)          MuJoCo (刚体动力学)
┌─────────────────────┐        ┌────────────────────┐
│  active fiber force  │        │                    │
│  (per-tet kernel)    │        │  motor actuator    │
│         │            │        │    ctrl[0] = F     │
│         ▼            │        │        │           │
│  mesh 变形           │──F──▶  │  spatial tendon    │
│  (pos, vel)          │        │        │           │
│         │            │        │  joint torque      │
│  ATTACH constraint   │◀─x──  │  bone targets      │
│  (锚定到骨骼)        │        │  (site_xpos)       │
└─────────────────────┘        └────────────────────┘
```

两个系统通过两条路径耦合：
- **力路径** (mesh → MuJoCo)：从 mesh 变形状态计算肌肉力 F，写入 `ctrl[0]`
- **位置路径** (MuJoCo → mesh)：从 MuJoCo 骨骼位姿更新 ATTACH 约束的 target

## 力产生：per-tet active fiber kernel

`accumulate_active_fiber_force_kernel` 在每个 tet 上计算应力并散布到顶点。

### 输入

- 当前顶点位置 `pos` 和速度 `vel`
- 静止边矩阵逆 `Dm_inv`、静止体积 `V0`
- 材料空间纤维方向 `d`、activation `a`

### 计算

1. **变形梯度**：`F = Ds @ Dm_inv`，其中 `Ds` 是当前边矩阵
2. **纤维拉伸**：`lm = |F·d|`，拉伸比 `r = lm / lm_opt`
3. **纤维速度**：从速度边矩阵 `Ds_dot` 计算 `dlm/dt = dot(n, Ds_dot @ w)`，
   归一化 `v_norm = dlm/dt / (V_max · lm_opt)`
4. **应力**：`tau = sigma0 * (a * f_L(r) * f_V(v_norm) + d_damp * v_norm)`
5. **节点力**：`f_j = -V0 * tau * w[j] * n`（j=0,1,2），`f_3 = -(f_0+f_1+f_2)`

其中 `w = Dm_inv @ d` 是形函数梯度，`n = F·d / |F·d|` 是当前纤维方向。

### 物理含义

这是一个沿纤维方向的单轴应力场。正 `tau` 产生收缩力（insertion 向 origin 方向），
负 `tau`（快速收缩时 damping 主导）产生制动力。力通过形函数散布到 tet 的 4 个
顶点，FEM 内部节点力自消抵，边界顶点上的合力等于 surface traction。

## 力提取：Hybrid 方案

kernel 产生的是分布式的内力，不能直接用于 MuJoCo。需要提取宏观肌肉力。

### 为什么不直接从 mesh 边界提取

尝试过两种直接提取方案，均失败：

- **Correction-based**：比较 XPBD 约束求解前后 insertion 顶点位移，反推力。
  问题：volume/ARAP 约束校正混入信号（噪声），且快速收缩时 `tau < 0` 导致
  校正方向反转，32% substep 力被 clip 到 0。
- **Direct kernel force sum**：直接读取 kernel 输出在 insertion 顶点的力。
  问题：缺少 ATTACH 约束的自限反馈，产生正反馈发散。

### Hybrid 方案

从 mesh 变形状态计算力，用 1D Hill 公式作为 constitutive law：

```
pos_np = sim.pos.numpy()
stretches = compute_fiber_stretches(pos_np, tet_idx, rest_matrices, fiber_dirs)
l_tilde = mean(stretches) / lm_opt          # mesh 平均纤维拉伸
v_norm = (l_tilde - l_tilde_prev) / dts / V_max   # 有限差分速度
F = clip(a * f_L(l_tilde) * f_V(v_norm) + f_PE(l_tilde) + d * v_norm, 0) * F_max
```

- `l_tilde` 来自 mesh 变形梯度的 per-tet 计算（3D 几何驱动），非 MuJoCo 运动学
- `compute_fiber_stretches` 对每个 tet 计算 `|F·d|`，取全 mesh 平均
- 1D Hill 公式（f_L, f_V, f_PE, damping）充当本构关系，将应变映射为力

## 力施加：MuJoCo motor actuator

```python
mj_data.ctrl[0] = muscle_force_3d
```

MuJoCo 中的 spatial tendon 连接 origin site 和 insertion site，motor actuator
以 `gear=-1` 作用于该 tendon。正的 `ctrl` 值产生沿 tendon 缩短方向的力，
即 biceps 的屈肘力矩。

## 位置耦合：ATTACH 约束

每个 substep 开始，从 MuJoCo 骨骼位姿更新 mesh 的 ATTACH targets：

1. 读取 `site_xpos[origin]` 和 `site_xpos[insertion]`
2. 根据 rigid tendon 模型计算当前纤维长度：`l_fiber = path_length - L_slack`
3. 沿纤维方向缩放 bone targets（纤维方向按 `l_fiber / mesh_length` 缩放，横向不变）
4. ATTACH 约束（k=1e10）将 mesh 端点锚定到这些 targets

这确保 mesh 几何跟随关节角度变化而拉伸/收缩。

## Substep 时序

```
for substep:
    1. mj_forward()                    # 读 MuJoCo 状态
    2. 更新 ATTACH targets             # MuJoCo → mesh 位置耦合
    3. accumulate_active_fiber_force()  # mesh 内力（含 f_V）
    4. integrate()                     # 位置/速度前进
    5. solve_constraints()             # ATTACH + volume + ARAP
    6. update_velocities()             # 从位置校正更新速度
    7. 计算 l_tilde, v_norm → F        # hybrid 力提取
    8. for mj_substep:                 # MuJoCo 子步
           ctrl[0] = F
           mj_step()
```

力在 XPBD substep 级别计算（dts = dt / num_substeps），
应用于 `mj_per_xpbd` 个 MuJoCo 子步。
