# SimpleArm 阶段 2: VBD 体积肌肉 + MuJoCo 骨骼

日期：2026-03-28

## 已完成

- 重写 `examples/example_vbd_mujoco_simple_arm.py` — VBD sigma0>0 + MuJoCo 刚体耦合
- **RMSE=1.75°，Max error=8.08°**，稳态 82.16° vs OpenSim 82.2°
- USD 动画导出：`output/anim/simple_arm_vbd.usda`（600帧 tet mesh）


提取肌肉力方法:
```python
stretch = |Fd| # fiber stretch from deformation gradient
stretch_to_ltilde = mesh_L / L_opt    # rest mesh 长度 / 最优纤维长度
l_tilde_vbd = mean(stretches) * stretch_to_ltilde
ten_length = float(mj_data.ten_length[0]) 
fiber_length = ten_length - L_slack # 刚性肌腱，fiber_length = ten_length - slack_length
l_tilde_1d = fiber_length / L_opt # 1D 纤维长度
delta_ltilde = l_tilde_vbd - (fiber_length / L_opt)  # VBD correction, 反映3D变形对纤维长度的影响
l_tilde_now = l_tilde_1d + delta_ltilde # 用于DGF 力计算的纤维长度
fl = float(active_force_length(l_tilde_now)) # DGF曲线计算出肌肉力
fpe = float(passive_force_length(l_tilde_now))
fv = float(force_velocity(np.clip(v_norm, -1.0, 1.0)))
muscle_force = (activation * fl * fv + fpe + d_damp * v_norm) * F_max
muscle_force = np.clip(muscle_force, 0.0, F_max * 2.0)
```

## 架构

### VBD vmuscle 主动参与（sigma0 > 0）

VBD 的 vmuscle kernel (`accumulate_fiber_force_and_hessian`) 在 solver 内部：
1. 计算 DGF 力-长度、力-速度、被动力曲线
2. 累加纤维力和 Hessian 到全局系统（与 Neo-Hookean 弹性力一起）
3. 求解 `dx = H⁻¹f` 得到 3D 弹性+肌肉平衡态

sigma0 = F_max / (π × r²) ≈ 159155 Pa

### 两级耦合

**VBD 层（慢，每 outer step = 0.016s）：**
1. 从 MuJoCo tendon length 计算 fiber_length
2. 重置 mesh 顶点到对应位移（origin+insertion kinematic）
3. 设置 tet_activations（传入当前激活水平）
4. VBD step（sigma0>0, iterations=50）→ 3D 弹性+肌肉平衡
5. 从 deformation gradient 提取 per-tet fiber stretch
6. 计算 volumetric delta = l_tilde_vbd - l_tilde_1D

**MuJoCo 层（快，每 substep = 0.002s）：**
1. 实时激活动力学（tau_act=15ms 需要小步长）
2. 读取实时 tendon length → 1D l_tilde
3. 应用 VBD 修正：l_tilde_now = 1D_l_tilde + delta
4. DGF 力 = (a × fL(l̃) × fV(ṽ) + fPE + d×ṽ) × Fmax
5. MuJoCo step

### VBD Mesh 配置

- 圆柱体 240 tets (8 circumferential × 10 axial)
- 创建在 initial fiber_length = 0.3044m
- sigma0 = 159155 Pa（VBD vmuscle 主动收缩）
- k_mu=1000, k_lambda=10000, k_damp=1.0, density=1060
- Top 11 + Bottom 11 vertices: kinematic (mass=0)
- iterations=20, vbd_substeps=20（小 dt 比多迭代更有效）

### 关键发现

1. **sigma0>0 让 VBD 真正计算肌肉力**：vmuscle kernel 在 solver 内部执行 DGF 曲线计算（力-长度、力-速度、被动力），产生的 force+Hessian 与弹性力一起参与 3D 求解。这与 sigma0=0（仅被动弹性）有本质区别。

2. **直圆柱体 delta≈0 是物理正确的**：对均匀纤维方向的直圆柱，3D fiber stretch = ||F @ fiber_dir|| = axial stretch = 1D stretch。VBD 的价值在非均匀几何（变截面、wrapping）中才体现。

3. **sigma0>0 + kinematic 边界的径向不稳定**：当 scale > ~1.2（纤维被拉长超过 rest 的 120%）时，sigma0 收缩力与 kinematic 拉伸严重冲突，能量进入径向变形导致 mesh 爆炸。解决：(a) scale guard——scale 超出 [0.5, 1.2] 时跳过 VBD，用运动学缩放替代；(b) stretch sanity check——过滤极端 l_tilde 值。

4. **时间戳精度重要**：config dt=0.0167s ≠ substeps*mj_dt=8*0.002=0.016s。必须使用实际物理时间（累加 mj_dt）记录，否则瞬态曲线漂移导致误差放大。

5. **激活动力学必须在 substep 级别更新**：tau_act=0.015s < outer_dt=0.016s，在 outer step 级别更新会丢失动态细节。

## 数值验证

| 量 | OpenSim DGF | VBD+MuJoCo | MuJoCo 1D (阶段1) |
|----|-------------|------------|-----|
| 稳态角度 | 82.24° | 82.16° | 82.16° |
| 稳态力 | ~36.6N | 36.4N | 36.4N |
| RMSE | — | 1.75° | 1.80° |
| Max error | — | 8.08° | 8.28° |

## 与 Sliding Ball 的对比

| 特性 | Sliding Ball | SimpleArm Stage 2 |
|------|-------------|-------------------|
| sigma0 | 3e5 Pa | 1.6e5 Pa |
| 边界 | Top kinematic, bottom FREE | Both kinematic |
| 负载 | 重力 (ball mass) | MuJoCo 骨骼动力学 |
| VBD 作用 | 驱动形变+力 | 3D 平衡+stretch 提取 |
| 力来源 | VBD 平衡态 stretch → DGF | VBD 平衡态 stretch → DGF |
| Delta | N/A (VBD 是唯一模拟器) | ≈0 (直圆柱，预期) |

## 下一步

- 测试非均匀肌肉（变截面 mesh、wrapping）验证 delta≠0
- 使用 SolverMuscleBoneCoupled + ATTACH 约束实现 sigma0>0 + 自由 insertion（完整动力学耦合）
- 对比不同材料参数对 VBD 稳定性的影响
