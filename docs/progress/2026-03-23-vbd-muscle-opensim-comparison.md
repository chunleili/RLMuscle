# VBD Muscle 实现与 OpenSim 对比验证

> 日期：2026-03-23

## 概述

在 Newton VBD 求解器中实现了 DGF Hill-type 肌肉纤维能量，并与 OpenSim DeGrooteFregly2016Muscle 的 1-DOF sliding-ball 模型进行了对比验证。通过参数调节，VBD 稳态 fiber length 与 OpenSim 仅差 3.3%。

## 已完成

### 核心实现（Newton 子模块）
- `vmuscle_kernels.py`：DGF 曲线（active F-L 3高斯、passive F-L 指数、F-V sinh⁻¹）+ 纤维能量导数 + PK1/Hessian
- `vmuscle_launch.py`：kernel launch 包装
- `solver_vbd.py`：插入 `particle_q_prev2` 二步历史缓冲 + 颜色组循环中调用肌肉 kernel
- `builder.py`：`vmuscle_*` 字段（fiber_dirs, sigma0, activations, max_contraction_velocity）

### 示例与工具（主项目）
- `examples/example_muscle_sliding_ball.py`：悬挂圆柱 + 集中质量 + activation ramp → CSV 输出
- `scripts/plot_sliding_ball.py`：读 CSV 画图，可叠加 OpenSim npz 参考数据
- `src/VMuscle/mesh_utils.py`：`create_cylinder_tet_mesh`, `set_vmuscle_properties`, `fix_tet_winding` 等

### 关键参数调节发现

1. **准静态 F-V=1 有助于拟合 OpenSim**
   - 显式滞后 F-V（从 q_prev/q_prev2 计算速度）在 VBD 隐式位置求解中引发速度振荡
   - F-V 曲线不对称（缩短侧 fV<1 下降快于拉伸侧 fV>1 上升）导致振荡下平均 fV<1，系统性减弱收缩力
   - 即使 dt 缩小到 1/240、加阻尼，仍无法收敛到正确平衡点（l̃≈0.92 vs 正确值 0.61）
   - **设 v_max=1e6（等效 fV≡1）后 VBD 正确收敛到 l̃=0.610，与 OpenSim nfl=0.590 仅差 3.3%**
   - 结论：准静态对比中应禁用 F-V；F-V 本身非 bug，但需要隐式/平滑速度估计才能在动态场景正常工作

2. **小被动弹性刚度有助于拟合 OpenSim**
   - k_mu 从 1e5（原值）降到 1e3（真实肌肉被动组织 ~1-10 kPa）
   - k_lambda 从 1e5 降到 1e4
   - 使 DGF 纤维力主导行为，Neo-Hookean 弹性仅提供最小结构支撑
   - k_mu=1e5 时 l̃=0.70（弹性抵抗占主导），k_mu=1e3 时 l̃=0.61（接近 OpenSim 0.59）

3. **其他参数**
   - 悬挂配置（固定顶端）：匹配 OpenSim 的 ceiling-to-ball 设置，重力对抗收缩
   - ball_mass=10kg：使 F_max/weight=3.84，平衡点在 F-L 曲线生理范围内
   - ramp_steps=3（~0.05s）：快速激活，避免 ramp 期间 ball 掉出 active F-L 范围

### 稳态对比结果

| 指标 | VBD | OpenSim | 误差 |
|------|-----|---------|------|
| 位移 | +0.038 m | +0.040 m | 5% |
| fiber length | 0.610 | 0.590 | 3.3% |

剩余 3.3% 差异来源：Neo-Hookean 弹性抵抗（~26N）、分布质量 vs 点质量、3D 径向膨胀效应。

### Bug fixes
- DGF Gaussian 2 除零（l̃≈0.15 处 denominator 过零）→ epsilon guard
- 零初始化 particle_q_prev 导致 F-V NaN → 初始化为 model.particle_q
- 负 d²Ψ/dI₅² 导致不定 Hessian → clamp ≥ 0（SPD 投影）
- 反向四面体（负体积）→ fix_tet_winding

## 下一步

- 修复 F-V 滞后问题：隐式速度估计或平滑滤波，使动态场景也能正确工作
- 用真实 USD 肌肉网格测试（add_soft_vmuscle_mesh + materialW/muscle_id/tendonmask）
- 更精细的网格分辨率测试
- 与 OpenSimExample 的独立 VBD 实现交叉验证
