# VBD vs OpenSim 滑动球肌肉对比实验

**日期**: 2026-03-24
**分支**: feat/vbd-muscle

## 实验目标

验证 VBD 体积肌肉模型（DGF Hill-type）与 OpenSim DeGrooteFregly2016Muscle 在 1-DOF 滑动球场景下的一致性。

## 模型配置

| 参数 | 值 |
|------|-----|
| muscle_length | 0.10 m |
| muscle_radius | 0.02 m |
| sigma0 | 3.0e5 Pa |
| ball_mass | 10.0 kg |
| F_max | 377.0 N |
| weight | 98.1 N |
| weight/F_max | 0.260 |

### VBD 参数
- k_mu=1.0, k_lambda=1e3 (小剪切、大体积保持)
- k_damp=0.5 (SNH Rayleigh 阻尼)
- fiber_damping=0.05 (DGF 纤维阻尼, 新增)
- dt=1/60, iterations=50
- Activation dynamics: DGF 一阶 ODE (tau_act=15ms, tau_deact=60ms), 子步进 ~1ms
- F-V: 禁用 (v_max=1e6)

### OpenSim 参数
- DeGrooteFregly2016Muscle
- fiber_damping=0.01, ignore_tendon_compliance=True
- ignore_activation_dynamics=False
- max_contraction_velocity=10.0

## 关键发现

### 1. 稳态一致性
- VBD 稳态: position=0.0383, norm_fiber=0.604
- OpenSim 稳态: position=0.0400, norm_fiber=0.590
- 两者稳态力都收敛到 weight/F_max=0.260
- 差异来源: VBD 的 SNH 弹性材料提供额外抗缩力

### 2. SNH 刚度影响
- k_mu, k_lambda 越大, VBD 与 OpenSim 偏差越大 (SNH 抗缩力)
- k_mu=0 不稳定 (Rayleigh 阻尼依赖 Hessian)
- 最优: k_mu=1 (极小剪切) + k_lambda=1e3 (体积保持)

### 3. 减振方法对比

| 方法 | 效果 | 备注 |
|------|------|------|
| **fiber_damping** | **最好** | 沿纤维方向的物理阻尼, 最有针对性 |
| k_damp (Rayleigh) | 好 | 但太大会 NaN (>0.8), 与 dt 耦合 |
| 增大迭代 | 无效 | 超调由惯性预测决定, 非收敛精度 |
| 减小 dt | 需配套减 k_damp | dt 减半 → Rayleigh 力翻倍, 容易发散 |
| 慢 ramp | 减振但增超调 | 慢激活 → 重力先拉球下坠 |
| activation dynamics | 有效 | 物理平滑, 但 dt=1/60 时需子步进 |

### 4. Fiber Damping 实现
在 `vmuscle_kernels.py` 中实现了 DGF fiber damping:
- 力: `F_damp = -fiber_damping * sigma0 * rest_volume * (dl~/dt) * dldx`
- Hessian: `H_damp = fiber_damping * sigma0 * rest_volume / dt * outer(dldx, dldx)`
- `dldx = dot(m, fiber_dir) * Fd0_hat` (沿变形后纤维方向)
- 通过 `builder.vmuscle_fiber_damping` 设置

### 5. Activation Dynamics
- VBD 端使用 `activation_dynamics_step_np` (src/VMuscle/activation.py)
- tau_act=15ms << dt=17ms, 需要子步进 (~17 sub-steps at 1ms)
- 与 OpenSim 的 DGF activation dynamics 一致

## 最终参数推荐

```python
k_mu=1.0, k_lambda=1e3, k_damp=0.5
fiber_damping=0.05
dt=1/60, iterations=50
activation dynamics with 1ms sub-stepping
```

## 输出文件
- VBD: `output/vbd_muscle_sliding_ball_default.npz`
- OpenSim: `output/opensim_sliding_ball.sto` (在 OpenSimExample 仓库)
- 对比图: `output/vbd_muscle_demo_default.png`
