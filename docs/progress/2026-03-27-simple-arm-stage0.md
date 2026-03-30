# SimpleArm 阶段 0: OpenSim 基线

日期：2026-03-28

## 已完成

- 创建 `data/simpleArm/config.json` 统一配置文件
- 重写 `scripts/osim_simple_arm_dgf.py` → 函数接口 `osim_simple_arm_dgf(cfg) -> dict`
- 重写 `scripts/osim_simple_arm_millard.py` → 函数接口 `osim_simple_arm_millard(cfg) -> dict`
- 创建对比脚本 `scripts/run_simple_arm_comparison.py`
- 输出对比图 `output/simple_arm_osim_dgf_vs_millard.png`
- 添加 Ellipsoid 显示几何体（.osim 文件可在 OpenSim GUI 中正确渲染）

## 关键发现

### 参数修正（两轮）
1. **`tendon_slack_length`**：plan 原定 0.001 → 修正为 **0.55**。"刚性肌腱"应通过 `ignore_tendon_compliance=True` 实现，而非将肌腱几何长度设为零。slack_length=0.001 导致归一化纤维长度 ~1.77，运动完全错误（角度反向增大到 178°）。
2. **`fiber_damping`**：0.01 → 0.1 → **0.5**。测试了 0.1/0.5/1.0 三档：稳态不变（~82.2°），但 0.1 仍有明显振荡，0.5 几乎无振荡，1.0 完全静止。选择 0.5 作为平衡点。

### DGF vs Millard 对比（最终结果，fiber_damping=0.5）
- 稳态角度：DGF **82.2°** vs Millard **88.2°**（差 ~6°，固有差异）
- **RMSE=7.15°, Max error=18.23°**（主要来自 0-2s 瞬态差异）
- 稳态 6° 差异原因：两个模型的 force-length 曲线形状不同，DGF 在 l̃=0.57 平衡（F≈36.6N），Millard 在 l̃=0.52 平衡（曲线在短纤维区更宽），不可通过参数调节消除

### 激活初始化差异
- DGF `equilibrateMuscles` 初始化 activation≈0.5（静力学平衡值），excitation=0.3 → 去激活（曲线下降）
- Millard 初始化 activation≈0.05（默认最小值），excitation=0.3 → 激活（曲线上升）
- 不影响稳态，仅影响瞬态

### V_max 与准静态分析
- OpenSim DGF 中阻尼项 `F_damp = d × F_max × v/(V_max × L_opt)`，V_max 越大阻尼越小
- force-velocity 也通过 V_max 归一化：大 V_max → f_V→1（去除速度依赖）
- **设大 V_max 会同时去除两个阻尼机制，导致更多振荡，非准静态**
- 注意：VBD 实现中阻尼项不经过 V_max 归一化（用绝对速度），与 OpenSim 不同

### 最终参数
- `tendon_slack_length=0.55`, `ignore_tendon_compliance=True`（刚性肌腱）
- `ignore_activation_dynamics=False`（激活动力学开启）
- `fiber_damping=0.5`

### 技术注意
- pyopensim 没有 `StatesTrajectoryReporter`，直接用 `manager.getStatesTable()`
- 肘关节坐标名为 `elbow_coord_0`（非 `elbow_angle`）

## 下一步

- 阶段 1：MuJoCo Spatial Tendon + DGF 力注入 vs OpenSim（以 DGF 为基线）
