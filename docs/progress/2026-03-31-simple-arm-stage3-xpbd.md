# SimpleArm Stage 3: XPBD Coupled 实现

日期：2026-03-31

## 概述

用 XPBD MuscleSim（ATTACH 弹性边界）替代 VBD（kinematic 边界），解决 VBD 在 step 17 的网格爆炸问题。力提取走 1D 肌腱模型（DGF 曲线），XPBD 网格用于 3D 变形可视化。

## 架构

```
每个 outer step (dt=0.0167s):
  1. MuJoCo forward → 读取 insertion site 位置
  2. 计算位移 delta = new_insertion - initial_insertion
  3. 更新 bone targets = initial_targets + delta
  4. 设置 activation + contraction_factor (DGF 反求)
  5. XPBD 30 substeps (TETVOLUME + TETFIBERDGF + TETARAP + PIN + ATTACH)
  6. MuJoCo substeps: 1D fiber length → DGF 力 → ctrl 注入
```

## 关键文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `examples/example_xpbd_coupled_simple_arm.py` | 新建 | XPBD 耦合主程序 |
| `scripts/run_simple_arm_comparison.py` | 修改 | 添加 `--mode xpbd` |
| `data/simpleArm/config.json` | 修改 | 添加 `xpbd` 配置段 |
| `tests/test_xpbd_simple_arm.py` | 新建 | 4 个冒烟测试 |

## 验证结果

XPBD Coupled vs MuJoCo-only (DGF), 600 steps, CPU:

| 指标 | 值 |
|------|-----|
| 关节角 RMSE | 0.13 deg |
| 关节角最大误差 | 0.47 deg |
| 稳态角度 | 82.2 deg (两者一致) |
| 力范围 | [4.2, 70.2] N |
| 归一化纤维长度 | [0.508, 0.821] |
| NaN | 无 |
| Mesh 爆炸 | 无 |

## 关键发现

### 1. XPBD mesh 力提取不可靠

XPBD mesh 的 deformation gradient 提取的 fiber stretch 在仿真过程中波动很大（l_tilde 从 0.5 到 3+），不适合直接用于力计算。原因：
- ARAP 与 fiber 约束竞争导致 mesh 变形不符合 1D 肌肉模型
- 两端固定（PIN + ATTACH）时 fiber 收缩导致内部压缩不稳定
- XPBD compliance 机制不保证 mesh 变形路径与连续力学一致

**当前方案**：力计算使用 MuJoCo 1D 肌腱长度（与 MuJoCo-only Stage 1 一致），XPBD mesh 仅用于 3D 变形可视化。

### 2. ARAP 与 fiber 约束不兼容

高 ARAP 刚度（1e6）会阻止 fiber 收缩，导致 solver 发散。低 ARAP 刚度（100）则无法有效防止 mesh 畸变。当前使用 arap_stiffness=100 作为折中。

### 3. Warm-up 必须 activation=0

两端固定 + 非零 activation → fiber 试图收缩但无处可去 → 内部压缩 → solver 发散。Warm-up 阶段不能有 activation。

### 4. Bone target 必须用位移法

`bone_targets = initial_vertex_positions + displacement`，不能直接设为 MuJoCo insertion 位置（那样会把所有 insertion 顶点塌缩到一个点）。

### 5. contraction_factor = 1 - lm_eq

`dgf_equilibrium_fiber_length` 返回的是 `lm_eq`（平衡点纤维长度），contraction_factor 需要 `1 - lm_eq`。这与 sliding ball 示例一致。

## 参数

```json
{
  "num_substeps": 30,
  "attach_stiffness": 1e6,
  "pin_stiffness": 1e8,
  "fiber_stiffness": 1000.0,
  "fiber_stiffness_scale": 200.0,
  "volume_stiffness": 1e5,
  "arap_stiffness": 100.0,
  "warmup_steps": 10
}
```

### 6. TETFIBERDGF + TETARAP 均导致 tet 翻转

**TETFIBERDGF**：两端固定 + fiber 收缩 → 内部等距压缩 → 第一个 substep 就翻转 tet。fss=10（极软）在函数内仍爆炸，fss=200（标准）在 sub 0 就产生 7 m/s 速度。

**TETARAP**：手臂摆动时大变形 → 边界 tet 翻转（step 77）。ARAP 阻碍合理变形。

**最终方案**：仅保留 TETVOLUME + PIN + ATTACH。600 步零翻转。TETFIBERDGF 需等 Millard 能量本构（`C=√(2Ψ)`）才能安全启用。

## 参数

```json
{
  "num_substeps": 30,
  "attach_stiffness": 1e6,
  "pin_stiffness": 1e8,
  "volume_stiffness": 1e5,
  "warmup_steps": 10
}
```

约束：TETVOLUME (240) + PIN (11) + ATTACH (11) = 262 总约束

## 下一步

1. **能量本构方案**：实现 Millard 样条 + C=√(2Ψ) 精确能量约束（docs/plans/2026-03-31-xpbd-energy-constitutive.md），使 XPBD mesh fiber 收缩不产生翻转
2. **XPBD mesh 力提取**：能量本构解决翻转问题后，用 deformation gradient 提取力（添加 3D 效应校正）
3. **CUDA 验证**：当前仅 CPU 验证，需测试 CUDA + colored GS
4. **复杂几何**：测试非圆柱几何（如从 USD 加载的真实肌肉 mesh）
