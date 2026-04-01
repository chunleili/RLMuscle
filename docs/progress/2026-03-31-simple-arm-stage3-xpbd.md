# SimpleArm Stage 3: XPBD Coupled 实现

日期：2026-03-31 ~ 2026-04-01

## 概述

用 XPBD MuscleSim（ATTACH 弹性边界）替代 VBD（kinematic 边界），解决 VBD 在 step 17 的网格爆炸问题。力提取走 1D 肌腱模型（DGF 曲线），XPBD 网格用于 3D 变形可视化。

## 最终架构

```
扁平 loop，每步 dts = dt / num_substeps = 0.000333s:
  1. MuJoCo forward → 读取 insertion site 位置
  2. 更新 bone targets (origin 固定 + insertion 跟随)
  3. 1 次 XPBD solve (integrate → clear → solve → update_vel)
  4. 10 次 MuJoCo substeps (dtmj = dts/10)
     activation dynamics → 1D DGF 力 → mj_step

每 num_substeps(30) 次 → 记录一帧 + 导出 PLY
```

### 约束

| 类型 | 数量 | 刚度 | 作用 |
|------|------|------|------|
| TETVOLUME | 480 | 1e6 | 体积守恒 |
| TETARAP | 480 | 1e6 | 形状保持 |
| ATTACH (origin) | ~18 | 1e37 | 固定近端（超强弹簧，非 kinematic） |
| ATTACH (insertion) | ~18 | 1e10 | 跟随远端骨骼 |

**关键：没有 PIN，没有 stopped=1。** 所有顶点都是动态的，origin 通过 ATTACH(1e37) "几乎"固定但允许微小弹性位移，避免刚性-弹性突变。

### 可视化输出（PLY per frame）

```
Origin site ──[tendon_prox 0.275m, r=5mm]── Muscle [0.304m, r=20mm] ──[tendon_dist 0.275m, r=5mm]── Insertion site
```

| 文件 | 内容 |
|------|------|
| `output/anim_xpbd/frame_XXXX.ply` | 肌肉 tet mesh 表面 |
| `output/anim_xpbd_bones/tendon_prox_XXXX.ply` | 近端肌腱 |
| `output/anim_xpbd_bones/tendon_dist_XXXX.ply` | 远端肌腱 |
| `output/anim_xpbd_bones/humerus_XXXX.ply` | 肱骨 capsule |
| `output/anim_xpbd_bones/radius_XXXX.ply` | 桡骨 capsule |

## 关键文件

| 文件 | 说明 |
|------|------|
| `examples/example_xpbd_coupled_simple_arm.py` | XPBD 耦合主程序 |
| `scripts/run_simple_arm_comparison.py` | `--mode xpbd` |
| `data/simpleArm/config.json` | `xpbd` 配置段 |
| `tests/test_xpbd_simple_arm.py` | 4 个冒烟测试 |

## 验证结果

1000 steps (10s), CPU, dt=0.01:

| 指标 | 值 |
|------|-----|
| 稳态角度 | 82.2 deg |
| 力范围 | [4.1, 68.4] N |
| NaN | 无 |
| Inverted tets | 零 |

## 迭代过程中的关键发现

### 1. stopped=1 是 mesh 爆炸的根因

`stopped=1` 让 origin 顶点 `invmass=0`，约束完全无法移动它们，形成刚性-弹性突变界面。在大变形时边界 tet 必然翻转。

**解决方案**：用 ATTACH(1e37) 替代 PIN+stopped。invmass 非零，约束系统可自然平衡。这与 `example_couple.py`（bicep.json 中 `stiffness=1e37`）的做法一致。

### 2. TETFIBERDGF 在两端约束时不可用

两端固定（PIN/ATTACH）+ fiber 收缩 → 内部等距压缩 → tet 翻转。即使 fiber_stiffness_scale=10（极软），函数内仍爆炸。

**根因**：XPBD 的 target-stretch 驱动方式在 isometric 场景下无法平衡。需要 Millard 能量本构（`C=√(2Ψ)`）用能量最小化替代 target-stretch。

### 3. 约束刚度必须对齐 example_couple 量级

原始 SimpleArm 的刚度（volume=1e5, arap=100, attach=1e6）比稳定的 bicep 示例（volume=1e10, arap=1e10, attach=1e37）低了 4-8 个数量级。提高到 1e6/1e37 后稳定。

### 4. 扁平 loop 消除大步位移

原始嵌套 loop（outer step → XPBD substeps → MuJoCo substeps）导致每个 outer step 的 bone target 跳变 ~10mm（半径的 50%）。改为扁平 loop（每 dts 更新一次 bone target），每步位移降至 ~0.1mm（0.5% 半径）。

### 5. OpenSim tendon 占比 64%

SimpleArm 的 `tendon_slack_length=0.55m` 占总路径 64%。肌肉 mesh 只覆盖 fiber 部分（0.3m），居中放置，两端各接 0.275m tendon 可视化。

## 参数

```json
{
  "solver": {"dt": 0.01, "n_steps": 1000},
  "xpbd": {
    "num_substeps": 30,
    "volume_stiffness": 1e6,
    "arap_stiffness": 1e6,
    "attach_origin_stiffness": 1e37,
    "attach_insertion_stiffness": 1e10,
    "warmup_steps": 10
  }
}
```

## 下一步

1. **能量本构方案**：实现 Millard 样条 + C=√(2Ψ) 精确能量约束，使 XPBD mesh fiber 收缩不产生翻转
2. **XPBD mesh 力提取**：能量本构解决翻转问题后，用 deformation gradient 提取力
3. **CUDA 验证**：当前仅 CPU 验证，需测试 CUDA + colored GS
4. **复杂几何**：测试非圆柱几何（如从 USD 加载的真实肌肉 mesh）
