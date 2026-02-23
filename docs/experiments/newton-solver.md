# 骨骼-肌肉耦合参数实验

## 稳定性指标

**mae（Mean Absolute Error）：越小越稳定。**

mae = 仿真全程中关节角度与目标角度的平均绝对误差。值越小说明系统跟踪目标越准、振荡越少。

## 测试环境

RTX 5070 Ti Laptop GPU / AMD Ryzen, Warp 1.11.1, CUDA 12.9, Taichi 1.7.4

场景：二头肌-桡骨肘关节耦合（1251顶点, 3938四面体, 11954约束, 10肌肉子步/步）

## 实验 1：Solver 真实性能对比

基于 `example_couple.py` 的完整耦合场景（Taichi PBD 肌肉 + Newton 骨骼），200 步 × 3 次取均值。
仅计时核心 `solver.step()` 循环，排除初始化、模型加载和可视化。

场景：二头肌-桡骨肘关节耦合（1251顶点, 3938四面体, 11954约束, 10肌肉子步/步）。

### 可用 Solver

| 名称 | 后端 | 设备 | 说明 |
|---|---|---|---|
| featherstone | Newton Featherstone | CPU | 显式积分，最快 CPU solver |
| mujoco | MuJoCo (mujoco_warp) | CPU/GPU | 默认 GPU 后端，`implicitfast` 积分器 |
| mujoco_cpu | MuJoCo-C | CPU | `use_mujoco_cpu=True`，纯 CPU 后端 |
| mujoco_cuda | MuJoCo (mujoco_warp) | CUDA | GPU 后端，`integrator="euler"` 绕开 LTO bug |
| xpbd | Newton XPBD | CPU | 约束求解，iterations=2 |
| semi_implicit | Newton SemiImplicit | CPU | 半隐式积分 |
| vbd | Newton VBD | CPU | 不支持 revolute joint，无法用于此场景 |

> **已知问题**：`mujoco_warp` 使用 `implicitfast` 积分器时，`derivative.py` 中 `wp.tile_matmul` LTO 编译失败。`mujoco_cuda` 通过 `integrator="euler"` 绕开此问题。

### 性能基准

测试环境：RTX 5070 Ti Laptop GPU / AMD Ryzen, Warp 1.11.1, CUDA 12.9

| solver | avg ms/step | avg fps | min_s | max_s | trials |
|---|---:|---:|---:|---:|---:|
| mujoco_cpu | 30.94 | 32.3 | 5.555 | 6.939 | 3 |
| semi_implicit | 33.10 | 30.2 | 6.562 | 6.675 | 3 |
| featherstone | 36.05 | 27.7 | 7.059 | 7.333 | 3 |
| xpbd | 42.36 | 23.6 | 7.184 | 9.471 | 3 |
| **mujoco_cuda** | **79.29** | **12.6** | 15.488 | 16.531 | 3 |

**结论**：
- **CPU solver 差距不大**：mujoco_cpu（30.9）< semi_implicit（33.1）< featherstone（36.1）< xpbd（42.4），瓶颈主要在 Taichi 肌肉 PBD 求解，骨骼 solver 差异约 10ms 范围内。
- **mujoco_cuda 最慢**（79.3 ms/step，12.6 fps），是最快 CPU solver 的 2.6 倍。原因：`integrator="euler"`（绕开 `tile_matmul` LTO bug）+ 单关节场景数据量太小，GPU kernel launch 开销远大于计算收益。mujoco_warp 的优势需在大规模多关节/多体场景中才能体现。
- **xpbd 波动较大**（min 7.2s vs max 9.5s），其他 solver 相对稳定。


## 实验 2：参数稳定性研究

基于真实 Newton + Taichi PBD 耦合仿真（非简化模型），对 5 种 solver 进行单变量参数扫描。
使用脉冲激活模式（act=0 → 0.8 → 0），每组 180 步（3s @60Hz）。

### 基准参数

```
k_coupling       = 5000.0    # 耦合弹簧刚度
max_torque       = 50.0      # 力矩截断上限 (N·m)
torque_smoothing = 0.3       # EMA 平滑系数
dt               = 1/60      # 时间步长
armature         = 1.0       # 关节惯量
friction         = 0.9       # 关节摩擦
```

### 指标说明

| 指标 | 说明 |
|---|---|
| peak_angle | 全程最大关节偏转角 (rad) |
| mae | 全程 \|angle\| 均值 |
| tail_mae | 最后 20% 步的 \|angle\| 均值（衰减质量） |
| energy_ratio | 最后10步角速度均值 / 最大角速度（残余能量比） |

### 2.1 k_coupling 扫描

| solver | k_coupling | peak | mae | tail_mae | energy |
|---|---:|---:|---:|---:|---:|
| featherstone | 1000 | 0.39 | 0.19 | 0.17 | 0.35 |
| featherstone | 3000 | 0.87 | 0.43 | 0.32 | 0.23 |
| featherstone | 5000 | 1.14 | 0.51 | 0.29 | 0.14 |
| featherstone | 8000 | 1.46 | 0.54 | 0.14 | 0.08 |
| featherstone | 12000 | 3.08 | 1.12 | 1.43 | 0.86 |
| mujoco_cpu | 1000 | 0.27 | 0.16 | 0.24 | 0.11 |
| mujoco_cpu | 3000 | 0.80 | 0.43 | 0.43 | 0.16 |
| mujoco_cpu | 5000 | 1.08 | 0.55 | 0.45 | 0.10 |
| mujoco_cpu | 8000 | 1.41 | 0.60 | 0.37 | 0.03 |
| mujoco_cpu | 12000 | 3.05 | 1.06 | 1.35 | 0.94 |
| mujoco_cuda | 1000 | 0.27 | 0.16 | 0.24 | 0.11 |
| mujoco_cuda | 5000 | 1.10 | 0.56 | 0.47 | 0.12 |
| mujoco_cuda | 12000 | 3.07 | 1.10 | 1.37 | 0.93 |
| xpbd | * | DIVERGED | - | - | - |
| semi_implicit | * | DIVERGED | - | - | - |

**结论**：
- k↑ → peak↑，k=12000 时 peak~3.0，能量几乎不衰减（energy≈0.9），系统不稳定。
- 最佳范围 k=3000–8000，其中 k=5000 时 mae 最小且 energy 较低。
- mujoco 系列表现更稳定（energy 更低），featherstone 在 k=8000 有最低 energy（0.08）但 peak 偏高。

### 2.2 max_torque 扫描

| solver | max_torque | peak | mae | tail_mae | energy |
|---|---:|---:|---:|---:|---:|
| featherstone | 10 | 1.06 | 0.50 | 0.36 | 0.25 |
| featherstone | 25 | 1.13 | 0.52 | 0.29 | 0.14 |
| featherstone | 50 | 1.14 | 0.52 | 0.30 | 0.14 |
| featherstone | 100 | 1.14 | 0.52 | 0.29 | 0.14 |
| featherstone | 200 | 1.14 | 0.52 | 0.30 | 0.14 |
| mujoco_cpu | 10 | 0.99 | 0.52 | 0.49 | 0.18 |
| mujoco_cpu | 50 | 1.08 | 0.56 | 0.46 | 0.10 |
| mujoco_cpu | 200 | 1.08 | 0.55 | 0.45 | 0.10 |

**结论**：
- max_torque≥25 后效果趋于一致，说明实际肌肉力矩 <25 N·m，截断不起作用。
- max_torque=10 有轻微限制效应（peak 略低、energy 略高）。
- **建议 max_torque≥50**，保留裕量。

### 2.3 torque_smoothing 扫描

| solver | smoothing | peak | mae | tail_mae | energy |
|---|---:|---:|---:|---:|---:|
| featherstone | 0.0 | 1.14 | 0.52 | 0.29 | 0.14 |
| featherstone | 0.3 | 1.13 | 0.51 | 0.29 | 0.14 |
| featherstone | 0.8 | 1.14 | 0.51 | 0.29 | 0.14 |
| mujoco_cpu | 0.0 | 1.08 | 0.55 | 0.46 | 0.10 |
| mujoco_cpu | 0.5 | 1.07 | 0.55 | 0.46 | 0.10 |
| mujoco_cpu | 0.8 | 1.08 | 0.56 | 0.46 | 0.10 |

**结论**：
- **torque_smoothing 对稳定性几乎无影响**。所有值变化 <1%。
- 该参数主要影响力矩的平滑程度，在当前场景中力矩变化不剧烈。

### 2.4 dt 扫描

| solver | dt | peak | mae | tail_mae | energy |
|---|---:|---:|---:|---:|---:|
| featherstone | 1/30 | 3.07 | 0.97 | 2.52 | 0.16 |
| featherstone | 1/60 | 1.14 | 0.52 | 0.30 | 0.14 |
| featherstone | 1/120 | 1.08 | 0.53 | 0.36 | 0.13 |
| mujoco_cpu | 1/30 | 3.05 | 0.86 | 2.12 | 0.53 |
| mujoco_cpu | 1/60 | 1.08 | 0.56 | 0.46 | 0.10 |
| mujoco_cpu | 1/120 | 1.03 | 0.56 | 0.52 | 0.10 |
| mujoco_cuda | 1/30 | 3.06 | 0.80 | 1.71 | 0.84 |
| mujoco_cuda | 1/60 | 1.10 | 0.56 | 0.46 | 0.12 |
| mujoco_cuda | 1/120 | 1.05 | 0.57 | 0.53 | 0.11 |
| xpbd | 1/30 | DIVERGED(16) | - | - | - |
| xpbd | 1/60 | DIVERGED(31) | - | - | - |
| xpbd | 1/120 | DIVERGED(62) | - | - | - |
| semi_implicit | * | DIVERGED(0) | - | - | - |

**结论**：
- **dt=1/30 对所有 solver 都不稳定**（peak~3.0），**dt=1/60 是最佳平衡点**。
- dt=1/120 与 1/60 差异很小，不值得加倍计算量。
- XPBD 的 diverge 步数与 dt 呈反比（16/31/62），均在 t≈0.5s（激活开始时）发散。

### 2.5 Solver 兼容性总结

| solver | 耦合稳定性 | 原因 |
|---|---|---|
| **featherstone** | **稳定** | 关节空间直接积分，`joint_f` 直接加入 `joint_tau` |
| **mujoco_cpu** | **稳定** | 广义坐标积分，正确处理 `joint_f` |
| **mujoco_cuda** | **稳定** | 同 mujoco_cpu（使用 euler 积分器） |
| xpbd | **发散** | 位置基约束求解，`joint_f` 转为 `body_f` 但约束覆盖导致位移发散 |
| semi_implicit | **发散** | 体空间半隐式积分，`body_f` 初始化/符号问题导致 step 0 即 NaN |

> XPBD/SemiImplicit 在纯骨骼场景（无耦合力）下可正常运行，但在肌肉耦合场景中 `joint_f` → `body_f` 的转换与约束求解冲突，导致数值发散。

---

## 实验 3：可控性研究

测试 activation 对关节角度/力矩的控制能力。每个 activation 级别独立运行 2s（120步 @60Hz），取后 30% 作为稳态。

### 激活模式

1. **独立级别测试**：act = 0, 0.25, 0.5, 0.75, 1.0，各自从静止运行至稳态
2. **阶跃响应**：act=0(1s) → act=0.8(2s) → act=0(3s)，测量超调和停止速度

### 可控性指标

| 指标 | 说明 |
|---|---|
| grade | 综合评级 A/B/C/D/F |
| monotonicity | act↑ → \|angle\|↑ 的正确比例（1.0=完美） |
| act_angle_corr | Pearson r(activation, steady-state angle) |
| overshoot | (peak - steady) / steady（阶跃响应） |
| settling_steps | 去激活后角速度收敛所需步数 |
| rest_stable | 最后 30 步角度标准差（是否振荡） |
| reversible | 去激活后角度是否回到 <80% 激活时的值 |

### 稳态角度 vs 激活级别

| solver | act=0 | act=0.25 | act=0.5 | act=0.75 | act=1.0 |
|---|---:|---:|---:|---:|---:|
| featherstone | 0.000 | 0.887 | 0.908 | 0.901 | 0.899 |
| mujoco_cpu | 0.000 | 0.961 | 0.958 | 0.944 | 0.957 |
| mujoco_cuda | 0.000 | 0.947 | 0.965 | 0.944 | 0.948 |
| xpbd | - | DIVERGED | DIVERGED | DIVERGED | DIVERGED |
| semi_implicit | DIVERGED | DIVERGED | DIVERGED | DIVERGED | DIVERGED |

### 可控性评级

| solver | grade | mono | angle_corr | torque_corr | overshoot | settle | rest_std | reversible |
|---|:---:|---:|---:|---:|---:|---:|---:|:---:|
| featherstone | **C** | 0.50 | 0.712 | 0.707 | 0.260 | 167 | 0.009 | Y |
| mujoco_cpu | **C** | 0.50 | 0.702 | 0.688 | 0.126 | 0 | 0.001 | Y |
| mujoco_cuda | **B** | 0.75 | 0.703 | 0.692 | 0.157 | 0 | 0.001 | Y |
| xpbd | **F** | - | - | - | - | - | - | - |
| semi_implicit | **F** | - | - | - | - | - | - | - |

### 结论

**所有工作 solver 的可控性都不理想（C-B 级）**，根本原因：

1. **激活饱和**：act=0.25 时稳态角度已达 ~0.9 rad，act=1.0 时仅 ~0.95 rad。不同激活水平的区分度极低（<0.08 rad），肌肉力在低激活时即已饱和。
2. **非单调响应**：act=0.5 时角度略高于 act=0.75 和 1.0。这是因为更高的激活产生更大的肌肉变形，改变了力臂方向，导致有效力矩反而降低。
3. **featherstone 停止较慢**：settling=167步（2.8s），而 MuJoCo 系列立即收敛（0步），说明 MuJoCo 的隐式积分有天然阻尼效果。
4. **超调可接受**：featherstone 0.26，mujoco 0.13-0.16。
5. **可逆性良好**：三个工作 solver 去激活后都能回到较低角度。

**改进方向**：
- 降低 `k_coupling`（如 2000）使激活-角度关系更线性
- 增大 `armature`（如 3.0）增加关节惯性，拉开不同激活水平的区分度
- 或重新设计耦合机制，使力矩与 activation 更直接耦合（当前是 activation → PBD fiber → reaction → torque 的间接路径）

---

## 运行命令

```bash
# 性能基准
uv run python tests/test_perf_couple.py

# 参数研究（完整）
uv run python tests/test_solver_parameters.py

# 参数研究（快速）
uv run python tests/test_solver_parameters.py --quick

# 仅稳定性 / 仅可控性
uv run python tests/test_solver_parameters.py --study stability
uv run python tests/test_solver_parameters.py --study control
```
