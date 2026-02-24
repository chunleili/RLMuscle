# 可控性改进：主动静止长度收缩 — 实施计划

## Context

骨骼-肌肉耦合系统的可控性不理想（C-B 级）。根因在 `muscle.py:1269-1302`：activation 仅调制纤维约束的 **刚度**（stiffness），而非约束的 **目标**（target）。

当前约束函数 `C = ||Fw||` 对所有 activation > 0 都试图将纤维"坍缩到零长度"，区别仅在于"有多用力"。由于 stiffness 在 act ≈ 0.1 时已达到实质刚性（25M），不同激活水平产生的肌肉形变几乎相同 → 反作用力相近 → 力矩相近 → 稳态角度只有 < 0.08 rad 的区分度。

**目标**：改为 `C = ||Fw|| - target_stretch(activation)`，使不同激活水平产生不同的纤维目标长度，从根本上建立 activation → 角度的正比关系。

## 数学原理

### 当前模型
```
C = ||Fw||          (线性能量，目标 = 0)
stiffness ∝ activation × 10000  (act=0 → 跳过约束, act=1 → 1e8)
```
所有非零 activation 都试图坍缩纤维，仅靠 volume/ARAP 约束平衡。结果：所有 act>0.1 产生几乎相同的平衡形态。

### 新模型
```
C = ||Fw|| - target_stretch     (线性能量，目标 = target_stretch)
target_stretch = 1.0 - belly_factor × activation × contraction_ratio
```
- act=0: target=1.0（静止长度，无力）
- act=0.25: target=0.925（收缩 7.5%）
- act=0.5: target=0.85（收缩 15%）
- act=1.0: target=0.7（收缩 30%）

### 梯度不变性
```
∂C/∂x = ∂(||Fw|| - target)/∂x = ∂||Fw||/∂x
```
target_stretch 是关于顶点位置的常数，**梯度计算完全不变**。唯一改变的是约束值 C 本身（`muscle.py:1741`）。

---

## 实施步骤

### Step 1: SimConfig 添加参数

**文件**: `src/VMuscle/muscle.py:12-37` (SimConfig dataclass)

```python
contraction_ratio: float = 0.3       # 最大纤维收缩比 (act=1.0 时纤维缩短到 rest 的 70%)
fiber_stiffness_scale: float = 500.0  # 纤维刚度倍率 (替代硬编码 10000.0)
```

- `contraction_ratio=0.3` 对应真实二头肌约 30% 最大缩短（生理范围 20%-50%）
- `fiber_stiffness_scale=500.0`：从 10000.0 降到 500.0，使约束从"无限刚性"变为"强但有弹性"，让不同激活水平有更好的力-位移梯度

### Step 2: MuscleSim 存储参数供 Taichi kernel 访问

**文件**: `src/VMuscle/muscle.py:407-431` (MuscleSim.__init__)

在 `self.use_jacobi = False` 附近添加：
```python
self.contraction_ratio = self.cfg.contraction_ratio
self.fiber_stiffness_scale = self.cfg.fiber_stiffness_scale
```

Taichi 1.7 中，kernel 可通过 `self.xxx` 访问 Python 标量属性（与 `self.dt` 机制相同）。

### Step 3: 修改约束分发代码 (核心修改 1/2)

**文件**: `src/VMuscle/muscle.py:1269-1302` (solve_constraints 中 TETFIBERNORM 分支)

原代码：
```python
acti = self.activation[tetid]
stiffness = self.cons[c].stiffness
_tendonmask = self.tendonmask[tetid]
fiberscale = self.transfer_tension(acti, _tendonmask)
stiffness = stiffness * fiberscale * 10000.0
if stiffness <= 0.0:
    continue
```

新代码：
```python
acti = self.activation[tetid]
_tendonmask = self.tendonmask[tetid]
belly_factor = 1.0 - _tendonmask

# 刚度: 通过 transfer_tension 保持激活依赖（肌肉激活时变硬）
fiberscale = self.transfer_tension(acti, _tendonmask)
stiffness = self.cons[c].stiffness * fiberscale * self.fiber_stiffness_scale
if stiffness <= 0.0:
    continue

# 目标纤维拉伸: activation 越高，目标越短
target_stretch = 1.0 - belly_factor * acti * self.contraction_ratio
```

然后在调用 `self.tet_fiber_update_xpbd(...)` 时追加 `target_stretch` 参数。

### Step 4: 修改纤维约束函数 (核心修改 2/2)

**文件**: `src/VMuscle/muscle.py:1608-1754` (tet_fiber_update_xpbd)

#### 4a. 函数签名添加参数

在 `stopped: ti.template()` 后添加：
```python
target_stretch: ti.f32 = 1.0,
```

#### 4b. 修改约束值计算

将第 1741 行：
```python
C = psi
```
改为：
```python
C = psi - target_stretch
```

**仅此一行改变**。梯度 (grad0-grad3)、XPBD 更新 (dL)、位置校正全部不变。

### Step 5: 更新配置文件

**文件**: `data/muscle/config/bicep.json`

在顶层添加：
```json
"contraction_ratio": 0.3,
"fiber_stiffness_scale": 500.0,
```

### Step 6: 同步 reset 和其他入口

检查 `MuscleSim.reset()` 等是否需要同步更新。由于 `contraction_ratio` 和 `fiber_stiffness_scale` 是从 `self.cfg` 读取的常量，reset 时不需要额外处理。

---

## 关键文件清单

| 文件 | 修改内容 |
|------|----------|
| `src/VMuscle/muscle.py:12-37` | SimConfig 添加 2 个新字段 |
| `src/VMuscle/muscle.py:429-431` | MuscleSim.__init__ 存储参数 |
| `src/VMuscle/muscle.py:1269-1302` | solve_constraints 分发：计算 target_stretch，替换刚度公式 |
| `src/VMuscle/muscle.py:1608-1628` | tet_fiber_update_xpbd 签名：添加 target_stretch 参数 |
| `src/VMuscle/muscle.py:1741` | 约束值：`C = psi` → `C = psi - target_stretch` |
| `data/muscle/config/bicep.json` | 添加 contraction_ratio, fiber_stiffness_scale |

**不需要修改的文件**：
- `solver_muscle_bone_coupled.py` — 耦合力矩提取逻辑不变
- `test_solver_parameters.py` — 测试逻辑不变，只需重新运行
- `example_couple.py` — 入口不变

---

## 验证方案

### 1. 冒烟测试：交互式可视化
```bash
uv run python examples/example_couple.py
```
确认肌肉收缩形变随 activation 滑块有明显可见的渐变效果（而非二值开关）。

### 2. 可控性量化测试
```bash
uv run python tests/test_solver_parameters.py --study control
```
期望改善：
- monotonicity: 0.50 → ≥ 0.75（理想 1.0）
- act_angle_corr: 0.71 → ≥ 0.90
- grade: C → B 或 A
- 稳态角度在不同激活水平间有明显区分度（> 0.1 rad 间隔）

### 3. 稳定性回归测试
```bash
uv run python tests/test_solver_parameters.py --study stability
```
确认 featherstone 和 mujoco 系列仍然稳定（不发散），mae 无显著恶化。

### 4. 参数敏感性
若可控性仍不理想，按以下顺序调参：
1. 增大 `contraction_ratio`（0.3 → 0.4 → 0.5）
2. 降低 `fiber_stiffness_scale`（500 → 200 → 100）使约束更柔顺
3. 降低 `k_coupling`（5000 → 3000）减小弹簧刚度

---

## 备选方案（方案 1 走不通时参考）

### 方案 2: 分离被动/主动力（Hill 模型）
将 activation 的作用拆成两个独立通道：被动弹性刚度（常驻）+ 主动收缩力（正比于 activation，沿纤维方向）。在 `integrate()` 阶段作为外力施加。更接近 Hill-type 模型，但需重构力模型。

### 方案 3: 混合直接力矩（最快见效）
保持 PBD 肌肉提供视觉形变，在力矩层叠加直接分量：
`tau_total = blend * tau_PBD + (1-blend) * activation * k_direct * axis`。
仅改 `solver_muscle_bone_coupled.py` 约 10 行，但物理性较弱。

### 方案 4: 纯调参
降低 `k_coupling`（→2000）、增大 `armature`（→3.0）、降低硬编码 `* 10000.0`。零代码修改但治标不治本。

### 方案 5: 力臂方向修正
使用 rest pose 的力臂方向（而非变形后），消除大变形导致的非单调响应。改 `_compute_torque_kernel`，可与方案 1 组合。