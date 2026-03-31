# XPBD 肌肉能量本构方案

> 日期: 2026-03-31
> 目标: 探索将肌肉力-长度曲线的能量函数直接嵌入 XPBD 约束，替代当前的"外部平衡反求 + 刚度调制"方案

## 背景

当前 `TETFIBERDGF` 约束的工作方式：
1. CPU 端每步反求 DGF f_L 曲线平衡点 → `contraction_factor`
2. GPU 端设 `target_stretch = 1 - a * contraction_factor`
3. GPU 端用 `f_total = a*f_L(λ) + f_PE(λ)` 调制约束刚度

精度可接受（vs OpenSim 误差 0.8%），但存在问题：
- 每步 CPU-GPU 通信（反求平衡点）
- 非能量驱动，target 是人工设定的
- 多肌肉竞争时无法自然平衡

---

## 1. DGF 能量函数分析

### 肌肉能量密度

从 PK1 应力（见 `docs/notes/dexterous_muscle_formulation.md`）：

```
P_m = f_hill(λ) / λ · F · (d⊗d)^T
f_hill(λ) = σ₀ · [a · f_L(λ̃) + f_PE(λ̃)]   (准静态)
```

能量是力的积分：

```
Ψ_muscle(λ) = σ₀ · ∫₁^λ [a · f_L(s) + f_PE(s)] ds
            = σ₀ · [a · Ψ_active(λ) + Ψ_passive(λ)]
```

### 被动能量 Ψ_passive：有闭式解 ✓

```
f_PE(s) = (exp(KPE·(s-1)/E0) - offset) / denom

Ψ_passive(λ) = ∫₁^λ f_PE(s) ds
             = (1/denom) · [(E0/KPE) · (exp(KPE·(λ-1)/E0) - 1) - offset · (λ - 1)]
```

KPE=4, E0=0.6, offset=exp(KPE·(0.2-1)/E0), denom=exp(KPE)-offset.

### 主动能量 Ψ_active：无闭式解 ✗

DGF f_L 前两项使用可变宽度分母：

```
g1 = b11 · exp(-0.5 · ((s - b21) / (b31 + b41·s))²)
g2 = b12 · exp(-0.5 · ((s - b22) / (b32 + b42·s))²)
g3 = b13 · exp(-0.5 · ((s - b23) / b33)²)           ← 仅此项有 erf 闭式解
```

令 u = (s - b21)/(b31 + b41·s)（Mobius 变换），逆变换 Jacobian 引入 1/(1 - b41·u)²，
积分变为 ∫ exp(-u²/2) / (1 - b41·u)² du，**无初等函数原函数**。

**结论**：DGF 能量需要数值积分（GPU 上 5-8 点 Gauss-Legendre）或查找表，不够优雅。

---

## 2. XPBD 能量-约束映射

### XPBD 势能结构

XPBD 约束产生的势能始终是二次形式：

```
E_xpbd = ½ · α⁻¹ · C²
```

其中 α = compliance, C = 约束函数。要从非二次能量 Ψ(x) 映射到 XPBD，标准做法是：

```
C(x) = √(2Ψ(x))     →    ½ · α⁻¹ · C² = α⁻¹ · Ψ(x)
α = 1/(V · dt²)      →    E_xpbd = V · Ψ(x)    ← 精确恢复
```

梯度：∇C = ∇Ψ / √(2Ψ) = ∇Ψ / C

参考：Macklin et al. 2021 "A Constraint-based Formulation of Stable Neo-Hookean Materials"，
本代码库 SNH 约束（`muscle_warp.py:1158-1313`）使用了同样的方法。

### 奇异性处理

λ ≈ 1 时 Ψ → 0，导致 C → 0, ∇C → ∞。Taylor 展开处理：

```
|λ - 1| < ε:
  Ψ ≈ ½ · σ₀ · f_hill(1) · (λ - 1)²
  C ≈ √(σ₀ · f_hill(1)) · |λ - 1|    （退化为线性弹簧）
```

现有代码已有类似处理模式（`tet_fiber_update_xpbd_fn` 第 418 行 `if psi > 1e-9`）。

---

## 3. 三种方案对比

### 方案 A：`C = √(2Ψ)` 精确能量约束 + DGF

```
C(λ) = √(2 · σ₀ · ∫₁^λ [a·f_L(s) + f_PE(s)] ds)
α = 1/(V · dt²)
```

| 优点 | 缺点 |
|------|------|
| 精确恢复能量 | DGF 需数值积分（GPU 开销） |
| 干净数学表述 | λ≈1 奇异性 |
| 与 VBD 方案理论一致 | 实现复杂 |

### 方案 B：动态目标力注入

```
C = λ - target_dynamic
target_dynamic = λ - σ₀ · [a·f_L(λ) + f_PE(λ)] / k_base
α = 1/(k_base · V · dt²)
```

等价于约束力 = k_base · (λ - target) = σ₀ · f_hill(λ)，精确匹配 Hill 力。

| 优点 | 缺点 |
|------|------|
| 消除外部平衡反求 | 非真正能量驱动 |
| 代码改动最小（~10 行） | 有效能量路径依赖 |
| 无需能量积分 | 多肌肉物理正确性存疑 |

### 方案 C：Millard 样条 + 精确能量约束 ← 推荐

Millard 2012 用三次 Hermite 样条定义 f_L 和 f_PE：

```
f_L(s) = a_i + b_i·(s-s_i) + c_i·(s-s_i)² + d_i·(s-s_i)³   (分段三次)

∫f_L ds = a_i·h + b_i·h²/2 + c_i·h³/3 + d_i·h⁴/4           (分段四次，闭式！)
```

再用方案 A 的 C = √(2Ψ) 得到精确能量约束。

| 优点 | 缺点 |
|------|------|
| 能量闭式，GPU 零额外开销 | 需新建 Millard 曲线模块 |
| 精确恢复势能 | 与 DGF 结果有 ~6° 差异 |
| OpenSim 默认肌肉模型 | 曲线参数需从文献提取 |
| GPU 友好（无 exp/log） | |
| 导数 C¹ 连续 | |

---

## 4. Millard 2012 模型规格

### 主动力-长度曲线 f_L

三次 Hermite 样条，关键节点（Millard et al. 2013）：

| λ̃ | f_L | df_L/dλ̃ |
|---|---|---|
| 0.4441 | 0.0 | 0.0 |
| 0.7331 | 0.7267 | (自动) |
| 1.0 | 1.0 | 0.0 |
| 1.1845 | 0.7714 | (自动) |
| 1.8123 | 0.0 | 0.0 |

### 被动力-长度曲线 f_PE

三次 Hermite 样条：
- λ̃ ≤ 1.0: f_PE = 0
- λ̃ = 1.0: f_PE = 0, slope = 0
- λ̃ = 1.0 + E0 (=1.6): f_PE = 1.0, slope = 1/E0

### XPBD 约束

```
Ψ_millard(λ) = σ₀ · [a · Ψ_L(λ) + Ψ_PE(λ)]     ← 四次多项式分段，闭式
C = √(2 · V · Ψ_millard(λ))
∇C = σ₀ · V · [a·f_L(λ) + f_PE(λ)] · ∇λ / C
α = 1/dt²
```

---

## 5. 实施计划

### Stage 1: Millard 曲线库
- 新建 `src/VMuscle/millard_curves.py`（NumPy + Warp @wp.func）
- 包含 f_L, f_PE, Ψ_L, Ψ_PE 四个函数
- 单元测试：与 DGF 曲线形状对比图

### Stage 2: Millard 能量 XPBD 约束
- 新约束类型 `TETFIBERMILLARD`
- `constraints.py` 新增 `create_tet_fiber_millard_constraint`
- `muscle_warp.py` 新增 `solve_tetfibermillard_kernel`
- 奇异性处理（λ≈1 Taylor 展开）

### Stage 3: 验证与对比
- sliding ball benchmark vs OpenSim Millard
- 与当前 TETFIBERDGF 对比
- 收敛速度、稳定性、力曲线精度

---

## 6. 关键文件参考

| 文件 | 用途 |
|------|------|
| `src/VMuscle/muscle_warp.py:957-1025` | 当前 DGF 约束内核（参考） |
| `src/VMuscle/muscle_warp.py:1158-1313` | SNH 能量约束（模式参考） |
| `src/VMuscle/muscle_warp.py:373-463` | tet_fiber_update_xpbd_fn（共享逻辑） |
| `src/VMuscle/dgf_curves.py` | DGF 曲线实现（对照） |
| `src/VMuscle/constraints.py:255-305` | DGF 约束构建器（模板） |
| `scripts/osim_simple_arm_millard.py` | OpenSim Millard 基准 |
| `docs/notes/dexterous_muscle_formulation.md` | PK1 应力推导 |
