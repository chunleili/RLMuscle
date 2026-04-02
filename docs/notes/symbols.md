# 符号表 / Symbol Table

本项目所有文档和代码统一使用以下符号。新增符号应先在此表登记。

## 肌肉本构

| 符号 | 代码变量 | 含义 | 单位 | 备注 |
|------|---------|------|------|------|
| lm | `lm_tilde`, `lm` | 归一化纤维拉伸（当前长度 / 最优长度） | 无量纲 | = ‖F·d‖；**不用 λ**，避免与 Lamé 参数和 XPBD 乘子冲突 |
| lm_eq | `lm_eq` | 平衡纤维拉伸 | 无量纲 | bisection 反求 |
| lm_min | — | f_L 左肩（Millard: 0.4441） | 无量纲 | 能量参考点 |
| lm_opt | — | 最优纤维长度对应的拉伸（= 1.0） | 无量纲 | f_L 峰值处 |
| f_L(lm) | `f_L_val` | 主动力-长度曲线 | 无量纲 | Millard: 5 段 quintic Bezier；DGF: 3 高斯 |
| f_PE(lm) | `f_PE_val` | 被动力-长度曲线 | 无量纲 | Millard: 2 段 quintic Bezier；DGF: 指数 |
| f_V(v̇) | `fv` | 力-速度曲线 | 无量纲 | 当前未启用，用 Rayleigh 阻尼替代 |
| f_total | `f_total` | 归一化总力 = a·f_L + f_PE | 无量纲 | |
| a | `acti`, `activation` | 肌肉激活水平 | [0, 1] | 由 excitation 经一阶滤波得到 |
| σ₀ | `sigma0` | 峰值等距应力 | Pa | 默认 300,000 Pa；存于 `restdir[0]` |
| Ψ_L(lm) | `psi_L` | 主动能量密度 = σ₀·a·∫_{lm_min}^{lm} f_L(lm') dlm' | J/m³ | 闭式10次多项式，非数值积分 |
| Ψ_PE(lm) | `psi_pe` | 被动能量密度 | J/m³ | 同上 |
| F_max | `F_max` | 峰值等距力 = σ₀ · A_cross | N | |

## 连续力学

| 符号 | 代码变量 | 含义 | 单位 | 备注 |
|------|---------|------|------|------|
| F | `_Ds @ Dminv` | 变形梯度 | 3×3 | F = D_s · D_m⁻¹ |
| D_s | `_Ds` | 当前边矩阵 | 3×3 | |
| D_m⁻¹ | `rest_matrix` | 静止边矩阵的逆 | 3×3 | 预计算 |
| d | `fiber_dir` | 材料空间纤维方向 | 单位向量 | 存于 `restvector[0:3]` |
| J | `J` | det(F)，体积比 | 无量纲 | |
| R | `R` | 极分解旋转 | 3×3 正交 | F = R·S |
| P_m | — | 纤维 PK1 应力 = f_hill/lm · F·(dd^T) | Pa | 见 dexterous 公式笔记 |
| μ | `mu` | 剪切模量（Neo-Hookean） | Pa | |
| λ_Lamé | `lam` | Lamé 第二参数（体积模量） | Pa | 代码中用 `lam`，**不缩写为 λ** |

## XPBD 求解器

| 符号 | 代码变量 | 含义 | 单位 | 备注 |
|------|---------|------|------|------|
| C | `C`, `C_act` | 约束违反量 | 视约束类型 | fiber: C=√(2Ψ)；volume: C=J-α |
| ∇C_i | `grad0`..`grad3` | 约束梯度（对顶点位置） | 3D 向量 | |
| Δλ_xpbd | `dlambda` | Lagrange 乘子增量 | — | **不用 Δλ**，避免与纤维拉伸混淆 |
| λ_xpbd | `cons[c].L` | 累积 Lagrange 乘子 | — | 同上 |
| α | `alpha` | compliance = 1/(k·dt²) | — | |
| γ | `gamma` | 阻尼因子 | — | |
| w_sum | `wsum` | 质量加权梯度范数平方和 | — | = Σ m_i⁻¹ ‖∇C_i‖² |
| m_i⁻¹ | `inv_mass0`.. | 顶点逆质量 | kg⁻¹ | 固定点为 0 |
| dt | `dt` | 时间步长 | s | |

## Bezier 曲线（Millard）

| 符号 | 代码变量 | 含义 | 备注 |
|------|---------|------|------|
| u | `u` | Bezier 参数 | [0, 1]，Newton 迭代求解 x(u)=lm |
| x_coeffs | `x_coeffs` | x(u) 幂基系数 | 6项（5次） |
| y_coeffs | `y_coeffs` | y(u) 幂基系数 | 6项（5次） |
| e_coeffs | `F_coeffs`, `e_coeffs` | Ψ(u) 闭式能量多项式系数 | 11项（10次） |
| seg_bounds | `seg_bounds` | 段边界 x 值 | n_seg+1 个 |

## 约束数据结构字段

| 字段 | 含义 |
|------|------|
| `restdir[0]` | σ₀（fiber 约束）或 λ_Lamé（SNH 约束） |
| `restdir[1]` | contraction_factor（当前）→ Stage 5 后废弃 |
| `restvector[0:3]` | 纤维方向 d（fiber）或 四元数 q（ARAP）或目标位置（attach） |
| `restlength` | tet 体积（fiber/volume）或目标距离（distance） |
| `pts[0:4]` | 顶点索引，-1 表示不使用 |
| `tetid` | tet 元素 ID |
| `L[0:4]` | 累积 Lagrange 乘子（按约束子类型分量） |
