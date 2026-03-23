# Dexterous Manipulation 论文肌肉力学公式分析

> 来源: Lee et al., "Dexterous Manipulation and Control with Volumetric Muscles", SIGGRAPH 2018

## 一、Dexterous 如何让 PD 配准 Hill-type Muscle

### 1. 核心问题：PD 不能处理任意非线性能量

Projective Dynamics 的核心是将隐式时间积分拆成两步交替：

```
Global Step:  解一个固定的线性系统 (L x = b)    <- 矩阵 L 预分解，每步复用
Local Step:   对每个约束，投影到约束集的最近点     <- 闭式解
```

**限制**：只有能写成 "形变梯度 F 到目标 F* 的距离平方" 形式的能量才能高效处理：

```
E_constraint = w/2 * ||F - F*||^2_F
```

而 Hill-type 肌肉的 f_L(lambda)、f_PE(lambda) 是高度非线性的（三高斯、指数），f_V(v) 还依赖速度——这些都不能直接写成上面的形式。

### 2. Dexterous 的解决方案：分三部分处理

#### Part A：各向同性基质 -> 标准 ARAP 约束

肌肉的基础组织（结缔组织、细胞外基质）用 Neo-Hookean / ARAP 建模：

```
E_iso = mu/2 * ||F - R||^2_F + kappa/2 * (J - 1)^2
```

其中 R 是 F 的最近旋转矩阵（极分解）。这是 PD 的标准约束，Local Step 就是求极分解 R = polar(F)。

#### Part B：纤维被动力 + 主动力 -> "Active Rest Length" 投影

这是 Dexterous 最核心的 trick。对于纤维方向 d0：

**被动弹性 (f_PE)**：纤维拉伸时产生回弹力。在 PD 中建模为纤维方向弹簧约束：

```
E_fiber = k_fiber/2 * (lambda - lambda_rest)^2
```

**主动收缩 (a * f_L)**：不是加一个力，而是改变静息长度：

```
lambda_rest(a) = 1.0 - a * Delta_contract
```

当 activation a 从 0 增加到 1 时，lambda_rest 减小，纤维的"目标长度"缩短，PD 自然驱动纤维收缩。

**与 Hill f_L 的关系**：

```
真实 Hill:     F_active = F0 * a * f_L(lambda)             <- 非线性，依赖当前 lambda
Dexterous PD:  F_active ~ k_fiber * (lambda - lambda_rest(a))  <- 线性弹簧近似
```

Dexterous 并没有精确嵌入 f_L 曲线，而是用线性弹簧近似了 f_L 在 lambda~1 附近的行为。f_L 的非线性形状（三高斯的宽度和不对称性）被丢失。

#### Part C：力-速度关系 (F-V) -> Rayleigh 阻尼

Hill F-V 曲线是速度依赖的耗散关系。Dexterous 的处理：**完全放弃 f_V 曲线**，用 Rayleigh 阻尼替代：

```
f_damp = k_d * H_elastic * v
```

其中 k_d 标定为匹配 f_V 在 v=0 处的斜率：

```
k_d = beta * sigma_0 / (V_max * l_opt)
```

Rayleigh 阻尼在 |v/V_max| < 0.3 区域与 Hill F-V 接近，但在高速区偏差很大。

### 3. 对比总结

| 方面 | Hill-type (OpenSim) | Dexterous (PD) | 我们的方案 (VBD) |
|------|-------------------|----------------|-----------------|
| f_L 曲线 | DGF 三高斯，精确 | 线性弹簧近似 | DGF 三高斯，精确 |
| f_PE 曲线 | DGF 指数，精确 | 简化为弹簧 | DGF 指数，精确 |
| f_V 曲线 | 对数型，非线性 | Rayleigh 线性阻尼 | Rayleigh 阻尼（后续可升级） |
| 求解器 | ODE | PD（固定线性系统） | VBD（逐顶点 Newton） |
| 能量约束 | N/A | 仅限 PD 可投影形式 | 任意可微能量 |

---

## 二、公式 7 详解：纤维 PK1 应力推导

### 公式

$$
\mathbf{P}_m := \frac{\partial \Psi_m}{\partial \mathbf{F}} = \frac{\partial \Psi_m}{\partial l} \cdot \frac{\partial l}{\partial \mathbf{F}} = f_{hill}(l, \dot{l}, a) \cdot \frac{1}{l} \mathbf{F} \mathbf{d}\mathbf{d}^T \quad (7)
$$

### 变量定义

| 符号 | 含义 |
|------|------|
| **F** | 形变梯度 (3x3)，描述 rest -> world 的局部变形 |
| **d** | 纤维方向，rest 构型中的单位向量 (3x1) |
| l = \|\|Fd\|\| | fiber stretch factor — 纤维在世界空间中的当前拉伸长度 |
| Psi_m(l) | 纤维应变能密度，仅依赖 l |
| f_hill(l, l_dot, a) | Hill 力曲线，= dPsi_m/dl |
| P_m | 纤维的 1st Piola-Kirchhoff 应力 (3x3) |

### 链式法则推导

目标是求 dPsi_m/dF（PK1 应力），Psi_m 通过标量 l 间接依赖 F，所以用链式法则：

```
dPsi_m/dF = (dPsi_m/dl) * (dl/dF)
              ----------    --------
                因子 1        因子 2
```

**因子 1**：dPsi_m/dl = f_hill(l, l_dot, a)

论文的关键洞察：**我们永远不需要写出 Psi_m(l) 的显式表达式**。FEM 仿真只需要梯度和 Hessian，而 Psi_m 的导数就是 Hill 力曲线——这个是已知的：

```
f_hill(l, l_dot, a) = sigma_0 * [a * f_L(l_tilde) * f_V(v_tilde) + f_PE(l_tilde)]
```

**因子 2**：dl/dF，其中 l = ||Fd||

```
l = ||Fd|| = sqrt(d^T F^T F d)

设 w = Fd (3x1 向量)，则 l = ||w||

dl/dF_{ij} = (1/||w||) * w_i * d_j     (标准向量范数对矩阵的导数)
```

写成矩阵形式：

```
dl/dF = (1/l) * (Fd) * d^T = (1/l) * F * (d d^T)
```

**合并**：

```
P_m = f_hill * (1/l) * F * (d d^T)                              ...(7)
```

---

## 三、为什么 dd^T 是纤维方向的投影矩阵

### 具体例子

设 d = (1, 0, 0)^T（纤维沿 x 轴）：

```
        [1]                     [1 0 0]
dd^T =  [0] * [1 0 0]   =      [0 0 0]
        [0]                     [0 0 0]
```

对任意向量 v = (v1, v2, v3)^T 作用：

```
          [1 0 0] [v1]     [v1]
dd^T v =  [0 0 0] [v2]  =  [0 ]   <- 只保留了 v 在 d 方向的分量
          [0 0 0] [v3]     [0 ]
```

### 一般情况的代数证明

对任意单位向量 d 和任意向量 v：

```
(dd^T) v = d (d^T v) = (d . v) * d
           ---------   -------   ---
           矩阵乘法    标量投影   方向
```

拆成两步：
1. d^T v = d . v：v 在 d 方向上的标量投影（一个数）
2. 乘以 d：把这个标量放回 d 方向

结果就是 v 在 d 方向上的正交投影向量。

### 验证投影矩阵的性质

投影矩阵 P 必须满足 P^2 = P（投影两次 = 投影一次）：

```
(dd^T)(dd^T) = d (d^T d) d^T = d * 1 * d^T = dd^T   (since ||d||^2 = 1)
```

---

## 四、为什么 F dd^T 提取 F 沿纤维方向的作用

### 分解 F 的作用

恒等矩阵可以分解为两个正交投影之和：

```
I = dd^T + (I - dd^T)
    -----   ----------
  纤维方向   纤维垂直面
```

所以形变梯度 F 也可以分解：

```
F = F * I = F * dd^T  +  F * (I - dd^T)
            --------     ---------------
           沿纤维的变形   垂直于纤维的变形
```

**F dd^T 就是 F 中只与纤维方向相关的那部分。**

### 具体例子

设 d = (1, 0, 0)^T，F 是一个一般的形变梯度：

```
          [F11 F12 F13]   [1 0 0]     [F11 0 0]
F * dd^T= [F21 F22 F23] * [0 0 0]  =  [F21 0 0]
          [F31 F32 F33]   [0 0 0]     [F31 0 0]
                                       ---------
                                       只剩 F 的第 1 列！
```

F 的第 1 列 = F * e1 = F * d = Fd，就是 rest 构型中沿纤维方向的单位向量经过变形后变成的向量。

### 对任意向量的作用

```
(F dd^T) v = F * d * (d^T v) = (d^T v) * Fd
                                -------   --
                               v在d方向的  d经F变形后
                               标量分量    的世界空间向量
```

含义：
1. 取 v 沿纤维的分量 -> 标量 (d^T v)
2. 乘以 Fd（纤维在世界空间的拉伸向量）
3. 结果：只有沿纤维输入的部分才会产生沿 Fd 方向的输出

---

## 五、公式 7 的物理含义总结

```
P_m = f_hill/l * F dd^T
```

- **F dd^T**：形变梯度中仅与纤维方向相关的部分（3x3 矩阵）
- **/l**：归一化（l = ||Fd||，去掉拉伸长度的量纲）
- **f_hill**：Hill 力曲线给出的标量力值

P_m 是一个**只沿纤维方向施加应力的 3x3 张量**，大小由 Hill 力曲线决定，方向由当前纤维的变形状态决定。对垂直于纤维的方向，贡献为零。

---

## 六、与我们 VBD 方案的等价性

我们计划中用 I4 = l^2 作为参数化变量：

```
我们:       P = 2 Psi'(I4) * F(d x d)
Dexterous:  P = f_hill/l * F(d x d)
```

验证等价：I4 = l^2，所以 dI4 = 2l*dl

```
Psi'(I4) = dPsi/dI4 = (dPsi/dl) * (dl/dI4) = f_hill * 1/(2l)

2 Psi'(I4) = f_hill / l    完全一致
```

唯一区别是参数化选择：Dexterous 用 l（物理直觉更直接），我们用 I4 = l^2（求导更简洁，Hessian 公式更干净，连续力学文献的标准做法）。

### VBD 的优势

论文紧接公式 7 后承认 PD 不能直接处理 f_hill 的完整非线性：

> "It is well understood that the Projective Dynamics formulation... supports a specific and somewhat narrow scope of materials"

VBD 可以直接使用公式 7 的完整非线性形式（包括 DGF 三高斯 f_L 和指数 f_PE），不需要任何线性化或投影近似。
