# XPBD 中的被动能量与主动纤维力

## 1. 建模原则

这里把组织模型拆成两部分：

```text
passive part:  由势能定义，可放进 XPBD constraint
active part:   由 Hill / Millard 型主动力给出，通常不当作 constraint
```

推荐写成

```text
Psi_passive = Psi_vol(J) + Psi_iso(I1_bar or I1_bar,I2_bar) + Psi_fiber_passive(lambda_f or I4_bar)
```

其中

```text
J        = det(F)
C        = F^T F
I1       = tr(C)
I2       = 0.5 * (I1^2 - tr(C^2))
I4       = d · C d = ||F d||^2
lambda_f = ||F d||
```

如果采用体积-偏差分裂，则常用

```text
C_bar      = J^(-2/3) C
I1_bar     = tr(C_bar)
I2_bar     = 0.5 * (I1_bar^2 - tr(C_bar^2))
I4_bar     = d · C_bar d
lambda_f_bar = sqrt(I4_bar)
```

核心原则：

```text
1. 被动项走 XPBD
2. 主动项直接作为附加力
```

## 2. Passive Part 如何设计 C

对每个单元 `e`，先定义被动单元能量

```text
U_e_passive = V0 * [ psi_vol(J) + psi_iso(...) + psi_fiber_passive(...) ]
```

在 XPBD 中，不建议把所有东西都揉成一个总 constraint。更稳妥的是分项：

```text
C_vol = J - 1
or
C_vol = sqrt(2 * V0 * psi_vol(J))

C_iso = sqrt(2 * V0 * psi_iso(...))

C_fib = sqrt(2 * V0 * psi_fiber_passive(...))
or, for a simple quadratic fiber penalty,
C_fib = lambda_f - lambda_rest
```

这样做的好处是：

```text
1. 体积、基质、纤维三部分更容易调
2. 数值条件通常更好
3. 不会把主动和被动混在一起
```

XPBD 更新仍然是标准形式：

```text
alpha_tilde = alpha / dt^2

delta_lambda =
    -( C + alpha_tilde * lambda_c )
    / ( sum_i w_i * ||grad_xi C||^2 + alpha_tilde )

delta_x_i = w_i * grad_xi C * delta_lambda
lambda_c += delta_lambda
```

这里只需要为每个 passive constraint 提供 `grad_xi C`。

## 3. 纤维方向上的关键导数

设

```text
g = F d
lambda_f = ||g||
n = g / max(lambda_f, eps_fiber)
```

则

```text
dlambda_f/dF = n ⊗ d
dI4/dF       = 2 * (F d) ⊗ d
```

若使用线性四面体或三角形，则

```text
grad_xi U_e = V0 * P * gradN_i^0
```

其中 `P = dPsi/dF`。

特别地，对于纤维方向上的标量应力 `tau_f(lambda_f)`：

```text
P_fiber = tau_f(lambda_f) * (n ⊗ d)
grad_xi U_e = V0 * tau_f(lambda_f) * n * (d · gradN_i^0)
```

## 4. 最重要的一点：fiber stretch ratio 的分母必须先对齐

这是把连续体模型和已有 Hill / Millard 曲线接起来时最容易出错的地方。

连续体里通常直接算

```text
lambda_f = ||F d|| = l_f / L_ref
```

但 Hill / Millard 曲线一般不是直接用这个量，而是某种归一化长度，例如

```text
r = l_f / l_opt
```

或者

```text
r = l_f / l_slack
```

或者其他文献定义的 stretch ratio。

所以在使用已有 `force-fiber stretch ratio` 曲线之前，必须先明确：

```text
1. 曲线里的 ratio 分母到底是什么
2. 模拟里的 lambda_f 相对于什么长度归一化
3. 二者之间如何转换
```

如果曲线是以最优长度 `l_opt` 为分母，而连续体里

```text
lambda_f = l_f / L_ref
```

那么必须写成

```text
r = lambda_f / lambda_opt
lambda_opt = l_opt / L_ref
```

即

```text
l_f / l_opt = (l_f / L_ref) / (l_opt / L_ref)
```

这一步必须先做对，否则：

```text
1. 峰值位置会错
2. 斜率会错
3. 主动力大小也会错
```

如果你的曲线是实验拟合出来的多项式

```text
f_FL(r)
```

那它的输入必须是对齐后的 `r`，而不是直接把 `lambda_f` 硬塞进去。

## 5. Active Force 为什么不必走 Constraint

主动纤维力通常来自 Hill / Millard 型关系，例如

```text
f_act = f_act(r, activation, velocity, history, ...)
```

这里面经常包含：

```text
1. activation 的时变
2. force-length
3. force-velocity
4. 其他历史依赖
```

这类主动力一般不是严格保守力，因此没有必要硬写成

```text
C(x) = sqrt(2U(x))
```

更直接也更清楚的做法是：

```text
把 active force 当作附加内力或附加外力
```

也就是只保留它的力学方向，而不把它包装成 constraint。

对单元纤维应力 `tau_act`，可写成

```text
P_act = tau_act * (n ⊗ d)
```

于是节点力为

```text
f_i_act = -V0 * P_act * gradN_i^0
        = -V0 * tau_act * n * (d · gradN_i^0)
```

这就是最直接的主动纤维力离散形式。

重点：

```text
active force 没必要走 constraint
```

因为即使它不是保守力，

```text
只要把 f_act 放进时间积分，
它一样会做功并驱动网格产生变形。
```

也就是说，网格变形并不要求每一项都来自 constraint；对 active part，时间积分本身就足够把力转成位移和速度变化。

补充一句：

```text
如果当前子步冻结 activation，且 active term 只依赖 lambda_f，
那么它可以瞬时保守化并改写成 energy/constraint；
但对已有 Hill / Millard force curve，直接当附加力通常更自然。
```

## 6. 如何把已有 Hill / Millard 曲线接进来

假设你已有一条一维曲线

```text
f_FL(r)
```

以及 activation `act`。

最简单的主动项写法是

```text
tau_act = sigma0 * act * f_FL(r)
```

如果还有 velocity 项，则

```text
tau_act = sigma0 * act * f_FL(r) * f_V(v_hat)
```

这里：

```text
sigma0 = 主动力标定尺度
```

如果你手里的曲线输出的是“力”而不是“应力”，需要先统一尺度，例如除以参考截面积，或者等价地把面积吸收到标定常数里。

然后按上一节直接形成单元节点力：

```text
P_act   = tau_act * (n ⊗ d)
f_i_act = -V0 * P_act * gradN_i^0
```

这里不需要设计额外的 `C_act`。

## 7. 推荐的整体流程

推荐把每个时间步分成两条线：

```text
1. active force accumulation
2. passive XPBD projection
```

具体来说：

```text
active:
  用当前配置算 lambda_f
  把 lambda_f 转成曲线真正需要的 ratio r
  从 Hill / Millard 曲线得到 tau_act
  组装 f_act

passive:
  用 XPBD 投影体积项、各向同性项、被动纤维项
```

这两条线不要混淆。

按这个分裂做时，active term 数值上通常是显式或滞后显式处理；这只是实现选择，不是力形式本身的硬限制。

## 8. 简要伪代码

```cpp
for each time step:

    // 1. update activation / muscle state
    act_n = update_activation(...)

    // 2. accumulate active fiber force on predicted state
    clear f_act
    for each element e:
        F = Ds(x) * invDm
        g = F * d
        lambda_f = norm(g)
        n = g / max(lambda_f, eps_fiber)

        // IMPORTANT: align denominator before evaluating the curve
        // Example: r = l_f / l_opt = lambda_f / lambda_opt
        r = convert_lambda_f_to_curve_ratio(lambda_f)

        tau_act = sigma0 * act_n * f_FL(r)
        // or tau_act = sigma0 * act_n * f_FL(r) * f_V(v_hat)

        for i in element nodes:
            f_act[i] += -V0 * tau_act * n * dot(d, gradN0[i])

    // 3. time integration with active force
    // In this split form, the active term is treated explicitly / lagged explicitly.
    v += dt * M^{-1} * (f_ext + f_act + other_forces)
    x_pred = x + dt * v

    // 4. XPBD solve for passive constraints only
    for iter = 1..num_iters:
        project volume constraints
        project isotropic passive constraints
        project passive fiber constraints

    // 5. finalize state
    v = (x_pred - x) / dt
    x = x_pred
```

## 9. 一句话总结

这个问题最清楚的处理方式是：

```text
passive part -> XPBD constraints
active part  -> direct additional force
```

在默认分裂实现下，active term 通常按显式或滞后显式处理。

而在把现有 Hill / Millard 曲线接进来之前，

```text
必须先把 fiber stretch ratio 的分母定义对齐，
否则曲线输入就是错的。
```
