# XPBD Millard2012 本构模型实现方案

## Context

当前 XPBD 框架使用 DGF (DeGrooteFregly) 本构，精度可接受（vs OpenSim ~0.8%），但 DGF 的三高斯拟合无法得到能量函数显式表达式。Millard2012 采用 quintic Bezier 样条，参数已知 → 可用符号推导得到精确多项式表达式，GPU 上高效求值。

### 关键发现

1. **OpenSim Millard 曲线 = quintic Bezier**（5次参数多项式），控制点由已知参数确定
2. **u-空间精确表达**：x(u), y(u) 是 5 次多项式，y(u)·x'(u) 是 9 次 → 积分 F(u)=∫y·dx 是 **10 次多项式，精确闭式**。用 SymPy 推导系数即可
3. **GPU 求值**：给定 x，Newton 迭代解 u(x)（3-5步，~50 FLOPs），再求 y(u) 或 F(u)（Horner ~10-20 FLOPs）。总计 ~70 FLOPs/tet，与 DGF 的 exp() 相当
4. **能量约束 C=sqrt(2Psi) 对主动力不可行**：Psi(lambda<1)<0。沿用 target-stretch + 刚度调制方案，被动 f_PE 能量留作后续
5. **初始化时计算**：控制点 → SymPy 推导 → 数值系数，一次计算，每次仿真复用

---

## Stage 1: Millard 曲线符号推导与模块

**新建文件**: `src/VMuscle/millard_curves.py`

### 1.1 SymPy 符号推导（初始化时运行一次）

对每段 quintic Bezier，给定控制点 Px[0..5], Py[0..5]：

```python
# 1. Bernstein → 幂基
x(u) = Σ Px[j]·B_j^5(u) = α₅u⁵ + α₄u⁴ + ... + α₀    # 5次
y(u) = Σ Py[j]·B_j^5(u) = β₅u⁵ + β₄u⁴ + ... + β₀    # 5次

# 2. 导数
x'(u) = 5α₅u⁴ + 4α₄u³ + ... + α₁                      # 4次

# 3. 被积函数
y(u)·x'(u) = 9次多项式                                    # 精确

# 4. 积分（能量函数）
F(u) = ∫₀ᵘ y(t)·x'(t) dt = 10次多项式                    # 精确闭式
```

用 SymPy 或 Mathematica 对已知 Millard 默认参数求解，得到每段的数值系数。

### 1.2 CPU 端：OpenSim 曲线构造

移植 OpenSim 的控制点计算算法：
- `quintic_bezier_corner_control_points()` — 移植自 `SegmentedQuinticBezierToolkit.cpp:142-230`
- `build_active_fl_curve()` — 5 段，参考 `SmoothSegmentedFunctionFactory.cpp:63-169`
  - 默认: x0=0.4441, x1=0.73, x2=1.0, x3=1.8123, ylow=0.1, slope=0.8616
- `build_passive_fpe_curve()` — 1-2 段，参考 `SmoothSegmentedFunctionFactory.cpp:582-692`
  - 默认: eZero=0.0, eIso=0.7, kLow=0.2, kIso=2.857, curviness=0.75

### 1.3 符号推导 → 数值系数

```python
class MillardSegment:
    """一段 quintic Bezier 的精确多项式表示."""
    x_coeffs: np.ndarray  # (6,) x(u) = Σ x_coeffs[k] * u^k
    y_coeffs: np.ndarray  # (6,) y(u)
    F_coeffs: np.ndarray  # (11,) F(u) = ∫₀ᵘ y·x' dt (能量积分)
    # x'(u) 从 x_coeffs 解析求导

class MillardCurve:
    """完整 Millard 曲线 = 多段 MillardSegment."""
    segments: list[MillardSegment]
    x_range: tuple[float, float]
    y_boundary: tuple[float, float, float, float]  # 外推用
```

初始化流程:
1. 调用 OpenSim 控制点算法 → 得到 Bezier 控制点
2. 用 SymPy 将 Bernstein 基展开为幂基 → `x_coeffs`, `y_coeffs`
3. 用 SymPy 求 y(u)·x'(u) 的不定积分 → `F_coeffs`
4. 数值系数存入 `MillardSegment`

### 1.4 GPU 端：Warp 函数

```python
@wp.func
def millard_find_segment_and_u(
    x: float,
    # 每段的 x(u) 系数，通过 wp.array 传入
    seg_x_coeffs: wp.array(dtype=wp.float32),  # (n_seg * 6,)
    n_segments: int,
) -> wp.vec2i:  # (segment_index, u)
    """给定 x，确定所在段并 Newton 迭代求 u.
    
    Newton: u_{n+1} = u_n - (x(u_n) - x_target) / x'(u_n)
    收敛: 3-5 步, ~50 FLOPs
    """

@wp.func
def millard_eval_y(u: float, seg_y_coeffs_offset: int,
                    y_coeffs: wp.array(dtype=wp.float32)) -> float:
    """Horner 求值 y(u), degree 5, ~10 FLOPs."""

@wp.func
def millard_eval_F(u: float, seg_F_coeffs_offset: int,
                    F_coeffs: wp.array(dtype=wp.float32)) -> float:
    """Horner 求值 F(u) = ∫y·dx, degree 10, ~20 FLOPs."""
```

GPU 每 tet 求值流程:
1. 线性扫描确定 lm_tilde 所在 Bezier 段 (~5 比较)
2. Newton 迭代 x→u (3-5 步, ~50 FLOPs)
3. Horner 求 f_L(u) 或 f_PE(u) (~10 FLOPs)
4. (可选) Horner 求能量 F(u) (~20 FLOPs)

**总计 ~70 FLOPs/tet**，与 DGF 的 3×exp() 相当，但精度为机器精度。

### 1.5 验证

- `tests/test_millard_curves.py`:
  - Bezier 控制点 vs OpenSim 参考值
  - SymPy 推导 vs 数值积分一致性
  - f_L, f_PE 曲线值 vs OpenSim 输出
- 输出曲线对比图到 `output/millard_curves.png`

---

## Stage 2: XPBD 约束 (TETFIBERMILLARD)

### 2.1 类型常量 & 约束构建器

**修改文件**: `src/VMuscle/constraints.py`

```python
TETFIBERMILLARD = -503662411  # 新类型常量

def create_tet_fiber_millard_constraint(self, params):
    # 几何逻辑与 create_tet_fiber_dgf_constraint (L255-305) 完全相同
    # type = TETFIBERMILLARD
    # restdir = [sigma0, contraction_factor, 0.0]
```

在 `_collect_raw_constraints` 中添加 `'fibermillard'` 分支。

### 2.2 Warp 内核

**修改文件**: `src/VMuscle/muscle_warp.py`

```python
@wp.kernel
def solve_tetfibermillard_kernel(
    # 与 DGF 内核相同的基础参数 +
    # Millard 曲线系数数组 (wp.array)
    fl_seg_x_coeffs, fl_seg_y_coeffs, fl_n_segments,
    fpe_seg_x_coeffs, fpe_seg_y_coeffs, fpe_n_segments,
    fl_x_lo, fl_x_hi, fl_y_lo, fl_y_hi, fl_dydx_lo, fl_dydx_hi,
    fpe_x_lo, fpe_x_hi, fpe_y_lo, fpe_y_hi, fpe_dydx_lo, fpe_dydx_hi,
):
    # 结构与 solve_tetfiberdgf_kernel (L957-1025) 相同:
    # 1. 读约束数据，计算变形梯度，得 lm_tilde
    # 2. f_L = millard_find_segment_and_u + millard_eval_y  <- 替换 DGF
    # 3. f_PE = 同上                                         <- 替换 DGF
    # 4. f_total = acti * f_L + f_PE
    # 5. stiffness = base * max(f_total, 0.01) * fiber_stiffness_scale
    # 6. target_stretch = 1 - acti * contraction_factor
    # 7. 调用 tet_fiber_update_xpbd_fn(...)  <- 复用现有
```

### 2.3 曲线数组初始化

在 `MuscleSim.build_constraints()` 中：
```python
if TETFIBERMILLARD in self.cons_ranges:
    from VMuscle.millard_curves import MillardCurves
    mc = MillardCurves()  # 内部调用 SymPy 推导（首次）或加载缓存
    # 上传系数到 GPU
    self.millard_fl_x_coeffs = wp.from_numpy(mc.fl.x_coeffs_flat, ...)
    self.millard_fl_y_coeffs = wp.from_numpy(mc.fl.y_coeffs_flat, ...)
    # ... fpe 同理
```

### 2.4 Dispatch & 约束更新

- `_dispatch_constraints`: 添加 TETFIBERMILLARD 分支
- 平衡点反求: `millard_equilibrium_fiber_length(a, load, fl_curve)` — CPU 端，用精确 y(u) 求逆
- 复用 `update_cons_restdir1_kernel` 每步更新 contraction_factor

---

## Stage 3: Sliding Ball 验证

**新建文件**:
- `data/slidingBall/config_xpbd_millard.json`
- `examples/example_xpbd_millard_sliding_ball.py`（模板: `example_xpbd_dgf_sliding_ball.py`）

验证内容:
| 指标 | XPBD-Millard | OpenSim-Millard | XPBD-DGF |
|------|-------------|-----------------|----------|
| lambda_eq | ? | ~0.59 | 0.5945 |
| 球位置 | ? | ~0.04m | 0.0405m |
| 稳态力 | ? | ~0.26 | ~0.26 |

输出:
- `output/millard_sliding_ball_result.png`
- `output/millard_vs_dgf_curves.png`（f_L, f_PE, 能量积分对比）

---

## Stage 4: Simple Arm 验证

将 `examples/example_xpbd_coupled_simple_arm.py` 中的 DGF 约束替换为 Millard：
- 配置: `data/simpleArm/config.json` 添加 `fibermillard` 约束选项
- 验证: 肘关节角度、力范围、稳定性（无 NaN、无 inverted tet）
- 对比 DGF 结果（1000 步, 10s）

输出:
- `output/millard_simple_arm_result.png`
- 肘角、力的时间序列对比图

---

## 关键文件清单

| 操作 | 文件 |
|------|------|
| 新建 | `src/VMuscle/millard_curves.py` |
| 新建 | `tests/test_millard_curves.py` |
| 新建 | `examples/example_xpbd_millard_sliding_ball.py` |
| 新建 | `data/slidingBall/config_xpbd_millard.json` |
| 修改 | `src/VMuscle/constraints.py` — TETFIBERMILLARD 类型 + builder |
| 修改 | `src/VMuscle/muscle_warp.py` — 内核 + dispatch + 曲线初始化 |
| 修改 | `examples/example_xpbd_coupled_simple_arm.py` — Millard 选项 |
| 修改 | `data/simpleArm/config.json` — Millard 配置 |

## 验证方法

1. `uv run python -m pytest tests/test_millard_curves.py` — 曲线精度 & 符号推导
2. `RUN=example_xpbd_millard_sliding_ball uv run main.py` — sliding ball
3. `RUN=example_xpbd_coupled_simple_arm uv run main.py` — simple arm（切换 Millard）
4. 对比 OpenSim Millard + DGF 基准数据
