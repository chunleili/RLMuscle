# couple3 post-smooth 网格质量改善

> 关联计划: [couple3 plan](../../.claude/plans/woolly-launching-lampson.md)
> 关联文档:
> - [couple3 mesh 稳定性](2026-04-03-couple3-mesh-stability.md) — 问题根因
> - [couple3 初始实现](2026-04-03-example-couple3.md) — couple3 设计
> - [方法对比](2026-04-03-couple3-approach-comparison.md) — 最终选用 TETFIBERMILLARD
> 时间: 2026-04-03
> 状态: 成功

## 问题

example_couple3 使用 TETFIBERMILLARD 能量约束驱动 bicep 收缩，在 activation=1.0 时产生大量网格畸变（~530 个反转 tets，13.5%），且去激活后无法恢复（~18 个 tets 永久反转）。

## 解决方案: Post-Smoothing (GS+Jacobi)

参照 Houdini 的 smoothing iteration 思路，在每个 substep 的约束求解后，额外运行若干轮只包含**非 fiber 约束**（volume、ARAP、attach）的平滑迭代。

核心思想：fiber 约束驱动收缩导致局部大变形，而 volume/ARAP/attach 约束负责维持网格质量。Post-smooth 给予这些约束额外的迭代来修复被 fiber 力扰乱的网格。

### 实现

**`muscle_warp.py` — `MuscleSim.post_smooth()`**:
```python
_FIBER_TYPES = frozenset([TETFIBERNORM, TETFIBERDGF, TETFIBERMILLARD])

def post_smooth(self):
    n_iters = int(getattr(self.cfg, 'post_smooth_iters', 0))
    if n_iters <= 0 or self.n_cons == 0:
        return
    if not hasattr(self, '_smooth_ranges'):
        self._smooth_ranges = {
            k: v for k, v in self.cons_ranges.items()
            if k not in self._FIBER_TYPES
        }
    for _ in range(n_iters):
        self.clear()
        self._dispatch_constraints(self._smooth_ranges)
        if self.use_jacobi:
            self.apply_dP()
```

**调用位置**（`muscle_common.py` 和 `solver_muscle_bone_coupled_warp.py`）:
```python
self.solve_constraints()
if self.use_jacobi:
    self.apply_dP()
self.post_smooth()          # <-- 新增
self.repair_inverted_tets()
self.update_velocities()
```

**Config**: `"post_smooth_iters": 3`（`bicep_fibermillard_coupled.json`）

### 向后兼容

- `post_smooth_iters` 默认 0，不影响现有 config（如 couple2 的 `bicep.json`）
- `sigma0=0` 时 fiber force 本就不启用，不影响纯 TETFIBERNORM 路径

## 测试结果

### 复现命令
```bash
uv run python examples/example_couple3.py --auto --steps 300 --no-usd
```

### 300 步完整运行数据

| 阶段 | Steps | Activation | 反转 Tets | Worst det(F) | Torque (N·m) |
|------|-------|:---:|:---:|:---:|:---:|
| 静止 | 1-60 | 0 | 0 | — | 0 |
| 激活上升 (exc=0.5) | 61-90 | 0.5 | 5→32 | -4.12 | 0→2.56 |
| 全激活 (exc=1.0) | 91-150 | 1.0 | 32→55 | -5.09 (tet 3705) | 2.56→6.50 (clamp) |
| 去激活 (exc=0.7→0.3) | 151-240 | 0.7→0.3 | 55→60 (peak) | -4.25 | 6.50→0.02 |
| 完全去激活 | 241-260 | 0 | 60→51 | -4.16 | 1.22→0 |
| 恢复 | 261-274 | 0 | 36→0 | 0.0096 | 0 |
| 次级振荡 | 275-289 | 0 | 1→5→2 | -1.11 (tet 3846) | 0 |
| 最终静止 | 290-300 | 0 | 0 | 0.003 | 0 |

### 方法对比

| 配置 | Peak 反转 Tets | Worst det(F) | 去激活恢复 |
|------|:---:|:---:|:---:|
| **Baseline（无修复）** | ~480 (12.2%) | -927 | ~19（永久） |
| couple2 (TETFIBERNORM) | ~15 (0.4%) | -3.9 | 0-1 |
| Old centroid repair | ~360 (9.1%) | -0.002 | ~350（trap!） |
| SVD Jacobi repair | ~260 (6.6%) | varies | ~260（trap!） |
| Post-smooth (3 iters) | ~62 (1.5%) | -5.09 | 0-1 |
| Post-smooth (5 iters) | ~10 (0.25%) | -2.48 | 1 |
| **smooth=5 + SVD(a=0.1, σ=0.05, i=1)** | **9 (0.23%)** | **-0.54** | **0-1** |

### 最终方案（默认配置）
`post_smooth_iters=5 + repair_alpha=0.1, repair_sigma_min=0.05, repair_iters=1`
- 反转 tets 峰值: 9 (0.23%)，从 480 降低 **98%**
- Worst det(F): -0.54（vs -927 baseline）
- 完全恢复：去激活后 0-1 个反转 tets
- 性能开销: +224%（全部来自 smooth=5，SVD i=1 零额外开销）
- 峰值 torque 6.5 N·m，joint angle -0.92 rad
- 详细参数扫描见 [SVD repair 实验](../experiments/2026-04-04-svd-repair-sweep.md)

## 其他修改

### MillardCurve bug fix（`millard_curves.py`）
- Line 284: `self._eval_integral_scalar(xi)` → `self.z_eval_integral_scalar(xi)`（方法重命名后调用点未更新）

### fiber kernel skip（已实现但无效）
在 `solve_tetfibermillard_kernel` 中加入 det(F) 检查，跳过近反转 tet 的 fiber 约束。实测无效（peak 59 vs baseline 62），因为问题在于邻居挤压而非自身 fiber 力。参数 `fiber_skip_detF` 默认 0 不启用。

### freeze_near_inverted（已实现但无效）
在 post_smooth 后检测 det(F) < threshold 的 tet，将其顶点回退到 pprev。实测无有效改善，且可能阻碍恢复。参数 `freeze_detF_threshold` 默认 0 不启用。

## 修改文件

| 文件 | 改动 |
|------|------|
| `data/muscle/config/bicep_fibermillard_coupled.json` | +`post_smooth_iters: 5`, +SVD repair 参数 |
| `src/VMuscle/muscle_warp.py` | +`post_smooth()`, +`freeze_near_inverted()`, +SVD kernels, +fiber skip, +`mat33_inverse_fn` |
| `src/VMuscle/muscle_common.py` | substep 循环中调用 `post_smooth()`, `freeze_near_inverted()` |
| `src/VMuscle/solver_muscle_bone_coupled_warp.py` | coupled substep 循环中调用 `post_smooth()`, `freeze_near_inverted()` |
| `src/VMuscle/millard_curves.py` | bug fix: `_eval_integral_scalar` → `z_eval_integral_scalar` |
| `scripts/experiments/sweep_post_smooth.py` | 参数扫描脚本 |
| `scripts/run_couple3_curves.py` | 曲线绘图脚本 |
| `docs/experiments/2026-04-04-svd-repair-sweep.md` | SVD 参数扫描实验记录 |
