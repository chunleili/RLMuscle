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
| **Baseline（无修复）** | ~530 (13.5%) | -1427 | ~18（永久） |
| couple2 (TETFIBERNORM) | ~15 (0.4%) | -3.9 | 0-1 |
| Old centroid repair | ~360 (9.1%) | -0.002 | ~350（trap!） |
| SVD Jacobi repair | ~260 (6.6%) | varies | ~260（trap!） |
| **Post-smooth (3 iters)** | **~60 (1.5%)** | **-5.09** | **0（完全恢复）** |

### 关键指标
- 反转 tets 峰值：从 530 → 60（**89% 改善**）
- 完全恢复：去激活后 0 个反转 tets（vs baseline 18 个永久反转）
- 无 NaN、无 crash、无 trap state
- 峰值 torque 6.5 N·m，joint angle -0.92 rad

## 其他修改

### MillardCurve bug fix（`millard_curves.py`）
- Line 284: `self._eval_integral_scalar(xi)` → `self.z_eval_integral_scalar(xi)`（方法重命名后调用点未更新）

### SVD 修复代码（保留但未启用）
`_svd_clamp_accumulate_kernel` 和 `_apply_svd_clamp_kernel` 已实现但默认 `repair_alpha=0`，不启用。Post-smooth 方法效果更好且更稳定。

## 修改文件

| 文件 | 改动 |
|------|------|
| `data/muscle/config/bicep_fibermillard_coupled.json` | +`post_smooth_iters: 3` |
| `src/VMuscle/muscle_warp.py` | +`post_smooth()`, +SVD kernels, +`mat33_inverse_fn` |
| `src/VMuscle/muscle_common.py` | substep 循环中调用 `post_smooth()` |
| `src/VMuscle/solver_muscle_bone_coupled_warp.py` | coupled substep 循环中调用 `post_smooth()` |
| `src/VMuscle/millard_curves.py` | bug fix: `_eval_integral_scalar` → `z_eval_integral_scalar` |
