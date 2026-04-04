# 性能分析：三个核心 Example

**日期**: 2026-04-04  
**分支**: dev  
**目标**: 在进入 RL 阶段前，分析 `example_couple3`、`example_xpbd_coupled_simple_arm`、`example_mujoco_simple_arm` 的性能瓶颈  
**环境**: CPU (x86_64), Warp 1.12.0, CUDA 12.9, RTX 4090 ×2（模拟运行在 CPU 上）

## 方法

1. 代码静态分析（识别热路径中的反模式）
2. cProfile 实际运行（**以 profile 数据为准**）

## Profile 结果

### 复现命令

```bash
uv run python scripts/profile_examples.py           # 全部
uv run python scripts/profile_examples.py couple3    # 单个
uv run python scripts/profile_examples.py xpbd
uv run python scripts/profile_examples.py mujoco
```

Profile 输出保存于 `output/profile_*.prof`。

---

### couple3 — 100 步, 61.87s (618ms/step)

| 排名 | 函数 | tottime | cumtime | 调用次数 | 占比 |
|------|------|---------|---------|---------|------|
| 1 | `context.py:invoke` (Warp kernel dispatch) | **47.97s** | 48.39s | 135,000 | **77.5%** |
| 2 | `mesh_utils.py:check_mesh_quality` | **2.07s** | 5.91s | 100 | **3.3%** |
| 3 | `context.py:pack_arg` | 1.68s | 2.71s | 1,089,000 | 2.7% |
| 4 | `column_stack` (from check_mesh_quality) | 1.07s | 1.42s | 393,800 | 1.7% |
| 5 | `np.linalg.det` (from check_mesh_quality) | 1.02s | 2.14s | 393,800 | 1.6% |
| 6 | `_sync_bone_positions` | 0.08s | 0.56s | 3,100 | 0.9% |
| 7 | `_compute_raw_muscle_torque` | 0.06s | 0.40s | 3,000 | 0.6% |

**关键热点分解**:
- `post_smooth` (5 iter × 30 substeps): cumtime = **37.0s** (59.8%) — 占 kernel dispatch 的主体
- `repair_inverted_tets`: cumtime = **4.9s** (7.9%)
- `solve_constraints` (主迭代): cumtime = **9.7s** (15.7%)
- `check_mesh_quality`: tottime = **2.07s** — 纯 Python per-tet 循环 (3938 tets × 100 steps)

**结论**: couple3 的瓶颈是 **Warp kernel launch 开销**（135k 次 launch 占 77.5% 时间）。`post_smooth` 是最大贡献者（每帧 750 次 kernel launch）。GPU-CPU sync（`_sync_bone_positions` + `_compute_raw_muscle_torque`）在绝对时间上**不显著**（< 1s），被静态分析高估了。`check_mesh_quality` 的 Python 循环是第二大纯开销（2s）。

---

### xpbd_coupled_simple_arm — 100 步, 6.82s (68ms/step)

| 排名 | 函数 | tottime | cumtime | 调用次数 | 占比 |
|------|------|---------|---------|---------|------|
| 1 | `context.py:invoke` (Warp kernel dispatch) | **1.18s** | 1.27s | 36,091 | **17.3%** |
| 2 | 主函数自身 (example loop) | **0.54s** | 6.82s | 1 | 7.9% |
| 3 | `activation.py:activation_dynamics_step_np` | **0.36s** | 0.47s | 30,000 | **5.3%** |
| 4 | `muscle_common.py:compute_fiber_stretches` | **0.29s** | 0.61s | 100 | **4.3%** |
| 5 | `millard_curves.py:_horner_eval_deriv` | 0.26s | 0.27s | 162,876 | 3.8% |
| 6 | `column_stack` | 0.26s | 0.34s | 96,480 | 3.8% |
| 7 | `millard_curves.py:_newton_find_u` | 0.18s | 0.57s | 30,000 | 2.6% |
| 8 | `millard_curves.py:eval` | 0.17s | 0.94s | 60,000 | 2.5% |
| 9 | `context.py:pack_arg` | 0.38s | 0.58s | 243,547 | 5.6% |
| 10 | `mujoco.mj_step` | 0.15s | 0.15s | 30,000 | 2.2% |

**结论**: xpbd 开销分布更均匀。Warp kernel dispatch 仍是最大单项（17%），但 **Python 端 Millard curve 评估**（`eval` + `_newton_find_u` + `_horner_eval_deriv` 合计 ~0.61s = **8.9%**）和 **activation dynamics**（0.36s = 5.3%）、**compute_fiber_stretches**（0.29s = 4.3%）都是重要贡献者。MuJoCo 步进本身只占 2.2%。

---

### mujoco_simple_arm — 200 步, 0.047s (0.24ms/step)

| 排名 | 函数 | tottime | cumtime | 调用次数 | 占比 |
|------|------|---------|---------|---------|------|
| 1 | `activation.py:activation_dynamics_step_np` | **0.011s** | 0.014s | 1,000 | **23.4%** |
| 2 | `dgf_curves.py:active_force_length` | **0.007s** | 0.007s | 1,000 | **14.9%** |
| 3 | 主函数自身 | 0.006s | 0.047s | 1 | 12.8% |
| 4 | `mujoco.mj_step` | 0.004s | 0.004s | 1,000 | 8.5% |
| 5 | `np.clip` 相关 | 0.004s | 0.010s | 3,000 | 8.5% |
| 6 | `dgf_curves.py:passive_force_length` | 0.003s | 0.007s | 1,000 | 6.4% |

**结论**: 极快（0.047s/200步）。相对占比中 `activation_dynamics_step_np`（23%）和 DGF curves（21%）是最大开销，但**绝对时间可忽略**。对 RL 训练无瓶颈。

---

## 分析：静态分析 vs Profile 结果对比

| 静态分析预测 | Profile 验证 | 结论 |
|-------------|-------------|------|
| GPU-CPU sync (`_sync_bone_positions`, `_compute_raw_muscle_torque`) 为 HIGH | couple3 中合计 < 1s / 62s (**< 2%**) | **高估**。kernel dispatch 远大于 sync 开销 |
| `check_mesh_quality` Python per-tet 循环为 HIGH | couple3: 2.07s tottime (**3.3%**) | **正确**，但非主要瓶颈 |
| `activation_dynamics_step_np` 标量 NumPy 为 HIGH | xpbd: 0.36s (**5.3%**), mujoco: 0.011s | **xpbd 中正确**，mujoco 中绝对值可忽略 |
| Millard curves Python 开销未预测 | xpbd: 合计 0.61s (**8.9%**) | **被低估**，是 xpbd 的重要瓶颈 |
| Warp kernel launch 开销未预测 | couple3: **47.97s (77.5%)** | **最大发现**，静态分析完全遗漏 |

## 修复优先级（基于 Profile）

### P0 — 减少 Warp kernel launch 数量（couple3 主瓶颈）

couple3 每步 1350 次 kernel launch，其中 `post_smooth`（5 iter × 30 substeps × 5 kernel/iter = **750/step**）是最大贡献者。

1. **减少 `post_smooth_iters`**: 从 5 降到 2-3，直接减少 ~300-450 kernel launch/step
2. **Kernel fusion**: 合并 `clear()` + `_dispatch_constraints()` 中的小 kernel
3. **减少 `num_substeps`**: 从 30 降低（需要验证稳定性）

### P1 — Python 端 Millard curves 和 activation（xpbd 瓶颈）

4. **Millard `_newton_find_u`**: 每次 `eval` 做 Newton 迭代求解分段参数 u，30k 次调用占 0.57s。考虑查表或 Warp kernel 实现
5. **`activation_dynamics_step_np`**: 添加纯 `math` 标量版本，避免 30k 次 `np.array` 分配（xpbd: 0.36s）
6. **`compute_fiber_stretches`**: 向量化 NumPy 替代 Python per-tet 循环（xpbd: 0.29s）

### P2 — 诊断开销（可选）

7. **`check_mesh_quality`**: 向量化或降低频率（couple3: 2s）
8. **USD export**: RL 训练时禁用
9. **日志 `.numpy()`**: 训练时关闭

### 不需要修复

- `_sync_bone_positions` / `_compute_raw_muscle_torque` — profile 证明开销极小
- `mujoco_simple_arm` — 已足够快，无需优化
- `np.clip` 标量开销 — 绝对值可忽略

---

## 优化实施结果

### Phase 1: 标量函数快路径 + 向量化

| 改动 | 文件 |
|------|------|
| `activation_dynamics_step_scalar` | `activation.py` |
| `MillardCurve.eval_scalar` | `millard_curves.py` |
| `active_force_length_scalar` 等 | `dgf_curves.py` |
| 向量化 `compute_fiber_stretches` | `muscle_common.py` |

| Example | 优化前 | 优化后 | 加速 |
|---------|--------|--------|------|
| mujoco (200步) | 0.047s | 0.014s | **3.4x** |
| xpbd (100步) | 6.82s | 4.95s | **1.38x** |
| couple3 (CPU) | 61.87s | ~同 | ~1% |

### Phase 2: GPU + Jacobi + CUDA Graph Capture

| 改动 | 文件 |
|------|------|
| GPU 自动启用 Jacobi（race-safe） | `muscle_common.py`, `muscle_warp.py` |
| `post_smooth` 强制 Jacobi 模式 | `muscle_warp.py` |
| `freeze_near_inverted` GPU 下跳过 | `muscle_warp.py` |
| `_capturable_substep()` | `muscle_warp.py` |
| `capture_muscle_graph()` + graph replay | `solver_muscle_bone_coupled_warp.py` |
| `_sync_bone_positions` in-place 写入 | `solver_muscle_bone_coupled_warp.py` |
| `control.joint_f` in-place 更新 | `solver_muscle_bone_coupled_warp.py` |
| `--use-cuda-graph` CLI flag | `example_couple3.py` |

| 模式 | couple3 每步耗时 | vs CPU baseline |
|------|----------------|----------------|
| CPU (plain GS) | 618ms | 1x |
| GPU (Jacobi, no graph) | 88.2ms | **7.0x** |
| GPU (Jacobi + CUDA graph) | 72.7ms | **8.5x** |

### 复现命令

```bash
# Phase 1 验证
uv run python scripts/profile_examples.py mujoco xpbd

# Phase 2: GPU + graph
RUN=example_couple3 uv run main.py --auto --steps 100 --no-usd --device cuda:0
RUN=example_couple3 uv run main.py --auto --steps 100 --no-usd --device cuda:0 --use-cuda-graph
```
