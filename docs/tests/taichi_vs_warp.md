# 测试文档

所有测试均通过 `uv run` 运行，无需额外安装。从项目根目录执行。

---

## 1. Taichi vs Warp 数值对比

**文件**: `tests/test_muscle_warp_vs_taichi.py`

对比 `muscle.py` (Taichi) 和 `muscle_warp.py` (Warp) 在相同配置下的顶点位置，验证数值一致性。两者均运行在 CPU 上。

```bash
# Jacobi 模式（推荐首测，确定性，阈值 1e-3）
uv run python tests/test_muscle_warp_vs_taichi.py --mode jacobi --steps 100

# Gauss-Seidel 模式（非确定性，阈值 0.1）
uv run python tests/test_muscle_warp_vs_taichi.py --mode gauss-seidel --steps 100

# 同时运行两种模式
uv run python tests/test_muscle_warp_vs_taichi.py --mode both --steps 100
```

**输出**: 每 10 步报告 max/mean 误差、体积误差、采样顶点误差，最终给出 PASS/FAIL。

**预期结果**:
| 模式 | 最大误差 | 阈值 |
|------|---------|------|
| Jacobi | ~2.3e-4 | 1e-3 |
| Gauss-Seidel | ~5.0e-2 | 0.1 |

---

## 2. 可视化对比

**文件**: `tests/test_visual_comparison.py`

Headless 运行两个版本 400 步，用 matplotlib 渲染表面网格快照并保存为 PNG。

```bash
uv run python tests/test_visual_comparison.py
```

**输出**:
- `output/taichi/step_XXXX.png` — Taichi 版各帧
- `output/warp/step_XXXX.png` — Warp 版各帧
- `output/comparison/step_XXXX.png` — 并排对比（含误差统计）

**快照帧**: step 1, 50, 100, 200, 400

**注意**: 默认使用 CUDA 运行 Warp。脚本头部有 `wp.set_device("cpu")` 可切换到 CPU。

---

## 3. Warp CPU vs CUDA 对比

**文件**: `tests/test_warp_cpu_vs_cuda.py`

同一 Warp 模拟分别在 CPU 和 CUDA 上运行，检测 GPU 竞争条件是否导致发散。

```bash
uv run python tests/test_warp_cpu_vs_cuda.py
```

**输出**: 每个 checkpoint 报告 NaN 数量和两设备间的 max/mean 差异。

**预期结果**: 修复 `atomic_add` 后，CUDA 上不应出现 NaN。Gauss-Seidel 模式下 CPU/CUDA 仍有小差异（并行调度不同），属正常。

---

## 4. CUDA Jacobi 稳定性

**文件**: `tests/test_warp_cuda_jacobi.py`

在 CUDA 上以 Jacobi 模式运行 100 步，验证 GPU 模拟稳定性。

```bash
uv run python tests/test_warp_cuda_jacobi.py
```

**输出**: 每个 checkpoint 报告 NaN 数量、体积误差、质心坐标。

**预期结果**: 0 NaN，体积误差稳定在 -0.25 左右。

---

## 测试顺序建议

1. 先跑 **数值对比 Jacobi**（最权威，确定性基准）
2. 再跑 **CUDA Jacobi 稳定性**（确认 GPU 不发散）
3. 可选跑 **CPU vs CUDA 对比** 和 **可视化对比**
