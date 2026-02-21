---
name: ai-test-workflow
description: Run automated tests and visually inspect PNG results for muscle simulation regression testing. Use when you need to verify simulation correctness after code changes.
---

# AI 自动化测试工作流

## 快速命令

```bash
# 一键全跑
uv run pytest tests/test_regression.py -v

# 跳过慢测试（只跑快速检查）
uv run pytest tests/test_regression.py -m "not slow" -v

# 跳过 CUDA 测试（无 GPU 环境）
uv run pytest tests/test_regression.py -m "not cuda" -v

# 只跑数值测试
uv run pytest tests/test_regression.py -m "not visual and not cuda" -v

# 手动运行单个测试（仍然可用）
uv run python tests/test_muscle_warp_vs_taichi.py --mode jacobi --steps 100
uv run python tests/test_visual_comparison.py
```

## 完整工作流

1. **运行 pytest**：`uv run pytest tests/test_regression.py -v`
2. **检查输出**：查看 PASS/FAIL 状态
3. **视觉检查**（对 `visual` 标记的测试）：
   - 用 Read 工具查看 `output/comparison/*.png`
   - AI 判断渲染结果是否合理：
     - 网格形状是否正常（肌肉形状，无爆炸/坍塌）
     - 无 NaN 导致的空白/缺失三角形
     - Taichi 和 Warp 两版视觉上一致
4. **汇总报告**：列出所有测试结果 + 视觉判断

## 测试标记说明

| 标记 | 含义 | 典型耗时 |
|------|------|----------|
| `slow` | 长时间模拟（>30s） | 1-5 min |
| `visual` | 生成 PNG 输出 | 2-5 min |
| `cuda` | 需要 CUDA GPU | 30s-2 min |

## 判断标准

- **Jacobi 数值一致性**：Warp vs Taichi 最大误差 < 1e-3
- **Gauss-Seidel 数值一致性**：最大误差 < 0.1（因并行非确定性）
- **CUDA 稳定性**：无 NaN 顶点
- **视觉比较**：PNG 文件存在且 > 1KB，无全 NaN 帧
