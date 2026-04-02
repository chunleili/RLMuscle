# XPBD-Millard 运行指南

## Sliding Ball 对比

```bash
# 默认: XPBD-Millard vs OpenSim-Millard
uv run python scripts/run_sliding_ball_comparison.py

# 其他模式
uv run python scripts/run_sliding_ball_comparison.py --mode xpbd-dgf       # XPBD-DGF vs OpenSim-DGF
uv run python scripts/run_sliding_ball_comparison.py --mode vbd            # VBD vs OpenSim-DGF

# 跳过 OpenSim（只跑仿真+出图）
uv run python scripts/run_sliding_ball_comparison.py --skip-opensim
```

输出: `output/sliding_ball_comparison_<mode>.png`

## Simple Arm 对比

```bash
# XPBD-Millard vs OpenSim-Millard
uv run python scripts/run_simple_arm_comparison.py --mode xpbd-millard

# 其他模式
uv run python scripts/run_simple_arm_comparison.py --mode xpbd             # XPBD-DGF vs OpenSim-DGF
uv run python scripts/run_simple_arm_comparison.py --mode osim             # OpenSim DGF vs Millard
uv run python scripts/run_simple_arm_comparison.py --mode all              # 所有模式
```

输出: `output/simple_arm_xpbd_millard_vs_osim.png`

## 单独运行仿真

```bash
RUN=example_xpbd_millard_sliding_ball uv run main.py
RUN=example_xpbd_coupled_simple_arm_millard uv run main.py
```

## 单元测试

```bash
uv run python -m pytest tests/test_millard_curves.py -v    # 曲线精度 (CPU)
uv run python tests/test_millard_gpu.py                     # GPU 一致性
```

## 验证结果

| 场景 | XPBD-Millard | OpenSim-Millard | 误差 |
|------|-------------|-----------------|------|
| Sliding Ball λ_eq | 0.5498 | 0.5461 | 0.7% |
| Sliding Ball 球位置 | 0.0449 m | 0.0444 m | 1.2% |
| Simple Arm 稳态肘角 | 88.25° | 88.15° | 0.10° |
| Simple Arm RMSE | - | - | 1.79° |
