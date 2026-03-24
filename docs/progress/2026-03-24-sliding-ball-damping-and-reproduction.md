# Sliding Ball 阻尼调优与对比复现流程

> 日期：2026-03-24

## 概述

在 VBD vmuscle kernel 中新增 fiber damping，配合 activation dynamics，大幅减少了 sliding-ball 示例的振荡和超调。整理了完整的 VBD vs OpenSim 对比复现流程。

## 已完成

### 1. Fiber Damping 实现
- `external/newton/newton/_src/solvers/vbd/vmuscle_kernels.py`：在 `evaluate_fiber_force_and_hessian` 中新增纤维方向阻尼
  - 力：`F_damp = -fiber_damping * sigma0 * V_rest * (dl~/dt) * dldx`
  - Hessian：`H_damp = fiber_damping * sigma0 * V_rest / dt * outer(dldx, dldx)`
  - `dldx = dot(m, fiber_dir) * Fd0_hat`（沿变形后纤维方向）
- `external/newton/newton/_src/solvers/vbd/vmuscle_launch.py`：传递 `fiber_damping` 参数
- `external/newton/newton/_src/sim/builder.py`：新增 `vmuscle_fiber_damping` 属性

### 2. Activation Dynamics 集成
- `examples/example_muscle_sliding_ball.py`：用 `activation_dynamics_step_np` 替代硬编码 ramp
- excitation=1.0 (step function) → activation dynamics 自然平滑
- tau_act=15ms < dt=17ms，需要 ~1ms 子步进才能正确积分

### 3. 配置外部化
- `data/slidingBall/config.json`：所有参数从 JSON 读取
- `examples/example_muscle_sliding_ball.py`：支持 `--config` 参数

### 4. OpenSim .sto 输出
- `OpenSimExample/vbd_muscle/osim_sliding_ball.py`：用 `osim.TimeSeriesTable` + `STOFileAdapter.write` 输出标准 .sto
- `OpenSimExample/scripts/plot_sliding_ball.py`：独立脚本，从 .npz/.sto 加载数据绘图

### 5. 阻尼策略对比实验
测试了 5 种减振方法：高 Rayleigh 阻尼、小 dt、多迭代、慢 ramp、fiber damping。
结论：fiber damping 最有效，因为直接作用在纤维伸缩方向上。

## 复现流程

### 前置条件
- RLMuscle-dev 仓库：`feat/vbd-muscle` 分支
- OpenSimExample 仓库：需要 conda `opensim` 环境（OpenSim 4.5+）

### Step 1: 运行 VBD 仿真

```bash
cd D:/dev/RLMuscle-dev
uv run python -m examples.example_muscle_sliding_ball
# 或指定配置
uv run python -m examples.example_muscle_sliding_ball --config data/slidingBall/config.json
```

输出：`output/vbd_muscle_sliding_ball_default.npz`

### Step 2: 运行 OpenSim 参考仿真

```bash
cd D:/Dev/OpenSimExample/OpenSimExample
D:/App/Miniconda/envs/opensim/python.exe -c "
import sys, os, importlib.util
spec = importlib.util.spec_from_file_location('osim', 'vbd_muscle/osim_sliding_ball.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
os.makedirs('output', exist_ok=True)
mod.osim_sliding_ball(
    muscle_length=0.10, ball_mass=10.0, sigma0=300000.0,
    muscle_radius=0.02, excitation_func=lambda t: min(t/0.05, 1.0),
    t_end=5.0, dt=0.001)
"
```

输出：`output/opensim_sliding_ball.sto`

**注意**：OpenSim 参数必须与 `config.json` 中一致（length, mass, sigma0, radius, t_end）。

### Step 3: 绘制对比图

```bash
cd D:/Dev/OpenSimExample/OpenSimExample
D:/App/Miniconda/envs/opensim/python.exe scripts/plot_sliding_ball.py \
    --vbd D:/dev/RLMuscle-dev/output/vbd_muscle_sliding_ball_default.npz \
    --sto output/opensim_sliding_ball.sto
```

输出：`output/vbd_muscle_demo_default.png`

## 最终参数（config.json）

| 参数 | 值 | 说明 |
|------|-----|------|
| k_mu | 1.0 | 极小剪切刚度 |
| k_lambda | 1000 | 体积保持 |
| k_damp | 0.5 | SNH Rayleigh 阻尼 |
| fiber_damping | 0.05 | 纤维方向阻尼 |
| dt | 1/60 | 时间步 |
| iterations | 50 | VBD 迭代 |
| activation | dynamics | tau_act=15ms, 1ms 子步进 |

## 涉及的源文件

### RLMuscle-dev
| 文件 | 改动 |
|------|------|
| `examples/example_muscle_sliding_ball.py` | 从 JSON 读配置, activation dynamics, fiber damping |
| `data/slidingBall/config.json` | 新建, 参数配置 |
| `src/VMuscle/activation.py` | 已有, activation dynamics |
| `src/VMuscle/dgf_curves.py` | 已有, DGF 曲线 |
| `src/VMuscle/mesh_utils.py` | 已有, 网格工具 |
| `external/newton/.../builder.py` | 新增 `vmuscle_fiber_damping` |
| `external/newton/.../vmuscle_kernels.py` | 新增 fiber damping 力/Hessian |
| `external/newton/.../vmuscle_launch.py` | 传递 fiber_damping 参数 |

### OpenSimExample
| 文件 | 改动 |
|------|------|
| `vbd_muscle/osim_sliding_ball.py` | 修复 .sto 输出 |
| `scripts/plot_sliding_ball.py` | 重写为独立脚本 |
| `scripts/plot_damping_comparison.py` | 新建, 阻尼策略对比 |

## 下一步

- 在 vmuscle kernel 中实现隐式/平滑 F-V 速度估计
- 用真实 USD 肌肉网格测试 fiber damping 效果
- 将 OpenSim 参考仿真也配置化（读同一份 config.json）
