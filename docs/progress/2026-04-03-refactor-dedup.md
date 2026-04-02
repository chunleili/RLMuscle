# 重构：消除 example 重复代码

## 完成内容

### Stage 1: SimpleArm 共享模块
- 创建 `src/VMuscle/simple_arm_helpers.py`，提取：
  - `build_mjcf()` — MuJoCo XML 生成（之前在 3 个文件中复制）
  - `build_attach_constraints()` — XPBD ATTACH 约束构建（2 个文件）
  - `compute_excitation()` — Hermite smoothstep 激励调度（5 个文件）
  - `write_sto()` — OpenSim .sto 文件写入（6 个文件）
- 更新 5 个 SimpleArm example 使用共享模块
- 删除 `sys.path.insert` hack 和跨 example 的 import
- 用 `rotation_matrix_align` 替换 `_rotation_matrix_from_z_to`
- **减少 ~450 行重复代码**

### Stage 2: Config 加载统一
- 添加 `load_config_dict()` 到 `src/VMuscle/config.py`
- 修复 `example_human_import2.py` 中的拼写错误 `load_conifg`

### Stage 3: Sliding-ball 共享模块
- 创建 `src/VMuscle/sliding_ball_helpers.py`，提取：
  - `build_xpbd_sliding_ball_sim()` — 程序化圆柱网格 + MuscleSim 构建
  - `compute_rest_matrices()` — 逆休息矩阵计算
  - `load_sliding_ball_config_base()` — 共享 config 展平
- 更新 3 个 sliding-ball example 使用共享模块
- **减少 ~430 行重复代码**

## 验证
- `scripts/run_simple_arm_comparison.py --mode xpbd-millard`: RMSE=1.79 deg（与重构前一致）
- `scripts/run_simple_arm_comparison.py --mode mujoco`: RMSE=1.76 deg（与重构前一致）
- `scripts/run_sliding_ball_comparison.py --mode xpbd-dgf --skip-opensim`: 正常运行

## 总计
- 删除 ~880 行重复代码，新增 ~540 行共享模块
- 净减少 ~340 行，消除所有主要重复
