# Newton VBD 代码分析

## 一、VBD 算法概述

**论文来源**: Anka He Chen et al., "Vertex Block Descent", ACM SIGGRAPH 2024

VBD 的核心思想是：**将全局非线性优化问题分解为逐顶点的局部 3×3 线性系统**，通过 Gauss-Seidel 式迭代（按图着色分组并行）求解。每个顶点求解 `H_i · Δx_i = f_i`，其中 `f_i` 是总力，`H_i` 是对应的 3×3 Hessian。

## 二、文件结构

| 文件 | 职责 |
|------|------|
| `external/newton/newton/_src/solvers/vbd/solver_vbd.py` (~2100行) | `SolverVBD` 类：初始化、step 主循环、邻接构建 |
| `external/newton/newton/_src/solvers/vbd/particle_vbd_kernels.py` (~3400行) | 粒子 Warp kernel：弹性力/Hessian、接触力、求解 |
| `external/newton/newton/_src/solvers/vbd/rigid_vbd_kernels.py` | 刚体 AVBD kernel：关节约束、刚体碰撞、对偶更新 |
| `external/newton/newton/_src/solvers/vbd/tri_mesh_collision.py` | 三角网格自碰撞检测（BVH） |
| `external/newton/newton/_src/sim/graph_coloring.py` | 图着色算法（MCS/Greedy） |

## 三、`step()` 主循环（3阶段）

位于 `solver_vbd.py:1149-1189`：

```
Phase 1: 初始化
  ├── _initialize_rigid_bodies: 刚体前向积分 + warmstart 罚系数
  └── _initialize_particles: forward_step(惯性目标) + 穿透截断

Phase 2: 迭代（默认10次）
  ├── _solve_rigid_body_iteration: AVBD 刚体求解 + 对偶更新
  └── _solve_particle_iteration: 按颜色分组求解弹性 + 接触

Phase 3: 收尾
  ├── _finalize_particles: vel = (pos - pos_prev) / dt
  └── _finalize_rigid_bodies: 刚体速度更新 + Dahl 状态更新
```

## 四、核心 kernel: `solve_elasticity`

位于 `particle_vbd_kernels.py:3148-3286`，对每个颜色组中的顶点：

1. **惯性项**: `f += m(inertia - pos)/dt²`, `H += m/dt² · I₃`
2. **三角面弹性 (StVK)**: 遍历邻接三角面累积力/Hessian
3. **边弯曲 (二面角)**: 遍历邻接边累积弯曲力
4. **四面体弹性 (Neo-Hookean)**: 遍历邻接四面体
5. **接触力**: 加上预累积的 `particle_forces` 和 `particle_hessians`
6. **求解**: `displacement += H⁻¹ · f`（3×3 矩阵求逆）

## 五、材料模型

| 模型 | 用途 | 位置 |
|------|------|------|
| **Neo-Hookean** | 四面体软体（体积） | `particle_vbd_kernels.py:347` |
| **StVK** | 三角面布料（膜） | `particle_vbd_kernels.py:880` |
| **二面角弯曲** | 布料弯曲刚度 | `particle_vbd_kernels.py:1075` |

## 六、关键设计要点

### 6.1 图着色并行

`builder.color()` 为顶点/刚体着色，同色顶点互不邻接，可无数据竞争并行求解。

### 6.2 邻接信息 CSR 格式

`ParticleForceElementAdjacencyInfo` 存储每个顶点的邻接面/边/弹簧/四面体，使用 CSR（压缩稀疏行）格式：
- `v_adj_faces` / `v_adj_faces_offsets`：邻接三角面（id + 顶点序号）
- `v_adj_edges` / `v_adj_edges_offsets`：邻接边
- `v_adj_springs` / `v_adj_springs_offsets`：邻接弹簧
- `v_adj_tets` / `v_adj_tets_offsets`：邻接四面体

### 6.3 AVBD 自适应罚系数

刚体约束使用自适应罚系数：
- Warmstart: `k = max(k_min, γ · k_prev)`（衰减但不低于下限）
- Dual update: `k = min(k_max, k + β · violation)`（违约越大 k 增长越快）

### 6.4 穿透自由截断

通过 planar DAT（Divide and Truncate）限制位移幅度防止穿透。

### 6.5 性能优化

- **Tile API 加速**: `solve_elasticity_tile` 利用 CUDA tile API 加速
- **CUDA Graph Capture**: 整个 simulate 循环可 capture 为 graph 重放
- **碰撞检测频率控制**: `particle_collision_detection_interval` 控制重新检测的频率

## 七、使用示例

```python
builder = newton.ModelBuilder()
builder.add_soft_grid(...)       # 添加软体
builder.color()                  # 图着色（VBD 必需）
model = builder.finalize()

solver = newton.solvers.SolverVBD(model, iterations=10)
state_0, state_1 = model.state(), model.state()
contacts = model.contacts()

for step in range(num_steps):
    state_0.clear_forces()
    model.collide(state_0, contacts)
    solver.step(state_0, state_1, control, contacts, dt)
    state_0, state_1 = state_1, state_0
```
