# example_couple2.py 完整调用流程详解

> 日期：2026-03-09

## 概述

`example_couple2.py` 是肌肉-骨骼耦合仿真的 Warp-only 版本示例。它将 Warp PBD 肌肉仿真（`MuscleSim`）与 Newton 刚体骨骼求解器双向耦合，模拟二头肌驱动肘关节弯曲。

---

## 一、程序入口

程序通过两种方式启动：

**方式 A：直接运行**
```
uv run python examples/example_couple2.py --auto
```
→ `example_couple2.py:233` `if __name__ == "__main__": main()`

**方式 B：通过 main.py**
```
RUN=example_couple2 uv run main.py
```
→ `main.py:32` `importlib.import_module("examples.example_couple2")`
→ `main.py:34` `module.main()` → 跳转到 `example_couple2.py:189`

---

## 二、`main()` 函数逐步分析 (`example_couple2.py:189-234`)

### Step 1: 解析命令行参数 (L190)
```python
args = _create_parser().parse_args()
```
→ **JUMP IN** `_create_parser()` (`example_couple2.py:175-186`)
- 创建 `argparse.ArgumentParser`
- 注册 `--auto`（是否自动激活调度）、`--steps`（默认300）、`--config`（默认 `data/muscle/config/bicep.json`）、`--device`（默认 `cpu`）

→ **JUMP OUT** 返回 parser

### Step 2: 配置日志 (L192)
```python
setup_logging(to_file=True)
```
→ **JUMP IN** `setup_logging()` (`example_couple2.py:30-44`)
- 设置 root logger 级别为 WARNING
- 设置 `"couple"` logger 级别为 DEBUG，不向上传播
- 添加 FileHandler 写到 `log.md` + StreamHandler 输出到终端

→ **JUMP OUT**

### Step 3: 初始化 Warp (L194-195)
```python
wp.init()
wp.set_device(args.device)  # 默认 "cpu"
```

### Step 4: 加载肌肉配置 (L197-199)
```python
cfg = load_config(args.config)   # "data/muscle/config/bicep.json"
cfg.gui = False
cfg.render_mode = None
```
→ **JUMP IN** `load_config()` (`muscle_warp.py:90-110`)
- 打开 JSON 文件，读取所有字段
- 遍历 `SimConfig` 的 dataclass fields，把匹配的 key 填入 kwargs
- 对 `geo_path`, `bone_geo_path` 做 `Path()` 转换
- 额外处理 `muscle_prim_path` 和 `bone_prim_paths`（USD 支持）
- 返回 `SimConfig` 实例

→ **JUMP OUT** 返回 `cfg`

### Step 5: 创建 MuscleSim (L201)
```python
sim = MuscleSim(cfg)
```
→ **JUMP IN** `MuscleSim.__init__()` (`muscle_warp.py:1514-1539`)

这是最复杂的初始化过程，依次调用：

#### 5a. 加载肌肉网格 (L1520-1523)
```python
mesh_data = load_mesh(cfg.geo_path, prim_path=...)
```
→ **JUMP IN** `load_mesh()` (`muscle_warp.py:279-291`)
- 根据路径后缀分派：`.geo` → `load_mesh_geo()`，`.json` → `load_mesh_json()`，USD → `load_mesh_usd()` 等
- 对于 `bicep.geo`：
  → **JUMP IN** `load_mesh_geo()` (`muscle_warp.py:136`)
  - 用 `Geo` 解析器读取 Houdini `.geo` 文件
  - 返回 `(positions, tets, fibers, tendon_mask, geo_obj)`
  → **JUMP OUT**

→ **JUMP OUT** 返回 `(pos0_np, tet_np, v_fiber_np, v_tendonmask_np, geo)`

赋值到 `self.pos0_np`, `self.tet_np`, `self.v_fiber_np`, `self.v_tendonmask_np`, `self.geo`

#### 5b. 加载骨骼几何 (L1526)
```python
self.load_bone_geo(cfg.bone_geo_path)
```
→ **JUMP IN** `load_bone_geo()` (`muscle_warp.py:1750-1820`)
- 对于 `.geo` 文件：用 `Geo` 解析器加载
- 读取顶点 `bone_pos`、索引 `bone_indices_np`
- 如果有 `muscle_id` 点属性，构建 `bone_muscle_ids` 字典（按 muscle_id 分组顶点索引）
- 创建 `bone_pos_field` Warp 数组用于 GPU 上的骨骼位置同步

→ **JUMP OUT**

#### 5c. 分配字段 (L1529)
```python
self._allocate_fields()
```
→ **JUMP IN** `_allocate_fields()` (`muscle_warp.py:1548-1566`)
- 为所有物理量分配 Warp 零数组：`pos`, `pprev`, `pos0`, `vel`, `force`, `mass`, `stopped`, `v_fiber_dir`, `dP`, `dPw`, `tet_indices`, `rest_volume`, `rest_matrix`, `activation`

→ **JUMP OUT**

#### 5d. 初始化字段 (L1530)
```python
self._init_fields()
```
→ **JUMP IN** `_init_fields()` (`muscle_warp.py:2162-2187`)
- 把 numpy 数据写入 Warp 数组（pos0, pos, tet_indices, v_fiber_dir）
- 计算 cell 级 tendon mask（启动 `compute_cell_tendon_mask_kernel` Warp kernel）
- 初始化 `activation` 为零

→ **JUMP OUT**

#### 5e. 预计算静息量 (L1531)
```python
self._precompute_rest()
```
→ **JUMP IN** `_precompute_rest()` (`muscle_warp.py:2190-2195`)
- 启动 `precompute_rest_kernel` Warp kernel 计算每个四面体的静息体积、静息矩阵、顶点质量

→ **JUMP OUT**

#### 5f. 构建表面三角形 (L1532)
```python
self._build_surface_tris()
```
→ **JUMP IN** `_build_surface_tris()` (`muscle_warp.py:1542-1545`)
  → **JUMP IN** `build_surface_tris()` (`muscle_warp.py:295-313`)
  - 从四面体提取边界面（只出现一次的面 = 表面面）
  → **JUMP OUT**

→ **JUMP OUT**

#### 5g. 构建约束 (L1533)
```python
self.build_constraints()
```
→ **JUMP IN** `build_constraints()` (`muscle_warp.py:2080-2159`)
- 遍历 `cfg.constraints` 列表中每个约束配置
- 根据类型分派创建：
  - `volume` → `create_tet_volume_constraint()`
  - `fiber` → `create_tet_fiber_constraint()` (L1647)
  - `pin` → `create_pin_constraints()` (L1694)
  - `attach` → `create_attach_constraints()` (L1836) — **最重要！** 建立肌肉-骨骼连接对
  - `distanceline`, `tetarap`, `triarap` 等
- 将所有约束打包成 Warp 结构化数组 `self.cons`

→ **JUMP OUT**

→ **JUMP OUT** 从 `MuscleSim.__init__` 返回 `sim` 对象

---

### Step 6: 构建肘关节 Newton 模型 (L204)
```python
model, state, radius_link, joint, selected_indices = build_elbow_model(sim)
```
→ **JUMP IN** `build_elbow_model()` (`example_couple2.py:88-132`)

#### 6a. 提取桡骨网格 (L97)
```python
group_name, selected_indices, radius_vertices, radius_faces = _extract_radius_mesh(sim)
```
→ **JUMP IN** `_extract_radius_mesh()` (`example_couple2.py:47-85`)
- 从 `sim.bone_pos` 获取全部骨骼顶点
- 从 `sim.bone_indices_np` 获取三角面索引，reshape 为 (N,3)
- 在 `sim.bone_muscle_ids` 中找包含 `"radius"` 的组
- 过滤出仅属于桡骨的三角面
- 重映射索引，返回局部顶点和局部面

→ **JUMP OUT** 返回 `(group_name, selected_indices, local_vertices, local_faces)`

#### 6b. 创建 Newton 刚体模型 (L94-128)
- `builder = newton.ModelBuilder(up_axis=Y, gravity=0)` — 无重力
- `newton.solvers.SolverMuJoCo.register_custom_attributes(builder)` — 注册 MuJoCo 特有属性
- `builder.add_link()` (L99) — 添加桡骨刚体链接
- `builder.add_shape_mesh()` (L100-109) — 用桡骨网格创建碰撞形状，自动计算惯性
- `builder.add_joint_revolute()` (L111-123) — 在 `ELBOW_PIVOT` 处创建转动关节
  - 旋转轴 `ELBOW_AXIS`
  - 角度范围 [-3.0, 3.0] rad
  - 具有 `armature`（虚拟惯性）和 `friction`
- `builder.add_articulation([joint], label="elbow")` (L124) — 将关节注册为关节体
- `builder.finalize()` (L126) → `model`
- `model.state()` (L127) → `state`（包含 body_q, joint_q 等状态）
- `newton.eval_fk(model, joint_q, joint_qd, state)` (L128) — 正向运动学，把关节角度转为刚体位姿

→ **JUMP OUT** 返回 `(model, state, radius_link, joint, selected_indices)`

---

### Step 7: 创建耦合求解器 (L206-211)
```python
solver = SolverMuscleBoneCoupledWarp(model, sim, k_coupling=5000.0, max_torque=50.0)
```
→ **JUMP IN** `SolverMuscleBoneCoupledWarp.__init__()` (`solver_muscle_bone_coupled_warp.py:34-62`)
- 保存 `model` 和 `core`（即 MuscleSim 实例）
- 尝试创建 `SolverMuJoCo`，失败则退回 `SolverFeatherstone` — 这是骨骼求解器
- 设置耦合参数：`k_coupling=5000.0`（弹簧刚度）、`max_torque=50.0`（力矩上限）、`torque_ema=0.2`（指数移动平均）

→ **JUMP OUT**

### Step 8: 配置耦合 (L213-221)
```python
solver.configure_coupling(
    bone_body_id=radius_link,
    bone_rest_verts=sim.bone_pos[selected_indices],
    bone_vertex_indices=selected_indices,
    joint_index=joint,
    joint_pivot=ELBOW_PIVOT,
    joint_axis=ELBOW_AXIS,
)
```
→ **JUMP IN** `configure_coupling()` (`solver_muscle_bone_coupled_warp.py:64-108`)
- 保存骨骼刚体 ID、静息顶点、顶点索引
- 从 `model.joint_qd_start` 计算关节自由度索引 `_joint_dof_index`
- 归一化关节旋转轴
- **关键：** 遍历 `core.attach_constraints`，找出 `src`（肌肉端顶点）↔ `tgt`（骨骼端顶点）中 tgt 在 `selected_indices` 范围内的配对 → 存入 `_attach_pairs`
- 置 `_coupling_configured = True`

→ **JUMP OUT**

### Step 9: 日志输出 (L223-228)
记录 dt、muscle_substeps、bone_substeps 等参数。

---

### Step 10: 仿真主循环 (L230)
```python
run_loop(solver, state, cfg, dt=dt, n_steps=args.steps, auto=args.auto)
```
→ **JUMP IN** `run_loop()` (`example_couple2.py:150-172`)

循环 `n_steps` 次（默认 300 步）：

#### 10a. 激活调度 (L153-154)
如果 `--auto` 启用：
```python
cfg.activation = _activation_schedule(step, n_steps)
```
→ **JUMP IN** `_activation_schedule()` (`example_couple2.py:135-147`)

| 进度 t | activation |
|--------|-----------|
| 0~20%  | 0.0（静止）|
| 20~30% | 0.5（半激活）|
| 30~50% | 1.0（全激活）|
| 50~70% | 0.7（减弱）|
| 70~80% | 0.3（进一步减弱）|
| 80~100%| 0.0（松弛）|

→ **JUMP OUT**

#### 10b. 求解器步进 (L156)
```python
solver.step(state, state, dt=dt)
```
→ **JUMP IN** `SolverMuscleBoneCoupledWarp.step()` (`solver_muscle_bone_coupled_warp.py:162-209`)

这是**核心耦合循环**，内部依次执行：

##### (i) 骨骼→肌肉同步 (L170-171)
```python
self._sync_bone_positions(state_in)
```
→ **JUMP IN** `_sync_bone_positions()` (`solver_muscle_bone_coupled_warp.py:118-131`)
- 从 `state.body_q` 获取桡骨的 `(position, quaternion)`
- 用四元数旋转静息顶点 + 平移 → 得到变形后的骨骼顶点
  → **JUMP IN** `_quat_rotate()` (`solver_muscle_bone_coupled_warp.py:111-116`)
  - 向量化四元数旋转公式：`p + qw*(2*q×p) + q×(2*q×p)`
  → **JUMP OUT**
- 写回 `core.bone_pos_field`（Warp 数组）

→ **JUMP OUT**

##### (ii) 肌肉仿真步 (L173-174)
```python
self.core.activation.fill_(self.core.cfg.activation)
self.core.step()
```
→ **JUMP IN** `MuscleSim.step()` (`muscle_warp.py:2226-2243`)
- `update_attach_targets()` (L2227) — 用 `bone_pos_field` 更新 attach 约束的目标位置
- 循环 `num_substeps` 次子步：
  1. `integrate_kernel` — 半隐式欧拉积分（重力 + 阻尼）
  2. `clear` — 清零 dP/dPw 累加器
  3. `solve_constraints_kernel` — 求解所有约束（体积、纤维、attach、ARAP 等）
  4. `apply_dP_kernel` — Jacobi 式更新位置
  5. `update_velocities_kernel` — 从位置差更新速度

→ **JUMP OUT**

##### (iii) 计算肌肉→骨骼力矩 (L176-188)
```python
torque = self._compute_muscle_torque()
```
→ **JUMP IN** `_compute_muscle_torque()` (`solver_muscle_bone_coupled_warp.py:133-160`)
- 遍历每个 `(src, tgt)` attach 对：
  - `disp = muscle_pos[src] - bone_pos[tgt]`（位移差）
  - `force = k_coupling * disp`（弹簧力）
  - `arm = bone_pos[tgt] - joint_pivot`（力臂）
  - `torque += cross(arm, force)`（力矩累加）
- 除以配对数取平均
- 裁剪到 `max_torque`
- EMA 平滑

→ **JUMP OUT**

然后将力矩投影到关节轴上，写入 `control.joint_f`。

##### (iv) 骨骼求解器子步 (L190-192)
```python
dt_sub = dt / bone_substeps
for _ in range(bone_substeps):
    self.bone_solver.step(state_in, state_out, control, None, dt_sub)
```
用 MuJoCo（或 Featherstone）积分刚体骨骼运动。

##### (v) 再次同步骨骼位置 (L194-195)
```python
self._sync_bone_positions(state_out)
```
确保肌肉看到最新骨骼位姿。

→ **JUMP OUT** 从 `solver.step()` 返回

#### 10c. 日志打印 (L158-172)
每25步打印一次状态：body 位置、关节角度、力矩大小。

→ **JUMP OUT** 从 `run_loop()` 返回

---

## 三、完整调用树

```
main.py:34 → example_couple2.main()
│
├── L190: _create_parser().parse_args()              [example_couple2.py:175]
├── L192: setup_logging(to_file=True)                [example_couple2.py:30]
├── L194-195: wp.init(), wp.set_device()
├── L197: load_config(args.config)                   [muscle_warp.py:90]
├── L201: MuscleSim(cfg)                             [muscle_warp.py:1514]
│   ├── L1520: load_mesh() → load_mesh_geo()         [muscle_warp.py:279→136]
│   ├── L1526: load_bone_geo()                       [muscle_warp.py:1750]
│   ├── L1529: _allocate_fields()                    [muscle_warp.py:1548]
│   ├── L1530: _init_fields()                        [muscle_warp.py:2162]
│   ├── L1531: _precompute_rest()                    [muscle_warp.py:2190]
│   ├── L1532: _build_surface_tris()                 [muscle_warp.py:1542→295]
│   └── L1533: build_constraints()                   [muscle_warp.py:2080]
│
├── L204: build_elbow_model(sim)                     [example_couple2.py:88]
│   ├── L97:  _extract_radius_mesh(sim)              [example_couple2.py:47]
│   ├── L94-109: Newton ModelBuilder + add_link + add_shape_mesh
│   ├── L111-124: add_joint_revolute + add_articulation
│   └── L126-128: finalize + state + eval_fk
│
├── L206: SolverMuscleBoneCoupledWarp(...)            [solver_...warp.py:34]
│   └── 创建 MuJoCo/Featherstone 骨骼求解器
│
├── L214: solver.configure_coupling(...)              [solver_...warp.py:64]
│   └── 匹配 attach_pairs (肌肉↔骨骼顶点配对)
│
└── L230: run_loop(solver, state, cfg, ...)           [example_couple2.py:150]
    └── 循环 n_steps 次:
        ├── _activation_schedule()                    [example_couple2.py:135]
        └── solver.step()                             [solver_...warp.py:162]
            ├── _sync_bone_positions(state_in)        [solver_...warp.py:118]
            │   └── _quat_rotate()                    [solver_...warp.py:111]
            ├── core.step()                           [muscle_warp.py:2226]
            │   ├── update_attach_targets()           [muscle_warp.py:2208]
            │   └── num_substeps × {integrate → clear → solve_constraints → apply_dP → update_vel}
            ├── _compute_muscle_torque()              [solver_...warp.py:133]
            ├── bone_solver.step() × bone_substeps    [Newton MuJoCo/Featherstone]
            └── _sync_bone_positions(state_out)       [solver_...warp.py:118]
```

---

## 四、核心数据流

```
 ┌──────────────┐         骨骼位姿 → 变形顶点          ┌──────────────┐
 │  Newton 刚体  │ ──── _sync_bone_positions() ──────→ │  Warp 肌肉    │
 │  (桡骨+关节)  │                                     │  (PBD 求解)   │
 │              │ ←─── _compute_muscle_torque() ────── │              │
 └──────────────┘      attach 位移 → 关节力矩           └──────────────┘
```

- **骨骼→肌肉**：刚体位姿通过四元数旋转写入 `bone_pos_field`，attach 约束目标跟随更新。
- **肌肉→骨骼**：attach 约束产生的位移差 × 弹簧刚度 = 力，叉乘力臂 = 力矩，投影到关节轴驱动旋转。

---

## 五、涉及的关键文件

| 文件 | 角色 |
|------|------|
| `examples/example_couple2.py` | 示例入口，构建 Newton 模型，运行仿真循环 |
| `src/VMuscle/muscle_warp.py` | `MuscleSim` 核心类，PBD 肌肉仿真 |
| `src/VMuscle/solver_muscle_bone_coupled_warp.py` | `SolverMuscleBoneCoupledWarp` 耦合求解器 |
| `src/VMuscle/geo.py` | Houdini `.geo` 文件解析器 |
| `data/muscle/config/bicep.json` | 二头肌仿真配置 |
| `data/muscle/model/bicep.geo` | 肌肉四面体网格 |
| `data/muscle/model/bicep_bone.geo` | 骨骼三角网格 |
