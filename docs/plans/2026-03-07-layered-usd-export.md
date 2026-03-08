# 方案 A：基于 UsdIO 的刚柔耦合 Layered USD 导出

## 目标

实现一个完整的 USD 工作流：**从同一个 USD 读取肌肉+骨骼 → Warp 仿真 → 写回 layered USD**，保持原始 USD 不被修改。

## 现状分析

### 数据源结构 (`bicep.usd`)

```
/character
├── /bone
│   ├── /L_scapula/L_scapulaShape   (Mesh, 1096 verts, 静态骨骼)
│   ├── /L_radius/L_radiusShape     (Mesh, 393 verts, 动态骨骼)
│   └── /L_humerus/L_humerusShape   (Mesh, 948 verts, 静态骨骼)
└── /muscle
    └── /bicep                       (TetMesh, 1251 verts, 984 tets)
                                      primvars: materialW, tendonmask, muscle_id, ...
                                      surfaceFaceVertexIndices: 1834 (表面三角形)
```

### 已有能力

| 组件 | 能做什么 | 缺什么 |
|------|---------|--------|
| `UsdIO.read()` | 读 Mesh + TetMesh → `UsdMesh`(含 vertices, faces, tets, prim_path) | 不读 primvar 属性 (fiber, tendonmask) |
| `UsdIO.start()` + `set_points()` | 按 prim_path 写变形顶点到 layer | 无 |
| `UsdIO.set_custom()` | 写任意标量属性到任意 prim | 不能写 transform (xformOp) |
| `muscle_warp.load_mesh_usd()` | 读 TetMesh + fiber + tendonmask + 全部 primvar | 独立于 UsdIO，不保留 prim path |
| `muscle_warp.load_bone_from_usd()` | 读骨骼 Mesh + 生成 muscle_id 映射 | 独立于 UsdIO |
| `SolverMuscleBoneCoupledWarp` | Warp 刚柔耦合仿真 | 无导出功能 |

### 核心 gap

1. **UsdIO 不读 primvar** — `UsdMesh` 只有 vertices/faces/tets，没有 fiber/tendonmask 等 PBD 所需属性
2. **muscle_warp 不保留 prim path** — 肌肉加载后丢失了原始 prim path（如 `/character/muscle/bicep`）
3. **骨骼是平摊大数组** — 多个骨骼 mesh 合并成一个大 bone_pos，丢失了每块骨骼的 prim path
4. **无 xformOp 写入** — UsdIO 的 `set_custom()` 理论上能写，但没有封装好 transform 写入

## 设计方案

### 数据流总览

```
bicep.usd ──UsdIO.read()──→ meshes[] (各自带 prim_path)
    │                             │
    │                    ┌────────┴────────┐
    │                    │ 肌肉 TetMesh     │ 骨骼 Mesh(es)
    │                    │ prim_path 已知   │ prim_path 已知
    │                    └────────┬────────┘
    │                             │
    │                   MuscleSim + Newton Model
    │                   (Warp PBD)  (刚体动力学)
    │                             │
    │                    每帧仿真 step()
    │                             │
    │              ┌──────────────┼──────────────┐
    │              │              │              │
    │      肌肉变形顶点    骨骼 body_q     自定义数据
    │     sim.pos.numpy()  state.body_q  activation, torque
    │              │              │              │
    │              ▼              ▼              ▼
    └──UsdIO.start()──→ output/bicep.anim.usda (layer)
              set_points()    set_xform()    set_runtime()
```

### 关键设计决策

1. **UsdIO 作为唯一 USD 入口** — 所有 mesh 从 UsdIO 读取，保留 prim_path 映射
2. **UsdMesh 扩展** — 给 `UsdMesh` 增加 `primvars: dict` 字段存储 primvar 数据
3. **建立 prim_path → 仿真数据映射** — 通过 UsdMesh.mesh_path 追溯原始 prim
4. **UsdIO 增加 `set_xform()`** — 封装刚体 transform 写入（translate + orient）

## 实现步骤

### Step 1: 扩展 UsdMesh 数据结构

**文件**: `src/VMuscle/usd_io.py`

在 `UsdMesh` 中增加 `primvars` 字段：

```python
@dataclass
class UsdMesh:
    mesh_path: str
    vertices: np.ndarray          # (N, 3) float32
    faces: np.ndarray             # (T, 3) int32
    color: ColorRgb = (0.7, 0.7, 0.7)
    tets: np.ndarray | None = None
    primvars: dict | None = None  # NEW: {"materialW": ndarray, "tendonmask": ndarray, ...}
```

### Step 2: UsdIO.read() 读取 primvar

**文件**: `src/VMuscle/usd_io.py`

在 `read()` 方法中，遍历 prim 的 primvar 并存入 `UsdMesh.primvars`：

```python
# 在 read() 中已有的 meshes.append(UsdMesh(...)) 之前：
pv_api = UsdGeom.PrimvarsAPI(prim)
primvars = {}
for pv in pv_api.GetPrimvars():
    name = pv.GetPrimvarName()
    if name in ("displayColor", "displayOpacity"):
        continue
    val = pv.Get()
    if val is not None:
        primvars[name] = np.asarray(val)

meshes.append(UsdMesh(..., primvars=primvars if primvars else None))
```

### Step 3: UsdIO 增加 set_xform() 方法

**文件**: `src/VMuscle/usd_io.py`

```python
def set_xform(self, prim_path: str, pos: np.ndarray, quat_wxyz: np.ndarray,
              *, frame: int | None = None) -> bool:
    """Write rigid body transform to a prim.

    Args:
        pos: Translation [x, y, z].
        quat_wxyz: Quaternion [w, x, y, z] (USD convention).
    """
    prim = self._stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return False

    from pxr import UsdGeom
    xf = UsdGeom.Xformable(prim)
    # Ensure xformOp order exists
    ops = xf.GetOrderedXformOps()
    if not ops:
        xf.AddTranslateOp()
        xf.AddOrientOp()
        ops = xf.GetOrderedXformOps()

    t = frame if frame is not None else self._Sdf.TimeCode.Default()
    if len(ops) >= 1:
        ops[0].Set(self._Gf.Vec3d(*pos.astype(float)), t)
    if len(ops) >= 2:
        q = quat_wxyz.astype(float)
        ops[1].Set(self._Gf.Quatf(q[0], q[1], q[2], q[3]), t)

    if frame is not None:
        self._mark_frame(frame)
    return True
```

### Step 4: 构建 prim_path 映射辅助

**文件**: `src/VMuscle/usd_io.py` (UsdIO 上新增方法)

```python
def find_mesh(self, keyword: str) -> UsdMesh | None:
    """Find first mesh whose prim_path contains keyword."""
    for m in self.meshes:
        if keyword in m.mesh_path:
            return m
    return None

def find_meshes(self, keyword: str) -> list[UsdMesh]:
    """Find all meshes whose prim_path contains keyword."""
    return [m for m in self.meshes if keyword in m.mesh_path]
```

### Step 5: 创建耦合导出示例

**文件**: `examples/example_couple_usd_export.py`

使用 Warp 版 MuscleSim + Newton + UsdIO，完整流程：

```python
"""Rigid-flexible coupling with layered USD export.

Reads muscle + bone from a single USD, runs Warp PBD coupled with Newton
rigid body, writes deformed vertices and bone transforms to a layered USD.
"""
import numpy as np
import warp as wp
import newton
import newton.examples
from VMuscle.usd_io import UsdIO, usd_args
from VMuscle.muscle_warp import MuscleSim, load_config
from VMuscle.solver_muscle_bone_coupled_warp import SolverMuscleBoneCoupledWarp

class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 60
        self.sim_time = 0.0
        self.sim_frame = 0

        # --- 1. 从 USD 读取所有 mesh（保留 prim_path） ---
        self.usd = UsdIO(args.usd_path, y_up_to_z_up=False).read()

        # 按 prim_path 分离肌肉和骨骼
        self.muscle_mesh = self.usd.find_mesh("muscle")
        self.bone_meshes = self.usd.find_meshes("bone")

        # --- 2. 初始化 Warp MuscleSim（传入 UsdMesh 数据） ---
        cfg = load_config("data/muscle/config/bicep.json")
        cfg.gui = False
        cfg.render_mode = None
        sim = MuscleSim(cfg)

        # --- 3. 构建 Newton 刚体模型 ---
        # ... (build elbow model from bone_meshes)

        # --- 4. 耦合 solver ---
        # self.solver = SolverMuscleBoneCoupledWarp(model, sim, ...)

        # --- 5. 开始 layered USD 写入 ---
        self.usd.start(args.output_path)
        self.usd.set_runtime("fps", self.fps)

        # Warp mesh data for viewer rendering
        self.warp_data = self.usd.warp_mesh_data()

        # 空 Newton model for viewer camera
        builder = newton.ModelBuilder()
        self.model = builder.finalize()
        self.state_0 = self.model.state()
        viewer.set_model(self.model)

    def step(self):
        # --- 仿真 (先用简单的 translate 代替) ---
        for wd in self.warp_data:
            # 简单测试：所有顶点向上移动
            pos_np = wd.pos.numpy()
            pos_np[:, 1] += 0.001  # Y-up
            wd.pos = wp.array(pos_np, dtype=wp.vec3)

        self.sim_frame += 1
        self.sim_time += 1.0 / self.fps

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        for wd in self.warp_data:
            self.viewer.log_mesh(wd.name, wd.pos, wd.tri_indices)
        self.viewer.end_frame()

        # --- 写入 layered USD ---
        for wd in self.warp_data:
            self.usd.set_points(wd.name, wd.pos, frame=self.sim_frame)

    def close(self):
        self.usd.close()
```

### Step 6: 接入真正的仿真（后续）

后续把 Step 5 中的 `step()` 替换为真正的耦合仿真：

```python
def step(self):
    self.solver.step(self.state_0, self.state_1, dt=1.0/60.0)
    self.state_0, self.state_1 = self.state_1, self.state_0

    # 肌肉变形 → set_points
    muscle_verts = self.sim.pos.numpy()
    self.usd.set_points(self.muscle_mesh.mesh_path, muscle_verts, frame=self.sim_frame)

    # 骨骼 transform → set_xform
    for bone_mesh in self.bone_meshes:
        body_id = self.bone_body_map[bone_mesh.mesh_path]
        body_q = self.state_0.body_q.numpy()[body_id]
        pos = body_q[:3]
        quat_xyzw = body_q[3:7]
        quat_wxyz = np.array([quat_xyzw[3], *quat_xyzw[:3]])
        self.usd.set_xform(bone_mesh.mesh_path, pos, quat_wxyz, frame=self.sim_frame)

    # 自定义数据
    self.usd.set_runtime("activation", self.cfg.activation, frame=self.sim_frame)
    self.usd.set_runtime("torque", float(np.linalg.norm(self.solver._muscle_torque)), frame=self.sim_frame)
```

## 输出文件结构

```
output/bicep.anim.usda (layer)
├── sublayer → data/muscle/model/bicep.usd (原始模型，不修改)
├── /character/muscle/bicep
│   └── points (time-sampled, 每帧变形顶点)
├── /character/bone/L_radius
│   ├── xformOp:translate (time-sampled)
│   └── xformOp:orient (time-sampled)
└── /anim/runtime
    ├── fps = 60
    ├── activation (time-sampled)
    └── torque (time-sampled)
```

## 实施顺序

| 序号 | 内容 | 改动文件 | 预期效果 |
|------|------|---------|---------|
| 1 | UsdMesh 加 primvars 字段 | `usd_io.py` | 读取到 fiber/tendonmask 等属性 |
| 2 | UsdIO.read() 读 primvar | `usd_io.py` | primvar 数据随 UsdMesh 一起返回 |
| 3 | UsdIO 加 set_xform() | `usd_io.py` | 能写刚体 transform 到 layer |
| 4 | UsdIO 加 find_mesh/find_meshes | `usd_io.py` | 按关键字查找 prim |
| 5 | 写 IO-only 示例 (仿真用 translate 代替) | `examples/example_couple_usd_export.py` | 验证完整 read → write 流程 |
| 6 | 接入真实仿真 (后续) | 示例文件 | 完整刚柔耦合 + 导出 |

## 注意事项

- 全程 **纯 Warp**，不依赖 Taichi
- 使用 `--viewer usd` 或 `--viewer null` 运行（headless 环境）
- `y_up_to_z_up=False` 保持 Y-up（与 Houdini 一致），写回时无需坐标转换
- 骨骼的 xformOp 写在 Xform 层（如 `/character/bone/L_radius`），不在 Shape 层
- primvar 写入暂不需要（fiber/tendonmask 是静态属性，已在源 USD 中）
