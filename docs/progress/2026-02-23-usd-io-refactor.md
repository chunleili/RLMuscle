# 2026-02-23: usd_io.py 重构

- **1143 → 553 行 (-52%)**，删除 931 行，新增 339 行
- 用 `newton.usd.get_mesh()` 加载 mesh（pxr fallback），不再自写三角化主路径
- primvar 读取：3 个函数合并为 `_read_primvars()`（基于 `PrimvarsAPI.GetAuthoredPrimvars()`）
- primvar 写入：保留 `set_primvar` / `set_display_color`，直接操作 `PrimvarsAPI`
- 删除：`UsdMeshSet` 中间类、Rodrigues 旋转公式（→ 硬编码 `_Y_TO_Z`）、`_mesh_color_from_path` 启发式、`_USE_NEWTON_USD_GET_MESH` 全局标记
- `UsdIO` 删除 `up_axis` 参数（从 `y_up_to_z_up` 推导）；`usd_args` 加默认值、修正 `__all__`
- 数值验证：新旧代码 vertices/faces/tets bit-exact 一致
