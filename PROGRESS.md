# 下一步

- 在 CUDA 上做完整的 Taichi vs Warp 可视化对比（当前 Taichi 在同进程下有 LLVM 崩溃问题，需分进程运行）
- 骨骼-肌肉耦合（`reaction_accum`）Warp 版尚未移植

# 已完成

## 2026-02-23: usd_io.py 重构

- **1143 → 553 行 (-52%)**，删除 931 行，新增 339 行
- 用 `newton.usd.get_mesh()` 加载 mesh（pxr fallback），不再自写三角化主路径
- primvar 读取：3 个函数合并为 `_read_primvars()`（基于 `PrimvarsAPI.GetAuthoredPrimvars()`）
- primvar 写入：保留 `set_primvar` / `set_display_color`，直接操作 `PrimvarsAPI`
- 删除：`UsdMeshSet` 中间类、Rodrigues 旋转公式（→ 硬编码 `_Y_TO_Z`）、`_mesh_color_from_path` 启发式、`_USE_NEWTON_USD_GET_MESH` 全局标记
- `UsdIO` 删除 `up_axis` 参数（从 `y_up_to_z_up` 推导）；`usd_args` 加默认值、修正 `__all__`
- 数值验证：新旧代码 vertices/faces/tets bit-exact 一致

## 2026-02-21: Warp 移植审查与修复

- CPU Jacobi 与 Taichi 数值一致 (max_err < 2.3e-4)；CUDA GS 发散已修复
- `use_jacobi` 默认改 True；GS 分支 6 处改 `wp.atomic_add`
- 缺失: `reaction_accum`（骨骼耦合）、`Visualizer` 类

# 测试命令

```bash
uv run pytest -v                    # 一键全跑
uv run pytest -m "not slow" -v      # 跳过慢测试
uv run pytest -m "not cuda" -v      # 跳过 CUDA 测试
```
