from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pxr import Sdf, Usd, UsdSkel


def get_skeleton_from_prim_path(stage: "Usd.Stage", prim_path: str | "Sdf.Path") -> "UsdSkel.Skeleton":
    """根据 USD stage 和 prim path 获取 kinematic tree（UsdSkel.Skeleton）。"""
    from pxr import Usd, UsdSkel

    if stage is None:
        raise ValueError("`stage` 不能为空。")

    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim or not prim.IsValid():
        raise ValueError(f"在 stage 中找不到 prim: {prim_path}")

    # 1) path 直接指向 skeleton
    if prim.IsA(UsdSkel.Skeleton):
        return UsdSkel.Skeleton(prim)

    # 2) 从绑定关系读取 skeleton（常见于 mesh / skinned prim）
    binding_api = UsdSkel.BindingAPI(prim)
    skeleton_rel = binding_api.GetSkeletonRel()
    if skeleton_rel:
        targets = skeleton_rel.GetTargets()
        if targets:
            skel_prim = stage.GetPrimAtPath(targets[0])
            if skel_prim and skel_prim.IsValid() and skel_prim.IsA(UsdSkel.Skeleton):
                return UsdSkel.Skeleton(skel_prim)

    # 3) 在该 prim 子树中查找第一个 skeleton
    for child in Usd.PrimRange(prim):
        if child.IsA(UsdSkel.Skeleton):
            return UsdSkel.Skeleton(child)

    raise ValueError(f"未在 prim `{prim_path}` 及其绑定/子树中找到 UsdSkel.Skeleton。")
