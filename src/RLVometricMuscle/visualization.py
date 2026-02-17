from __future__ import annotations

"""Minimal viewer helpers with fixed defaults.

Public API (2 only):
- focus_camera_on_points(viewer, points)
- ViewerVisualization(viewer, wide_points, focus_points=None)
"""

from typing import Any

import numpy as np
import warp as wp


def _as_points(points: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected {name} shape (N,3), got {arr.shape}.")
    return arr


def _bbox_min_max(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return points.min(axis=0), points.max(axis=0)


def _camera_angles_from_look_dir(viewer: Any, look_dir: np.ndarray) -> tuple[float, float]:
    d = look_dir / (np.linalg.norm(look_dir) + 1.0e-12)
    up_axis = int(getattr(getattr(viewer, "camera", None), "up_axis", 2))

    if up_axis == 0:  # X-up
        pitch = np.degrees(np.arcsin(np.clip(d[0], -1.0, 1.0)))
        yaw = np.degrees(np.arctan2(d[2], d[1]))
    elif up_axis == 1:  # Y-up
        pitch = np.degrees(np.arcsin(np.clip(d[1], -1.0, 1.0)))
        yaw = np.degrees(np.arctan2(d[2], d[0]))
    else:  # Z-up
        pitch = np.degrees(np.arcsin(np.clip(d[2], -1.0, 1.0)))
        yaw = np.degrees(np.arctan2(d[1], d[0]))
    return float(pitch), float(yaw)


def _set_camera_wide(viewer: Any, points: np.ndarray) -> None:
    if not hasattr(viewer, "set_camera"):
        return

    bbox_min, bbox_max = _bbox_min_max(points)
    center = 0.5 * (bbox_min + bbox_max)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    dist = max(1.2, 2.0 * diag)

    cam_pos = np.array(
        [center[0] + dist, center[1] + 0.35 * dist, center[2] + 0.4 * dist],
        dtype=np.float32,
    )
    look_dir = center - cam_pos
    pitch, yaw = _camera_angles_from_look_dir(viewer, look_dir)
    viewer.set_camera(pos=wp.vec3(*cam_pos.tolist()), pitch=pitch, yaw=yaw)

    camera = getattr(viewer, "camera", None)
    if camera is not None and hasattr(camera, "fov"):
        camera.fov = 45.0


def _make_origin_gizmo(viewer: Any, points: np.ndarray) -> tuple[Any, Any, Any]:
    bbox_min, bbox_max = _bbox_min_max(points)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    L = max(0.1, 0.4 * diag)

    starts = np.zeros((3, 3), dtype=np.float32)
    ends = np.array([[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]], dtype=np.float32)
    colors = np.array([[1.0, 0.2, 0.2], [0.2, 1.0, 0.2], [0.2, 0.4, 1.0]], dtype=np.float32)

    device = getattr(viewer, "device", None)
    return (
        wp.array(starts, dtype=wp.vec3, device=device),
        wp.array(ends, dtype=wp.vec3, device=device),
        wp.array(colors, dtype=wp.vec3, device=device),
    )


def focus_camera_on_points(viewer: Any, points: np.ndarray) -> None:
    """Refocus camera on points while preserving view direction when possible."""

    if not hasattr(viewer, "set_camera"):
        return

    pts = _as_points(points, "points")
    bbox_min, bbox_max = _bbox_min_max(pts)
    center = 0.5 * (bbox_min + bbox_max)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    camera = getattr(viewer, "camera", None)

    if camera is not None and hasattr(camera, "get_front"):
        front = np.asarray(camera.get_front(), dtype=np.float32)
        front_norm = float(np.linalg.norm(front))
        if front_norm > 1.0e-6:
            front = front / front_norm
            fov = float(np.clip(float(getattr(camera, "fov", 45.0)), 15.0, 90.0))
            tan_half = max(np.tan(np.radians(0.5 * fov)), 1.0e-3)
            dist = max(0.6, 0.75 * diag / tan_half)
            cam_pos = center - front * dist
            viewer.set_camera(
                pos=wp.vec3(*cam_pos.tolist()),
                pitch=float(getattr(camera, "pitch", 0.0)),
                yaw=float(getattr(camera, "yaw", 0.0)),
            )
            return

    _set_camera_wide(viewer, pts)


class ViewerVisualization:
    """Fixed-default visualization helper with minimal surface area."""

    _FOCUS_HOTKEY = "f"
    _ORIGIN_WIDTH = 0.01

    def __init__(
        self,
        viewer: Any,
        wide_points: np.ndarray,
        focus_points: np.ndarray | None = None,
    ):
        self.viewer = viewer
        self._wide_points = _as_points(wide_points, "wide_points")
        self._focus_points = self._wide_points if focus_points is None else _as_points(focus_points, "focus_points")
        self._origin_starts, self._origin_ends, self._origin_colors = _make_origin_gizmo(viewer, self._wide_points)
        self._prev_focus_key_down = False
        self._is_first_frame = True

        _set_camera_wide(self.viewer, self._wide_points)
        focus_camera_on_points(self.viewer, self._focus_points)

    def update_focus_hotkey(self) -> None:
        key_down = bool(self.viewer.is_key_down(self._FOCUS_HOTKEY)) if hasattr(self.viewer, "is_key_down") else False
        if key_down and not self._prev_focus_key_down:
            focus_camera_on_points(self.viewer, self._focus_points)
        self._prev_focus_key_down = key_down

    def log_debug_visuals(self) -> None:
        self.viewer.log_lines(
            "/debug/origin_axes",
            self._origin_starts,
            self._origin_ends,
            self._origin_colors,
            width=self._ORIGIN_WIDTH,
            hidden=False,
        )

    def handle_post_frame(self) -> None:
        if not self._is_first_frame:
            return
        renderer = getattr(self.viewer, "renderer", None)
        if renderer is not None and hasattr(renderer, "draw_shadows"):
            renderer.draw_shadows = False
        self._is_first_frame = False
        
__all__ = [
    "ViewerVisualization",
    "focus_camera_on_points",
]
