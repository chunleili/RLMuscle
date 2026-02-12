from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp


def _bbox_min_max(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return points.min(axis=0), points.max(axis=0)


def _camera_angles_from_look_dir(viewer: Any, look_dir: np.ndarray) -> tuple[float, float]:
    d = look_dir / (np.linalg.norm(look_dir) + 1.0e-12)
    up_axis = getattr(getattr(viewer, "camera", None), "up_axis", 2)

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


def init_origin_gizmo_arrays(
    viewer: Any,
    axis_length: float,
    enabled: bool,
) -> tuple[Any | None, Any | None, Any | None]:
    if not enabled:
        return None, None, None

    starts = np.zeros((3, 3), dtype=np.float32)
    L = float(axis_length)
    ends = np.array(
        [
            [L, 0.0, 0.0],
            [0.0, L, 0.0],
            [0.0, 0.0, L],
        ],
        dtype=np.float32,
    )
    colors = np.array(
        [
            [1.0, 0.2, 0.2],
            [0.2, 1.0, 0.2],
            [0.2, 0.4, 1.0],
        ],
        dtype=np.float32,
    )

    device = getattr(viewer, "device", None)
    return (
        wp.array(starts, dtype=wp.vec3, device=device),
        wp.array(ends, dtype=wp.vec3, device=device),
        wp.array(colors, dtype=wp.vec3, device=device),
    )


def set_camera_wide(
    viewer: Any,
    focus_points: np.ndarray,
) -> None:
    if not hasattr(viewer, "set_camera"):
        return

    bbox_min, bbox_max = _bbox_min_max(focus_points)
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

    if hasattr(viewer, "camera") and hasattr(viewer.camera, "fov"):
        viewer.camera.fov = 45.0


def focus_camera_on_points(
    viewer: Any,
    focus_points: np.ndarray,
    focus_keep_view: bool = True,
) -> None:
    if not hasattr(viewer, "set_camera"):
        return

    bbox_min, bbox_max = _bbox_min_max(focus_points)
    center = 0.5 * (bbox_min + bbox_max)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    camera = getattr(viewer, "camera", None)

    if focus_keep_view and camera is not None:
        front = np.array(camera.get_front(), dtype=np.float32)
        front_norm = float(np.linalg.norm(front))
        if front_norm > 1.0e-6:
            front = front / front_norm
            fov = float(getattr(camera, "fov", 45.0))
            fov = float(np.clip(fov, 15.0, 90.0))
            tan_half = max(np.tan(np.radians(0.5 * fov)), 1.0e-3)
            dist = max(0.6, 0.75 * diag / tan_half)
            cam_pos = center - front * dist
            viewer.set_camera(
                pos=wp.vec3(*cam_pos.tolist()),
                pitch=float(camera.pitch),
                yaw=float(camera.yaw),
            )
            return

    up_axis = int(getattr(camera, "up_axis", 2)) if camera is not None else 2
    side_axis = (up_axis + 1) % 3
    cam_pos = center.copy()
    cam_pos[up_axis] += max(0.5, 1.2 * diag)
    cam_pos[side_axis] += 0.35 * max(diag, 0.2)
    look_dir = center - cam_pos
    pitch, yaw = _camera_angles_from_look_dir(viewer, look_dir)
    viewer.set_camera(pos=wp.vec3(*cam_pos.tolist()), pitch=pitch, yaw=yaw)


@dataclass(frozen=True)
class ViewerVisualConfig:
    show_origin_gizmo: bool = False
    focus_keep_view: bool = True
    orbit_around_focus: bool = False


def viewer_visual_config_from(
    source: Any,
    *,
    orbit_around_focus: bool = False,
) -> ViewerVisualConfig:
    return ViewerVisualConfig(
        show_origin_gizmo=bool(getattr(source, "show_origin_gizmo", False)),
        focus_keep_view=bool(getattr(source, "focus_keep_view", True)),
        orbit_around_focus=bool(orbit_around_focus),
    )


class ViewerVisualization:
    """High-level visualization controller for Newton viewer demos."""

    _FOCUS_HOTKEY = "f"
    _ORIGIN_LINE_WIDTH = 0.01

    def __init__(self, viewer: Any, config: ViewerVisualConfig):
        self.viewer = viewer
        self.config = config

        self._wide_points: np.ndarray | None = None
        self._focus_points: np.ndarray | None = None
        self._focus_center: np.ndarray | None = None
        self._orbit_distance: float | None = None
        self._origin_starts: Any | None = None
        self._origin_ends: Any | None = None
        self._origin_colors: Any | None = None

        self._is_first_frame = True
        self._prev_focus_key_down = False

        if self.config.orbit_around_focus:
            self._register_orbit_callback()

    def set_points(self, wide_points: np.ndarray, focus_points: np.ndarray | None = None) -> None:
        wide = np.asarray(wide_points, dtype=np.float32)
        if wide.ndim != 2 or wide.shape[1] != 3:
            raise ValueError(f"Expected wide_points shape (N,3), got {wide.shape}.")

        if focus_points is None:
            focus = wide
        else:
            focus = np.asarray(focus_points, dtype=np.float32)
            if focus.ndim != 2 or focus.shape[1] != 3:
                raise ValueError(f"Expected focus_points shape (N,3), got {focus.shape}.")

        self._wide_points = wide
        self._focus_points = focus
        self._focus_center = self._compute_points_center(focus)
        self._refresh_orbit_distance()

    def setup(
        self,
        *,
        wide_points: np.ndarray,
        focus_points: np.ndarray | None = None,
        origin_axis_length: float | None = None,
    ) -> None:
        self.set_points(wide_points=wide_points, focus_points=focus_points)
        self._origin_starts, self._origin_ends, self._origin_colors = init_origin_gizmo_arrays(
            viewer=self.viewer,
            axis_length=float(origin_axis_length or 0.0),
            enabled=self.config.show_origin_gizmo,
        )
        if self._wide_points is not None:
            set_camera_wide(
                viewer=self.viewer,
                focus_points=self._wide_points,
            )
        self.focus_camera_on_model()

    def focus_camera_on_model(self) -> None:
        if self._focus_points is None:
            return
        focus_camera_on_points(
            viewer=self.viewer,
            focus_points=self._focus_points,
            focus_keep_view=self.config.focus_keep_view,
        )
        self._refresh_orbit_distance()

    def update_focus_hotkey(self) -> None:
        key = self._FOCUS_HOTKEY
        key_down = bool(self.viewer.is_key_down(key)) if hasattr(self.viewer, "is_key_down") else False
        if key_down and not self._prev_focus_key_down:
            self.focus_camera_on_model()
        self._prev_focus_key_down = key_down
        if self.config.orbit_around_focus:
            self._refresh_orbit_distance()

    def log_debug_visuals(self) -> None:
        if self.config.show_origin_gizmo and self._origin_starts is not None:
            self.viewer.log_lines(
                "/debug/origin_axes",
                self._origin_starts,
                self._origin_ends,
                self._origin_colors,
                width=self._ORIGIN_LINE_WIDTH,
                hidden=False,
            )

    def handle_post_frame(self) -> None:
        if not self._is_first_frame:
            return
        if hasattr(self.viewer, "renderer") and hasattr(self.viewer.renderer, "draw_shadows"):
            self.viewer.renderer.draw_shadows = False
        self._is_first_frame = False

    @staticmethod
    def _compute_points_center(points: np.ndarray) -> np.ndarray:
        bbox_min, bbox_max = _bbox_min_max(points)
        return (0.5 * (bbox_min + bbox_max)).astype(np.float32)

    def _get_camera_pos(self) -> np.ndarray | None:
        camera = getattr(self.viewer, "camera", None)
        if camera is None or not hasattr(camera, "pos"):
            return None
        return np.asarray(camera.pos, dtype=np.float32)

    def _refresh_orbit_distance(self) -> None:
        if not self.config.orbit_around_focus or self._focus_center is None:
            return
        cam_pos = self._get_camera_pos()
        if cam_pos is None:
            return
        self._orbit_distance = float(np.linalg.norm(cam_pos - self._focus_center))

    def _register_orbit_callback(self) -> None:
        renderer = getattr(self.viewer, "renderer", None)
        if renderer is not None and hasattr(renderer, "register_mouse_drag"):
            renderer.register_mouse_drag(self._on_mouse_drag_orbit)

    def _on_mouse_drag_orbit(self, x: float, y: float, dx: float, dy: float, buttons: int, modifiers: int) -> None:
        del x, y, dx, dy, modifiers
        if not self.config.orbit_around_focus:
            return
        if self._focus_center is None:
            return
        if hasattr(self.viewer, "ui") and self.viewer.ui is not None and self.viewer.ui.is_capturing():
            return

        left_mask = 1
        try:
            import pyglet  # noqa: PLC0415

            left_mask = int(pyglet.window.mouse.LEFT)
        except Exception:
            pass
        if (int(buttons) & left_mask) == 0:
            return

        camera = getattr(self.viewer, "camera", None)
        if camera is None:
            return
        front = np.asarray(camera.get_front(), dtype=np.float32)
        front_norm = float(np.linalg.norm(front))
        if front_norm <= 1.0e-6:
            return
        front = front / front_norm

        if self._orbit_distance is None:
            self._refresh_orbit_distance()
        dist = max(0.1, float(self._orbit_distance or 1.0))
        cam_pos = self._focus_center - front * dist

        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(*cam_pos.tolist()), pitch=float(camera.pitch), yaw=float(camera.yaw))
        else:
            camera.pos = type(camera.pos)(*cam_pos.tolist())
