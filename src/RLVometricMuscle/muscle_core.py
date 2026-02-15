from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import warp as wp


@dataclass
class SimConfig:
    name: str = "MuscleSimWarp"
    geo_path: Path | None = Path("data/muscle/model/bicep.geo")
    dt: float = 1e-3
    gravity: float = -9.8
    density: float = 1000.0
    veldamping: float = 0.02
    activation: float = 0.3
    stiffness: float = 25.0
    activation_gain: float = 10.0


def load_config(path: Path) -> SimConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    path_fields = {"geo_path"}
    kwargs = {}
    for fld in fields(SimConfig):
        name = fld.name
        if name in data:
            value = data[name]
            if name in path_fields:
                kwargs[name] = Path(value) if value else None
            else:
                kwargs[name] = value
    return SimConfig(**kwargs)


def load_mesh_geo(path: Path):
    from .geo import Geo

    geo = Geo(str(path))
    positions = np.asarray(geo.positions, dtype=np.float32)
    tets = np.asarray(geo.vert, dtype=np.int32)
    fibers = np.asarray(geo.materialW, dtype=np.float32) if hasattr(geo, "materialW") else None
    tendon_mask = np.asarray(geo.tendonmask, dtype=np.float32) if hasattr(geo, "tendonmask") else None
    return positions, tets, fibers, tendon_mask


def load_mesh(path: Path | None):
    if path is None:
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        tets = np.array([[0, 2, 1, 3]], dtype=np.int32)
        fibers = np.array([[0.0, 1.0, 0.0]] * 4, dtype=np.float32)
        tendon_mask = np.zeros((4,), dtype=np.float32)
        return positions, tets, fibers, tendon_mask

    if str(path).endswith(".geo"):
        return load_mesh_geo(path)
    raise ValueError(f"Unsupported mesh file format: {path}")


@wp.kernel
def _integrate_particles(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    rest_pos: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=wp.float32),
    fiber: wp.array(dtype=wp.vec3),
    activation: wp.array(dtype=wp.float32),
    dt: float,
    gravity_y: float,
    damping: float,
    stiffness: float,
    activation_gain: float,
):
    tid = wp.tid()
    if inv_mass[tid] <= 0.0:
        vel[tid] = wp.vec3(0.0, 0.0, 0.0)
        pos[tid] = rest_pos[tid]
        return

    p = pos[tid]
    v = vel[tid]
    p0 = rest_pos[tid]

    force = wp.vec3(0.0, gravity_y, 0.0) / inv_mass[tid]
    force += stiffness * (p0 - p)

    f = fiber[tid]
    f_norm = wp.length(f)
    if f_norm > 1e-6:
        force += activation_gain * activation[tid] * (f / f_norm)

    v = v + dt * force
    v = v * (1.0 - damping)
    pos[tid] = p + dt * v
    vel[tid] = v


class MuscleCore:
    """Lightweight warp-based volumetric muscle core for solver coupling."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        positions, tets, fibers, _ = load_mesh(cfg.geo_path)

        self.n_verts = int(positions.shape[0])
        self.n_tets = int(tets.shape[0])

        if fibers is None or fibers.shape[0] != self.n_verts:
            fibers = np.zeros((self.n_verts, 3), dtype=np.float32)
            fibers[:, 1] = 1.0

        self._activation_host = np.full((self.n_tets,), float(cfg.activation), dtype=np.float32)
        self._activation_vertex_host = np.full((self.n_verts,), float(cfg.activation), dtype=np.float32)

        mass = np.full((self.n_verts,), 1.0 / max(cfg.density, 1.0), dtype=np.float32)

        self.rest_pos = wp.array(positions, dtype=wp.vec3)
        self.pos = wp.array(positions, dtype=wp.vec3)
        self.vel = wp.zeros(self.n_verts, dtype=wp.vec3)
        self.inv_mass = wp.array(mass, dtype=wp.float32)
        self.fiber = wp.array(fibers.astype(np.float32), dtype=wp.vec3)
        self.activation = wp.array(self._activation_host, dtype=wp.float32)
        self.activation_vertex = wp.array(self._activation_vertex_host, dtype=wp.float32)
        self.tets = wp.array(tets.astype(np.int32), dtype=wp.vec4i)

    def set_state(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        if positions.shape[0] != self.n_verts:
            raise ValueError("Muscle vertex count mismatch while setting state.")
        self.pos = wp.array(positions.astype(np.float32), dtype=wp.vec3)
        self.vel = wp.array(velocities.astype(np.float32), dtype=wp.vec3)

    def set_activation_from_tets(self, tet_activation: np.ndarray) -> None:
        if tet_activation.size == 0:
            return
        if tet_activation.shape[0] != self.n_tets:
            value = float(np.mean(tet_activation))
            self._activation_host.fill(value)
        else:
            self._activation_host[:] = tet_activation.astype(np.float32)

        value = float(np.mean(self._activation_host))
        self._activation_vertex_host.fill(value)

        self.activation = wp.array(self._activation_host, dtype=wp.float32)
        self.activation_vertex = wp.array(self._activation_vertex_host, dtype=wp.float32)

    def step(self, dt: float) -> None:
        wp.launch(
            kernel=_integrate_particles,
            dim=self.n_verts,
            inputs=[
                self.pos,
                self.vel,
                self.rest_pos,
                self.inv_mass,
                self.fiber,
                self.activation_vertex,
                float(dt),
                float(self.cfg.gravity),
                float(self.cfg.veldamping),
                float(self.cfg.stiffness),
                float(self.cfg.activation_gain),
            ],
        )
