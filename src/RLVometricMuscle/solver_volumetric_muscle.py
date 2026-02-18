import numpy as np
import warp as wp

import newton
from newton._src.sim.model import Model
from newton._src.sim.state import State
from newton._src.sim.contacts import Contacts
from newton._src.sim.control import Control
from newton.solvers import SolverBase
from .muscle import MuscleSim, load_config

# Y-up → Z-up rotation matrix (90° around X)
_R_Y2Z = np.array([[1, 0, 0],
                    [0, 0, -1],
                    [0, 1, 0]], dtype=np.float32)


class SolverVolumetricMuscle(SolverBase):
    def __init__(
        self,
        model: Model,
    ):
        super().__init__(model=model)

        cfg = load_config("data/muscle/config/bicep.json")
        cfg.gui = False
        cfg.render_mode = None
        self.core = MuscleSim(cfg)

        # Coordinate transform (set via set_coord_transform)
        self._center_shift = np.zeros(3, dtype=np.float32)
        self._has_coord_transform = False
        self._activation = 0.0  # default from bicep.json; nonzero values need stable constraint tuning

    def set_coord_transform(self, center_shift: np.ndarray):
        """Enable Y-up ↔ Z-up coordinate conversion for Newton integration.

        Args:
            center_shift: shift applied by UsdIO centering, shape (3,).
        """
        self._center_shift = np.asarray(center_shift, dtype=np.float32)
        self._has_coord_transform = True

    def update_bone_positions(self, indices: np.ndarray, positions: np.ndarray):
        """Update specific bone vertices in MuscleSim's bone_pos_field.

        Args:
            indices: vertex indices into bone_pos_field, shape (N,)
            positions: new world positions in MuscleSim Y-up coordinate system, shape (N, 3)
        """
        if not hasattr(self.core, "bone_pos_field"):
            return
        for i, idx in enumerate(indices):
            self.core.bone_pos_field[int(idx)] = positions[i].tolist()

    def _yup_to_zup_centered(self, pos_yup: np.ndarray) -> np.ndarray:
        """Convert positions from Y-up (.geo) to Z-up centered (Newton)."""
        pos_zup = pos_yup @ _R_Y2Z.T
        return pos_zup - self._center_shift

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control = None,
        contacts: Contacts = None,
        dt: float = 0.0):

        # MuscleSim maintains its own state in Y-up .geo coordinates.
        # Do NOT overwrite from state_in.particle_q (which is Z-up centered).
        # Use MuscleSim's configured dt (cfg.dt / cfg.num_substeps), NOT Newton's dt,
        # because MuscleSim.step() internally runs num_substeps sub-iterations.

        self.core.activation.fill(self._activation)
        # Run PBD substeps manually: move update_velocities() outside the loop
        # to prevent velocity feedback from amplifying constraint corrections.
        # In standard XPBD, velocities are computed once from total position change.
        self.core.update_attach_targets()
        for _ in range(self.core.cfg.num_substeps):
            self.core.integrate()
            self.core.clear()
            self.core.solve_constraints()
            if self.core.use_jacobi:
                self.core.apply_dP()
        self.core.update_velocities()

        # Write back muscle positions to Newton state for visualization.
        pos_yup = self.core.pos.to_numpy().reshape(-1, 3)
        vel_yup = self.core.vel.to_numpy().reshape(-1, 3)

        if self._has_coord_transform:
            pos_out = self._yup_to_zup_centered(pos_yup)
            vel_out = vel_yup @ _R_Y2Z.T
        else:
            pos_out = pos_yup
            vel_out = vel_yup

        state_out.particle_q = wp.from_numpy(pos_out, dtype=wp.vec3f, device=self.model.device)
        state_out.particle_qd = wp.from_numpy(vel_out, dtype=wp.vec3f, device=self.model.device)
        state_out.particle_f = wp.zeros_like(state_out.particle_q)
