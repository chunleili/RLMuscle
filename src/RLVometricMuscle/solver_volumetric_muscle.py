from pathlib import Path

import numpy as np
import warp as wp

from .muscle_core import MuscleCore, SimConfig, load_config


def _resolve_data_path(path: Path | None) -> Path | None:
    if path is None:
        return None

    p = Path(path)
    candidates = [p]
    p_str = p.as_posix()

    if p_str.startswith("data/model/"):
        suffix = p_str[len("data/model/"):]
        candidates.append(Path("data/muscle/model") / suffix)

    for cand in candidates:
        if cand.exists():
            return cand
    return p


class SolverVolumetricMuscle:
    def __init__(self, model, **kwargs):
        self.model = model
        config_path = Path(kwargs.get("config_path", "data/muscle/config/bicep.json"))
        cfg = load_config(config_path) if config_path.exists() else SimConfig()

        if "geo_path" in kwargs:
            cfg.geo_path = Path(kwargs["geo_path"]) if kwargs["geo_path"] else None
        if "muscle_arch" in kwargs and kwargs["muscle_arch"] is not None:
            # keep API compatibility: Warp backend is now the default path.
            _ = kwargs["muscle_arch"]

        cfg.geo_path = _resolve_data_path(cfg.geo_path)
        self.core = MuscleCore(cfg)

        model_particle_count = getattr(model, "particle_count", 0)
        if model_particle_count and model_particle_count != self.core.n_verts:
            raise ValueError(
                f"Model particle_count ({model_particle_count}) does not match muscle mesh vertices ({self.core.n_verts})."
            )

    def step(self, state_in, state_out, control, contacts, dt):
        if state_in.particle_q is not None and state_in.particle_qd is not None:
            self.core.set_state(state_in.particle_q.numpy(), state_in.particle_qd.numpy())

        if control is not None and getattr(control, "tet_activations", None) is not None:
            self.core.set_activation_from_tets(control.tet_activations.numpy())
        elif control is not None and getattr(control, "muscle_activations", None) is not None:
            muscle_act = control.muscle_activations.numpy()
            if muscle_act.size > 0:
                self.core.set_activation_from_tets(np.array([float(muscle_act[0])], dtype=np.float32))

        self.core.step(dt)

        if state_out.particle_q is not None and state_out.particle_qd is not None:
            state_out.particle_q.assign(wp.array(self.core.pos.numpy(), dtype=wp.vec3, device=self.model.device))
            state_out.particle_qd.assign(wp.array(self.core.vel.numpy(), dtype=wp.vec3, device=self.model.device))
