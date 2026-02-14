from pathlib import Path
import numpy as np
import warp as wp
from .muscle import SimConfig, MuscleSim, load_config


def _resolve_data_path(path: Path | None) -> Path | None:
    if path is None:
        return None

    p = Path(path)
    candidates = [p]
    p_str = p.as_posix()

    if p_str.startswith("data/model/"):
        suffix = p_str[len("data/model/"):]
        candidates.append(Path("data/muscle/model") / suffix)
    if p_str.startswith("data/animation/"):
        suffix = p_str[len("data/animation/"):]
        candidates.append(Path("data/muscle/animation") / suffix)

    for cand in candidates:
        if cand.exists():
            return cand
    return p

class SolverVolumetricMuscle:
    def __init__(self, model, **kwargs):
        self.model = model
        config_path = Path(kwargs.get("config_path", "data/muscle/config/bicep.json"))
        if config_path.exists():
            cfg = load_config(config_path)
        else:
            cfg = SimConfig()

        if "geo_path" in kwargs:
            cfg.geo_path = Path(kwargs["geo_path"]) if kwargs["geo_path"] else None
        if "bone_geo_path" in kwargs:
            cfg.bone_geo_path = Path(kwargs["bone_geo_path"]) if kwargs["bone_geo_path"] else None
        if "bone_animation_path" in kwargs:
            cfg.bone_animation_path = Path(kwargs["bone_animation_path"]) if kwargs["bone_animation_path"] else None
        if "constraints" in kwargs and kwargs["constraints"] is not None:
            cfg.constraints = kwargs["constraints"]
        if "muscle_arch" in kwargs and kwargs["muscle_arch"] is not None:
            cfg.arch = kwargs["muscle_arch"]

        cfg.geo_path = _resolve_data_path(cfg.geo_path)
        cfg.bone_geo_path = _resolve_data_path(cfg.bone_geo_path)
        cfg.bone_animation_path = _resolve_data_path(cfg.bone_animation_path)

        cfg.gui = False
        cfg.render_mode = None
        cfg.save_image = False
        cfg.show_auxiliary_meshes = False

        self.core = MuscleSim(cfg)
        model_particle_count = getattr(model, "particle_count", 0)
        if model_particle_count and model_particle_count != self.core.n_verts:
            raise ValueError(
                f"Model particle_count ({model_particle_count}) does not match muscle mesh vertices ({self.core.n_verts})."
            )

    def step(self, state_in, state_out, control, contacts, dt):
        self.core.cfg.dt = dt

        if state_in.particle_q is not None and state_in.particle_qd is not None:
            self.core.pos.from_numpy(state_in.particle_q.numpy().astype(np.float32))
            self.core.vel.from_numpy(state_in.particle_qd.numpy().astype(np.float32))

        if control is not None and getattr(control, "tet_activations", None) is not None:
            self.core.activation.from_numpy(control.tet_activations.numpy().astype(np.float32))
        elif control is not None and getattr(control, "muscle_activations", None) is not None:
            muscle_act = control.muscle_activations.numpy()
            if muscle_act.size > 0:
                self.core.activation.fill(float(muscle_act[0]))

        # TODO: enable muscle simulation when particle data is wired up
        # self.core.step()

        if state_out.particle_q is not None and state_out.particle_qd is not None:
            state_out.particle_q.assign(wp.array(self.core.pos.to_numpy(), dtype=wp.vec3, device=self.model.device))
            state_out.particle_qd.assign(wp.array(self.core.vel.to_numpy(), dtype=wp.vec3, device=self.model.device))
