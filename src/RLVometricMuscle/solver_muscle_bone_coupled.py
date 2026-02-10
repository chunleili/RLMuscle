# src/RLVometricMuscle/coupling.py

import inspect

from newton.solvers import SolverFeatherstone
from .solver_volumetric_muscle import SolverVolumetricMuscle


class SolverMuscleBoneCoupled:
    def __init__(self, model, **solver_kwargs):
        self.model = model
        bone_sig = inspect.signature(SolverFeatherstone.__init__)
        bone_kwargs = {
            k: v for k, v in solver_kwargs.items() if k in bone_sig.parameters and k != "model"
        }
        self.bone_solver = SolverFeatherstone(model, **bone_kwargs)
        self.muscle_solver = SolverVolumetricMuscle(model, **solver_kwargs)

    def step(self, state_in, state_out, control, contacts, dt):      
        if control is None:
            control = self.model.control(clone_variables=False)

        self.muscle_solver.step(state_in, state_out, control, contacts, dt)
        self.bone_solver.step(state_in, state_out, control, contacts, dt)
