# src/RLVometricMuscle/coupling.py

from newton.solvers import SolverFeatherstone
from .solver_volumetric_muscle import SolverVolumetricMuscle


class SolverMuscleBoneCoupled:
    def __init__(self, model, **solver_kwargs):
        self.model = model
        self.bone_solver = SolverFeatherstone(model, **solver_kwargs)
        self.muscle_solver = SolverVolumetricMuscle(model, **solver_kwargs)  

    def step(self, state_in, state_out, control, contacts, dt):      
        if control is None:
            control = self.model.control(clone_variables=False)

        self.muscle_solver.step(state_in, state_out, dt, contacts, control)
        self.bone_solver.step(state_in, state_out, control, contacts, dt)
