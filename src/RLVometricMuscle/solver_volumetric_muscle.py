class SolverVolumetricMuscle:
    def __init__(self, model, **kwargs):
        self.model = model
        self.step_cnt = 0

    def step(self, state_in, state_out, control, contacts, dt):
        # Placeholder for muscle dynamics computation
        print(f"VolumetricMuscle solver step {self.step_cnt} with dt={dt}")
        self.step_cnt += 1
        pass
