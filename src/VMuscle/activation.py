"""DGF activation dynamics for volumetric muscle.

Provides both Warp kernel and NumPy implementations of the first-order
activation dynamics ODE from DeGroote-Fregly 2016.
"""

import numpy as np
import warp as wp


@wp.func
def dgf_activation_dynamics(excitation: float, activation: float, dt: float) -> float:
    """One step of DGF activation dynamics (implicit Euler).

    da/dt = f(e, a) * (e - a)
    where f = (0.5+0.5*tanh(b*(e-a)))/(tau_a*(0.5+1.5*a))
            + (0.5-0.5*tanh(b*(e-a)))*(0.5+1.5*a)/tau_d
    """
    tau_a = 0.015  # activation time constant [s]
    tau_d = 0.060  # deactivation time constant [s]
    b = 10.0  # smoothing factor

    t = wp.tanh(b * (excitation - activation))
    f_act = (0.5 + 0.5 * t) / (tau_a * (0.5 + 1.5 * activation))
    f_deact = (0.5 - 0.5 * t) * (0.5 + 1.5 * activation) / tau_d
    da_dt = (f_act + f_deact) * (excitation - activation)

    a_new = activation + dt * da_dt
    return wp.clamp(a_new, 0.01, 1.0)


@wp.kernel
def update_activations(
    excitations: wp.array(dtype=wp.float32),
    activations: wp.array(dtype=wp.float32),
    dt: float,
):
    """Update all tet activations from excitation signals."""
    tid = wp.tid()
    activations[tid] = dgf_activation_dynamics(excitations[tid], activations[tid], dt)


def activation_dynamics_step_np(
    excitation: np.ndarray,
    activation: np.ndarray,
    dt: float,
    tau_act: float = 0.015,
    tau_deact: float = 0.060,
) -> np.ndarray:
    """NumPy version of first-order activation dynamics."""
    b = 10.0
    t = np.tanh(b * (excitation - activation))
    f_act = (0.5 + 0.5 * t) / (tau_act * (0.5 + 1.5 * activation))
    f_deact = (0.5 - 0.5 * t) * (0.5 + 1.5 * activation) / tau_deact
    da_dt = (f_act + f_deact) * (excitation - activation)
    a_new = activation + dt * da_dt
    return np.clip(a_new, 0.01, 1.0)
