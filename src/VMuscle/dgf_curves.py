"""DeGroote-Fregly 2016 muscle curve functions (NumPy).

Pure-NumPy implementations of the DGF Hill-type muscle curves, matching
the Warp kernels in newton/_src/solvers/vbd/vmuscle_kernels.py.

References:
    DeGroote et al. (2016), "Evaluation of Direct Collocation Optimal Control
    Problem Formulations for Solving the Muscle Redundancy Problem"
"""

import math

import numpy as np

# Active force-length: 3 Gaussian-like terms
_b11, _b21, _b31, _b41 = 0.815, 1.055, 0.162, 0.063
_b12, _b22, _b32, _b42 = 0.433, 0.717, -0.030, 0.200
_b13, _b23, _b33 = 0.1, 1.0, 0.354

# Passive force-length
_KPE = 4.0
_E0 = 0.6
_LM_MIN = 0.2

# Force-velocity
_D1 = -0.3211346127989808
_D2 = -8.149
_D3 = -0.374
_D4 = 0.8825327733249912


def active_force_length(lm_tilde):
    """DGF 2016 active force-length curve f_L(l~). Vectorized."""
    lm = np.asarray(lm_tilde, dtype=float)
    EPS = 1e-6
    d1 = np.maximum(np.abs(_b31 + _b41 * lm), EPS)
    g1 = _b11 * np.exp(-0.5 * ((lm - _b21) / d1) ** 2)
    d2 = np.maximum(np.abs(_b32 + _b42 * lm), EPS)
    g2 = _b12 * np.exp(-0.5 * ((lm - _b22) / d2) ** 2)
    g3 = _b13 * np.exp(-0.5 * ((lm - _b23) / _b33) ** 2)
    return g1 + g2 + g3


def passive_force_length(lm_tilde):
    """DGF 2016 passive force-length curve f_PE(l~). Vectorized."""
    lm = np.asarray(lm_tilde, dtype=float)
    offset = np.exp(_KPE * (_LM_MIN - 1.0) / _E0)
    denom = np.exp(_KPE) - offset
    result = (np.exp(np.clip(_KPE * (lm - 1.0) / _E0, -50, 50)) - offset) / denom
    return np.maximum(result, 0.0)


def force_velocity(v_norm):
    """DGF 2016 force-velocity curve f_V(v~). Vectorized."""
    v = np.asarray(v_norm, dtype=float)
    x = _D2 * v + _D3
    return _D1 * np.log(x + np.sqrt(x ** 2 + 1.0)) + _D4


def active_force_length_scalar(lm_tilde: float) -> float:
    """DGF 2016 active force-length curve (scalar, no numpy overhead)."""
    EPS = 1e-6
    d1 = max(abs(_b31 + _b41 * lm_tilde), EPS)
    g1 = _b11 * math.exp(-0.5 * ((lm_tilde - _b21) / d1) ** 2)
    d2 = max(abs(_b32 + _b42 * lm_tilde), EPS)
    g2 = _b12 * math.exp(-0.5 * ((lm_tilde - _b22) / d2) ** 2)
    g3 = _b13 * math.exp(-0.5 * ((lm_tilde - _b23) / _b33) ** 2)
    return g1 + g2 + g3


def passive_force_length_scalar(lm_tilde: float) -> float:
    """DGF 2016 passive force-length curve (scalar, no numpy overhead)."""
    offset = math.exp(_KPE * (_LM_MIN - 1.0) / _E0)
    denom = math.exp(_KPE) - offset
    arg = max(-50.0, min(50.0, _KPE * (lm_tilde - 1.0) / _E0))
    return max((math.exp(arg) - offset) / denom, 0.0)


def force_velocity_scalar(v_norm: float) -> float:
    """DGF 2016 force-velocity curve (scalar, no numpy overhead)."""
    x = _D2 * v_norm + _D3
    return _D1 * math.log(x + math.sqrt(x * x + 1.0)) + _D4


def compute_fiber_forces(stretches, activation, v_norm=0.0, include_passive=True):
    """Compute normalized fiber forces from stretches and activation.

    Args:
        stretches: Per-tet normalized fiber lengths (N,).
        activation: Scalar activation level [0, 1].
        v_norm: Scalar normalized contraction velocity (for f_V).

    Returns:
        dict with f_active, f_passive, f_total (mean over tets), f_velocity.
    """
    fl = active_force_length(stretches)
    fpe = passive_force_length(stretches) if include_passive else np.zeros_like(stretches)
    fv = float(force_velocity(np.clip(v_norm, -1.0, 1.0)))
    f_active = activation * fl * fv
    f_total = f_active + fpe
    return {
        'f_active': float(f_active.mean()),
        'f_passive': float(fpe.mean()),
        'f_total': float(f_total.mean()),
        'f_velocity': fv,
    }
