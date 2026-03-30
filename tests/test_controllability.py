import numpy as np

from VMuscle.activation import activation_dynamics_step_np
from VMuscle.controllability import (
    ActivationController,
    activation_to_gain,
    build_coupling_config,
    shape_torque_target,
)


def test_activation_dynamics_step_can_decay_to_zero():
    activation = np.array([1.0], dtype=np.float32)
    excitation = np.array([0.0], dtype=np.float32)
    for _ in range(240):
        activation = activation_dynamics_step_np(
            excitation,
            activation,
            1.0 / 120.0,
            tau_act=0.03,
            tau_deact=0.12,
            min_activation=0.0,
        )
    assert activation[0] < 1e-3


def test_activation_controller_is_bounded_and_releases_to_zero():
    cfg = build_coupling_config("smooth_nonlinear")
    controller = ActivationController(cfg)

    values = [controller.step(0.3, 1.0 / 60.0) for _ in range(30)]
    assert values[0] >= 0.0
    assert values[-1] <= 0.3 + 1e-6
    assert abs(values[-1] - 0.3) < 1e-3

    for _ in range(240):
        released = controller.step(0.0, 1.0 / 60.0)
    assert released == 0.0


def test_rate_limited_torque_shaping_limits_step_change():
    cfg = build_coupling_config("rate_limited", max_torque=100.0, torque_slew_rate=12.0, torque_ema=1.0)
    prev = np.zeros(3, dtype=np.float32)
    raw = np.array([50.0, 0.0, 0.0], dtype=np.float32)
    shaped = shape_torque_target(raw, 1.0, prev, 0.1, cfg)
    assert np.isclose(np.linalg.norm(shaped), 1.2, atol=1e-4)


def test_torque_shaping_respects_max_torque_clip():
    cfg = build_coupling_config("linear_tuned", max_torque=2.0, torque_slew_rate=0.0, torque_ema=1.0)
    shaped = shape_torque_target(np.array([10.0, 0.0, 0.0], dtype=np.float32), 1.0, np.zeros(3, dtype=np.float32), 0.1, cfg)
    assert np.isclose(np.linalg.norm(shaped), 2.0, atol=1e-4)


def test_activation_to_gain_is_monotonic_for_supported_modes():
    levels = np.linspace(0.0, 1.0, num=11)
    for preset in ("linear_tuned", "smooth_nonlinear", "rate_limited"):
        cfg = build_coupling_config(preset)
        gains = [activation_to_gain(level, cfg) for level in levels]
        assert np.all(np.diff(gains) >= -1e-6)
