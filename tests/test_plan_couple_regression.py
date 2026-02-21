"""Regression checks for the lightweight coupling benchmark."""

from __future__ import annotations

import unittest

from tests.plan_couple_sim import SOLVERS, SimConfig, run_sim


class TestPlanCoupleRegression(unittest.TestCase):
    def _cfg(self, **kwargs: float | str) -> SimConfig:
        base = dict(
            mode="hybrid",
            solver=SOLVERS["mujoco"],
            activation_profile="pulse",
            mass=1.2,
            inertia=0.32,
            stiffness=7.2,
            damping=6.8,
            dt=1.0 / 180.0,
            steps=2400,
        )
        base.update(kwargs)
        return SimConfig(**base)

    def test_torque_is_unstable_vs_hybrid(self) -> None:
        torque = run_sim(self._cfg(mode="torque"))
        hybrid = run_sim(self._cfg(mode="hybrid"))
        self.assertGreater(torque["peak_theta"], hybrid["peak_theta"] * 5.0)
        self.assertGreater(torque["mae"], hybrid["mae"] * 5.0)

    def test_inertia_increase_reduces_peak(self) -> None:
        low_inertia = run_sim(self._cfg(inertia=0.24))
        high_inertia = run_sim(self._cfg(inertia=0.32))
        self.assertLess(high_inertia["peak_theta"], low_inertia["peak_theta"])

    def test_hybrid_tail_better_than_weak(self) -> None:
        hybrid = run_sim(self._cfg(mode="hybrid"))
        weak = run_sim(self._cfg(mode="weak"))
        self.assertLess(hybrid["tail_mae"], weak["tail_mae"])

    def test_step_has_higher_mae_than_pulse(self) -> None:
        step = run_sim(self._cfg(activation_profile="step"))
        pulse = run_sim(self._cfg(activation_profile="pulse"))
        self.assertGreater(step["mae"], pulse["mae"])


if __name__ == "__main__":
    unittest.main()
