"""Smoke tests for XPBD coupled SimpleArm Stage 3."""
import numpy as np
import pytest


@pytest.fixture(scope="module")
def xpbd_result():
    """Run XPBD simulation once for all tests (30 steps, CPU)."""
    from examples.example_xpbd_coupled_simple_arm import (
        xpbd_coupled_simple_arm, load_config)
    cfg = load_config("data/simpleArm/config.json")
    cfg["solver"]["n_steps"] = 30
    cfg["solver"]["arch"] = "cpu"
    return xpbd_coupled_simple_arm(cfg, verbose=False)


def test_xpbd_completes(xpbd_result):
    """XPBD simulation completes 30 steps without crash."""
    assert xpbd_result is not None
    assert len(xpbd_result["times"]) == 30


def test_xpbd_no_nan(xpbd_result):
    """No NaN in any output array."""
    assert not np.any(np.isnan(xpbd_result["elbow_angles"]))
    assert not np.any(np.isnan(xpbd_result["forces"]))
    assert not np.any(np.isnan(xpbd_result["norm_fiber_lengths"]))


def test_xpbd_force_reasonable(xpbd_result):
    """Forces stay in physically reasonable range [0, 2*F_max]."""
    F_max = xpbd_result["max_iso_force"]
    assert np.all(xpbd_result["forces"] >= 0)
    assert np.all(xpbd_result["forces"] <= F_max * 2.0)


def test_xpbd_fiber_stretch_physiological(xpbd_result):
    """Normalized fiber lengths are in physiological range [0.3, 2.0]."""
    nfl = xpbd_result["norm_fiber_lengths"]
    assert np.all(nfl > 0.3), f"min nfl={nfl.min():.4f}"
    assert np.all(nfl < 2.0), f"max nfl={nfl.max():.4f}"
