"""Tests for Millard 2012 muscle curve module.

Validates:
1. Quintic Bezier control point computation
2. Power-basis conversion accuracy
3. Active f_L and passive f_PE curve values at key points
4. Energy integral consistency (F' ≈ f)
5. Comparison with DGF curves
6. Equilibrium solver
"""

import numpy as np
import pytest

from VMuscle.millard_curves import (
    MillardCurves,
    _bernstein_to_power_basis,
    _horner_eval,
    _newton_find_u,
    build_active_fl_bezier,
    build_passive_fpe_bezier,
    millard_equilibrium_fiber_length,
    quintic_bezier_corner_control_points,
    scale_curviness,
)


class TestScaleCurviness:
    def test_range(self):
        assert scale_curviness(0.0) == pytest.approx(0.1)
        assert scale_curviness(1.0) == pytest.approx(0.9)
        assert scale_curviness(0.5) == pytest.approx(0.5)


class TestQuinticBezierControlPoints:
    def test_simple_corner(self):
        """A simple 45-degree corner should produce valid control points."""
        xPts, yPts = quintic_bezier_corner_control_points(
            0.0, 0.0, 1.0,   # start: (0,0) slope=1
            1.0, 0.0, -1.0,  # end: (1,0) slope=-1
            0.5,              # curviness
        )
        assert len(xPts) == 6
        assert len(yPts) == 6
        # Endpoints must match
        assert xPts[0] == pytest.approx(0.0)
        assert xPts[5] == pytest.approx(1.0)
        assert yPts[0] == pytest.approx(0.0)
        assert yPts[5] == pytest.approx(0.0)
        # Doubled control points (indices 1=2, 3=4)
        assert xPts[1] == pytest.approx(xPts[2])
        assert yPts[1] == pytest.approx(yPts[2])
        assert xPts[3] == pytest.approx(xPts[4])
        assert yPts[3] == pytest.approx(yPts[4])


class TestBernsteinToPowerBasis:
    def test_identity(self):
        """B(u) = u should give coeffs [0, 1, 0, 0, 0, 0]."""
        # For B(u) = u, the Bezier control points are [0, 1/5, 2/5, 3/5, 4/5, 1]
        pts = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        coeffs = _bernstein_to_power_basis(pts)
        assert coeffs[0] == pytest.approx(0.0, abs=1e-14)
        assert coeffs[1] == pytest.approx(1.0, abs=1e-14)
        for k in range(2, 6):
            assert coeffs[k] == pytest.approx(0.0, abs=1e-14)

    def test_constant(self):
        """B(u) = 3.0 should give coeffs [3, 0, 0, 0, 0, 0]."""
        pts = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
        coeffs = _bernstein_to_power_basis(pts)
        assert coeffs[0] == pytest.approx(3.0, abs=1e-14)
        for k in range(1, 6):
            assert coeffs[k] == pytest.approx(0.0, abs=1e-14)

    def test_evaluation_matches(self):
        """Power-basis evaluation must match Bernstein evaluation."""
        pts = np.array([1.0, 2.5, 0.3, 4.1, -0.5, 3.0])
        coeffs = _bernstein_to_power_basis(pts)

        for u in np.linspace(0, 1, 20):
            # Bernstein evaluation
            t = u
            s = 1 - t
            B = [s**5, 5*t*s**4, 10*t**2*s**3, 10*t**3*s**2, 5*t**4*s, t**5]
            val_bernstein = sum(pts[j] * B[j] for j in range(6))
            # Power-basis evaluation
            val_power = _horner_eval(u, coeffs)
            assert val_power == pytest.approx(val_bernstein, abs=1e-10)


class TestActiveForceLengthCurve:
    @pytest.fixture
    def mc(self):
        return MillardCurves()

    def test_peak_at_optimal(self, mc):
        """f_L(1.0) must be 1.0 (peak isometric force)."""
        assert mc.fl.eval(1.0) == pytest.approx(1.0, abs=1e-6)

    def test_shoulder_values(self, mc):
        """f_L at shoulders should equal ylow=0.1."""
        assert mc.fl.eval(0.4441) == pytest.approx(0.1, abs=1e-4)
        assert mc.fl.eval(1.8123) == pytest.approx(0.1, abs=1e-4)

    def test_monotonic_ascending(self, mc):
        """f_L should be monotonically increasing on [x0, 1.0]."""
        x = np.linspace(0.4441, 1.0, 100)
        y = mc.fl.eval(x)
        assert np.all(np.diff(y) >= -1e-10)

    def test_monotonic_descending(self, mc):
        """f_L should be monotonically decreasing on [1.0, x3]."""
        x = np.linspace(1.0, 1.8123, 100)
        y = mc.fl.eval(x)
        assert np.all(np.diff(y) <= 1e-10)

    def test_extrapolation(self, mc):
        """Outside domain, should extrapolate linearly to ylow."""
        assert mc.fl.eval(0.3) == pytest.approx(0.1, abs=1e-6)
        assert mc.fl.eval(2.0) == pytest.approx(0.1, abs=1e-6)

    def test_segment_count(self, mc):
        """Should have exactly 5 segments."""
        assert len(mc.fl.segments) == 5


class TestPassiveForceLengthCurve:
    @pytest.fixture
    def mc(self):
        return MillardCurves()

    def test_zero_at_rest(self, mc):
        """f_PE(1.0) must be 0 (no passive force at rest length)."""
        assert mc.fpe.eval(1.0) == pytest.approx(0.0, abs=1e-6)

    def test_one_at_iso_strain(self, mc):
        """f_PE(1.7) must be 1.0 (unit force at iso strain)."""
        assert mc.fpe.eval(1.7) == pytest.approx(1.0, abs=1e-4)

    def test_monotonic_increasing(self, mc):
        """f_PE should be monotonically increasing on [1.0, 1.7]."""
        x = np.linspace(1.0, 1.7, 100)
        y = mc.fpe.eval(x)
        assert np.all(np.diff(y) >= -1e-10)

    def test_zero_below_rest(self, mc):
        """f_PE should be 0 for lambda <= 1.0."""
        for lm in [0.5, 0.8, 0.99]:
            assert mc.fpe.eval(lm) == pytest.approx(0.0, abs=1e-6)

    def test_segment_count(self, mc):
        """Should have exactly 2 segments."""
        assert len(mc.fpe.segments) == 2


class TestEnergyIntegral:
    @pytest.fixture
    def mc(self):
        return MillardCurves()

    def test_integral_derivative_consistency_fl(self, mc):
        """d/dx [∫f_L dx] ≈ f_L (numerical derivative)."""
        eps = 1e-7
        for lm in [0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8]:
            F_plus = mc.fl.eval_integral(lm + eps)
            F_minus = mc.fl.eval_integral(lm - eps)
            numerical_deriv = (F_plus - F_minus) / (2 * eps)
            analytical = mc.fl.eval(lm)
            assert numerical_deriv == pytest.approx(analytical, abs=1e-5), (
                f"Integral derivative mismatch at lm={lm}"
            )

    def test_integral_derivative_consistency_fpe(self, mc):
        """d/dx [∫f_PE dx] ≈ f_PE (numerical derivative)."""
        eps = 1e-7
        for lm in [1.1, 1.3, 1.5, 1.7]:
            F_plus = mc.fpe.eval_integral(lm + eps)
            F_minus = mc.fpe.eval_integral(lm - eps)
            numerical_deriv = (F_plus - F_minus) / (2 * eps)
            analytical = mc.fpe.eval(lm)
            assert numerical_deriv == pytest.approx(analytical, abs=1e-5)

    def test_integral_at_start_is_zero(self, mc):
        """Integral at domain start should be ~0."""
        assert mc.fl.eval_integral(mc.fl.x_lo) == pytest.approx(0.0, abs=1e-10)
        assert mc.fpe.eval_integral(mc.fpe.x_lo) == pytest.approx(0.0, abs=1e-10)

    def test_integral_positive_beyond_rest(self, mc):
        """Active and passive energy integrals should be positive for lm > x_lo."""
        for lm in [0.5, 0.8, 1.0, 1.5]:
            assert mc.fl.eval_integral(lm) >= -1e-10
        for lm in [1.1, 1.5]:
            assert mc.fpe.eval_integral(lm) >= -1e-10


class TestEquilibriumSolver:
    def test_basic_equilibrium(self):
        mc = MillardCurves()
        # At full activation, ascending limb: find where f_L = 0.5
        lm_eq = millard_equilibrium_fiber_length(1.0, 0.5, mc.fl)
        assert mc.fl.eval(lm_eq) == pytest.approx(0.5, abs=1e-4)

    def test_zero_load(self):
        mc = MillardCurves()
        # Zero load should converge to near x_lo (minimum length)
        lm_eq = millard_equilibrium_fiber_length(1.0, 0.0, mc.fl)
        assert lm_eq == pytest.approx(mc.fl.x_lo, abs=1e-3)


class TestDGFComparison:
    """Compare Millard curves with DGF for qualitative consistency."""

    def test_fl_peak_same_location(self):
        """Both Millard and DGF should peak at lm ≈ 1.0."""
        from VMuscle.dgf_curves import active_force_length as dgf_fl

        mc = MillardCurves()
        lm = np.linspace(0.4, 1.8, 500)
        dgf_vals = dgf_fl(lm)
        mill_vals = mc.fl.eval(lm)

        dgf_peak = lm[np.argmax(dgf_vals)]
        mill_peak = lm[np.argmax(mill_vals)]
        assert abs(dgf_peak - mill_peak) < 0.05

    def test_fl_general_shape(self):
        """Millard f_L should be broadly similar to DGF f_L."""
        from VMuscle.dgf_curves import active_force_length as dgf_fl

        mc = MillardCurves()
        # At optimal length, both should be near 1.0
        assert dgf_fl(1.0) == pytest.approx(1.0, abs=0.05)
        assert mc.fl.eval(1.0) == pytest.approx(1.0, abs=0.05)

    def test_fpe_general_shape(self):
        """Millard f_PE should be broadly similar to DGF f_PE."""
        from VMuscle.dgf_curves import passive_force_length as dgf_fpe

        mc = MillardCurves()
        # At lm=1.0, both should be ~0
        assert dgf_fpe(1.0) == pytest.approx(0.0, abs=0.05)
        assert mc.fpe.eval(1.0) == pytest.approx(0.0, abs=0.05)
        # At lm=1.5, both should have significant passive force
        assert dgf_fpe(1.5) > 0.1
        assert mc.fpe.eval(1.5) > 0.1
