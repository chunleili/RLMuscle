"""Regression test suite. Wraps existing standalone test scripts as pytest cases.

Usage:
    uv run pytest tests/test_regression.py -v
    uv run pytest tests/test_regression.py -m "not slow" -v
    uv run pytest tests/test_regression.py -m "not cuda" -v
"""
import pytest
from pathlib import Path


# --- Numerical tests (Taichi vs Warp) ---

@pytest.mark.slow
def test_jacobi_numerical_consistency():
    """Warp vs Taichi, Jacobi mode, max error < 1e-3."""
    from test_muscle_warp_vs_taichi import run_comparison
    max_err = run_comparison(n_steps=50, use_jacobi=True)
    assert max_err < 1e-3, f"Jacobi max error {max_err} >= 1e-3"


@pytest.mark.slow
def test_gauss_seidel_numerical_consistency():
    """Warp vs Taichi, Gauss-Seidel mode, max error < 0.1."""
    from test_muscle_warp_vs_taichi import run_comparison
    max_err = run_comparison(n_steps=50, use_jacobi=False)
    assert max_err < 0.1, f"GS max error {max_err} >= 0.1"


# --- CUDA stability tests ---

@pytest.mark.cuda
def test_cuda_jacobi_stable():
    """CUDA Jacobi: 50 steps, no NaN."""
    from test_warp_cuda_jacobi import run_test
    result = run_test(n_steps=50)
    assert result["n_nan"] == 0, f"CUDA Jacobi produced {result['n_nan']} NaN vertices"
    assert result["passed"], "CUDA Jacobi test did not pass"


@pytest.mark.cuda
def test_cpu_vs_cuda_no_nan():
    """CPU vs CUDA: no NaN on CUDA side."""
    from test_warp_cpu_vs_cuda import run_test
    result = run_test()
    assert result["cuda_nans"] == 0, f"CUDA side has {result['cuda_nans']} NaN vertices"
    assert result["passed"], "CPU vs CUDA test did not pass"


# --- Visual comparison tests ---

@pytest.mark.slow
@pytest.mark.visual
def test_visual_comparison_generates_images():
    """Generate comparison PNGs, verify files exist and no all-NaN frames."""
    from test_visual_comparison import run_test
    result = run_test(snapshot_steps=[1, 50, 100])

    # Check images were generated
    assert len(result["image_paths"]) > 0, "No images generated"
    for img_path in result["image_paths"]:
        p = Path(img_path)
        assert p.exists(), f"Image not found: {img_path}"
        assert p.stat().st_size > 1024, f"Image too small (likely blank): {img_path}"

    # Check no all-NaN frames
    import math
    for step, err in result["errors_per_step"].items():
        assert not math.isnan(err), f"Step {step} has all-NaN vertices"

    assert result["passed"], "Visual comparison test did not pass"
