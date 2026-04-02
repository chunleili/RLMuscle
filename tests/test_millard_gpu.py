"""Test Millard curve GPU evaluation via Warp."""
import numpy as np
import warp as wp

from VMuscle.millard_curves import MillardCurves
from VMuscle.muscle_warp import millard_eval_wp, millard_energy_eval_wp

wp.init()


@wp.kernel
def test_millard_eval_kernel(
    x_vals: wp.array(dtype=wp.float32),
    y_out: wp.array(dtype=wp.float32),
    xc: wp.array(dtype=wp.float32),
    yc: wp.array(dtype=wp.float32),
    bounds: wp.array(dtype=wp.float32),
    n_seg: int,
    x_lo: float, x_hi: float,
    y_lo: float, y_hi: float,
    dydx_lo: float, dydx_hi: float,
):
    i = wp.tid()
    y_out[i] = millard_eval_wp(float(x_vals[i]), xc, yc, bounds, n_seg,
                                x_lo, x_hi, y_lo, y_hi, dydx_lo, dydx_hi)


def _upload_curve(curve):
    n_seg = len(curve.segments)
    x_c = np.zeros((n_seg, 6), dtype=np.float32)
    y_c = np.zeros((n_seg, 6), dtype=np.float32)
    F_c = np.zeros((n_seg, 11), dtype=np.float32)
    bounds = np.zeros(n_seg + 1, dtype=np.float32)
    for i, seg in enumerate(curve.segments):
        x_c[i] = seg.x_coeffs.astype(np.float32)
        y_c[i] = seg.y_coeffs.astype(np.float32)
        F_c[i] = seg.F_coeffs.astype(np.float32)
        bounds[i] = np.float32(seg.x_start)
    bounds[n_seg] = np.float32(curve.segments[-1].x_end)
    return (wp.from_numpy(x_c.flatten(), dtype=wp.float32),
            wp.from_numpy(y_c.flatten(), dtype=wp.float32),
            wp.from_numpy(F_c.flatten(), dtype=wp.float32),
            wp.from_numpy(bounds, dtype=wp.float32),
            n_seg)


def test_gpu_fl_evaluation():
    mc = MillardCurves()
    fl_xc, fl_yc, fl_Fc, fl_bounds, fl_n = _upload_curve(mc.fl)

    test_lm = np.array([0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8], dtype=np.float32)
    x_arr = wp.from_numpy(test_lm, dtype=wp.float32)
    y_arr = wp.zeros(len(test_lm), dtype=wp.float32)

    wp.launch(test_millard_eval_kernel, dim=len(test_lm), inputs=[
        x_arr, y_arr, fl_xc, fl_yc, fl_bounds, fl_n,
        mc.fl.x_lo, mc.fl.x_hi, mc.fl.y_lo, mc.fl.y_hi,
        mc.fl.dydx_lo, mc.fl.dydx_hi,
    ])
    wp.synchronize()

    gpu_vals = y_arr.numpy()
    cpu_vals = np.array([mc.fl.eval(float(x)) for x in test_lm], dtype=np.float32)

    print("=== GPU vs CPU: f_L ===")
    for i in range(len(test_lm)):
        print(f"  lm={test_lm[i]:.1f}: GPU={gpu_vals[i]:.6f}  CPU={cpu_vals[i]:.6f}  "
              f"err={abs(gpu_vals[i]-cpu_vals[i]):.2e}")

    max_err = np.max(np.abs(gpu_vals - cpu_vals))
    print(f"  MaxErr: {max_err:.2e}")
    assert max_err < 1e-3, f"GPU/CPU mismatch: {max_err}"


def test_gpu_fpe_evaluation():
    mc = MillardCurves()
    fpe_xc, fpe_yc, fpe_Fc, fpe_bounds, fpe_n = _upload_curve(mc.fpe)

    test_lm = np.array([0.8, 1.0, 1.1, 1.3, 1.5, 1.7, 1.9, 2.0], dtype=np.float32)
    x_arr = wp.from_numpy(test_lm, dtype=wp.float32)
    y_arr = wp.zeros(len(test_lm), dtype=wp.float32)

    wp.launch(test_millard_eval_kernel, dim=len(test_lm), inputs=[
        x_arr, y_arr, fpe_xc, fpe_yc, fpe_bounds, fpe_n,
        mc.fpe.x_lo, mc.fpe.x_hi, mc.fpe.y_lo, mc.fpe.y_hi,
        mc.fpe.dydx_lo, mc.fpe.dydx_hi,
    ])
    wp.synchronize()

    gpu_vals = y_arr.numpy()
    cpu_vals = np.array([mc.fpe.eval(float(x)) for x in test_lm], dtype=np.float32)

    print("=== GPU vs CPU: f_PE ===")
    for i in range(len(test_lm)):
        print(f"  lm={test_lm[i]:.1f}: GPU={gpu_vals[i]:.6f}  CPU={cpu_vals[i]:.6f}  "
              f"err={abs(gpu_vals[i]-cpu_vals[i]):.2e}")

    max_err = np.max(np.abs(gpu_vals - cpu_vals))
    print(f"  MaxErr: {max_err:.2e}")
    assert max_err < 1e-3, f"GPU/CPU mismatch: {max_err}"


@wp.kernel
def test_millard_energy_kernel(
    x_vals: wp.array(dtype=wp.float32),
    psi_out: wp.array(dtype=wp.float32),
    xc: wp.array(dtype=wp.float32),
    Fc: wp.array(dtype=wp.float32),
    bounds: wp.array(dtype=wp.float32),
    n_seg: int,
    x_lo: float, x_hi: float,
    y_lo: float, y_hi: float,
    dydx_lo: float, dydx_hi: float,
):
    i = wp.tid()
    psi_out[i] = millard_energy_eval_wp(float(x_vals[i]), xc, Fc, bounds, n_seg,
                                         x_lo, x_hi, y_lo, y_hi, dydx_lo, dydx_hi)


def test_gpu_fl_energy():
    mc = MillardCurves()
    fl_xc, fl_yc, fl_Fc, fl_bounds, fl_n = _upload_curve(mc.fl)

    # Test across full range including extrapolation
    test_lm = np.array([0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0,
                         1.2, 1.5, 1.7, 1.81, 1.9], dtype=np.float32)
    x_arr = wp.from_numpy(test_lm, dtype=wp.float32)
    psi_arr = wp.zeros(len(test_lm), dtype=wp.float32)

    wp.launch(test_millard_energy_kernel, dim=len(test_lm), inputs=[
        x_arr, psi_arr, fl_xc, fl_Fc, fl_bounds, fl_n,
        mc.fl.x_lo, mc.fl.x_hi, mc.fl.y_lo, mc.fl.y_hi,
        mc.fl.dydx_lo, mc.fl.dydx_hi,
    ])
    wp.synchronize()

    gpu_vals = psi_arr.numpy()
    cpu_vals = np.array([mc.fl.eval_integral(float(x)) for x in test_lm], dtype=np.float32)

    print("=== GPU vs CPU: Psi_L (energy integral) ===")
    for i in range(len(test_lm)):
        print(f"  lm={test_lm[i]:.2f}: GPU={gpu_vals[i]:.6f}  CPU={cpu_vals[i]:.6f}  "
              f"err={abs(gpu_vals[i]-cpu_vals[i]):.2e}")

    max_err = np.max(np.abs(gpu_vals - cpu_vals))
    print(f"  MaxErr: {max_err:.2e}")
    assert max_err < 1e-3, f"GPU/CPU energy mismatch: {max_err}"


def test_gpu_fpe_energy():
    mc = MillardCurves()
    fpe_xc, fpe_yc, fpe_Fc, fpe_bounds, fpe_n = _upload_curve(mc.fpe)

    test_lm = np.array([0.8, 1.0, 1.1, 1.3, 1.5, 1.7, 1.9], dtype=np.float32)
    x_arr = wp.from_numpy(test_lm, dtype=wp.float32)
    psi_arr = wp.zeros(len(test_lm), dtype=wp.float32)

    wp.launch(test_millard_energy_kernel, dim=len(test_lm), inputs=[
        x_arr, psi_arr, fpe_xc, fpe_Fc, fpe_bounds, fpe_n,
        mc.fpe.x_lo, mc.fpe.x_hi, mc.fpe.y_lo, mc.fpe.y_hi,
        mc.fpe.dydx_lo, mc.fpe.dydx_hi,
    ])
    wp.synchronize()

    gpu_vals = psi_arr.numpy()
    cpu_vals = np.array([mc.fpe.eval_integral(float(x)) for x in test_lm], dtype=np.float32)

    print("=== GPU vs CPU: Psi_PE (energy integral) ===")
    for i in range(len(test_lm)):
        print(f"  lm={test_lm[i]:.2f}: GPU={gpu_vals[i]:.6f}  CPU={cpu_vals[i]:.6f}  "
              f"err={abs(gpu_vals[i]-cpu_vals[i]):.2e}")

    max_err = np.max(np.abs(gpu_vals - cpu_vals))
    print(f"  MaxErr: {max_err:.2e}")
    assert max_err < 1e-3, f"GPU/CPU energy mismatch: {max_err}"


if __name__ == "__main__":
    test_gpu_fl_evaluation()
    print()
    test_gpu_fpe_evaluation()
    print()
    test_gpu_fl_energy()
    print()
    test_gpu_fpe_energy()
    print("\nAll GPU tests passed!")
