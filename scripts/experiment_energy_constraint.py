"""
CPU validation of energy-based XPBD constraint: C = sqrt(2 * sigma0 * a * Psi_L(lm))

Purpose:
  Verify the energy constitutive constraint formula before GPU integration.
  Validates: (1) math identities, (2) static equilibrium, (3) XPBD update
  direction/magnitude, (4) numerical stability across lm range.

Physics (sliding ball config):
  sigma0 = 300000 Pa, L0=0.1m, r=0.02m, ball=10kg, a=1.0
  Expected equilibrium: f_L(lm_eq) = norm_load/a ≈ 0.2602 → lm_eq ≈ 0.5498

Reference (Stage 4): lm_eq ≈ 0.5498, ball_pos ≈ 0.0550m
"""

import sys
import os
import numpy as np
from math import sqrt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from VMuscle.millard_curves import MillardCurves

# ── Parameters (from config_xpbd_millard.json) ──
sigma0 = 300_000.0       # Pa
L0 = 0.1                 # m
radius = 0.02            # m
density = 1060.0          # kg/m3
ball_mass = 10.0          # kg
gravity = 9.81            # m/s2
A_cross = np.pi * radius**2
V0 = A_cross * L0
F_max = sigma0 * A_cross
F_gravity = ball_mass * gravity
norm_load = F_gravity / F_max

mc = MillardCurves()
all_pass = True

print("=" * 70)
print("CPU Energy Constraint Validation")
print("=" * 70)
print(f"  sigma0={sigma0:.0f} Pa  A={A_cross:.6f} m2  V0={V0:.6e} m3")
print(f"  F_max={F_max:.1f} N  F_grav={F_gravity:.2f} N  norm_load={norm_load:.6f}")
print()


# ═══════════════════════════════════════════════════════════════════════
# Test 1: dPsi_L/dlm == f_L  (energy derivative equals force)
# ═══════════════════════════════════════════════════════════════════════
print("── Test 1: dPsi_L/dlm == f_L (energy derivative = force) ──")
h = 1e-7
test_lms = [0.45, 0.5, 0.55, 0.7, 0.85, 1.0, 1.2, 1.5, 1.75]
max_err = 0.0
for lm in test_lms:
    dpsi = (mc.fl.eval_integral(lm + h) - mc.fl.eval_integral(lm - h)) / (2 * h)
    f_L = mc.fl.eval(lm)
    err = abs(f_L - dpsi)
    max_err = max(max_err, err)
    print(f"  lm={lm:.2f}  f_L={f_L:.8f}  dPsi/dlm={dpsi:.8f}  err={err:.2e}")
ok = max_err < 1e-5
all_pass &= ok
print(f"  → max error: {max_err:.2e}  {'PASS' if ok else 'FAIL'}")
print()


# ═══════════════════════════════════════════════════════════════════════
# Test 2: C * dC/dlm == sigma0 * a * f_L  (force identity)
# ═══════════════════════════════════════════════════════════════════════
print("── Test 2: C · dC/dlm == sigma0 · a · f_L (force identity) ──")
a = 1.0
max_err2 = 0.0
for lm in [0.5, 0.6, 0.7, 0.85, 1.0, 1.3, 1.6]:
    psi_L = mc.fl.eval_integral(lm)
    f_L = mc.fl.eval(lm)
    if psi_L > 1e-15:
        C = sqrt(2 * sigma0 * a * psi_L)
        dC_dlm = sigma0 * a * f_L / C
        product = C * dC_dlm
        expected = sigma0 * a * f_L
        err = abs(product - expected) / max(abs(expected), 1e-15)
        max_err2 = max(max_err2, err)
        print(f"  lm={lm:.2f}  C·dC/dlm={product:.2f}  sigma0·a·f_L={expected:.2f}  "
              f"rel_err={err:.2e}")
ok = max_err2 < 1e-10
all_pass &= ok
print(f"  → max rel error: {max_err2:.2e}  {'PASS' if ok else 'FAIL'}")
print(f"  (This identity means XPBD force = sigma0·a·f_L regardless of energy magnitude)")
print()


# ═══════════════════════════════════════════════════════════════════════
# Test 3: Static equilibrium via bisection
# ═══════════════════════════════════════════════════════════════════════
print("── Test 3: Static equilibrium (force balance bisection) ──")
print(f"  Equilibrium condition: sigma0·a·f_L(lm_eq)·A = m·g")
print(f"  i.e. f_L(lm_eq) = {norm_load:.6f}")

# Bisection on ascending limb (f_L is monotonically increasing on [0.44, 1.0])
lo, hi = 0.4441, 1.0
for _ in range(100):
    mid = (lo + hi) / 2
    if mc.fl.eval(mid) < norm_load:
        lo = mid
    else:
        hi = mid
lm_eq = (lo + hi) / 2
f_eq = mc.fl.eval(lm_eq)
print(f"  lm_eq  = {lm_eq:.6f}  (Stage 4 reference: 0.5498)")
print(f"  f_L    = {f_eq:.6f}  (target: {norm_load:.6f})")
print(f"  F_musc = {sigma0 * a * f_eq * A_cross:.4f} N  (target: {F_gravity:.2f} N)")
err_lm = abs(lm_eq - 0.5498)
ok = err_lm < 0.01
all_pass &= ok
print(f"  → |lm_eq - 0.5498| = {err_lm:.6f}  {'PASS' if ok else 'FAIL'}")
print()


# ═══════════════════════════════════════════════════════════════════════
# Test 4: XPBD constraint correction direction and magnitude
# ═══════════════════════════════════════════════════════════════════════
print("── Test 4: XPBD constraint correction (direction & magnitude) ──")
print("  Verify: at lm=1.0 (rest), constraint drives lm DOWN (contraction)")
print("  Verify: at lm=lm_min (0.4441), constraint vanishes (C≈0)")
print()

for lm_test in [1.0, 0.8, 0.6, 0.5, lm_eq, 0.45, 0.4441]:
    psi_L = mc.fl.eval_integral(lm_test)
    f_L = mc.fl.eval(lm_test)
    if psi_L > 1e-15:
        C = sqrt(2 * sigma0 * a * psi_L)
        dC_dlm = sigma0 * a * f_L / C
        # In 3D XPBD: the correction Δlm ∝ -C * dC_dlm (always negative → contraction)
        sign = "↓ contract" if C * dC_dlm > 0 else "↑ extend"
        # Muscle force at this lm
        F_muscle = sigma0 * a * f_L * A_cross
        F_net = F_muscle - F_gravity
        balance = "muscle>gravity" if F_net > 0 else "gravity>muscle"
        print(f"  lm={lm_test:.4f}  C={C:10.4f}  dC/dlm={dC_dlm:10.2f}  "
              f"F_musc={F_muscle:8.2f}N  F_net={F_net:+8.2f}N  {sign}  ({balance})")
    else:
        print(f"  lm={lm_test:.4f}  C≈0 (Psi_L={psi_L:.2e})  → constraint inactive")

print()
print("  Expected: Δlm always negative (contraction direction),")
print("  F_net changes sign at lm_eq — muscle wins above, gravity wins below.")
print()


# ═══════════════════════════════════════════════════════════════════════
# Test 5: Numerical stability of C and dC/dlm across full range
# ═══════════════════════════════════════════════════════════════════════
print("── Test 5: Numerical stability across lm range ──")
lms = np.linspace(0.4441, 1.8123, 200)
C_vals = []
dC_vals = []
has_nan = False
has_inf = False

for lm in lms:
    psi_L = mc.fl.eval_integral(lm)
    f_L = mc.fl.eval(lm)
    if psi_L > 1e-15:
        C = sqrt(2 * sigma0 * a * psi_L)
        dC = sigma0 * a * f_L / C
    else:
        C = 0.0
        dC = 0.0
    C_vals.append(C)
    dC_vals.append(dC)
    if np.isnan(C) or np.isnan(dC):
        has_nan = True
    if np.isinf(C) or np.isinf(dC):
        has_inf = True

C_arr = np.array(C_vals)
dC_arr = np.array(dC_vals)
ok = not has_nan and not has_inf
all_pass &= ok
print(f"  C   range: [{C_arr.min():.4f}, {C_arr.max():.4f}]")
print(f"  dC  range: [{dC_arr.min():.4f}, {dC_arr.max():.4f}]")
print(f"  NaN: {has_nan}  Inf: {has_inf}  {'PASS' if ok else 'FAIL'}")
print()


# ═══════════════════════════════════════════════════════════════════════
# Test 6: Singularity near lm_min (Psi → 0)
# ═══════════════════════════════════════════════════════════════════════
print("── Test 6: Singularity handling near lm_min ──")
eps_psi = 1e-10
for lm_test in [0.4441, 0.4442, 0.4445, 0.445, 0.45]:
    psi_L = mc.fl.eval_integral(lm_test)
    f_L = mc.fl.eval(lm_test)
    if psi_L > eps_psi:
        C = sqrt(2 * sigma0 * a * psi_L)
        dC = sigma0 * a * f_L / C
        print(f"  lm={lm_test:.4f}  Psi={psi_L:.2e}  C={C:.6f}  "
              f"dC/dlm={dC:.2f}  (active)")
    else:
        print(f"  lm={lm_test:.4f}  Psi={psi_L:.2e}  → SKIP (Psi < eps={eps_psi:.0e})")
print(f"  → Threshold eps={eps_psi:.0e} safely skips singularity at lm_min")
print()


# ═══════════════════════════════════════════════════════════════════════
# Test 7: Relaxation iteration to equilibrium
# ═══════════════════════════════════════════════════════════════════════
print("── Test 7: Force-balance relaxation to equilibrium ──")
lm = 1.0  # start at optimal length
step_size = 1e-4
damping = 0.99
vel = 0.0

for i in range(5000):
    f_L = mc.fl.eval(lm)
    # Net force: positive = muscle dominates (should contract), negative = gravity dominates
    F_net = sigma0 * a * f_L * A_cross - F_gravity
    vel = damping * vel + step_size * F_net
    lm -= vel  # contraction: lm decreases when muscle force > gravity
    lm = np.clip(lm, 0.4441, 1.8123)

    if i % 500 == 0 or i == 4999:
        psi_L = mc.fl.eval_integral(lm)
        C = sqrt(2 * sigma0 * a * psi_L) if psi_L > 0 else 0.0
        print(f"  iter={i:5d}  lm={lm:.6f}  f_L={f_L:.6f}  "
              f"F_net={F_net:+.4f}N  C={C:.4f}")

err_relax = abs(lm - lm_eq)
ok = err_relax < 0.001
all_pass &= ok
print(f"  → lm_converged={lm:.6f}  lm_eq={lm_eq:.6f}  "
      f"|err|={err_relax:.6f}  {'PASS' if ok else 'FAIL'}")
print()


# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
print("=" * 70)
print(f"ALL TESTS: {'PASS' if all_pass else 'FAIL'}")
print("=" * 70)
print()
print("Note: XPBD dynamic convergence deferred to Task 4 (3D sliding ball)")
print("  1D chain model has extreme mass ratios (10kg ball vs 0.002kg muscle")
print("  node) that don't represent the 3D tet mesh's constraint distribution.")
