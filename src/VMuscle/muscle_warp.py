import time

import numpy as np
import warp as wp
import warp.render

from VMuscle.config import load_config  # noqa: F401
from VMuscle.constraints import (
    PIN, ATTACH, TETVOLUME, TETFIBERNORM, DISTANCELINE,
    TETARAP, TETARAPNORM, TETFIBERDGF, TETSNH,
    LINEARENERGY, NORMSTIFFNESS,
)
from VMuscle.mesh_io import build_surface_tris
from VMuscle.muscle_common import MuscleSimBase, get_bbox  # noqa: F401
from VMuscle.vis_warp import WarpRenderer


# ---------------------------------------------------------------------------
# Utility @wp.kernel for filling arrays
# ---------------------------------------------------------------------------

@wp.kernel
def fill_float_kernel(arr: wp.array(dtype=wp.float32), val: wp.float32):
    i = wp.tid()
    arr[i] = val

@wp.kernel
def fill_vec3_kernel(arr: wp.array(dtype=wp.vec3), val: wp.vec3):
    i = wp.tid()
    arr[i] = val


# ---------------------------------------------------------------------------
# @wp.struct definitions
# ---------------------------------------------------------------------------

@wp.struct
class SVDResult:
    U: wp.mat33
    sigma: wp.vec3
    V: wp.mat33

@wp.struct
class PolarResult:
    S: wp.mat33
    R: wp.mat33

@wp.struct
class Constraint:
    type: wp.int32
    cidx: wp.int32
    pts: wp.vec4i
    stiffness: wp.float32
    dampingratio: wp.float32
    tetid: wp.int32
    L: wp.vec3
    restlength: wp.float32
    restvector: wp.vec4f
    restdir: wp.vec3
    compressionstiffness: wp.float32


# ---------------------------------------------------------------------------
# Utility @wp.func helpers
# ---------------------------------------------------------------------------

@wp.func
def get_L_component(L: wp.vec3, loff: int) -> float:
    return wp.where(loff == 0, L[0], wp.where(loff == 1, L[1], L[2]))

@wp.func
def set_L_component(L: wp.vec3, loff: int, val: float) -> wp.vec3:
    return wp.vec3(
        wp.where(loff == 0, val, L[0]),
        wp.where(loff == 1, val, L[1]),
        wp.where(loff == 2, val, L[2]))


# ---------------------------------------------------------------------------
# Math @wp.func
# ---------------------------------------------------------------------------

@wp.func
def update_dP(dP: wp.array(dtype=wp.vec3), dPw: wp.array(dtype=wp.float32),
              dp: wp.vec3, pt: int):
    wp.atomic_add(dP, pt, dp)
    wp.atomic_add(dPw, pt, 1.0)

@wp.func
def fem_flags_fn(ctype: int) -> int:
    flags = 0
    if ctype == TETARAP or ctype == TETARAPNORM:
        flags = flags | LINEARENERGY
    if ctype == TETARAPNORM or ctype == TETFIBERNORM or ctype == TETFIBERDGF:
        flags = flags | NORMSTIFFNESS
    return flags

@wp.func
def project_to_line_fn(p: wp.vec3, orig: wp.vec3, direction: wp.vec3) -> wp.vec3:
    return orig + direction * wp.dot(p - orig, direction)

@wp.func
def get_inv_mass_fn(idx: int, mass: wp.array(dtype=wp.float32),
                    stopped: wp.array(dtype=wp.int32)) -> float:
    res = float(0.0)
    if stopped[idx] != 0:
        res = 0.0
    else:
        m = mass[idx]
        res = wp.where(m > 0.0, 1.0 / m, 0.0)
    return res


# ---------------------------------------------------------------------------
# DGF (DeGroote-Fregly 2016) muscle curve @wp.func
# Constants mirror dgf_curves.py
# ---------------------------------------------------------------------------

@wp.func
def dgf_active_force_length_wp(lm_tilde: float) -> float:
    """Active force-length curve: 3 Gaussian-like terms."""
    b11 = 0.815;  b21 = 1.055;  b31 = 0.162;  b41 = 0.063
    b12 = 0.433;  b22 = 0.717;  b32 = -0.030; b42 = 0.200
    b13 = 0.1;    b23 = 1.0;    b33 = 0.354
    EPS = 1.0e-6

    d1 = wp.max(wp.abs(b31 + b41 * lm_tilde), EPS)
    g1 = b11 * wp.exp(-0.5 * ((lm_tilde - b21) / d1) * ((lm_tilde - b21) / d1))

    d2 = wp.max(wp.abs(b32 + b42 * lm_tilde), EPS)
    g2 = b12 * wp.exp(-0.5 * ((lm_tilde - b22) / d2) * ((lm_tilde - b22) / d2))

    g3 = b13 * wp.exp(-0.5 * ((lm_tilde - b23) / b33) * ((lm_tilde - b23) / b33))
    return g1 + g2 + g3


@wp.func
def dgf_passive_force_length_wp(lm_tilde: float) -> float:
    """Passive force-length curve: exponential."""
    KPE = 4.0
    E0 = 0.6
    LM_MIN = 0.2

    offset = wp.exp(KPE * (LM_MIN - 1.0) / E0)
    denom = wp.exp(KPE) - offset
    arg = wp.clamp(KPE * (lm_tilde - 1.0) / E0, -50.0, 50.0)
    result = (wp.exp(arg) - offset) / denom
    return wp.max(result, 0.0)


@wp.func
def dgf_force_velocity_wp(v_norm: float) -> float:
    """Force-velocity curve: hyperbolic (asinh-based)."""
    D1 = -0.3211346127989808
    D2 = -8.149
    D3 = -0.374
    D4 = 0.8825327733249912

    x = D2 * v_norm + D3
    return D1 * wp.log(x + wp.sqrt(x * x + 1.0)) + D4


@wp.func
def ssvd_fn(F: wp.mat33) -> SVDResult:
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sigma = wp.vec3(0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    wp.svd3(F, U, sigma, V)

    detU = wp.determinant(U)
    if detU < 0.0:
        U = wp.mat33(
            U[0, 0], U[0, 1], -U[0, 2],
            U[1, 0], U[1, 1], -U[1, 2],
            U[2, 0], U[2, 1], -U[2, 2])
        sigma = wp.vec3(sigma[0], sigma[1], -sigma[2])

    detV = wp.determinant(V)
    if detV < 0.0:
        V = wp.mat33(
            V[0, 0], V[0, 1], -V[0, 2],
            V[1, 0], V[1, 1], -V[1, 2],
            V[2, 0], V[2, 1], -V[2, 2])
        sigma = wp.vec3(sigma[0], sigma[1], -sigma[2])

    result = SVDResult()
    result.U = U
    result.sigma = sigma
    result.V = V
    return result


@wp.func
def polar_decomposition_fn(F: wp.mat33) -> PolarResult:
    svd = ssvd_fn(F)
    U = svd.U
    sigma_vec = svd.sigma
    V = svd.V

    R = U * wp.transpose(V)
    sig_mat = wp.mat33(
        sigma_vec[0], 0.0, 0.0,
        0.0, sigma_vec[1], 0.0,
        0.0, 0.0, sigma_vec[2])
    S = V * sig_mat * wp.transpose(V)

    result = PolarResult()
    result.S = S
    result.R = R
    return result

@wp.func
def squared_norm3_fn(a: wp.mat33) -> float:
    s = float(0.0)
    for i in range(3):
        for j in range(3):
            s = s + a[i, j] * a[i, j]
    return s

@wp.func
def mat3_to_quat_fn(m: wp.mat33) -> wp.vec4f:
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    qw = float(0.0)
    qx = float(0.0)
    qy = float(0.0)
    qz = float(0.0)
    if trace > 0.0:
        s = wp.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = wp.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = wp.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = wp.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    return wp.vec4f(qx, qy, qz, qw)


@wp.func
def transfer_tension_fn(muscletension: float, tendonmask: float,
                        minfiberscale: float, maxfiberscale: float) -> float:
    fiberscale = minfiberscale + (1.0 - tendonmask) * muscletension * (maxfiberscale - minfiberscale)
    return fiberscale


# ---------------------------------------------------------------------------
# Constraint solver @wp.func
# ---------------------------------------------------------------------------

@wp.func
def tet_volume_update_xpbd_fn(
    use_jacobi: int,
    cidx: int,
    cons: wp.array(dtype=Constraint),
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    restlength: float,
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    tetid: int,
    pts: wp.vec4i,
    dt: float,
    stiffness: float,
    kstiffcompress: float,
    kdampratio: float,
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    has_compress_stiffness: int,
):
    inv_mass0 = get_inv_mass_fn(pts[0], mass, stopped)
    inv_mass1 = get_inv_mass_fn(pts[1], mass, stopped)
    inv_mass2 = get_inv_mass_fn(pts[2], mass, stopped)
    inv_mass3 = get_inv_mass_fn(pts[3], mass, stopped)

    p0 = pos[pts[0]]
    p1 = pos[pts[1]]
    p2 = pos[pts[2]]
    p3 = pos[pts[3]]

    d1 = p1 - p0
    d2 = p2 - p0
    d3 = p3 - p0
    grad1 = wp.cross(d3, d2) / 6.0
    grad2 = wp.cross(d1, d3) / 6.0
    grad3 = wp.cross(d2, d1) / 6.0
    grad0 = -(grad1 + grad2 + grad3)

    w_sum = (inv_mass0 * wp.length_sq(grad0) +
             inv_mass1 * wp.length_sq(grad1) +
             inv_mass2 * wp.length_sq(grad2) +
             inv_mass3 * wp.length_sq(grad3))

    if w_sum > 1e-9:
        volume = wp.dot(wp.cross(d2, d1), d3) / 6.0

        comp = 0
        if has_compress_stiffness != 0:
            if volume < restlength:
                comp = 1
        kstiff = wp.where(comp != 0, kstiffcompress, stiffness)
        loff = wp.min(comp, 2)

        if kstiff != 0.0:
            l = get_L_component(cons[cidx].L, loff)

            alpha = 1.0 / kstiff
            alpha = alpha / (dt * dt)

            C = volume - restlength

            dsum = float(0.0)
            gamma = float(1.0)
            if kdampratio > 0.0:
                prev0 = pprev[pts[0]]
                prev1 = pprev[pts[1]]
                prev2 = pprev[pts[2]]
                prev3 = pprev[pts[3]]
                beta = kstiff * kdampratio * dt * dt
                gamma = alpha * beta / dt
                dsum = (wp.dot(grad0, pos[pts[0]] - prev0) +
                        wp.dot(grad1, pos[pts[1]] - prev1) +
                        wp.dot(grad2, pos[pts[2]] - prev2) +
                        wp.dot(grad3, pos[pts[3]] - prev3))
                dsum = dsum * gamma
                gamma = gamma + 1.0

            dlambda = (-C - alpha * l - dsum) / (gamma * w_sum + alpha)

            if use_jacobi != 0:
                update_dP(dP, dPw, dlambda * inv_mass0 * grad0, pts[0])
                update_dP(dP, dPw, dlambda * inv_mass1 * grad1, pts[1])
                update_dP(dP, dPw, dlambda * inv_mass2 * grad2, pts[2])
                update_dP(dP, dPw, dlambda * inv_mass3 * grad3, pts[3])
            else:
                pos[pts[0]] = pos[pts[0]] + dlambda * inv_mass0 * grad0
                pos[pts[1]] = pos[pts[1]] + dlambda * inv_mass1 * grad1
                pos[pts[2]] = pos[pts[2]] + dlambda * inv_mass2 * grad2
                pos[pts[3]] = pos[pts[3]] + dlambda * inv_mass3 * grad3
                cons[cidx].L = set_L_component(cons[cidx].L, loff, get_L_component(cons[cidx].L, loff) + dlambda)


@wp.func
def tet_fiber_update_xpbd_fn(
    use_jacobi: int,
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    cidx: int,
    cons: wp.array(dtype=Constraint),
    pts: wp.vec4i,
    dt: float,
    fiber: wp.vec3,
    stiffness: float,
    Dminv: wp.mat33,
    kdampratio: float,
    restlength: float,
    restvector: wp.vec4f,
    acti: float,
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    target_stretch: float,
):
    inv_mass0 = get_inv_mass_fn(pts[0], mass, stopped)
    inv_mass1 = get_inv_mass_fn(pts[1], mass, stopped)
    inv_mass2 = get_inv_mass_fn(pts[2], mass, stopped)
    inv_mass3 = get_inv_mass_fn(pts[3], mass, stopped)

    l = cons[cidx].L[0]
    alpha = 1.0 / stiffness
    alpha = alpha / restlength
    alpha = alpha / (dt * dt)
    grad_scale = float(1.0)
    psi = float(0.0)

    c0 = pos[pts[0]] - pos[pts[3]]
    c1 = pos[pts[1]] - pos[pts[3]]
    c2 = pos[pts[2]] - pos[pts[3]]
    _Ds = wp.mat33(
        c0[0], c1[0], c2[0],
        c0[1], c1[1], c2[1],
        c0[2], c1[2], c2[2])

    wTDminvT = wp.vec3(restvector[0], restvector[1], restvector[2])
    FwT = _Ds * wTDminvT
    psi = 0.5 * wp.length_sq(FwT)

    if psi > 1e-9:
        psi_sqrt = wp.sqrt(2.0 * psi)
        grad_scale = 1.0 / psi_sqrt
        psi = psi_sqrt

    Ht = wp.outer(wTDminvT, FwT)

    grad0 = grad_scale * wp.vec3(Ht[0, 0], Ht[0, 1], Ht[0, 2])
    grad1 = grad_scale * wp.vec3(Ht[1, 0], Ht[1, 1], Ht[1, 2])
    grad2 = grad_scale * wp.vec3(Ht[2, 0], Ht[2, 1], Ht[2, 2])
    grad3 = -grad0 - grad1 - grad2

    w_sum = (inv_mass0 * wp.length_sq(grad0) +
             inv_mass1 * wp.length_sq(grad1) +
             inv_mass2 * wp.length_sq(grad2) +
             inv_mass3 * wp.length_sq(grad3))

    if w_sum > 1e-9:
        dsum = float(0.0)
        gamma = float(1.0)
        if kdampratio > 0.0:
            beta = stiffness * kdampratio * dt * dt
            beta = beta * restlength
            gamma = alpha * beta / dt

            dsum = (wp.dot(grad0, pos[pts[0]] - pprev[pts[0]]) +
                    wp.dot(grad1, pos[pts[1]] - pprev[pts[1]]) +
                    wp.dot(grad2, pos[pts[2]] - pprev[pts[2]]) +
                    wp.dot(grad3, pos[pts[3]] - pprev[pts[3]]))
            dsum = dsum * gamma
            gamma = gamma + 1.0

        C = psi - target_stretch
        dL = (-C - alpha * l - dsum) / (gamma * w_sum + alpha)
        if use_jacobi != 0:
            update_dP(dP, dPw, dL * inv_mass0 * grad0, pts[0])
            update_dP(dP, dPw, dL * inv_mass1 * grad1, pts[1])
            update_dP(dP, dPw, dL * inv_mass2 * grad2, pts[2])
            update_dP(dP, dPw, dL * inv_mass3 * grad3, pts[3])
        else:
            pos[pts[0]] = pos[pts[0]] + dL * inv_mass0 * grad0
            pos[pts[1]] = pos[pts[1]] + dL * inv_mass1 * grad1
            pos[pts[2]] = pos[pts[2]] + dL * inv_mass2 * grad2
            pos[pts[3]] = pos[pts[3]] + dL * inv_mass3 * grad3
            cons[cidx].L = wp.vec3(cons[cidx].L[0] + dL, cons[cidx].L[1], cons[cidx].L[2])


@wp.func
def distance_pos_update_xpbd_fn(
    use_jacobi: int,
    cidx: int,
    cons: wp.array(dtype=Constraint),
    pt0: int,
    pt1: int,
    p0: wp.vec3,
    p1: wp.vec3,
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    restlength: float,
    kstiff: float,
    kdampratio: float,
    kstiffcompress: float,
    dt: float,
    has_compress_stiffness: int,
):
    invmass0 = float(0.0)
    invmass1 = get_inv_mass_fn(pt1, mass, stopped)

    if pt0 >= 0:
        invmass0 = get_inv_mass_fn(pt0, mass, stopped)

    wsum = invmass0 + invmass1

    if wsum != 0.0:
        p1_current = pos[pt1]
        p0_current = p0
        if pt0 >= 0:
            p0_current = pos[pt0]

        n = p1_current - p0_current
        d = wp.length(n)
        if d >= 1e-6:
            comp = 0
            if has_compress_stiffness != 0:
                if d < restlength:
                    comp = 1
            kstiff_val = wp.where(comp != 0, kstiffcompress, kstiff)
            loff = wp.min(comp, 2)
            if kstiff_val != 0.0:
                l = get_L_component(cons[cidx].L, loff)

                alpha = 1.0 / kstiff_val
                alpha = alpha / (dt * dt)

                C = d - restlength
                n = n / d
                gradC = n

                dsum = float(0.0)
                gamma = float(1.0)
                if kdampratio > 0.0:
                    if pt0 >= 0:
                        prev0 = pprev[pt0]
                        prev1 = pprev[pt1]
                        beta = kstiff_val * kdampratio * dt * dt
                        gamma = alpha * beta / dt
                        dsum = gamma * (-wp.dot(gradC, p0_current - prev0) + wp.dot(gradC, p1_current - prev1))
                    else:
                        prev1 = pprev[pt1]
                        beta = kstiff_val * kdampratio * dt * dt
                        gamma = alpha * beta / dt
                        dsum = gamma * wp.dot(gradC, p1_current - prev1)
                    gamma = gamma + 1.0

                dL = (-C - alpha * l - dsum) / (gamma * wsum + alpha)
                dp = n * (-dL)

                if use_jacobi != 0:
                    if pt0 >= 0:
                        update_dP(dP, dPw, invmass0 * dp, pt0)
                    update_dP(dP, dPw, -invmass1 * dp, pt1)
                else:
                    if pt0 >= 0:
                        pos[pt0] = pos[pt0] + invmass0 * dp
                    pos[pt1] = pos[pt1] - invmass1 * dp
                    cons[cidx].L = set_L_component(cons[cidx].L, loff, get_L_component(cons[cidx].L, loff) + dL)


@wp.func
def attach_bilateral_update_fn(
    use_jacobi: int,
    cidx: int,
    cons: wp.array(dtype=Constraint),
    pt1: int,
    p0: wp.vec3,
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    restlength: float,
    kstiff: float,
    kdampratio: float,
    kstiffcompress: float,
    dt: float,
    has_compress_stiffness: int,
    reaction_accum: wp.array(dtype=wp.vec3),
):
    invmass1 = get_inv_mass_fn(pt1, mass, stopped)
    wsum = invmass1

    if wsum != 0.0:
        p1_current = pos[pt1]
        p0_current = p0

        n = p1_current - p0_current
        d = wp.length(n)
        if d >= 1e-6:
            comp = 0
            if has_compress_stiffness != 0:
                if d < restlength:
                    comp = 1
            kstiff_val = wp.where(comp != 0, kstiffcompress, kstiff)
            loff = wp.min(comp, 2)
            if kstiff_val != 0.0:
                l = get_L_component(cons[cidx].L, loff)

                alpha = 1.0 / kstiff_val
                alpha = alpha / (dt * dt)

                C = d - restlength
                n = n / d

                dsum = float(0.0)
                gamma = float(1.0)
                if kdampratio > 0.0:
                    prev1 = pprev[pt1]
                    beta = kstiff_val * kdampratio * dt * dt
                    gamma = alpha * beta / dt
                    dsum = gamma * wp.dot(n, p1_current - prev1)
                    gamma = gamma + 1.0

                dL = (-C - alpha * l - dsum) / (gamma * wsum + alpha)
                dp = n * (-dL)

                if use_jacobi != 0:
                    update_dP(dP, dPw, -invmass1 * dp, pt1)
                else:
                    pos[pt1] = pos[pt1] - invmass1 * dp
                    cons[cidx].L = set_L_component(cons[cidx].L, loff, get_L_component(cons[cidx].L, loff) + dL)

                wp.atomic_add(reaction_accum, cidx, C * n)


@wp.func
def tet_arap_update_xpbd_fn(
    use_jacobi: int,
    cidx: int,
    cons: wp.array(dtype=Constraint),
    pts: wp.vec4i,
    dt: float,
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    restlength: float,
    restmatrix: wp.mat33,
    kstiff: float,
    kdampratio: float,
    flags: int,
):
    pt0 = pts[0]
    pt1 = pts[1]
    pt2 = pts[2]
    pt3 = pts[3]
    p0 = pos[pt0]
    p1 = pos[pt1]
    p2 = pos[pt2]
    p3 = pos[pt3]

    invmass0 = get_inv_mass_fn(pt0, mass, stopped)
    invmass1 = get_inv_mass_fn(pt1, mass, stopped)
    invmass2 = get_inv_mass_fn(pt2, mass, stopped)
    invmass3 = get_inv_mass_fn(pt3, mass, stopped)

    c0 = p0 - p3
    c1 = p1 - p3
    c2 = p2 - p3
    Ds = wp.mat33(
        c0[0], c1[0], c2[0],
        c0[1], c1[1], c2[1],
        c0[2], c1[2], c2[2])

    F = Ds * restmatrix

    polar = polar_decomposition_fn(F)
    R = polar.R
    d = F - R

    cons[cidx].restvector = mat3_to_quat_fn(R)

    psi = squared_norm3_fn(d)
    gradscale = float(2.0)
    if (flags & (LINEARENERGY | NORMSTIFFNESS)) != 0:
        psi = wp.sqrt(psi)
        gradscale = 1.0 / psi

    if psi >= 1e-6:
        Ht = restmatrix * wp.transpose(d)
        grad0 = gradscale * wp.vec3(Ht[0, 0], Ht[0, 1], Ht[0, 2])
        grad1 = gradscale * wp.vec3(Ht[1, 0], Ht[1, 1], Ht[1, 2])
        grad2 = gradscale * wp.vec3(Ht[2, 0], Ht[2, 1], Ht[2, 2])
        grad3 = -grad0 - grad1 - grad2

        wsum = (invmass0 * wp.dot(grad0, grad0) +
                invmass1 * wp.dot(grad1, grad1) +
                invmass2 * wp.dot(grad2, grad2) +
                invmass3 * wp.dot(grad3, grad3))

        if wsum != 0.0:
            alpha = 1.0 / kstiff
            if (flags & NORMSTIFFNESS) != 0:
                alpha = alpha / restlength
            alpha = alpha / (dt * dt)

            dsum = float(0.0)
            gamma = float(1.0)
            if kdampratio > 0.0:
                prev0 = pprev[pt0]
                prev1 = pprev[pt1]
                prev2 = pprev[pt2]
                prev3 = pprev[pt3]
                beta = kstiff * kdampratio * dt * dt
                if (flags & NORMSTIFFNESS) != 0:
                    beta = beta * restlength
                gamma = alpha * beta / dt
                dsum = (wp.dot(grad0, p0 - prev0) +
                        wp.dot(grad1, p1 - prev1) +
                        wp.dot(grad2, p2 - prev2) +
                        wp.dot(grad3, p3 - prev3))
                dsum = dsum * gamma
                gamma = gamma + 1.0

            C = psi
            dL = (-C - alpha * cons[cidx].L[0] - dsum) / (gamma * wsum + alpha)
            if use_jacobi != 0:
                update_dP(dP, dPw, dL * invmass0 * grad0, pt0)
                update_dP(dP, dPw, dL * invmass1 * grad1, pt1)
                update_dP(dP, dPw, dL * invmass2 * grad2, pt2)
                update_dP(dP, dPw, dL * invmass3 * grad3, pt3)
            else:
                pos[pt0] = pos[pt0] + dL * invmass0 * grad0
                pos[pt1] = pos[pt1] + dL * invmass1 * grad1
                pos[pt2] = pos[pt2] + dL * invmass2 * grad2
                pos[pt3] = pos[pt3] + dL * invmass3 * grad3
                cons[cidx].L = wp.vec3(cons[cidx].L[0] + dL, cons[cidx].L[1], cons[cidx].L[2])


# ---------------------------------------------------------------------------
# Simulation @wp.kernel
# ---------------------------------------------------------------------------

@wp.kernel
def precompute_rest_kernel(
    pos0: wp.array(dtype=wp.vec3),
    tet_indices: wp.array(dtype=wp.vec4i),
    rest_volume: wp.array(dtype=wp.float32),
    rest_matrix: wp.array(dtype=wp.mat33),
    mass: wp.array(dtype=wp.float32),
    density: float,
    total_rest_volume: wp.array(dtype=wp.float32),
):
    c = wp.tid()
    pts = tet_indices[c]
    c0 = pos0[pts[0]] - pos0[pts[3]]
    c1 = pos0[pts[1]] - pos0[pts[3]]
    c2 = pos0[pts[2]] - pos0[pts[3]]
    Dm = wp.mat33(
        c0[0], c1[0], c2[0],
        c0[1], c1[1], c2[1],
        c0[2], c1[2], c2[2])
    vol = wp.abs(wp.determinant(Dm)) / 6.0
    rest_volume[c] = vol
    wp.atomic_add(total_rest_volume, 0, vol)
    mass_contrib = vol * density / 4.0
    for i in range(4):
        wp.atomic_add(mass, pts[i], mass_contrib)
    rest_matrix[c] = wp.inverse(Dm)


@wp.kernel
def integrate_kernel(
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    gravity: float,
    veldamping: float,
    dt: float,
):
    i = wp.tid()
    extacc = wp.vec3(0.0, gravity, 0.0)
    pprev[i] = pos[i]
    v = (1.0 - veldamping) * vel[i]
    v = v + dt * extacc
    vel[i] = v
    pos[i] = pos[i] + dt * v


@wp.kernel
def update_velocities_kernel(
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    dt: float,
):
    i = wp.tid()
    vel[i] = (pos[i] - pprev[i]) / dt


@wp.kernel
def clear_dP_kernel(
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    dP[i] = wp.vec3(0.0)
    dPw[i] = 0.0

@wp.kernel
def clear_cons_L_kernel(
    cons: wp.array(dtype=Constraint),
):
    i = wp.tid()
    cons[i].L = wp.vec3(0.0)


@wp.kernel
def clear_reaction_kernel(
    reaction_accum: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    reaction_accum[i] = wp.vec3(0.0)


@wp.kernel
def apply_dP_kernel(
    pos: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
):
    idx = wp.tid()
    w = dPw[idx]
    if w > 1e-9:
        pos[idx] = pos[idx] + dP[idx] / w


@wp.kernel
def calc_vol_error_kernel(
    tet_indices: wp.array(dtype=wp.vec4i),
    pos: wp.array(dtype=wp.vec3),
    rest_volume: wp.array(dtype=wp.float32),
    total_vol_out: wp.array(dtype=wp.float32),
):
    c = wp.tid()
    pts = tet_indices[c]
    p0 = pos[pts[0]]
    p1 = pos[pts[1]]
    p2 = pos[pts[2]]
    p3 = pos[pts[3]]
    d1 = p1 - p0
    d2 = p2 - p0
    d3 = p3 - p0
    volume = wp.dot(wp.cross(d2, d1), d3) / 6.0
    wp.atomic_add(total_vol_out, 0, volume)


@wp.kernel
def compute_cell_tendon_mask_kernel(
    tet_indices: wp.array(dtype=wp.vec4i),
    v_tendonmask: wp.array(dtype=wp.float32),
    tendonmask: wp.array(dtype=wp.float32),
):
    c = wp.tid()
    pts = tet_indices[c]
    sum_mask = float(0.0)
    for i in range(4):
        sum_mask = sum_mask + v_tendonmask[pts[i]]
    tendonmask[c] = sum_mask / 4.0


@wp.kernel
def update_attach_targets_kernel(
    cons: wp.array(dtype=Constraint),
    bone_pos_field: wp.array(dtype=wp.vec3),
    n_cons: int,
):
    c = wp.tid()
    if c >= n_cons:
        return
    ctype = cons[c].type
    if ctype == ATTACH:
        tgt_idx = cons[c].pts[2]
        if tgt_idx >= 0:
            target_pos = bone_pos_field[tgt_idx]
            rv = cons[c].restvector
            cons[c].restvector = wp.vec4f(target_pos[0], target_pos[1], target_pos[2], rv[3])
    elif ctype == DISTANCELINE:
        tgt_idx = cons[c].pts[1]
        if tgt_idx >= 0:
            target_pos = bone_pos_field[tgt_idx]
            rv = cons[c].restvector
            cons[c].restvector = wp.vec4f(target_pos[0], target_pos[1], target_pos[2], rv[3])


# ---------------------------------------------------------------------------
# Per-type constraint solver @wp.kernel
# ---------------------------------------------------------------------------

@wp.kernel
def solve_tetvolume_kernel(
    cons: wp.array(dtype=Constraint),
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    dt: float,
    use_jacobi: int,
    has_compress_stiffness: int,
    offset: int,
):
    c = offset + wp.tid()
    cstiffness = cons[c].stiffness
    if cstiffness <= 0.0:
        return
    kstiffcompress = cons[c].compressionstiffness
    kstiffcompress = wp.where(kstiffcompress >= 0.0, kstiffcompress, cstiffness)
    pts = cons[c].pts
    tetid = cons[c].tetid
    tet_volume_update_xpbd_fn(
        use_jacobi, c, cons, pos, pprev,
        cons[c].restlength, dP, dPw, tetid, pts,
        dt, cstiffness, kstiffcompress, cons[c].dampingratio,
        mass, stopped, has_compress_stiffness)


@wp.kernel
def solve_tetfibernorm_kernel(
    cons: wp.array(dtype=Constraint),
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    rest_matrix: wp.array(dtype=wp.mat33),
    v_fiber_dir: wp.array(dtype=wp.vec3),
    activation: wp.array(dtype=wp.float32),
    tendonmask: wp.array(dtype=wp.float32),
    dt: float,
    use_jacobi: int,
    contraction_ratio: float,
    fiber_stiffness_scale: float,
    offset: int,
):
    c = offset + wp.tid()
    cstiffness = cons[c].stiffness
    if cstiffness <= 0.0:
        return
    pts = cons[c].pts
    tetid = cons[c].tetid
    fiber_dir = (v_fiber_dir[pts[0]] + v_fiber_dir[pts[1]] +
                 v_fiber_dir[pts[2]] + v_fiber_dir[pts[3]]) / 4.0
    Dminv = rest_matrix[tetid]
    acti = activation[tetid]
    _tendonmask = tendonmask[tetid]
    belly_factor = 1.0 - _tendonmask

    fiberscale = transfer_tension_fn(acti, _tendonmask, 0.0, 10.0)
    stiffness_val = cstiffness * fiberscale * fiber_stiffness_scale
    if stiffness_val > 0.0:
        target_stretch = 1.0 - belly_factor * acti * contraction_ratio

        tet_fiber_update_xpbd_fn(
            use_jacobi, pos, pprev, dP, dPw,
            c, cons, pts, dt, fiber_dir, stiffness_val, Dminv,
            cons[c].dampingratio, cons[c].restlength, cons[c].restvector,
            acti, mass, stopped, target_stretch)


@wp.kernel
def solve_tetfiberdgf_kernel(
    cons: wp.array(dtype=Constraint),
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    rest_matrix: wp.array(dtype=wp.mat33),
    v_fiber_dir: wp.array(dtype=wp.vec3),
    activation: wp.array(dtype=wp.float32),
    tendonmask: wp.array(dtype=wp.float32),
    dt: float,
    use_jacobi: int,
    fiber_stiffness_scale: float,
    offset: int,
):
    """XPBD fiber constraint with DGF force-length and passive curves (quasi-static).

    Fixed target (no f_L feedback), DGF curves modulate stiffness only.
    target_stretch = 1 - activation * contraction_factor  (same as TETFIBERNORM)
    stiffness scaled by f_total so force magnitude follows DGF profile.
    """
    c = offset + wp.tid()
    cstiffness = cons[c].stiffness
    if cstiffness <= 0.0:
        return
    pts = cons[c].pts
    tetid = cons[c].tetid
    fiber_dir = (v_fiber_dir[pts[0]] + v_fiber_dir[pts[1]] +
                 v_fiber_dir[pts[2]] + v_fiber_dir[pts[3]]) / 4.0
    Dminv = rest_matrix[tetid]
    acti = activation[tetid]
    contraction_factor = cons[c].restdir[2]

    # Compute current fiber stretch from deformation gradient
    c0 = pos[pts[0]] - pos[pts[3]]
    c1 = pos[pts[1]] - pos[pts[3]]
    c2 = pos[pts[2]] - pos[pts[3]]
    _Ds = wp.mat33(
        c0[0], c1[0], c2[0],
        c0[1], c1[1], c2[1],
        c0[2], c1[2], c2[2])
    wTDminvT = wp.vec3(cons[c].restvector[0], cons[c].restvector[1], cons[c].restvector[2])
    FwT = _Ds * wTDminvT
    lm_tilde = wp.sqrt(wp.max(wp.length_sq(FwT), 1.0e-12))

    # Constant stiffness matching TETFIBERNORM: fiberscale = 10 * activation
    fiberscale = 10.0 * acti
    stiffness_val = cstiffness * fiberscale * fiber_stiffness_scale

    if stiffness_val > 0.0:
        target_stretch = 1.0 - acti * contraction_factor

        tet_fiber_update_xpbd_fn(
            use_jacobi, pos, pprev, dP, dPw,
            c, cons, pts, dt, fiber_dir, stiffness_val, Dminv,
            cons[c].dampingratio, cons[c].restlength, cons[c].restvector,
            acti, mass, stopped, target_stretch)


@wp.kernel
def solve_attach_kernel(
    cons: wp.array(dtype=Constraint),
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    reaction_accum: wp.array(dtype=wp.vec3),
    dt: float,
    use_jacobi: int,
    has_compress_stiffness: int,
    offset: int,
):
    c = offset + wp.tid()
    cstiffness = cons[c].stiffness
    if cstiffness <= 0.0:
        return
    kstiffcompress = cons[c].compressionstiffness
    kstiffcompress = wp.where(kstiffcompress >= 0.0, kstiffcompress, cstiffness)
    pts = cons[c].pts
    pt_src = pts[0]
    rv = cons[c].restvector
    p0_target = wp.vec3(rv[0], rv[1], rv[2])
    attach_bilateral_update_fn(
        use_jacobi, c, cons, pt_src, p0_target,
        pos, pprev, dP, dPw, mass, stopped,
        cons[c].restlength, cstiffness, cons[c].dampingratio,
        kstiffcompress, dt, has_compress_stiffness, reaction_accum)


@wp.kernel
def solve_pin_kernel(
    cons: wp.array(dtype=Constraint),
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    dt: float,
    use_jacobi: int,
    has_compress_stiffness: int,
    offset: int,
):
    c = offset + wp.tid()
    cstiffness = cons[c].stiffness
    if cstiffness <= 0.0:
        return
    kstiffcompress = cons[c].compressionstiffness
    kstiffcompress = wp.where(kstiffcompress >= 0.0, kstiffcompress, cstiffness)
    pts = cons[c].pts
    pt_src = pts[0]
    rv = cons[c].restvector
    p0_target = wp.vec3(rv[0], rv[1], rv[2])
    p1_src = pos[pt_src]
    distance_pos_update_xpbd_fn(
        use_jacobi, c, cons, -1, pt_src,
        p0_target, p1_src,
        pos, pprev, dP, dPw, mass, stopped,
        cons[c].restlength, cstiffness, cons[c].dampingratio,
        kstiffcompress, dt, has_compress_stiffness)


@wp.kernel
def solve_distanceline_kernel(
    cons: wp.array(dtype=Constraint),
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    dt: float,
    use_jacobi: int,
    has_compress_stiffness: int,
    offset: int,
):
    c = offset + wp.tid()
    cstiffness = cons[c].stiffness
    if cstiffness <= 0.0:
        return
    kstiffcompress = cons[c].compressionstiffness
    kstiffcompress = wp.where(kstiffcompress >= 0.0, kstiffcompress, cstiffness)
    pts = cons[c].pts
    pt_src = pts[0]
    p_src = pos[pt_src]
    rv = cons[c].restvector
    line_origin = wp.vec3(rv[0], rv[1], rv[2])
    line_dir = cons[c].restdir

    p_projected = project_to_line_fn(p_src, line_origin, line_dir)

    distance_pos_update_xpbd_fn(
        use_jacobi, c, cons, -1, pt_src,
        p_projected, p_src,
        pos, pprev, dP, dPw, mass, stopped,
        cons[c].restlength, cstiffness, cons[c].dampingratio,
        kstiffcompress, dt, has_compress_stiffness)


@wp.kernel
def solve_tetarap_kernel(
    cons: wp.array(dtype=Constraint),
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    rest_matrix: wp.array(dtype=wp.mat33),
    dt: float,
    use_jacobi: int,
    offset: int,
):
    c = offset + wp.tid()
    cstiffness = cons[c].stiffness
    if cstiffness <= 0.0:
        return
    pts = cons[c].pts
    tetid = cons[c].tetid
    ctype = cons[c].type
    flags = fem_flags_fn(ctype)
    tet_arap_update_xpbd_fn(
        use_jacobi, c, cons, pts, dt, pos, pprev, dP, dPw,
        mass, stopped, cons[c].restlength, rest_matrix[tetid],
        cstiffness, cons[c].dampingratio, flags)


@wp.func
def tet_snh_update_xpbd_fn(
    use_jacobi: int,
    cidx: int,
    cons: wp.array(dtype=Constraint),
    pts: wp.vec4i,
    dt: float,
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    restvolume: float,
    restmatrix: wp.mat33,
    mu: float,
    lam: float,
    kdampratio: float,
):
    """Stable Neo-Hookean XPBD constraint (coupled deviatoric + volumetric).

    Deviatoric: C_D = ||F||_F - sqrt(3)
    Volumetric: C_H = J - alpha  where alpha = 1 + mu/lam
    Coupled 2x2 solve for both Lagrange multipliers.
    Reference: Macklin "A Constraint-based Formulation of Stable Neo-Hookean Materials"
    """
    pt0 = pts[0]
    pt1 = pts[1]
    pt2 = pts[2]
    pt3 = pts[3]
    p0 = pos[pt0]
    p1 = pos[pt1]
    p2 = pos[pt2]
    p3 = pos[pt3]

    invmass0 = get_inv_mass_fn(pt0, mass, stopped)
    invmass1 = get_inv_mass_fn(pt1, mass, stopped)
    invmass2 = get_inv_mass_fn(pt2, mass, stopped)
    invmass3 = get_inv_mass_fn(pt3, mass, stopped)

    # Deformation gradient F = Ds * Dm^-1
    c0 = p0 - p3
    c1 = p1 - p3
    c2 = p2 - p3
    Ds = wp.mat33(
        c0[0], c1[0], c2[0],
        c0[1], c1[1], c2[1],
        c0[2], c1[2], c2[2])
    F = Ds * restmatrix

    # --- Deviatoric constraint: C_D = ||F||_F - sqrt(3) ---
    Ic = (F[0, 0] * F[0, 0] + F[0, 1] * F[0, 1] + F[0, 2] * F[0, 2] +
          F[1, 0] * F[1, 0] + F[1, 1] * F[1, 1] + F[1, 2] * F[1, 2] +
          F[2, 0] * F[2, 0] + F[2, 1] * F[2, 1] + F[2, 2] * F[2, 2])
    r_s = wp.sqrt(wp.max(Ic, 1.0e-12))
    dC = r_s - 1.7320508075688772  # sqrt(3)

    if r_s < 1.0e-6:
        return

    # Deviatoric gradient: dC_D/dF = F/||F||, projected to position space
    inv_rs = 1.0 / r_s
    Fnorm = F * inv_rs
    Ht = restmatrix * wp.transpose(Fnorm)
    dgrad0 = wp.vec3(Ht[0, 0], Ht[0, 1], Ht[0, 2])
    dgrad1 = wp.vec3(Ht[1, 0], Ht[1, 1], Ht[1, 2])
    dgrad2 = wp.vec3(Ht[2, 0], Ht[2, 1], Ht[2, 2])
    dgrad3 = -dgrad0 - dgrad1 - dgrad2

    dsum_w = (invmass0 * wp.dot(dgrad0, dgrad0) +
              invmass1 * wp.dot(dgrad1, dgrad1) +
              invmass2 * wp.dot(dgrad2, dgrad2) +
              invmass3 * wp.dot(dgrad3, dgrad3))
    if dsum_w == 0.0:
        return

    # --- Volumetric constraint: C_H = J - alpha ---
    # Clamped: only resist volume expansion (J > alpha)
    alpha_snh = 1.0 + mu / lam
    J = wp.determinant(F)
    hC = J - alpha_snh

    # Volume gradients: dJ/dx_i via cross products of edge vectors
    inv_detDm = 1.0 / wp.max(wp.abs(restvolume) * 6.0, 1.0e-20)
    hgrad0 = wp.cross(c1, c2) * inv_detDm
    hgrad1 = wp.cross(c2, c0) * inv_detDm
    hgrad2 = wp.cross(c0, c1) * inv_detDm
    hgrad3 = -hgrad0 - hgrad1 - hgrad2

    hsum_w = (invmass0 * wp.dot(hgrad0, hgrad0) +
              invmass1 * wp.dot(hgrad1, hgrad1) +
              invmass2 * wp.dot(hgrad2, hgrad2) +
              invmass3 * wp.dot(hgrad3, hgrad3))
    if hsum_w == 0.0:
        return

    # Coupled cross-term
    dhsum = (invmass0 * wp.dot(dgrad0, hgrad0) +
             invmass1 * wp.dot(dgrad1, hgrad1) +
             invmass2 * wp.dot(dgrad2, hgrad2) +
             invmass3 * wp.dot(dgrad3, hgrad3))

    # XPBD compliances: alpha = 1/(stiffness * volume * dt^2)
    abs_vol = wp.max(wp.abs(restvolume), 1.0e-20)
    dalpha = 1.0 / (mu * abs_vol * dt * dt)
    halpha = 1.0 / (lam * abs_vol * dt * dt)

    # Damping
    ddamp = float(0.0)
    dgamma = float(1.0)
    hdamp = float(0.0)
    hgamma = float(1.0)
    if kdampratio > 0.0:
        prev0 = pprev[pt0]
        prev1 = pprev[pt1]
        prev2 = pprev[pt2]
        prev3 = pprev[pt3]
        dbeta = mu * abs_vol * kdampratio * dt * dt
        hbeta = lam * abs_vol * kdampratio * dt * dt
        dgamma = dalpha * dbeta / dt
        hgamma = halpha * hbeta / dt
        ddamp = (wp.dot(dgrad0, p0 - prev0) + wp.dot(dgrad1, p1 - prev1) +
                 wp.dot(dgrad2, p2 - prev2) + wp.dot(dgrad3, p3 - prev3))
        hdamp = (wp.dot(hgrad0, p0 - prev0) + wp.dot(hgrad1, p1 - prev1) +
                 wp.dot(hgrad2, p2 - prev2) + wp.dot(hgrad3, p3 - prev3))
        ddamp = ddamp * dgamma
        hdamp = hdamp * hgamma
        dgamma = dgamma + 1.0
        hgamma = hgamma + 1.0

    dL = cons[cidx].L[0]
    hL = cons[cidx].L[2]

    # Coupled 2x2 linear solve for both Lagrange multipliers
    Axx = dgamma * dsum_w + dalpha
    Ayy = hgamma * hsum_w + halpha
    det_A = Axx * Ayy - dhsum * dhsum
    if det_A < 1.0e-8:
        return
    inv_det = 1.0 / det_A
    db = -dC - dalpha * dL - ddamp
    hb = -hC - halpha * hL - hdamp
    ddL = inv_det * (Ayy * db - dhsum * hb)
    hdL = inv_det * (Axx * hb - dhsum * db)

    if use_jacobi != 0:
        update_dP(dP, dPw, invmass0 * (ddL * dgrad0 + hdL * hgrad0), pt0)
        update_dP(dP, dPw, invmass1 * (ddL * dgrad1 + hdL * hgrad1), pt1)
        update_dP(dP, dPw, invmass2 * (ddL * dgrad2 + hdL * hgrad2), pt2)
        update_dP(dP, dPw, invmass3 * (ddL * dgrad3 + hdL * hgrad3), pt3)
    else:
        pos[pt0] = p0 + invmass0 * (ddL * dgrad0 + hdL * hgrad0)
        pos[pt1] = p1 + invmass1 * (ddL * dgrad1 + hdL * hgrad1)
        pos[pt2] = p2 + invmass2 * (ddL * dgrad2 + hdL * hgrad2)
        pos[pt3] = p3 + invmass3 * (ddL * dgrad3 + hdL * hgrad3)
        cons[cidx].L = wp.vec3(dL + ddL, cons[cidx].L[1], hL + hdL)


@wp.kernel
def solve_tetsnh_kernel(
    cons: wp.array(dtype=Constraint),
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    rest_matrix: wp.array(dtype=wp.mat33),
    dt: float,
    use_jacobi: int,
    offset: int,
):
    c = offset + wp.tid()
    mu = cons[c].stiffness
    if mu <= 0.0:
        return
    pts = cons[c].pts
    tetid = cons[c].tetid
    lam = cons[c].restdir[0]
    tet_snh_update_xpbd_fn(
        use_jacobi, c, cons, pts, dt, pos, pprev, dP, dPw,
        mass, stopped, cons[c].restlength, rest_matrix[tetid],
        mu, lam, cons[c].dampingratio)


# ---------------------------------------------------------------------------
# MuscleSim (Warp backend)
# ---------------------------------------------------------------------------

class MuscleSim(MuscleSimBase):

    def _init_backend(self):
        wp.init()

    def _build_surface_tris(self):
        self.surface_tris = build_surface_tris(self.tet_np)
        self.n_tris = self.surface_tris.shape[0]
        self.surface_tris_flat = self.surface_tris.reshape(-1).astype(np.int32)

    def _allocate_fields(self):
        n_v = self.n_verts
        n_tet = self.tet_np.shape[0]

        self.pos = wp.zeros(n_v, dtype=wp.vec3)
        self.pprev = wp.zeros(n_v, dtype=wp.vec3)
        self.pos0 = wp.zeros(n_v, dtype=wp.vec3)
        self.vel = wp.zeros(n_v, dtype=wp.vec3)
        self.force = wp.zeros(n_v, dtype=wp.vec3)
        self.mass = wp.zeros(n_v, dtype=wp.float32)
        self.stopped = wp.zeros(n_v, dtype=wp.int32)
        self.v_fiber_dir = wp.zeros(n_v, dtype=wp.vec3)
        self.dP = wp.zeros(n_v, dtype=wp.vec3)
        self.dPw = wp.zeros(n_v, dtype=wp.float32)
        self.tet_indices = wp.zeros(n_tet, dtype=wp.vec4i)

        self.rest_volume = wp.zeros(n_tet, dtype=wp.float32)
        self.rest_matrix = wp.zeros(n_tet, dtype=wp.mat33)
        self.activation = wp.zeros(n_tet, dtype=wp.float32)

    def _create_bone_fields(self):
        if self.bone_pos.shape[0] > 0:
            self.bone_pos_field = wp.from_numpy(self.bone_pos.astype(np.float32), dtype=wp.vec3)
            if self.bone_indices_np.shape[0] > 0:
                self.bone_indices_field = wp.from_numpy(self.bone_indices_np.astype(np.int32), dtype=wp.int32)
            else:
                self.bone_indices_field = None
        else:
            self.bone_pos_field = None
            self.bone_indices_field = None

    def build_constraints(self):
        print("Building constraints...")
        all_constraints, _dt_collect = self._collect_raw_constraints()

        # Sort by type for per-type kernel dispatch
        all_constraints.sort(key=lambda c: c['type'])

        n_cons = len(all_constraints)
        self.n_cons = n_cons
        self.cons_ranges = {}
        if n_cons > 0:
            prev_type = None
            start_idx = 0
            for i, c in enumerate(all_constraints):
                c['cidx'] = i
                ctype = c['type']
                if ctype != prev_type:
                    if prev_type is not None:
                        self.cons_ranges[prev_type] = (start_idx, i - start_idx)
                    start_idx = i
                    prev_type = ctype
            if prev_type is not None:
                self.cons_ranges[prev_type] = (start_idx, n_cons - start_idx)

            type_arr = np.array([c['type'] for c in all_constraints], dtype=np.int32)
            cidx_arr = np.arange(n_cons, dtype=np.int32)
            pts_arr = np.array([c['pts'] for c in all_constraints], dtype=np.int32)
            stiffness_arr = np.array([c['stiffness'] for c in all_constraints], dtype=np.float32)
            dampingratio_arr = np.array([c['dampingratio'] for c in all_constraints], dtype=np.float32)
            tetid_arr = np.array([c['tetid'] for c in all_constraints], dtype=np.int32)
            L_arr = np.array([c['L'] for c in all_constraints], dtype=np.float32)
            restlength_arr = np.array([c['restlength'] for c in all_constraints], dtype=np.float32)
            restvector_arr = np.array([c['restvector'] for c in all_constraints], dtype=np.float32)
            restdir_arr = np.array([c['restdir'] for c in all_constraints], dtype=np.float32)
            compressionstiffness_arr = np.array([c['compressionstiffness'] for c in all_constraints], dtype=np.float32)

            cons_np = np.zeros(n_cons, dtype=Constraint.numpy_dtype())
            cons_np['type'] = type_arr
            cons_np['cidx'] = cidx_arr
            cons_np['pts'] = pts_arr
            cons_np['stiffness'] = stiffness_arr
            cons_np['dampingratio'] = dampingratio_arr
            cons_np['tetid'] = tetid_arr
            cons_np['L'] = L_arr
            cons_np['restlength'] = restlength_arr
            cons_np['restvector'] = restvector_arr
            cons_np['restdir'] = restdir_arr
            cons_np['compressionstiffness'] = compressionstiffness_arr

            self.cons = wp.array(cons_np, dtype=Constraint)
        else:
            self.cons = wp.zeros(0, dtype=Constraint)
            self.raw_constraints = []

        print(f"Built {n_cons} constraints total. [{_dt_collect*1000:.0f}ms]")

        self.reaction_accum = wp.zeros(max(n_cons, 1), dtype=wp.vec3)

        # Build colored constraint groups for parallel Gauss-Seidel
        if self.use_colored_gs and n_cons > 0:
            self._build_colored_gs(all_constraints)

    def _build_colored_gs(self, all_constraints):
        """Reorder constraints by (color, type) and build per-color dispatch ranges."""
        from .constraints import build_constraint_color_groups
        import time

        t0 = time.perf_counter()
        color_groups = build_constraint_color_groups(all_constraints)
        n_colors = len(color_groups)
        print(f"  Graph coloring: {n_colors} colors, "
              f"sizes: {[len(g) for g in color_groups]} "
              f"[{(time.perf_counter()-t0)*1000:.0f}ms]")

        # Reorder constraints: (color, type) sorted
        reordered = []
        self.color_type_ranges = []
        global_offset = 0
        for color_idx, group in enumerate(color_groups):
            color_cons = [all_constraints[i] for i in group]
            color_cons.sort(key=lambda c: c['type'])

            type_ranges = {}
            prev_type = None
            start = 0
            for i, c in enumerate(color_cons):
                ctype = c['type']
                if ctype != prev_type:
                    if prev_type is not None:
                        type_ranges[prev_type] = (global_offset + start, i - start)
                    start = i
                    prev_type = ctype
            if prev_type is not None:
                type_ranges[prev_type] = (global_offset + start, len(color_cons) - start)

            self.color_type_ranges.append(type_ranges)
            reordered.extend(color_cons)
            global_offset += len(color_cons)

        # Rebuild constraint array with new order
        n_cons = len(reordered)
        for i, c in enumerate(reordered):
            c['cidx'] = i

        cons_np = np.zeros(n_cons, dtype=Constraint.numpy_dtype())
        cons_np['type'] = np.array([c['type'] for c in reordered], dtype=np.int32)
        cons_np['cidx'] = np.arange(n_cons, dtype=np.int32)
        cons_np['pts'] = np.array([c['pts'] for c in reordered], dtype=np.int32)
        cons_np['stiffness'] = np.array([c['stiffness'] for c in reordered], dtype=np.float32)
        cons_np['dampingratio'] = np.array([c['dampingratio'] for c in reordered], dtype=np.float32)
        cons_np['tetid'] = np.array([c['tetid'] for c in reordered], dtype=np.int32)
        cons_np['L'] = np.array([c['L'] for c in reordered], dtype=np.float32)
        cons_np['restlength'] = np.array([c['restlength'] for c in reordered], dtype=np.float32)
        cons_np['restvector'] = np.array([c['restvector'] for c in reordered], dtype=np.float32)
        cons_np['restdir'] = np.array([c['restdir'] for c in reordered], dtype=np.float32)
        cons_np['compressionstiffness'] = np.array([c['compressionstiffness'] for c in reordered], dtype=np.float32)
        self.cons = wp.array(cons_np, dtype=Constraint)

    def _init_fields(self):
        self.pos0 = wp.from_numpy(self.pos0_np.astype(np.float32), dtype=wp.vec3)
        self.pos = wp.from_numpy(self.pos0_np.astype(np.float32), dtype=wp.vec3)
        self.vel.zero_()
        self.force.zero_()
        self.mass.zero_()
        self.stopped.zero_()
        self.tet_indices = wp.from_numpy(self.tet_np.astype(np.int32), dtype=wp.vec4i)

        if self.v_fiber_np is None:
            self.v_fiber_np = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (self.n_verts, 1))
        self.v_fiber_dir = wp.from_numpy(self.v_fiber_np.astype(np.float32), dtype=wp.vec3)

        n_tet = self.tet_np.shape[0]
        self.tendonmask = wp.zeros(n_tet, dtype=wp.float32)
        if self.v_tendonmask_np is not None:
            v_tendonmask_wp = wp.from_numpy(self.v_tendonmask_np.astype(np.float32), dtype=wp.float32)
            wp.launch(compute_cell_tendon_mask_kernel, dim=n_tet,
                      inputs=[self.tet_indices, v_tendonmask_wp, self.tendonmask])

        self.activation.zero_()
        self.total_rest_volume = wp.zeros(1, dtype=wp.float32)

        print("Initialized fields done.")

    def _precompute_rest(self):
        n_tet = self.tet_np.shape[0]
        self.total_rest_volume.zero_()
        self.mass.zero_()
        wp.launch(precompute_rest_kernel, dim=n_tet,
                  inputs=[self.pos0, self.tet_indices, self.rest_volume,
                          self.rest_matrix, self.mass, self.cfg.density,
                          self.total_rest_volume])

    def _init_renderer(self):
        self.renderer = None
        if self.cfg.gui and self.cfg.render_mode in ("human", "usd"):
            stage_path = "muscle_sim.usd" if self.cfg.render_mode == "usd" else None
            self.renderer = WarpRenderer(mode=self.cfg.render_mode, stage_path=stage_path)

    def reset(self):
        self.pos = wp.from_numpy(self.pos0_np.astype(np.float32), dtype=wp.vec3)
        self.pprev = wp.from_numpy(self.pos0_np.astype(np.float32), dtype=wp.vec3)
        self.vel.zero_()
        self.force.zero_()
        self.activation.zero_()
        self.clear()
        self.step_cnt = 1

    def update_attach_targets(self):
        if (
            hasattr(self, "bone_pos_field")
            and self.bone_pos_field is not None
            and self.bone_pos_field.shape[0] > 0
            and hasattr(self, "attach_constraints")
            and len(self.attach_constraints) > 0
            and self.n_cons > 0
        ):
            wp.launch(update_attach_targets_kernel, dim=self.n_cons,
                      inputs=[self.cons, self.bone_pos_field, self.n_cons])

    def integrate(self):
        wp.launch(integrate_kernel, dim=self.n_verts,
                  inputs=[self.pos, self.pprev, self.vel,
                          self.cfg.gravity, self.cfg.veldamping, self.dt])

    def update_velocities(self):
        wp.launch(update_velocities_kernel, dim=self.n_verts,
                  inputs=[self.pos, self.pprev, self.vel, self.dt])

    def calc_vol_error(self):
        total_vol = wp.zeros(1, dtype=wp.float32)
        n_tet = self.tet_np.shape[0]
        wp.launch(calc_vol_error_kernel, dim=n_tet,
                  inputs=[self.tet_indices, self.pos, self.rest_volume, total_vol])
        total_vol_np = total_vol.numpy()[0]
        total_rest_np = self.total_rest_volume.numpy()[0]
        if total_rest_np == 0.0:
            return 0.0
        return (total_vol_np - total_rest_np) / total_rest_np

    def clear(self):
        wp.launch(clear_dP_kernel, dim=self.n_verts,
                  inputs=[self.dP, self.dPw])
        if self.n_cons > 0:
            wp.launch(clear_cons_L_kernel, dim=self.n_cons,
                      inputs=[self.cons])

    def clear_reaction(self):
        if self.n_cons > 0:
            wp.launch(clear_reaction_kernel, dim=max(self.n_cons, 1),
                      inputs=[self.reaction_accum])

    def apply_dP(self):
        wp.launch(apply_dP_kernel, dim=self.n_verts,
                  inputs=[self.pos, self.dP, self.dPw])

    def _dispatch_constraints(self, type_ranges):
        """Dispatch constraint solver kernels for a given {type: (offset, count)} mapping."""
        has_compress = 1 if self.has_compressstiffness else 0
        use_jacobi_int = 1 if self.use_jacobi else 0
        for ctype, (offset, count) in type_ranges.items():
            if ctype == TETVOLUME:
                wp.launch(solve_tetvolume_kernel, dim=count, inputs=[
                    self.cons, self.pos, self.pprev,
                    self.dP, self.dPw, self.mass, self.stopped,
                    self.dt, use_jacobi_int, has_compress, offset])
            elif ctype == TETFIBERNORM:
                wp.launch(solve_tetfibernorm_kernel, dim=count, inputs=[
                    self.cons, self.pos, self.pprev,
                    self.dP, self.dPw, self.mass, self.stopped,
                    self.rest_matrix, self.v_fiber_dir,
                    self.activation, self.tendonmask,
                    self.dt, use_jacobi_int,
                    self.contraction_ratio, self.fiber_stiffness_scale,
                    offset])
            elif ctype == ATTACH:
                wp.launch(solve_attach_kernel, dim=count, inputs=[
                    self.cons, self.pos, self.pprev,
                    self.dP, self.dPw, self.mass, self.stopped,
                    self.reaction_accum,
                    self.dt, use_jacobi_int, has_compress, offset])
            elif ctype == PIN:
                wp.launch(solve_pin_kernel, dim=count, inputs=[
                    self.cons, self.pos, self.pprev,
                    self.dP, self.dPw, self.mass, self.stopped,
                    self.dt, use_jacobi_int, has_compress, offset])
            elif ctype == DISTANCELINE:
                wp.launch(solve_distanceline_kernel, dim=count, inputs=[
                    self.cons, self.pos, self.pprev,
                    self.dP, self.dPw, self.mass, self.stopped,
                    self.dt, use_jacobi_int, has_compress, offset])
            elif ctype == TETARAP:
                wp.launch(solve_tetarap_kernel, dim=count, inputs=[
                    self.cons, self.pos, self.pprev,
                    self.dP, self.dPw, self.mass, self.stopped,
                    self.rest_matrix,
                    self.dt, use_jacobi_int, offset])
            elif ctype == TETFIBERDGF:
                wp.launch(solve_tetfiberdgf_kernel, dim=count, inputs=[
                    self.cons, self.pos, self.pprev,
                    self.dP, self.dPw, self.mass, self.stopped,
                    self.rest_matrix, self.v_fiber_dir,
                    self.activation, self.tendonmask,
                    self.dt, use_jacobi_int,
                    self.fiber_stiffness_scale,
                    offset])
            elif ctype == TETSNH:
                wp.launch(solve_tetsnh_kernel, dim=count, inputs=[
                    self.cons, self.pos, self.pprev,
                    self.dP, self.dPw, self.mass, self.stopped,
                    self.rest_matrix,
                    self.dt, use_jacobi_int, offset])

    def solve_constraints(self):
        if self.n_cons == 0:
            return
        if self.use_colored_gs and hasattr(self, 'color_type_ranges'):
            for type_ranges in self.color_type_ranges:
                self._dispatch_constraints(type_ranges)
        else:
            self._dispatch_constraints(self.cons_ranges)


    def render(self):
        if self.renderer is None:
            return
        pos_np = self.pos.numpy()
        bone_pos_np = None
        bone_idx = None
        if hasattr(self, 'bone_pos_field') and self.bone_pos_field is not None:
            bone_pos_np = self.bone_pos_field.numpy()
            bone_idx = self.bone_indices_np if hasattr(self, 'bone_indices_np') else None
        sim_time = self.step_cnt * self.cfg.dt
        self.renderer.update(sim_time, pos_np, self.surface_tris_flat, bone_pos_np, bone_idx)

    def run(self):
        self.step_cnt = 1

        if self.renderer is not None:
            def on_key(symbol, modifiers):
                import pyglet
                if symbol == pyglet.window.key.R:
                    self.reset()
                    return pyglet.event.EVENT_HANDLED
                if symbol == pyglet.window.key.SPACE:
                    self.cfg.pause = not self.cfg.pause
                    return pyglet.event.EVENT_HANDLED
            self.renderer.register_key_press_callback(on_key)

        while self.step_cnt <= self.cfg.nsteps:
            if self.renderer is not None and not self.renderer.is_running():
                break
            if self.cfg.reset:
                self.reset()
                self.step_cnt = 1
                self.cfg.reset = False
            if not self.cfg.pause:
                wp.launch(fill_float_kernel, dim=self.activation.shape[0],
                          inputs=[self.activation, self.cfg.activation])
                self.step_start_time = time.perf_counter()
                self.step()
                self.step_cnt += 1
                self.step_end_time = time.perf_counter()
            self.render()

        if self.renderer is not None:
            self.renderer.save()


def main():
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "data/muscle/config/bicep.json"
    print("Using config:", config_path)
    cfg = load_config(config_path)
    cfg.gui = False
    sim = MuscleSim(cfg)
    print("Running for", cfg.nsteps, "steps.")
    sim.run()


if __name__ == "__main__":
    main()
