import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp
import warp.render
from scipy.spatial import cKDTree

@dataclass
class SimConfig:
    name: str = "MuscleSim"
    geo_path: Path = Path("data/muscle/model/bicep.geo")
    bone_geo_path: Path = Path("data/muscle/model/bicep_bone.geo")
    ground_mesh_path: Path = Path("data/muscle/model/ground.obj")
    coord_mesh_path: Path = Path("data/muscle/model/coord.obj")
    dt: float = 1e-3
    nsteps: int = 400
    num_substeps: int = 10  # number of substeps per main timestep for stability
    gravity: float = -9.8
    density: float = 1000.0
    veldamping: float = 0.02
    activation: float = 0.3
    constraints: list = None
    arch: str = "cuda"
    gui: bool = True
    render_mode: str = "human"  # "human" or "rgb_array" or None
    save_image: bool = False
    show_auxiliary_meshes: bool = False
    pause: bool = False
    reset: bool = False
    show_wireframe: bool = False
    render_fps: int = 24
    color_bones: bool = False
    color_muscles: str = "tendonmask"
    contraction_ratio: float = 0.4
    fiber_stiffness_scale: float = 200.0
    HAS_compressstiffness = False


# enum of constraint types, reference from pbd_types.h
DISTANCE      =  -264586729
BEND          =  5106433
STRETCHSHEAR  =  1143749888
BENDTWIST     =  1915235160
PIN           =  157323
ATTACH        =  1650556
PINORIENT     =  1780757740
PRESSURE      =  1396538961
TRIAREA       =  788656672
TETVOLUME     =  -215389979
TRIANGLEBEND  =  -120001733
ANGLE         =  187510551
TETFIBER      =  892515453
TETFIBERNORM  =  -303462111
PTPRIM        =  -600175536
DISTANCELINE  =  1621136047
DISTANCEPLANE =  -139877165
TRIARAP       =  788656539
TRIARAPNL     =  1634014773
TRIARAPNORM   =  -711728545
TETARAP       =  -92199131
TETARAPNL     =  -1666554577
TETARAPVOL    =  -1532966034
TETARAPNLVOL  =  1593379856
TETARAPNORM   =  -885573303
TETARAPNORMVOL=  -305911678
SHAPEMATCH    =  -841721998

# ARAP flags (from pbd_types.h)
LINEARENERGY = 1 << 0
NORMSTIFFNESS = 1 << 1


# ---------------------------------------------------------------------------
# Pure Python helpers
# ---------------------------------------------------------------------------

def constraint_alias(name: str) -> str:
    name = name.lower()
    if name == "stitch" or name == "branchstitch":
        return "distance"
    if name == "attachnormal":
        return "distanceline"
    return name


def load_config(path: Path) -> SimConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    from dataclasses import fields
    from types import SimpleNamespace
    kwargs = {}
    for fld in fields(SimConfig):
        name = fld.name
        if name in data:
            if name == "geo_path" or name == "bone_geo_path":
                kwargs[name] = Path(data[name])
            else:
                kwargs[name] = data[name]

    cfg = SimConfig(**kwargs)

    if "coupling" in data:
        cfg.coupling = SimpleNamespace(**data["coupling"])

    return cfg

def load_mesh_json(path):
    import json
    data = json.load(open(path, 'r'))
    positions = np.array(data["P"]).reshape(-1,3).copy()
    tets = np.array(data["tet"],dtype=int).reshape(-1,4).copy()
    nv = positions.shape[0]
    nt = tets.shape[0]
    tendon_mask = np.zeros((nv,), dtype=np.float32)
    fibers = np.zeros((nv,3), dtype=np.float32)
    return positions, tets, fibers, tendon_mask, None

def load_mesh_tetgen(path):
    import meshio
    def read_tet(filename):
        from pathlib import Path
        if Path(filename).suffix == "":
            filename += ".node"
        mesh = meshio.read(filename)
        pos = mesh.points
        tet_indices = mesh.cells_dict["tetra"]
        return pos, tet_indices
    positions,tets = read_tet(str(path))
    return positions, tets, None, None, None

def load_mesh_geo(path: Path):
    from VMuscle.geo import Geo
    geo = Geo(str(path))
    positions = np.asarray(geo.positions, dtype=np.float32)
    tets = np.asarray(geo.vert, dtype=np.int32)
    fibers = np.asarray(geo.materialW, dtype=np.float32) if hasattr(geo, "materialW") else None
    tendon_mask = (
        np.asarray(geo.tendonmask, dtype=np.float32) if hasattr(geo, "tendonmask") else None
    )
    return positions, tets, fibers, tendon_mask, geo

# just for testing
def load_mesh_one_tet(path: Path=None):
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    tets = np.array([
        [0, 2, 1, 3],
    ], dtype=np.int32)
    fibers = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    tendon_mask = np.array([
        0, 0, 0, 0,
    ], dtype=np.float32)
    return positions, tets, fibers, tendon_mask, None

def load_mesh_usd(path: Path):
    from VMuscle.mesh_io import load_mesh_usd as _load_mesh_usd
    return _load_mesh_usd(path, y_up_to_z_up=False)

def load_mesh(path: Path):
    if path is None:
        print("Using built-in one-tet mesh for testing.")
        return load_mesh_one_tet()
    elif str(path).endswith(".json"):
        return load_mesh_json(str(path))
    elif str(path).endswith(".node"):
        return load_mesh_tetgen(str(path))
    elif str(path).endswith(".geo"):
        return load_mesh_geo(path)
    elif str(path).endswith((".usd", ".usdc", ".usda")):
        return load_mesh_usd(path)


def build_surface_tris(tets: np.ndarray) -> np.ndarray:
    """Extract boundary faces (triangles) from tetrahedra."""
    faces = [
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3),
    ]
    counts = {}
    for tet in tets:
        for f in faces:
            tri = (tet[f[0]], tet[f[1]], tet[f[2]])
            key = tuple(sorted(tri))
            counts.setdefault(key, []).append(tri)
    surface = []
    for tris in counts.values():
        if len(tris) == 1:
            surface.append(tris[0])
    return np.asarray(surface, dtype=np.int32)


def get_bbox(pos):
    lowest_x = np.min(pos[:, 0])
    highest_x = np.max(pos[:, 0])
    lowest_y = np.min(pos[:, 1])
    highest_y = np.max(pos[:, 1])
    lowest_z = np.min(pos[:, 2])
    highest_z = np.max(pos[:, 2])
    bbox = np.array(
        [
            [lowest_x, lowest_y, lowest_z],
            [highest_x, highest_y, highest_z],
        ]
    )
    return bbox


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
class TriXformResult:
    xform: wp.mat33
    area: wp.float32

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
    if (ctype == TRIARAP or ctype == TETARAP or ctype == TETARAPVOL or
            ctype == TETARAPNORM or ctype == TETARAPNORMVOL):
        flags = flags | LINEARENERGY
    if (ctype == TRIARAPNORM or ctype == TETARAPNORM or ctype == TETARAPNORMVOL or
            ctype == TETFIBERNORM):
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


@wp.func
def ssvd_fn(F: wp.mat33) -> SVDResult:
    # MUST use local vars for svd3 output
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sigma = wp.vec3(0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    wp.svd3(F, U, sigma, V)

    # Check determinant of U: if negative, flip column 2 and sigma[2]
    detU = wp.determinant(U)
    if detU < 0.0:
        # flip column 2 of U
        U = wp.mat33(
            U[0, 0], U[0, 1], -U[0, 2],
            U[1, 0], U[1, 1], -U[1, 2],
            U[2, 0], U[2, 1], -U[2, 2])
        sigma = wp.vec3(sigma[0], sigma[1], -sigma[2])

    # Check determinant of V: if negative, flip column 2 and sigma[2]
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
    # sigma as diagonal matrix
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
def invariant4_fn(F: wp.mat33, fiber: wp.vec3) -> float:
    I4 = float(1.0)
    if wp.determinant(F) < 0.0:
        polar = polar_decomposition_fn(F)
        S = polar.S
        Sw = S * fiber
        I4 = wp.dot(fiber, Sw)
    return I4

@wp.func
def squared_norm2_fn(a00: float, a01: float, a10: float, a11: float) -> float:
    return a00 * a00 + a01 * a01 + a10 * a10 + a11 * a11

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
def triangle_xform_and_area_fn(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3) -> TriXformResult:
    e0 = p1 - p0
    e1 = p2 - p0
    n = wp.cross(e1, e0)
    nlen = wp.length(n)
    area = 0.5 * nlen
    xform = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if nlen > 1e-12:
        z = n / nlen
        y = wp.cross(e0, z)
        ylen = wp.length(y)
        if ylen > 1e-12:
            y = y / ylen
        x = wp.cross(y, z)
        # xform rows are x, y, z
        xform = wp.mat33(
            x[0], x[1], x[2],
            y[0], y[1], y[2],
            z[0], z[1], z[2])
    result = TriXformResult()
    result.xform = xform
    result.area = area
    return result


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

        # compression band check
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

            # Damping
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
    # isLINEARENERGY = True, isNORMSTIFFNESS = True, use_anisotropic_arap = False
    inv_mass0 = get_inv_mass_fn(pts[0], mass, stopped)
    inv_mass1 = get_inv_mass_fn(pts[1], mass, stopped)
    inv_mass2 = get_inv_mass_fn(pts[2], mass, stopped)
    inv_mass3 = get_inv_mass_fn(pts[3], mass, stopped)

    l = cons[cidx].L[0]
    alpha = 1.0 / stiffness
    # isNORMSTIFFNESS: divide by restlength
    alpha = alpha / restlength
    alpha = alpha / (dt * dt)
    grad_scale = float(1.0)
    psi = float(0.0)

    # _Ds = cols([pos[pts[i]] - pos[pts[3]] for i in range(3)])
    c0 = pos[pts[0]] - pos[pts[3]]
    c1 = pos[pts[1]] - pos[pts[3]]
    c2 = pos[pts[2]] - pos[pts[3]]
    # columns from c0, c1, c2 -> row-major mat33
    _Ds = wp.mat33(
        c0[0], c1[0], c2[0],
        c0[1], c1[1], c2[1],
        c0[2], c1[2], c2[2])

    F = _Ds * Dminv

    # Non-anisotropic path:
    # wTDminvT stored in restvector.xyz
    wTDminvT = wp.vec3(restvector[0], restvector[1], restvector[2])
    # FwT = wTDminvT @ _Ds^T = transpose(_Ds) * wTDminvT
    FwT = wp.transpose(_Ds) * wTDminvT
    psi = 0.5 * wp.length_sq(FwT)

    # Linear energy: C = sqrt(2*psi), gradscale = 1/sqrt(2*psi)
    # (isLINEARENERGY or isNORMSTIFFNESS) is always True here
    if psi > 1e-9:
        psi_sqrt = wp.sqrt(2.0 * psi)
        grad_scale = 1.0 / psi_sqrt
        psi = psi_sqrt

    # Piola-Kirchhoff: Ht = outer(wTDminvT, FwT)
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
            # isNORMSTIFFNESS
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
def distance_update_xpbd_fn(
    use_jacobi: int,
    cidx: int,
    cons: wp.array(dtype=Constraint),
    pts: wp.vec4i,
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
    pt0 = pts[0]
    pt1 = pts[1]
    p0 = pos[pt0]
    p1 = pos[pt1]
    invmass0 = get_inv_mass_fn(pt0, mass, stopped)
    invmass1 = get_inv_mass_fn(pt1, mass, stopped)
    wsum = invmass0 + invmass1
    if wsum != 0.0:
        n = p1 - p0
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
                    prev0 = pprev[pt0]
                    prev1 = pprev[pt1]
                    beta = kstiff_val * kdampratio * dt * dt
                    gamma = alpha * beta / dt
                    dsum = gamma * (-wp.dot(gradC, p0 - prev0) + wp.dot(gradC, p1 - prev1))
                    gamma = gamma + 1.0

                dL = (-C - alpha * l - dsum) / (gamma * wsum + alpha)
                dp = n * (-dL)

                if use_jacobi != 0:
                    update_dP(dP, dPw, invmass0 * dp, pt0)
                    update_dP(dP, dPw, -invmass1 * dp, pt1)
                else:
                    pos[pt0] = pos[pt0] + invmass0 * dp
                    pos[pt1] = pos[pt1] - invmass1 * dp
                    cons[cidx].L = set_L_component(cons[cidx].L, loff, get_L_component(cons[cidx].L, loff) + dL)


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

                # Muscle side correction
                if use_jacobi != 0:
                    update_dP(dP, dPw, -invmass1 * dp, pt1)
                else:
                    pos[pt1] = pos[pt1] - invmass1 * dp
                    cons[cidx].L = set_L_component(cons[cidx].L, loff, get_L_component(cons[cidx].L, loff) + dL)

                # Bilateral reaction accumulator
                wp.atomic_add(reaction_accum, cidx, C * n)


@wp.func
def tri_arap_update_xpbd_fn(
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
    restvector: wp.vec4f,
    kstiff: float,
    kdampratio: float,
    kstiffcompress: float,
    flags: int,
    has_compress_stiffness: int,
):
    pt0 = pts[0]
    pt1 = pts[1]
    pt2 = pts[2]
    p0 = pos[pt0]
    p1 = pos[pt1]
    p2 = pos[pt2]

    invmass0 = get_inv_mass_fn(pt0, mass, stopped)
    invmass1 = get_inv_mass_fn(pt1, mass, stopped)
    invmass2 = get_inv_mass_fn(pt2, mass, stopped)

    # Dminv 2x2 stored in restvector
    Dminv00 = restvector[0]
    Dminv01 = restvector[1]
    Dminv10 = restvector[2]
    Dminv11 = restvector[3]

    tri_res = triangle_xform_and_area_fn(p0, p1, p2)
    xform = tri_res.xform
    area = tri_res.area
    if area != 0.0:
        comp = 0
        if has_compress_stiffness != 0:
            if area < restlength:
                comp = 1
        kstiff_val = wp.where(comp != 0, kstiffcompress, kstiff)
        loff = wp.min(comp, 2)
        if kstiff_val != 0.0:
            l = get_L_component(cons[cidx].L, loff)

            # Project to 2D via xform rows
            row0 = wp.vec3(xform[0, 0], xform[0, 1], xform[0, 2])
            row1 = wp.vec3(xform[1, 0], xform[1, 1], xform[1, 2])
            P0x = wp.dot(row0, p0)
            P0y = wp.dot(row1, p0)
            P1x = wp.dot(row0, p1)
            P1y = wp.dot(row1, p1)
            P2x = wp.dot(row0, p2)
            P2y = wp.dot(row1, p2)

            # Ds 2x2: cols([P0-P2, P1-P2])
            Ds00 = P0x - P2x
            Ds10 = P0y - P2y
            Ds01 = P1x - P2x
            Ds11 = P1y - P2y

            # F = Ds @ Dminv (2x2 matmul)
            F00 = Ds00 * Dminv00 + Ds01 * Dminv10
            F01 = Ds00 * Dminv01 + Ds01 * Dminv11
            F10 = Ds10 * Dminv00 + Ds11 * Dminv10
            F11 = Ds10 * Dminv01 + Ds11 * Dminv11

            # m = [F00+F11, F10-F01]
            mx = F00 + F11
            my = F10 - F01
            mlen = wp.sqrt(mx * mx + my * my)
            if mlen >= 1e-9:
                mx = mx / mlen
                my = my / mlen
                # R = [[mx, -my], [my, mx]]
                R00 = mx
                R01 = -my
                R10 = my
                R11 = mx

                # d = F - R
                d00 = F00 - R00
                d01 = F01 - R01
                d10 = F10 - R10
                d11 = F11 - R11
                psi = squared_norm2_fn(d00, d01, d10, d11)
                gradscale = float(2.0)
                if (flags & (LINEARENERGY | NORMSTIFFNESS)) != 0:
                    psi = wp.sqrt(psi)
                    gradscale = 1.0 / psi

                if psi >= 1e-6:
                    # Ht = Dminv @ d^T (2x2)
                    # d^T: [d00,d10; d01,d11]
                    Ht00 = Dminv00 * d00 + Dminv01 * d10
                    Ht01 = Dminv00 * d01 + Dminv01 * d11
                    Ht10 = Dminv10 * d00 + Dminv11 * d10
                    Ht11 = Dminv10 * d01 + Dminv11 * d11

                    grad0_2d_x = Ht00 * gradscale
                    grad0_2d_y = Ht01 * gradscale
                    grad1_2d_x = Ht10 * gradscale
                    grad1_2d_y = Ht11 * gradscale

                    # 3D grad in 2D tri space with z=0
                    g0_local = wp.vec3(grad0_2d_x, grad0_2d_y, 0.0)
                    g1_local = wp.vec3(grad1_2d_x, grad1_2d_y, 0.0)
                    g2_local = -g0_local - g1_local

                    # Transform back to world: xform^T * g
                    xformT = wp.transpose(xform)
                    grad0 = xformT * g0_local
                    grad1 = xformT * g1_local
                    grad2 = xformT * g2_local

                    wsum = (invmass0 * wp.dot(grad0, grad0) +
                            invmass1 * wp.dot(grad1, grad1) +
                            invmass2 * wp.dot(grad2, grad2))

                    if wsum != 0.0:
                        alpha = 1.0 / kstiff_val
                        if (flags & NORMSTIFFNESS) != 0:
                            alpha = alpha / restlength
                        alpha = alpha / (dt * dt)

                        dsum = float(0.0)
                        gamma = float(1.0)
                        if kdampratio > 0.0:
                            prev0 = pprev[pt0]
                            prev1 = pprev[pt1]
                            prev2 = pprev[pt2]
                            beta = kstiff_val * kdampratio * dt * dt
                            if (flags & NORMSTIFFNESS) != 0:
                                beta = beta * restlength
                            gamma = alpha * beta / dt
                            dsum = (wp.dot(grad0, p0 - prev0) +
                                    wp.dot(grad1, p1 - prev1) +
                                    wp.dot(grad2, p2 - prev2))
                            dsum = dsum * gamma
                            gamma = gamma + 1.0

                        C = psi
                        dL = (-C - alpha * l - dsum) / (gamma * wsum + alpha)
                        if use_jacobi != 0:
                            update_dP(dP, dPw, dL * invmass0 * grad0, pt0)
                            update_dP(dP, dPw, dL * invmass1 * grad1, pt1)
                            update_dP(dP, dPw, dL * invmass2 * grad2, pt2)
                        else:
                            pos[pt0] = pos[pt0] + dL * invmass0 * grad0
                            pos[pt1] = pos[pt1] + dL * invmass1 * grad1
                            pos[pt2] = pos[pt2] + dL * invmass2 * grad2
                            cons[cidx].L = set_L_component(cons[cidx].L, loff, get_L_component(cons[cidx].L, loff) + dL)


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

    # Ds = cols([p0-p3, p1-p3, p2-p3])
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

    # Store quaternion of R in restvector
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
    # Dm = cols([pos0[pts[i]] - pos0[pts[3]] for i in range(3)])
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
def compute_cell_fiber_dir_kernel(
    tet_indices: wp.array(dtype=wp.vec4i),
    v_fiber_dir: wp.array(dtype=wp.vec3),
    c_fiber_dir: wp.array(dtype=wp.vec3),
):
    c = wp.tid()
    pts = tet_indices[c]
    v_dirs = wp.vec3(0.0)
    for i in range(4):
        v_dirs = v_dirs + v_fiber_dir[pts[i]]
    norm = wp.length(v_dirs) + 1e-8
    c_fiber_dir[c] = v_dirs / norm


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


@wp.kernel
def solve_constraints_kernel(
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
    reaction_accum: wp.array(dtype=wp.vec3),
    dt: float,
    use_jacobi: int,
    contraction_ratio: float,
    fiber_stiffness_scale: float,
    has_compress_stiffness: int,
    n_cons: int,
):
    c = wp.tid()
    if c >= n_cons:
        return

    # Common setup
    cstiffness = cons[c].stiffness
    if cstiffness <= 0.0:
        return

    # Compression stiffness: defaults to -1, meaning use regular stiffness
    kstiffcompress = cons[c].compressionstiffness
    kstiffcompress = wp.where(kstiffcompress >= 0.0, kstiffcompress, cstiffness)

    ctype = cons[c].type
    pts = cons[c].pts

    if ctype == TETVOLUME:
        tetid = cons[c].tetid
        tet_volume_update_xpbd_fn(
            use_jacobi, c, cons, pos, pprev,
            cons[c].restlength, dP, dPw, tetid, pts,
            dt, cstiffness, kstiffcompress, cons[c].dampingratio,
            mass, stopped, has_compress_stiffness)

    elif ctype == TETFIBERNORM:
        tetid = cons[c].tetid
        fiber_dir = (v_fiber_dir[pts[0]] + v_fiber_dir[pts[1]] +
                     v_fiber_dir[pts[2]] + v_fiber_dir[pts[3]]) / 4.0
        Dminv = rest_matrix[tetid]
        acti = activation[tetid]
        _tendonmask = tendonmask[tetid]
        belly_factor = 1.0 - _tendonmask

        # Stiffness: activation-dependent via transfer_tension
        fiberscale = transfer_tension_fn(acti, _tendonmask, 0.0, 10.0)
        stiffness_val = cstiffness * fiberscale * fiber_stiffness_scale
        if stiffness_val > 0.0:
            # Target fiber stretch
            target_stretch = 1.0 - belly_factor * acti * contraction_ratio

            tet_fiber_update_xpbd_fn(
                use_jacobi, pos, pprev, dP, dPw,
                c, cons, pts, dt, fiber_dir, stiffness_val, Dminv,
                cons[c].dampingratio, cons[c].restlength, cons[c].restvector,
                acti, mass, stopped, target_stretch)

    elif ctype == DISTANCE:
        distance_update_xpbd_fn(
            use_jacobi, c, cons, pts, pos, pprev, dP, dPw,
            mass, stopped, cons[c].restlength, cstiffness,
            cons[c].dampingratio, kstiffcompress, dt, has_compress_stiffness)

    elif ctype == ATTACH:
        pt_src = pts[0]
        rv = cons[c].restvector
        p0_target = wp.vec3(rv[0], rv[1], rv[2])
        attach_bilateral_update_fn(
            use_jacobi, c, cons, pt_src, p0_target,
            pos, pprev, dP, dPw, mass, stopped,
            cons[c].restlength, cstiffness, cons[c].dampingratio,
            kstiffcompress, dt, has_compress_stiffness, reaction_accum)

    elif ctype == PIN:
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

    elif ctype == DISTANCELINE:
        pt_src = pts[0]
        p_src = pos[pt_src]
        rv = cons[c].restvector
        line_origin = wp.vec3(rv[0], rv[1], rv[2])
        line_dir = cons[c].restdir

        # Project source point onto the line
        p_projected = project_to_line_fn(p_src, line_origin, line_dir)

        distance_pos_update_xpbd_fn(
            use_jacobi, c, cons, -1, pt_src,
            p_projected, p_src,
            pos, pprev, dP, dPw, mass, stopped,
            cons[c].restlength, cstiffness, cons[c].dampingratio,
            kstiffcompress, dt, has_compress_stiffness)

    elif ctype == TETARAP:
        tetid = cons[c].tetid
        flags = fem_flags_fn(ctype)
        tet_arap_update_xpbd_fn(
            use_jacobi, c, cons, pts, dt, pos, pprev, dP, dPw,
            mass, stopped, cons[c].restlength, rest_matrix[tetid],
            cstiffness, cons[c].dampingratio, flags)

    elif ctype == TRIARAP:
        flags = fem_flags_fn(ctype)
        tri_arap_update_xpbd_fn(
            use_jacobi, c, cons, pts, dt, pos, pprev, dP, dPw,
            mass, stopped, cons[c].restlength, cons[c].restvector,
            cstiffness, cons[c].dampingratio, kstiffcompress,
            flags, has_compress_stiffness)


# ---------------------------------------------------------------------------
# WarpRenderer — OpenGL (interactive) and USD (offline) rendering
# ---------------------------------------------------------------------------

class WarpRenderer:
    """Wraps wp.render.OpenGLRenderer or wp.render.UsdRenderer.

    Args:
        mode: "human" for OpenGL interactive, "usd" for USD file output, None to disable.
        stage_path: Output USD file path (only used when mode="usd").
    """

    def __init__(self, mode: str = "human", stage_path: str = "muscle_sim.usd"):
        self.mode = mode
        self.renderer = None
        if mode == "human":
            self.renderer = wp.render.OpenGLRenderer(vsync=True)
        elif mode == "usd":
            self.renderer = wp.render.UsdRenderer(stage_path)
        self.sim_time = 0.0

    def is_running(self) -> bool:
        if self.renderer is None:
            return True
        if self.mode == "human":
            return self.renderer.is_running()
        return True

    def update(self, sim_time: float, pos_np: np.ndarray, tri_indices: np.ndarray,
               bone_pos_np: np.ndarray = None, bone_indices: np.ndarray = None):
        """Render one frame: muscle mesh + optional bone mesh."""
        if self.renderer is None:
            return
        self.sim_time = sim_time

        self.renderer.begin_frame(sim_time)
        self.renderer.render_mesh(
            name="muscle",
            points=pos_np,
            indices=tri_indices,
            colors=(0.8, 0.3, 0.2),
        )
        if bone_pos_np is not None and bone_indices is not None and len(bone_indices) > 0:
            self.renderer.render_mesh(
                name="bone",
                points=bone_pos_np,
                indices=bone_indices,
                colors=(0.9, 0.9, 0.85),
            )
        self.renderer.end_frame()

    def save(self):
        """Save USD file (no-op for OpenGL)."""
        if self.renderer is not None and self.mode == "usd":
            self.renderer.save()

    def register_key_press_callback(self, callback):
        """Register keyboard callback (OpenGL only)."""
        if self.mode == "human" and self.renderer is not None:
            self.renderer.register_key_press_callback(callback)


# ---------------------------------------------------------------------------
# MuscleSim class
# ---------------------------------------------------------------------------

class MuscleSim:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        wp.init()

        self.constraint_configs = self.cfg.constraints if self.cfg.constraints else []

        print("Loading mesh from:", cfg.geo_path)
        mesh_data = load_mesh(cfg.geo_path)
        self.pos0_np, self.tet_np, self.v_fiber_np, self.v_tendonmask_np, self.geo = mesh_data
        self.n_verts = self.pos0_np.shape[0]
        print(f"Loaded mesh: {self.n_verts} vertices, {self.tet_np.shape[0]} tetrahedra.")

        print("Loading bone mesh:", cfg.bone_geo_path)
        self.load_bone_geo(cfg.bone_geo_path)

        print("Allocating&initializing fields...")
        self._allocate_fields()
        self._init_fields()
        self._precompute_rest()
        self._build_surface_tris()
        self.build_constraints()

        self.use_jacobi = False
        self.contraction_ratio = self.cfg.contraction_ratio
        self.fiber_stiffness_scale = self.cfg.fiber_stiffness_scale
        self.dt = self.cfg.dt / self.cfg.num_substeps
        self.step_cnt = 0

        # Renderer setup
        self.renderer = None
        if cfg.gui and cfg.render_mode in ("human", "usd"):
            stage_path = "muscle_sim.usd" if cfg.render_mode == "usd" else None
            self.renderer = WarpRenderer(mode=cfg.render_mode, stage_path=stage_path)

        print("All initialization done.")


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


    # Reference: pbd_constraints.h:L1291
    def _batch_compute_tet_rest_matrices(self):
        if hasattr(self, '_cached_tet_rest'):
            return self._cached_tet_rest
        tet_pos = self.pos0_np[self.tet_np]
        cols = tet_pos[:, :3, :] - tet_pos[:, 3:4, :]
        M = np.transpose(cols, (0, 2, 1))
        dets = np.linalg.det(M)
        volumes = dets / 6.0
        valid = np.abs(dets) > 1e-30
        restmatrices = np.zeros_like(M)
        if np.any(valid):
            restmatrices[valid] = np.linalg.inv(M[valid])
        self._cached_tet_rest = (restmatrices, volumes, valid)
        return self._cached_tet_rest

    def compute_tet_rest_matrix(self, pt0, pt1, pt2, pt3, scale=1.0):
        p = self.pos0_np
        p0 = p[pt0]
        p1 = p[pt1]
        p2 = p[pt2]
        p3 = p[pt3]
        M = scale * np.stack([p0 - p3, p1 - p3, p2 - p3]).T
        detM = np.linalg.det(M)
        if detM == 0:
            return None, 0.0
        restmatrix = np.linalg.inv(M)
        volume = detM / 6.0
        return restmatrix, volume

    def compute_tri_rest_matrix(self, pt0, pt1, pt2, scale=1.0):
        p = self.pos0_np
        p0 = p[pt0]
        p1 = p[pt1]
        p2 = p[pt2]
        e0 = p1 - p0
        e1 = p2 - p0
        n = np.cross(e1, e0)
        nlen = np.linalg.norm(n)
        if nlen < 1e-12:
            return None, 0.0
        z = n / nlen
        y = np.cross(e0, z)
        ylen = np.linalg.norm(y)
        if ylen < 1e-12:
            return None, 0.0
        y = y / ylen
        x = np.cross(y, z)

        xform = np.stack([x, y, z], axis=0).T
        P0 = p0 @ xform
        P1 = p1 @ xform
        P2 = p2 @ xform
        M = scale * np.column_stack([P0[:2] - P2[:2], P1[:2] - P2[:2]])
        detM = np.linalg.det(M)
        if detM == 0:
            return None, 0.0
        restmatrix = np.linalg.inv(M)
        area = abs(detM / 2.0)
        return restmatrix, area

    def compute_tet_fiber_rest_length(self, pt0, pt1, pt2, pt3):
        restm, volume = self.compute_tet_rest_matrix(pt0, pt1, pt2, pt3, scale=1.0)
        if restm is None:
            return 0.0, np.array([0.0, 0.0, 1.0], dtype=np.float32)

        materialW = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if self.v_fiber_np is not None:
            w = self.v_fiber_np[pt0] + self.v_fiber_np[pt1] + self.v_fiber_np[pt2] + self.v_fiber_np[pt3]
            norm = np.linalg.norm(w)
            if norm > 1e-8:
                materialW = w / norm
        materialW = materialW @ restm.T
        return volume, materialW


    # Reference: pbd_constraints.h:L1291
    def create_tet_fiber_constraint(self, params):
        """Vectorized: compute fiber constraints for all tets using batch rest matrices."""
        stiffness = params.get('stiffness', 1.0)
        dampingratio = params.get('dampingratio', 0.0)
        restmatrices, volumes, valid = self._batch_compute_tet_rest_matrices()

        n_tet = len(self.tet_np)
        if self.v_fiber_np is not None:
            fiber_verts = self.v_fiber_np[self.tet_np]
            w = fiber_verts.sum(axis=1)
            norms = np.linalg.norm(w, axis=1, keepdims=True)
            default_w = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (n_tet, 1))
            norm_ok = (norms > 1e-8).ravel()
            materialW = default_w.copy()
            materialW[norm_ok] = w[norm_ok] / norms[norm_ok]
        else:
            materialW = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (n_tet, 1))

        materialW_transformed = np.einsum('nj,nkj->nk', materialW, restmatrices)

        constraints = []
        for i in range(n_tet):
            tet = self.tet_np[i]
            if not valid[i]:
                vol = 0.0
                mw_t = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                vol = float(volumes[i])
                mw_t = materialW_transformed[i]
            restvec4 = [float(mw_t[0]), float(mw_t[1]), float(mw_t[2]), 1.0]
            c = dict(
                type=TETFIBERNORM,
                pts=[int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3])],
                stiffness=stiffness,
                dampingratio=dampingratio,
                tetid=i,
                L=[0.0, 0.0, 0.0],
                restlength=float(vol),
                restvector=restvec4,
                restdir=[0.0, 0.0, 0.0],
                compressionstiffness=-1.0
            )
            constraints.append(c)
        return constraints

    def create_pin_constraints(self, params):
        constraints = []
        pin_mask = None
        try:
            from geo import Geo
            geo = Geo(str(self.cfg.geo_path))
            if hasattr(geo, "gluetoanimation"):
                pin_mask = np.asarray(geo.gluetoanimation, dtype=np.float32)
            elif hasattr(geo, "pin"):
                pin_mask = np.asarray(geo.pin, dtype=np.float32)
        except Exception:
            pin_mask = None

        if pin_mask is None:
            return constraints

        stiffness = params.get('stiffness', 1e10)
        dampingratio = params.get('dampingratio', 0.0)

        for i, val in enumerate(pin_mask):
            if val > 0.5:
                restpos = self.pos0_np[i]
                c = dict(
                    type=PIN,
                    pts=[int(i), -1, 0, 0],
                    stiffness=stiffness,
                    dampingratio=dampingratio,
                    tetid=0,
                    L=[0.0, 0.0, 0.0],
                    restlength=0.0,
                    restvector=[float(restpos[0]), float(restpos[1]), float(restpos[2]), 1.0],
                    restdir=[0.0, 0.0, 0.0],
                    compressionstiffness=-1.0,
                )
                constraints.append(c)
        return constraints


    def map_pts2tets(self, tet):
        pt2tet = {}
        for i, tet_verts in enumerate(tet):
            for v in tet_verts:
                if v not in pt2tet:
                    pt2tet[v] = []
                pt2tet[v].append(i)
        return pt2tet


    def one2multi_dict_to_np(self, one2multi_dict, n_verts):
        max_len = max(len(v) for v in one2multi_dict.values())
        result = -np.ones((n_verts, max_len), dtype=np.int32)
        for k, v in one2multi_dict.items():
            result[k, :len(v)] = v
        return result


    def load_bone_geo(self, target_path):
        if not hasattr(self, 'bone_pos') and Path(target_path).exists():
            is_usd = str(target_path).endswith((".usd", ".usdc", ".usda"))
            if is_usd:
                self._load_bone_from_usd(target_path)
            else:
                self._load_bone_from_geo(target_path)

            if self.bone_pos.shape[0] > 0:
                self.bone_pos_field = wp.from_numpy(self.bone_pos.astype(np.float32), dtype=wp.vec3)
                if self.bone_indices_np.shape[0] > 0:
                    self.bone_indices_field = wp.from_numpy(self.bone_indices_np.astype(np.int32), dtype=wp.int32)
                else:
                    self.bone_indices_field = None
                if self.bone_vertex_colors is not None:
                    self.bone_colors_np = self.bone_vertex_colors
                else:
                    self.bone_colors_np = None
            else:
                self.bone_pos_field = None
                self.bone_indices_field = None
                self.bone_colors_np = None

        if hasattr(self, 'bone_pos'):
            return self.bone_geo, self.bone_pos
        else:
            return None, np.zeros((0,3), dtype=np.float32)

    def _load_bone_from_geo(self, target_path):
        from VMuscle.geo import Geo
        self.bone_geo = Geo(target_path)
        if len(self.bone_geo.positions) == 0:
            print(f"Warning: No vertices found in {target_path}")
            self.bone_pos = np.zeros((0, 3), dtype=np.float32)
            self.bone_muscle_ids = {}
        else:
            self.bone_pos = np.asarray(self.bone_geo.positions, dtype=np.float32)
            if hasattr(self.bone_geo, 'indices'):
                self.bone_indices_np = np.asarray(self.bone_geo.indices, dtype=np.int32)
            elif hasattr(self.bone_geo, 'vert'):
                self.bone_indices_np = np.array(self.bone_geo.vert, dtype=np.int32).flatten()
            else:
                self.bone_indices_np = np.zeros(0, dtype=np.int32)

            self.bone_muscle_ids = {}
            self.bone_vertex_colors = None

            if hasattr(self.bone_geo, 'pointattr') and 'muscle_id' in self.bone_geo.pointattr:
                muscle_ids = self.bone_geo.pointattr['muscle_id']
                self._build_bone_muscle_id_mapping(muscle_ids)

    def _load_bone_from_usd(self, usd_path):
        from VMuscle.mesh_io import load_bone_usd_data
        positions, indices, muscle_id_per_vertex = load_bone_usd_data(usd_path)
        self.bone_geo = None
        if len(positions) == 0:
            print(f"Warning: No bone vertices found in {usd_path}")
            self.bone_pos = np.zeros((0, 3), dtype=np.float32)
            self.bone_muscle_ids = {}
        else:
            self.bone_pos = positions
            self.bone_indices_np = indices
            self.bone_muscle_ids = {}
            self.bone_vertex_colors = None
            if muscle_id_per_vertex:
                self._build_bone_muscle_id_mapping(muscle_id_per_vertex)

    def _build_bone_muscle_id_mapping(self, muscle_ids):
        for v_idx, mid in enumerate(muscle_ids):
            if mid not in self.bone_muscle_ids:
                self.bone_muscle_ids[mid] = []
            self.bone_muscle_ids[mid].append(v_idx)
        for mid in self.bone_muscle_ids:
            self.bone_muscle_ids[mid] = np.array(self.bone_muscle_ids[mid], dtype=np.int32)

        print(f"Bone muscle_id groups: {list(self.bone_muscle_ids.keys())}")

        if self.cfg.color_bones:
            unique_ids = sorted(self.bone_muscle_ids.keys())
            self.bone_id_colors = generate_muscle_id_colors(unique_ids)
            self.bone_vertex_colors = np.zeros((len(muscle_ids), 3), dtype=np.float32)
            for v_idx, mid in enumerate(muscle_ids):
                self.bone_vertex_colors[v_idx] = self.bone_id_colors[mid]
            print("Bone coloring by muscle_id enabled")


    def create_attach_constraints(self, params):
        constraints = []
        mask_name = params.get('mask_name')
        target_path = params.get('target_path')
        mask_threshold = params.get('mask_threshold', 0.75)
        stiffness = params.get('stiffness', 1e10)
        dampingratio = params.get('dampingratio', 0.0)

        self.bone_geo, self.bone_pos = self.load_bone_geo(target_path)

        mask = np.asarray(getattr(self.geo, mask_name), dtype=np.float32) if hasattr(self.geo, mask_name) else None

        if mask is None:
            Warning(f"Warning: mask '{mask_name}' not found in geometry.")
            return []

        valid_src_indices = np.where(mask > mask_threshold)[0].astype(np.int32)

        if len(valid_src_indices) == 0:
            Warning(f"Warning: No vertices with mask > {mask_threshold}")
            return []

        bone_tree = cKDTree(self.bone_pos)
        src_positions = self.pos0_np[valid_src_indices]
        dists, tgt_indices = bone_tree.query(src_positions, k=1)

        if not hasattr(self, 'pt2tet'):
            self.pt2tet = self.map_pts2tets(self.tet_np)

        for j, src_idx in enumerate(valid_src_indices):
            tgt_idx = int(tgt_indices[j])
            target_pos = self.bone_pos[tgt_idx]
            src_pos = src_positions[j]
            tetid = self.pt2tet.get(src_idx, [-1])[0]
            restlength = float(dists[j])

            c = dict(
                type=ATTACH,
                pts=[int(src_idx), -1, int(tgt_idx), 0],
                stiffness=stiffness,
                dampingratio=dampingratio,
                tetid=tetid,
                L=[0.0, 0.0, 0.0],
                restlength=restlength,
                restvector=[float(target_pos[0]), float(target_pos[1]), float(target_pos[2]), 1.0],
                restdir=[0.0, 0.0, 0.0],
                compressionstiffness=-1.0,
            )
            constraints.append(c)

        return constraints

    def create_distance_line_constraints(self, params):
        constraints = []
        mask_name = params.get('mask_name')
        target_path = params.get('target_path')
        mask_threshold = params.get('mask_threshold', 0.75)
        stiffness = params.get('stiffness', 1e10)
        dampingratio = params.get('dampingratio', 0.0)

        self.bone_geo, self.bone_pos = self.load_bone_geo(target_path)

        mask = np.asarray(getattr(self.geo, mask_name), dtype=np.float32) if hasattr(self.geo, mask_name) else None

        if mask is None:
            Warning(f"Warning: mask '{mask_name}' not found in geometry.")
            return []

        valid_src_indices = np.where(mask > mask_threshold)[0].astype(np.int32)

        if len(valid_src_indices) == 0:
            Warning(f"Warning: No vertices with mask > {mask_threshold}")
            return []

        bone_tree = cKDTree(self.bone_pos)
        src_positions = self.pos0_np[valid_src_indices]
        dists, tgt_indices = bone_tree.query(src_positions, k=1)

        target_positions = self.bone_pos[tgt_indices]
        directions = target_positions - src_positions
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        if np.any(norms < 1e-9):
            bad = np.where(norms.ravel() < 1e-9)[0]
            raise ValueError(f"source and target points too close at indices: {valid_src_indices[bad]}")
        directions = directions / norms

        if not hasattr(self, 'pt2tet'):
            self.pt2tet = self.map_pts2tets(self.tet_np)

        for j, src_idx in enumerate(valid_src_indices):
            tgt_idx = int(tgt_indices[j])
            target_pos = target_positions[j]
            direction = directions[j]
            tetid = self.pt2tet.get(src_idx, [-1])[0]

            c = dict(
                type=DISTANCELINE,
                pts=[int(src_idx), int(tgt_idx), 0, 0],
                stiffness=stiffness,
                dampingratio=dampingratio,
                tetid=tetid,
                L=[0.0, 0.0, 0.0],
                restlength=0.0,
                restvector=[float(target_pos[0]), float(target_pos[1]), float(target_pos[2]), 1.0],
                restdir=[float(direction[0]), float(direction[1]), float(direction[2])],
                compressionstiffness=-1.0,
            )
            constraints.append(c)

        return constraints

    def create_tet_arap_constraints(self, params):
        stiffness = params.get('stiffness', 1e10)
        dampingratio = params.get('dampingratio', 0.0)
        restmatrices, volumes, valid = self._batch_compute_tet_rest_matrices()

        valid_indices = np.where(valid)[0]
        constraints = []
        for i in valid_indices:
            tet = self.tet_np[i]
            c = dict(
                type=TETARAP,
                pts=[int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3])],
                stiffness=stiffness,
                dampingratio=dampingratio,
                tetid=int(i),
                L=[0.0, 0.0, 0.0],
                restlength=float(volumes[i]),
                restvector=[0.0, 0.0, 0.0, 1.0],
                restdir=[0.0, 0.0, 0.0],
                compressionstiffness=-1.0,
            )
            constraints.append(c)
        return constraints

    def create_tri_arap_constraints(self, params):
        stiffness = params.get('stiffness', 1e10)
        dampingratio = params.get('dampingratio', 0.0)

        if not hasattr(self, 'surface_tris'):
            self.surface_tris = build_surface_tris(self.tet_np)
        tris = self.surface_tris
        if len(tris) == 0:
            return []

        p = self.pos0_np
        tri_arr = np.asarray(tris)
        p0 = p[tri_arr[:, 0]]
        p1 = p[tri_arr[:, 1]]
        p2 = p[tri_arr[:, 2]]
        e0 = p1 - p0
        e1 = p2 - p0
        n = np.cross(e1, e0)
        nlen = np.linalg.norm(n, axis=1)
        valid1 = nlen > 1e-12
        z = np.zeros_like(n)
        z[valid1] = n[valid1] / nlen[valid1, None]
        y = np.cross(e0, z)
        ylen = np.linalg.norm(y, axis=1)
        valid2 = ylen > 1e-12
        valid = valid1 & valid2
        y[valid] = y[valid] / ylen[valid, None]
        x = np.cross(y, z)

        xform = np.stack([x, y, z], axis=1)
        xformT = np.transpose(xform, (0, 2, 1))
        P0 = np.einsum('ni,nij->nj', p0, xformT)
        P1 = np.einsum('ni,nij->nj', p1, xformT)
        P2 = np.einsum('ni,nij->nj', p2, xformT)
        col0 = P0[:, :2] - P2[:, :2]
        col1 = P1[:, :2] - P2[:, :2]
        M = np.stack([col0, col1], axis=2)
        dets = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
        valid = valid & (np.abs(dets) > 1e-30)
        areas = np.abs(dets / 2.0)
        restm_all = np.zeros((len(tris), 2, 2), dtype=np.float64)
        inv_dets = np.zeros(len(tris))
        inv_dets[valid] = 1.0 / dets[valid]
        restm_all[valid, 0, 0] = M[valid, 1, 1] * inv_dets[valid]
        restm_all[valid, 0, 1] = -M[valid, 0, 1] * inv_dets[valid]
        restm_all[valid, 1, 0] = -M[valid, 1, 0] * inv_dets[valid]
        restm_all[valid, 1, 1] = M[valid, 0, 0] * inv_dets[valid]

        valid_indices = np.where(valid)[0]
        constraints = []
        for i in valid_indices:
            tri = tri_arr[i]
            rm = restm_all[i]
            restvec4 = [float(rm[0, 0]), float(rm[0, 1]), float(rm[1, 0]), float(rm[1, 1])]
            c = dict(
                type=TRIARAP,
                pts=[int(tri[0]), int(tri[1]), int(tri[2]), 0],
                stiffness=stiffness,
                dampingratio=dampingratio,
                tetid=int(i),
                L=[0.0, 0.0, 0.0],
                restlength=float(areas[i]),
                restvector=restvec4,
                restdir=[0.0, 0.0, 0.0],
                compressionstiffness=-1.0,
            )
            constraints.append(c)
        return constraints

    # Reference: pbd_constraints.h:L980
    def create_tet_volume_constraint(self, params):
        stiffness = params.get('stiffness', 1e10)
        dampingratio = params.get('dampingratio', 0.0)
        restmatrices, volumes, valid = self._batch_compute_tet_rest_matrices()

        valid_indices = np.where(valid)[0]
        constraints = []
        for i in valid_indices:
            tet = self.tet_np[i]
            c = dict(
                type=TETVOLUME,
                pts=[int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3])],
                stiffness=stiffness,
                dampingratio=dampingratio,
                compressionstiffness=-1.0,
                tetid=int(i),
                L=[0.0, 0.0, 0.0],
                restlength=float(volumes[i]),
                restvector=[0.0, 0.0, 0.0, 0.0],
                restdir=[0.0, 0.0, 0.0],
            )
            constraints.append(c)
        return constraints


    def build_constraints(self):
        print("Building constraints...")

        self.tetvolume_constraints = []
        self.tetfiber_constraints = []
        self.pin_constraints = []
        self.attach_constraints = []
        self.distanceline_constraints = []
        self.triarap_constraints = []
        self.tetarap_constraints = []

        all_constraints = []
        _t_total = time.perf_counter()
        for params in self.constraint_configs:
            ctype = constraint_alias(params['type'])
            new_constraints = []
            if ctype == 'volume':
                new_constraints = self.create_tet_volume_constraint(params)
                self.tetvolume_constraints.extend(new_constraints)
            elif ctype == 'fiber':
                new_constraints = self.create_tet_fiber_constraint(params)
                self.tetfiber_constraints.extend(new_constraints)
            elif ctype == 'pin':
                new_constraints = self.create_pin_constraints(params)
                self.pin_constraints.extend(new_constraints)
            elif ctype == 'attach':
                new_constraints = self.create_attach_constraints(params)
                self.attach_constraints.extend(new_constraints)
            elif ctype == 'distanceline':
                new_constraints = self.create_distance_line_constraints(params)
                self.distanceline_constraints.extend(new_constraints)
            elif ctype == 'tetarap':
                new_constraints = self.create_tet_arap_constraints(params)
                self.tetarap_constraints.extend(new_constraints)
            elif ctype == 'triarap':
                new_constraints = self.create_tri_arap_constraints(params)
                self.triarap_constraints.extend(new_constraints)

            if new_constraints:
                all_constraints.extend(new_constraints)
                print(f"  {params.get('name', ctype)} ({ctype}): {len(new_constraints)} constraints")

        self.raw_constraints = all_constraints.copy()

        n_cons = len(all_constraints)
        self.n_cons = n_cons
        if n_cons > 0:
            for i, c in enumerate(all_constraints):
                c['cidx'] = i

            # Build numpy structured array matching Constraint struct layout
            # Then create warp array
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

            # Build constraint array using numpy structured array
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

        _dt_total = time.perf_counter() - _t_total
        print(f"Built {n_cons} constraints total. [{_dt_total*1000:.0f}ms]")

        # Reaction accumulator for bilateral attach coupling
        self.reaction_accum = wp.zeros(max(n_cons, 1), dtype=wp.vec3)


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

        # Compute per-cell tendon mask
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

    def solve_constraints(self):
        if self.n_cons == 0:
            return
        has_compress = 1 if self.cfg.HAS_compressstiffness else 0
        use_jacobi_int = 1 if self.use_jacobi else 0
        wp.launch(solve_constraints_kernel, dim=self.n_cons,
                  inputs=[
                      self.cons, self.pos, self.pprev,
                      self.dP, self.dPw, self.mass, self.stopped,
                      self.rest_matrix, self.v_fiber_dir,
                      self.activation, self.tendonmask,
                      self.reaction_accum,
                      self.dt, use_jacobi_int,
                      self.contraction_ratio, self.fiber_stiffness_scale,
                      has_compress, self.n_cons,
                  ])

    def step(self):
        self.update_attach_targets()
        for _ in range(self.cfg.num_substeps):
            self.integrate()
            self.clear()
            self.solve_constraints()
            if self.use_jacobi:
                self.apply_dP()
            self.update_velocities()

    def render(self):
        """Render current state via WarpRenderer (if available)."""
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

    def get_fps(self):
        if not hasattr(self, 'step_start_time') or not hasattr(self, 'step_end_time'):
            return 0.0
        dur = self.step_end_time - self.step_start_time
        if dur == 0:
            return 0.0
        else:
            return 1.0 / dur

    def run(self):
        self.step_cnt = 1

        # OpenGL: register R key for reset
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
            # Check if renderer window closed
            if self.renderer is not None and not self.renderer.is_running():
                break
            if self.cfg.reset:
                self.reset()
                self.step_cnt = 1
                self.cfg.reset = False
            if not self.cfg.pause:
                self.step_start_time = time.perf_counter()
                self.step()
                self.step_cnt += 1
                self.step_end_time = time.perf_counter()
            self.render()

        # Save USD file if applicable
        if self.renderer is not None:
            self.renderer.save()


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def generate_muscle_id_colors(muscle_ids):
    """Generate unique colors for each muscle_id."""
    import colorsys
    colors = {}
    n = len(muscle_ids)
    for i, mid in enumerate(muscle_ids):
        hue = i / max(n, 1)
        saturation = 0.7 + (i % 3) * 0.1
        value = 0.8 + (i % 2) * 0.15
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors[mid] = np.array(rgb, dtype=np.float32)
    return colors


def get_config_path():
    parser = argparse.ArgumentParser(description="Muscle simulation.")
    parser.add_argument("--config", type=Path, default=Path("data/muscle/config/bicep.json"), help="Path to JSON config.")
    return parser.parse_args().config


def main():
    config_path = get_config_path()
    print("Using config:", config_path)
    cfg = load_config(config_path)
    cfg.gui = False  # headless mode for warp version
    sim = MuscleSim(cfg)
    print("Running for", cfg.nsteps, "steps.")
    sim.run()


if __name__ == "__main__":
    main()
