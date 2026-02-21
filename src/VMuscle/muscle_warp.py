# Warp version refactor of MuscleSim. Only rewrite the taichi kernel with warp.

import argparse
import json
import time
import zipfile
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path

import numpy as np
import warp as wp

wp.init()

@dataclass
class SimConfig:
    name: str = "MuscleSim"
    geo_path: Path = Path("data/muscle/model/bicep.geo")
    bone_geo_path: Path = Path("data/muscle/model/bicep_bone.geo")
    ground_mesh_path: Path = Path("data/muscle/model/ground.obj")
    coord_mesh_path: Path = Path("data/muscle/model/coord.obj")
    dt: float = 1e-3
    nsteps: int = 400
    num_substeps: int = 10  # 每个主时间步的子时间步数，用于提高稳定性
    gravity: float = -9.8
    density: float = 1000.0
    veldamping: float = 0.02
    activation: float = 0.3
    constraints: list = None
    arch: str = "cuda"
    gui: bool = True
    render_mode: str = "human"  # "human" or "rgb_array" or None, if None, no rendering
    save_image: bool = False
    show_auxiliary_meshes: bool = True
    pause: bool = False
    show_wireframe: bool = False
    render_fps: int = 24
    color_bones: bool = False  # 是否按 muscle_id 给骨骼着色
    color_muscles: str = "tendonmask"  # 肌肉着色模式: None, "muscle_id", "tendonmask"
    HAS_compressstiffness = False  # 如需关闭压缩带，设为False

# enum of constraint types, referecne from pbd_types.h
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
    kwargs = {}
    for fld in fields(SimConfig):
        name = fld.name
        if name in data:
            if name == "geo_path" or name == "bone_geo_path":
                kwargs[name] = Path(data[name])
            else:
                kwargs[name] = data[name]

    return SimConfig(**kwargs)

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


# ============================================================================
# Warp struct for constraints
# ============================================================================

@wp.struct
class Constraint:
    type: int
    cidx: int
    pts: wp.vec4i
    stiffness: float
    dampingratio: float
    tetid: int
    L: wp.vec3
    restlength: float
    restvector: wp.vec4
    restdir: wp.vec3
    compressionstiffness: float


# ============================================================================
# Warp @wp.func helper functions (module-level)
# ============================================================================

@wp.func
def flatten_mat33(mat: wp.mat33) -> wp.vec3:
    # Note: returning a vec3 of first row only for a simplified version.
    # The original returned a 9-element vector. Warp doesn't have vec9,
    # but this function is not used in the simulation, so we provide a stub.
    return wp.vec3(mat[0, 0], mat[0, 1], mat[0, 2])


@wp.func
def update_dP(dP: wp.array(dtype=wp.vec3), dPw: wp.array(dtype=float),
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
def outer_product_fn(v: wp.vec3) -> wp.mat33:
    return wp.outer(v, v)


@wp.func
def ssvd_fn(F: wp.mat33):
    U, sigma, V = wp.svd3(F)
    # sigma is wp.vec3 (diagonal values)
    # Reconstruct as mat33 for determinant check
    det_U = wp.determinant(U)
    if det_U < 0.0:
        U = wp.mat33(
            U[0, 0], U[0, 1], -U[0, 2],
            U[1, 0], U[1, 1], -U[1, 2],
            U[2, 0], U[2, 1], -U[2, 2],
        )
        sigma = wp.vec3(sigma[0], sigma[1], -sigma[2])
    det_V = wp.determinant(V)
    if det_V < 0.0:
        V = wp.mat33(
            V[0, 0], V[0, 1], -V[0, 2],
            V[1, 0], V[1, 1], -V[1, 2],
            V[2, 0], V[2, 1], -V[2, 2],
        )
        sigma = wp.vec3(sigma[0], sigma[1], -sigma[2])
    return U, sigma, V


@wp.func
def polar_decomposition_fn(F: wp.mat33):
    U, sigma, V = ssvd_fn(F)
    R = U * wp.transpose(V)
    sig_mat = wp.diag(sigma)
    S = V * sig_mat * wp.transpose(V)
    return S, R


@wp.func
def invariant4_fn(F: wp.mat33, fiber: wp.vec3) -> float:
    I4 = 1.0
    if wp.determinant(F) < 0.0:
        S = wp.mat33()
        R = wp.mat33()
        S, R = polar_decomposition_fn(F)
        Sw = S * fiber
        I4 = wp.dot(fiber, Sw)
    return I4


@wp.func
def invariant5_fn(F: wp.mat33, fiber: wp.vec3) -> float:
    Ff = F * fiber
    return wp.dot(Ff, Ff)


@wp.func
def squared_norm2_fn(a: wp.mat22) -> float:
    # Frobenius norm squared of a 2x2 matrix
    return a[0, 0] * a[0, 0] + a[0, 1] * a[0, 1] + a[1, 0] * a[1, 0] + a[1, 1] * a[1, 1]


@wp.func
def squared_norm3_fn(a: wp.mat33) -> float:
    # Frobenius norm squared of a 3x3 matrix: trace(A^T A) = sum of squares of all entries
    s = float(0.0)
    for i in range(3):
        for j in range(3):
            s = s + a[i, j] * a[i, j]
    return s


@wp.func
def mat3_to_quat_fn(m: wp.mat33) -> wp.vec4:
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
    return wp.vec4(qx, qy, qz, qw)


@wp.func
def triangle_xform_and_area_fn(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3):
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
        xform = wp.mat33(
            x[0], x[1], x[2],
            y[0], y[1], y[2],
            z[0], z[1], z[2],
        )
    return xform, area


@wp.func
def get_inv_mass_fn(idx: int, mass: wp.array(dtype=float), stopped: wp.array(dtype=int)) -> float:
    res = 0.0
    if stopped[idx] != 0:
        res = 0.0
    else:
        m = mass[idx]
        if m > 0.0:
            res = 1.0 / m
        else:
            res = 0.0
    return res


@wp.func
def transfer_tension_fn(muscletension: float, tendonmask: float,
                         minfiberscale: float, maxfiberscale: float) -> float:
    fiberscale = minfiberscale + (1.0 - tendonmask) * muscletension * (maxfiberscale - minfiberscale)
    return fiberscale


@wp.func
def inCompressBand_fn(curlen: float, restlen: float, has_compressstiffness: int) -> int:
    res = 0
    if has_compressstiffness != 0:
        if curlen < restlen:
            res = 1
        else:
            res = 0
    else:
        res = 0
    return res


@wp.func
def mat33_from_cols(c0: wp.vec3, c1: wp.vec3, c2: wp.vec3) -> wp.mat33:
    """Construct a mat33 from 3 column vectors (like ti.Matrix.cols)."""
    return wp.mat33(
        c0[0], c1[0], c2[0],
        c0[1], c1[1], c2[1],
        c0[2], c1[2], c2[2],
    )


# ============================================================================
# Constraint solver @wp.func (module-level)
# ============================================================================

@wp.func
def get_compression_stiffness_fn(cons: wp.array(dtype=Constraint), c: int) -> float:
    kstiff = cons[c].stiffness
    kstiffcompress = cons[c].compressionstiffness
    kstiffcompress = wp.where(kstiffcompress >= 0.0, kstiffcompress, kstiff)
    return kstiffcompress


@wp.func
def tet_volume_update_xpbd_fn(
    use_jacobi: int,
    cidx: int,
    cons: wp.array(dtype=Constraint),
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    restlength: float,
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=float),
    tetid: int,
    pts: wp.vec4i,
    dt: float,
    stiffness: float,
    kstiffcompress: float,
    kdampratio: float,
    mass: wp.array(dtype=float),
    stopped: wp.array(dtype=int),
    loff: int,
    has_compressstiffness: int,
):
    loff_local = 0
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

    w_sum = (inv_mass0 * wp.dot(grad0, grad0) +
             inv_mass1 * wp.dot(grad1, grad1) +
             inv_mass2 * wp.dot(grad2, grad2) +
             inv_mass3 * wp.dot(grad3, grad3))

    if w_sum > 1e-9:
        volume = wp.dot(wp.cross(d2, d1), d3) / 6.0

        comp = inCompressBand_fn(volume, restlength, has_compressstiffness)
        kstiff = wp.where(comp != 0, kstiffcompress, stiffness)
        loff_local = loff_local + comp
        loff_local = wp.min(loff_local, 2)

        if kstiff != 0.0:
            l = cons[cidx].L[loff_local]

            alpha = 1.0 / kstiff
            alpha = alpha / (dt * dt)

            C = volume - restlength

            dsum = 0.0
            gamma = 1.0
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
                wp.atomic_add(pos, pts[0], dlambda * inv_mass0 * grad0)
                wp.atomic_add(pos, pts[1], dlambda * inv_mass1 * grad1)
                wp.atomic_add(pos, pts[2], dlambda * inv_mass2 * grad2)
                wp.atomic_add(pos, pts[3], dlambda * inv_mass3 * grad3)
                new_L = cons[cidx].L
                new_val = new_L[loff_local] + dlambda
                if loff_local == 0:
                    new_L = wp.vec3(new_val, new_L[1], new_L[2])
                elif loff_local == 1:
                    new_L = wp.vec3(new_L[0], new_val, new_L[2])
                else:
                    new_L = wp.vec3(new_L[0], new_L[1], new_val)
                cons[cidx].L = new_L


@wp.func
def tet_fiber_update_xpbd_fn(
    use_jacobi: int,
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=float),
    cidx: int,
    cons: wp.array(dtype=Constraint),
    pts: wp.vec4i,
    dt: float,
    fiber: wp.vec3,
    stiffness: float,
    Dminv: wp.mat33,
    kdampratio: float,
    restlength: float,
    restvector: wp.vec4,
    acti: float,
    mass: wp.array(dtype=float),
    stopped: wp.array(dtype=int),
):
    isLINEARENERGY = True
    isNORMSTIFFNESS = True

    inv_mass0 = get_inv_mass_fn(pts[0], mass, stopped)
    inv_mass1 = get_inv_mass_fn(pts[1], mass, stopped)
    inv_mass2 = get_inv_mass_fn(pts[2], mass, stopped)
    inv_mass3 = get_inv_mass_fn(pts[3], mass, stopped)

    l = cons[cidx].L[0]
    alpha = 1.0 / stiffness
    if isNORMSTIFFNESS:
        alpha = alpha / restlength
    alpha = alpha / (dt * dt)
    grad_scale = 1.0
    psi = 0.0

    c0 = pos[pts[0]] - pos[pts[3]]
    c1 = pos[pts[1]] - pos[pts[3]]
    c2 = pos[pts[2]] - pos[pts[3]]
    _Ds = mat33_from_cols(c0, c1, c2)
    F = _Ds * Dminv

    # Non-anisotropic path (the default)
    wTDminvT = wp.vec3(restvector[0], restvector[1], restvector[2])
    # FwT = wTDminvT @ _Ds^T  (row-vec * mat = Ds * w as column-vec)
    # In Taichi: FwT = wTDminvT @ _Ds.transpose()
    #   result[i] = sum_j wTDminvT[j] * _Ds[i,j] = (Ds * w)[i]
    # In Warp: M * v computes result[i] = sum_j M[i,j] * v[j], so _Ds * w gives the same.
    FwT = _Ds * wTDminvT
    psi = 0.5 * wp.dot(FwT, FwT)

    if (isLINEARENERGY or isNORMSTIFFNESS):
        if psi > 1e-9:
            psi_sqrt = wp.sqrt(2.0 * psi)
            grad_scale = 1.0 / psi_sqrt
            psi = psi_sqrt

    # Ht = outer(wTDminvT, FwT) -> shape (3,3)
    Ht = wp.outer(wTDminvT, FwT)

    grad0 = grad_scale * wp.vec3(Ht[0, 0], Ht[0, 1], Ht[0, 2])
    grad1 = grad_scale * wp.vec3(Ht[1, 0], Ht[1, 1], Ht[1, 2])
    grad2 = grad_scale * wp.vec3(Ht[2, 0], Ht[2, 1], Ht[2, 2])
    grad3 = -grad0 - grad1 - grad2

    w_sum = (inv_mass0 * wp.dot(grad0, grad0) +
             inv_mass1 * wp.dot(grad1, grad1) +
             inv_mass2 * wp.dot(grad2, grad2) +
             inv_mass3 * wp.dot(grad3, grad3))

    if w_sum > 1e-9:
        dsum = 0.0
        gamma = 1.0
        if kdampratio > 0.0:
            beta = stiffness * kdampratio * dt * dt
            if isNORMSTIFFNESS:
                beta = beta * restlength
            gamma = alpha * beta / dt

            dsum = (wp.dot(grad0, pos[pts[0]] - pprev[pts[0]]) +
                    wp.dot(grad1, pos[pts[1]] - pprev[pts[1]]) +
                    wp.dot(grad2, pos[pts[2]] - pprev[pts[2]]) +
                    wp.dot(grad3, pos[pts[3]] - pprev[pts[3]]))
            dsum = dsum * gamma
            gamma = gamma + 1.0

        C = psi
        dL = (-C - alpha * l - dsum) / (gamma * w_sum + alpha)
        if use_jacobi != 0:
            update_dP(dP, dPw, dL * inv_mass0 * grad0, pts[0])
            update_dP(dP, dPw, dL * inv_mass1 * grad1, pts[1])
            update_dP(dP, dPw, dL * inv_mass2 * grad2, pts[2])
            update_dP(dP, dPw, dL * inv_mass3 * grad3, pts[3])
        else:
            wp.atomic_add(pos, pts[0], dL * inv_mass0 * grad0)
            wp.atomic_add(pos, pts[1], dL * inv_mass1 * grad1)
            wp.atomic_add(pos, pts[2], dL * inv_mass2 * grad2)
            wp.atomic_add(pos, pts[3], dL * inv_mass3 * grad3)
            new_L = cons[cidx].L
            new_L = wp.vec3(new_L[0] + dL, new_L[1], new_L[2])
            cons[cidx].L = new_L


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
    dPw: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    stopped: wp.array(dtype=int),
    restlength: float,
    kstiff: float,
    kdampratio: float,
    kstiffcompress: float,
    dt: float,
    has_compressstiffness: int,
):
    invmass0 = 0.0
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
            loff = inCompressBand_fn(d, restlength, has_compressstiffness)
            kstiff_val = wp.where(loff != 0, kstiffcompress, kstiff)
            if kstiff_val != 0.0:
                l = cons[cidx].L[loff]

                alpha = 1.0 / kstiff_val
                alpha = alpha / (dt * dt)

                C = d - restlength
                n = n / d
                gradC = n

                dsum = 0.0
                gamma = 1.0
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

                dL_val = (-C - alpha * l - dsum) / (gamma * wsum + alpha)
                dp = n * (-dL_val)

                if use_jacobi != 0:
                    if pt0 >= 0:
                        update_dP(dP, dPw, invmass0 * dp, pt0)
                    update_dP(dP, dPw, -invmass1 * dp, pt1)
                else:
                    if pt0 >= 0:
                        wp.atomic_add(pos, pt0, invmass0 * dp)
                    wp.atomic_add(pos, pt1, -invmass1 * dp)
                    new_L = cons[cidx].L
                    new_val = new_L[loff] + dL_val
                    if loff == 0:
                        new_L = wp.vec3(new_val, new_L[1], new_L[2])
                    elif loff == 1:
                        new_L = wp.vec3(new_L[0], new_val, new_L[2])
                    else:
                        new_L = wp.vec3(new_L[0], new_L[1], new_val)
                    cons[cidx].L = new_L


@wp.func
def distance_update_xpbd_fn(
    use_jacobi: int,
    cidx: int,
    cons: wp.array(dtype=Constraint),
    pts: wp.vec4i,
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    stopped: wp.array(dtype=int),
    restlength: float,
    kstiff: float,
    kdampratio: float,
    kstiffcompress: float,
    dt: float,
    has_compressstiffness: int,
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
            loff = inCompressBand_fn(d, restlength, has_compressstiffness)
            kstiff_val = wp.where(loff != 0, kstiffcompress, kstiff)
            if kstiff_val != 0.0:
                l = cons[cidx].L[loff]

                alpha = 1.0 / kstiff_val
                alpha = alpha / (dt * dt)

                C = d - restlength
                n = n / d
                gradC = n

                dsum = 0.0
                gamma = 1.0
                if kdampratio > 0.0:
                    prev0 = pprev[pt0]
                    prev1 = pprev[pt1]
                    beta = kstiff_val * kdampratio * dt * dt
                    gamma = alpha * beta / dt
                    dsum = gamma * (-wp.dot(gradC, p0 - prev0) + wp.dot(gradC, p1 - prev1))
                    gamma = gamma + 1.0

                dL_val = (-C - alpha * l - dsum) / (gamma * wsum + alpha)
                dp = n * (-dL_val)

                if use_jacobi != 0:
                    update_dP(dP, dPw, invmass0 * dp, pt0)
                    update_dP(dP, dPw, -invmass1 * dp, pt1)
                else:
                    wp.atomic_add(pos, pt0, invmass0 * dp)
                    wp.atomic_add(pos, pt1, -invmass1 * dp)
                    new_L = cons[cidx].L
                    new_val = new_L[loff] + dL_val
                    if loff == 0:
                        new_L = wp.vec3(new_val, new_L[1], new_L[2])
                    elif loff == 1:
                        new_L = wp.vec3(new_L[0], new_val, new_L[2])
                    else:
                        new_L = wp.vec3(new_L[0], new_L[1], new_val)
                    cons[cidx].L = new_L


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
    dPw: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    stopped: wp.array(dtype=int),
    restlength: float,
    restvector: wp.vec4,
    kstiff: float,
    kdampratio: float,
    kstiffcompress: float,
    flags: int,
    has_compressstiffness: int,
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

    Dminv = wp.mat22(restvector[0], restvector[1],
                     restvector[2], restvector[3])

    xform = wp.mat33()
    area = float(0.0)
    xform, area = triangle_xform_and_area_fn(p0, p1, p2)
    if area != 0.0:
        loff = inCompressBand_fn(area, restlength, has_compressstiffness)
        kstiff_val = wp.where(loff != 0, kstiffcompress, kstiff)
        if kstiff_val != 0.0:
            l = cons[cidx].L[loff]

            row0 = wp.vec3(xform[0, 0], xform[0, 1], xform[0, 2])
            row1 = wp.vec3(xform[1, 0], xform[1, 1], xform[1, 2])
            P0 = wp.vec2(wp.dot(row0, p0), wp.dot(row1, p0))
            P1 = wp.vec2(wp.dot(row0, p1), wp.dot(row1, p1))
            P2 = wp.vec2(wp.dot(row0, p2), wp.dot(row1, p2))

            # Ds = mat22 from cols [P0-P2, P1-P2]
            col0 = P0 - P2
            col1 = P1 - P2
            Ds_2 = wp.mat22(col0[0], col1[0],
                            col0[1], col1[1])
            F2 = Ds_2 * Dminv

            m = wp.vec2(F2[0, 0] + F2[1, 1], F2[1, 0] - F2[0, 1])
            mlen = wp.length(m)
            if mlen >= 1e-9:
                m = m / mlen
                R = wp.mat22(m[0], -m[1],
                             m[1],  m[0])

                d2 = F2 - R
                psi = squared_norm2_fn(d2)
                gradscale = 2.0
                if (flags & (LINEARENERGY | NORMSTIFFNESS)) != 0:
                    psi = wp.sqrt(psi)
                    gradscale = 1.0 / psi

                if psi >= 1e-6:
                    d2T = wp.transpose(d2)
                    Ht2 = Dminv * d2T
                    grad0_2d = wp.vec3(Ht2[0, 0], Ht2[0, 1], 0.0) * gradscale
                    grad1_2d = wp.vec3(Ht2[1, 0], Ht2[1, 1], 0.0) * gradscale
                    grad2_2d = -grad0_2d - grad1_2d

                    xformT = wp.transpose(xform)
                    grad0_3d = xformT * grad0_2d
                    grad1_3d = xformT * grad1_2d
                    grad2_3d = xformT * grad2_2d

                    wsum = (invmass0 * wp.dot(grad0_3d, grad0_3d) +
                            invmass1 * wp.dot(grad1_3d, grad1_3d) +
                            invmass2 * wp.dot(grad2_3d, grad2_3d))

                    if wsum != 0.0:
                        alpha = 1.0 / kstiff_val
                        if (flags & NORMSTIFFNESS) != 0:
                            alpha = alpha / restlength
                        alpha = alpha / (dt * dt)

                        dsum = 0.0
                        gamma = 1.0
                        if kdampratio > 0.0:
                            prev0 = pprev[pt0]
                            prev1 = pprev[pt1]
                            prev2 = pprev[pt2]
                            beta = kstiff_val * kdampratio * dt * dt
                            if (flags & NORMSTIFFNESS) != 0:
                                beta = beta * restlength
                            gamma = alpha * beta / dt
                            dsum = (wp.dot(grad0_3d, p0 - prev0) +
                                    wp.dot(grad1_3d, p1 - prev1) +
                                    wp.dot(grad2_3d, p2 - prev2))
                            dsum = dsum * gamma
                            gamma = gamma + 1.0

                        C = psi
                        dL_val = (-C - alpha * l - dsum) / (gamma * wsum + alpha)
                        if use_jacobi != 0:
                            update_dP(dP, dPw, dL_val * invmass0 * grad0_3d, pt0)
                            update_dP(dP, dPw, dL_val * invmass1 * grad1_3d, pt1)
                            update_dP(dP, dPw, dL_val * invmass2 * grad2_3d, pt2)
                        else:
                            wp.atomic_add(pos, pt0, dL_val * invmass0 * grad0_3d)
                            wp.atomic_add(pos, pt1, dL_val * invmass1 * grad1_3d)
                            wp.atomic_add(pos, pt2, dL_val * invmass2 * grad2_3d)
                            new_L = cons[cidx].L
                            new_val = new_L[loff] + dL_val
                            if loff == 0:
                                new_L = wp.vec3(new_val, new_L[1], new_L[2])
                            elif loff == 1:
                                new_L = wp.vec3(new_L[0], new_val, new_L[2])
                            else:
                                new_L = wp.vec3(new_L[0], new_L[1], new_val)
                            cons[cidx].L = new_L


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
    dPw: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    stopped: wp.array(dtype=int),
    restlength: float,
    restvector: wp.vec4,
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

    Ds = mat33_from_cols(p0 - p3, p1 - p3, p2 - p3)
    F = Ds * restmatrix

    S = wp.mat33()
    R = wp.mat33()
    S, R = polar_decomposition_fn(F)
    d3 = F - R
    cons[cidx].restvector = mat3_to_quat_fn(R)

    psi = squared_norm3_fn(d3)
    gradscale = 2.0
    if (flags & (LINEARENERGY | NORMSTIFFNESS)) != 0:
        psi = wp.sqrt(psi)
        gradscale = 1.0 / psi

    if psi >= 1e-6:
        Ht = restmatrix * wp.transpose(d3)
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

            dsum = 0.0
            gamma = 1.0
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
            l0 = cons[cidx].L[0]
            dL_val = (-C - alpha * l0 - dsum) / (gamma * wsum + alpha)
            if use_jacobi != 0:
                update_dP(dP, dPw, dL_val * invmass0 * grad0, pt0)
                update_dP(dP, dPw, dL_val * invmass1 * grad1, pt1)
                update_dP(dP, dPw, dL_val * invmass2 * grad2, pt2)
                update_dP(dP, dPw, dL_val * invmass3 * grad3, pt3)
            else:
                wp.atomic_add(pos, pt0, dL_val * invmass0 * grad0)
                wp.atomic_add(pos, pt1, dL_val * invmass1 * grad1)
                wp.atomic_add(pos, pt2, dL_val * invmass2 * grad2)
                wp.atomic_add(pos, pt3, dL_val * invmass3 * grad3)
                new_L = cons[cidx].L
                new_L = wp.vec3(new_L[0] + dL_val, new_L[1], new_L[2])
                cons[cidx].L = new_L


# ============================================================================
# Warp kernels (module-level)
# ============================================================================

@wp.kernel
def fill_float_kernel(arr: wp.array(dtype=float), val: float):
    i = wp.tid()
    arr[i] = val


@wp.kernel
def precompute_rest_kernel(
    tet_indices: wp.array(dtype=wp.vec4i),
    pos0: wp.array(dtype=wp.vec3),
    rest_volume: wp.array(dtype=float),
    rest_matrix: wp.array(dtype=wp.mat33),
    mass: wp.array(dtype=float),
    total_rest_volume: wp.array(dtype=float),
    density: float,
):
    c = wp.tid()
    pts = tet_indices[c]
    c0 = pos0[pts[0]] - pos0[pts[3]]
    c1 = pos0[pts[1]] - pos0[pts[3]]
    c2 = pos0[pts[2]] - pos0[pts[3]]
    Dm = mat33_from_cols(c0, c1, c2)
    vol = wp.abs(wp.determinant(Dm)) / 6.0
    rest_volume[c] = vol
    wp.atomic_add(total_rest_volume, 0, vol)
    mass_contrib = vol * density / 4.0
    wp.atomic_add(mass, pts[0], mass_contrib)
    wp.atomic_add(mass, pts[1], mass_contrib)
    wp.atomic_add(mass, pts[2], mass_contrib)
    wp.atomic_add(mass, pts[3], mass_contrib)
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
    vel[i] = (1.0 - veldamping) * vel[i]
    vel[i] = vel[i] + dt * extacc
    pos[i] = pos[i] + dt * vel[i]


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
    dPw: wp.array(dtype=float),
):
    i = wp.tid()
    dP[i] = wp.vec3(0.0, 0.0, 0.0)
    dPw[i] = 0.0


@wp.kernel
def clear_cons_L_kernel(cons: wp.array(dtype=Constraint)):
    i = wp.tid()
    cons[i].L = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def apply_dP_kernel(
    pos: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=float),
):
    idx = wp.tid()
    w = dPw[idx]
    if w > 1e-9:
        pos[idx] = pos[idx] + dP[idx] / w


@wp.kernel
def compute_cell_tendon_mask_kernel(
    tet_indices: wp.array(dtype=wp.vec4i),
    v_tendonmask: wp.array(dtype=float),
    tendon_mask: wp.array(dtype=float),
):
    c = wp.tid()
    pts = tet_indices[c]
    sum_mask = float(0.0)
    sum_mask = sum_mask + v_tendonmask[pts[0]]
    sum_mask = sum_mask + v_tendonmask[pts[1]]
    sum_mask = sum_mask + v_tendonmask[pts[2]]
    sum_mask = sum_mask + v_tendonmask[pts[3]]
    tendon_mask[c] = sum_mask / 4.0


@wp.kernel
def update_attach_targets_kernel(
    cons: wp.array(dtype=Constraint),
    bone_pos: wp.array(dtype=wp.vec3),
):
    c = wp.tid()
    ctype = cons[c].type
    if ctype == ATTACH:
        tgt_idx = cons[c].pts[2]
        if tgt_idx >= 0:
            target_pos = bone_pos[tgt_idx]
            rv = cons[c].restvector
            cons[c].restvector = wp.vec4(target_pos[0], target_pos[1], target_pos[2], rv[3])
    elif ctype == DISTANCELINE:
        tgt_idx = cons[c].pts[1]
        if tgt_idx >= 0:
            target_pos = bone_pos[tgt_idx]
            rv = cons[c].restvector
            cons[c].restvector = wp.vec4(target_pos[0], target_pos[1], target_pos[2], rv[3])


@wp.kernel
def solve_constraints_kernel(
    cons: wp.array(dtype=Constraint),
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    stopped: wp.array(dtype=int),
    v_fiber_dir: wp.array(dtype=wp.vec3),
    rest_matrix: wp.array(dtype=wp.mat33),
    activation: wp.array(dtype=float),
    tendonmask: wp.array(dtype=float),
    dt: float,
    use_jacobi: int,
    has_compressstiffness: int,
):
    c = wp.tid()
    kstiffcompress = get_compression_stiffness_fn(cons, c)

    if cons[c].stiffness <= 0.0:
        return

    ctype = cons[c].type

    if ctype == TETVOLUME:
        pts = cons[c].pts
        tetid = cons[c].tetid
        tet_volume_update_xpbd_fn(
            use_jacobi,
            c,
            cons,
            pos,
            pprev,
            cons[c].restlength,
            dP, dPw,
            tetid,
            pts,
            dt,
            cons[c].stiffness,
            kstiffcompress,
            cons[c].dampingratio,
            mass, stopped,
            0,
            has_compressstiffness,
        )
    elif ctype == TETFIBERNORM:
        pts = cons[c].pts
        fiber_dir = (v_fiber_dir[pts[0]] + v_fiber_dir[pts[1]] + v_fiber_dir[pts[2]] + v_fiber_dir[pts[3]]) / 4.0
        tetid = cons[c].tetid
        Dminv = rest_matrix[tetid]
        acti = activation[tetid]

        stiffness = cons[c].stiffness
        _tendonmask = tendonmask[tetid]
        fiberscale = transfer_tension_fn(acti, _tendonmask, 0.0, 10.0)
        stiffness = stiffness * fiberscale * 10000.0
        if stiffness <= 0.0:
            return
        tet_fiber_update_xpbd_fn(
            use_jacobi,
            pos, pprev,
            dP, dPw,
            c,
            cons,
            pts,
            dt,
            fiber_dir,
            stiffness,
            Dminv,
            cons[c].dampingratio,
            cons[c].restlength,
            cons[c].restvector,
            acti,
            mass, stopped,
        )
    elif ctype == DISTANCE:
        pts = cons[c].pts
        distance_update_xpbd_fn(
            use_jacobi,
            c,
            cons,
            pts,
            pos, pprev,
            dP, dPw,
            mass, stopped,
            cons[c].restlength,
            cons[c].stiffness,
            cons[c].dampingratio,
            kstiffcompress,
            dt,
            has_compressstiffness,
        )
    elif ctype == ATTACH:
        pts = cons[c].pts
        pt_src = pts[0]
        rv = cons[c].restvector
        target_pos = wp.vec3(rv[0], rv[1], rv[2])
        distance_pos_update_xpbd_fn(
            use_jacobi,
            c,
            cons,
            -1,
            pt_src,
            target_pos,
            pos[pt_src],
            pos, pprev,
            dP, dPw,
            mass, stopped,
            cons[c].restlength,
            cons[c].stiffness,
            cons[c].dampingratio,
            kstiffcompress,
            dt,
            has_compressstiffness,
        )
    elif ctype == PIN:
        if cons[c].stiffness <= 0.0:
            return
        pts = cons[c].pts
        pt_src = pts[0]
        rv = cons[c].restvector
        target_pos = wp.vec3(rv[0], rv[1], rv[2])
        distance_pos_update_xpbd_fn(
            use_jacobi,
            c,
            cons,
            -1,
            pt_src,
            target_pos,
            pos[pt_src],
            pos, pprev,
            dP, dPw,
            mass, stopped,
            cons[c].restlength,
            cons[c].stiffness,
            cons[c].dampingratio,
            kstiffcompress,
            dt,
            has_compressstiffness,
        )
    elif ctype == DISTANCELINE:
        pts = cons[c].pts
        pt_src = pts[0]
        p_src = pos[pt_src]
        rv = cons[c].restvector
        line_origin = wp.vec3(rv[0], rv[1], rv[2])
        line_dir = cons[c].restdir

        p_projected = project_to_line_fn(p_src, line_origin, line_dir)

        distance_pos_update_xpbd_fn(
            use_jacobi,
            c,
            cons,
            -1,
            pt_src,
            p_projected,
            p_src,
            pos, pprev,
            dP, dPw,
            mass, stopped,
            cons[c].restlength,
            cons[c].stiffness,
            cons[c].dampingratio,
            kstiffcompress,
            dt,
            has_compressstiffness,
        )
    elif ctype == TETARAP:
        pts = cons[c].pts
        tetid = cons[c].tetid
        tet_arap_update_xpbd_fn(
            use_jacobi,
            c,
            cons,
            pts,
            dt,
            pos, pprev,
            dP, dPw,
            mass, stopped,
            cons[c].restlength,
            cons[c].restvector,
            rest_matrix[tetid],
            cons[c].stiffness,
            cons[c].dampingratio,
            fem_flags_fn(ctype),
        )
    elif ctype == TRIARAP:
        pts = cons[c].pts
        tri_arap_update_xpbd_fn(
            use_jacobi,
            c,
            cons,
            pts,
            dt,
            pos, pprev,
            dP, dPw,
            mass, stopped,
            cons[c].restlength,
            cons[c].restvector,
            cons[c].stiffness,
            cons[c].dampingratio,
            kstiffcompress,
            fem_flags_fn(ctype),
            has_compressstiffness,
        )


# ============================================================================
# MuscleSim class
# ============================================================================

class MuscleSim:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg

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

        self.use_jacobi = True
        self.dt = self.cfg.dt / self.cfg.num_substeps
        self.step_cnt = 0

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
        self.mass = wp.zeros(n_v, dtype=float)
        self.stopped = wp.zeros(n_v, dtype=int)
        self.v_fiber_dir = wp.zeros(n_v, dtype=wp.vec3)
        self.dP = wp.zeros(n_v, dtype=wp.vec3)
        self.dPw = wp.zeros(n_v, dtype=float)
        self.tet_indices = wp.zeros(n_tet, dtype=wp.vec4i)

        self.rest_volume = wp.zeros(n_tet, dtype=float)
        self.rest_matrix = wp.zeros(n_tet, dtype=wp.mat33)
        self.activation = wp.zeros(n_tet, dtype=float)


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


    def create_tet_fiber_constraint(self, params):
        constraints = []
        stiffness = params.get('stiffness', 1.0)
        dampingratio = params.get('dampingratio', 0.0)

        for i, tet in enumerate(self.tet_np):
            pt0, pt1, pt2, pt3 = tet
            volume, materialW = self.compute_tet_fiber_rest_length(pt0, pt1, pt2, pt3)
            restvec4 = materialW.astype(float).tolist()[:3] + [1.0]
            c = dict(
                type=TETFIBERNORM,
                pts=[int(pt0), int(pt1), int(pt2), int(pt3)],
                stiffness=stiffness,
                dampingratio=dampingratio,
                tetid=i,
                L=[0.0, 0.0, 0.0],
                restlength=float(volume),
                restvector=restvec4,
                restdir=[0.0, 0.0, 0.0],
                compressionstiffness=-1.0,
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
        if not hasattr(self, 'bone_pos_np') and Path(target_path).exists():
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
                    for v_idx, mid in enumerate(muscle_ids):
                        if mid not in self.bone_muscle_ids:
                            self.bone_muscle_ids[mid] = []
                        self.bone_muscle_ids[mid].append(v_idx)
                    for mid in self.bone_muscle_ids:
                        self.bone_muscle_ids[mid] = np.array(self.bone_muscle_ids[mid], dtype=np.int32)

                    print(f"Bone muscle_id groups: {list(self.bone_muscle_ids.keys())}")

                    if self.cfg.color_bones:
                        unique_ids = sorted(self.bone_muscle_ids.keys())
                        self.bone_id_colors = self._generate_muscle_id_colors(unique_ids)

                        self.bone_vertex_colors = np.zeros((len(muscle_ids), 3), dtype=np.float32)
                        for v_idx, mid in enumerate(muscle_ids):
                            self.bone_vertex_colors[v_idx] = self.bone_id_colors[mid]
                        print("Bone coloring by muscle_id enabled")

            if self.bone_pos.shape[0] > 0:
                self.bone_pos_np = self.bone_pos.copy()
                self.bone_pos_field = wp.array(self.bone_pos, dtype=wp.vec3)

        if hasattr(self, 'bone_pos'):
            return self.bone_geo, self.bone_pos
        else:
            return None, np.zeros((0,3), dtype=np.float32)


    def _generate_muscle_id_colors(self, unique_ids):
        import colorsys
        colors = {}
        n = len(unique_ids)
        for i, mid in enumerate(unique_ids):
            hue = i / max(n, 1)
            saturation = 0.7 + (i % 3) * 0.1
            value = 0.8 + (i % 2) * 0.15
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors[mid] = np.array(rgb, dtype=np.float32)
        return colors


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

        for src_idx in valid_src_indices:
            src_pos = self.pos0_np[src_idx]

            diff = self.bone_pos - src_pos
            dists_sq = np.einsum('ij,ij->i', diff, diff)
            tgt_idx = np.argmin(dists_sq)

            target_pos = self.bone_pos[tgt_idx]

            if not hasattr(self, 'pt2tet'):
                self.pt2tet = self.map_pts2tets(self.tet_np)
            tetid = self.pt2tet.get(src_idx, [-1])[0]

            restlength = np.linalg.norm(target_pos - src_pos)

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

        for src_idx in valid_src_indices:
            src_pos = self.pos0_np[src_idx]

            diff = self.bone_pos - src_pos
            dists_sq = np.einsum('ij,ij->i', diff, diff)
            tgt_idx = np.argmin(dists_sq)

            target_pos = self.bone_pos[tgt_idx]

            if not hasattr(self, 'pt2tet'):
                self.pt2tet = self.map_pts2tets(self.tet_np)
            tetid = self.pt2tet.get(src_idx, [-1])[0]

            direction = target_pos - src_pos
            norm = np.linalg.norm(direction)
            if norm > 1e-9:
                direction = direction / norm
            else:
                raise ValueError(f"source point and target point are too close: src_idx={src_idx}, tgt_idx={tgt_idx}")

            if not hasattr(self, 'pt2tet'):
                self.pt2tet = self.map_pts2tets(self.tet_np)
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
        constraints = []
        stiffness = params.get('stiffness', 1e10)
        dampingratio = params.get('dampingratio', 0.0)

        for i, tet in enumerate(self.tet_np):
            pt0, pt1, pt2, pt3 = tet
            restm, volume = self.compute_tet_rest_matrix(pt0, pt1, pt2, pt3, scale=1.0)
            if restm is None:
                continue
            c = dict(
                type=TETARAP,
                pts=[int(pt0), int(pt1), int(pt2), int(pt3)],
                stiffness=stiffness,
                dampingratio=dampingratio,
                tetid=i,
                L=[0.0, 0.0, 0.0],
                restlength=float(volume),
                restvector=[0.0, 0.0, 0.0, 1.0],
                restdir=[0.0, 0.0, 0.0],
                compressionstiffness=-1.0,
            )
            constraints.append(c)
        return constraints

    def create_tri_arap_constraints(self, params):
        constraints = []
        stiffness = params.get('stiffness', 1e10)
        dampingratio = params.get('dampingratio', 0.0)

        if not hasattr(self, 'surface_tris'):
            self.surface_tris = build_surface_tris(self.tet_np)
        tris = self.surface_tris
        for i, tri in enumerate(tris):
            pt0, pt1, pt2 = tri
            restm, area = self.compute_tri_rest_matrix(pt0, pt1, pt2, scale=1.0)
            if restm is None:
                continue
            restvec4 = [restm[0, 0], restm[0, 1], restm[1, 0], restm[1, 1]]
            c = dict(
                type=TRIARAP,
                pts=[int(pt0), int(pt1), int(pt2), 0],
                stiffness=stiffness,
                dampingratio=dampingratio,
                tetid=i,
                L=[0.0, 0.0, 0.0],
                restlength=float(area),
                restvector=restvec4,
                restdir=[0.0, 0.0, 0.0],
                compressionstiffness=-1.0,
            )
            constraints.append(c)
        return constraints

    def create_tet_volume_constraint(self, params):
        constraints = []
        stiffness = params.get('stiffness', 1e10)
        dampingratio = params.get('dampingratio', 0.0)

        for i, tet in enumerate(self.tet_np):
            pt0, pt1, pt2, pt3 = tet
            p = self.pos0_np
            restm, volume = self.compute_tet_rest_matrix(pt0, pt1, pt2, pt3, scale=1.0)
            if restm is None:
                continue
            v = volume

            c = dict(
                type=TETVOLUME,
                pts=[int(pt0), int(pt1), int(pt2), int(pt3)],
                stiffness=stiffness,
                dampingratio=dampingratio,
                compressionstiffness=-1.0,
                tetid=i,
                L=[0.0, 0.0, 0.0],
                restlength=float(v),
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
        if n_cons > 0:
            # Build numpy structured array for constraints and convert to warp
            cons_np = np.zeros(n_cons, dtype=np.dtype([
                ('type', np.int32),
                ('cidx', np.int32),
                ('pts', np.int32, (4,)),
                ('stiffness', np.float32),
                ('dampingratio', np.float32),
                ('tetid', np.int32),
                ('L', np.float32, (3,)),
                ('restlength', np.float32),
                ('restvector', np.float32, (4,)),
                ('restdir', np.float32, (3,)),
                ('compressionstiffness', np.float32),
            ]))
            for i, c in enumerate(all_constraints):
                c['cidx'] = i
                cons_np[i]['type'] = c['type']
                cons_np[i]['cidx'] = i
                cons_np[i]['pts'] = c['pts']
                cons_np[i]['stiffness'] = c['stiffness']
                cons_np[i]['dampingratio'] = c['dampingratio']
                cons_np[i]['tetid'] = c['tetid']
                cons_np[i]['L'] = c['L']
                cons_np[i]['restlength'] = c['restlength']
                cons_np[i]['restvector'] = c['restvector']
                cons_np[i]['restdir'] = c['restdir']
                cons_np[i]['compressionstiffness'] = c['compressionstiffness']
            self.cons = wp.array(cons_np, dtype=Constraint)
        else:
            self.cons = wp.zeros(0, dtype=Constraint)
            self.raw_constraints = []

        print(f"Built {self.cons.shape[0]} constraints total.")


    def _init_fields(self):
        self.pos0 = wp.array(self.pos0_np.astype(np.float32), dtype=wp.vec3)
        self.pos = wp.array(self.pos0_np.astype(np.float32), dtype=wp.vec3)
        self.vel.zero_()
        self.force.zero_()
        self.mass.zero_()
        self.stopped.zero_()
        self.tet_indices = wp.array(self.tet_np.astype(np.int32), dtype=wp.vec4i)

        if self.v_fiber_np is None:
            self.v_fiber_np = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (self.n_verts, 1))
        self.v_fiber_dir = wp.array(self.v_fiber_np.astype(np.float32), dtype=wp.vec3)

        # Compute cell tendon mask
        if self.v_tendonmask_np is not None:
            self.v_tendonmask = wp.array(self.v_tendonmask_np.astype(np.float32), dtype=float)
        else:
            self.v_tendonmask = wp.zeros(self.n_verts, dtype=float)
        self.tendonmask = wp.zeros(self.tet_np.shape[0], dtype=float)
        wp.launch(compute_cell_tendon_mask_kernel, dim=self.tet_np.shape[0],
                  inputs=[self.tet_indices, self.v_tendonmask, self.tendonmask])

        self.activation.zero_()
        self.total_rest_volume = wp.zeros(1, dtype=float)

        print("Initialized fields done.")


    def _precompute_rest(self):
        n_tet = self.tet_np.shape[0]
        self.total_rest_volume.zero_()
        wp.launch(precompute_rest_kernel, dim=n_tet,
                  inputs=[self.tet_indices, self.pos0, self.rest_volume, self.rest_matrix,
                          self.mass, self.total_rest_volume, self.cfg.density])


    def reset(self):
        self.pos = wp.array(self.pos0_np.astype(np.float32), dtype=wp.vec3)
        self.pprev = wp.array(self.pos0_np.astype(np.float32), dtype=wp.vec3)
        self.vel.zero_()
        self.force.zero_()
        self.activation.zero_()
        self.clear()
        self.step_cnt = 1


    def update_attach_targets(self):
        if (
            hasattr(self, "bone_pos_field")
            and self.bone_pos_field.shape[0] > 0
            and hasattr(self, "attach_constraints")
            and len(self.attach_constraints) > 0
        ):
            wp.launch(update_attach_targets_kernel, dim=self.cons.shape[0],
                      inputs=[self.cons, self.bone_pos_field])


    def clear(self):
        n_v = self.pos.shape[0]
        wp.launch(clear_dP_kernel, dim=n_v, inputs=[self.dP, self.dPw])
        if self.cons.shape[0] > 0:
            wp.launch(clear_cons_L_kernel, dim=self.cons.shape[0], inputs=[self.cons])


    def step(self):
        self.update_attach_targets()
        for _ in range(self.cfg.num_substeps):
            wp.launch(integrate_kernel, dim=self.n_verts,
                      inputs=[self.pos, self.pprev, self.vel, self.cfg.gravity, self.cfg.veldamping, self.dt])
            self.clear()
            if self.cons.shape[0] > 0:
                has_compress = 1 if self.cfg.HAS_compressstiffness else 0
                use_jacobi_int = 1 if self.use_jacobi else 0
                wp.launch(solve_constraints_kernel, dim=self.cons.shape[0],
                          inputs=[self.cons, self.pos, self.pprev, self.dP, self.dPw,
                                  self.mass, self.stopped, self.v_fiber_dir, self.rest_matrix,
                                  self.activation, self.tendonmask, self.dt, use_jacobi_int, has_compress])
            if self.use_jacobi:
                wp.launch(apply_dP_kernel, dim=self.n_verts,
                          inputs=[self.pos, self.dP, self.dPw])
            wp.launch(update_velocities_kernel, dim=self.n_verts,
                      inputs=[self.pos, self.pprev, self.vel, self.dt])


    def calc_vol_error(self):
        """Calculate volume error as a fraction of rest volume. Returns a float on the host."""
        pos_np = self.pos.numpy()
        tet_np = self.tet_indices.numpy()
        rest_vol_np = self.total_rest_volume.numpy()
        total_rest = rest_vol_np[0]

        total_vol = 0.0
        for i in range(tet_np.shape[0]):
            pts = tet_np[i]
            p0 = pos_np[pts[0]]
            p1 = pos_np[pts[1]]
            p2 = pos_np[pts[2]]
            p3 = pos_np[pts[3]]
            d1 = p1 - p0
            d2 = p2 - p0
            d3 = p3 - p0
            volume = np.dot(np.cross(d2, d1), d3) / 6.0
            total_vol += volume

        if total_rest == 0.0:
            return 0.0
        return (total_vol - total_rest) / total_rest


    def get_fps(self):
        if not hasattr(self, 'step_start_time') or not hasattr(self, 'step_end_time'):
            return 0.0
        dur = self.step_end_time - self.step_start_time
        if dur == 0:
            return 0.0
        else:
            return 1.0 / dur


def generate_muscle_id_colors(muscle_ids):
    """为每个 muscle_id 生成独特的颜色"""
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
