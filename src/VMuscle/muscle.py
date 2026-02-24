import argparse
import json
import time
import zipfile
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path

import numpy as np
import taichi as ti
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
    show_auxiliary_meshes: bool = False
    pause: bool = False
    show_wireframe: bool = False
    render_fps: int = 24
    color_bones: bool = False  # 是否按 muscle_id 给骨骼着色
    color_muscles: str = "tendonmask"  # 肌肉着色模式: None, "muscle_id", "tendonmask"
    contraction_ratio: float = 0.4       # max fiber contraction ratio (act=1.0 → fiber shortens to 60% of rest length)
    fiber_stiffness_scale: float = 200.0  # fiber stiffness multiplier (replaces hardcoded 10000.0)
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
    # if name == "attach": # we seperate ATTACH from PIN 
    #     return "pin"
    if name == "stitch" or name == "branchstitch":
        return "distance"
    if name == "attachnormal":
        return "distanceline"
    return name


def pick_arch(name: str):
    name = name.lower()
    if name == "vulkan":
        return ti.vulkan
    if name == "cpu":
        return ti.cpu
    if name == "cuda":
        return ti.cuda
    return ti.cpu


def load_config(path: Path) -> SimConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Use dataclass field list so defaults live in SimConfig and we only override provided keys
    from dataclasses import fields
    kwargs = {}
    for fld in fields(SimConfig):
        name = fld.name
        if name in data:
            # special-case Path field
            if name == "geo_path" or name == "bone_geo_path":  
                kwargs[name] = Path(data[name])
            else:
                kwargs[name] = data[name]

    return SimConfig(**kwargs)

def load_mesh_json(path):
    import json
    # path="data/model/dragon.json"
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
    # path = "data/model1/bunny_small/bunny_small.node"
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


def read_auxiliary_meshes(ground_path="data/model/ground.obj", coord_path="data/model/coord.obj"):
    """
    读取辅助网格，包括地面和坐标系。

    Examples::

        # (before the render loop)
        ground, coord, ground_indices, coord_indices = read_auxiliary_meshes()
        # ...
        # (in the render loop)
        scene.mesh(ground, indices=ground_indices, color=(0.5,0.5,0.5))
        scene.mesh(coord, indices=coord_indices, color=(0.5, 0, 0))
    """
    def read_mesh(mesh_path, scale=[1.0, 1.0, 1.0], shift=[0, 0, 0]):
        import trimesh
        print("Using trimesh read ", mesh_path)
        mesh = trimesh.load(mesh_path)
        mesh.vertices *= scale
        mesh.vertices += shift
        return mesh.vertices, mesh.faces

    ground_, ground_indices_ = read_mesh(ground_path)
    coord_, coord_indices_ = read_mesh(coord_path)
    ground_indices_ = ground_indices_.flatten()
    coord_indices_ = coord_indices_.flatten()
    ground = ti.Vector.field(3, dtype=ti.f32, shape=ground_.shape[0])
    ground.from_numpy(ground_)
    coord = ti.Vector.field(3, dtype=ti.f32, shape=coord_.shape[0])
    coord.from_numpy(coord_)
    ground_indices = ti.field(dtype=ti.i32, shape=ground_indices_.shape[0])
    ground_indices.from_numpy(ground_indices_)
    coord_indices = ti.field(dtype=ti.i32, shape=coord_indices_.shape[0])
    coord_indices.from_numpy(coord_indices_)

    return ground, coord, ground_indices, coord_indices


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


@ti.func
def flatten(mat):
    # 只支持3x3矩阵，返回长度为9的向量
    ret = ti.Vector.zero(ti.f32, 9)
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            ret[i * 3 + j] = mat[i, j]
    return ret


# pbd_constraints.cl:L99
# 将dp累加到dP[pt]，并将dPw[pt]加1。dPw用于记录累加次数，最后更新时需要除以dPw以获得平均值。
@ti.func
def updatedP(dP, dPw, dp: ti.types.vector(3, ti.f32), pt: ti.i32):
    for j in ti.static(range(3)):
        ti.atomic_add(dP[pt][j], dp[j])
    ti.atomic_add(dPw[pt], 1.0)

@ti.func
def fem_flags(ctype: ti.i32) -> ti.i32:
    flags = 0
    if (ctype == TRIARAP or ctype == TETARAP or ctype == TETARAPVOL or
            ctype == TETARAPNORM or ctype == TETARAPNORMVOL):
        flags |= LINEARENERGY
    if (ctype == TRIARAPNORM or ctype == TETARAPNORM or ctype == TETARAPNORMVOL or
            ctype == TETFIBERNORM):
        flags |= NORMSTIFFNESS
    return flags

@ti.func
def project_to_line(p: ti.types.vector(3, ti.f32),
                    orig: ti.types.vector(3, ti.f32),
                    direction: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32):
    return orig + direction * (p - orig).dot(direction)


# math utils
@ti.func
def outer_product( v):
    return ti.Matrix([[v[i] * v[j] for j in ti.static(range(3))] for i in ti.static(range(3))])

@ti.func
def ssvd( F):
    U, sig, V = ti.svd(F)
    if U.determinant() < 0:
        for i in ti.static(range(3)):
            U[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    if V.determinant() < 0:
        for i in ti.static(range(3)):
            V[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    return U, sig, V

@ti.func
def polar_decomposition( F):
    U, sig, V = ssvd(F)
    R = U @ V.transpose()
    S = V @ sig @ V.transpose()
    return S, R

@ti.func
def invariant4(F, fiber):
    I4 = 1.0
    if F.determinant() < 0.0:
        S, _ = polar_decomposition(F)
        Sw = S @ fiber
        I4 = fiber.dot(Sw)
    return I4

@ti.func
def invariant5(F, fiber):
    return (F @ fiber).norm_sqr()

@ti.func
def squared_norm2(a: ti.types.matrix(2, 2, ti.f32)) -> ti.f32:
    return a.norm_sqr()

@ti.func
def squared_norm3(a: ti.types.matrix(3, 3, ti.f32)) -> ti.f32:
    return a.norm_sqr()

@ti.func
def mat3_to_quat(m: ti.types.matrix(3, 3, ti.f32)) -> ti.types.vector(4, ti.f32):
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    qw, qx, qy, qz = 0.0, 0.0, 0.0, 0.0
    if trace > 0.0:
        s = ti.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = ti.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = ti.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = ti.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    return ti.Vector([qx, qy, qz, qw])

@ti.func
def triangle_xform_and_area(p0: ti.types.vector(3, ti.f32),
                            p1: ti.types.vector(3, ti.f32),
                            p2: ti.types.vector(3, ti.f32)):
    e0 = p1 - p0
    e1 = p2 - p0
    n = e1.cross(e0)
    nlen = n.norm()
    area = 0.5 * nlen
    xform = ti.Matrix.zero(ti.f32, 3, 3)
    if nlen > 1e-12:
        z = n / nlen
        y = e0.cross(z)
        ylen = y.norm()
        if ylen > 1e-12:
            y = y / ylen
        x = y.cross(z)
        xform[0, 0], xform[0, 1], xform[0, 2] = x
        xform[1, 0], xform[1, 1], xform[1, 2] = y
        xform[2, 0], xform[2, 1], xform[2, 2] = z
    return xform, area


@ti.func
def get_inv_mass(idx: ti.i32, mass: ti.template(), stopped: ti.template())-> ti.f32:
    res = 0.0
    if (stopped[idx]):
        res = 0.0
    else:
        m = mass[idx]
        res = 1.0 / m if m > 0.0 else 0.0
    return res
    
@ti.data_oriented
class MuscleSim:   
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        ti.init(arch=pick_arch(cfg.arch))

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
        self._build_surface_tris_ti_field()
        self.build_constraints()
            
        self.use_jacobi = False
        self.contraction_ratio = self.cfg.contraction_ratio
        self.fiber_stiffness_scale = self.cfg.fiber_stiffness_scale
        self.dt = self.cfg.dt / self.cfg.num_substeps
        self.step_cnt = 0
        
        print("Initializing visualization...")
        print("Renderer mode:", cfg.render_mode)
        self.vis = Visualizer(cfg, self)
        # 生成肌肉顶点颜色
        self.vis._generate_muscle_colors()
        # 为 attach 约束准备可视化数据
        self.vis._init_attach_vis(self.attach_constraints)
        print("All initialization done.")


    def _build_surface_tris_ti_field(self):
        self.surface_tris = build_surface_tris(self.tet_np)
        self.n_tris = self.surface_tris.shape[0]
        self.surface_tris_field = ti.field(dtype=ti.i32, shape=(self.n_tris * 3,))
        self.surface_tris_field.from_numpy(self.surface_tris.reshape(-1))


    def _allocate_fields(self):
        n_v = self.n_verts
        n_tet = self.tet_np.shape[0]

        self.pos = ti.Vector.field(3, dtype=ti.f32, shape=n_v)
        self.pprev = ti.Vector.field(3, dtype=ti.f32, shape=n_v)
        self.pos0 = ti.Vector.field(3, dtype=ti.f32, shape=n_v)
        self.vel = ti.Vector.field(3, dtype=ti.f32, shape=n_v)
        self.force = ti.Vector.field(3, dtype=ti.f32, shape=n_v)
        self.mass = ti.field(dtype=ti.f32, shape=n_v)
        self.stopped = ti.field(dtype=ti.i32, shape=n_v)
        self.v_fiber_dir = ti.Vector.field(3, dtype=ti.f32, shape=n_v)
        self.dP = ti.Vector.field(3, dtype=ti.f32, shape=n_v)
        self.dPw = ti.field(dtype=ti.f32, shape=n_v)
        self.tet_indices = ti.Vector.field(4, dtype=ti.i32, shape=n_tet)

        self.rest_volume = ti.field(dtype=ti.f32, shape=n_tet)
        self.rest_matrix = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_tet)
        self.activation = ti.field(dtype=ti.f32, shape=n_tet)



    # Reference: C:\Program Files\Side Effects Software\Houdini 21.0.440\houdini\vex\include\pbd_constraints.h:L1291 
    def _batch_compute_tet_rest_matrices(self):
        """Batch compute rest matrices for all tets using vectorized numpy.
        Returns cached (restmatrices, volumes, valid) tuple.
        restmatrices: (N_tet, 3, 3), volumes: (N_tet,), valid: (N_tet,) bool mask."""
        if hasattr(self, '_cached_tet_rest'):
            return self._cached_tet_rest
        tet_pos = self.pos0_np[self.tet_np]             # (N_tet, 4, 3)
        # M = transpose(set(p0-p3, p1-p3, p2-p3)), i.e. columns are edge vectors
        cols = tet_pos[:, :3, :] - tet_pos[:, 3:4, :]   # (N_tet, 3, 3): rows are edges
        M = np.transpose(cols, (0, 2, 1))                # (N_tet, 3, 3): columns are edges
        dets = np.linalg.det(M)                          # (N_tet,)
        volumes = dets / 6.0
        valid = np.abs(dets) > 1e-30
        restmatrices = np.zeros_like(M)
        if np.any(valid):
            restmatrices[valid] = np.linalg.inv(M[valid])
        self._cached_tet_rest = (restmatrices, volumes, valid)
        return self._cached_tet_rest

    def compute_tet_rest_matrix(self, pt0, pt1, pt2, pt3, scale=1.0):
        """
        计算四面体的逆材料矩阵(restmatrix)和体积(volume)。
        参数:
            pt0, pt1, pt2, pt3: 顶点索引
            scale: 缩放因子，默认为1.0
        返回:
            (restmatrix, volume) 其中restmatrix为3x3矩阵，volume为float
            若行列式为0，返回(None, 0.0)
        """
        p = self.pos0_np
        p0 = p[pt0]
        p1 = p[pt1]
        p2 = p[pt2]
        p3 = p[pt3]
        # 计算材料矩阵M
            # matrix3 M = transpose(scale * set(p0 - p3, p1 - p3, p2 - p3));
        M = scale * np.stack([p0 - p3, p1 - p3, p2 - p3]).T  # shape (3,3)
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

        # xform rows are triangle-space axes in world space
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
        """
        计算四面体的rest matrix、materialW和体积。
        返回: volume, materialW (3,)
        """
        # 1. 计算rest matrix和体积
        restm, volume = self.compute_tet_rest_matrix(pt0, pt1, pt2, pt3, scale=1.0)
        if restm is None:
            return 0.0, np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # 2. materialW
        materialW = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if self.v_fiber_np is not None:
            w = self.v_fiber_np[pt0] + self.v_fiber_np[pt1] + self.v_fiber_np[pt2] + self.v_fiber_np[pt3]
            norm = np.linalg.norm(w)
            if norm > 1e-8:
                materialW = w / norm
        # materialW = materialW * transpose(restm)
        materialW = materialW @ restm.T
        return volume, materialW


    # Reference: C:\Program Files\Side Effects Software\Houdini 21.0.440\houdini\vex\include\pbd_constraints.h:L1291
    def create_tet_fiber_constraint(self, params):
        """Vectorized: compute fiber constraints for all tets using batch rest matrices."""
        stiffness = params.get('stiffness', 1.0)
        dampingratio = params.get('dampingratio', 0.0)
        restmatrices, volumes, valid = self._batch_compute_tet_rest_matrices()

        n_tet = len(self.tet_np)
        # Batch compute materialW for all tets
        if self.v_fiber_np is not None:
            fiber_verts = self.v_fiber_np[self.tet_np]  # (N_tet, 4, 3)
            w = fiber_verts.sum(axis=1)                 # (N_tet, 3)
            norms = np.linalg.norm(w, axis=1, keepdims=True)
            default_w = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (n_tet, 1))
            norm_ok = (norms > 1e-8).ravel()
            materialW = default_w.copy()
            materialW[norm_ok] = w[norm_ok] / norms[norm_ok]
        else:
            materialW = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (n_tet, 1))

        # materialW[i] @ restmatrices[i].T = einsum('j,kj->k', materialW[i], restmatrices[i])
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
                pts=ti.Vector([int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3])]),
                stiffness=stiffness,
                dampingratio=dampingratio,
                tetid=i,
                L=ti.Vector([0.0, 0.0, 0.0]),
                restlength=float(vol),
                restvector=ti.Vector(restvec4),
                restdir=ti.Vector([0.0, 0.0, 0.0]),
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
                    pts=ti.Vector([int(i), -1, 0, 0]),
                    stiffness=stiffness,
                    dampingratio=dampingratio,
                    tetid=0,
                    L=ti.Vector([0.0, 0.0, 0.0]),
                    restlength=0.0,
                    restvector=ti.Vector([float(restpos[0]), float(restpos[1]), float(restpos[2]), 1.0]),
                    restdir=ti.Vector([0.0, 0.0, 0.0]),
                    compressionstiffness=-1.0,
                )
                constraints.append(c)
        return constraints


    def map_pts2tets(self, tet):
        """
        建立从点到四面体的映射。
        参数:
            tet: (n_tet,4) numpy array
        返回:
        pt2tet: dict, key为点索引，value为包含该点的四面体索引列表
        """
        pt2tet = {}
        for i, tet_verts in enumerate(tet):
            for v in tet_verts:
                if v not in pt2tet:
                    pt2tet[v] = []
                pt2tet[v].append(i)
        return pt2tet


    def one2multi_dict_to_np(self, one2multi_dict, n_verts):
        """
        将点到多值的dict转换为等长的numpy array。
        参数:
            one2multi_dict: dict, key为点索引，value为包含该点的值列表
            n_verts: int, 点的总数
        返回:
            result: np.ndarray, shape (n_verts, max_len)，不足的部分填-1
        """
        max_len = max(len(v) for v in one2multi_dict.values())
        result = -np.ones((n_verts, max_len), dtype=np.int32)
        for k, v in one2multi_dict.items():
            result[k, :len(v)] = v
        return result



    def load_bone_geo(self, target_path):
        if not hasattr(self, 'bone_pos_field') and Path(target_path).exists():
            from VMuscle.geo import Geo
            self.bone_geo = Geo(target_path)
            if len(self.bone_geo.positions) == 0:
                print(f"Warning: No vertices found in {target_path}")
                self.bone_pos = np.zeros((0, 3), dtype=np.float32)
                self.bone_muscle_ids = {}  # muscle_id -> vertex indices
            else:
                self.bone_pos = np.asarray(self.bone_geo.positions, dtype=np.float32)
                # Load indices for visualization
                if hasattr(self.bone_geo, 'indices'):
                    self.bone_indices_np = np.asarray(self.bone_geo.indices, dtype=np.int32)
                elif hasattr(self.bone_geo, 'vert'):
                    # Flatten vert if it's a list of lists (e.g. triangles/quads)
                    self.bone_indices_np = np.array(self.bone_geo.vert, dtype=np.int32).flatten()
                else:
                    self.bone_indices_np = np.zeros(0, dtype=np.int32)

                # Parse muscle_id from point attributes
                self.bone_muscle_ids = {}  # muscle_id -> vertex indices
                self.bone_vertex_colors = None  # 顶点颜色数组
                
                if hasattr(self.bone_geo, 'pointattr') and 'muscle_id' in self.bone_geo.pointattr:
                    muscle_ids = self.bone_geo.pointattr['muscle_id']
                    # muscle_ids is per-vertex, group vertices by muscle_id
                    for v_idx, mid in enumerate(muscle_ids):
                        if mid not in self.bone_muscle_ids:
                            self.bone_muscle_ids[mid] = []
                        self.bone_muscle_ids[mid].append(v_idx)
                    # Convert lists to numpy arrays
                    for mid in self.bone_muscle_ids:
                        self.bone_muscle_ids[mid] = np.array(self.bone_muscle_ids[mid], dtype=np.int32)
                    
                    print(f"Bone muscle_id groups: {list(self.bone_muscle_ids.keys())}")
                    
                    # 仅当 color_bones=True 时生成颜色
                    if self.cfg.color_bones:
                        # 为每个 muscle_id 分配颜色
                        unique_ids = sorted(self.bone_muscle_ids.keys())
                        self.bone_id_colors = self._generate_muscle_id_colors(unique_ids)
                        
                        # 创建顶点颜色数组
                        self.bone_vertex_colors = np.zeros((len(muscle_ids), 3), dtype=np.float32)
                        for v_idx, mid in enumerate(muscle_ids):
                            self.bone_vertex_colors[v_idx] = self.bone_id_colors[mid]
                        print("Bone coloring by muscle_id enabled")

            if self.bone_pos.shape[0] > 0:
                self.bone_pos_field = ti.Vector.field(3, dtype=ti.f32, shape=self.bone_pos.shape[0])
                self.bone_pos_field.from_numpy(self.bone_pos)
                if self.bone_indices_np.shape[0] > 0:
                    self.bone_indices_field = ti.field(dtype=ti.i32, shape=self.bone_indices_np.shape[0])
                    self.bone_indices_field.from_numpy(self.bone_indices_np)
                # 创建颜色 field
                if self.bone_vertex_colors is not None:
                    self.bone_colors_field = ti.Vector.field(3, dtype=ti.f32, shape=self.bone_vertex_colors.shape[0])
                    self.bone_colors_field.from_numpy(self.bone_vertex_colors)

        if hasattr(self, 'bone_pos'):
            return self.bone_geo, self.bone_pos
        else: # file does not exist or is empty
            return None, np.zeros((0,3), dtype=np.float32)




    def create_attach_constraints(self, params):
        """
        创建attach类型约束（将肌肉点attach到骨骼上）。
        使用mask过滤和自动最近点搜索。
        
        在Houdini当中，实际上使用的是target_prim & target_uv来指定target位置，每次运行kernel前利用VEX提前计算target_pos然后保存到restvector中。
        kernel update时只需要restvector即可完成投影计算。
        因此其pts中只有source点索引。
        但是我们这里额外保存了target_pt到pts的第二个位置中。

        参数:
            params: dict，包含：
                - mask_name: str，肌肉mask属性名（如'muscletobonemask'）
                - target_path: str, e.g. "data/model/bicep_bone.geo" 骨骼模型路径
                - mask_threshold: float，mask阈值，默认0.75
                - stiffness: float，约束刚度，默认1e10
                - dampingratio: float，阻尼比，默认0.0
        """
        constraints = []
        
        # 获取mask属性名和阈值
        mask_name = params.get('mask_name')
        target_path = params.get('target_path')
        mask_threshold = params.get('mask_threshold', 0.75)
        stiffness = params.get('stiffness', 1e10)
        dampingratio = params.get('dampingratio', 0.0)

        # 加载骨骼数据
        self.bone_geo, self.bone_pos = self.load_bone_geo(target_path)

        # 获取mask数据
        mask = np.asarray(getattr(self.geo, mask_name), dtype=np.float32) if hasattr(self.geo, mask_name) else None
        
        if mask is None:
            Warning(f"Warning: mask '{mask_name}' not found in geometry.")
            return []

        # 找出mask值大于阈值的点
        valid_src_indices = np.where(mask > mask_threshold)[0].astype(np.int32)
        
        if len(valid_src_indices) == 0:
            Warning(f"Warning: No vertices with mask > {mask_threshold}")
            return []
            
        # Build KDTree for bone positions and batch nearest-neighbor search
        bone_tree = cKDTree(self.bone_pos)
        src_positions = self.pos0_np[valid_src_indices]
        dists, tgt_indices = bone_tree.query(src_positions, k=1)

        # Build pt2tet mapping once
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
                pts=ti.Vector([int(src_idx), -1, int(tgt_idx), 0]),
                stiffness=stiffness,
                dampingratio=dampingratio,
                tetid=tetid,
                L=ti.Vector([0.0, 0.0, 0.0]),
                restlength=restlength,
                restvector=ti.Vector([float(target_pos[0]), float(target_pos[1]), float(target_pos[2]), 1.0]),
                restdir=ti.Vector([0.0, 0.0, 0.0]),
                compressionstiffness=-1.0,
            )
            constraints.append(c)

        return constraints

    def create_distance_line_constraints(self, params):
        """
        创建distanceline（attachnormal）类型约束。
        使用mask过滤和自动最近点搜索。
        
        与attach不同之处在于，
        source点会被投影到以source点为原点、restdir为方向的射线上，
        然后进行distance约束求解。

        在Houdini当中，实际上使用的是target_prim & target_uv来指定target位置，每次运行kernel前利用VEX提前计算target_pos然后保存到restvector中。
        kernel update时只需要restvector即可完成投影计算。
        因此其pts中只有source点索引。
        但是我们这里额外保存了target_pt到pts的第二个位置中。

        参数:
            params: dict，包含：
                - mask_name: str，肌肉mask属性名（如'muscleendmask'）
                - target_path: str, e.g. "data/model/bicep_bone.geo" 骨骼模型路径
                - mask_threshold: float，mask阈值，默认0.75
                - stiffness: float，约束刚度，默认1e5
                - dampingratio: float，阻尼比，默认0.1
        """
        constraints = []
        
        # 获取mask属性名和阈值
        mask_name = params.get('mask_name')
        target_path = params.get('target_path')
        mask_threshold = params.get('mask_threshold', 0.75)
        stiffness = params.get('stiffness', 1e10)
        dampingratio = params.get('dampingratio', 0.0)

        # 加载骨骼数据
        self.bone_geo, self.bone_pos = self.load_bone_geo(target_path)

        # 获取mask数据
        mask = np.asarray(getattr(self.geo, mask_name), dtype=np.float32) if hasattr(self.geo, mask_name) else None
        
        if mask is None:
            Warning(f"Warning: mask '{mask_name}' not found in geometry.")
            return []

        # 找出mask值大于阈值的点
        valid_src_indices = np.where(mask > mask_threshold)[0].astype(np.int32)
        
        if len(valid_src_indices) == 0:
            Warning(f"Warning: No vertices with mask > {mask_threshold}")
            return []
            
        # Build KDTree for bone positions and batch nearest-neighbor search
        bone_tree = cKDTree(self.bone_pos)
        src_positions = self.pos0_np[valid_src_indices]
        dists, tgt_indices = bone_tree.query(src_positions, k=1)

        # Vectorized direction computation
        target_positions = self.bone_pos[tgt_indices]
        directions = target_positions - src_positions
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        if np.any(norms < 1e-9):
            bad = np.where(norms.ravel() < 1e-9)[0]
            raise ValueError(f"source and target points too close at indices: {valid_src_indices[bad]}")
        directions = directions / norms

        # Build pt2tet mapping once
        if not hasattr(self, 'pt2tet'):
            self.pt2tet = self.map_pts2tets(self.tet_np)

        for j, src_idx in enumerate(valid_src_indices):
            tgt_idx = int(tgt_indices[j])
            target_pos = target_positions[j]
            direction = directions[j]
            tetid = self.pt2tet.get(src_idx, [-1])[0]

            c = dict(
                type=DISTANCELINE,
                pts=ti.Vector([int(src_idx), int(tgt_idx), 0, 0]),
                stiffness=stiffness,
                dampingratio=dampingratio,
                tetid=tetid,
                L=ti.Vector([0.0, 0.0, 0.0]),
                restlength=0.0,
                restvector=ti.Vector([float(target_pos[0]), float(target_pos[1]), float(target_pos[2]), 1.0]),
                restdir=ti.Vector([float(direction[0]), float(direction[1]), float(direction[2])]),
                compressionstiffness=-1.0,
            )
            constraints.append(c)

        return constraints

    def create_tet_arap_constraints(self, params):
        """Vectorized: uses cached batch rest matrices."""
        stiffness = params.get('stiffness', 1e10)
        dampingratio = params.get('dampingratio', 0.0)
        restmatrices, volumes, valid = self._batch_compute_tet_rest_matrices()

        valid_indices = np.where(valid)[0]
        constraints = []
        for i in valid_indices:
            tet = self.tet_np[i]
            c = dict(
                type=TETARAP,
                pts=ti.Vector([int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3])]),
                stiffness=stiffness,
                dampingratio=dampingratio,
                tetid=int(i),
                L=ti.Vector([0.0, 0.0, 0.0]),
                restlength=float(volumes[i]),
                restvector=ti.Vector([0.0, 0.0, 0.0, 1.0]),
                restdir=ti.Vector([0.0, 0.0, 0.0]),
                compressionstiffness=-1.0,
            )
            constraints.append(c)
        return constraints

    def create_tri_arap_constraints(self, params):
        """Vectorized: batch compute 2x2 rest matrices for surface triangles."""
        stiffness = params.get('stiffness', 1e10)
        dampingratio = params.get('dampingratio', 0.0)

        if not hasattr(self, 'surface_tris'):
            self.surface_tris = build_surface_tris(self.tet_np)
        tris = self.surface_tris
        if len(tris) == 0:
            return []

        # Batch compute tri rest matrices
        p = self.pos0_np
        tri_arr = np.asarray(tris)  # (N_tri, 3)
        p0 = p[tri_arr[:, 0]]  # (N, 3)
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

        xform = np.stack([x, y, z], axis=1)  # (N, 3, 3)
        xformT = np.transpose(xform, (0, 2, 1))
        P0 = np.einsum('ni,nij->nj', p0, xformT)
        P1 = np.einsum('ni,nij->nj', p1, xformT)
        P2 = np.einsum('ni,nij->nj', p2, xformT)
        col0 = P0[:, :2] - P2[:, :2]
        col1 = P1[:, :2] - P2[:, :2]
        M = np.stack([col0, col1], axis=2)  # (N, 2, 2)
        dets = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
        valid = valid & (np.abs(dets) > 1e-30)
        areas = np.abs(dets / 2.0)
        # Batch 2x2 inverse using analytical formula
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
                pts=ti.Vector([int(tri[0]), int(tri[1]), int(tri[2]), 0]),
                stiffness=stiffness,
                dampingratio=dampingratio,
                tetid=int(i),
                L=ti.Vector([0.0, 0.0, 0.0]),
                restlength=float(areas[i]),
                restvector=ti.Vector(restvec4),
                restdir=ti.Vector([0.0, 0.0, 0.0]),
                compressionstiffness=-1.0,
            )
            constraints.append(c)
        return constraints

    # Reference: C:\Program Files\Side Effects Software\Houdini 21.0.440\houdini\vex\include\pbd_constraints.h:L980
    def create_tet_volume_constraint(self, params):
        """Vectorized: compute volume constraints for all tets using batch rest matrices."""
        stiffness = params.get('stiffness', 1e10)
        dampingratio = params.get('dampingratio', 0.0)
        restmatrices, volumes, valid = self._batch_compute_tet_rest_matrices()

        valid_indices = np.where(valid)[0]
        constraints = []
        for i in valid_indices:
            tet = self.tet_np[i]
            c = dict(
                type=TETVOLUME,
                pts=ti.Vector([int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3])]),
                stiffness=stiffness,
                dampingratio=dampingratio,
                compressionstiffness=-1.0,
                tetid=int(i),
                L=ti.Vector([0.0, 0.0, 0.0]),
                restlength=float(volumes[i]),
                restvector=ti.Vector([0.0, 0.0, 0.0, 0.0]),
                restdir=ti.Vector([0.0, 0.0, 0.0]),
            )
            constraints.append(c)
        return constraints

    
    def build_constraints(self):
        print("Building constraints...")
        constraint_struct = ti.types.struct(
            type=ti.i32,
            cidx=ti.i32, # constraint index
            pts=ti.types.vector(4, ti.i32),
            stiffness=ti.f32,
            dampingratio=ti.f32,
            tetid=ti.i32,
            L=ti.types.vector(3, ti.f32),
            restlength=ti.f32,
            restvector=ti.types.vector(4, ti.f32),
            restdir=ti.types.vector(3, ti.f32),
            compressionstiffness=ti.f32,
        )

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
                ...

        self.raw_constraints = all_constraints.copy()

        n_cons = len(all_constraints)
        if n_cons > 0:
            # Assign cidx to each constraint
            for i, c in enumerate(all_constraints):
                c['cidx'] = i

            # Batch extract constraint data into numpy arrays for from_numpy
            type_arr = np.array([c['type'] for c in all_constraints], dtype=np.int32)
            cidx_arr = np.arange(n_cons, dtype=np.int32)
            pts_arr = np.array([[c['pts'][j] for j in range(4)] for c in all_constraints], dtype=np.int32)
            stiffness_arr = np.array([c['stiffness'] for c in all_constraints], dtype=np.float32)
            dampingratio_arr = np.array([c['dampingratio'] for c in all_constraints], dtype=np.float32)
            tetid_arr = np.array([c['tetid'] for c in all_constraints], dtype=np.int32)
            L_arr = np.array([[c['L'][j] for j in range(3)] for c in all_constraints], dtype=np.float32)
            restlength_arr = np.array([c['restlength'] for c in all_constraints], dtype=np.float32)
            restvector_arr = np.array([[c['restvector'][j] for j in range(4)] for c in all_constraints], dtype=np.float32)
            restdir_arr = np.array([[c['restdir'][j] for j in range(3)] for c in all_constraints], dtype=np.float32)
            compressionstiffness_arr = np.array([c['compressionstiffness'] for c in all_constraints], dtype=np.float32)

            self.cons = constraint_struct.field(shape=n_cons)
            self.cons.type.from_numpy(type_arr)
            self.cons.cidx.from_numpy(cidx_arr)
            self.cons.pts.from_numpy(pts_arr)
            self.cons.stiffness.from_numpy(stiffness_arr)
            self.cons.dampingratio.from_numpy(dampingratio_arr)
            self.cons.tetid.from_numpy(tetid_arr)
            self.cons.L.from_numpy(L_arr)
            self.cons.restlength.from_numpy(restlength_arr)
            self.cons.restvector.from_numpy(restvector_arr)
            self.cons.restdir.from_numpy(restdir_arr)
            self.cons.compressionstiffness.from_numpy(compressionstiffness_arr)
        else:
            self.cons = constraint_struct.field(shape=0)
            self.raw_constraints = []

        _dt_total = time.perf_counter() - _t_total
        print(f"Built {self.cons.shape[0]} constraints total. [{_dt_total*1000:.0f}ms]")

        # Reaction accumulator for bilateral attach coupling (mass-independent C×n)
        self.reaction_accum = ti.Vector.field(3, dtype=ti.f32, shape=max(n_cons, 1))


    def _init_fields(self):
        self.pos0.from_numpy(self.pos0_np)
        self.pos.from_numpy(self.pos0_np)
        self.vel.fill(0)
        self.force.fill(0)
        self.mass.fill(0)
        self.stopped.fill(0)
        self.tet_indices.from_numpy(self.tet_np)

        if self.v_fiber_np is None:
            self.v_fiber_np = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (self.n_verts, 1))
        self.v_fiber_dir.from_numpy(self.v_fiber_np)

        self.tendonmask = ti.field(dtype=ti.f32, shape=(self.tet_np.shape[0],))
        self._compute_cell_tendon_mask(self.tet_indices, self.v_tendonmask_np, self.tendonmask)

        self.activation.fill(0.0)
        self.total_rest_volume = ti.field(dtype=ti.f32, shape=())
        
        print("Initialized fields done.")


    def reset(self):
        self.pos.from_numpy(self.pos0_np)
        self.pprev.from_numpy(self.pos0_np)
        self.vel.fill(0)
        self.force.fill(0)  
        self.activation.fill(0.0)
        self.clear()
        self.step_cnt = 1

    @ti.kernel
    def _update_attach_targets_kernel(self):
        for c in range(self.cons.shape[0]):
            ctype = self.cons[c].type
            if ctype == ATTACH:
                # An attach constraint where pts[2] stores the bone vertex index
                tgt_idx = self.cons[c].pts[2]
                if tgt_idx >= 0:
                    # Get the current bone position from the field
                    target_pos = self.bone_pos_field[tgt_idx]
                    # Update the restvector for the constraint solver
                    self.cons[c].restvector.xyz = target_pos
            elif ctype == DISTANCELINE:
                # For attachnormal, pts[1] stores the bone vertex index
                tgt_idx = self.cons[c].pts[1]
                if tgt_idx >= 0:
                    # Get the current bone position from the field
                    target_pos = self.bone_pos_field[tgt_idx]
                    # Update the restvector (line_origin) for the constraint solver
                    self.cons[c].restvector.xyz = target_pos
                    # NOTE: This does not update restdir. The line direction remains static.
                    # This is a simplification from the VEX reference implementation.

    def update_attach_targets(self):
        """
        Updates the target positions for all 'attach' and 'attachnormal' constraints.
        This should be called each frame before the simulation step if the bone geometry is animated.
        It reads the current position of bone vertices and updates the `restvector` of the constraints.
        """
        if (
            hasattr(self, "bone_pos_field")
            and self.bone_pos_field.shape[0] > 0
            and hasattr(self, "attach_constraints")
            and len(self.attach_constraints) > 0
        ):
            self._update_attach_targets_kernel()


    @ti.kernel
    def close(self):
        if hasattr(self, "window"):
            self.window.running = False
            self.window.destroy()
            
    @ti.kernel
    def _compute_cell_tendon_mask(self, tet_indices: ti.template(), v_tendonmask_np: ti.types.ndarray(), tendon_mask: ti.template()):
        for c in tet_indices:
            sum_mask = 0.0
            for i in ti.static(range(4)):
                v_idx = tet_indices[c][i]
                sum_mask += v_tendonmask_np[v_idx]
            tendon_mask[c] = sum_mask / 4.0

    @ti.kernel
    def _compute_cell_fiber_dir(self, tet_indices: ti.template(),v_fiber_dir: ti.template(), c_fiber_dir: ti.template()):
        for c in tet_indices:
            pts = tet_indices[c]
            v_dirs = ti.Vector.zero(ti.f32, 3)
            for i in ti.static(range(4)):
                v_dirs += v_fiber_dir[pts[i]]
            norm = v_dirs.norm() + 1e-8
            c_fiber_dir[c] = v_dirs / norm



    @ti.func
    def Ds_rest(self, pts):
        return ti.Matrix.cols([self.pos0[pts[i]] - self.pos0[pts[3]] for i in range(3)])

    @ti.func
    def Ds(self, pts):
        return ti.Matrix.cols([self.pos[pts[i]] - self.pos[pts[3]] for i in range(3)])


    @ti.kernel
    def _precompute_rest(self):
        n_tet = self.rest_volume.shape[0]
        self.total_rest_volume[None] = 0.0
        for c in range(n_tet):
            pts = self.tet_indices[c]
            Dm = self.Ds_rest(pts)
            self.rest_volume[c] = ti.abs(Dm.determinant()) / 6.0
            self.total_rest_volume[None] += self.rest_volume[c]
            for i in range(4):
                ti.atomic_add(self.mass[pts[i]], self.rest_volume[c] * self.cfg.density / 4.0)
            self.rest_matrix[c] = Dm.inverse()

    @ti.func
    def get_compression_stiffness(self, c: int) -> float:
        # Compression Stiffness defaults to -1, which means use regular stiffness.
        kstiff = self.cons[c].stiffness
        kstiffcompress = self.cons[c].compressionstiffness 
        kstiffcompress = ti.select(kstiffcompress >= 0.0, kstiffcompress, kstiff)
        return kstiffcompress
    

    @ti.func
    def transfer_tension(self, muscletension, tendonmask, minfiberscale=0.0, maxfiberscale=10.0):
        """
        Fit the muscle tension to [minfiberscale, maxfiberscale] based on the tendon mask value, then ramp it based on tendonmask.
        Corresponds to Houdini's transfer_tension in /obj/muscleTestDebug/musclesolvervellum1/vellumconfiguremuscles1

        //f@__vellumfiberscale applied in VellumConstraintProperties Dop
        // tendons have minimum range applied right away, belly of muscle responds to muscletension
        float tendontension = minfiberscale;
        float bellytension = fit(f@muscletension, 0, 1, minfiberscale, maxfiberscale);
        float rampedtension = tendontension * tendonmask + bellytension * (1-tendonmask);
        f@fiberscale = rampedtension;
        """
        fiberscale = minfiberscale + (1.0 - tendonmask) * muscletension * (maxfiberscale - minfiberscale)
        return fiberscale
    

    @ti.func
    def transfer_shape_and_bulge(
        tpose,
        muscletension,
        fibervolumescale=10.0,
    ):
        """
        tpose(targetshape) :  shape (3,)
        muscletension      : alpha ∈ [0,1]
        muscletension(bulgeamt & shapeamt): alpha ∈ [0,1]
        fibervolumescale  : 0,..inf     
        Reference: 
        transfer_shape_and_bulge aw in /obj/muscleTestDebug/musclesolvervellum1/vellumconfiguremuscles1
        float volfactor = cbrt( fit(bulgeamt, 0, 1, 1, bulgescale) );
        // used to blend with shape target in vellum's editable dive target
        v@restP = ( lerp(tpose, targetshape, shapeamt) )  * volfactor;    
        """
        # volume compensation factor
        volfactor = (1.0 + muscletension * (fibervolumescale - 1.0)) ** (1.0 / 3.0)

        # shape blend
        p_shape = (1.0 - muscletension) * tpose + muscletension * fibervolumescale

        # final rest position
        restP = p_shape * volfactor

        return restP


    @ti.kernel
    def solve_constraints(self):
        for c in range(self.cons.shape[0]):
            # common setup for all constraints
            # set stiffness for compression
            kstiffcompress = self.get_compression_stiffness(c)
            # jump over zero stiffness constraints
            if self.cons[c].stiffness <= 0.0:
                continue

            if self.cons[c].type == TETVOLUME:
                # Volume constraint
                pts = self.cons[c].pts
                loff = 0
                tetid = self.cons[c].tetid
                self.tet_volume_update_xpbd(
                    self.use_jacobi,
                    c,
                    self.cons,
                    self.pos,
                    self.pprev,
                    self.cons[c].restlength,
                    self.dP,
                    self.dPw,
                    tetid,
                    pts,
                    self.dt,
                    self.cons[c].stiffness,
                    kstiffcompress,
                    self.cons[c].dampingratio,
                    self.mass,
                    self.stopped,
                    loff
                )
            elif self.cons[c].type == TETFIBERNORM:
                # tet fiber norm constraint
                pts = self.cons[c].pts
                fiber_dir = (self.v_fiber_dir[pts[0]] + self.v_fiber_dir[pts[1]] + self.v_fiber_dir[pts[2]] + self.v_fiber_dir[pts[3]]) / 4.0
                tetid = self.cons[c].tetid
                Dminv = self.rest_matrix[tetid]
                acti = self.activation[tetid]
                _tendonmask = self.tendonmask[tetid]
                belly_factor = 1.0 - _tendonmask

                # stiffness: activation-dependent via transfer_tension (muscle gets stiffer when activated)
                fiberscale = self.transfer_tension(acti, _tendonmask)
                stiffness = self.cons[c].stiffness * fiberscale * self.fiber_stiffness_scale
                if stiffness <= 0.0: #if activation is zero, stiffness can be zero
                    continue

                # target fiber stretch: higher activation → shorter target length
                target_stretch = 1.0 - belly_factor * acti * self.contraction_ratio

                self.tet_fiber_update_xpbd(
                    self.use_jacobi,
                    self.pos,
                    self.pprev,
                    self.dP,
                    self.dPw,
                    c,
                    self.cons,
                    pts,
                    self.dt,
                    fiber_dir,
                    stiffness,
                    Dminv,
                    self.cons[c].dampingratio,
                    self.cons[c].restlength,
                    self.cons[c].restvector,
                    acti,
                    self.mass,
                    self.stopped,
                    target_stretch,
                )
            elif self.cons[c].type == DISTANCE:
                # distance constraint
                pts = self.cons[c].pts
                self.distance_update_xpbd(
                    self.use_jacobi,
                    c,
                    self.cons,
                    pts,
                    self.pos,
                    self.pprev,
                    self.dP,
                    self.dPw,
                    self.mass,
                    self.stopped,
                    self.cons[c].restlength,
                    self.cons[c].stiffness,
                    self.cons[c].dampingratio,
                    kstiffcompress,
                )
            elif self.cons[c].type == ATTACH:
                # attach constraint: bilateral update (same correction + reaction accum)
                pts = self.cons[c].pts
                pt_src = pts[0]
                self.attach_bilateral_update(
                    self.use_jacobi,
                    c,
                    self.cons,
                    pt_src,
                    self.cons[c].restvector.xyz,
                    self.pos[pt_src],
                    self.pos,
                    self.pprev,
                    self.dP,
                    self.dPw,
                    self.mass,
                    self.stopped,
                    self.cons[c].restlength,
                    self.cons[c].stiffness,
                    self.cons[c].dampingratio,
                    kstiffcompress,
                )
            elif self.cons[c].type == PIN:
                if self.cons[c].stiffness <= 0.0:
                    continue
                # pin constraint: fixed pin (constrain point to fixed position)
                pts = self.cons[c].pts
                pt_src = pts[0]
                
                # Fixed pin constraint: constrain point to fixed position
                self.distance_pos_update_xpbd(
                    self.use_jacobi,
                    c,
                    self.cons,
                    -1,
                    pt_src,
                    self.cons[c].restvector.xyz,
                    self.pos[pt_src],
                    self.pos,
                    self.pprev,
                    self.dP,
                    self.dPw,
                    self.mass,
                    self.stopped,
                    self.cons[c].restlength,
                    self.cons[c].stiffness,
                    self.cons[c].dampingratio,
                    kstiffcompress,
                )
            elif self.cons[c].type == DISTANCELINE: # distanceline is an alias of attachnormal
                # distance line constraint: source point is projected onto a line, then constrained to that projection
                pts = self.cons[c].pts
                pt_src = pts[0]
                pt_tgt = pts[1]
                p_src = self.pos[pt_src]
                line_origin = self.cons[c].restvector.xyz
                line_dir = self.cons[c].restdir
                
                # Project source point onto the line
                p_projected = project_to_line(p_src, line_origin, line_dir)
                
                # Apply distance constraint between projected point (p0) and source point (p1)
                self.distance_pos_update_xpbd(
                    self.use_jacobi,
                    c,
                    self.cons,
                    -1,  # pt0 = -1 means p0 is a fixed projected point
                    pt_src,  # pt1 is the source point to constrain
                    p_projected,  # p0 is the projected point (fixed on line)
                    p_src,  # p1 is the current source position
                    self.pos,
                    self.pprev,
                    self.dP,
                    self.dPw,
                    self.mass,
                    self.stopped,
                    self.cons[c].restlength,
                    self.cons[c].stiffness,
                    self.cons[c].dampingratio,
                    kstiffcompress,
                )
            elif self.cons[c].type == TETARAP:
                # tet arap constraint
                pts = self.cons[c].pts
                tetid = self.cons[c].tetid
                self.tet_arap_update_xpbd(
                    self.use_jacobi,
                    c,
                    self.cons,
                    pts,
                    self.dt,
                    self.pos,
                    self.pprev,
                    self.dP,
                    self.dPw,
                    self.mass,
                    self.stopped,
                    self.cons[c].restlength,
                    self.cons[c].restvector,
                    self.rest_matrix[tetid],
                    self.cons[c].stiffness,
                    self.cons[c].dampingratio,
                    fem_flags(self.cons[c].type),
                )
            elif self.cons[c].type == TRIARAP:
                # tri arap constraint
                pts = self.cons[c].pts
                self.tri_arap_update_xpbd(
                    self.use_jacobi,
                    c,
                    self.cons,
                    pts,
                    self.dt,
                    self.pos,
                    self.pprev,
                    self.dP,
                    self.dPw,
                    self.mass,
                    self.stopped,
                    self.cons[c].restlength,
                    self.cons[c].restvector,
                    self.cons[c].stiffness,
                    self.cons[c].dampingratio,
                    kstiffcompress,
                    fem_flags(self.cons[c].type),
                )

    @ti.kernel
    def integrate(self):
        for i in range(self.pos.shape[0]):
            extacc = ti.Vector([0.0, self.cfg.gravity, 0.0])  
            self.pprev[i] = self.pos[i]
            self.vel[i] = (1.0 - self.cfg.veldamping) * self.vel[i] 
            self.vel[i] += self.dt * extacc
            self.pos[i] += self.dt * self.vel[i]
    
    @ti.kernel
    def update_velocities(self):
        for i in range(self.pos.shape[0]):
            self.vel[i] = (self.pos[i] - self.pprev[i]) / self.dt

    @ti.kernel
    def calc_vol_error(self)-> ti.f32:
        total_vol = 0.0
        for c in range(self.rest_volume.shape[0]):
            pts = self.tet_indices[c]
            p0, p1, p2, p3 = self.pos[pts[0]], self.pos[pts[1]], self.pos[pts[2]], self.pos[pts[3]]
            d1 = p1 - p0
            d2 = p2 - p0
            d3 = p3 - p0
            volume = ((d2).cross(d1)).dot(d3) / 6.0
            total_vol += volume

        vol_err = (total_vol - self.total_rest_volume[None]) / self.total_rest_volume[None]
        return vol_err

    @ti.kernel
    def clear(self):
        for i in self.dP:
            self.dP[i] = ti.Vector([0.0, 0.0, 0.0])
            self.dPw[i] = 0.0
        for i in self.cons:
            self.cons[i].L = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def clear_reaction(self):
        """Clear reaction accumulator (call once per frame, NOT per substep)."""
        for i in range(self.cons.shape[0]):
            self.reaction_accum[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def apply_dP(self):
        for idx in self.pos:
            w = self.dPw[idx]
            if w > 1e-9:
                self.pos[idx] += self.dP[idx] / w


    @ti.func
    def inCompressBand(self,curlen: ti.f32, restlen: ti.f32) -> ti.i32:
        # 如果定义了HAS_compressstiffness，则返回(curlen < restlen)，否则返回0。
        res = 0
        if self.cfg.HAS_compressstiffness:
            if curlen < restlen:
                res = 1
            else:
                res = 0
        else:
            res = 0
        return res



    # Reference: C:\Program Files\Side Effects Software\Houdini 21.0.440\houdini\ocl\sim\pbd_constraints.cl:L956 tetVolumeUpdateXPBD
    @ti.func
    def tet_volume_update_xpbd(
        self,
        use_jacobi: ti.template(),
        cidx: ti.i32,
        cons: ti.template(),
        pos: ti.template(),
        pprev: ti.template(),
        restlength: ti.template(),
        dP: ti.template(),
        dPw: ti.template(),
        tetid: ti.i32,
        pts: ti.types.vector(4, ti.i32),
        dt: ti.f32,
        stiffness: ti.f32,
        kstiffcompress: ti.f32,
        kdampratio: ti.f32,
        mass: ti.template(),
        stopped: ti.template(),
        loff: ti.i32 
    ):   
        loff=0
        inv_masses = ti.Vector([0.0, 0.0, 0.0, 0.0])
        for i in ti.static(range(4)):
            inv_masses[i] = get_inv_mass(pts[i], mass, stopped)

        p0, p1, p2, p3 = pos[pts[0]], pos[pts[1]], pos[pts[2]], pos[pts[3]]

        d1 = p1 - p0
        d2 = p2 - p0
        d3 = p3 - p0
        grad1 = (d3.cross(d2)) / 6.0
        grad2 = (d1.cross(d3)) / 6.0
        grad3 = (d2.cross(d1)) / 6.0
        grad0 = -(grad1 + grad2 + grad3)

        w_sum = (inv_masses[0] * grad0.norm_sqr() +
                inv_masses[1] * grad1.norm_sqr() +
                inv_masses[2] * grad2.norm_sqr() +
                inv_masses[3] * grad3.norm_sqr())

        if w_sum > 1e-9:
            volume = ((d2).cross(d1)).dot(d3) / 6.0
            # print("volume:", volume)

            # if has kstiffcompress, then L use L[1], else L[0]
            comp = self.inCompressBand(volume, restlength)
            kstiff = ti.select(comp, kstiffcompress, stiffness)
            loff += comp
            loff = ti.min(loff, 2)

            if kstiff != 0.0:
                l = cons[cidx].L[loff]

                alpha = 1.0 / kstiff
                alpha /= dt * dt

                C = volume - restlength

                # Damping terms (use provided kdampratio)
                dsum = 0.0
                gamma = 1.0
                if kdampratio > 0.0:
                    prev0 = pprev[pts[0]]
                    prev1 = pprev[pts[1]]
                    prev2 = pprev[pts[2]]
                    prev3 = pprev[pts[3]]
                    beta = kstiff * kdampratio * dt * dt
                    gamma = alpha * beta / dt
                    dsum = grad0.dot(pos[pts[0]] - prev0) +grad1.dot(pos[pts[1]] - prev1) + grad2.dot(pos[pts[2]] - prev2) + grad3.dot(pos[pts[3]] - prev3)
                    dsum *= gamma
                    gamma += 1.0

                dlambda = (-C - alpha * l - dsum) / (gamma * w_sum + alpha)

                # Apply updates: either accumulate (Jacobi) or immediate (GS-like)
                if use_jacobi:
                    updatedP(dP, dPw, dlambda * inv_masses[0] * grad0, pts[0])
                    updatedP(dP, dPw, dlambda * inv_masses[1] * grad1, pts[1])
                    updatedP(dP, dPw, dlambda * inv_masses[2] * grad2, pts[2])
                    updatedP(dP, dPw, dlambda * inv_masses[3] * grad3, pts[3])
                else:
                    # Apply GS 
                    pos[pts[0]] +=   dlambda * inv_masses[0] * grad0
                    pos[pts[1]] +=   dlambda * inv_masses[1] * grad1
                    pos[pts[2]] +=   dlambda * inv_masses[2] * grad2
                    pos[pts[3]] +=   dlambda * inv_masses[3] * grad3
                    cons[cidx].L[loff] = cons[cidx].L[loff] + dlambda


    # Reference: C:\Program Files\Side Effects Software\Houdini 21.0.440\houdini\ocl\sim\pbd_constraints.cl:L1093 tetFiberUpdateXPBD
    @ti.func
    def tet_fiber_update_xpbd(
        self,
        use_jacobi: ti.template(),
        pos: ti.template(),
        pprev: ti.template(),
        dP: ti.template(),
        dPw: ti.template(),
        cidx: ti.i32,
        cons: ti.template(),
        pts: ti.types.vector(4, ti.i32),
        dt: ti.f32,
        fiber: ti.types.vector(3, ti.f32),
        stiffness: ti.f32,
        Dminv: ti.types.matrix(3, 3, ti.f32),
        kdampratio: ti.f32,
        restlength: ti.f32,
        restvector: ti.types.vector(4, ti.f32),
        acti: ti.f32,
        mass: ti.template(),
        stopped: ti.template(),
        target_stretch: ti.f32 = 1.0,
    ):
        isLINEARENERGY = True
        isNORMSTIFFNESS = True
        use_anisotropic_arap = False

        inv_masses = ti.Vector([0.0, 0.0, 0.0, 0.0])
        for i in ti.static(range(4)):
            inv_masses[i] = get_inv_mass(pts[i], mass, stopped)
        l = cons[cidx].L[0]
        alpha = 1.0 / stiffness
        if isNORMSTIFFNESS:
            alpha /= restlength
        alpha /= dt * dt
        grad_scale = 1.0
        psi = 0.0
        Ht = ti.Matrix([[0] * 3 for _ in range(3)], ti.f32)

        _Ds = ti.Matrix.cols([pos[pts[i]] - pos[pts[3]] for i in range(3)])
        F = _Ds @ Dminv

        # if (restmatrix && squaredNorm3(Dminv) > 0)
        if use_anisotropic_arap:
            fiberscale = 1.0
            w = fiber
            
            # Applying user correction: mat3Tvecmul(a, b) is b * a^T, which in Taichi is a.transpose() @ b
            Dminv_w = Dminv.transpose() @ w
            FwT = F.transpose() @ w
            
            I5 = FwT.norm_sqr()
            I4 = invariant4(F, fiber)
            SI4 = 1.0 if I4 > 0.0 else -1.0
            
            # Fiber energy
            sqrt_I5 = ti.sqrt(I5)
            diff = sqrt_I5 - SI4 * fiberscale
            psi = 0.5 * diff * diff
            
            # Gradient scale
            sqrt_I5_inv = 1.0 / sqrt_I5 if sqrt_I5 > 1e-9 else 0.0
            dPsidI5 = (1.0 - SI4 * fiberscale * sqrt_I5_inv)
            
            # Singularity handling for gradient scale
            if ti.abs(I4) > 1e-4:
                grad_scale = dPsidI5
            else:
                grad_scale = -(1.0 - fiberscale / 2.0)

            # Piola-Kirchhoff stress tensor derivative Ht
            Ht = ti.Matrix.outer_product(Dminv_w, FwT)
        else:
            # C = psi = 0.5 * ||Fw||^2, 
            # where F is deformation gradient, w is fiber direction in rest pose.
            # gradC = F w w^T Dm^-T
            # 
            # Proof: 
            # Since we set the energy density as C, this derevation is the same with FEM. 
            # See pp.22 in Sifakis.
            # 𝜕C/𝜕𝑥=𝜕Ψ/𝜕𝑥=𝜕Ψ/𝜕F:𝜕F/𝜕𝑥=P D_m^(−T), where P is the PK1. 
            # So we need to find P=𝜕Ψ/𝜕F
            # ||F𝐰||^2=𝑡𝑟(𝐹𝑤 (𝐹𝑤)^𝑇 )=𝑡𝑟((𝐹𝑤)^𝑇  (𝐹𝑤))=𝑡𝑟(𝑤^𝑇 𝐹^𝑇 𝐹 𝑤)=𝑡𝑟(𝐹^𝑇 𝐹 𝑤 𝑤^𝑇)
            # Identity: (𝜕𝑡𝑟(𝐹𝐹^𝑇 𝐴))/𝜕𝐹=𝐹(𝐴+𝐴^𝑇 ), here A = w w^T
            # So, (𝜕𝑡𝑟(𝐹𝐹^𝑇 𝑤𝑤^𝑇))/𝜕𝐹=2𝐹𝑤𝑤^𝑇
            # P = F w w^T
            # gradC = P Dm^-T = F w w^T Dm^-T
            # volume is in alpha, so we don't need to multiply volume here.

            # Pre-multiplied w^T * Dm^-T stored in restvector.
            wTDminvT = restvector.xyz
            # // F = Ds * Dm^-1
            # // FwT = (F w)^T = (w^T * Dm^-T) * Ds^T
            FwT = wTDminvT @ _Ds.transpose()
            psi = 0.5 * FwT.norm_sqr()


            # if use the linear energy, C = psi =  ||Fw||, 
            # then gradC = F w w^T Dm^-T / ||Fw||
            if (isLINEARENERGY or isNORMSTIFFNESS):
                if psi > 1e-9:
                    psi_sqrt = ti.sqrt(2.0 * psi)
                    grad_scale = 1.0 / psi_sqrt
                    psi = psi_sqrt

            # Piola-Kirchhoff stress tensor derivative Ht (H^T)
            Ht = ti.Matrix.outer_product(wTDminvT, FwT)


        grad0 = grad_scale * ti.Vector([Ht[0,0], Ht[0,1], Ht[0,2]])
        grad1 = grad_scale * ti.Vector([Ht[1,0], Ht[1,1], Ht[1,2]])
        grad2 = grad_scale * ti.Vector([Ht[2,0], Ht[2,1], Ht[2,2]])
        grad3 = -grad0 - grad1 - grad2

        w_sum = (inv_masses[0] * grad0.norm_sqr() +
                 inv_masses[1] * grad1.norm_sqr() +
                 inv_masses[2] * grad2.norm_sqr() +
                 inv_masses[3] * grad3.norm_sqr())

        if w_sum > 1e-9:
            dsum = 0.0
            gamma = 1.0
            if kdampratio > 0:
                beta = stiffness * kdampratio * dt * dt
                if (isNORMSTIFFNESS):
                    beta *= restlength
                gamma = alpha * beta / dt

                dsum = (grad0.dot(pos[pts[0]] - pprev[pts[0]]) +
                        grad1.dot(pos[pts[1]] - pprev[pts[1]]) +
                        grad2.dot(pos[pts[2]] - pprev[pts[2]]) +
                        grad3.dot(pos[pts[3]] - pprev[pts[3]]))
                dsum *= gamma
                gamma += 1.0

            C = psi - target_stretch
            dL = (-C - alpha * l - dsum) / (gamma * w_sum + alpha)
            if use_jacobi:
                updatedP(dP, dPw, dL * inv_masses[0] * grad0, pts[0])
                updatedP(dP, dPw, dL * inv_masses[1] * grad1, pts[1])
                updatedP(dP, dPw, dL * inv_masses[2] * grad2, pts[2])
                updatedP(dP, dPw, dL * inv_masses[3] * grad3, pts[3])
            else:
                pos[pts[0]] +=  dL * inv_masses[0] * grad0
                pos[pts[1]] +=  dL * inv_masses[1] * grad1
                pos[pts[2]] +=  dL * inv_masses[2] * grad2
                pos[pts[3]] +=  dL * inv_masses[3] * grad3

                cons[cidx].L[0] += dL

    @ti.func
    def attach_bilateral_update(
        self,
        use_jacobi: ti.template(),
        cidx: ti.i32,
        cons: ti.template(),
        pt1: ti.i32,
        p0: ti.types.vector(3, ti.f32),
        p1: ti.types.vector(3, ti.f32),
        pos: ti.template(),
        pprev: ti.template(),
        dP: ti.template(),
        dPw: ti.template(),
        mass: ti.template(),
        stopped: ti.template(),
        restlength: ti.f32,
        kstiff: ti.f32,
        kdampratio: ti.f32,
        kstiffcompress: ti.f32,
    ):
        """Bilateral attach: same position correction as distance_pos_update_xpbd (pt0=-1)
        plus accumulates constraint violation C×n into reaction_accum for bone coupling."""
        invmass1 = get_inv_mass(pt1, mass, stopped)
        wsum = invmass1

        if wsum != 0.0:
            p1_current = pos[pt1]
            p0_current = p0

            n = p1_current - p0_current
            d = n.norm()
            if d >= 1e-6:
                loff = self.inCompressBand(d, restlength)
                kstiff_val = kstiffcompress if loff else kstiff
                if kstiff_val != 0.0:
                    l = cons[cidx].L[loff]

                    alpha = 1.0 / kstiff_val
                    alpha /= self.dt * self.dt

                    C = d - restlength
                    n = n / d

                    dsum = 0.0
                    gamma = 1.0
                    if kdampratio > 0.0:
                        prev1 = pprev[pt1]
                        beta = kstiff_val * kdampratio * self.dt * self.dt
                        gamma = alpha * beta / self.dt
                        dsum = gamma * n.dot(p1_current - prev1)
                        gamma += 1.0

                    dL = (-C - alpha * l - dsum) / (gamma * wsum + alpha)
                    dp = n * (-dL)

                    # Muscle side: same position correction as original
                    if use_jacobi:
                        updatedP(dP, dPw, -invmass1 * dp, pt1)
                    else:
                        pos[pt1] -= invmass1 * dp
                        cons[cidx].L[loff] = cons[cidx].L[loff] + dL

                    # Bilateral reaction: accumulate C×n (mass-independent)
                    # C×n = constraint violation vector, represents the "virtual
                    # displacement" the bone side would receive (Newton's 3rd law)
                    self.reaction_accum[cidx] += C * n

    # FIXME: the pt0 is target and pt1 is source, which is opposite to intuitive
    @ti.func
    def distance_pos_update_xpbd(
        self,
        use_jacobi: ti.template(),
        cidx: ti.i32,
        cons: ti.template(),
        pt0: ti.i32,
        pt1: ti.i32,
        p0: ti.types.vector(3, ti.f32),
        p1: ti.types.vector(3, ti.f32),
        pos: ti.template(),
        pprev: ti.template(),
        dP: ti.template(),
        dPw: ti.template(),
        mass: ti.template(),
        stopped: ti.template(),
        restlength: ti.f32,
        kstiff: ti.f32,
        kdampratio: ti.f32,
        kstiffcompress: ti.f32,
    ):
        """
        更新两点之间的distance约束。
        pt0 可以是固定的（invmass0=0）或动态的。
        如果 pt0 < 0，表示约束是以固定线上的点为target。
        注意：p0 和 p1 是传入的位置值，可能是 pos[pt0]/pos[pt1]，也可能是投影点等计算出的临时值
        """
        invmass0 = 0.0
        invmass1 = get_inv_mass(pt1, mass, stopped)
        
        if pt0 >= 0:
            invmass0 = get_inv_mass(pt0, mass, stopped)
        
        wsum = invmass0 + invmass1
        
        if wsum != 0.0:
            # 重新从 pos 读取 p1，确保使用最新值
            p1_current = pos[pt1]
            p0_current = p0
            if pt0 >= 0:
                p0_current = pos[pt0]
            
            n = p1_current - p0_current
            d = n.norm()
            if d >= 1e-6:
                loff = self.inCompressBand(d, restlength)
                kstiff_val = kstiffcompress if loff else kstiff
                if kstiff_val != 0.0:
                    l = cons[cidx].L[loff]

                    alpha = 1.0 / kstiff_val
                    alpha /= self.dt * self.dt

                    C = d - restlength
                    n = n / d
                    gradC = n

                    dsum = 0.0
                    gamma = 1.0
                    if kdampratio > 0.0:
                        if pt0 >= 0:
                            prev0 = pprev[pt0]
                            prev1 = pprev[pt1]
                            beta = kstiff_val * kdampratio * self.dt * self.dt
                            gamma = alpha * beta / self.dt
                            dsum = gamma * (-gradC.dot(p0_current - prev0) + gradC.dot(p1_current - prev1))
                        else:
                            prev1 = pprev[pt1]
                            beta = kstiff_val * kdampratio * self.dt * self.dt
                            gamma = alpha * beta / self.dt
                            dsum = gamma * gradC.dot(p1_current - prev1)
                        gamma += 1.0

                    dL = (-C - alpha * l - dsum) / (gamma * wsum + alpha)
                    dp = n * (-dL)

                    if use_jacobi:
                        if pt0 >= 0:
                            updatedP(dP, dPw, invmass0 * dp, pt0)
                        updatedP(dP, dPw, -invmass1 * dp, pt1)
                    else:
                        if pt0 >= 0:
                            pos[pt0] += invmass0 * dp
                        pos[pt1] -= invmass1 * dp
                        cons[cidx].L[loff] = cons[cidx].L[loff] + dL

    @ti.func
    def distance_update_xpbd(
        self,
        use_jacobi: ti.template(),
        cidx: ti.i32,
        cons: ti.template(),
        pts: ti.types.vector(4, ti.i32),
        pos: ti.template(),
        pprev: ti.template(),
        dP: ti.template(),
        dPw: ti.template(),
        mass: ti.template(),
        stopped: ti.template(),
        restlength: ti.f32,
        kstiff: ti.f32,
        kdampratio: ti.f32,
        kstiffcompress: ti.f32,
    ):
        pt0 = pts[0]
        pt1 = pts[1]
        p0 = pos[pt0]
        p1 = pos[pt1]
        invmass0 = get_inv_mass(pt0, mass, stopped)
        invmass1 = get_inv_mass(pt1, mass, stopped)
        wsum = invmass0 + invmass1
        if wsum != 0.0:
            n = p1 - p0
            d = n.norm()
            if d >= 1e-6:
                loff = self.inCompressBand(d, restlength)
                kstiff_val = kstiffcompress if loff else kstiff
                if kstiff_val != 0.0:
                    l = cons[cidx].L[loff]

                    alpha = 1.0 / kstiff_val
                    alpha /= self.dt * self.dt

                    C = d - restlength
                    n = n / d
                    gradC = n

                    dsum = 0.0
                    gamma = 1.0
                    if kdampratio > 0.0:
                        prev0 = pprev[pt0]
                        prev1 = pprev[pt1]
                        beta = kstiff_val * kdampratio * self.dt * self.dt
                        gamma = alpha * beta / self.dt
                        dsum = gamma * (-gradC.dot(p0 - prev0) + gradC.dot(p1 - prev1))
                        gamma += 1.0

                    dL = (-C - alpha * l - dsum) / (gamma * wsum + alpha)
                    dp = n * (-dL)

                    if use_jacobi:
                        updatedP(dP, dPw, invmass0 * dp, pt0)
                        updatedP(dP, dPw, -invmass1 * dp, pt1)
                    else:
                        pos[pt0] += invmass0 * dp
                        pos[pt1] -= invmass1 * dp
                        cons[cidx].L[loff] = cons[cidx].L[loff] + dL

    @ti.func
    def tri_arap_update_xpbd(
        self,
        use_jacobi: ti.template(),
        cidx: ti.i32,
        cons: ti.template(),
        pts: ti.types.vector(4, ti.i32),
        dt: ti.f32,
        pos: ti.template(),
        pprev: ti.template(),
        dP: ti.template(),
        dPw: ti.template(),
        mass: ti.template(),
        stopped: ti.template(),
        restlength: ti.f32,
        restvector: ti.types.vector(4, ti.f32),
        kstiff: ti.f32,
        kdampratio: ti.f32,
        kstiffcompress: ti.f32,
        flags: ti.i32,
    ):
        pt0 = pts[0]
        pt1 = pts[1]
        pt2 = pts[2]
        p0 = pos[pt0]
        p1 = pos[pt1]
        p2 = pos[pt2]

        invmass0 = get_inv_mass(pt0, mass, stopped)
        invmass1 = get_inv_mass(pt1, mass, stopped)
        invmass2 = get_inv_mass(pt2, mass, stopped)

        Dminv = ti.Matrix([[restvector[0], restvector[1]],
                           [restvector[2], restvector[3]]], ti.f32)

        xform, area = triangle_xform_and_area(p0, p1, p2)
        if area != 0.0:
            loff = self.inCompressBand(area, restlength)
            kstiff_val = kstiffcompress if loff else kstiff
            if kstiff_val != 0.0:
                l = cons[cidx].L[loff]

                row0 = ti.Vector([xform[0, 0], xform[0, 1], xform[0, 2]])
                row1 = ti.Vector([xform[1, 0], xform[1, 1], xform[1, 2]])
                P0 = ti.Vector([row0.dot(p0), row1.dot(p0)])
                P1 = ti.Vector([row0.dot(p1), row1.dot(p1)])
                P2 = ti.Vector([row0.dot(p2), row1.dot(p2)])

                Ds = ti.Matrix.cols([P0 - P2, P1 - P2])
                F = Ds @ Dminv

                m = ti.Vector([F[0, 0] + F[1, 1], F[1, 0] - F[0, 1]])
                mlen = m.norm()
                if mlen >= 1e-9:
                    m = m / mlen
                    R = ti.Matrix([[m[0], -m[1]],
                                   [m[1],  m[0]]], ti.f32)

                    d = F - R
                    psi = squared_norm2(d)
                    gradscale = 2.0
                    if (flags & (LINEARENERGY | NORMSTIFFNESS)) != 0:
                        psi = ti.sqrt(psi)
                        gradscale = 1.0 / psi
                    
                    if psi >= 1e-6:
                        Ht = Dminv @ d.transpose()
                        grad0 = ti.Vector([Ht[0, 0], Ht[0, 1], 0.0]) * gradscale
                        grad1 = ti.Vector([Ht[1, 0], Ht[1, 1], 0.0]) * gradscale
                        grad2 = -grad0 - grad1

                        xformT = xform.transpose()
                        grad0 = xformT @ grad0
                        grad1 = xformT @ grad1
                        grad2 = xformT @ grad2

                        wsum = (invmass0 * grad0.dot(grad0) +
                                invmass1 * grad1.dot(grad1) +
                                invmass2 * grad2.dot(grad2))
                        
                        if wsum != 0.0:
                            alpha = 1.0 / kstiff_val
                            if (flags & NORMSTIFFNESS) != 0:
                                alpha /= restlength
                            alpha /= dt * dt

                            dsum = 0.0
                            gamma = 1.0
                            if kdampratio > 0.0:
                                prev0 = pprev[pt0]
                                prev1 = pprev[pt1]
                                prev2 = pprev[pt2]
                                beta = kstiff_val * kdampratio * dt * dt
                                if (flags & NORMSTIFFNESS) != 0:
                                    beta *= restlength
                                gamma = alpha * beta / dt
                                dsum = (grad0.dot(p0 - prev0) +
                                        grad1.dot(p1 - prev1) +
                                        grad2.dot(p2 - prev2))
                                dsum *= gamma
                                gamma += 1.0

                            C = psi
                            dL = (-C - alpha * l - dsum) / (gamma * wsum + alpha)
                            if use_jacobi:
                                updatedP(dP, dPw, dL * invmass0 * grad0, pt0)
                                updatedP(dP, dPw, dL * invmass1 * grad1, pt1)
                                updatedP(dP, dPw, dL * invmass2 * grad2, pt2)
                            else:
                                pos[pt0] += dL * invmass0 * grad0
                                pos[pt1] += dL * invmass1 * grad1
                                pos[pt2] += dL * invmass2 * grad2
                                cons[cidx].L[loff] = cons[cidx].L[loff] + dL

    @ti.func
    def tet_arap_update_xpbd(
        self,
        use_jacobi: ti.template(),
        cidx: ti.i32,
        cons: ti.template(),
        pts: ti.types.vector(4, ti.i32),
        dt: ti.f32,
        pos: ti.template(),
        pprev: ti.template(),
        dP: ti.template(),
        dPw: ti.template(),
        mass: ti.template(),
        stopped: ti.template(),
        restlength: ti.f32,
        restvector: ti.types.vector(4, ti.f32),
        restmatrix: ti.types.matrix(3, 3, ti.f32),
        kstiff: ti.f32,
        kdampratio: ti.f32,
        flags: ti.i32,
    ):
        pt0 = pts[0]
        pt1 = pts[1]
        pt2 = pts[2]
        pt3 = pts[3]
        p0 = pos[pt0]
        p1 = pos[pt1]
        p2 = pos[pt2]
        p3 = pos[pt3]

        invmass0 = get_inv_mass(pt0, mass, stopped)
        invmass1 = get_inv_mass(pt1, mass, stopped)
        invmass2 = get_inv_mass(pt2, mass, stopped)
        invmass3 = get_inv_mass(pt3, mass, stopped)

        Ds = ti.Matrix.cols([p0 - p3, p1 - p3, p2 - p3])
        F = Ds @ restmatrix

        _, R = polar_decomposition(F)
        d = F - R
        cons[cidx].restvector = mat3_to_quat(R)

        psi = squared_norm3(d)
        gradscale = 2.0
        if (flags & (LINEARENERGY | NORMSTIFFNESS)) != 0:
            psi = ti.sqrt(psi)
            gradscale = 1.0 / psi
        
        if psi >= 1e-6:
            Ht = restmatrix @ d.transpose()
            grad0 = gradscale * ti.Vector([Ht[0, 0], Ht[0, 1], Ht[0, 2]])
            grad1 = gradscale * ti.Vector([Ht[1, 0], Ht[1, 1], Ht[1, 2]])
            grad2 = gradscale * ti.Vector([Ht[2, 0], Ht[2, 1], Ht[2, 2]])
            grad3 = -grad0 - grad1 - grad2

            wsum = (invmass0 * grad0.dot(grad0) +
                    invmass1 * grad1.dot(grad1) +
                    invmass2 * grad2.dot(grad2) +
                    invmass3 * grad3.dot(grad3))
            
            if wsum != 0.0:
                alpha = 1.0 / kstiff
                if (flags & NORMSTIFFNESS) != 0:
                    alpha /= restlength
                alpha /= dt * dt

                dsum = 0.0
                gamma = 1.0
                if kdampratio > 0.0:
                    prev0 = pprev[pt0]
                    prev1 = pprev[pt1]
                    prev2 = pprev[pt2]
                    prev3 = pprev[pt3]
                    beta = kstiff * kdampratio * dt * dt
                    if (flags & NORMSTIFFNESS) != 0:
                        beta *= restlength
                    gamma = alpha * beta / dt
                    dsum = (grad0.dot(p0 - prev0) +
                            grad1.dot(p1 - prev1) +
                            grad2.dot(p2 - prev2) +
                            grad3.dot(p3 - prev3))
                    dsum *= gamma
                    gamma += 1.0

                C = psi
                dL = (-C - alpha * cons[cidx].L[0] - dsum) / (gamma * wsum + alpha)
                if use_jacobi:
                    updatedP(dP, dPw, dL * invmass0 * grad0, pt0)
                    updatedP(dP, dPw, dL * invmass1 * grad1, pt1)
                    updatedP(dP, dPw, dL * invmass2 * grad2, pt2)
                    updatedP(dP, dPw, dL * invmass3 * grad3, pt3)
                else:
                    pos[pt0] += dL * invmass0 * grad0
                    pos[pt1] += dL * invmass1 * grad1
                    pos[pt2] += dL * invmass2 * grad2
                    pos[pt3] += dL * invmass3 * grad3
                    cons[cidx].L[0] += dL

    def step(self):
        self.update_attach_targets()
        # self.dt = self.cfg.dt / self.cfg.num_substeps
        for _ in range(self.cfg.num_substeps):
            self.integrate()
            self.clear()
            self.solve_constraints()
            if self.use_jacobi:
                self.apply_dP()
            self.update_velocities()

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
        while self.step_cnt <= self.cfg.nsteps:
            self.vis._render_control()
            if not self.cfg.pause:
                self.step_start_time = time.perf_counter()
                self.step()
                self.step_cnt += 1
                self.step_end_time = time.perf_counter()
            if self.cfg.gui or self.cfg.save_image:
                self.vis._render_frame(self.step_cnt, self.cfg.save_image)
        while self.cfg.gui and self.window.running:
            self.vis._render_frame(self.step_cnt, self.cfg.save_image)
    
@ti.data_oriented
class Visualizer:
    def __init__(self,cfg: SimConfig, muscle=None):
        self._init_render(cfg, muscle)

    def _init_render(self, cfg: SimConfig, muscle=None):
        self.cfg = cfg
        self.muscle = muscle
        if self.cfg.render_mode is None or not self.cfg.gui:
            self.cfg.gui = False
            return
        
        self.res = (1080, 720)
        self.window = ti.ui.Window(self.cfg.name, self.res, vsync=True, fps_limit=self.cfg.render_fps)
        self.canvas = self.window.get_canvas()
        self.scene = self.window.get_scene()
        self.camera = ti.ui.Camera()
        self.camera.position(0.5, 1.0, 1.95)
        self.camera.lookat(0.5, 0.3, 0.5)
        self.camera.fov(45)
        self.gui = self.window.get_gui()

        if self.cfg.show_auxiliary_meshes and (hasattr(self.cfg, 'ground_mesh_path') and hasattr(self.cfg, 'coord_mesh_path')):
            self.ground, self.coord, self.ground_indices, self.coord_indices = read_auxiliary_meshes(self.cfg.ground_mesh_path, self.cfg.coord_mesh_path)
        
        if self.muscle is not None:
            bbox = get_bbox(self.muscle.pos0_np)
            self._focus_camera_on_model(bbox) # focus camera on model at start
        
        # 初始化 attach 可视化相关属性
        self.num_attach_lines = 0
        self.attach_lines_vertices = None


    def _focus_camera_on_model(self, bbox):
        center = (bbox[0] + bbox[1]) * 0.5
        self.camera.position(center[0], center[1], center[2] + 0.5)
        self.camera.lookat(center[0], center[1], center[2])


    def update(self):
        self._render_frame()
    

    def _render_frame(self, step: int, save_image: bool = False):
        if self.cfg.render_mode == None:
            return
        
        if self.muscle is None:
            return
        
        self.camera.track_user_inputs(self.window, movement_speed=0.01, hold_key=ti.ui.RMB)
        self.scene.set_camera(self.camera)
        
        # Render muscle mesh
        if hasattr(self, 'muscle_colors_field') and self.muscle_colors_field is not None:
            # 使用 per_vertex_color 着色
            self.scene.mesh(vertices=self.muscle.pos, indices=self.muscle.surface_tris_field, 
                           per_vertex_color=self.muscle_colors_field, two_sided=True, show_wireframe=self.cfg.show_wireframe)
        else:
            # 默认蓝色
            self.scene.mesh(vertices=self.muscle.pos, indices=self.muscle.surface_tris_field, 
                           color=(0.2, 0.6, 1.0), two_sided=True, show_wireframe=self.cfg.show_wireframe)
        
        # Render bone mesh
        if self.muscle.bone_indices_field is not None:
            if hasattr(self, 'bone_colors_field') and self.bone_colors_field is not None:
                # 使用 per_vertex_color 按 muscle_id 着色
                self.scene.mesh(vertices=self.muscle.bone_pos_field, indices=self.muscle.bone_indices_field, 
                               per_vertex_color=self.bone_colors_field, two_sided=True, show_wireframe=self.cfg.show_wireframe)
            else:
                # 默认浅蓝色
                self.scene.mesh(vertices=self.muscle.bone_pos_field, indices=self.muscle.bone_indices_field, 
                               color=(0.1, 0.4, 0.8), two_sided=True, show_wireframe=self.cfg.show_wireframe)
        elif hasattr(self.muscle, 'bone_pos_field') and self.muscle.bone_pos_field is not None:
            # If no indices, just show particles? Or maybe the user prefers a mesh.
            self.scene.particles(self.muscle.bone_pos_field, radius=0.005, color=(0.1, 0.4, 0.8))

        # Render attach lines
        if self.num_attach_lines > 0:
            self.muscle._update_attach_targets_kernel()
            self._update_attach_vis(self.muscle.cons, self.muscle.pos, self.attach_cidx)
            self.scene.lines(vertices=self.attach_lines_vertices, width=1, color=(1.0, 0.0, 0.0))

        if self.cfg.show_auxiliary_meshes:
            self.scene.mesh(self.ground, indices=self.ground_indices, color=(0.5, 0.5, 0.5),)
            self.scene.mesh(self.coord, indices=self.coord_indices, color=(0.5, 0, 0), )

        with self.gui.sub_window("Options", 0, 0, 0.25, 0.3) as w:
            self.gui.text(f"Step: {step}")
            if self.muscle is not None and self.muscle.step_cnt > 1:
                self.gui.text(f"FPS: {self.muscle.get_fps():.1f}")
                self.gui.text(f"Volume error: {self.muscle.calc_vol_error() * 100:.2f} %")
            if w.button("Pause" if not self.cfg.pause else "Resume"):
                self.cfg.pause = not self.cfg.pause
            if w.button("Toggle Wireframe"):
                self.cfg.show_wireframe = not self.cfg.show_wireframe
            if w.button("Reset Simulation"):
                if self.muscle is not None:
                    self.muscle.reset()
            if w.button("save camera settings"):
                pass
            if w.button("load camera settings"):
                pass
            if w.button("focus camera on model"):
                if self.muscle is not None:
                    bbox = get_bbox(self.muscle.pos0_np)
                    center = (bbox[0] + bbox[1]) * 0.5
                    self.camera.position(center[0], center[1], center[2] + 0.5)
                    self.camera.lookat(center[0], center[1], center[2])
            self.cfg.activation = w.slider_float("activation", self.cfg.activation, 0.0, 1.0)
            if self.muscle is not None:
                self.muscle.activation.fill(self.cfg.activation)
            if hasattr(self, 'extra_text') and self.extra_text:
                self.gui.text(self.extra_text)
            

        self.scene.ambient_light((0.4, 0.4, 0.4))
        self.scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.6, 0.6, 0.6))
        self.scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.6, 0.6, 0.6))
        
        self.canvas.scene(self.scene)
        if self.cfg.render_mode == "human":
            self.window.show()
        if save_image:
            self.window.save_image(f"output/{step:04d}.png")
        if self.cfg.render_mode == "rgb_array":
            self.rgb_array = self.window.get_image_buffer_as_numpy()
            return self.rgb_array

    def _render_muscles(self):
        pass

    def _render_bones(self):
        pass

    def _render_control(self):
        if not self.cfg.gui or self.cfg.render_mode != "human":
            return
        for e in self.window.get_events(ti.ui.PRESS):
            if e.key in [ti.ui.ESCAPE]:
                exit()
            elif e.key == 'r' or e.key == 'R':
                if self.muscle is not None:
                    self.muscle.reset()
                print("Simulation reset.")
            elif e.key == 'f' or e.key == 'F':
                self.cfg.show_wireframe = not self.cfg.show_wireframe
                print("show_wireframe:", self.cfg.show_wireframe)
            elif e.key == ti.ui.SPACE:
                self.cfg.pause = not self.cfg.pause
                if self.cfg.pause:
                    print("Simulation paused.")
                else:
                    print("Simulation resumed.")
            elif e.key == 'g' or e.key == 'G':
                if self.muscle is not None:
                    bbox = get_bbox(self.muscle.pos0_np)
                    center = (bbox[0] + bbox[1]) * 0.5
                    self.camera.position(center[0], center[1], center[2] + 0.5)
                    self.camera.lookat(center[0], center[1], center[2])
            else:
                pass


    @ti.kernel
    def _update_attach_vis(self, cons: ti.template(), pos: ti.template(), attach_cidx: ti.template()):
        for i in range(self.num_attach_lines):
            cidx = attach_cidx[i]  # Get the actual global constraint index
            pts = cons[cidx].pts
            src_pt = pts[0]
            self.attach_lines_vertices[2 * i] = pos[src_pt]
            self.attach_lines_vertices[2 * i + 1] = cons[cidx].restvector.xyz


    def _init_attach_vis(self, attach_constraints):
        """
        为 attach / attachnormal 约束准备可视化数据。
        """
        if not attach_constraints:
            return
        self.num_attach_lines = len(attach_constraints)
        if self.num_attach_lines > 0:
            self.attach_lines_vertices = ti.Vector.field(3, dtype=ti.f32, shape=self.num_attach_lines * 2)
            # Store the cidx values for each attach constraint， this will convert local attach index to global constraint index
            self.attach_cidx = ti.field(dtype=ti.i32, shape=self.num_attach_lines)
            for i, c in enumerate(attach_constraints):
                self.attach_cidx[i] = c['cidx']

    
    def _generate_muscle_colors(self):
        """根据 cfg.color_muscles 生成肌肉顶点颜色"""
        self.muscle_vertex_colors = None
        self.muscle_colors_field = None
        
        if self.cfg.color_muscles is None or self.muscle is None:
            return
        
        if self.cfg.color_muscles == "muscle_id":
            # 按 muscle_id 着色
            if hasattr(self.muscle, 'geo') and hasattr(self.muscle.geo, 'pointattr') and 'muscle_id' in self.muscle.geo.pointattr:
                muscle_ids = self.muscle.geo.pointattr['muscle_id']
                unique_ids = sorted(set(muscle_ids))
                id_colors = generate_muscle_id_colors(unique_ids)
                
                self.muscle_vertex_colors = np.zeros((self.muscle.n_verts, 3), dtype=np.float32)
                for v_idx, mid in enumerate(muscle_ids):
                    self.muscle_vertex_colors[v_idx] = id_colors[mid]
                
                print(f"Muscle coloring by muscle_id enabled ({len(unique_ids)} groups)")
            else:
                print("Warning: muscle_id not found in geometry, cannot color by muscle_id")
        
        elif self.cfg.color_muscles == "tendonmask":
            # 按 tendonmask 着色（白色=肌腱，红色=肌肉腹部）
            if hasattr(self.muscle, 'v_tendonmask_np') and self.muscle.v_tendonmask_np is not None:
                self.muscle_vertex_colors = np.zeros((self.muscle.n_verts, 3), dtype=np.float32)
                for v_idx in range(self.muscle.n_verts):
                    mask_value = self.muscle.v_tendonmask_np[v_idx]
                    # mask_value: 0=肌肉, 1=肌腱
                    # 插值颜色：红色(0.9,0.1,0.1) -> 白色(1.0,1.0,1.0)
                    r = 0.9 + mask_value * 0.1
                    g = 0.1 + mask_value * 0.9
                    b = 0.1 + mask_value * 0.9
                    self.muscle_vertex_colors[v_idx] = [r, g, b]
                
                print("Muscle coloring by tendonmask enabled")
            else:
                print("Warning: tendonmask not found in geometry, cannot color by tendonmask")
        
        # 创建 Taichi field
        if self.muscle_vertex_colors is not None:
            self.muscle_colors_field = ti.Vector.field(3, dtype=ti.f32, shape=self.muscle.n_verts)
            self.muscle_colors_field.from_numpy(self.muscle_vertex_colors)


def generate_muscle_id_colors(muscle_ids):
    """为每个 muscle_id 生成独特的颜色"""
    import colorsys
    colors = {}
    n = len(muscle_ids)
    for i, mid in enumerate(muscle_ids):
        # 使用 HSV 色彩空间生成均匀分布的颜色
        hue = i / max(n, 1)
        saturation = 0.7 + (i % 3) * 0.1  # 0.7-0.9 之间
        value = 0.8 + (i % 2) * 0.15  # 0.8-0.95 之间
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
    sim = MuscleSim(cfg)
    print("Running for", cfg.nsteps, "steps.")
    print("Simulation starting (press SPACE to run)...")
    sim.run()


if __name__ == "__main__":
    main()
