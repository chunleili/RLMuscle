"""
Constraint building for MuscleSim (pure Python / numpy).

Contains constraint type constants, aliases, surface triangle extraction,
and the ConstraintBuilderMixin that provides all create_* methods.
Backend-specific build_constraints lives in muscle.py / muscle_warp.py.
"""
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


# Constraint type constants (from pbd_types.h)
PIN           =  157323
ATTACH        =  1650556
TETVOLUME     =  -215389979
TETFIBERNORM  =  -303462111
DISTANCELINE  =  1621136047
TETARAP       =  -92199131
TETARAPNORM   =  -885573303

# ARAP flags (from pbd_types.h)
LINEARENERGY = 1 << 0
NORMSTIFFNESS = 1 << 1


def constraint_alias(name: str) -> str:
    name = name.lower()
    if name == "attachnormal":
        return "distanceline"
    return name


class ConstraintBuilderMixin:
    """Mixin providing constraint creation methods (backend-agnostic).

    Requires self to have: pos0_np, tet_np, v_fiber_np, geo, cfg,
    load_bone_geo(), constraint_configs.
    """

    # -- rest matrix helpers (pure numpy) --

    def _batch_compute_tet_rest_matrices(self):
        """Batch compute rest matrices for all tets.
        Returns cached (restmatrices, volumes, valid) tuple."""
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
        M = scale * np.stack([p[pt0] - p[pt3], p[pt1] - p[pt3], p[pt2] - p[pt3]]).T
        detM = np.linalg.det(M)
        if detM == 0:
            return None, 0.0
        return np.linalg.inv(M), detM / 6.0

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

    def map_pts2tets(self, tet):
        pt2tet = {}
        for i, tet_verts in enumerate(tet):
            for v in tet_verts:
                if v not in pt2tet:
                    pt2tet[v] = []
                pt2tet[v].append(i)
        return pt2tet

    # -- constraint creators (return list[dict] with plain lists, no ti/wp) --

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

    def create_tet_fiber_constraint(self, params):
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
            c = dict(
                type=TETFIBERNORM,
                pts=[int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3])],
                stiffness=stiffness,
                dampingratio=dampingratio,
                tetid=i,
                L=[0.0, 0.0, 0.0],
                restlength=float(vol),
                restvector=[float(mw_t[0]), float(mw_t[1]), float(mw_t[2]), 1.0],
                restdir=[0.0, 0.0, 0.0],
                compressionstiffness=-1.0
            )
            constraints.append(c)
        return constraints

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
            tetid = self.pt2tet.get(src_idx, [-1])[0]
            restlength = float(dists[j])
            c = dict(
                type=ATTACH,
                pts=[int(src_idx), -1, int(tgt_idx), -1],
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
                pts=[int(src_idx), -1, int(tgt_idx), -1],
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

    def _collect_raw_constraints(self):
        """Dispatch constraint_configs -> create_* methods, return all_constraints list."""
        self.attach_constraints = []
        self.distanceline_constraints = []

        all_constraints = []
        _t_total = time.perf_counter()
        for params in self.constraint_configs:
            ctype = constraint_alias(params['type'])
            new_constraints = []
            if ctype == 'volume':
                new_constraints = self.create_tet_volume_constraint(params)
            elif ctype == 'fiber':
                new_constraints = self.create_tet_fiber_constraint(params)
            elif ctype == 'attach':
                new_constraints = self.create_attach_constraints(params)
                self.attach_constraints.extend(new_constraints)
            elif ctype == 'distanceline':
                new_constraints = self.create_distance_line_constraints(params)
                self.distanceline_constraints.extend(new_constraints)
            elif ctype == 'tetarap':
                new_constraints = self.create_tet_arap_constraints(params)

            if new_constraints:
                all_constraints.extend(new_constraints)
                print(f"  {params.get('name', ctype)} ({ctype}): {len(new_constraints)} constraints")

        self.raw_constraints = all_constraints.copy()
        _dt_total = time.perf_counter() - _t_total
        return all_constraints, _dt_total
