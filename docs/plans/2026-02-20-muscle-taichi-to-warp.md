# Muscle Taichi-to-Warp Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite all Taichi kernels in `muscle_warp.py` to NVIDIA Warp, keeping the same public API so `solver_muscle_bone_coupled.py` and `example_couple.py` work without changes. Validate by comparing numerical output of `muscle.py` (Taichi) vs `muscle_warp.py` (Warp).

**Architecture:** The `MuscleSim` class keeps its same public interface (`step()`, `integrate()`, `clear()`, `solve_constraints()`, `apply_dP()`, `update_velocities()`, fields like `pos`, `vel`, `pprev`, `cons`, `activation`, `bone_pos_field`). Internally, all `ti.field` become `wp.array`, all `@ti.kernel`/`@ti.func` become `@wp.kernel`/`@wp.func`, and `ti.types.struct` becomes `@wp.struct`. The `Visualizer` class is removed from `muscle_warp.py` (it depends on `ti.ui` which is Taichi-specific and is not part of the muscle physics). Non-kernel Python code (mesh loading, constraint building, etc.) stays as pure NumPy.

**Tech Stack:** `warp-lang`, `numpy`, Python 3.10+

---

## Scope Clarification

### What Changes
- `src/VMuscle/muscle_warp.py`: Complete rewrite of Taichi primitives to Warp equivalents
  - `ti.field` / `ti.Vector.field` / `ti.Matrix.field` → `wp.array(dtype=...)`
  - `@ti.kernel` → `@wp.kernel` with explicit `wp.launch()`
  - `@ti.func` → `@wp.func`
  - `ti.types.struct` → `@wp.struct`
  - `@ti.data_oriented` → removed (not needed)
  - `ti.init()` → `wp.init()`
  - `ti.atomic_add` → `wp.atomic_add`
  - `ti.svd` → `wp.svd3`
  - `ti.select` → `wp.select`
  - `Visualizer` class → removed (Taichi-specific GUI)

### What Does NOT Change
- `src/VMuscle/muscle.py` — the original Taichi version stays untouched
- `src/VMuscle/solver_muscle_bone_coupled.py` — must work with either version (test validates this)
- `examples/` — no changes
- All pure-Python/NumPy code in `muscle_warp.py` (mesh loading, constraint building logic) stays the same

### Public API Contract (must remain identical)
```python
class MuscleSim:
    # Fields (now wp.array instead of ti.field, but .numpy() works the same)
    pos: wp.array(dtype=wp.vec3)         # was ti.Vector.field(3, ...)
    pprev: wp.array(dtype=wp.vec3)
    pos0: wp.array(dtype=wp.vec3)
    vel: wp.array(dtype=wp.vec3)
    force: wp.array(dtype=wp.vec3)
    mass: wp.array(dtype=float)
    stopped: wp.array(dtype=int)
    v_fiber_dir: wp.array(dtype=wp.vec3)
    dP: wp.array(dtype=wp.vec3)
    dPw: wp.array(dtype=float)
    tet_indices: wp.array(dtype=wp.vec4i)
    rest_volume: wp.array(dtype=float)
    rest_matrix: wp.array(dtype=wp.mat33)
    activation: wp.array(dtype=float)
    tendonmask: wp.array(dtype=float)
    bone_pos_field: wp.array(dtype=wp.vec3)
    cons: wp.array(dtype=Constraint)     # Constraint is @wp.struct

    # Methods
    def step(self)
    def integrate(self)
    def clear(self)
    def solve_constraints(self)
    def apply_dP(self)
    def update_velocities(self)
    def update_attach_targets(self)
    def reset(self)
```

---

## Task 1: Warp Struct & Array Infrastructure

**Files:**
- Modify: `src/VMuscle/muscle_warp.py`

**Step 1: Replace imports and initialization**

Replace `import taichi as ti` with `import warp as wp`, remove `ti.init()`, add `wp.init()`. Remove `@ti.data_oriented` decorators. Remove `pick_arch()` function (Warp handles device selection differently). Remove `Visualizer` class entirely.

**Step 2: Define Constraint struct with `@wp.struct`**

Replace `ti.types.struct(...)` with:
```python
@wp.struct
class Constraint:
    type: int
    cidx: int
    pts: wp.vec4i          # 4 int indices
    stiffness: float
    dampingratio: float
    tetid: int
    L: wp.vec3
    restlength: float
    restvector: wp.vec4
    restdir: wp.vec3
    compressionstiffness: float
```

Note: Taichi uses `ti.Vector([...])` when building constraints — replace these with `wp.vec4i(...)`, `wp.vec3(...)`, `wp.vec4(...)` in all `create_*_constraint` methods.

**Step 3: Replace field allocations in `_allocate_fields()`**

```python
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
```

**Step 4: Replace `_init_fields()` — use `wp.array()` from numpy**

Replace `field.from_numpy(arr)` with `wp.array(arr, dtype=...)`. Replace `field.fill(0)` with `self.field.zero_()` or re-assign `wp.zeros(...)`.

**Step 5: Replace `build_constraints()` — use `wp.array(dtype=Constraint)`**

The constraint building loop stays as NumPy. At the end, pack into a `wp.array`:
```python
# Build structured numpy array, then convert
cons_list = [Constraint() for ...] # not possible directly
# Instead: build arrays of each field, then copy
```

Actually, for Warp structs, we need to build them on the CPU and copy. The most practical approach: keep a list of Constraint instances, then use `wp.array(cons_list, dtype=Constraint)`.

**Step 6: Replace bone field allocations**

In `load_bone_geo()`, replace:
- `ti.Vector.field(3, dtype=ti.f32, shape=N)` → `wp.zeros(N, dtype=wp.vec3)`
- `ti.field(dtype=ti.i32, shape=N)` → `wp.zeros(N, dtype=int)`
- `field.from_numpy(arr)` → `wp.array(arr, dtype=...)`

**Step 7: Run smoke test**

```bash
uv run python -c "from VMuscle.muscle_warp import MuscleSim, SimConfig; print('Import OK')"
```
Expected: `Import OK` (no Taichi import errors)

**Step 8: Commit**

```bash
git add src/VMuscle/muscle_warp.py
git commit -m "refactor(muscle_warp): replace taichi fields/structs with warp arrays/structs"
```

---

## Task 2: Math Utility Functions (Warp @wp.func)

**Files:**
- Modify: `src/VMuscle/muscle_warp.py`

**Step 1: Convert `@ti.func` math helpers to `@wp.func`**

These are module-level functions. Convert each one:

```python
@wp.func
def flatten_mat33(mat: wp.mat33) -> wp.vec(length=9, dtype=float):
    # Flatten 3x3 matrix to 9-element vector
    ...

@wp.func
def update_dP(dP: wp.array(dtype=wp.vec3), dPw: wp.array(dtype=float),
              dp: wp.vec3, pt: int):
    wp.atomic_add(dP, pt, dp)
    wp.atomic_add(dPw, pt, 1.0)

@wp.func
def fem_flags_fn(ctype: int) -> int:
    flags = 0
    if ctype == TETARAP or ctype == TETARAPVOL or ctype == TETARAPNORM or ctype == TETARAPNORMVOL:
        flags = flags | LINEARENERGY
    if ctype == TRIARAPNORM or ctype == TETARAPNORM or ctype == TETARAPNORMVOL or ctype == TETFIBERNORM:
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
    if wp.determinant(U) < 0.0:
        # flip third column of U and negate sigma[2,2]
        ...
    if wp.determinant(V) < 0.0:
        ...
    return U, sigma, V

@wp.func
def polar_decomposition_fn(F: wp.mat33):
    U, sigma, V = ssvd_fn(F)
    R = U * wp.transpose(V)
    S = V * sigma * wp.transpose(V)
    return S, R

@wp.func
def get_inv_mass_fn(idx: int, mass: wp.array(dtype=float),
                    stopped: wp.array(dtype=int)) -> float:
    if stopped[idx] != 0:
        return 0.0
    m = mass[idx]
    if m > 0.0:
        return 1.0 / m
    return 0.0
```

Key Warp differences:
- `ti.svd(F)` → `wp.svd3(F)` (returns `U, sigma, V`)
- `mat.determinant()` → `wp.determinant(mat)`
- `v.norm()` → `wp.length(v)`
- `v.norm_sqr()` → `wp.dot(v, v)` (or `wp.length_sq(v)`)
- `v.dot(w)` → `wp.dot(v, w)`
- `v.cross(w)` → `wp.cross(v, w)`
- `ti.Matrix.outer_product(a, b)` → `wp.outer(a, b)`
- `ti.abs(x)` → `wp.abs(x)`
- `ti.sqrt(x)` → `wp.sqrt(x)`
- `ti.select(cond, a, b)` → `wp.select(cond, a, b)` (Note: warp's select is `wp.select(cond, true_val, false_val)`)
- `mat @ vec` → `mat * vec` in Warp (Warp uses `*` for matrix-vector multiply)
- `mat.transpose()` → `wp.transpose(mat)`
- `mat.inverse()` → `wp.inverse(mat)` (only for small matrices — check support)

Note on `wp.svd3`: Warp's SVD returns `(U, sigma, V)` where sigma is a `wp.vec3` (diagonal), not a matrix. Adjust accordingly.

**Step 2: Convert `mat3_to_quat` and `triangle_xform_and_area`**

These require careful translation — `triangle_xform_and_area` builds a 3x3 rotation matrix from cross products.

**Step 3: Convert `transfer_tension` and `transfer_shape_and_bulge`**

These are simple scalar math — straightforward conversion.

**Step 4: Commit**

```bash
git add src/VMuscle/muscle_warp.py
git commit -m "refactor(muscle_warp): convert ti.func math utilities to wp.func"
```

---

## Task 3: Core Simulation Kernels

**Files:**
- Modify: `src/VMuscle/muscle_warp.py`

The Taichi `MuscleSim` class uses `@ti.kernel` methods on the class (via `@ti.data_oriented`). Warp does not support class method kernels. All kernels must be module-level `@wp.kernel` functions, launched via `wp.launch()` from the class methods.

**Step 1: Convert `_precompute_rest` kernel**

Taichi version accesses `self.pos0`, `self.tet_indices`, etc. Warp version must take all arrays as explicit parameters:

```python
@wp.kernel
def precompute_rest_kernel(
    pos0: wp.array(dtype=wp.vec3),
    tet_indices: wp.array(dtype=wp.vec4i),
    rest_volume: wp.array(dtype=float),
    rest_matrix: wp.array(dtype=wp.mat33),
    mass: wp.array(dtype=float),
    total_rest_volume: wp.array(dtype=float),
    density: float,
):
    c = wp.tid()
    pts = tet_indices[c]
    # Ds_rest: columns are (pos0[pts[i]] - pos0[pts[3]]) for i=0,1,2
    col0 = pos0[pts[0]] - pos0[pts[3]]
    col1 = pos0[pts[1]] - pos0[pts[3]]
    col2 = pos0[pts[2]] - pos0[pts[3]]
    Dm = wp.mat33(col0[0], col1[0], col2[0],
                  col0[1], col1[1], col2[1],
                  col0[2], col1[2], col2[2])
    vol = wp.abs(wp.determinant(Dm)) / 6.0
    rest_volume[c] = vol
    wp.atomic_add(total_rest_volume, 0, vol)
    mass_contrib = vol * density / 4.0
    for i in range(4):
        wp.atomic_add(mass, pts[i], mass_contrib)
    rest_matrix[c] = wp.inverse(Dm)
```

Then in the class:
```python
def _precompute_rest(self):
    self.total_rest_volume = wp.zeros(1, dtype=float)
    wp.launch(precompute_rest_kernel, dim=self.tet_np.shape[0],
              inputs=[self.pos0, self.tet_indices, self.rest_volume,
                      self.rest_matrix, self.mass, self.total_rest_volume,
                      self.cfg.density])
```

**Step 2: Convert `integrate` kernel**

```python
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
    v = (1.0 - veldamping) * vel[i] + dt * extacc
    vel[i] = v
    pos[i] = pos[i] + dt * v
```

**Step 3: Convert `update_velocities` kernel**

```python
@wp.kernel
def update_velocities_kernel(
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    dt: float,
):
    i = wp.tid()
    vel[i] = (pos[i] - pprev[i]) / dt
```

**Step 4: Convert `clear` kernel**

```python
@wp.kernel
def clear_kernel(
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=float),
    cons: wp.array(dtype=Constraint),
):
    i = wp.tid()
    if i < dP.shape[0]:
        dP[i] = wp.vec3(0.0, 0.0, 0.0)
        dPw[i] = 0.0
    # Clear constraint L separately (different array size)

@wp.kernel
def clear_cons_L_kernel(cons: wp.array(dtype=Constraint)):
    i = wp.tid()
    cons[i].L = wp.vec3(0.0, 0.0, 0.0)
```

**Step 5: Convert `apply_dP` kernel**

```python
@wp.kernel
def apply_dP_kernel(
    pos: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=float),
):
    i = wp.tid()
    w = dPw[i]
    if w > 1.0e-9:
        pos[i] = pos[i] + dP[i] / w
```

**Step 6: Convert `_compute_cell_tendon_mask` kernel**

```python
@wp.kernel
def compute_cell_tendon_mask_kernel(
    tet_indices: wp.array(dtype=wp.vec4i),
    v_tendonmask: wp.array(dtype=float),
    tendon_mask: wp.array(dtype=float),
):
    c = wp.tid()
    pts = tet_indices[c]
    s = v_tendonmask[pts[0]] + v_tendonmask[pts[1]] + v_tendonmask[pts[2]] + v_tendonmask[pts[3]]
    tendon_mask[c] = s / 4.0
```

**Step 7: Convert `_update_attach_targets_kernel`**

```python
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
            cons[c].restvector = wp.vec4(target_pos[0], target_pos[1], target_pos[2], 1.0)
    elif ctype == DISTANCELINE:
        tgt_idx = cons[c].pts[1]
        if tgt_idx >= 0:
            target_pos = bone_pos[tgt_idx]
            cons[c].restvector = wp.vec4(target_pos[0], target_pos[1], target_pos[2], 1.0)
```

**Step 8: Commit**

```bash
git add src/VMuscle/muscle_warp.py
git commit -m "refactor(muscle_warp): convert core simulation kernels to warp"
```

---

## Task 4: Constraint Solver Kernels

**Files:**
- Modify: `src/VMuscle/muscle_warp.py`

This is the largest and most complex task. The Taichi version has one mega-kernel `solve_constraints` that dispatches to `@ti.func` methods based on constraint type. In Warp, we convert these to `@wp.func` helpers called from a `@wp.kernel`.

**Step 1: Convert `tet_volume_update_xpbd` to `@wp.func`**

Key differences:
- `ti.types.vector(4, ti.i32)` → `wp.vec4i`
- `grad.norm_sqr()` → `wp.dot(grad, grad)`
- All field access via explicit array parameters

**Step 2: Convert `tet_fiber_update_xpbd` to `@wp.func`**

Key differences:
- `ti.Matrix.cols([...])` → manual column construction with `wp.mat33(...)`
- `_Ds @ Dminv` → `_Ds * Dminv` (Warp uses `*` for mat-mat multiply)
- `ti.Matrix.outer_product(a, b)` → `wp.outer(a, b)` (for vec3 outer product → mat33)
- `Ht[0,0], Ht[0,1], Ht[0,2]` → extract row from mat33

Note: `restvector.xyz` in Taichi returns a vec3. In Warp, access `wp.vec3(rv[0], rv[1], rv[2])` from a vec4.

**Step 3: Convert `distance_pos_update_xpbd` and `distance_update_xpbd` to `@wp.func`**

Relatively straightforward scalar math with vector operations.

**Step 4: Convert `tri_arap_update_xpbd` to `@wp.func`**

Needs `triangle_xform_and_area`, polar decomposition via 2D rotation extraction.

**Step 5: Convert `tet_arap_update_xpbd` to `@wp.func`**

Uses `polar_decomposition` (SVD-based), `mat3_to_quat`.

**Step 6: Write the main `solve_constraints` kernel**

```python
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
    ctype = cons[c].type
    stiffness = cons[c].stiffness
    if stiffness <= 0.0:
        return
    kstiffcompress = get_compression_stiffness_fn(cons, c, has_compressstiffness)

    if ctype == TETVOLUME:
        tet_volume_update_xpbd_fn(use_jacobi, c, cons, pos, pprev, dP, dPw, ...)
    elif ctype == TETFIBERNORM:
        # compute fiberscale from activation and tendonmask
        ...
        tet_fiber_update_xpbd_fn(...)
    elif ctype == DISTANCE:
        distance_update_xpbd_fn(...)
    elif ctype == ATTACH:
        distance_pos_update_xpbd_fn(...)
    elif ctype == PIN:
        distance_pos_update_xpbd_fn(...)
    elif ctype == DISTANCELINE:
        # project, then distance_pos_update
        ...
    elif ctype == TETARAP:
        tet_arap_update_xpbd_fn(...)
    elif ctype == TRIARAP:
        tri_arap_update_xpbd_fn(...)
```

**Step 7: Update class `step()` method to use `wp.launch()`**

```python
def step(self):
    self.update_attach_targets()
    for _ in range(self.cfg.num_substeps):
        wp.launch(integrate_kernel, dim=self.n_verts, inputs=[...])
        wp.launch(clear_kernel, dim=self.n_verts, inputs=[...])
        wp.launch(clear_cons_L_kernel, dim=self.cons.shape[0], inputs=[self.cons])
        wp.launch(solve_constraints_kernel, dim=self.cons.shape[0], inputs=[...])
        if self.use_jacobi:
            wp.launch(apply_dP_kernel, dim=self.n_verts, inputs=[...])
        wp.launch(update_velocities_kernel, dim=self.n_verts, inputs=[...])
```

**Step 8: Commit**

```bash
git add src/VMuscle/muscle_warp.py
git commit -m "refactor(muscle_warp): convert constraint solver kernels to warp"
```

---

## Task 5: Compatibility Layer for SolverMuscleBoneCoupled

**Files:**
- Modify: `src/VMuscle/muscle_warp.py`

`solver_muscle_bone_coupled.py` accesses `MuscleSim` fields via Taichi APIs (`.from_numpy()`, `field[idx]`, `field.fill()`). We need to ensure Warp arrays support equivalent access patterns.

**Step 1: Ensure `.numpy()` compatibility**

Warp arrays already support `.numpy()`. No changes needed.

**Step 2: Ensure element access compatibility**

Taichi: `self.core.cons[cidx].restvector.xyz` — returns a vec3.
Warp: `self.core.cons.numpy()[cidx]` then access fields. However, for GPU arrays, direct element access like `cons[i]` works but returns to CPU.

The solver accesses:
- `self.core.cons[cidx].restvector.xyz` — used in `_compute_torque_kernel` (a Taichi kernel)
- `self.core.pos[src_idx]` — used in the same Taichi kernel
- `self.core.bone_pos_field[idx]` — written to in `_sync_bone_kernel`
- `self.core.activation.fill(val)` — fill activation

For the coupled solver to work, we have two options:
1. Also convert `solver_muscle_bone_coupled.py` to Warp (out of scope per user request)
2. Add numpy-based compatibility wrappers

Since the user says "keep everything else unchanged", we need to make `muscle_warp.py`'s MuscleSim provide attribute access that the Taichi-based coupled solver can use. This means the coupled solver needs to be updated to detect which backend is used. However, this contradicts "keep everything else unchanged."

**Resolution:** The most practical approach — make `muscle_warp.py` export arrays that can be accessed by index from Python (Warp supports this natively: `wp_array[i]` reads a single element). For `field.fill(val)`, add a convenience method. The coupled solver's Taichi kernels that directly access muscle fields will need to use numpy as intermediary — but since the user wants to keep solver_muscle_bone_coupled unchanged, we need to provide taichi-compatible `.fill()` and element access.

Actually, looking more carefully: `solver_muscle_bone_coupled.py` has its own `@ti.kernel` that reads `self.core.bone_pos_field`, `self.core.pos`, and `self.core.cons`. These are Taichi kernels reading Taichi fields. If we change those fields to Warp arrays, the Taichi kernels in the coupled solver will break.

**Revised approach:** We should add a thin compatibility layer. The `MuscleSim` class provides:
- `.fill(val)` method on arrays (wrap `wp.array` to support this)
- For the coupled solver's Taichi kernels to read warp data, we sync via numpy at the coupling boundary

However, this is getting complex. A cleaner approach: **the test compares muscle.py vs muscle_warp.py standalone** (not through the coupled solver). The coupled solver integration is a separate concern.

**Step 3: Add `.fill()` helper**

```python
# Utility: fill a warp array with a scalar value
def wp_fill(arr, val):
    arr.zero_()
    if val != 0.0:
        np_arr = arr.numpy()
        np_arr[:] = val
        wp.copy(arr, wp.array(np_arr, dtype=arr.dtype))
```

Or use a fill kernel:
```python
@wp.kernel
def fill_float_kernel(arr: wp.array(dtype=float), val: float):
    i = wp.tid()
    arr[i] = val
```

**Step 4: Commit**

```bash
git add src/VMuscle/muscle_warp.py
git commit -m "refactor(muscle_warp): add compatibility helpers for field access"
```

---

## Task 6: Comparison Test

**Files:**
- Create: `tests/test_muscle_warp_vs_taichi.py`

This test verifies that `muscle_warp.py` (Warp) produces the same numerical output as `muscle.py` (Taichi) for the same input.

**Step 1: Write the comparison test**

```python
"""Compare muscle.py (Taichi) vs muscle_warp.py (Warp) output."""
import numpy as np
import sys, os

def test_one_tet_step():
    """Run both versions on a single tetrahedron for N steps, compare positions."""
    # --- Taichi version ---
    from VMuscle.muscle import MuscleSim as MuscleSim_Ti, SimConfig as SimConfig_Ti
    cfg_ti = SimConfig_Ti(
        geo_path=None,  # triggers load_mesh_one_tet
        bone_geo_path="nonexistent",
        gui=False,
        render_mode=None,
        arch="cpu",
        nsteps=10,
        num_substeps=5,
        constraints=[
            {"type": "volume", "stiffness": 1e4},
        ],
    )
    sim_ti = MuscleSim_Ti(cfg_ti)
    for _ in range(cfg_ti.nsteps):
        sim_ti.step()
    pos_ti = sim_ti.pos.to_numpy()

    # --- Warp version ---
    from VMuscle.muscle_warp import MuscleSim as MuscleSim_Wp, SimConfig as SimConfig_Wp
    cfg_wp = SimConfig_Wp(
        geo_path=None,
        bone_geo_path="nonexistent",
        gui=False,
        render_mode=None,
        nsteps=10,
        num_substeps=5,
        constraints=[
            {"type": "volume", "stiffness": 1e4},
        ],
    )
    sim_wp = MuscleSim_Wp(cfg_wp)
    for _ in range(cfg_wp.nsteps):
        sim_wp.step()
    pos_wp = sim_wp.pos.numpy()

    # Compare
    np.testing.assert_allclose(pos_ti, pos_wp, atol=1e-5, rtol=1e-4,
                               err_msg="Taichi vs Warp positions diverged")

def test_fiber_constraint():
    """Test with fiber constraints and activation."""
    # Similar setup with fiber constraints, activation=0.3
    ...

def test_multiple_constraint_types():
    """Test with volume + fiber + arap constraints."""
    ...
```

**Step 2: Run the test**

```bash
uv run python -m pytest tests/test_muscle_warp_vs_taichi.py -v
```

Expected: All tests PASS with positions matching within tolerance.

Note: Due to floating-point differences between Taichi and Warp backends (different GPU kernels, different reduction orderings), exact match is unlikely. Use `atol=1e-4` initially. If Gauss-Seidel (non-Jacobi) mode is used, the parallel execution order may cause larger differences — test with `use_jacobi=True` first for deterministic comparison.

**Step 3: Commit**

```bash
git add tests/test_muscle_warp_vs_taichi.py
git commit -m "test: add comparison test for muscle taichi vs warp"
```

---

## Task 7: Integration Test with .geo Mesh

**Files:**
- Modify: `tests/test_muscle_warp_vs_taichi.py`

**Step 1: Add test with actual bicep.geo mesh**

```python
def test_bicep_geo_step():
    """Run both versions on bicep.geo for a few steps, compare positions."""
    geo_path = "data/muscle/model/bicep.geo"
    bone_path = "data/muscle/model/bicep_bone.geo"
    config_path = "data/muscle/config/bicep.json"

    from VMuscle.muscle import MuscleSim as MuscleSim_Ti, load_config as load_config_ti
    cfg_ti = load_config_ti(config_path)
    cfg_ti.gui = False
    cfg_ti.render_mode = None
    cfg_ti.nsteps = 3
    sim_ti = MuscleSim_Ti(cfg_ti)
    sim_ti.activation.fill(0.3)
    for _ in range(cfg_ti.nsteps):
        sim_ti.step()
    pos_ti = sim_ti.pos.to_numpy()

    from VMuscle.muscle_warp import MuscleSim as MuscleSim_Wp, load_config as load_config_wp
    cfg_wp = load_config_wp(config_path)
    cfg_wp.gui = False
    cfg_wp.render_mode = None
    cfg_wp.nsteps = 3
    sim_wp = MuscleSim_Wp(cfg_wp)
    # sim_wp.activation needs fill
    wp_fill(sim_wp.activation, 0.3)
    for _ in range(cfg_wp.nsteps):
        sim_wp.step()
    pos_wp = sim_wp.pos.numpy()

    np.testing.assert_allclose(pos_ti, pos_wp, atol=1e-4, rtol=1e-3)
```

**Step 2: Run the test**

```bash
uv run python -m pytest tests/test_muscle_warp_vs_taichi.py::test_bicep_geo_step -v
```

**Step 3: Commit**

```bash
git add tests/test_muscle_warp_vs_taichi.py
git commit -m "test: add bicep.geo integration test for warp vs taichi"
```

---

## Task 8: Cleanup & Final Verification

**Files:**
- Modify: `src/VMuscle/muscle_warp.py` (if needed)

**Step 1: Remove any remaining `import taichi` references**

```bash
grep -n "taichi\|ti\." src/VMuscle/muscle_warp.py
```
Expected: No matches (except in comments/docstrings).

**Step 2: Run full test suite**

```bash
uv run python -m pytest tests/ -v
```

**Step 3: Verify muscle.py is untouched**

```bash
git diff src/VMuscle/muscle.py
```
Expected: No changes.

**Step 4: Final commit**

```bash
git add -A
git commit -m "refactor(muscle_warp): complete taichi-to-warp migration with tests"
```

---

## Key Taichi → Warp Translation Reference

| Taichi | Warp | Notes |
|--------|------|-------|
| `ti.init(arch=...)` | `wp.init()` | Warp auto-detects GPU |
| `ti.field(dtype=ti.f32, shape=N)` | `wp.zeros(N, dtype=float)` | |
| `ti.Vector.field(3, ti.f32, shape=N)` | `wp.zeros(N, dtype=wp.vec3)` | |
| `ti.Matrix.field(3,3, ti.f32, shape=N)` | `wp.zeros(N, dtype=wp.mat33)` | |
| `field.from_numpy(arr)` | `wp.array(arr, dtype=...)` | Reassign variable |
| `field.to_numpy()` | `arr.numpy()` | |
| `field.fill(val)` | Custom fill kernel | |
| `@ti.data_oriented` | (not needed) | |
| `@ti.kernel` (class method) | `@wp.kernel` (module-level) + `wp.launch()` | |
| `@ti.func` | `@wp.func` | |
| `ti.types.struct(...)` | `@wp.struct class ...` | |
| `for i in field:` | `i = wp.tid()` | |
| `ti.atomic_add(field[i], val)` | `wp.atomic_add(arr, i, val)` | |
| `v.norm()` | `wp.length(v)` | |
| `v.norm_sqr()` | `wp.length_sq(v)` or `wp.dot(v,v)` | |
| `v.dot(w)` | `wp.dot(v, w)` | |
| `v.cross(w)` | `wp.cross(v, w)` | |
| `m.determinant()` | `wp.determinant(m)` | |
| `m.inverse()` | `wp.inverse(m)` | Only for mat22/mat33 |
| `m.transpose()` | `wp.transpose(m)` | |
| `m @ v` (mat-vec) | `m * v` | Warp uses `*` |
| `m1 @ m2` (mat-mat) | `m1 * m2` | Warp uses `*` |
| `ti.svd(F)` | `wp.svd3(F)` | sigma is vec3 in Warp |
| `ti.select(c, a, b)` | `wp.select(c, a, b)` | |
| `ti.abs(x)` | `wp.abs(x)` | |
| `ti.sqrt(x)` | `wp.sqrt(x)` | |
| `ti.Vector([x,y,z])` | `wp.vec3(x,y,z)` | |
| `ti.Matrix.cols([c0,c1,c2])` | Manual `wp.mat33(...)` construction | |
| `ti.Matrix.outer_product(a,b)` | `wp.outer(a,b)` | |
| `ti.Matrix.zero(ti.f32, 3, 3)` | `wp.mat33(0.0,...)` or `wp.identity(n=3)*0` | |
| `ti.loop_config(serialize=True)` | (no equivalent — use atomic ops) | |
