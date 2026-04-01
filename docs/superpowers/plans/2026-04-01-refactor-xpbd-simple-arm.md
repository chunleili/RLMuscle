# Refactor XPBD SimpleArm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `example_xpbd_coupled_simple_arm.py` from a 718-line monolith into a clean, modular structure matching the `example_couple2.py` pattern, with all config from JSON, no hacks, and identical numerical output.

**Architecture:** Extract shared utilities (`build_mjcf`) to `src/VMuscle/`, split the XPBD example into focused modules (mesh helpers, sim builder, coupled loop), read all parameters from `data/simpleArm/config.json`. The refactored example should be ~100 lines calling well-factored library code.

**Tech Stack:** Python, Warp (wp), MuJoCo, NumPy

---

## Baseline

The baseline output is saved at:
- `output/SimpleArm_XPBD_Coupled_states_BASELINE.sto` — numerical trajectory
- `output/simple_arm_xpbd_vs_osim_BASELINE.png` — comparison plot
- Key metrics: RMSE=1.70 deg, Max error=7.97 deg, final angle=82.2 deg

**Regression criterion:** After refactoring, re-running `uv run python scripts/run_simple_arm_comparison.py --mode xpbd` must produce STO output with max per-field deviation < 1e-6 from baseline.

---

## File Structure

### Files to create:
- `src/VMuscle/simple_arm_mujoco.py` — Extract `build_mjcf()` from `examples/example_mujoco_simple_arm.py` (shared by multiple examples)
- `src/VMuscle/simple_arm_xpbd.py` — XPBD sim builder + coupled step logic (extracted from example)

### Files to modify:
- `examples/example_xpbd_coupled_simple_arm.py` — Rewrite to thin ~100 line orchestrator
- `examples/example_mujoco_simple_arm.py` — Import `build_mjcf` from new location, keep backward compat
- `data/simpleArm/config.json` — Add any missing XPBD params that are currently hardcoded

### Files unchanged (library code, already clean):
- `src/VMuscle/activation.py` — `activation_dynamics_step_np`
- `src/VMuscle/dgf_curves.py` — `active_force_length`, `force_velocity`, `passive_force_length`
- `src/VMuscle/mesh_utils.py` — `create_cylinder_tet_mesh`
- `src/VMuscle/mesh_io.py` — `MeshExporter`, `save_ply`
- `src/VMuscle/muscle_warp.py` — `MuscleSim`, `Constraint`
- `src/VMuscle/constraints.py` — constraint type constants

---

## Task 1: Extract `build_mjcf` to shared module

**Context:** `build_mjcf(cfg)` in `examples/example_mujoco_simple_arm.py:39-115` generates the MJCF XML for the simple arm MuJoCo model. It is imported by the XPBD example via `from example_mujoco_simple_arm import build_mjcf`. This cross-example dependency is a hack. Move it to `src/VMuscle/simple_arm_mujoco.py`.

**Files:**
- Create: `src/VMuscle/simple_arm_mujoco.py`
- Modify: `examples/example_mujoco_simple_arm.py`
- Modify: `examples/example_xpbd_coupled_simple_arm.py` (update import — will be rewritten later but needs to work now)

- [ ] **Step 1: Create `src/VMuscle/simple_arm_mujoco.py`**

Copy `build_mjcf` function verbatim from `examples/example_mujoco_simple_arm.py:39-115` into a new file:

```python
"""MuJoCo SimpleArm model builder — shared by all SimpleArm examples."""

import textwrap


def build_mjcf(cfg):
    """Generate MJCF XML for the simple arm model.
    
    Args:
        cfg: Config dict with "geometry" section containing humerus_length,
             radius_length, muscle_origin_on_humerus, muscle_insertion_on_radius.
    
    Returns:
        MJCF XML string.
    """
    geo = cfg["geometry"]
    L_h = geo["humerus_length"]
    L_r = geo["radius_length"]
    mo = geo["muscle_origin_on_humerus"]
    mi = geo["muscle_insertion_on_radius"]
    ox, oy, oz = mo[0], -(L_h - mo[1]), mo[2] if len(mo) > 2 else 0
    ix, iy, iz = mi[0], -(L_r - mi[1]), mi[2] if len(mi) > 2 else 0

    return textwrap.dedent(f"""\
    <?xml version="1.0" ?>
    <mujoco model="simple_arm">
      <option timestep="0.002" gravity="0 -9.81 0"
              integrator="implicit" solver="Newton"
              iterations="50" tolerance="1e-10"/>
      <compiler angle="radian"/>

      <worldbody>
        <body name="humerus" pos="0 0 0">
          <geom type="capsule" size="0.04" fromto="0 0 0 0 {-L_h} 0"
                rgba="0.7 0.7 0.7 0.8" mass="0"
                contype="0" conaffinity="0"/>
          <site name="muscle_origin" pos="{ox} {oy} {oz}" size="0.015"
                rgba="1 0 0 1"/>

          <body name="radius" pos="0 {-L_h} 0">
            <joint name="elbow" type="hinge" axis="0 0 1"
                   limited="true" range="0 3.14159"
                   damping="0" armature="0"/>
            <inertial pos="0 {-L_r} 0" mass="1"
                      diaginertia="0.001 0.001 0.001"/>
            <geom type="capsule" size="0.03" fromto="0 0 0 0 {-L_r} 0"
                  rgba="0.5 0.5 0.8 0.8" mass="0"
                  contype="0" conaffinity="0"/>
            <site name="muscle_insertion" pos="{ix} {iy} {iz}" size="0.015"
                  rgba="0 0 1 1"/>
          </body>
        </body>
      </worldbody>

      <tendon>
        <spatial name="biceps_tendon" stiffness="0" damping="0"
                 width="0.008" rgba="0.8 0.2 0.2 1">
          <site site="muscle_origin"/>
          <site site="muscle_insertion"/>
        </spatial>
      </tendon>

      <actuator>
        <motor name="biceps_motor" tendon="biceps_tendon" gear="-1"/>
      </actuator>
    </mujoco>""")
```

- [ ] **Step 2: Update `examples/example_mujoco_simple_arm.py` to import from new location**

Replace the `build_mjcf` function definition (lines 39-115) with an import:

```python
from VMuscle.simple_arm_mujoco import build_mjcf
```

Keep the rest of the file unchanged.

- [ ] **Step 3: Update XPBD example import**

In `examples/example_xpbd_coupled_simple_arm.py`, replace lines 420-423:
```python
_ed = os.path.dirname(os.path.abspath(__file__))
if _ed not in sys.path:
    sys.path.insert(0, _ed)
from example_mujoco_simple_arm import build_mjcf
```
with:
```python
from VMuscle.simple_arm_mujoco import build_mjcf
```

- [ ] **Step 4: Verify both examples still work**

Run: `uv run python scripts/run_simple_arm_comparison.py --mode xpbd 2>&1 | tail -5`
Expected: Same RMSE=1.70 deg, Max error=7.97 deg

- [ ] **Step 5: Commit**

```bash
git add src/VMuscle/simple_arm_mujoco.py examples/example_mujoco_simple_arm.py examples/example_xpbd_coupled_simple_arm.py
git commit -m "refactor: extract build_mjcf to src/VMuscle/simple_arm_mujoco.py"
```

---

## Task 2: Extract XPBD sim builder and helpers to `src/VMuscle/simple_arm_xpbd.py`

**Context:** The current `example_xpbd_coupled_simple_arm.py` has ~400 lines of utility functions and sim building code that should live in `src/`. These include: rotation helpers, constraint upload, `build_xpbd_muscle_sim()`, capsule mesh creation, fiber stretch computation, and the coupled step logic.

**Files:**
- Create: `src/VMuscle/simple_arm_xpbd.py`
- Test: Run regression after extraction

- [ ] **Step 1: Create `src/VMuscle/simple_arm_xpbd.py`**

Extract the following functions verbatim from `examples/example_xpbd_coupled_simple_arm.py`:

```python
"""XPBD SimpleArm — sim builder, mesh helpers, and coupled stepping logic."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import warp as wp

from VMuscle.activation import activation_dynamics_step_np
from VMuscle.constraints import ATTACH, TETARAP, TETVOLUME
from VMuscle.dgf_curves import active_force_length, force_velocity, passive_force_length
from VMuscle.mesh_io import MeshExporter, save_ply
from VMuscle.mesh_utils import create_cylinder_tet_mesh
from VMuscle.muscle_warp import Constraint, MuscleSim


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def rotation_matrix_from_z_to(target_dir):
    # ... copy verbatim from example lines 42-55 ...

def rotation_between(u, v):
    # ... copy verbatim from example lines 181-195 ...

def create_capsule_mesh(p0, p1, radius, n_circ=12, n_axial=8, n_cap=4):
    # ... copy verbatim from example lines 72-166 ...

def transform_capsule(verts, body_pos, body_quat):
    # ... copy verbatim from example lines 169-178 ...


# ---------------------------------------------------------------------------
# Fiber stretch computation
# ---------------------------------------------------------------------------

def compute_fiber_stretches(pos, tet_idx, rest_matrices, fiber_dirs):
    # ... copy verbatim from example lines 58-69 ...


# ---------------------------------------------------------------------------
# Constraint upload
# ---------------------------------------------------------------------------

def upload_all_constraints(sim, all_constraints):
    # ... copy verbatim from example lines 198-227 (rename from _upload_all_constraints) ...


# ---------------------------------------------------------------------------
# Build XPBD MuscleSim
# ---------------------------------------------------------------------------

def build_xpbd_muscle_sim(vertices, tets, fiber_dirs_per_tet,
                          origin_ids, insertion_ids, bone_targets_np, *,
                          dts, device="cpu",
                          volume_stiffness=1e6, arap_stiffness=1e6,
                          attach_origin_stiffness=1e37,
                          attach_insertion_stiffness=1e10,
                          density=1060.0, veldamping=0.02):
    # ... copy verbatim from example lines 234-368 ...
    # Use upload_all_constraints instead of _upload_all_constraints


# ---------------------------------------------------------------------------
# Coupled simulation loop
# ---------------------------------------------------------------------------

def run_xpbd_coupled_loop(cfg):
    """Run XPBD + MuJoCo coupled SimpleArm simulation.

    This is the main entry point. All parameters come from cfg dict.

    Returns:
        dict with times, elbow_angles, forces, norm_fiber_lengths,
        activations, max_iso_force, muscle_type.
    """
    # ... copy verbatim from example xpbd_coupled_simple_arm() lines 375-703 ...
    # Replace all internal references to use the functions defined above
    # Replace `from example_mujoco_simple_arm import build_mjcf`
    #   with `from VMuscle.simple_arm_mujoco import build_mjcf`
```

Copy ALL functions exactly as-is. The only changes allowed are:
1. Remove leading underscores from `_rotation_matrix_from_z_to`, `_rotation_between`, `_upload_all_constraints` (now public API)
2. Update internal cross-references to use new names
3. Replace the MuJoCo build_mjcf import with `from VMuscle.simple_arm_mujoco import build_mjcf`

- [ ] **Step 2: Verify the module imports cleanly**

Run: `uv run python -c "from VMuscle.simple_arm_xpbd import run_xpbd_coupled_loop; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/VMuscle/simple_arm_xpbd.py
git commit -m "refactor: extract XPBD sim builder and helpers to src/VMuscle/simple_arm_xpbd.py"
```

---

## Task 3: Rewrite `example_xpbd_coupled_simple_arm.py` as thin orchestrator

**Context:** Now that all logic lives in `src/VMuscle/simple_arm_xpbd.py`, rewrite the example to be a thin orchestrator (~30 lines), matching `example_couple2.py` style.

**Files:**
- Modify: `examples/example_xpbd_coupled_simple_arm.py` (complete rewrite)

- [ ] **Step 1: Rewrite the example**

```python
"""XPBD coupled SimpleArm example (Stage 3x).

Builds a MuscleSim programmatically from a cylinder mesh with
TETVOLUME + TETARAP + ATTACH constraints, coupled to MuJoCo rigid-body sim.

Usage:
    RUN=example_xpbd_coupled_simple_arm uv run main.py
    uv run python examples/example_xpbd_coupled_simple_arm.py
    uv run python examples/example_xpbd_coupled_simple_arm.py --config data/simpleArm/config.json
"""

import argparse
import json

import numpy as np

from VMuscle.simple_arm_xpbd import run_xpbd_coupled_loop


def load_config(path="data/simpleArm/config.json"):
    with open(path) as f:
        return json.load(f)


def xpbd_coupled_simple_arm(cfg, verbose=True):
    """Public API called by run_simple_arm_comparison.py."""
    return run_xpbd_coupled_loop(cfg, verbose=verbose)


def main():
    parser = argparse.ArgumentParser(description="XPBD coupled SimpleArm")
    parser.add_argument("--config", default="data/simpleArm/config.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = xpbd_coupled_simple_arm(cfg)
    if result:
        print(f"\nFinal elbow angle: {np.degrees(result['elbow_angles'][-1]):.2f}deg")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run regression test**

Run: `uv run python scripts/run_simple_arm_comparison.py --mode xpbd 2>&1 | tail -5`
Expected: Same RMSE=1.70 deg, Max error=7.97 deg

- [ ] **Step 3: Verify STO file matches baseline numerically**

Run a Python snippet to compare:
```python
import numpy as np
baseline = np.loadtxt("output/SimpleArm_XPBD_Coupled_states_BASELINE.sto", skiprows=7)
current = np.loadtxt("output/SimpleArm_XPBD_Coupled_states.sto", skiprows=7)
diff = np.abs(baseline - current).max()
print(f"Max deviation: {diff}")
assert diff < 1e-6, f"Regression! Max deviation={diff}"
```

- [ ] **Step 4: Commit**

```bash
git add examples/example_xpbd_coupled_simple_arm.py
git commit -m "refactor: rewrite XPBD example as thin orchestrator"
```

---

## Task 4: Clean up `simple_arm_xpbd.py` — remove hacks and redundancy

**Context:** The extracted code is a verbatim copy. Now clean it up:
1. Remove the `sys.path` hacks (no longer needed since we import from VMuscle)
2. Remove duplicate `load_config` (use `VMuscle.config.load_config` or inline JSON load)  
3. Move hardcoded constants (density=1060, veldamping=0.02, mj_per_xpbd=10, capsule params) into the config JSON
4. Remove `os.sys.path` manipulation
5. Remove dead code (unused imports, commented-out code)

**Files:**
- Modify: `src/VMuscle/simple_arm_xpbd.py`
- Modify: `data/simpleArm/config.json`

- [ ] **Step 1: Add missing parameters to config.json**

Add to the `"xpbd"` section:
```json
{
  "xpbd": {
    "num_substeps": 30,
    "volume_stiffness": 1e6,
    "arap_stiffness": 1e6,
    "attach_origin_stiffness": 1e10,
    "attach_insertion_stiffness": 1e10,
    "warmup_steps": 10,
    "mj_per_xpbd": 10,
    "density": 1060.0,
    "veldamping": 0.02
  }
}
```

- [ ] **Step 2: Update `run_xpbd_coupled_loop` to read all params from config**

Replace hardcoded values with `xpbd_cfg.get(key, default)`:
- `mj_per_xpbd = xpbd_cfg.get("mj_per_xpbd", 10)`
- `density = xpbd_cfg.get("density", 1060.0)`
- `veldamping = xpbd_cfg.get("veldamping", 0.02)`

Remove `sys.path` manipulation and `os` import if no longer needed.

- [ ] **Step 3: Run regression test**

Run: `uv run python scripts/run_simple_arm_comparison.py --mode xpbd 2>&1 | tail -5`
Expected: Same RMSE=1.70 deg, Max error=7.97 deg

Compare STO files:
```python
import numpy as np
baseline = np.loadtxt("output/SimpleArm_XPBD_Coupled_states_BASELINE.sto", skiprows=7)
current = np.loadtxt("output/SimpleArm_XPBD_Coupled_states.sto", skiprows=7)
diff = np.abs(baseline - current).max()
print(f"Max deviation: {diff}")
assert diff < 1e-6, f"Regression! Max deviation={diff}"
```

- [ ] **Step 4: Commit**

```bash
git add src/VMuscle/simple_arm_xpbd.py data/simpleArm/config.json
git commit -m "refactor: remove hacks, read all XPBD params from config.json"
```

---

## Task 5: Update `example_mujoco_simple_arm.py` to use shared `build_mjcf`

**Context:** Task 1 moved `build_mjcf` to `src/VMuscle/simple_arm_mujoco.py`. The `example_mujoco_simple_arm.py` should also be updated to verify no other examples break.

**Files:**
- Verify: `examples/example_mujoco_simple_arm.py` (should already use `from VMuscle.simple_arm_mujoco import build_mjcf`)

- [ ] **Step 1: Verify MuJoCo example works**

Run: `uv run python scripts/run_simple_arm_comparison.py --mode mujoco 2>&1 | tail -5`
Expected: Completes without error

- [ ] **Step 2: Verify all modes work**

Run: `uv run python scripts/run_simple_arm_comparison.py --mode xpbd 2>&1 | tail -5`
Expected: Same RMSE=1.70, Max error=7.97

- [ ] **Step 3: Commit (if any fixes needed)**

---

## Task 6: Final regression verification and cleanup

**Context:** Final pass to ensure everything is clean and matches baseline.

**Files:**
- All modified files

- [ ] **Step 1: Full numerical regression check**

```python
import numpy as np
baseline = np.loadtxt("output/SimpleArm_XPBD_Coupled_states_BASELINE.sto", skiprows=7)
current = np.loadtxt("output/SimpleArm_XPBD_Coupled_states.sto", skiprows=7)
diff = np.abs(baseline - current).max()
print(f"Max deviation: {diff}")
assert diff < 1e-6, f"Regression! Max deviation={diff}"
print("PASS: Numerical regression check passed")
```

- [ ] **Step 2: Verify no unused imports in all modified files**

Check: `src/VMuscle/simple_arm_mujoco.py`, `src/VMuscle/simple_arm_xpbd.py`, `examples/example_xpbd_coupled_simple_arm.py`, `examples/example_mujoco_simple_arm.py`

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "refactor: complete XPBD SimpleArm restructure with regression verification"
```
