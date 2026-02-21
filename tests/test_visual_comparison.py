"""Visual comparison between Taichi and Warp muscle simulations.

Runs both simulators headless and renders surface mesh snapshots using
matplotlib (Agg backend) at key frames. Saves images to:
    output/taichi/step_XXX.png
    output/warp/step_XXX.png
    output/comparison/step_XXX.png  (side-by-side)

Usage:
    uv run python tests/test_visual_comparison.py
"""
import sys, os
sys.path.insert(0, "src")
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
import warp as wp

# ---- init frameworks ----
wp.init()
wp.set_device("cpu")  # match Agent 1: both on CPU to avoid GPU race conditions

from VMuscle.muscle import MuscleSim as TaichiSim, SimConfig as TaichiCfg, load_config, build_surface_tris
from VMuscle.muscle_warp import MuscleSim as WarpSim, SimConfig as WarpCfg, fill_float_kernel

SNAPSHOT_STEPS = [1, 50, 100, 200, 400]
ACTIVATION = 0.3
OUT_T = Path("output/taichi")
OUT_W = Path("output/warp")
OUT_C = Path("output/comparison")

for d in [OUT_T, OUT_W, OUT_C]:
    d.mkdir(parents=True, exist_ok=True)


def render_mesh(ax, pos, tris, title, color="steelblue", lims=None):
    """Render a surface mesh on a matplotlib 3D axis."""
    # filter out NaN/Inf vertices for rendering
    valid = np.isfinite(pos).all(axis=1)
    if not valid.all():
        n_bad = (~valid).sum()
        print(f"    WARNING: {n_bad}/{pos.shape[0]} vertices are NaN/Inf in '{title}'")

    verts = pos[tris]  # (n_tri, 3, 3)
    # skip tris with any NaN
    tri_valid = np.isfinite(verts).all(axis=(1, 2))
    verts = verts[tri_valid]

    if verts.shape[0] > 0:
        poly = Poly3DCollection(verts, alpha=0.6, edgecolor="k", linewidth=0.15)
        poly.set_facecolor(color)
        ax.add_collection3d(poly)

    # use shared limits if provided, otherwise compute from finite data
    if lims is not None:
        mins, maxs, margin = lims
    else:
        finite_pos = pos[valid]
        if finite_pos.shape[0] == 0:
            mins, maxs, margin = np.zeros(3), np.ones(3), 0.1
        else:
            mins = finite_pos.min(axis=0)
            maxs = finite_pos.max(axis=0)
            margin = (maxs - mins).max() * 0.1

    ax.set_xlim(float(mins[0] - margin), float(maxs[0] + margin))
    ax.set_ylim(float(mins[1] - margin), float(maxs[1] + margin))
    ax.set_zlim(float(mins[2] - margin), float(maxs[2] + margin))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")


def snapshot(step, pos_t, pos_w, tris):
    """Save individual + comparison images for one step."""
    # compute shared axis limits from both pos arrays (finite only)
    all_pos = np.concatenate([pos_t[np.isfinite(pos_t).all(axis=1)],
                               pos_w[np.isfinite(pos_w).all(axis=1)]], axis=0)
    if all_pos.shape[0] > 0:
        mins = all_pos.min(axis=0)
        maxs = all_pos.max(axis=0)
        margin = (maxs - mins).max() * 0.1
    else:
        mins, maxs, margin = np.zeros(3), np.ones(3), 0.1
    shared_lims = (mins, maxs, margin)

    # individual
    for pos, out_dir, label, clr in [
        (pos_t, OUT_T, "Taichi", "steelblue"),
        (pos_w, OUT_W, "Warp", "darkorange"),
    ]:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        render_mesh(ax, pos, tris, f"{label} step={step}", color=clr, lims=shared_lims)
        ax.view_init(elev=25, azim=-60)
        fig.savefig(out_dir / f"step_{step:04d}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    # side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                     subplot_kw={"projection": "3d"})
    render_mesh(ax1, pos_t, tris, f"Taichi step={step}", "steelblue", lims=shared_lims)
    render_mesh(ax2, pos_w, tris, f"Warp step={step}", "darkorange", lims=shared_lims)
    ax1.view_init(elev=25, azim=-60)
    ax2.view_init(elev=25, azim=-60)

    # numerical stats (handle NaN)
    finite_mask = np.isfinite(pos_t).all(axis=1) & np.isfinite(pos_w).all(axis=1)
    diff = np.abs(pos_t[finite_mask] - pos_w[finite_mask])
    n_nan_t = (~np.isfinite(pos_t).all(axis=1)).sum()
    n_nan_w = (~np.isfinite(pos_w).all(axis=1)).sum()
    if diff.size > 0:
        max_err = diff.max()
        mean_err = diff.mean()
    else:
        max_err = mean_err = float("nan")
    fig.suptitle(f"Step {step}  |  max_err={max_err:.6f}  mean_err={mean_err:.6f}"
                 f"  nan_t={n_nan_t}  nan_w={n_nan_w}", fontsize=10)
    fig.savefig(OUT_C / f"step_{step:04d}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [step {step:4d}] max_err={max_err:.6f}  mean_err={mean_err:.6f}"
          f"  nan_t={n_nan_t}  nan_w={n_nan_w}")


def run_test(snapshot_steps=None, activation=ACTIVATION):
    """Run visual comparison and generate PNGs.

    Returns dict with keys: image_paths, errors_per_step, passed.
    """
    if snapshot_steps is None:
        snapshot_steps = SNAPSHOT_STEPS

    # ---- Taichi sim ----
    cfg_t = load_config("data/muscle/config/bicep.json")
    cfg_t.gui = False
    cfg_t.render_mode = None
    taichi_sim = TaichiSim(cfg_t)

    # ---- Warp sim ----
    cfg_w = WarpCfg(
        geo_path=cfg_t.geo_path,
        bone_geo_path=cfg_t.bone_geo_path,
        dt=cfg_t.dt,
        nsteps=cfg_t.nsteps,
        num_substeps=cfg_t.num_substeps,
        gravity=cfg_t.gravity,
        density=cfg_t.density,
        veldamping=cfg_t.veldamping,
        activation=cfg_t.activation,
        constraints=cfg_t.constraints,
        gui=False,
        render_mode=None,
    )
    warp_sim = WarpSim(cfg_w)

    assert taichi_sim.n_verts == warp_sim.n_verts, "Vertex count mismatch!"

    # surface triangles (shared mesh)
    tris = build_surface_tris(np.asarray(taichi_sim.tet_np, dtype=np.int32))

    total_steps = max(snapshot_steps)
    print(f"Running {total_steps} steps, snapshots at {snapshot_steps} ...")

    image_paths = []
    errors_per_step = {}

    for step in range(1, total_steps + 1):
        # set activation for both
        cfg_t.activation = activation
        wp.launch(fill_float_kernel, dim=warp_sim.activation.shape[0],
                  inputs=[warp_sim.activation, activation])

        taichi_sim.step()
        warp_sim.step()

        if step in snapshot_steps:
            pos_t = taichi_sim.pos.to_numpy()
            pos_w = warp_sim.pos.numpy()
            snapshot(step, pos_t, pos_w, tris)
            img_path = OUT_C / f"step_{step:04d}.png"
            image_paths.append(str(img_path))
            # compute error for this step
            finite_mask = np.isfinite(pos_t).all(axis=1) & np.isfinite(pos_w).all(axis=1)
            if finite_mask.any():
                diff = np.abs(pos_t[finite_mask] - pos_w[finite_mask])
                errors_per_step[step] = float(diff.max())
            else:
                errors_per_step[step] = float("nan")

    print(f"\nDone. Images saved to {OUT_T}, {OUT_W}, {OUT_C}")

    # Check all images exist and are non-trivial
    passed = all(Path(p).exists() and Path(p).stat().st_size > 1024 for p in image_paths)
    return {"image_paths": image_paths, "errors_per_step": errors_per_step, "passed": passed}


def main():
    """Entry point for manual execution."""
    run_test()


if __name__ == "__main__":
    main()
