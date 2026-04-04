"""Render a USD/USDA/USDC mesh animation to MP4 using matplotlib.

Requires: pxr (usd-core), matplotlib, ffmpeg (in PATH)

Usage:
    # Basic: render full animation (fps & frame range from USD metadata)
    uv run python scripts/usd_to_mp4.py output/anim/sliding_ball_default.usda

    # Custom output path
    uv run python scripts/usd_to_mp4.py input.usd -o output.mp4

    # Override fps and frame range
    uv run python scripts/usd_to_mp4.py input.usdc --fps 30 --start 0 --end 120

    # Adjust camera angle and resolution
    uv run python scripts/usd_to_mp4.py input.usda --elev 30 --azim -45 --dpi 150
"""

import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Render USD animation to MP4")
    parser.add_argument("input", help="Path to .usd/.usda/.usdc file")
    parser.add_argument("-o", "--output", help="Output .mp4 path (default: same name as input)")
    parser.add_argument("--fps", type=int, default=0, help="Output FPS (0 = use USDA timeCodesPerSecond)")
    parser.add_argument("--start", type=int, default=-1, help="Start frame (-1 = use USDA startTimeCode)")
    parser.add_argument("--end", type=int, default=-1, help="End frame (-1 = use USDA endTimeCode)")
    parser.add_argument("--dpi", type=int, default=100, help="Figure DPI")
    parser.add_argument("--elev", type=float, default=25, help="Camera elevation angle")
    parser.add_argument("--azim", type=float, default=-60, help="Camera azimuth angle")
    args = parser.parse_args()

    from pxr import Usd, Gf
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.animation as animation

    # Open USD stage
    stage = Usd.Stage.Open(args.input)
    fps = args.fps or int(stage.GetTimeCodesPerSecond())
    start = args.start if args.start >= 0 else int(stage.GetStartTimeCode())
    end = args.end if args.end >= 0 else int(stage.GetEndTimeCode())
    frames = list(range(start, end + 1))

    # Find mesh prim
    prim = None
    for p in stage.Traverse():
        if "Mesh" in p.GetTypeName():
            prim = p
            break
    if prim is None:
        print("No mesh prim found in USD stage")
        return

    pts_attr = prim.GetAttribute("points")
    # Get face indices - try surfaceFaceVertexIndices (TetMesh) then faceVertexIndices
    face_attr = prim.GetAttribute("surfaceFaceVertexIndices")
    if face_attr and face_attr.Get(start) is not None:
        raw_faces = face_attr.Get(start)
        tri_indices = np.array([[f[0], f[1], f[2]] for f in raw_faces])
    else:
        face_attr = prim.GetAttribute("faceVertexIndices")
        cnt_attr = prim.GetAttribute("faceVertexCounts")
        if face_attr and cnt_attr:
            idxs = np.array(face_attr.Get(start))
            counts = np.array(cnt_attr.Get(start))
            # Assume all triangles
            tri_indices = idxs.reshape(-1, 3)
        else:
            print("No face indices found")
            return

    # Preload all frames
    print(f"Loading {len(frames)} frames...")
    all_points = []
    for f in frames:
        pts = np.array(pts_attr.Get(f))
        all_points.append(pts)

    # Compute global bounds for consistent axes
    all_pts = np.concatenate(all_points, axis=0)
    center = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
    half_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.2

    # Setup figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    def draw_frame(i):
        ax.cla()
        pts = all_points[i]
        verts = pts[tri_indices]
        poly = Poly3DCollection(verts, alpha=0.8, edgecolors="k", linewidths=0.3)
        poly.set_facecolor((0.8, 0.3, 0.2, 0.8))
        ax.add_collection3d(poly)
        ax.set_xlim(center[0] - half_range, center[0] + half_range)
        ax.set_ylim(center[1] - half_range, center[1] + half_range)
        ax.set_zlim(center[2] - half_range, center[2] + half_range)
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Time {frames[i]*1.0/fps:.2f}s (frame {frames[i]})")
        return [poly]

    out_path = args.output or str(Path(args.input).with_suffix(".mp4"))
    print(f"Rendering {len(frames)} frames at {fps} fps -> {out_path}")

    ani = animation.FuncAnimation(fig, draw_frame, frames=len(frames), interval=1000 / fps, blit=False)
    ani.save(out_path, writer="ffmpeg", fps=fps, dpi=args.dpi)
    print(f"Done: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
