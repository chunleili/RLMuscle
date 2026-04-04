"""Profile three core examples with cProfile.

Usage:
    uv run python scripts/profile_examples.py [couple3|mujoco|xpbd]

Runs a short simulation (100 steps) with cProfile and prints top 40 cumulative-time entries.
If no argument given, profiles all three sequentially.
"""
import cProfile
import json
import os
import pstats
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("WARP_CACHE_PATH", os.path.join(PROJECT_ROOT, ".cache", "warp"))

# Init warp early
import warp
warp.init()

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def profile_couple3():
    """Profile example_couple3 for 100 steps, no USD."""
    from examples.example_couple3 import setup_couple3, run_loop

    solver, sim, state, cfg, dt = setup_couple3()

    pr = cProfile.Profile()
    pr.enable()
    run_loop(solver, state, cfg, dt=dt, n_steps=100, auto=True, usd=None, sim=sim)
    pr.disable()

    stats = pstats.Stats(pr)
    stats.strip_dirs()
    dump_path = os.path.join(OUTPUT_DIR, "profile_couple3.prof")
    stats.dump_stats(dump_path)
    print(f"\n{'='*80}")
    print("PROFILE: example_couple3 (100 steps)")
    print(f"{'='*80}")
    stats.sort_stats("cumulative")
    stats.print_stats(40)
    stats.sort_stats("tottime")
    print("\n--- Top 40 by tottime ---")
    stats.print_stats(40)
    return dump_path


def profile_mujoco():
    """Profile example_mujoco_simple_arm with reduced steps."""
    from examples.example_mujoco_simple_arm import mujoco_simple_arm

    with open(os.path.join(PROJECT_ROOT, "data/simpleArm/config.json")) as f:
        cfg = json.load(f)
    # Override to 200 steps for profiling
    cfg["solver"]["n_steps"] = 200

    pr = cProfile.Profile()
    pr.enable()
    mujoco_simple_arm(cfg, verbose=False)
    pr.disable()

    stats = pstats.Stats(pr)
    stats.strip_dirs()
    dump_path = os.path.join(OUTPUT_DIR, "profile_mujoco.prof")
    stats.dump_stats(dump_path)
    print(f"\n{'='*80}")
    print("PROFILE: example_mujoco_simple_arm (200 steps)")
    print(f"{'='*80}")
    stats.sort_stats("cumulative")
    stats.print_stats(40)
    stats.sort_stats("tottime")
    print("\n--- Top 40 by tottime ---")
    stats.print_stats(40)
    return dump_path


def profile_xpbd():
    """Profile example_xpbd_coupled_simple_arm with reduced steps."""
    from examples.example_xpbd_coupled_simple_arm import xpbd_coupled_simple_arm

    with open(os.path.join(PROJECT_ROOT, "data/simpleArm/config.json")) as f:
        cfg = json.load(f)
    # Override to 100 steps for profiling
    cfg["solver"]["n_steps"] = 100

    pr = cProfile.Profile()
    pr.enable()
    xpbd_coupled_simple_arm(cfg, verbose=False)
    pr.disable()

    stats = pstats.Stats(pr)
    stats.strip_dirs()
    dump_path = os.path.join(OUTPUT_DIR, "profile_xpbd.prof")
    stats.dump_stats(dump_path)
    print(f"\n{'='*80}")
    print("PROFILE: example_xpbd_coupled_simple_arm (100 steps)")
    print(f"{'='*80}")
    stats.sort_stats("cumulative")
    stats.print_stats(40)
    stats.sort_stats("tottime")
    print("\n--- Top 40 by tottime ---")
    stats.print_stats(40)
    return dump_path


TARGETS = {
    "couple3": profile_couple3,
    "mujoco": profile_mujoco,
    "xpbd": profile_xpbd,
}

if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(TARGETS.keys())
    for name in targets:
        if name not in TARGETS:
            print(f"Unknown target: {name}. Choose from: {list(TARGETS.keys())}")
            sys.exit(1)
    for name in targets:
        print(f"\n>>> Profiling {name} ...")
        path = TARGETS[name]()
        print(f"    Saved: {path}")
