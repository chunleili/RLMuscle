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
