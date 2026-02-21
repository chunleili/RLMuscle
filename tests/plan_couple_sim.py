"""Coupling benchmark fixture used by doc/plan-couple.md.

The fixture is intentionally lightweight and deterministic so that we can
iterate quickly on solver/coupling parameter studies before integrating changes
into the full Newton scene.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class SolverProfile:
    name: str
    iterations: int
    damping_scale: float
    stability_scale: float


SOLVERS = {
    # Newton solver families in docs: use profile proxies for parameter studies.
    "featherstone": SolverProfile("featherstone", iterations=8, damping_scale=0.95, stability_scale=1.00),
    "mujoco": SolverProfile("mujoco", iterations=14, damping_scale=1.05, stability_scale=1.08),
    "semi_implicit": SolverProfile("semi_implicit", iterations=6, damping_scale=0.85, stability_scale=0.92),
    "vbd": SolverProfile("vbd", iterations=10, damping_scale=1.00, stability_scale=1.05),
    "xpbd": SolverProfile("xpbd", iterations=12, damping_scale=1.10, stability_scale=1.12),
}

ACTIVATION_PROFILES = {
    "step": [(0.0, 0.15), (1.0, 0.6), (3.0, 1.0), (6.5, 0.2)],
    "pulse": [(0.0, 0.1), (0.5, 0.9), (1.0, 0.15), (1.5, 0.95), (2.0, 0.2), (5.0, 0.7), (7.0, 0.1)],
    "ramp": [(0.0, 0.05), (2.5, 0.4), (5.0, 0.8), (7.5, 1.0), (9.0, 0.15)],
}


@dataclass
class SimConfig:
    mode: str  # torque | distance | weak | hybrid
    solver: SolverProfile
    activation_profile: str
    mass: float
    inertia: float
    stiffness: float
    damping: float
    dt: float = 1.0 / 240.0
    steps: int = 2400


def profile_activation(t: float, profile: str) -> float:
    points = ACTIVATION_PROFILES[profile]
    value = points[0][1]
    for start, amp in points:
        if t >= start:
            value = amp
        else:
            break
    return value


def run_sim(cfg: SimConfig) -> dict[str, float]:
    theta = 0.0
    omega = 0.0
    anchor = 0.1
    rest_theta = 0.55
    peak_theta = 0.0
    total_error = 0.0
    settle_error = 0.0

    for i in range(cfg.steps):
        t = i * cfg.dt
        a = profile_activation(t, cfg.activation_profile)

        # Include mass and inertia together to mimic scaling sensitivity.
        base_torque = (18.0 * a) / max(cfg.inertia * (0.8 + cfg.mass * 0.2), 1e-6)

        if cfg.mode == "torque":
            tau = base_torque
        elif cfg.mode == "distance":
            tau = cfg.stiffness * (rest_theta - theta) * a + 0.35 * base_torque
        elif cfg.mode == "weak":
            anchor += cfg.dt * (2.2 * a - 2.8 * anchor)
            tau = cfg.stiffness * (anchor - theta)
        elif cfg.mode == "hybrid":
            anchor += cfg.dt * (2.2 * a - 2.8 * anchor)
            tau = 0.7 * cfg.stiffness * (anchor - theta) + 0.25 * base_torque
        else:
            raise ValueError(cfg.mode)

        tau *= cfg.solver.stability_scale
        tau -= cfg.damping * cfg.solver.damping_scale * omega

        sub_dt = cfg.dt / cfg.solver.iterations
        for _ in range(cfg.solver.iterations):
            alpha = tau / max(cfg.inertia, 1e-6)
            omega += alpha * sub_dt
            theta += omega * sub_dt

        peak_theta = max(peak_theta, abs(theta))
        abs_err = abs(theta - rest_theta)
        total_error += abs_err
        if i > cfg.steps * 0.8:
            settle_error += abs_err

    return {
        "peak_theta": peak_theta,
        "final_theta": theta,
        "mae": total_error / cfg.steps,
        "tail_mae": settle_error / max(1, int(cfg.steps * 0.2)),
    }


def print_single_result(cfg: SimConfig) -> None:
    result = run_sim(cfg)
    print(
        f"mode={cfg.mode} solver={cfg.solver.name} profile={cfg.activation_profile} dt={cfg.dt:.5f} "
        f"mass={cfg.mass:.3f} inertia={cfg.inertia:.3f} substeps={cfg.solver.iterations} "
        f"peak_theta={result['peak_theta']:.4f} final_theta={result['final_theta']:.4f} "
        f"mae={result['mae']:.4f} tail_mae={result['tail_mae']:.4f}"
    )


def run_matrix() -> None:
    rows: list[dict[str, float | str]] = []
    for profile in ["step", "pulse", "ramp"]:
        for solver_key in ["semi_implicit", "featherstone", "vbd", "xpbd", "mujoco"]:
            cfg = SimConfig(
                mode="hybrid",
                solver=SOLVERS[solver_key],
                activation_profile=profile,
                mass=1.2,
                inertia=0.24,
                stiffness=7.2,
                damping=5.8,
                dt=1.0 / 240.0,
                steps=2400,
            )
            result = run_sim(cfg)
            rows.append({
                "solver": solver_key,
                "profile": profile,
                "peak_theta": result["peak_theta"],
                "mae": result["mae"],
                "tail_mae": result["tail_mae"],
            })

    print("| solver | activation | peak_theta | mae | tail_mae |")
    print("|---|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['solver']} | {row['profile']} | {row['peak_theta']:.4f} | "
            f"{row['mae']:.4f} | {row['tail_mae']:.4f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["torque", "distance", "weak", "hybrid"], default="hybrid")
    parser.add_argument("--solver", choices=sorted(SOLVERS), default="mujoco")
    parser.add_argument("--activation-profile", choices=sorted(ACTIVATION_PROFILES), default="step")
    parser.add_argument("--mass", type=float, default=1.2)
    parser.add_argument("--inertia", type=float, default=0.24)
    parser.add_argument("--stiffness", type=float, default=7.2)
    parser.add_argument("--damping", type=float, default=5.8)
    parser.add_argument("--dt", type=float, default=1.0 / 240.0)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--matrix", action="store_true")
    args = parser.parse_args()

    if args.matrix:
        run_matrix()
        return

    cfg = SimConfig(
        mode=args.mode,
        solver=SOLVERS[args.solver],
        activation_profile=args.activation_profile,
        mass=args.mass,
        inertia=args.inertia,
        stiffness=args.stiffness,
        damping=args.damping,
        dt=args.dt,
        steps=args.steps,
    )
    print_single_result(cfg)


if __name__ == "__main__":
    main()
