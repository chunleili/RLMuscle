"""OpenSim SimpleArm with DGF or Millard muscle.

Config-driven simulation: rigid tendon, activation dynamics.
Returns dict for comparison with MuJoCo and XPBD.

Usage:
    uv run python scripts/osim_simple_arm.py --hill-model-type dgf
    uv run python scripts/osim_simple_arm.py --hill-model-type millard
"""

import json
import os

import numpy as np

from scripts.osim_compat import import_opensim


def osim_simple_arm(cfg, hill_model_type="millard"):
    """Build and run OpenSim SimpleArm with the specified Hill model.

    Args:
        cfg: dict with keys geometry, muscle, activation, solver, initial_conditions.
        hill_model_type: "dgf" or "millard".

    Returns:
        dict with times, elbow_angles, forces, norm_fiber_lengths, activations,
        max_iso_force, muscle_type.  Or None if OpenSim unavailable.
    """
    osim = import_opensim()
    if osim is None:
        return None

    label = hill_model_type.upper()
    geo = cfg["geometry"]
    mus = cfg["muscle"]
    act_cfg = cfg["activation"]
    sol = cfg["solver"]
    ic = cfg["initial_conditions"]
    t_end = sol["n_steps"] * sol["dt"]

    # --- Build model ---
    model = osim.Model()
    model.setName(f"SimpleArm_{label}")
    model.set_gravity(osim.Vec3(0, -9.81, 0))

    humerus = osim.Body("humerus", 1.0, osim.Vec3(0), osim.Inertia(0, 0, 0))
    radius = osim.Body("radius", 1.0, osim.Vec3(0), osim.Inertia(0, 0, 0))
    model.addBody(humerus)
    model.addBody(radius)

    shoulder = osim.PinJoint(
        "shoulder",
        model.getGround(), osim.Vec3(0), osim.Vec3(0),
        humerus, osim.Vec3(0, geo["humerus_length"], 0), osim.Vec3(0))
    elbow = osim.PinJoint(
        "elbow",
        humerus, osim.Vec3(0), osim.Vec3(0),
        radius, osim.Vec3(0, geo["radius_length"], 0), osim.Vec3(0))
    model.addJoint(shoulder)
    model.addJoint(elbow)

    elbow_coord = elbow.getCoordinate()
    elbow_coord.setRangeMin(0.0)
    elbow_coord.setRangeMax(osim.SimTK_PI)
    elbow_coord.set_clamped(True)

    # Create muscle based on type
    if hill_model_type == "dgf":
        biceps = osim.DeGrooteFregly2016Muscle()
        biceps.setName("biceps")
        biceps.set_max_isometric_force(mus["max_isometric_force"])
        biceps.set_optimal_fiber_length(mus["optimal_fiber_length"])
        biceps.set_tendon_slack_length(mus["tendon_slack_length"])
        biceps.set_pennation_angle_at_optimal(0.0)
        biceps.set_fiber_damping(mus["fiber_damping"])
        biceps.set_max_contraction_velocity(mus["max_contraction_velocity"])
        biceps.set_ignore_tendon_compliance(True)
        biceps.set_ignore_activation_dynamics(False)
        downcast_cls = osim.DeGrooteFregly2016Muscle
    else:
        biceps = osim.Millard2012EquilibriumMuscle(
            "biceps",
            mus["max_isometric_force"],
            mus["optimal_fiber_length"],
            mus["tendon_slack_length"],
            0.0)
        biceps.set_ignore_tendon_compliance(True)
        biceps.set_ignore_activation_dynamics(False)
        downcast_cls = osim.Millard2012EquilibriumMuscle

    biceps.addNewPathPoint("origin", humerus,
                           osim.Vec3(*geo["muscle_origin_on_humerus"]))
    biceps.addNewPathPoint("insertion", radius,
                           osim.Vec3(*geo["muscle_insertion_on_radius"]))
    model.addForce(biceps)

    # Controller: StepFunction excitation
    brain = osim.PrescribedController()
    brain.addActuator(biceps)
    brain.prescribeControlForActuator(
        "biceps",
        osim.StepFunction(
            act_cfg["excitation_start_time"],
            act_cfg["excitation_end_time"],
            act_cfg["excitation_off"],
            act_cfg["excitation_on"]))
    model.addController(brain)

    # Display geometry
    bodyGeometry = osim.Ellipsoid(0.1, 0.5, 0.1)
    bodyGeometry.setColor(osim.Gray)
    humerusCenter = osim.PhysicalOffsetFrame()
    humerusCenter.setName("humerusCenter")
    humerusCenter.setParentFrame(humerus)
    humerusCenter.setOffsetTransform(osim.Transform(osim.Vec3(0, 0.5, 0)))
    humerus.addComponent(humerusCenter)
    humerusCenter.attachGeometry(bodyGeometry.clone())

    radiusCenter = osim.PhysicalOffsetFrame()
    radiusCenter.setName("radiusCenter")
    radiusCenter.setParentFrame(radius)
    radiusCenter.setOffsetTransform(osim.Transform(osim.Vec3(0, 0.5, 0)))
    radius.addComponent(radiusCenter)
    radiusCenter.attachGeometry(bodyGeometry.clone())

    model.finalizeConnections()

    # --- Init & simulate ---
    state = model.initSystem()
    shoulder.getCoordinate().setLocked(state, True)
    elbow.getCoordinate().setValue(state, np.radians(ic["elbow_angle_deg"]))
    model.equilibrateMuscles(state)

    manager = osim.Manager(model)
    manager.setIntegratorAccuracy(1e-6)
    state.setTime(0)
    manager.initialize(state)

    print(f"[{label}] Simulating {t_end:.1f}s, F_max={mus['max_isometric_force']:.0f}N, "
          f"L_opt={mus['optimal_fiber_length']:.2f}m")
    state = manager.integrate(t_end)

    # --- Extract results ---
    states_table = manager.getStatesTable()
    col_labels = list(states_table.getColumnLabels())
    n_rows = states_table.getNumRows()

    msl = downcast_cls.safeDownCast(model.getForceSet().get("biceps"))

    times, elbow_angles, forces = [], [], []
    norm_fiber_lengths, activations = [], []

    state = model.initSystem()
    for i in range(n_rows):
        t = states_table.getIndependentColumn()[i]
        row = states_table.getRowAtIndex(i)
        state.setTime(t)
        for j, name in enumerate(col_labels):
            model.setStateVariableValue(state, name, row[j])
        model.realizeDynamics(state)

        ea = model.getCoordinateSet().get("elbow_coord_0").getValue(state)
        force = abs(msl.getActuation(state))
        nfl = msl.getNormalizedFiberLength(state)
        act_val = msl.getActivation(state)

        times.append(t)
        elbow_angles.append(ea)
        forces.append(force)
        norm_fiber_lengths.append(nfl)
        activations.append(act_val)

    print(f"[{label}] Done: {len(times)} points, final angle={np.degrees(elbow_angles[-1]):.1f}°")

    # Save .osim and .sto
    os.makedirs("output", exist_ok=True)
    model.printToXML(f"output/SimpleArm_{label}.osim")
    sto_path = f"output/SimpleArm_{label}_states.sto"
    osim.STOFileAdapter.write(states_table, sto_path)
    with open(sto_path, "r") as f:
        lines = f.readlines()
    lines[0] = f"SimpleArm_OpenSim_{label}\n"
    with open(sto_path, "w") as f:
        f.writelines(lines)

    return {
        "times": np.array(times),
        "elbow_angles": np.array(elbow_angles),
        "forces": np.array(forces),
        "norm_fiber_lengths": np.array(norm_fiber_lengths),
        "activations": np.array(activations),
        "max_iso_force": mus["max_isometric_force"],
        "muscle_type": label,
    }


# Backward-compatible wrappers for comparison scripts
def osim_simple_arm_dgf(cfg):
    return osim_simple_arm(cfg, hill_model_type="dgf")


def osim_simple_arm_millard(cfg):
    return osim_simple_arm(cfg, hill_model_type="millard")


def load_config(path="data/simpleArm/config.json"):
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data/simpleArm/config.json")
    parser.add_argument("--hill-model-type", choices=["dgf", "millard"], default="millard")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = osim_simple_arm(cfg, hill_model_type=args.hill_model_type)
    if result:
        print(f"Final elbow angle: {np.degrees(result['elbow_angles'][-1]):.2f}°")
