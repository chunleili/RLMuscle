"""OpenSim SimpleArm with Millard2012EquilibriumMuscle.

Config-driven simulation: rigid tendon, activation dynamics, Millard muscle.
Returns dict for comparison with DGF and MuJoCo.

Usage:
    uv run python scripts/osim_simple_arm_millard.py
    uv run python scripts/osim_simple_arm_millard.py --config data/simpleArm/config.json
"""

import json
import os
import sys

import numpy as np


def _import_opensim():
    """Import opensim, supporting both conda opensim and pip pyopensim."""
    try:
        import opensim as osim
        return osim
    except ImportError:
        pass
    try:
        import pyopensim as osim
    except ImportError:
        print("OpenSim not available — skipping.")
        return None
    for attr, sub in [
        ("PinJoint", "simulation"),
        ("Millard2012EquilibriumMuscle", "actuators"),
        ("PiecewiseLinearFunction", "common"),
        ("TimeSeriesTable", "common"),
        ("STOFileAdapter", "common"),
    ]:
        if not hasattr(osim, attr):
            mod = getattr(osim, sub, None)
            if mod and hasattr(mod, attr):
                setattr(osim, attr, getattr(mod, attr))
    return osim


def osim_simple_arm_millard(cfg):
    """Build and run OpenSim SimpleArm with Millard muscle.

    Args:
        cfg: dict with keys geometry, muscle, activation, solver, initial_conditions.

    Returns:
        dict with times, elbow_angles, forces, norm_fiber_lengths, activations,
        max_iso_force.  Or None if OpenSim unavailable.
    """
    osim = _import_opensim()
    if osim is None:
        return None

    geo = cfg["geometry"]
    mus = cfg["muscle"]
    act_cfg = cfg["activation"]
    sol = cfg["solver"]
    ic = cfg["initial_conditions"]
    t_end = sol["n_steps"] * sol["dt"]

    # --- Build model ---
    model = osim.Model()
    model.setName("SimpleArm_Millard")
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

    # Elbow limits
    elbow_coord = elbow.getCoordinate()
    elbow_coord.setRangeMin(0.0)
    elbow_coord.setRangeMax(osim.SimTK_PI)
    elbow_coord.set_clamped(True)

    # Millard muscle
    biceps = osim.Millard2012EquilibriumMuscle(
        "biceps",
        mus["max_isometric_force"],
        mus["optimal_fiber_length"],
        mus["tendon_slack_length"],
        0.0)  # pennation angle
    biceps.set_ignore_tendon_compliance(True)       # rigid tendon
    biceps.set_ignore_activation_dynamics(False)    # activation dynamics ON
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

    print(f"[Millard] Simulating {t_end:.1f}s, F_max={mus['max_isometric_force']:.0f}N, "
          f"L_opt={mus['optimal_fiber_length']:.2f}m")
    state = manager.integrate(t_end)

    # --- Extract results ---
    states_table = manager.getStatesTable()
    col_labels = list(states_table.getColumnLabels())
    n_rows = states_table.getNumRows()

    msl = osim.Millard2012EquilibriumMuscle.safeDownCast(
        model.getForceSet().get("biceps"))

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

    print(f"[Millard] Done: {len(times)} points, final angle={np.degrees(elbow_angles[-1]):.1f}°")

    # Save .osim and .sto
    os.makedirs("output", exist_ok=True)
    model.printToXML("output/SimpleArm_Millard.osim")
    sto_path = "output/SimpleArm_Millard_states.sto"
    osim.STOFileAdapter.write(states_table, sto_path)
    with open(sto_path, "r") as f:
        lines = f.readlines()
    lines[0] = "SimpleArm_OpenSim_Millard\n"
    with open(sto_path, "w") as f:
        f.writelines(lines)

    return {
        "times": np.array(times),
        "elbow_angles": np.array(elbow_angles),
        "forces": np.array(forces),
        "norm_fiber_lengths": np.array(norm_fiber_lengths),
        "activations": np.array(activations),
        "max_iso_force": mus["max_isometric_force"],
        "muscle_type": "Millard",
    }


def load_config(path="data/simpleArm/config.json"):
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data/simpleArm/config.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = osim_simple_arm_millard(cfg)
    if result:
        print(f"Final elbow angle: {np.degrees(result['elbow_angles'][-1]):.2f}°")
