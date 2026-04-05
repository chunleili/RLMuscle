"""OpenSim forward-dynamics sliding-ball with DGF or Millard muscle.

Builds a 1-DOF model (SliderJoint along Y) with a muscle hanging from
the ceiling. Runs forward simulation with prescribed excitation via Manager.

Works with pyopensim (pip install pyopensim) — no conda required.
"""

import numpy as np

from scripts.osim_compat import import_opensim


def osim_sliding_ball(muscle_length, ball_mass, sigma0, muscle_radius,
                      excitation_func, t_end, dt=0.001, hill_model_type="millard"):
    """Build OpenSim 1-DOF model and run forward dynamics.

    Args:
        muscle_length: Optimal fiber length [m].
        ball_mass: Ball mass [kg].
        sigma0: Peak isometric stress [Pa].
        muscle_radius: Muscle cross-section radius [m] (for PCSA).
        excitation_func: callable(t) -> excitation [0,1].
        t_end: Simulation end time [s].
        dt: Report interval [s] (integrator uses adaptive stepping).
        hill_model_type: "dgf" or "millard".

    Returns:
        dict with 'times', 'positions', 'forces', 'norm_fiber_lengths',
        'active_forces', 'passive_forces', 'max_iso_force',
        or None if opensim unavailable.
    """
    osim = import_opensim()
    if osim is None:
        return None

    label = hill_model_type.upper()
    pcsa = np.pi * muscle_radius ** 2
    max_isometric_force = sigma0 * pcsa

    # --- Build model ---
    model = osim.Model()
    model.setName(f"sliding_ball_{hill_model_type}")
    model.set_gravity(osim.Vec3(0, -9.81, 0))

    body = osim.Body("ball", ball_mass, osim.Vec3(0), osim.Inertia(0.001))
    model.addBody(body)

    joint = osim.SliderJoint(
        "slider",
        model.getGround(), osim.Vec3(0), osim.Vec3(0, 0, np.pi / 2),
        body, osim.Vec3(0), osim.Vec3(0, 0, np.pi / 2))
    coord = joint.updCoordinate()
    coord.setName("height")
    coord.setDefaultValue(0.0)
    coord.setRangeMin(-1.0)
    coord.setRangeMax(1.0)
    coord.set_clamped(False)
    model.addJoint(joint)

    # Create muscle based on type
    if hill_model_type == "dgf":
        muscle = osim.DeGrooteFregly2016Muscle()
        muscle.setName("muscle")
        muscle.set_max_isometric_force(max_isometric_force)
        muscle.set_optimal_fiber_length(muscle_length)
        muscle.set_tendon_slack_length(0.001)
        muscle.set_pennation_angle_at_optimal(0.0)
        muscle.set_fiber_damping(0.0)
        muscle.set_max_contraction_velocity(10.0)
        muscle.set_ignore_tendon_compliance(True)
        muscle.set_ignore_activation_dynamics(False)
        muscle.set_ignore_passive_fiber_force(True)
        downcast_cls = osim.DeGrooteFregly2016Muscle
    else:
        muscle = osim.Millard2012EquilibriumMuscle()
        muscle.setName("muscle")
        muscle.set_max_isometric_force(max_isometric_force)
        muscle.set_optimal_fiber_length(muscle_length)
        muscle.set_tendon_slack_length(0.001)
        muscle.set_pennation_angle_at_optimal(0.0)
        muscle.set_fiber_damping(0.001)
        muscle.set_max_contraction_velocity(10.0)
        muscle.set_ignore_tendon_compliance(True)
        muscle.set_ignore_activation_dynamics(False)
        downcast_cls = osim.Millard2012EquilibriumMuscle

    muscle.addNewPathPoint("origin", model.updGround(),
                           osim.Vec3(0, muscle_length, 0))
    muscle.addNewPathPoint("insertion", body, osim.Vec3(0, 0, 0))
    model.addForce(muscle)

    # Prescribed excitation as PiecewiseLinearFunction
    exc_func = osim.PiecewiseLinearFunction()
    n_samples = max(int(t_end / dt) + 1, 2)
    for i in range(n_samples):
        t = min(i * dt, t_end)
        exc_func.addPoint(t, float(excitation_func(t)))

    controller = osim.PrescribedController()
    controller.setName("excitation_controller")
    controller.addActuator(muscle)
    controller.prescribeControlForActuator("muscle", exc_func)
    model.addController(controller)

    body.attachGeometry(osim.Sphere(0.05))
    model.finalizeConnections()

    import os
    os.makedirs("output", exist_ok=True)
    osim_path = f"output/sliding_ball_{hill_model_type}.osim"
    model.printToXML(osim_path)
    print(f"Wrote {osim_path}")

    # --- Forward simulation ---
    state = model.initSystem()
    manager = osim.Manager(model)
    manager.setIntegratorAccuracy(1e-6)
    manager.initialize(state)

    print(f"[{label}] Forward sim: F_max={max_isometric_force:.1f}N, "
          f"L_opt={muscle_length:.3f}m, ball={ball_mass}kg, t_end={t_end}s")

    state = manager.integrate(t_end)

    # --- Extract results from states table ---
    states_table = manager.getStatesTable()
    col_labels = list(states_table.getColumnLabels())
    n_rows = states_table.getNumRows()

    msl = downcast_cls.safeDownCast(model.getForceSet().get("muscle"))

    times, positions, forces = [], [], []
    norm_fiber_lengths, active_forces, passive_forces = [], [], []

    state = model.initSystem()
    for i in range(n_rows):
        t = states_table.getIndependentColumn()[i]
        row = states_table.getRowAtIndex(i)

        state.setTime(t)
        for j, name in enumerate(col_labels):
            model.setStateVariableValue(state, name, row[j])

        model.realizeDynamics(state)

        ball_y = model.getCoordinateSet().get("height").getValue(state)
        force = abs(msl.getActuation(state))
        nfl = msl.getNormalizedFiberLength(state)
        af = msl.getActiveFiberForce(state)
        pf = msl.getPassiveFiberForce(state)

        times.append(t)
        positions.append(ball_y)
        forces.append(force)
        norm_fiber_lengths.append(nfl)
        active_forces.append(af)
        passive_forces.append(pf)

    # Subsample to ~dt intervals if integrator produced too many points
    times_arr = np.array(times)
    if len(times_arr) > 2 * n_samples:
        target_times = np.arange(0, t_end + dt / 2, dt)
        indices = np.searchsorted(times_arr, target_times, side='left')
        indices = np.clip(indices, 0, len(times_arr) - 1)
        indices = np.unique(indices)
        times = [times[i] for i in indices]
        positions = [positions[i] for i in indices]
        forces = [forces[i] for i in indices]
        norm_fiber_lengths = [norm_fiber_lengths[i] for i in indices]
        active_forces = [active_forces[i] for i in indices]
        passive_forces = [passive_forces[i] for i in indices]

    print(f"[{label}] Done: {len(times)} points, final y={positions[-1]:.4f}, "
          f"nfl={norm_fiber_lengths[-1]:.4f}")

    result = {
        'times': np.array(times),
        'positions': np.array(positions),
        'forces': np.array(forces),
        'norm_fiber_lengths': np.array(norm_fiber_lengths),
        'active_forces': np.array(active_forces),
        'passive_forces': np.array(passive_forces),
        'max_iso_force': max_isometric_force,
    }

    # Save states .sto
    sto_path = f"output/opensim_sliding_ball_{hill_model_type}.sto"
    osim.STOFileAdapter.write(states_table, sto_path)
    print(f"Wrote {sto_path} ({states_table.getNumRows()} rows)")

    return result


# Backward-compatible wrapper
def osim_sliding_ball_millard(muscle_length, ball_mass, sigma0, muscle_radius,
                               excitation_func, t_end, dt=0.001):
    return osim_sliding_ball(muscle_length, ball_mass, sigma0, muscle_radius,
                             excitation_func, t_end, dt, hill_model_type="millard")
