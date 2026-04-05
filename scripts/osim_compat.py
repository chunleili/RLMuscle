"""Shared OpenSim import utility for comparison scripts."""


def import_opensim():
    """Import opensim, supporting both conda opensim and pip pyopensim.

    pyopensim puts some classes in submodules (simulation, actuators, common).
    We patch them onto the top-level module for a uniform API.
    Returns None if neither package is available.
    """
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
        ("SliderJoint", "simulation"),
        ("Sphere", "simulation"),
        ("DeGrooteFregly2016Muscle", "actuators"),
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
