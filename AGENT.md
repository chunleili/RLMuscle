# RL Muscle Demo Plan (Minimal)

1. Add a runnable `examples/example_minimal_couple.py` based on `example_minimal_joint.py`.
2. Load muscle tetra mesh from `data/muscle/model/bicep.geo` and build it into Newton model as soft particles/tets.
3. Keep articulated bones driven by `SolverFeatherstone` through `SolverMuscleBoneCoupled`.
4. Add one user-facing activation scalar (single muscle control) and map it to `control.tet_activations`.
5. Wire `main.py` to run the couple demo entry (already present) and keep changes minimal.
6. Verify syntax and provide exact run command.
