import numpy as np
import warp as wp

from newton._src.sim.graph_coloring import (
    ColoringAlgorithm,
    color_graph,
    construct_tetmesh_graph_edges,
    validate_graph_coloring,
)


def test_graph_coloring():
    from pxr import Usd, UsdGeom

    usd_stage = Usd.Stage.Open("./data/muscle/model/bicep.usd")
    tetmesh = UsdGeom.TetMesh(usd_stage.GetPrimAtPath("/character/muscle/bicep"))
    points = np.asarray(tetmesh.GetPointsAttr().Get())
    tet_indices = np.asarray(tetmesh.GetTetVertexIndicesAttr().Get()).reshape(-1, 4)

    num_nodes = len(points)
    print(f"Mesh: {num_nodes} vertices, {len(tet_indices)} tets")

    # Build graph edges from tet connectivity
    edges_np = construct_tetmesh_graph_edges(tet_indices, tet_active_mask=None)
    edge_indices = wp.array(edges_np, dtype=int, device="cpu")
    print(f"Graph: {len(edges_np)} edges")

    # Color with MCS algorithm
    color_groups = color_graph(
        num_nodes,
        edge_indices,
        balance_colors=True,
        target_max_min_color_ratio=1.1,
        algorithm=ColoringAlgorithm.MCS,
    )
    print(f"MCS coloring: {len(color_groups)} colors, sizes: {[len(g) for g in color_groups]}")

    # Validate: assign per-vertex color array and check no edge shares a color
    particle_colors = np.full(num_nodes, -1, dtype=np.int32)
    for color_idx, group in enumerate(color_groups):
        particle_colors[group] = color_idx

    assert np.all(particle_colors >= 0), "Some vertices were not colored"

    particle_colors_wp = wp.array(particle_colors, dtype=int, device="cpu")
    wp.launch(
        kernel=validate_graph_coloring,
        inputs=[edge_indices, particle_colors_wp],
        dim=edge_indices.shape[0],
        device="cpu",
    )
    print("Vertex coloring validation passed")


def build_constraint_dual_edges(tet_indices: np.ndarray) -> np.ndarray:
    """Build dual graph edges: two tets sharing a vertex are adjacent constraints."""
    from collections import defaultdict

    vertex_to_tets = defaultdict(list)
    for tet_id, tet in enumerate(tet_indices):
        for v in tet:
            vertex_to_tets[v].append(tet_id)

    edge_set = set()
    for tet_list in vertex_to_tets.values():
        for i in range(len(tet_list)):
            for j in range(i + 1, len(tet_list)):
                a, b = tet_list[i], tet_list[j]
                edge_set.add((min(a, b), max(a, b)))

    if not edge_set:
        return np.empty((0, 2), dtype=np.int32)
    return np.array(sorted(edge_set), dtype=np.int32)


def test_constraint_coloring():
    """XPBD constraint coloring: color tets so that same-color tets share no vertex."""
    from pxr import Usd, UsdGeom

    usd_stage = Usd.Stage.Open("./data/muscle/model/bicep.usd")
    tetmesh = UsdGeom.TetMesh(usd_stage.GetPrimAtPath("/character/muscle/bicep"))
    points = np.asarray(tetmesh.GetPointsAttr().Get())
    tet_indices = np.asarray(tetmesh.GetTetVertexIndicesAttr().Get()).reshape(-1, 4)

    num_tets = len(tet_indices)
    print(f"\nMesh: {len(points)} vertices, {num_tets} tets")

    # Build dual graph: nodes=tets, edges=tets sharing a vertex
    dual_edges = build_constraint_dual_edges(tet_indices)
    dual_edge_indices = wp.array(dual_edges, dtype=int, device="cpu")
    print(f"Dual graph: {len(dual_edges)} edges")

    # Color constraints
    constraint_groups = color_graph(
        num_tets,
        dual_edge_indices,
        balance_colors=True,
        target_max_min_color_ratio=1.1,
        algorithm=ColoringAlgorithm.MCS,
    )
    print(f"Constraint coloring: {len(constraint_groups)} colors, sizes: {[len(g) for g in constraint_groups]}")

    # Validate: same-color tets must not share any vertex
    for color_idx, group in enumerate(constraint_groups):
        vertices_in_group = set()
        for tet_id in group:
            tet_verts = set(tet_indices[tet_id])
            overlap = vertices_in_group & tet_verts
            assert not overlap, (
                f"Color {color_idx}: tet {tet_id} shares vertices {overlap} with another same-color tet"
            )
            vertices_in_group.update(tet_verts)

    print("Constraint coloring validation passed")


def test_mixed_constraint_coloring():
    """Test cross-type constraint coloring using build_constraint_color_groups."""
    from pxr import Usd, UsdGeom
    from VMuscle.constraints import build_constraint_color_groups, TETVOLUME, ATTACH

    usd_stage = Usd.Stage.Open("./data/muscle/model/bicep.usd")
    tetmesh = UsdGeom.TetMesh(usd_stage.GetPrimAtPath("/character/muscle/bicep"))
    points = np.asarray(tetmesh.GetPointsAttr().Get())
    tet_indices = np.asarray(tetmesh.GetTetVertexIndicesAttr().Get()).reshape(-1, 4)

    # Build mixed constraints: tet volume + fake attach on some vertices
    all_constraints = []
    for i, tet in enumerate(tet_indices[:100]):
        all_constraints.append(dict(
            type=TETVOLUME,
            pts=[int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3])],
        ))
    # Add attach constraints on vertices that overlap with tet constraints
    attach_verts = set()
    for tet in tet_indices[:100]:
        attach_verts.update(tet.tolist())
    for v in list(attach_verts)[:20]:
        all_constraints.append(dict(
            type=ATTACH,
            pts=[v, -1, 9999, -1],  # src vertex, padding, bone tgt, padding
        ))

    n_cons = len(all_constraints)
    print(f"\nMixed constraints: {n_cons} total (100 tet + {n_cons-100} attach)")

    color_groups = build_constraint_color_groups(all_constraints)
    print(f"Coloring: {len(color_groups)} colors, sizes: {[len(g) for g in color_groups]}")

    # Validate: same-color constraints share no vertex (pts >= 0)
    for color_idx, group in enumerate(color_groups):
        vertices_in_group = set()
        for cid in group:
            c = all_constraints[cid]
            verts = {v for v in c['pts'] if v >= 0}
            overlap = vertices_in_group & verts
            assert not overlap, (
                f"Color {color_idx}: constraint {cid} (type={c['type']}) "
                f"shares vertices {overlap} with another same-color constraint"
            )
            vertices_in_group.update(verts)

    print("Mixed constraint coloring validation passed")


def _run_sim(arch="cpu", nsteps=100):
    """Run taichi sim with activation ramp, return final positions.

    arch="cpu" → regular GS, arch="cuda" → colored GS (auto-detected).
    """
    from VMuscle.config import load_config
    from VMuscle.muscle_taichi import MuscleSim
    from VMuscle.muscle_common import activation_ramp

    cfg = load_config("data/muscle/config/bicep.json")
    cfg.gui = False
    cfg.nsteps = nsteps
    cfg.arch = arch
    sim = MuscleSim(cfg)
    for step in range(1, nsteps + 1):
        act = activation_ramp(step / nsteps)
        sim.activation.fill(act)
        sim.step()
    return sim.pos.to_numpy().copy()


def test_gs_vs_colored_gs():
    """Compare plain GS (cpu) vs colored GS (cuda) with activation ramp."""
    nsteps = 100
    pos_gs = _run_sim(arch="cpu", nsteps=nsteps)
    pos_cgs = _run_sim(arch="cuda", nsteps=nsteps)

    diff = np.abs(pos_gs - pos_cgs)
    max_diff = diff.max()
    mean_diff = diff.mean()
    print(f"\nGS vs Colored GS comparison ({nsteps} steps with activation ramp):")
    print(f"  Max diff:  {max_diff:.2e}")
    print(f"  Mean diff: {mean_diff:.2e}")

    assert not np.any(np.isnan(pos_gs)), "GS produced NaN"
    assert not np.any(np.isnan(pos_cgs)), "Colored GS produced NaN"
    assert np.all(np.abs(pos_gs) < 1e6), "GS positions exploded"
    assert np.all(np.abs(pos_cgs) < 1e6), "Colored GS positions exploded"
    print("Both simulations are numerically stable")


if __name__ == "__main__":
    wp.init()
    test_graph_coloring()
    test_constraint_coloring()
    test_mixed_constraint_coloring()
    test_gs_vs_colored_gs()