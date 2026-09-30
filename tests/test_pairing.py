"""Bronchoarterial pairing on a synthetic pair of tubes."""
import numpy as np

from thalweg.graph import Edge, Node, Points, Provenance, Structure, TubeGraph
from thalweg.pairing import airway_rows, pair


def _line(a, b, n=40):
    return np.linspace(a, b, n)


def _doc(tubes):
    """tubes: {name: [(points, radius), ...]} -> a TubeGraph, one root-to-tip edge per tube."""
    nodes, edges, pos, rad, structures = [], [], [], [], []
    for name, lines in tubes.items():
        root = len(nodes)
        for k, (p, r) in enumerate(lines):
            if k == 0:
                nodes.append(Node(id=len(nodes), kind="root", position=tuple(p[0]), structure=name))
            nodes.append(Node(id=len(nodes), kind="tip", position=tuple(p[-1]), structure=name))
            a = len(pos)
            pos.extend(map(tuple, p))
            rad.extend([r] * len(p))
            edges.append(Edge(id=len(edges), structure=name, start_node=root, end_node=len(nodes) - 1,
                              point_range=(a, a + len(p)),
                              length_mm=float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()),
                              provenance=Provenance(method="field")))
        structures.append(Structure(name=name, roots=[root], method="test"))
    return TubeGraph(structures=structures, nodes=nodes, edges=edges, points=Points(position=pos, radius=rad))


def test_parallel_neighbor_pairs_and_crossing_one_does_not():
    airway = [(_line([0, 0, 0], [0, 0, 30]), 1.0)]
    # the partner runs parallel 5 mm away; a second artery edge leaves the same root, climbs away at a
    # steep angle (|cos| 0.57 to the airway), then crosses 3 mm from it, perpendicular
    crossing = np.concatenate([_line([5, 0, 0], [-3, -20, 15])[:-1], _line([-3, -20, 15], [-3, 20, 15])])
    artery = [(_line([5, 0, 0], [5, 0, 30]), 2.0), (crossing, 4.0)]
    g = _doc({"air": airway, "art": artery})
    partner, ra, rv, _ = pair(g, "air", "art")
    assert (partner == 1).all()                                  # every sample pairs with the parallel one
    rows = [dict(edge=0)]
    s = airway_rows(g, "air", "art", rows)
    assert rows[0]["paired_artery_edge"] == 1 and abs(rows[0]["bronchus_to_artery_ratio"] - 0.5) < 1e-12
    assert s["paired_branches"] == 1 and s["share_of_paired_branches_above_1"] == 0.0


def test_nothing_within_reach_leaves_the_branch_unpaired():
    g = _doc({"air": [(_line([0, 0, 0], [0, 0, 30]), 1.0)], "art": [(_line([20, 0, 0], [20, 0, 30]), 2.0)]})
    rows = [dict(edge=0)]
    s = airway_rows(g, "air", "art", rows)
    assert rows[0]["bronchus_to_artery_ratio"] is None and rows[0]["paired_fraction"] == 0.0
    assert s["paired_branches"] == 0 and s["bronchus_to_artery_ratio_median"] is None
