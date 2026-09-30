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


def test_the_reported_artery_edge_holds_most_of_the_paired_samples():
    """An airway beside one artery edge for 2/3 of its length and a wider one for the rest: the
    branch is paired with the first, every sample counts, and the ratio is the median over them."""
    air = [(_line([0, 0, 0], [0, 0, 40], 41), 1.0)]
    art = [(_line([4, 0, 28], [4, 0, -1], 30), 2.0), (_line([4, 0, 28], [4, 0, 34], 7), 4.0)]
    g = _doc({"air": air, "art": art})
    rows = [dict(edge=0)]
    airway_rows(g, "air", "art", rows)
    r = rows[0]
    assert r["paired_artery_edge"] == 1                             # not the minority edge (2)
    assert r["paired_sample_count"] == 41 and r["paired_fraction"] == 1.0
    assert abs(r["bronchus_to_artery_ratio"] - 0.5) < 1e-12        # the median sample sits on edge 1


def test_part_of_a_branch_out_of_reach_lowers_the_paired_fraction():
    g = _doc({"air": [(_line([0, 0, 0], [0, 0, 40], 41), 1.0)],
              "art": [(_line([4, 0, 0], [4, 0, 20], 21), 2.0)]})
    rows = [dict(edge=0)]
    s = airway_rows(g, "air", "art", rows)
    assert 0.5 < rows[0]["paired_fraction"] < 0.75                  # 20 mm beside it, plus the reach
    assert rows[0]["paired_sample_count"] == round(rows[0]["paired_fraction"] * 41)
    assert abs(s["paired_sample_share"] - rows[0]["paired_fraction"]) < 1e-12


def test_a_missing_structure_is_an_error():
    import pytest
    from thalweg.errors import ThalwegError
    g = _doc({"air": [(_line([0, 0, 0], [0, 0, 30]), 1.0)]})
    with pytest.raises(ThalwegError):
        pair(g, "air", "art")


def test_run_pairs_airways_with_arteries_not_veins():
    from thalweg.case import PAIRS
    assert PAIRS == {"lung_airways": "lung_arteries"}


def _y_doc(trees):
    """trees: {name: (root, fork, left tip, right tip, radius)}. Edge ids per tree, in order:
    stem, left, right."""
    nodes, edges, pos, rad, structures = [], [], [], [], []
    for name, (root, fork, left, right, r) in trees.items():
        n0 = len(nodes)
        for k, (kind, q) in enumerate([("root", root), ("junction", fork), ("tip", left), ("tip", right)]):
            nodes.append(Node(id=n0 + k, kind=kind, position=tuple(map(float, q)), structure=name))
        for a, b in ((0, 1), (1, 2), (1, 3)):
            p = _line(nodes[n0 + a].position, nodes[n0 + b].position, 21)
            at = len(pos)
            pos.extend(map(tuple, p))
            rad.extend([r] * len(p))
            edges.append(Edge(id=len(edges), structure=name, start_node=n0 + a, end_node=n0 + b,
                              point_range=(at, at + len(p)),
                              length_mm=float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()),
                              provenance=Provenance(method="field")))
        structures.append(Structure(name=name, roots=[n0], method="test"))
    return TubeGraph(structures=structures, nodes=nodes, edges=edges, points=Points(position=pos, radius=rad))


def test_a_pairing_that_follows_both_trees_is_consistent():
    """An airway Y beside an artery Y, 3 mm apart: stem with stem, each daughter with its own."""
    air = ([0, 0, 0], [0, 0, 20], [-10, 0, 35], [10, 0, 35], 1.0)
    art = ([0, 3, 0], [0, 3, 20], [-10, 3, 35], [10, 3, 35], 2.0)
    g = _y_doc({"air": air, "art": art})
    rows = [dict(edge=k) for k in (0, 1, 2)]
    s = airway_rows(g, "air", "art", rows)
    assert [r["paired_artery_edge"] for r in rows] == [3, 4, 5]
    assert [r["paired_artery_consistent"] for r in rows] == [True, True, True]
    assert s["consistent_paired_branches"] == 3 and s["consistent_bronchus_to_artery_ratio_median"] == 0.5


def test_siblings_on_one_artery_edge_and_a_partner_upstream_are_not_consistent():
    from thalweg.pairing import consistency
    air = ([0, 0, 0], [0, 0, 20], [-10, 0, 35], [10, 0, 35], 1.0)
    art = ([0, 3, 0], [0, 3, 20], [-10, 3, 35], [10, 3, 35], 2.0)
    g = _y_doc({"air": air, "art": art})
    # both daughters on the artery's left daughter: shared
    assert consistency(g, "air", "art", {0: 3, 1: 4, 2: 4}) == {0: True, 1: False, 2: False}
    # the stem paired with a daughter, a daughter with the stem: upstream of its parent's partner
    assert consistency(g, "air", "art", {0: 4, 1: 3, 2: None}) == {0: True, 1: False, 2: None}
    # an unpaired stem: the daughters are checked against each other only
    assert consistency(g, "air", "art", {0: None, 1: 4, 2: 5}) == {0: None, 1: True, 2: True}
    # a daughter still beside the parent's artery edge (the artery forks later) is consistent
    assert consistency(g, "air", "art", {0: 3, 1: 3, 2: 5}) == {0: True, 1: True, 2: True}
