"""Tree statistics on synthetic graphs with known answers, and the per-point radius interval."""
import numpy as np

from thalweg.graph import Edge, Node, Points, Provenance, Structure, TubeGraph
from thalweg.statistics import horton, orientation_entropy, small_vessel_volume_fraction, streams


def binary_tree(depth=4, length=10.0, radius=lambda d: 4.0 / 1.5 ** d):
    """A perfect binary tree: every edge `length` long, radius shrinking by 1.5 per level."""
    nodes = [Node(id=0, kind="root", position=(0.0, 0.0, 0.0), structure="t")]
    edges, pos, rad = [], [], []
    frontier = [(0, np.zeros(3), 0)]
    rng = np.random.default_rng(0)
    while frontier:
        n, p, d = frontier.pop()
        kids = 1 if d == 0 else 2
        if d == depth:
            continue
        for _ in range(kids):
            u = rng.normal(size=3)
            q = p + length * u / np.linalg.norm(u)
            m = len(nodes)
            kind = "tip" if d + 1 == depth else "junction"
            nodes.append(Node(id=m, kind=kind, position=tuple(q), structure="t"))
            line = np.linspace(p, q, 11)
            a = len(pos)
            pos.extend(map(tuple, line))
            rad.extend([radius(d)] * 11)
            edges.append(Edge(id=len(edges), structure="t", start_node=n, end_node=m, point_range=(a, a + 11),
                              length_mm=float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum()),
                              provenance=Provenance(method="field")))
            frontier.append((m, q, d + 1))
    return TubeGraph(structures=[Structure(name="t", roots=[0], method="test")], nodes=nodes, edges=edges,
                     points=Points(position=pos, radius=rad))


def test_horton_on_a_perfect_binary_tree():
    g = binary_tree(depth=5)
    h = horton(g, "t")
    counts = [h["orders"][k]["streams"] for k in sorted(h["orders"])]
    assert counts == [16, 8, 4, 2, 1]                      # each level is its own order
    assert abs(h["bifurcation_ratio"] - 2.0) < 1e-9
    assert abs(h["diameter_ratio"] - 1.5) < 0.06           # the top stream spans two levels
    assert sum(len(s["edges"]) for s in streams(g, "t")) == len(g.edges)


def test_small_vessel_fraction():
    g = binary_tree(depth=3, radius=lambda d: [3.0, 2.0, 1.0][d])      # areas 28.3, 12.6, 3.1 mm^2
    s = small_vessel_volume_fraction(g, "t")
    vol = [1 * np.pi * 9 * 10, 2 * np.pi * 4 * 10, 4 * np.pi * 1 * 10]
    assert abs(s["small_vessel_volume_fraction"] - vol[2] / sum(vol)) < 1e-3
    assert abs(s["volume_mm3"] - sum(vol)) < 0.1


def test_orientation_entropy_extremes():
    one_way = binary_tree(depth=1)
    assert orientation_entropy(one_way, "t") == 0.0
    spread = binary_tree(depth=9, length=5.0)              # 511 random directions
    assert orientation_entropy(spread, "t") > 0.9


def test_radius_interval_brackets_the_radius():
    """On the Y phantom the +-2 logit interval is +-2/slope of radius, about 0.19 mm each way."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from phantoms import SLOPE, tube_field, y_tree
    from thalweg.centerlines import graph_from_tree, radius_interval
    from thalweg.graph import Source
    from thalweg.kernel import medial
    m, geo = tube_field(*y_tree())
    T = medial.trace(m, geo)
    nodes, edges, pos, rad, s = graph_from_tree(T, "y", m, geo, Source(), {"connectivity": "field"})
    g = TubeGraph(structures=[s], nodes=nodes, edges=edges, points=Points(position=pos, radius=rad))
    h = radius_interval(g, "y", m, geo)
    r = h.radii()
    lo = np.array(h.points.columns["radius_lower_mm"], float)
    hi = np.array(h.points.columns["radius_upper_mm"], float)
    ok = r > 0
    assert (lo[ok] <= r[ok]).all() and (hi[ok] >= r[ok]).all()
    assert abs(np.median((hi - lo)[ok]) / 2 - 2.0 / SLOPE) < 0.03
