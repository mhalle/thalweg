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


def _lines(segments, radius=1.0):
    """A graph of separate straight edges from one root: [(start, end), ...]; not a valid tree of
    tubes, but enough for the direction statistics."""
    nodes = [Node(id=0, kind="root", position=tuple(map(float, segments[0][0])), structure="t")]
    edges, pos, rad = [], [], []
    for a, b in segments:
        a, b = np.asarray(a, float), np.asarray(b, float)
        nodes.append(Node(id=len(nodes), kind="tip", position=tuple(b), structure="t"))
        line = np.linspace(a, b, 3)
        line[0] = nodes[0].position
        at = len(pos)
        pos.extend(map(tuple, line))
        rad.extend([radius] * 3)
        edges.append(Edge(id=len(edges), structure="t", start_node=0, end_node=len(nodes) - 1,
                          point_range=(at, at + 3),
                          length_mm=float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum()),
                          provenance=Provenance(method="field")))
    return TubeGraph(structures=[Structure(name="t", roots=[0], method="test")], nodes=nodes, edges=edges,
                     points=Points(position=pos, radius=rad))


def test_orientation_is_undirected_on_every_axis():
    """A line and its reverse are one direction: entropy 0, also for the axis-aligned cases where
    the fold's deciding component is exactly zero."""
    for axis in np.eye(3):
        g = _lines([(np.zeros(3), 10 * axis), (np.zeros(3), -10 * axis)])
        assert orientation_entropy(g, "t") == 0.0, axis
    u = np.array([0.3, -0.5, 0.2])
    assert orientation_entropy(_lines([(np.zeros(3), u), (np.zeros(3), -u)]), "t") == 0.0


def test_orientation_bins_have_equal_area_and_weigh_by_length():
    """Evenly spread directions fill the bins evenly (entropy near 1: bins uniform in polar angle
    would not), and a long edge outweighs many short ones."""
    from thalweg.statistics import ORIENTATION_BINS
    rng = np.random.default_rng(1)
    u = rng.normal(size=(20000, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    even = orientation_entropy(_lines([(np.zeros(3), v) for v in u]), "t")
    assert even > 0.9995, even
    # two directions: equal lengths give log 2; 99:1 by length gives far less
    a, b = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
    equal = orientation_entropy(_lines([(np.zeros(3), a), (np.zeros(3), b)]), "t")
    assert abs(equal - np.log(2) / np.log(ORIENTATION_BINS)) < 1e-9
    skew = orientation_entropy(_lines([(np.zeros(3), 99 * a), (np.zeros(3), b)]), "t")
    assert skew < 0.2 * equal


def _tree(spec):
    """spec: [(start node, end node, end kind, end position, radius)], node 0 the root at the
    origin; straight edges."""
    nodes = {0: Node(id=0, kind="root", position=(0.0, 0.0, 0.0), structure="t")}
    edges, pos, rad = [], [], []
    for a, b, kind, q, r in spec:
        nodes[b] = Node(id=b, kind=kind, position=tuple(map(float, q)), structure="t")
        line = np.linspace(nodes[a].position, nodes[b].position, 5)
        at = len(pos)
        pos.extend(map(tuple, line))
        rad.extend([r] * 5)
        edges.append(Edge(id=len(edges), structure="t", start_node=a, end_node=b, point_range=(at, at + 5),
                          length_mm=float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum()),
                          provenance=Provenance(method="field")))
    return TubeGraph(structures=[Structure(name="t", roots=[0], method="test")],
                     nodes=[nodes[k] for k in sorted(nodes)], edges=edges,
                     points=Points(position=pos, radius=rad))


def test_a_stream_chains_edges_of_one_order():
    """A trunk that sheds a twig and goes on: its two edges (order 2) are one stream, 20 mm long,
    with the length-weighted mean diameter; the three tips are order-1 streams."""
    g = _tree([(0, 1, "junction", (0, 0, 10), 3.0), (1, 2, "tip", (8, 0, 10), 1.0),
               (1, 3, "junction", (0, 0, 20), 2.0), (3, 4, "tip", (6, 0, 26), 1.0),
               (3, 5, "tip", (-6, 0, 26), 1.0)])
    st = sorted(streams(g, "t"), key=lambda s: -s["order"])
    assert [s["order"] for s in st] == [2, 1, 1, 1]
    assert st[0]["edges"] == [0, 2] and abs(st[0]["length_mm"] - 20.0) < 1e-9
    assert abs(st[0]["diameter_mm"] - 5.0) < 1e-9                   # 2 x the mean of radii 3 and 2


def test_horton_ratios_have_their_sign_and_scale():
    """A binary tree whose edges grow 1.3x longer and 1.5x wider per level toward the root: the
    length and diameter ratios are those factors, and diameters are twice the radius."""
    spec, frontier, depth = [], [(0, np.zeros(3), 0)], 5
    rng = np.random.default_rng(3)
    while frontier:
        n, p, d = frontier.pop()
        if d == depth:
            continue
        for _ in range(1 if d == 0 else 2):
            u = rng.normal(size=3)
            q = p + 10.0 * 1.3 ** (depth - 1 - d) * u / np.linalg.norm(u)
            m = len(spec) + 1
            spec.append((n, m, "tip" if d + 1 == depth else "junction", q, 4.0 / 1.5 ** d))
            frontier.append((m, q, d + 1))
    h = horton(_tree(spec), "t")
    assert abs(h["bifurcation_ratio"] - 2.0) < 1e-9
    assert abs(h["length_ratio"] - 1.3) < 1e-6 and abs(h["diameter_ratio"] - 1.5) < 1e-6
    assert abs(h["orders"]["1"]["mean_diameter_mm"] - 2 * 4.0 / 1.5 ** 4) < 1e-3
    assert abs(h["orders"]["1"]["mean_length_mm"] - 10.0) < 1e-3


def test_radius_interval_clamps_and_zeroes_the_lower_bound_on_a_thin_tube():
    """A thin tube whose margin reaches only 1.5 logits on its axis: the +2 surface does not exist,
    so the lower bound is 0; the upper bound is never below the radius."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from phantoms import capsule_distance
    from rankfield.geometry import Geometry
    from thalweg.centerlines import graph_from_tree, radius_interval
    from thalweg.graph import Source
    from thalweg.kernel import medial
    from thalweg.kernel.field import sample
    h, shape, lo = 0.25, (33, 33, 113), np.array([-4.0, -4.0, -4.0])
    geo = Geometry(shape=shape, directions=((h, 0, 0), (0, h, 0), (0, 0, h)), origin=tuple(lo))
    X = lo + h * np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1).reshape(-1, 3)
    d = capsule_distance(X, np.zeros(3), np.array([0.0, 0, 20]), 0.6, 0.6)
    m = np.clip(2.5 * d, -8, 8).astype(np.float32).reshape(shape)   # 1.5 logits on the axis
    T = medial.trace(m, geo)
    nodes, edges, pos, rad, s = graph_from_tree(T, "t", m, geo, Source(), {"connectivity": "field"})
    g = TubeGraph(structures=[s], nodes=nodes, edges=edges, points=Points(position=pos, radius=rad))
    h = radius_interval(g, "t", m, geo)
    lo = np.array(h.points.columns["radius_lower_mm"], float)
    hi = np.array(h.points.columns["radius_upper_mm"], float)
    here = sample(m, geo, h.positions())
    assert (here <= 2.0).all() and (lo == 0.0).all()
    r = h.radii()
    assert (hi[r > 0] >= r[r > 0]).all() and (lo <= np.maximum(r, 0)).all()
