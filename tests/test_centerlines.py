"""The pipeline's tracer wrapper: truncated ends, source checks, errors."""
import numpy as np
import pytest

from thalweg.centerlines import _on_boundary, check_source, graph_from_tree
from thalweg.errors import ThalwegError
from thalweg.graph import Points, Source, TubeGraph
from thalweg.kernel import medial
from phantoms import tube_field


def _graph(m, geo, source=None):
    T = medial.trace(m, geo)
    nodes, edges, pos, rad, s = graph_from_tree(T, "t", m, geo, source or Source(), {"connectivity": "field"})
    return TubeGraph(structures=[s], nodes=nodes, edges=edges, points=Points(position=pos, radius=rad))


def _cut_far_x(m, geo, x_max):
    """The field with everything beyond world x = x_max cut off (the grid ends there)."""
    d = np.asarray(geo.directions, float)
    ax = int(np.argmax(np.abs(d[:, 0])))                           # the array axis that runs along x
    x = np.asarray(geo.origin, float)[0] + np.arange(m.shape[ax]) * d[ax, 0]
    keep = np.nonzero(x <= x_max)[0]
    return np.take(m, keep, axis=ax).copy() if keep[0] == 0 else None


def test_a_tube_off_the_grid_has_a_truncated_end():
    """A tube running out of the field's box: that end is truncated, the capped one a tip."""
    m, geo = tube_field([np.array([[0.0, 0, 0], [0, 0, 30]])], [np.array([2.0, 2.0])], pad=4.0)
    cut = m[:, :, :-12].copy()                                     # the box now ends inside the tube
    g = _graph(cut, geo)
    kinds = sorted(nd.kind for nd in g.nodes if nd.kind in ("tip", "truncated") or nd.attributes)
    assert "truncated" in kinds or any(nd.attributes.get("on_grid_boundary") for nd in g.nodes)
    ends = [nd for nd in g.nodes if nd.kind in ("tip", "truncated")]
    for nd in ends:
        near_face = nd.position[2] > 20
        assert (nd.kind == "truncated") == near_face


def test_on_boundary_is_false_inside():
    m, geo = tube_field([np.array([[0.0, 0, 0], [0, 0, 30]])], [np.array([2.0, 2.0])], pad=4.0)
    assert not _on_boundary(m, geo, [0.0, 0.0, 15.0], 2.0)


def test_check_source_refuses_another_field():
    from thalweg.graph import Grid
    m, geo = tube_field([np.array([[0.0, 0, 0], [0, 0, 30]])], [np.array([2.0, 2.0])])
    grid = Grid(shape=tuple(m.shape), directions=tuple(map(tuple, np.asarray(geo.directions))),
                origin=tuple(geo.origin))
    g = _graph(m, geo, Source(grid=grid, label_value=3))
    s = g.structures[0]
    check_source(s, geo, None)                                     # the same field passes
    from rankfield.geometry import Geometry
    other = Geometry(shape=geo.shape, directions=geo.directions, origin=tuple(np.add(geo.origin, 1.0)))
    with pytest.raises(ThalwegError):
        check_source(s, other, None)


def test_reroot_keeps_a_valid_tree():
    """Re-rooting at a daughter's tip reverses the path to it; the result is a valid, oriented tree."""
    from thalweg.centerlines import reroot
    from phantoms import y_tree
    m, geo = tube_field(*y_tree())
    g = _graph(m, geo)
    tip = next(nd.id for nd in g.nodes if nd.kind == "tip")
    h = reroot(g, "t", tip)
    TubeGraph.model_validate_json(h.dumps())                       # every invariant still holds
    t = h.tree("t")
    assert t.root == tip and len(t.order) == len(h.edges)
    assert h.nodes[tip].kind == "root" and h.nodes[tip].attributes["end_kind"] == "tip"
    old = g.structures[0].roots[0]
    assert h.nodes[old].kind in ("tip", "joint", "junction")
    assert h.structures[0].statistics["deepest_point"] == list(g.nodes[old].position)
    # the same polylines, some reversed: total length and point multiset unchanged
    assert abs(sum(e.length_mm for e in h.edges) - sum(e.length_mm for e in g.edges)) < 1e-9
    assert sorted(map(tuple, h.positions())) == sorted(map(tuple, g.positions()))


def test_inlet_prefers_an_end_off_the_field():
    """A tube running out of the field: its truncated end is the inlet, though a side branch is
    wider (r 3 against 2: the off-field end is over half as wide as the widest end)."""
    from thalweg.centerlines import inlet, reroot
    lines = [np.array([[0.0, 0, 0], [0, 0, 30]]), np.array([[0.0, 0, 15], [12.0, 0, 15]])]
    m, geo = tube_field(lines, [np.array([2.0, 2.0]), np.array([3.0, 3.0])], pad=4.0)
    cut = m[:, :, :-12].copy()                                     # the box ends inside the first tube
    g = _graph(cut, geo)
    n = inlet(g, "t")
    assert g.nodes[n].kind == "truncated" or g.nodes[n].attributes.get("on_grid_boundary")
    h = reroot(g, "t", n)
    assert h.nodes[n].kind == "root" and h.nodes[n].attributes.get("on_grid_boundary")


def test_inlet_is_not_a_thin_twig_that_runs_off_the_field():
    """A 4 mm trunk ending inside the field with a 1 mm twig cut by the grid (a field of view that
    clips a peripheral branch): the inlet is the trunk's end, not the twig's."""
    from thalweg.centerlines import end_width, inlet
    lines = [np.array([[0.0, 0, 0], [0, 0, 40]]), np.array([[0.0, 0, 20], [30.0, 0, 20]])]
    m, geo = tube_field(lines, [np.array([4.0, 4.0]), np.array([1.0, 1.0])], pad=5.0)
    g = _graph(_cut_far_x(m, geo, 22.0), geo)
    off = [nd for nd in g.nodes if nd.kind == "truncated"]
    assert off and max(end_width(g, nd.id) for nd in off) < 1.5     # the twig does run off the field
    n = inlet(g, "t")
    assert end_width(g, n) > 3.5 and g.nodes[n].kind != "truncated"


def test_reroot_reverses_radii_and_columns_with_the_points():
    """Every sample keeps its own radius and column value through a re-root, the demoted root
    forgets what it was, and the tracer's deepest point stays on record."""
    from thalweg.centerlines import reroot
    from phantoms import y_tree
    m, geo = tube_field(*y_tree())
    g = _graph(m, geo)
    n = len(g.points.position)
    g = g.model_copy(update={"points": g.points.model_copy(update={"columns": {"tag": list(range(n))}})})
    at = {}
    for p, r, t in zip(g.points.position, g.points.radius, g.points.columns["tag"]):
        at.setdefault(tuple(p), set()).add((r, t))
    tips = [nd.id for nd in g.nodes if nd.kind == "tip"]
    deepest = g.structures[0].statistics["deepest_point"]
    h = reroot(g, "t", tips[0])
    assert h.points.columns["tag"] != g.points.columns["tag"]       # something was reversed
    for p, r, t in zip(h.points.position, h.points.radius, h.points.columns["tag"]):
        assert (r, t) in at[tuple(p)]
    k = reroot(h, "t", tips[1])                                     # the first inlet is demoted
    assert "end_kind" not in k.nodes[tips[0]].attributes and k.nodes[tips[0]].kind == "tip"
    assert k.structures[0].statistics["deepest_point"] == deepest
    TubeGraph.model_validate_json(k.dumps())


@pytest.mark.data
@pytest.mark.slow
def test_artery_inlet_is_the_pulmonary_trunk(vessels_data):
    """C3N-00704: the inlet is the end of the widest terminal edge (the pulmonary trunk, r ~ 14 mm),
    not the deepest point inside it; the arteries then have one fewer tip than rooted at depth."""
    from thalweg.centerlines import centerline_graph
    from thalweg.store import open_store
    st = open_store(vessels_data / "runs" / "C3N-00704_ctpa0625.lung_vessels.duckn.zip")
    g = centerline_graph(st, "lung_arteries")
    root = g.structures[0].roots[0]
    first = next(e for e in g.edges if e.start_node == root)
    assert np.percentile(g.edge_radius(first), 75) > 10.0
    assert g.degree()[root] == 1 and g.nodes[root].attributes["end_kind"] == "tip"
    deep = centerline_graph(st, "lung_arteries", root="deepest")
    assert g.structures[0].statistics["tips"] == deep.structures[0].statistics["tips"] - 1


def _ends(trunk, a, b):
    """root -(trunk)- junction, with two ends that run off the field (radii a and b)."""
    from test_statistics import _tree
    return _tree([(0, 1, "junction", (0, 0, 20), trunk), (1, 2, "truncated", (15, 0, 20), a),
                  (1, 3, "truncated", (-15, 0, 20), b)])


@pytest.mark.parametrize("a,b,expected", [(1.9, 1.0, 0), (2.0, 1.0, 2), (2.1, 3.0, 3), (5.0, 1.0, 2)])
def test_inlet_threshold(a, b, expected):
    """A 4 mm trunk ending inside the field and two ends off it: the widest off-field end is the
    inlet from half the widest end's width up, and the trunk's end below that."""
    from thalweg.centerlines import inlet
    assert inlet(_ends(4.0, a, b), "t") == expected


def test_graph_pieces_with_offsets_record_the_deepest_point():
    from phantoms import y_tree
    m, geo = tube_field(*y_tree())
    T = medial.trace(m, geo)
    plain = graph_from_tree(T, "t", m, geo, Source(), {})
    moved = graph_from_tree(T, "t", m, geo, Source(), {}, node_offset=7, edge_offset=3, point_offset=11)
    root = next(nd for nd in moved[0] if nd.kind == "root")
    assert root.id == moved[4].roots[0] == plain[4].roots[0] + 7
    assert moved[4].statistics["deepest_point"] == list(root.position) == plain[4].statistics["deepest_point"]


def test_combine_keeps_point_columns_in_order():
    from thalweg.centerlines import combine
    from phantoms import y_tree
    m, geo = tube_field(*y_tree())
    g = _graph(m, geo)
    n = len(g.points.position)
    cols = {k: [0] * n for k in ("zeta", "alpha", "mid")}
    a = g.model_copy(update={"points": g.points.model_copy(update={"columns": cols})})
    b = g.model_copy(update={"structures": [g.structures[0].model_copy(update={"name": "u"})],
                             "nodes": [x.model_copy(update={"structure": "u"}) for x in g.nodes],
                             "edges": [x.model_copy(update={"structure": "u"}) for x in g.edges],
                             "points": g.points.model_copy(update={"columns": {"extra": [1] * n}})})
    c = combine([a, b])
    assert list(c.points.columns) == ["zeta", "alpha", "mid", "extra"]
    assert c.points.columns["extra"] == [None] * n + [1] * n
    assert c.points.columns["mid"] == [0] * n + [None] * n
