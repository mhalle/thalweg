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
    assert h.nodes[tip].kind == "root" and h.nodes[tip].attributes["end"] == "tip"
    old = g.structures[0].roots[0]
    assert h.nodes[old].kind in ("tip", "joint", "junction")
    assert h.structures[0].statistics["deepest_point"] == list(g.nodes[old].position)
    # the same polylines, some reversed: total length and point multiset unchanged
    assert abs(sum(e.length_mm for e in h.edges) - sum(e.length_mm for e in g.edges)) < 1e-9
    assert sorted(map(tuple, h.positions())) == sorted(map(tuple, g.positions()))


def test_inlet_prefers_an_end_off_the_field():
    """A tube running out of the field: its truncated end is the inlet, whatever the widths."""
    from thalweg.centerlines import inlet, reroot
    lines = [np.array([[0.0, 0, 0], [0, 0, 30]]), np.array([[0.0, 0, 15], [12.0, 0, 15]])]
    m, geo = tube_field(lines, [np.array([2.0, 2.0]), np.array([3.0, 3.0])], pad=4.0)
    cut = m[:, :, :-12].copy()                                     # the box ends inside the first tube
    g = _graph(cut, geo)
    n = inlet(g, "t")
    assert g.nodes[n].kind == "truncated" or g.nodes[n].attributes.get("on_grid_boundary")
    h = reroot(g, "t", n)
    assert h.nodes[n].kind == "root" and h.nodes[n].attributes.get("on_grid_boundary")


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
    assert g.degree()[root] == 1 and g.nodes[root].attributes["end"] == "tip"
    deep = centerline_graph(st, "lung_arteries", root="deepest")
    assert g.structures[0].statistics["tips"] == deep.structures[0].statistics["tips"] - 1
