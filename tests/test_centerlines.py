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
