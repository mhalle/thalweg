"""Lobes per branch: synthetic lobe fields (always) and both stores' naming (data)."""
import numpy as np
import pytest
from rankfield.geometry import Geometry

from thalweg.centerlines import graph_from_tree
from thalweg.graph import Points, Source, TubeGraph
from thalweg.kernel import medial
from thalweg.lobes import LOBES, LobeFields, annotate, lobe_rows, lobe_summary, lobe_volumes
from phantoms import tube_field, y_tree


def _y():
    m, geo = tube_field(*y_tree())
    T = medial.trace(m, geo)
    nodes, edges, pos, rad, s = graph_from_tree(T, "y", m, geo, Source(), {"connectivity": "field"})
    return TubeGraph(structures=[s], nodes=nodes, edges=edges, points=Points(position=pos, radius=rad))


def _halves(g):
    """Five lobe margins on a 1 mm grid: lobe 1 is x < 0 above z = 20, lobe 5 is x > 0 above it,
    the other three are empty; below z = 20 is outside every lobe (a 'hilum')."""
    P = g.positions()
    lo, hi = P.min(0) - 5, P.max(0) + 5
    shape = tuple(int(np.ceil(b - a)) + 1 for a, b in zip(lo, hi))
    geo = Geometry(shape=shape, directions=((1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0)), origin=tuple(lo))
    X = np.stack(np.meshgrid(*[lo[a] + np.arange(shape[a]) for a in range(3)], indexing="ij"), -1)
    above = X[..., 2] > 20
    ms = np.full((5,) + shape, -8.0, np.float32)
    ms[0][above & (X[..., 0] < 0)] = 8.0
    ms[4][above & (X[..., 0] >= 0)] = 8.0
    return LobeFields(ms, geo, 0, "name")


def test_edges_take_the_lobe_of_most_of_their_length():
    g = _y()
    F = _halves(g)
    h, el = annotate(g, F)
    assert len(h.points.columns["lobe"]) == len(h.points.position)
    lobes = {}
    for e in h.edges:
        mid = h.edge_points(e)[len(h.edge_points(e)) // 2]
        lobes[e.id] = el["y"][e.id][0]
        if mid[2] < 18:
            assert lobes[e.id] == 0                               # the trunk runs below the 'lobes'
        elif mid[2] > 25:
            assert lobes[e.id] == (1 if mid[0] < 0 else 5)
    rows = [dict(edge=e.id) for e in h.edges]
    lobe_rows(rows, el["y"])
    assert {r["lobe"] for r in rows} <= {None, LOBES[1], LOBES[5]}
    s = lobe_summary(h, "y", el["y"], lobe_volumes(F))
    assert s["lung_upper_lobe_right"]["edges"] == 0 and s[LOBES[1]]["lobe_volume_ml"] > 0
    assert sum(v["edges"] for v in s.values()) == len(h.edges)


@pytest.mark.data
@pytest.mark.parametrize("run,named_by", [("MSB-02664_ctape0625_v013", "name"),
                                          ("C3N-00704_ctpa0625", "value (misnamed store)")])
def test_store_lobes(vessels_data, run, named_by):
    from thalweg.lobes import lobe_fields
    from thalweg.store import open_store
    F = lobe_fields(open_store(vessels_data / "runs" / f"{run}.lung_vessels.duckn.zip"))
    assert F.named_by == named_by and F.margins.shape[0] == 5
    vol = lobe_volumes(F)
    assert all(v > 50e3 for v in vol.values())                    # every lobe over 50 ml
