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
            assert 0.55 < el["y"][e.id][1] < 0.8                  # 20 of its 30 mm lie outside them
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
    # the numbering is anatomical (LPS: +x is the patient's left, +z is up)
    d, o = np.asarray(F.geometry.directions, float), np.asarray(F.geometry.origin, float)
    c = {k: o + np.argwhere(F.margins[k - 1] > 0).mean(0) @ d for k in LOBES}
    assert min(c[1][0], c[2][0]) > max(c[3][0], c[4][0], c[5][0])
    assert c[1][2] > c[2][2] and c[3][2] > max(c[4][2], c[5][2])    # upper lobes on top
    assert c[4][1] < c[5][1]                                          # the middle lobe is in front
    assert LOBES[1] == "lung_upper_lobe_left" and LOBES[4] == "lung_middle_lobe_right"


class _Store:
    """The least of a store that lobe_fields needs: five classes named by value, each a box."""

    def __init__(self, x_of):
        from pathlib import Path
        from types import SimpleNamespace
        self.path = Path("fake.duckn.zip")
        self.geo = Geometry(shape=(20, 8, 8), directions=((1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0)),
                            origin=(-10.0, 0.0, 0.0))
        self.x_of = x_of                                          # label value -> (x index range)
        self.structures = [SimpleNamespace(name=f"label_{v}", label_value=v, part=0) for v in x_of]

    def ref_by_name_or_value(self, name, value=None):
        from thalweg.store import TOTAL_VALUES
        return next(s for s in self.structures if s.label_value == TOTAL_VALUES[name])

    def margin(self, name, part):
        m = np.full(self.geo.shape, -8.0, np.float32)
        a, b = self.x_of[int(name.split("_")[1])]
        m[a:b] = 8.0
        return m, self.geo, None


def test_lobes_found_by_value_must_lie_like_lobes():
    """Values 10-11 (left lobes) at +x of 12-14: accepted. Mirrored: refused, the classes found by
    value are not lung lobes (or the store's left and right are swapped)."""
    from thalweg.errors import ThalwegError
    from thalweg.lobes import lobe_fields
    good = {10: (12, 16), 11: (16, 20), 12: (0, 3), 13: (3, 6), 14: (6, 9)}
    F = lobe_fields(_Store(good))
    assert F.named_by == "value (misnamed store)"
    assert F.margins[0, 13, 0, 0] > 0 and F.margins[4, 7, 0, 0] > 0   # value 10 is lobe 1, 14 lobe 5
    mirrored = {10: (0, 4), 11: (4, 8), 12: (10, 13), 13: (13, 16), 14: (16, 20)}
    with pytest.raises(ThalwegError):
        lobe_fields(_Store(mirrored))
