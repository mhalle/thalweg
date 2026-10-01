"""The branch table on the analytic Y phantom: orders, areas, angles against the truth."""
import numpy as np
import pytest

from thalweg.centerlines import graph_from_tree
from thalweg.graph import Points, Source, TubeGraph
from thalweg.kernel import medial
from thalweg.measure import branch_table, strahler, write_table
from phantoms import tube_field, y_tree

TRUE_R = {"trunk": 3.0, "left": 2.2, "right": 1.8}


@pytest.fixture(scope="module")
def y():
    lines, radii = y_tree()
    m, geo = tube_field(lines, radii)
    T = medial.trace(m, geo)
    nodes, edges, pos, rad, s = graph_from_tree(T, "y", m, geo, Source(), {"graph": "field"})
    g = TubeGraph(structures=[s], nodes=nodes, edges=edges, points=Points(position=pos, radius=rad))
    stations = []
    rows = branch_table(g, "y", m, geo, step=1.0, stations_out=stations)
    return lines, g, rows, stations


def which(g, e, lines):
    """trunk / left / right, by where the edge's middle lies."""
    mid = g.edge_points(e)[len(g.edge_points(e)) // 2]
    return "trunk" if mid[2] < lines[0][1][2] else ("left" if mid[0] < 0 else "right")


def test_strahler_of_a_y(y):
    lines, g, rows, _ = y
    order = strahler(g, "y")
    kinds = {which(g, e, lines): order[e.id] for e in g.edges}
    assert kinds == {"trunk": 2, "left": 1, "right": 1}


def test_areas_match_the_tubes(y):
    lines, g, rows, _ = y
    for r in rows:
        name = which(g, g.edges[r["edge"]], lines)
        true = np.pi * TRUE_R[name] ** 2
        assert abs(r["area_mm2"] - true) / true < 0.03, (name, r["area_mm2"], true)
        # the model's interval brackets the boundary area: +2 logits smaller, -2 larger
        assert r["area_lower_mm2"] < r["area_mm2"] < r["area_upper_mm2"]
        assert r["aspect_ratio"] > 0.97


def deg(a, b):
    return np.degrees(np.arccos(np.clip(a @ b, -1, 1)))


def test_angles_match_the_geometry(y):
    lines, g, rows, _ = y
    trunk = lines[0][1] - lines[0][0]
    trunk /= np.linalg.norm(trunk)
    d = {k: (L[1] - L[0]) / np.linalg.norm(L[1] - L[0]) for k, L in (("left", lines[1]), ("right", lines[2]))}
    for r in rows:
        name = which(g, g.edges[r["edge"]], lines)
        if name == "trunk":
            assert r["deflection_deg"] is None and r["sibling_angle_deg"] is None
            continue
        assert abs(r["deflection_deg"] - deg(d[name], trunk)) < 2.0, (name, r["deflection_deg"])
        assert abs(r["sibling_angle_deg"] - deg(d["left"], d["right"])) < 6.0


def test_stations_profile_and_parquet(y, tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    _, g, rows, stations = y
    assert sum(r["station_count"] for r in rows) == len(stations)
    write_table(rows, tmp_path / "b.parquet")
    back = pq.read_table(tmp_path / "b.parquet").to_pylist()
    assert back == rows


def _graph_of(lines, radii, name="t", **kw):
    m, geo = tube_field(lines, radii)
    T = medial.trace(m, geo, **kw)
    nodes, edges, pos, rad, s = graph_from_tree(T, name, m, geo, Source(), {"connectivity": "field"})
    return m, geo, TubeGraph(structures=[s], nodes=nodes, edges=edges,
                             points=Points(position=pos, radius=rad))


@pytest.mark.parametrize("half,expect_sibling", [(45.0, 90.0), (70.0, 140.0)])
def test_wide_angles_are_not_biased_low(half, expect_sibling):
    """Chords start outside the junction's ball: a 70-degree half-angle reads ~70, not ~59."""
    h = np.radians(half)
    top = np.array([0.0, 0, 30])
    lines = [np.array([[0, 0, 0.0], top]), np.array([top, top + 25 * np.array([np.sin(h), 0, np.cos(h)])]),
             np.array([top, top + 25 * np.array([-np.sin(h), 0, np.cos(h)])])]
    radii = [np.array([2.5, 2.5]), np.array([1.6, 1.6]), np.array([1.6, 1.6])]
    m, geo, g = _graph_of(lines, radii)
    rows = [r for r in branch_table(g, "t", m, geo) if r["deflection_deg"] is not None]
    assert len(rows) == 2
    for r in rows:
        assert abs(r["deflection_deg"] - half) < 3.0, r["deflection_deg"]
        assert abs(r["sibling_angle_deg"] - expect_sibling) < 5.0, r["sibling_angle_deg"]
        assert r["angle_reliable"]


def test_t_junction():
    top = np.array([0.0, 0, 30])
    lines = [np.array([[0, 0, 0.0], top]), np.array([top, top + [0, 0, 25.0]]),
             np.array([top, top + [25.0, 0, 0]])]
    radii = [np.array([2.5, 2.5]), np.array([2.4, 2.4]), np.array([1.5, 1.5])]
    m, geo, g = _graph_of(lines, radii)
    side = [r for r in branch_table(g, "t", m, geo)
            if r["deflection_deg"] is not None and r["deflection_deg"] > 45]
    assert len(side) == 1 and abs(side[0]["deflection_deg"] - 90.0) < 4.0, side


def _near_coincident_junctions():
    """A trunk whose two branchings are 0.5 mm apart: the tracer leaves a stub parent between them."""
    top = np.array([0.0, 0, 30])
    lines = [np.array([[0, 0, 0.0], top]),
             np.array([top, top + 22 * np.array([np.sin(np.radians(59)), 0, np.cos(np.radians(59))])]),
             np.array([top + [0, 0, 0.5], top + [0, 0, 0.5] + 22 * np.array([0, np.sin(np.radians(56)),
                                                                               np.cos(np.radians(56))])]),
             np.array([top, top + 22 * np.array([-np.sin(np.radians(51)), 0, np.cos(np.radians(51))])])]
    radii = [np.array([2.5, 2.5])] + [np.array([1.4, 1.4])] * 3
    return lines, radii


def test_short_edges_report_no_shape_and_no_sections():
    """Edges shorter than max(4 r, 3 mm) have no shape statistics; the checked set is not empty."""
    lines, radii = _near_coincident_junctions()
    m, geo, g = _graph_of(lines, radii)
    rows = branch_table(g, "t", m, geo)
    short = [r for r in rows if r["length_mm"] < max(4 * (r["radius_mean_mm"] or 0.5), 3.0)]
    assert short, "the phantom must produce a short edge"
    for r in short:
        assert not r["shape_reliable"] and r["curvature_max_per_mm"] is None and r["distance_metric"] is None
    for r in rows:
        if r["station_count"] == 0:
            assert r["area_mm2"] is None and r["aspect_ratio"] is None


def test_angles_walk_past_stub_parents():
    """Each daughter's deflection is measured against the trunk, not the 0.5 mm stub between the
    two junctions: 59, 56 and 51 degrees, within 3."""
    lines, radii = _near_coincident_junctions()
    m, geo, g = _graph_of(lines, radii)
    rows = [r for r in branch_table(g, "t", m, geo)
            if r["deflection_deg"] is not None and r["length_mm"] > 10]
    got = sorted(r["deflection_deg"] for r in rows)
    assert len(got) == 3 and np.allclose(got, [51, 56, 59], atol=3.0), got


def test_undefined_angles_are_none_not_ninety():
    lines, radii = _near_coincident_junctions()
    m, geo, g = _graph_of(lines, radii)
    for r in branch_table(g, "t", m, geo):
        for k in ("deflection_deg", "sibling_angle_deg"):
            assert r[k] is None or abs(r[k] - 90.0) > 1e-9


def test_strahler_ignores_truncated_ends():
    """A tree whose one daughter runs off the field: that edge has no order and does not count."""
    import json as _json
    lines, radii = y_tree()
    m, geo, g = _graph_of(lines, radii, "y")
    doc = _json.loads(g.dumps())
    tip = next(n for n in doc["nodes"] if n["kind"] == "tip")
    tip["kind"] = "truncated"
    h = TubeGraph.model_validate(doc)
    order = strahler(h, "y")
    into = next(e for e in h.edges if e.end_node == tip["id"])
    assert order[into.id] is None
    assert max(o for o in order.values() if o is not None) == 1          # the trunk no longer reaches 2


def _lumen_and_wall(r_in=2.0, r_out=3.0, length=30.0, spacing=0.5):
    """Three classes like a store's: lumen (d < r_in), wall (r_in < d < r_out), background; their
    margins are each logit minus the best other, clipped like a ranked store."""
    from rankfield.geometry import Geometry
    from phantoms import CLIP, SLOPE
    lo = np.array([-r_out - 4, -r_out - 4, -4.0])
    shape = (int(2 * (r_out + 4) / spacing) + 1,) * 2 + (int((length + 8) / spacing) + 1,)
    geo = Geometry(shape=shape, directions=((spacing, 0, 0), (0, spacing, 0), (0, 0, spacing)),
                   origin=tuple(lo))
    X = np.stack(np.meshgrid(*[lo[a] + np.arange(shape[a]) * spacing for a in range(3)], indexing="ij"), -1)
    d = np.hypot(X[..., 0], X[..., 1])
    along = np.clip(X[..., 2], 0, length)
    d = np.maximum(d, np.abs(X[..., 2] - along))                       # closed ends
    lumen = SLOPE * (r_in - d)
    wall = SLOPE * np.minimum(d - r_in, r_out - d)
    bg = SLOPE * (d - r_out)
    m_lumen = np.clip(lumen - np.maximum(wall, bg), -CLIP, CLIP).astype(np.float32)
    m_wall = np.clip(wall - np.maximum(lumen, bg), -CLIP, CLIP).astype(np.float32)
    return m_lumen, m_wall, geo


def test_wall_measures_on_a_lumen_in_its_wall():
    """A 2 mm lumen in a 1 mm wall: wall area pi (9 - 4) = 15.7 mm^2, WA% 55.6, thickness 1.0 mm."""
    m_lumen, m_wall, geo = _lumen_and_wall()
    T = medial.trace(m_lumen, geo)
    nodes, edges, pos, rad, s = graph_from_tree(T, "a", m_lumen, geo, Source(), {"connectivity": "field"})
    g = TubeGraph(structures=[s], nodes=nodes, edges=edges, points=Points(position=pos, radius=rad))
    stations = []
    rows = branch_table(g, "a", m_lumen, geo, outer=np.maximum(m_lumen, m_wall), stations_out=stations)
    main = max(rows, key=lambda r: r["length_mm"])
    assert main["wall_station_count"] > 10
    assert abs(main["wall_area_mm2"] - 5 * np.pi) / (5 * np.pi) < 0.05, main["wall_area_mm2"]
    assert abs(main["wall_area_percent"] - 500 / 9) < 2.0
    assert abs(main["wall_thickness_mm"] - 1.0) < 0.05
    from thalweg.measure import pi10
    p = pi10(stations)
    assert p["wall_station_count"] > 10
    # the phantom's own constant wall: sqrt(pi t (10 / pi + t)) with t = 1 mm
    assert abs(p["constant_wall_pi10_mm"] - np.sqrt(np.pi * (10 / np.pi + 1))) < 0.1
    inner = [s_["internal_perimeter_mm"] for s_ in stations if s_["wall_area_mm2"] is not None]
    assert abs(np.median(inner) - 4 * np.pi) < 0.3                     # the lumen's perimeter, not the wall's


def test_pi10_is_the_fitted_line_at_a_perimeter_of_10():
    from thalweg.measure import pi10
    st = [dict(internal_perimeter_mm=float(x), wall_area_mm2=float((1.0 + 0.3 * x) ** 2),
               wall_thickness_mm=1.2) for x in np.linspace(5, 30, 40)]
    p = pi10(st)
    assert abs(p["pi10_mm"] - 4.0) < 1e-9 and abs(p["slope"] - 0.3) < 1e-9 and p["wall_station_count"] == 40
    assert p["internal_perimeter_range_mm"] == [5.0, 30.0]
    assert abs(p["constant_wall_pi10_mm"] - np.sqrt(np.pi * 1.2 * (10 / np.pi + 1.2))) < 1e-9
    assert pi10(st[:5])["pi10_mm"] is None                             # too few stations
    same = [dict(s_, internal_perimeter_mm=12.0) for s_ in st]
    assert pi10(same)["pi10_mm"] is None                               # one perimeter: no line


def test_a_wall_needs_a_closed_outer_contour_no_smaller_than_the_lumen(monkeypatch):
    """The outer contour open (it leaves the section window: a neighbor's wall touches) or smaller
    than the lumen: no wall measure, rather than a wrong one."""
    from thalweg import measure
    lumen = dict(area=10.0, perimeter=11.0)
    for outer in (dict(area=30.0, closed=False, perimeter=20.0), dict(area=8.0, closed=True, perimeter=9.0),
                  dict(area=0.0, closed=True, perimeter=0.0)):
        monkeypatch.setattr(measure.S, "describe", lambda img, g, level, o=outer: o)
        assert measure._wall(None, None, lumen) == {k: None for k in measure.WALL_KEYS}
    closed = dict(area=30.0, closed=True, perimeter=20.0)
    monkeypatch.setattr(measure.S, "describe", lambda img, g, level: closed)
    w = measure._wall(None, None, lumen)
    assert w["wall_area_mm2"] == 20.0 and w["internal_perimeter_mm"] == 11.0
    assert abs(w["wall_thickness_mm"] - 20.0 / 15.5) < 1e-12 and abs(w["wall_area_percent"] - 200 / 3) < 1e-9


def test_table_keeps_columns_only_later_rows_have(tmp_path):
    """Airway rows come after artery rows and carry extra columns: none may be dropped."""
    pq = pytest.importorskip("pyarrow.parquet")
    rows = [dict(structure="lung_arteries", edge=0, length_mm=1.0),
            dict(structure="lung_airways", edge=1, length_mm=2.0, wall_area_mm2=3.0, paired_artery_edge=0)]
    write_table(rows, tmp_path / "t.parquet")
    back = pq.read_table(tmp_path / "t.parquet").to_pylist()
    assert back[1]["wall_area_mm2"] == 3.0 and back[1]["paired_artery_edge"] == 0
    assert back[0]["wall_area_mm2"] is None and set(back[0]) == set(back[1])


def test_a_branch_with_too_few_wall_stations_gets_no_wall_measures():
    """The same phantom sectioned every 12 mm: one or two wall stations on the branch, so its wall
    medians are withheld, though the count and the stations themselves are reported."""
    from thalweg.measure import MIN_WALL_STATIONS, WALL_KEYS
    m_lumen, m_wall, geo = _lumen_and_wall()
    T = medial.trace(m_lumen, geo)
    nodes, edges, pos, rad, s = graph_from_tree(T, "a", m_lumen, geo, Source(), {"connectivity": "field"})
    g = TubeGraph(structures=[s], nodes=nodes, edges=edges, points=Points(position=pos, radius=rad))
    stations = []
    rows = branch_table(g, "a", m_lumen, geo, step=12.0, outer=np.maximum(m_lumen, m_wall),
                        stations_out=stations)
    main = max(rows, key=lambda r: r["length_mm"])
    assert 0 < main["wall_station_count"] < MIN_WALL_STATIONS
    assert all(main[k] is None for k in WALL_KEYS) and main["area_mm2"] is not None
    assert sum(s_["wall_area_mm2"] is not None for s_ in stations) >= main["wall_station_count"]


def test_wall_medians_start_at_the_minimum_station_count():
    """Over several section spacings: a branch with exactly MIN_WALL_STATIONS wall stations has
    wall medians, one with fewer has none."""
    from thalweg.measure import MIN_WALL_STATIONS
    m_lumen, m_wall, geo = _lumen_and_wall()
    T = medial.trace(m_lumen, geo)
    nodes, edges, pos, rad, s = graph_from_tree(T, "a", m_lumen, geo, Source(), {"connectivity": "field"})
    g = TubeGraph(structures=[s], nodes=nodes, edges=edges, points=Points(position=pos, radius=rad))
    seen = {}
    for step in (5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0):
        rows = branch_table(g, "a", m_lumen, geo, step=step, outer=np.maximum(m_lumen, m_wall))
        main = max(rows, key=lambda r: r["length_mm"])
        seen[main["wall_station_count"]] = main["wall_thickness_mm"] is not None
    assert MIN_WALL_STATIONS == 3 and seen[3] is True and seen[2] is False
    assert all(has == (n >= 3) for n, has in seen.items())


def test_pi10_constant_wall_uses_the_median_thickness():
    from thalweg.measure import pi10
    st = [dict(internal_perimeter_mm=float(x), wall_area_mm2=float((1.0 + 0.3 * x) ** 2),
               wall_thickness_mm=1.0 if k < 30 else 9.0) for k, x in enumerate(np.linspace(5, 30, 40))]
    assert abs(pi10(st)["constant_wall_pi10_mm"] - np.sqrt(np.pi * (10 / np.pi + 1.0))) < 1e-9
    assert set(pi10(st[:3])) == set(pi10(st))                          # the same keys when there is no fit


def test_summary_keys():
    """The serialized names of one structure's summary (docs/format/thalweg-json.md)."""
    from thalweg.case import summarize
    from phantoms import tube_field, y_tree
    m, geo = tube_field(*y_tree())
    T = medial.trace(m, geo)
    nodes, edges, pos, rad, s = graph_from_tree(T, "y", m, geo, Source(), {"connectivity": "field"})
    g = TubeGraph(structures=[s], nodes=nodes, edges=edges, points=Points(position=pos, radius=rad))
    sm = summarize(g, "y", [dict(area_mm2=1.0), dict(area_mm2=None)])
    assert set(sm) == {"edge_count", "node_count_by_kind", "length_mm", "radius_percentiles_mm",
                       "strahler_order", "unordered", "sectioned_branch_count"}
    assert sm["edge_count"] == len(g.edges) and sm["sectioned_branch_count"] == 1
    assert sm["node_count_by_kind"] == {"root": 1, "junction": 1, "tip": 2} or sum(
        sm["node_count_by_kind"].values()) == len(g.nodes)
    assert all(set(v) == {"edge_count", "length_mm"} for v in sm["strahler_order"].values())
    assert set(sm["unordered"]) == {"edge_count", "length_mm"}
    assert set(sm["radius_percentiles_mm"]) == {"p10", "p50", "p90"}
