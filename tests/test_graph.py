"""The .thalweg.json model: validation, round trip, schema, and the pipeline on the Y phantom."""
import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from thalweg.centerlines import combine, graph_from_tree
from thalweg.graph import Points, Source, TubeGraph, json_schema
from thalweg.kernel import medial
from phantoms import tube_field, y_tree

SCHEMA = Path(__file__).parents[1] / "docs" / "format" / "thalweg-0.1.schema.json"


def y_graph(name="y"):
    m, geo = tube_field(*y_tree())
    T = medial.trace(m, geo)
    nodes, edges, pos, rad, s = graph_from_tree(T, name, m, geo, Source(), {"graph": "field"})
    return TubeGraph(structures=[s], nodes=nodes, edges=edges, points=Points(position=pos, radius=rad))


def test_round_trip(tmp_path):
    g = y_graph()
    for name in ("y.thalweg.json", "y.thalweg.json.gz"):
        g.write(tmp_path / name)
        h = TubeGraph.read(tmp_path / name)
        assert h == g
    assert np.allclose(g.edge_points(0)[0], g.nodes[g.edges[0].start_node].position)
    assert np.allclose(g.edge_points(0)[-1], g.nodes[g.edges[0].end_node].position)


def _broken(mutate):
    doc = json.loads(y_graph().dumps())
    mutate(doc)
    with pytest.raises(ValidationError):
        TubeGraph.model_validate(doc)


def _end(doc, eid=0):
    return doc["edges"][eid]["point_range"][1] - 1


@pytest.mark.parametrize("name,mutate", [
    ("range beyond the table", lambda d: d["edges"][0].__setitem__("point_range", [0, 10 ** 9])),
    ("unknown node", lambda d: d["edges"][0].__setitem__("end_node", 10 ** 6)),
    ("column length", lambda d: d["points"].__setitem__("columns", {"area": [1.0]})),
    ("ids are not positions", lambda d: d["edges"].reverse()),
    ("end sample off its node", lambda d: d["points"]["position"].__setitem__(_end(d), [9e3, 9e3, 9e3])),
    ("duplicate structure", lambda d: d["structures"].append(dict(d["structures"][0]))),
    ("bridged without a gap", lambda d: d["edges"][0].__setitem__("provenance", {"method": "bridged"})),
    ("field with a gap",
     lambda d: d["edges"][0].__setitem__("provenance", {"method": "field", "gap_mm": 1.0})),
    ("overlapping ranges",
     lambda d: d["edges"][1].__setitem__("point_range", d["edges"][0]["point_range"])),
    ("negative length", lambda d: d["edges"][0].__setitem__("length_mm", -1.0)),
    ("negative radius", lambda d: d["points"]["radius"].__setitem__(0, -3.0)),
    ("tip of degree 3",
     lambda d: [n.__setitem__("kind", "tip") for n in d["nodes"] if n["kind"] == "junction"]),
    ("version", lambda d: d.__setitem__("version", "banana")),
    ("root of another structure", lambda d: d["structures"][0].__setitem__("roots", [10 ** 5])),
])
def test_invariants(name, mutate):
    _broken(mutate)


def test_non_finite_numbers_are_refused():
    doc = json.loads(y_graph().dumps())
    doc["points"]["position"][3] = [float("nan"), 0.0, 0.0]
    with pytest.raises(ValidationError):
        TubeGraph.model_validate(doc)


def test_tree_and_cycles():
    """A cycle validates (the format allows it) but tree() refuses it, so no consumer can hang."""
    from thalweg.errors import ThalwegError
    from thalweg.measure import strahler
    g = y_graph()
    t = g.tree("y")
    assert t.root == g.structures[0].roots[0] and len(t.order) == len(g.edges)
    doc = json.loads(g.dumps())
    tips = [n["id"] for n in doc["nodes"] if n["kind"] == "tip"]
    a, b = tips[:2]
    k = len(doc["points"]["position"])
    doc["points"]["position"] += [doc["nodes"][a]["position"], doc["nodes"][b]["position"]]
    doc["points"]["radius"] += [1.0, 1.0]
    for n in doc["nodes"]:
        if n["id"] in (a, b):
            n["kind"] = "joint"
    doc["edges"].append(dict(id=len(doc["edges"]), structure="y", start_node=a, end_node=b,
                             point_range=[k, k + 2], provenance={"method": "field"},
                             length_mm=float(np.linalg.norm(np.subtract(doc["nodes"][a]["position"],
                                                                        doc["nodes"][b]["position"])))))
    cyc = TubeGraph.model_validate(doc)
    with pytest.raises(ThalwegError):
        cyc.tree("y")
    with pytest.raises(ThalwegError):
        strahler(cyc, "y")


def test_read_refuses_other_formats(tmp_path):
    p = tmp_path / "x.thalweg.json"
    p.write_text(json.dumps({"format": "other"}))
    with pytest.raises(ValueError):
        TubeGraph.read(p)
    p.write_text(json.dumps({"format": "thalweg", "version": "2.0"}))
    with pytest.raises(ValueError, match="version"):
        TubeGraph.read(p)


def test_combine_renumbers():
    a, b = y_graph("a"), y_graph("b")
    c = combine([a, b])
    assert len(c.nodes) == len(a.nodes) + len(b.nodes)
    assert len(c.points.position) == len(a.points.position) + len(b.points.position)
    for e in c.edges:
        assert c.nodes[e.start_node].structure == e.structure == c.nodes[e.end_node].structure
    assert np.allclose(c.edge_points(len(a.edges)), b.edge_points(0))
    with pytest.raises(ValueError):
        combine([a, a])


def test_schema_file_is_current():
    """docs/format/thalweg-0.1.schema.json is generated: `thalweg schema -o` rewrites it."""
    assert json.loads(SCHEMA.read_text()) == json_schema()


def test_structure_helpers():
    """structure_edges / point_rows keep to one structure; edge_segments gives each segment's
    length, midpoint and mean end radius, with no radius where either end has none."""
    from thalweg.graph import Edge, Node, Points, Provenance, Structure
    pts = [(0.0, 0, 0), (0, 0, 2), (0, 0, 6), (5.0, 0, 0), (5, 0, 3)]
    g = TubeGraph(
        structures=[Structure(name="a", roots=[0], method="t"), Structure(name="b", roots=[2], method="t")],
        nodes=[Node(id=0, kind="root", position=pts[0], structure="a"),
               Node(id=1, kind="tip", position=pts[2], structure="a"),
               Node(id=2, kind="root", position=pts[3], structure="b"),
               Node(id=3, kind="tip", position=pts[4], structure="b")],
        edges=[Edge(id=0, structure="a", start_node=0, end_node=1, point_range=(0, 3), length_mm=6.0,
                    provenance=Provenance(method="field")),
               Edge(id=1, structure="b", start_node=2, end_node=3, point_range=(3, 5), length_mm=3.0,
                    provenance=Provenance(method="field"))],
        points=Points(position=pts, radius=[3.0, -1.0, 1.0, 2.0, 4.0]))
    assert [e.id for e in g.structure_edges("a")] == [0] and [e.id for e in g.structure_edges("b")] == [1]
    assert g.point_rows("a").tolist() == [0, 1, 2] and g.point_rows("b").tolist() == [3, 4]
    assert g.point_rows("none").tolist() == []
    seg, mid, r = g.edge_segments(0)
    assert seg.tolist() == [2.0, 4.0] and mid.tolist() == [[0, 0, 1], [0, 0, 4]]
    assert r.tolist() == [-1.0, -1.0]                          # each segment has an end without a radius
    seg, mid, r = g.edge_segments(1)
    assert seg.tolist() == [3.0] and mid.tolist() == [[5, 0, 1.5]] and r.tolist() == [3.0]
