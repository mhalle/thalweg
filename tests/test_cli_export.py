"""The ``thalweg export`` verb: every output on the Y phantom (the store stood in for), its errors,
and a data-marked scale run on the case's airways."""
import json
import time
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from click.testing import CliRunner

from thalweg.centerlines import graph_from_tree
from thalweg.cli import main
from thalweg.graph import Node, Points, Source, Structure, TubeGraph
from thalweg.kernel import medial
from phantoms import tube_field, y_tree


class _Store:
    """What the verb asks of a store: the structure's margin, its geometry and its reference."""

    def __init__(self, m, geo):
        self.m, self.geo = m, geo

    def margin(self, name, part=None):
        return self.m, self.geo, None


@pytest.fixture(scope="module")
def y_phantom():
    m, geo = tube_field(*y_tree())
    T = medial.trace(m, geo)
    nodes, edges, pos, rad, s = graph_from_tree(T, "y", m, geo, Source(), {"graph": "field"})
    g = TubeGraph(structures=[s], nodes=nodes, edges=edges, points=Points(position=pos, radius=rad))
    return m, geo, g


@pytest.fixture
def files(y_phantom, tmp_path, monkeypatch):
    m, geo, g = y_phantom
    monkeypatch.setattr("thalweg.store.open_store", lambda path, **kw: _Store(m, geo))
    graph = g.write(tmp_path / "y.thalweg.json")
    store = tmp_path / "store.zip"
    store.write_bytes(b"")                                  # the argument must exist; the stand-in reads it
    return str(graph), str(store), tmp_path


def _run(*args):
    return CliRunner().invoke(main, [str(a) for a in args])


def test_export_writes_every_output(files, y_phantom):
    graph, store, d = files
    _, _, g = y_phantom
    r = _run("export", graph, store, "-s", "y", "--mesh", d / "y.vtp", "--vmtk-centerlines", d / "c.vtp",
             "--swc", d / "y.swc", "--markups", d / "y.mrk.json", "--cap-kinds", "tip,root")
    assert r.exit_code == 0, r.output
    assert "3 ends (tip/root): 3 capped" in r.output
    doc = json.loads((d / "y.vtp.boundaries.json").read_text())
    assert [b["id"] for b in doc["boundaries"]] == [0, 1, 2, 3] and doc["skipped"] == []
    piece = ET.parse(d / "c.vtp").getroot().find("PolyData/Piece")
    arrays = {a.get("Name"): a.get("type") for a in piece.iter("DataArray")}
    assert arrays["GroupIds"] == "Int32" and int(piece.get("NumberOfLines")) > 0
    rows = [ln for ln in (d / "y.swc").read_text().splitlines() if not ln.startswith("#")]
    assert len(rows) == len(g.points.position) - len(g.edges) + 1
    assert len(json.loads((d / "y.mrk.json").read_text())["markups"]) == len(g.edges)


def test_export_reports_every_end(files):
    """Default kinds: the Y's two tips (its root is an inlet, capped only when asked)."""
    graph, store, d = files
    r = _run("export", graph, store, "-s", "y", "--mesh", d / "y.vtp")
    assert r.exit_code == 0, r.output
    assert "2 ends (tip/truncated): 2 capped" in r.output
    doc = json.loads((d / "y.vtp.boundaries.json").read_text())
    assert len(doc["boundaries"]) - 1 + len(doc["skipped"]) == 2


def test_export_vmtk_exact_runs(files):
    graph, store, d = files
    r = _run("export", graph, store, "-s", "y", "--vmtk-centerlines", d / "c.vtp", "--vmtk-exact")
    assert r.exit_code == 0, r.output
    assert (d / "c.vtp").exists()


def test_export_needs_an_output(files):
    graph, store, _ = files
    r = _run("export", graph, store, "-s", "y")
    assert r.exit_code != 0 and "nothing to export" in r.output


def test_export_refuses_unknown_cap_kinds(files):
    graph, store, d = files
    r = _run("export", graph, store, "-s", "y", "--mesh", d / "y.vtp", "--cap-kinds", "tip,junction")
    assert r.exit_code != 0 and "--cap-kinds" in r.output


def test_export_of_a_root_only_structure_is_a_reported_failure(tmp_path):
    g = TubeGraph(structures=[Structure(name="r", roots=[0], method="test")],
                  nodes=[Node(id=0, kind="root", position=(0.0, 0.0, 0.0), structure="r")])
    graph = g.write(tmp_path / "r.thalweg.json")
    store = tmp_path / "store.zip"
    store.write_bytes(b"")
    r = _run("export", graph, store, "-s", "r", "--vmtk-centerlines", tmp_path / "c.vtp")
    assert r.exit_code == 1 and isinstance(r.exception, SystemExit)          # a ClickException, no trace
    lines = r.output.strip().splitlines()                    # the output's reason, then the verdict
    assert len(lines) == 2 and "not written" in lines[0] and "no edges" in lines[0]
    assert lines[1].startswith("Error: 1 of 1 outputs not written")


def test_export_keeps_going_past_an_output_it_cannot_make(files, tmp_path):
    """A single tube has no bifurcation: the sections are skipped with a notice and the other outputs
    are still written (the round-10 review found the whole export aborted)."""
    import numpy as np
    from thalweg.graph import Edge, Points, Provenance
    g = TubeGraph(structures=[Structure(name="t", roots=[0], method="test")],
                  nodes=[Node(id=0, kind="root", position=(0.0, 0.0, 0.0), structure="t"),
                         Node(id=1, kind="tip", position=(10.0, 0.0, 0.0), structure="t")],
                  edges=[Edge(id=0, structure="t", start_node=0, end_node=1, point_range=(0, 11),
                              length_mm=10.0, provenance=Provenance(method="field"))],
                  points=Points(position=[(float(x), 0.0, 0.0) for x in np.linspace(0, 10, 11)],
                                radius=[1.0] * 11))
    graph = g.write(tmp_path / "t.thalweg.json")
    _, store, _ = files
    r = _run("export", graph, store, "-s", "t", "--bifurcation-sections", tmp_path / "s.parquet",
             "--swc", tmp_path / "t.swc", "--markups", tmp_path / "t.mrk.json",
             "--zero-d", tmp_path / "t.json")
    assert r.exit_code == 0, r.output
    assert "no bifurcation" in r.output and not (tmp_path / "s.parquet").exists()
    assert all((tmp_path / f).exists() for f in ("t.swc", "t.mrk.json", "t.json"))


def test_an_output_folder_that_does_not_exist_is_refused_before_work(files, tmp_path):
    graph, store, _ = files
    r = _run("export", graph, store, "-s", "y", "--swc", tmp_path / "nope" / "x.swc")
    assert r.exit_code == 2 and "folder does not exist" in r.output


def test_export_unknown_structure_is_a_one_line_error(files):
    graph, store, d = files
    r = _run("export", graph, store, "-s", "nope", "--swc", d / "x.swc")
    assert r.exit_code == 1 and r.output.startswith("Error:") and len(r.output.strip().splitlines()) == 1


@pytest.mark.data
@pytest.mark.slow
def test_export_mesh_of_the_case_airways_within_budget(vessels_data, tmp_path):
    """The C3N airways (~130 ends) meshed, capped and checked in well under a minute."""
    from thalweg.centerlines import centerline_graph
    from thalweg.export import ends, mesh_defects
    store = vessels_data / "runs" / "C3N-00704_ctpa0625.lung_vessels.duckn.zip"
    if not store.exists():
        pytest.skip(f"no {store}")
    from thalweg.store import open_store
    g = centerline_graph(open_store(store), "lung_airways")
    graph = g.write(tmp_path / "air.thalweg.json.gz")
    t = time.time()
    r = _run("export", graph, store, "-s", "lung_airways", "--mesh", tmp_path / "air.vtp")
    took = time.time() - t
    assert r.exit_code == 0, r.output
    assert took < 30, took                                  # 5.5 s on an M2 (2026-09-30)
    doc = json.loads((tmp_path / "air.vtp.boundaries.json").read_text())
    n_ends = len(ends(g, "lung_airways"))
    assert len(doc["boundaries"]) - 1 + len(doc["skipped"]) == n_ends        # every end capped or reported
    assert len(doc["boundaries"]) - 1 > 0.7 * n_ends
    # the written file is the closed surface
    piece = ET.parse(tmp_path / "air.vtp").getroot().find("PolyData/Piece")
    V = np.array(piece.find("Points/DataArray").text.split(), float).reshape(-1, 3)
    F = np.array(piece.find("Polys/DataArray[@Name='connectivity']").text.split(), np.int64).reshape(-1, 3)
    d = mesh_defects(V, F)
    assert d["boundary_edges"] == d["nonmanifold_vertices"] == d["repeated_directed_edges"] == 0
    assert d["signed_volume_mm3"] > 0


def test_an_option_without_its_output_is_refused(files):
    """--inflow without --zero-d (and the like) did nothing, silently (the round-10 review)."""
    graph, store, d = files
    for extra, want in ((["--swc", d / "x.swc", "--inflow", "2"], "--zero-d"),
                        (["--swc", d / "x.swc", "--cap-kinds", "tip"], "--mesh"),
                        (["--swc", d / "x.swc", "--distance-spheres", "2"], "--bifurcation-sections"),
                        (["--swc", d / "x.swc", "--vmtk-exact"], "--vmtk-centerlines")):
        r = _run("export", graph, store, "-s", "y", *extra)
        assert r.exit_code == 2 and want in r.output, (extra, r.output)
