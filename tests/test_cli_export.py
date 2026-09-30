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
    monkeypatch.setattr("thalweg.store.open_store", lambda path: _Store(m, geo))
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


def test_export_of_a_root_only_structure_is_a_one_line_error(tmp_path):
    g = TubeGraph(structures=[Structure(name="r", roots=[0], method="test")],
                  nodes=[Node(id=0, kind="root", position=(0.0, 0.0, 0.0), structure="r")])
    graph = g.write(tmp_path / "r.thalweg.json")
    store = tmp_path / "store.zip"
    store.write_bytes(b"")
    r = _run("export", graph, store, "-s", "r", "--vmtk-centerlines", tmp_path / "c.vtp")
    assert r.exit_code == 1 and isinstance(r.exception, SystemExit)          # a ClickException, no trace
    assert r.output.startswith("Error:") and "no edges" in r.output
    assert len(r.output.strip().splitlines()) == 1


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
