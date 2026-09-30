"""The batch product (phase 3): one case decoded once, traced, measured, summarized, checked (data)."""
import json

import pytest
from click.testing import CliRunner

from thalweg.cli import main

STORE = "runs/C3N-00704_ctpa0625.lung_vessels.duckn.zip"


@pytest.fixture(scope="module")
def airways(vessels_data):
    from thalweg.case import Case
    case = Case.open(vessels_data / STORE)
    return case, case.run(["lung_airways"], step=2.0)


@pytest.mark.data
@pytest.mark.slow
def test_run_is_consistent(airways):
    case, res = airways
    g, rows, summary, qc = res["graph"], res["rows"], res["summary"], res["qc"]
    assert len(rows) == len(g.edges) == summary["lung_airways"]["edge_count"]
    assert {r["edge"] for r in rows} == {e.id for e in g.edges}
    sm = summary["lung_airways"]
    ordered = sum(v["edge_count"] for v in sm["strahler_order"].values())
    assert ordered + sm["unordered"]["edge_count"] == len(g.edges)
    q = qc["structures"]["lung_airways"]
    assert q["unrefined_point_count"] == 0 and q["field_loop_count"] >= 0
    assert q["length_outside_field_mm"] < 0.01 * q["length_mm"]
    assert qc["acquisition"]["1"]["coarse_slices"] is False
    assert qc["acquisition"]["1"]["source_spacing_mm"][0] == pytest.approx(0.625)
    # one decode per class across trace, measure and qc: the airways and their wall
    assert case.decodes == 2 == len(case.margins)
    wall = sm["pi10"]
    assert wall["pi10_mm"] > 0 and wall["wall_station_count"] <= wall["station_count"] == len(res["stations"])
    # everything the batch promises is wired in
    cols = g.points.columns
    assert {"lobe_number", "radius_lower_mm", "radius_upper_mm"} <= set(cols)
    assert set(cols["lobe_number"]) <= set(range(6)) and len(set(cols["lobe_number"])) > 3
    assert all("lobe_name" in r and "lobe_length_fraction" in r for r in rows)
    ts = sm["tree_statistics"]
    assert ts["horton"]["bifurcation_ratio"] > 1.5 and 0 < ts["orientation_entropy"] <= 1
    assert "outside_lobes" in sm["lobes"]
    assert qc["artery_vein"]["plausible"] is None and "needs both" in qc["artery_vein"]["reason"]
    assert qc["lobes"]["named_by"] is not None
    st = g.structures[0].statistics
    from thalweg.centerlines import end_width
    deg = g.degree()
    widths = [end_width(g, nd.id) for nd in g.nodes if deg[nd.id] == 1]
    assert st["inlet_end_width_mm"] == pytest.approx(end_width(g, g.structures[0].roots[0]), abs=1e-4)
    assert st["widest_end_width_mm"] == pytest.approx(max(widths), abs=1e-4) and len(st["deepest_point"]) == 3
    assert {"edge_count", "node_count_by_kind", "tree_statistics", "pi10", "lobes"} <= set(sm)
    assert {"thalweg_version", "store", "acquisition", "lobes", "artery_vein", "structures"} <= set(qc)
    assert res["summary"]["lung_airways"]["coarse_slices"] is False


@pytest.mark.data
@pytest.mark.slow
def test_run_verb_writes_the_package(vessels_data, tmp_path):
    r = CliRunner().invoke(main, ["run", str(vessels_data / STORE), "-o", str(tmp_path), "-s", "lung_airways",
                                  "--step", "2", "-q", "--branch-volumes"])
    assert r.exit_code == 0, r.output
    pq = pytest.importorskip("pyarrow.parquet")
    rows = pq.read_table(tmp_path / "branches.parquet").to_pylist()
    total = json.loads((tmp_path / "summary.json").read_text())["lung_airways"]["partition_volume_mm3"]
    assert 30e3 < total < 60e3 and sum(x["volume_mm3"] for x in rows) == pytest.approx(total, abs=0.5)
    assert sum(x["volume_mm3"] > 0 for x in rows) > 0.95 * len(rows)
    # the partition covers the traced piece exactly: not the fragments the tracer dropped
    import gzip
    import numpy as np
    st = json.loads(gzip.open(tmp_path / "graph.thalweg.json.gz").read())["structures"][0]
    voxel = abs(np.linalg.det(np.array(st["source"]["grid"]["directions"])))
    assert total == pytest.approx(st["statistics"]["traced_lattice_point_count"] * voxel, abs=0.5)
    assert st["statistics"]["lattice_point_count"] > st["statistics"]["traced_lattice_point_count"]
    names = {p.name for p in tmp_path.iterdir()}
    assert names == {"graph.thalweg.json.gz", "branches.parquet", "stations.parquet", "summary.json",
                     "qc.json"}
    qc = json.loads((tmp_path / "qc.json").read_text())
    assert "lung_airways" in qc["structures"] and qc["thalweg_version"]


@pytest.mark.data
@pytest.mark.slow
def test_default_order_run_writes_the_airway_columns(vessels_data, tmp_path):
    """Arteries first, airways second (the default order): the airway-only wall and pairing columns
    must still be in branches.parquet and the wall columns in stations.parquet."""
    pq = pytest.importorskip("pyarrow.parquet")
    args = ["run", str(vessels_data / STORE), "-o", str(tmp_path), "-s", "lung_arteries",
            "-s", "lung_airways", "--step", "2", "--ridge-passes", "1", "-q"]
    r = CliRunner().invoke(main, args)
    assert r.exit_code == 0, r.output
    rows = pq.read_table(tmp_path / "branches.parquet").to_pylist()
    air = [x for x in rows if x["structure"] == "lung_airways"]
    assert any(x["wall_area_percent"] is not None for x in air)
    assert any(x["bronchus_to_artery_ratio"] is not None for x in air)
    assert all(x["wall_area_percent"] is None for x in rows if x["structure"] == "lung_arteries")
    stations = pq.read_table(tmp_path / "stations.parquet").to_pylist()
    assert any(x.get("wall_thickness_mm") is not None for x in stations)


def test_missing_store_is_a_clear_error(tmp_path):
    r = CliRunner().invoke(main, ["centerlines", str(tmp_path), "-s", "x",
                                  "-o", str(tmp_path / "o.thalweg.json")])
    assert r.exit_code != 0 and "no store" in r.output.lower()


@pytest.mark.data
@pytest.mark.slow
def test_the_other_verbs_run_end_to_end(vessels_data, tmp_path):
    """centerlines (not quiet: its log reads the statistics), summary, table and export with wall
    maps, on the airways."""
    import numpy as np
    graph, maps = tmp_path / "a.thalweg.json.gz", tmp_path / "w.npz"
    run = CliRunner()
    r = run.invoke(main, ["centerlines", str(vessels_data / STORE), "-s", "lung_airways", "-o", str(graph)])
    assert r.exit_code == 0, r.output + str(r.exception)
    r = run.invoke(main, ["summary", str(graph)])
    assert r.exit_code == 0 and "lung_airways" in r.output, r.output + str(r.exception)
    r = run.invoke(main, ["table", str(graph), str(vessels_data / STORE), "-o", str(tmp_path / "b.parquet"),
                          "--step", "3"])
    assert r.exit_code == 0, r.output + str(r.exception)
    r = run.invoke(main, ["export", str(graph), str(vessels_data / STORE), "-s", "lung_airways",
                          "--wall-maps", str(maps), "--wall-map-step", "1"])
    assert r.exit_code == 0, r.output + str(r.exception)
    mesh = tmp_path / "a.vtp"
    r = run.invoke(main, ["export", str(graph), str(vessels_data / STORE), "-s", "lung_airways",
                          "--mesh", str(mesh), "--flow-extensions", "3", "--extension-transition", "0.5"])
    assert r.exit_code == 0, r.output + str(r.exception)
    caps = json.loads((tmp_path / "a.vtp.boundaries.json").read_text())["boundaries"][1:]
    assert len(caps) > 50
    for c in caps:
        assert c["extension_length_mm"] == pytest.approx(3 * c["extension_radius_mm"], abs=1e-5)
        assert c["extension_transition"] == 0.5 and c["extension_vertices_inside_structure"] >= 0
        assert 0.8 * c["extension_radius_mm"] < c["ring_mean_radius_mm"] <= c["extension_radius_mm"] + 1e-6
    r = run.invoke(main, ["export", str(graph), str(vessels_data / STORE), "-s", "lung_airways",
                          "--mesh", str(tmp_path / "b.vtp"), "--curvature", "--distance-to-centerlines",
                          "--flow-extensions", "2"])
    assert r.exit_code == 0, r.output + str(r.exception)
    import xml.etree.ElementTree as ET
    arrays = ET.parse(tmp_path / "b.vtp").getroot().find("PolyData/Piece/PointData").iter("DataArray")
    pdata = {a.get("Name"): np.array(a.text.split(), float) for a in arrays}
    assert set(pdata) == {"MeanCurvature", "DistanceToCenterlines", "CenterlineRadius"}
    h = pdata["MeanCurvature"]
    assert 0.5 < np.isfinite(h).mean() < 1.0 and 0.1 < np.nanmedian(h) < 0.5      # NaN on the extensions
    assert np.isfinite(pdata["DistanceToCenterlines"]).all()
    r = run.invoke(main, ["export", str(graph), str(vessels_data / STORE), "-s", "lung_airways",
                          "--swc", str(tmp_path / "a.swc"), "--flow-extensions", "3"])
    assert r.exit_code != 0 and "need --mesh" in r.output
    r = run.invoke(main, ["export", str(graph), str(vessels_data / STORE), "-s", "lung_airways",
                          "--mesh", str(mesh), "--extension-transition", "0.5"])
    assert r.exit_code != 0 and "needs --flow-extensions" in r.output
    r = run.invoke(main, ["export", str(graph), str(vessels_data / STORE), "-s", "lung_airways",
                          "--bifurcation-sections", str(tmp_path / "s.parquet"), "--distance-spheres", "2"])
    assert r.exit_code == 0, r.output + str(r.exception)
    pq = pytest.importorskip("pyarrow.parquet")
    sections = pq.read_table(tmp_path / "s.parquet").to_pylist()
    assert len(sections) > 100 and all(x["distance_spheres"] == 2 for x in sections)
    assert sum(x["closed"] for x in sections) > 0.7 * len(sections)
    z = np.load(maps)
    assert len(z["edges"]) > 100 and len(z["angle_rad"]) == 72
    steps = np.concatenate([np.diff(z[f"edge_{int(e)}_arc_length_mm"]) for e in z["edges"]])
    assert abs(np.median(steps) - 1.0) < 0.05                          # --wall-map-step 1
    k = int(z["edges"][0])
    assert z[f"edge_{k}_radius_mm"].shape[1] == 72 and np.isfinite(z[f"edge_{k}_radius_mm"]).any()
