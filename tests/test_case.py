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
    assert len(rows) == len(g.edges) == summary["lung_airways"]["edges"]
    assert {r["edge"] for r in rows} == {e.id for e in g.edges}
    sm = summary["lung_airways"]
    assert sum(v["edges"] for v in sm["strahler_order"].values()) + sm["unordered"]["edges"] == len(g.edges)
    q = qc["structures"]["lung_airways"]
    assert q["unrefined_points"] == 0 and q["field_loops"] >= 0 and q["outside_mm"] < 0.01 * q["length_mm"]
    assert qc["acquisition"]["1"]["coarse_slices"] is False
    assert qc["acquisition"]["1"]["source_spacing_mm"][0] == pytest.approx(0.625)
    # one decode per class across trace, measure and qc: the airways and their wall
    assert case.decodes == 2 == len(case.margins)
    wall = sm["wall"]
    assert wall["pi10_mm"] > 0 and wall["stations"] <= wall["lumen_stations"] == len(res["stations"])
    # everything the batch promises is wired in
    cols = g.points.columns
    assert {"lobe", "radius_lower_mm", "radius_upper_mm"} <= set(cols)
    assert set(cols["lobe"]) <= set(range(6)) and len(set(cols["lobe"])) > 3
    assert all("lobe" in r and "lobe_length_fraction" in r for r in rows)
    assert sm["tree"]["horton"]["bifurcation_ratio"] > 1.5 and 0 < sm["tree"]["orientation_entropy"] <= 1
    assert "outside_lobes" in sm["lobes"]
    assert qc["artery_vein"]["plausible"] is None and "needs both" in qc["artery_vein"]["reason"]
    assert qc["lobes"]["named_by"] is not None
    st = g.structures[0].statistics
    assert st["inlet_end_width_mm"] >= 0.5 * st["widest_end_width_mm"] and len(st["deepest_point"]) == 3
    assert res["summary"]["lung_airways"]["coarse_slices"] is False


@pytest.mark.data
@pytest.mark.slow
def test_run_verb_writes_the_package(vessels_data, tmp_path):
    r = CliRunner().invoke(main, ["run", str(vessels_data / STORE), "-o", str(tmp_path), "-s", "lung_airways",
                                  "--step", "2", "-q"])
    assert r.exit_code == 0, r.output
    names = {p.name for p in tmp_path.iterdir()}
    assert names == {"graph.thalweg.json.gz", "branches.parquet", "stations.parquet", "summary.json",
                     "qc.json"}
    qc = json.loads((tmp_path / "qc.json").read_text())
    assert "lung_airways" in qc["structures"]


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
