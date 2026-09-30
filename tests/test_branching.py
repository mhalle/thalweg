"""vmtk's grouping chain on thalweg's graph: the Y phantom (always) and field-to-vmtk end to end (data)."""
import numpy as np
import pytest

from thalweg.branching import annotate, vmtk_branching
from thalweg.centerlines import graph_from_tree
from thalweg.graph import Points, Source, TubeGraph
from thalweg.kernel import medial
from phantoms import tube_field, y_tree


@pytest.fixture(scope="module")
def y_graph():
    m, geo = tube_field(*y_tree())
    T = medial.trace(m, geo)
    nodes, edges, pos, rad, s = graph_from_tree(T, "y", m, geo, Source(), {"graph": "field"})
    return TubeGraph(structures=[s], nodes=nodes, edges=edges, points=Points(position=pos, radius=rad))


def test_y_groups_frame_and_angles(y_graph):
    b = vmtk_branching(y_graph, "y")
    g = b.split.cell_data["GroupIds"]
    bl = b.split.cell_data["Blanking"]
    assert len(set(g[bl == 0].tolist())) == 3 and len(set(g[bl == 1].tolist())) == 1
    h = annotate(y_graph, "y", b)
    TubeGraph.model_validate_json(h.dumps())                     # still a valid document
    junction = [nd for nd in h.nodes if nd.kind == "junction"]
    assert len(junction) == 1 and len(junction[0].attributes["bifurcation_frames"]) == 1
    f = junction[0].attributes["bifurcation_frames"][0]
    assert abs(np.linalg.norm(f["normal"]) - 1) < 1e-9 and abs(np.dot(f["normal"], f["up_normal"])) < 1e-9
    region = np.array(h.points.columns["bifurcation_region"])
    assert (region == 1).any() and (region == 0).any()
    daughters = [e for e in h.edges if e.start_node == junction[0].id]
    assert len(daughters) == 2 and all("bifurcation_vector" in e.attributes for e in daughters)
    # the phantom's daughters leave on opposite sides of the trunk axis: opposite in-plane signs
    a = [e.attributes["bifurcation_vector"]["in_plane_angle_rad"] for e in daughters]
    assert a[0] * a[1] < 0
    # vmtk's in-plane angles are each daughter's deviation from straight on; their magnitudes add up
    # to the angle between the daughters (79.2 degrees in the phantom)
    assert abs(np.degrees(abs(a[0]) + abs(a[1])) - 79.2) < 2.0
    # the daughters' vectors refer to the junction's own bifurcation
    assert {e.attributes["bifurcation_vector"]["bifurcation_group"] for e in daughters} == {f["group"]}


@pytest.mark.data
@pytest.mark.slow
def test_frames_follow_groups_on_the_case(vessels_data):
    """Every daughter's vector refers to one of its junction's bifurcations (the consistency the
    nearest-origin matching broke), and no structure without a bifurcation crashes the chain."""
    pytest.importorskip("pyarrow")
    from thalweg.centerlines import centerline_graph
    from cases import reference_source
    g = centerline_graph(vessels_data / "runs" / "C3N-00704_ctpa0625.lung_vessels.duckn.zip", "lung_arteries",
                         ridge_passes=1)                         # the graph vmtk was fed
    b = vmtk_branching(g, "lung_arteries", source=reference_source(g))
    h = annotate(g, "lung_arteries", b)
    bad = 0
    for nd in h.nodes:
        groups = {f["group"] for f in nd.attributes.get("bifurcation_frames", [])}
        for e in h.edges:
            v = e.attributes.get("bifurcation_vector")
            if e.start_node == nd.id and v and groups and v["bifurcation_group"] not in groups:
                bad += 1
    assert bad == 0
    stats = h.structure("lung_arteries").statistics["vmtk_bifurcations"]
    assert stats["junctions_with_one"] > 0


def test_a_tube_without_a_bifurcation():
    lines = [np.array([[0.0, 0, 0], [0, 0, 30]])]
    m, geo = tube_field(lines, [np.array([2.0, 2.0])])
    T = medial.trace(m, geo)
    nodes, edges, pos, rad, s = graph_from_tree(T, "i", m, geo, Source(), {"connectivity": "field"})
    g = TubeGraph(structures=[s], nodes=nodes, edges=edges, points=Points(position=pos, radius=rad))
    b = vmtk_branching(g, "i")
    assert len(b.frames.points) == 0 and len(b.vectors.points) == 0
    annotate(g, "i", b)


@pytest.mark.data
@pytest.mark.slow
def test_field_to_vmtk_groups_end_to_end(vessels_data, case_oracle):
    """Our field -> our graph -> vmtk convention -> ported extractor, all flags on: vmtk's own groups."""
    from thalweg.centerlines import centerline_graph
    from thalweg.vmtk import Centerlines
    from cases import reference_source
    g = centerline_graph(vessels_data / "runs" / "C3N-00704_ctpa0625.lung_vessels.duckn.zip", "lung_arteries",
                         ridge_passes=1)                         # the graph vmtk was fed
    b = vmtk_branching(g, "lung_arteries", source=reference_source(g), vmtk_compatible=True)
    ref = Centerlines.from_npz(case_oracle["extract"])
    assert [len(c) for c in b.split.cells] == [len(c) for c in ref.cells]
    for k in ("GroupIds", "Blanking", "CenterlineIds", "TractIds"):
        assert np.array_equal(b.split.cell_data[k], ref.cell_data[k]), k
    assert np.abs(b.split.points - ref.points).max() == 0.0
