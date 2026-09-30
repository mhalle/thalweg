"""The branch partition: vmtk's labeling rule (against vmtkBranchClipper's own output), the exact
pruning, and the label volume on a phantom."""
import numpy as np
import pytest

from thalweg.centerlines import graph_from_tree
from thalweg.graph import Points, Source, TubeGraph
from thalweg.kernel import medial
from thalweg.partition import edge_tubes, label_field, label_points, label_volumes, volume_rows
from thalweg.vmtk.centerlines import Centerlines
from thalweg.vmtk.partition import LabeledTubes, group_tubes, lowest_label
from thalweg.vmtk.polyball import TubeSegments
from phantoms import tube_field, y_tree


def _random_tubes(rng, n_lines=12, wide=True):
    """Polylines scattered in a 60 mm box, radii from 0.3 mm to 9 mm, one label per line."""
    p0, p1, r0, r1, lab = [], [], [], [], []
    for k in range(n_lines):
        start = rng.uniform(0, 60, 3)
        steps = rng.normal(size=(15, 3)) * 1.2
        line = start + np.cumsum(steps, axis=0)
        r = np.abs(rng.normal(size=15)) * (3.0 if wide and k % 4 == 0 else 0.4) + 0.3
        p0.append(line[:-1]), p1.append(line[1:]), r0.append(r[:-1]), r1.append(r[1:])
        lab.append(np.full(14, k))
    seg = TubeSegments(*(np.concatenate(a) for a in (p0, p1, r0, r1)), np.concatenate(lab),
                       np.tile(np.arange(14), n_lines))
    return LabeledTubes(seg, np.concatenate(lab).astype(np.int64))


def test_the_pruned_search_gives_the_exhaustive_answer():
    rng = np.random.default_rng(0)
    tubes = _random_tubes(rng)
    x = rng.uniform(-10, 70, (4000, 3))
    fast = label_points(x, tubes)
    full = label_points(x, tubes, exhaustive=True)
    assert (fast[0] == full[0]).all() and np.array_equal(fast[1], full[1])
    assert (full[1] < 0).sum() > 10 and len(np.unique(full[0])) > 8       # inside tubes and outside


def test_ties_go_to_the_lowest_label_and_unusable_segments_are_skipped():
    a, b = np.array([[0.0, 0, 0]]), np.array([[10.0, 0, 0]])
    same = TubeSegments(np.concatenate([a, a]), np.concatenate([b, b]), np.ones(2), np.ones(2),
                        np.array([0, 1]), np.zeros(2, np.int64))
    x = np.array([[5.0, 3, 0], [5.0, 0.5, 0]])
    for label in ([7, 3], [3, 7]):
        tubes = LabeledTubes(same, np.array(label))
        for got in (lowest_label(x, tubes), label_points(x, tubes)):
            assert got[0].tolist() == [3, 3] and np.allclose(got[1], [8.0, -0.75])
    # a segment whose radius changes as fast as it advances has no tube (vmtk skips it)
    cone = LabeledTubes(TubeSegments(a, np.array([[1.0, 0, 0]]), np.array([0.0]), np.array([1.0]),
                                     np.zeros(1, np.int64), np.zeros(1, np.int64)), np.array([0]))
    for got in (lowest_label(x, cone), label_points(x, cone)):
        assert got[0].tolist() == [-1, -1]


def test_group_tubes_leave_out_blanked_cells():
    cl = Centerlines.from_lines([[[0, 0, 0], [0, 0, 5]], [[0, 0, 5], [0, 0, 8]], [[0, 0, 8], [4, 0, 12]]],
                                [[2, 2], [2, 2], [1, 1]])
    cl.cell_data.update(GroupIds=np.array([0, 1, 2]), Blanking=np.array([0, 1, 0]))
    tubes = group_tubes(cl)
    assert sorted(set(tubes.label.tolist())) == [0, 2]
    lab, val = lowest_label([[0, 0, 6.5], [0, 0, 1], [4, 0, 12]], tubes)
    assert lab.tolist()[1:] == [0, 2] and lab[0] in (0, 2) and val[1] == -4.0


def _y_graph():
    m, geo = tube_field(*y_tree())
    T = medial.trace(m, geo)
    nodes, edges, pos, rad, s = graph_from_tree(T, "y", m, geo, Source(), {"connectivity": "field"})
    g = TubeGraph(structures=[s], nodes=nodes, edges=edges, points=Points(position=pos, radius=rad))
    return m, geo, g


def test_the_label_volume_of_a_y():
    """Every inside lattice point gets an edge; the volumes add up to the structure's; the trunk's
    (radius 3, 30 mm, a rounded end, part of the junction) is near pi r^2 L; each point far from
    the junction belongs to the edge it lies on."""
    m, geo, g = _y_graph()
    tubes = edge_tubes(g, "y")
    labels = label_field(m, geo, tubes, points=g.positions())
    assert labels.dtype == np.int32 and ((labels >= 0) == (m > 0)).all()
    vol = label_volumes(labels, geo)
    voxel = abs(np.linalg.det(np.asarray(geo.directions, float)))
    assert abs(sum(vol.values()) - (m > 0).sum() * voxel) < 1e-6
    assert set(vol) <= {e.id for e in g.edges}
    main = {e.id: e for e in g.edges}
    trunk = max(vol, key=vol.get)
    assert main[trunk].start_node == g.structures[0].roots[0] or main[trunk].length_mm > 20
    assert 0.85 < vol[trunk] / (np.pi * 9 * 30) < 1.25
    for e in g.edges:
        p = g.edge_points(e)
        if e.length_mm > 15:
            mid = p[len(p) // 2]
            assert label_points(mid[None], tubes)[0][0] == e.id
    rows = [dict(edge=e.id) for e in g.edges] + [dict(edge=999)]
    volume_rows(rows, vol)
    assert rows[-1]["volume_mm3"] == 0.0
    assert sum(r["volume_mm3"] for r in rows) == pytest.approx(sum(vol.values()))


def test_pieces_the_tracer_dropped_are_not_labeled():
    m, geo, g = _y_graph()
    m = m.copy()
    m[:3, :3, :3] = 8.0                                              # a fragment in the corner
    labels = label_field(m, geo, edge_tubes(g, "y"), points=g.positions())
    assert (labels[:3, :3, :3] == -1).all() and (labels >= 0).sum() == (m > 0).sum() - 27
    everything = label_field(m, geo, edge_tubes(g, "y"))
    assert (everything[:3, :3, :3] >= 0).all()


@pytest.mark.data
@pytest.mark.parametrize("which", ["vmtk", "ours"])
def test_labels_are_vmtk_branch_clippers(vessels_data, which):
    """vmtkBranchClipper's output on the C3N-00704 subtree, from the split centerlines it was given
    (vmtk's own, and ours). Its points are the input surface's vertices plus the points it inserts
    along each cut. At every vertex the group is the one vmtk assigned, dense and pruned alike. The
    inserted points lie where two groups' tube values tie (that is what the cut is), so there the
    label is either neighbor's: ours differs on about half of them, by a value gap that is tiny."""
    from scipy.spatial import cKDTree
    from thalweg.vmtk.polyball import segment_values
    f = vessels_data / f"C3N-00704_ctpa0625_vmtk_branch_{which}.npz"
    surface = vessels_data / "C3N-00704_ctpa0625_vmtk_input.npz"
    if not f.exists() or not surface.exists():
        pytest.skip(f"{f.name} or {surface.name} is not there (research/vessels/vmtk_branch.py)")
    B = np.load(f)
    cells = np.split(B["cell_ids"], np.cumsum(B["cell_len"])[:-1])
    cl = Centerlines(B["split_points"], cells, {"MaximumInscribedSphereRadius": B["split_radius"]},
                     {"GroupIds": B["cell_group"], "Blanking": B["cell_blank"]})
    tubes = group_tubes(cl)
    x, theirs = B["clip_points"], B["clip_group"]
    lab, _ = label_points(x, tubes)
    vertex = cKDTree(np.load(surface)["verts"]).query(x)[0] < 1e-6
    assert vertex.sum() > 20000 and (lab[vertex] == theirs[vertex]).all()
    some = np.nonzero(vertex)[0][::40]
    assert (lowest_label(x[some], tubes)[0] == theirs[some]).all()
    cut = np.nonzero(~vertex & (lab != theirs))[0]
    assert 3000 < len(cut) < 3500 and vertex.sum() == 21221
    gaps = []
    for a in range(0, len(cut), 256):
        idx = cut[a:a + 256]
        v = segment_values(x[idx], tubes.segments)
        for row, i in zip(v, idx):
            gaps.append(row[tubes.label == theirs[i]].min() - row[tubes.label == lab[i]].min())
    gaps = np.array(gaps)
    assert gaps.min() >= 0 and np.median(gaps) < 1e-3 and gaps.max() < 0.7      # mm^2: on the tie
