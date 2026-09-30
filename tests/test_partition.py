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


def test_a_segment_without_a_usable_radius_makes_no_tube_either_way():
    """A NaN radius: that segment is skipped, in the dense and the pruned search alike."""
    rng = np.random.default_rng(2)
    tubes = _random_tubes(rng, n_lines=5)
    tubes.segments.r0[3] = np.nan
    x = rng.uniform(0, 60, (500, 3))
    dense, pruned = lowest_label(x, tubes), label_points(x, tubes)
    assert (dense[0] >= 0).all() and (dense[0] == pruned[0]).all() and np.array_equal(dense[1], pruned[1])


def test_chunks_do_not_change_the_labels(monkeypatch):
    import thalweg.partition as P
    rng = np.random.default_rng(3)
    tubes = _random_tubes(rng)
    x = rng.uniform(0, 60, (100, 3))
    full = label_points(x, tubes, exhaustive=True)
    monkeypatch.setattr(P, "POINT_CHUNK", 7)
    got = label_points(x, tubes)
    assert (got[0] == full[0]).all() and np.array_equal(got[1], full[1])


def test_edge_tubes_take_radius_zero_where_a_sample_has_none():
    m, geo, g = _y_graph()
    rad = list(g.points.radius)
    rad[1] = -1.0
    g = g.model_copy(update={"points": g.points.model_copy(update={"radius": rad})})
    seg = edge_tubes(g, "y").segments
    assert seg.r0.min() == 0.0 and seg.r1.min() == 0.0 and np.isfinite(seg.r0).all()


def test_the_label_volume_on_an_oblique_anisotropic_grid():
    """A tube along world z on a grid whose axes are permuted, flipped and unequal: each lattice
    point inside takes the label of the half (z below or above 20) it lies in."""
    from rankfield.geometry import Geometry
    from phantoms import capsule_distance
    shape = (30, 60, 25)
    d = np.array([[0.0, -0.8, 0.0], [0.0, 0.0, 0.7], [0.5, 0.0, 0.0]])       # axis 0 -> -y, 1 -> z, 2 -> x
    origin = np.array([-6.0, 12.0, -1.0])
    geo = Geometry(shape=shape, directions=tuple(map(tuple, d)), origin=tuple(origin))
    idx = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1).reshape(-1, 3)
    world = origin + idx @ d
    a, b = np.array([0.0, 0, 3]), np.array([0.0, 0, 37])
    m = np.clip(10 * capsule_distance(world, a, b, 3.0, 3.0), -8, 8).astype(np.float32).reshape(shape)
    mid = np.array([0.0, 0, 20])
    seg = TubeSegments(np.array([a, mid]), np.array([mid, b]), np.full(2, 3.0), np.full(2, 3.0),
                       np.array([0, 1]), np.zeros(2, np.int64))
    labels = label_field(m, geo, LabeledTubes(seg, np.array([10, 20])))
    inside = labels.reshape(-1) >= 0
    assert inside.sum() > 1000 and (inside == (m.reshape(-1) > 0)).all()
    z = world[inside, 2]
    got = labels.reshape(-1)[inside]
    clear = np.abs(z - 20) > 1e-6
    assert (got[clear] == np.where(z[clear] < 20, 10, 20)).all()
    vol = label_volumes(labels, geo)
    assert abs(vol[10] / vol[20] - 1.0) < 0.06 and abs(sum(vol.values()) - inside.sum() * 0.28) < 1e-6


def test_a_fragment_touching_only_at_a_corner_is_not_the_traced_piece():
    """Pieces are the field's own: a voxel whose only contact with the structure is a corner
    (26-connected, but the interpolant dips below zero between them) stays unlabeled."""
    from rankfield.geometry import Geometry
    from thalweg.partition import traced_mask
    m = np.full((12, 12, 12), -8.0, np.float32)
    m[2:6, 2:6, 2:6] = 8.0
    m[6, 6, 6] = 8.0                                                  # a corner neighbor of (5, 5, 5)
    geo = Geometry(shape=m.shape, directions=((1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0)), origin=(0.0, 0, 0))
    mask = traced_mask(m, geo, np.array([[3.0, 3, 3], [3.4, 3.4, 3.4], [50.0, 0, 0]]))
    assert mask.sum() == 64 and not mask[6, 6, 6]


def test_tube_function_and_distance_to_centerlines_of_a_straight_tube():
    """Along a straight tube of radius 2: the tube function is d^2 - r^2 at distance d from the
    axis, the distance to the centerline is d, and its radius there is 2."""
    from rankfield.geometry import Geometry
    from thalweg.partition import distance_to_centerlines, tube_function
    from thalweg.graph import Edge, Node, Provenance, Structure
    z = np.linspace(0, 30, 31)
    pos = [(0.0, 0.0, float(v)) for v in z]
    g = TubeGraph(structures=[Structure(name="t", roots=[0], method="test")],
                  nodes=[Node(id=0, kind="root", position=pos[0], structure="t"),
                         Node(id=1, kind="tip", position=pos[-1], structure="t")],
                  edges=[Edge(id=0, structure="t", start_node=0, end_node=1, point_range=(0, 31),
                              length_mm=30.0, provenance=Provenance(method="field"))],
                  points=Points(position=pos, radius=[2.0] * 31))
    geo = Geometry(shape=(9, 9, 5), directions=((1.0, 0, 0), (0, 1.0, 0), (0, 0, 5.0)),
                   origin=(-4.0, -4.0, 5.0))
    f = tube_function(edge_tubes(g, "t"), geo, (9, 9, 5))
    i, j = np.meshgrid(np.arange(9) - 4.0, np.arange(9) - 4.0, indexing="ij")
    assert np.allclose(f, (i ** 2 + j ** 2 - 4.0)[..., None], atol=1e-5)
    x = np.array([[3.0, 0, 12.3], [0, 0.5, 7.0], [1.0, 1.0, 40.0]])
    d, r = distance_to_centerlines(x, g, "t")
    assert np.allclose(d, [3.0, 0.5, np.sqrt(2 + 100)]) and np.allclose(r, 2.0)
    d, r = distance_to_centerlines(x, g, "t", use_radius=True)
    assert np.allclose(d[:2], [3.0, 0.5])


@pytest.mark.data
def test_distance_to_centerlines_is_vmtks(vessels_data):
    """vmtkDistanceToCenterlines (its defaults) at the 28,449 mapped surface points of the C3N-00704
    subtree, from the centerlines vmtk used: the same distances to rounding."""
    from thalweg.vmtk.partition import distance_to_centerlines
    f = vessels_data / "C3N-00704_ctpa0625_vmtk_mapping.npz"
    if not f.exists():
        pytest.skip(f"{f.name} is not there (research/vessels/vmtk_mapping.py)")
    M = np.load(f)
    cells = np.split(M["cell_ids"], np.cumsum(M["cell_len"])[:-1])
    cl = Centerlines(M["cl_points"], cells, {"MaximumInscribedSphereRadius": M["cl_radius"]}, {})
    some = slice(None, None, 5)
    d, c, r = distance_to_centerlines(M["surf_points"][some], cl)
    assert np.abs(d - M["surf_dist"][some]).max() < 1e-9 and (r > 0).all()
