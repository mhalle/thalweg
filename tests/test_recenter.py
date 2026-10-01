"""Recentering (thalweg.kernel.recenter) and wall pruning at the root, on flattened and round
phantoms (docs/validation.md §1)."""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from thalweg.kernel import medial
from thalweg.kernel import recenter as rc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
import phantom_suite as PS  # noqa: E402
import vmtk_phantom as VP  # noqa: E402

A, B, L = 4.5, 1.5, 40.0                        # a 3:1 elliptic tube along x


def flat(oblique=False, roll=0.0, a=A):
    f = VP.elliptic_sdf(a, B, L, roll)
    box = (-4.0, -a - 4, -a - 4), (L + 4, a + 4, a + 4)
    if oblique:
        m, geo = PS.field_oblique(f, *box)
        return m, geo, lambda p: (np.asarray(p, float) - PS.SHIFT) @ PS.Q        # world -> tube frame
    m, geo = PS.field_of(f, *box, 0.7)
    return m, geo, lambda p: np.asarray(p, float)


def ends(tree):
    deg = {nd["id"]: 0 for nd in tree.nodes}
    for sg in tree.segments:
        deg[sg["a"]] += 1
        deg[sg["b"]] += 1
    return sum(1 for v in deg.values() if v == 1)


def off_axis(tree, frame):
    """Distance of every traced point clear of the flat ends (1.5 semi-major axes) from the axis."""
    P = frame(np.concatenate([np.array(sg["points"]) for sg in tree.segments]))
    keep = (P[:, 0] > 1.5 * A) & (P[:, 0] < L - 1.5 * A)
    return np.linalg.norm(P[keep, 1:], axis=1)


def test_a_flat_tube_is_traced_as_one_path_on_its_axis():
    m, geo, frame = flat()
    before = medial.trace(m, geo, prune="wall")
    after = medial.trace(m, geo, prune="wall", recenter=True)
    assert ends(before) == ends(after) == 2                 # wall pruning, the root's lobes included
    d0, d = off_axis(before, frame), off_axis(after, frame)
    assert np.median(d0) > 0.3                              # the tracer wanders across the width
    assert np.median(d) < 0.01 and np.percentile(d, 95) < 0.05
    assert after.stats["recentered_points"] > 0
    r = np.concatenate([sg["radius"] for sg in after.segments])
    assert abs(np.median(r) - B) < 0.05                     # the inscribed radius: half the depth


def test_a_rolled_oblique_flat_tube():
    m, geo, frame = flat(oblique=True, roll=np.radians(35))
    t = medial.trace(m, geo, prune="wall", recenter=True)
    assert ends(t) == 2
    d = off_axis(t, frame)
    assert np.median(d) < 0.01 and np.percentile(d, 95) < 0.05


def test_length_pruning_keeps_the_lobes_and_the_lobes_hold_the_path():
    """With the reference spur rule a flat tube keeps its side lobes; their junctions hold the path,
    so recentering pairs with wall pruning."""
    m, geo, frame = flat()
    t = medial.trace(m, geo, prune="length", recenter=True)
    assert ends(t) > 10
    assert np.median(off_axis(t, frame)) > 0.3


def test_a_round_tube_barely_moves():
    segs = [((0.0, 0.0, 0.0), (30.0, 0.0, 0.0), 2.0, 2.0)]
    m, geo = PS.field_of(PS.chain_distance(segs), (-4, -6, -6), (34, 6, 6), 0.7)
    a = medial.trace(m, geo, prune="wall")
    b = medial.trace(m, geo, prune="wall", recenter=True)
    pa = np.concatenate([sg["points"] for sg in a.segments])
    pb = np.concatenate([sg["points"] for sg in b.segments])
    assert len(pa) == len(pb)
    assert np.linalg.norm(pb - pa, axis=1).max() < 0.05
    ra = np.concatenate([sg["radius"] for sg in a.segments])
    rb = np.concatenate([sg["radius"] for sg in b.segments])
    assert np.abs(rb - ra).max() < 0.03


def test_wall_pruning_keeps_a_root_branch_that_continues_the_axis():
    """The deepest point mid-tube (a bulge), so both halves leave the root: the second half is
    behind the first branch's start, not beside it, and stays (as the pulmonary trunk does)."""
    segs = [((0.0, 0.0, 0.0), (20.0, 0.0, 0.0), 2.0, 4.0), ((20.0, 0.0, 0.0), (40.0, 0.0, 0.0), 4.0, 2.0)]
    m, geo = PS.field_of(PS.chain_distance(segs), (-4, -8, -8), (44, 8, 8), 0.7)
    t = medial.trace(m, geo, prune="wall")
    assert sum(b["parent"] < 0 for b in t.branches) == 2
    assert ends(t) == 2
    assert sum(sg["length_mm"] for sg in t.segments) > 35


def test_a_y_moves_nothing_at_its_junction():
    """The nodes are held: junction and tips stay put (the merged-lumen rejection itself is
    pinned by test_a_section_reaching_into_another_path_is_claimed)."""
    segs, _ = PS.y_phantom(60)
    pts = np.concatenate([[s[0], s[1]] for s in segs]).astype(float)
    m, geo = PS.field_of(PS.chain_distance(segs), pts.min(0) - 7, pts.max(0) + 7, 0.7)
    a = medial.trace(m, geo, prune="wall")
    b = medial.trace(m, geo, prune="wall", recenter=True)
    assert [nd["point"] for nd in a.nodes] == [nd["point"] for nd in b.nodes]


def test_tangents_and_end_holds():
    P = np.c_[np.linspace(0, 10, 11), np.zeros(11), np.zeros(11)]
    T = rc.tangents(P, np.full(11, 2.0))
    assert np.allclose(T, [1, 0, 0])
    held = rc.end_holds(P, np.full(11, 0.5), at=[5])             # radius + 1 mm = 1.5 mm
    assert held.tolist() == [True, True, False, False, True, True, True, False, False, True, True]


def test_a_section_reaching_into_another_path_is_claimed():
    segs, _ = PS.y_phantom(30)
    pts = np.concatenate([[s[0], s[1]] for s in segs]).astype(float)
    m, geo = PS.field_of(PS.chain_distance(segs), pts.min(0) - 7, pts.max(0) + 7, 0.7)
    t = medial.trace(m, geo, prune="wall")
    paths = [np.array(b["points"]) for b in t.branches]
    radii = [np.maximum(np.array(b["radius"]), 0.5) for b in t.branches]
    p0 = np.concatenate([p[:-1] for p in paths])
    p1 = np.concatenate([p[1:] for p in paths])
    r0 = np.concatenate([r[:-1] for r in radii])
    r1 = np.concatenate([r[1:] for r in radii])
    lab = np.concatenate([np.full(len(p) - 1, i) for i, p in enumerate(paths)])
    tree = cKDTree(0.5 * (p0 + p1))
    half_seg = 0.5 * np.linalg.norm(p1 - p0, axis=1).max()
    child = next(i for i, b in enumerate(t.branches) if b["parent"] >= 0)
    P, R = paths[child], radii[child]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    T = rc.tangents(P, R)

    def claimed(k):
        xy, n1, n2, _ = rc._sections(m, geo, P[k:k + 1], T[k:k + 1], R[k:k + 1])[0]
        return rc._claimed(P[k], xy, n1, n2, child, (p0, p1, r0, r1, lab), tree, R.max() + 3, half_seg)

    assert claimed(int(np.argmin(np.abs(s - 2.0))))                 # 2 mm out: still the merged lumen
    assert not claimed(int(np.argmin(np.abs(s - 0.6 * s[-1]))))      # well down the daughter


def test_tube_settings_by_structure():
    from thalweg.centerlines import FLAT_TUBES, tube_settings
    assert tube_settings("esophagus") == ("wall", True)
    assert tube_settings("lung_arteries") == ("length", False)
    assert tube_settings("colon", prune="length") == ("length", True)        # explicit values win
    assert tube_settings("lung_airways", recenter=True) == ("length", True)
    assert tube_settings("trachea", "auto", False) == ("wall", False)
    assert "aorta" not in FLAT_TUBES                                         # round: vessel settings


def test_centerline_graph_records_the_settings_it_used(monkeypatch):
    import pytest
    from thalweg import centerlines
    from thalweg.centerlines import centerline_graph
    from thalweg.errors import ThalwegError
    monkeypatch.setattr(centerlines, "open_store", lambda s: s)              # a stand-in store

    class Ref:
        scheme, part, label_value = "test", None, None

    class Store:
        path = "memory"
        structures = []

    m, geo, _ = flat()
    g = centerline_graph(Store(), "esophagus", margin=(m, geo, Ref()))
    p = g.structures[0].parameters
    assert (p["prune"], p["recenter"]) == ("wall", True)
    assert g.structures[0].statistics["recentered_point_count"] > 0
    g = centerline_graph(Store(), "vessel", margin=(m, geo, Ref()))
    p = g.structures[0].parameters
    assert (p["prune"], p["recenter"]) == ("length", False)
    with pytest.raises(ThalwegError):
        centerline_graph(Store(), "vessel", margin=(m, geo, Ref()), prune="spurs")


def test_a_six_to_one_tube_is_recentered_too():
    """The window grows to MAX_HALF_RADII path radii: a 9 x 1.5 mm section closes in it."""
    m, geo, frame = flat(a=9.0)
    t = medial.trace(m, geo, prune="wall", recenter=True)
    assert t.stats["recentered_points"] > 20
    P = frame(np.concatenate([np.array(sg["points"]) for sg in t.segments]))
    keep = (P[:, 0] > 1.5 * 9.0) & (P[:, 0] < L - 1.5 * 9.0)
    assert np.median(np.linalg.norm(P[keep, 1:], axis=1)) < 0.01


def test_the_shift_ramps_up_from_held_points():
    m, geo, frame = flat()
    X = cKDTree(medial.crossings(m, geo))
    s = np.arange(6.0, 34.0, 0.7)
    P = np.c_[s, np.full(len(s), 1.0), np.zeros(len(s))]               # 1 mm off the axis, inside
    held = np.zeros(len(s), bool)
    held[:3] = True
    Q, R, moved = rc.recenter(m, geo, [P], [np.full(len(s), 1.5)], [held], X)
    shift = np.linalg.norm(Q[0] - P, axis=1)
    gap = np.maximum(s - s[2], 0.0)
    assert np.all(shift <= rc.RAMP * gap + 1e-9)
    assert np.all(shift[:3] == 0) and moved[0][5:].all()
    far = gap > 2.5 / rc.RAMP
    assert np.abs(Q[0][far, 1]).max() < 0.02                           # on the axis beyond the ramp


def test_wall_pruning_keeps_a_root_branch_when_the_first_branch_turns():
    """The first root branch turns 20 mm out; a ray across the root's axis from there would run
    down the turned leg and read no wall, so those stations are skipped and the side branch stays."""
    segs = [((0, 0, 0), (0.01, 0, 0), 8.0, 8.0), ((0, 0, 0), (20, 0, 0), 5.0, 5.0),
            ((20, 0, 0), (20, 80, 0), 5.0, 5.0), ((0, 0, 0), (-5, 30, 0), 3.5, 3.5)]
    m, geo = PS.field_of(PS.chain_distance(segs), (-20, -14, -14), (30, 88, 14), 0.7)
    a = medial.trace(m, geo, prune="length")
    b = medial.trace(m, geo, prune="wall")
    assert ends(a) == ends(b) == 2


def test_a_pass_through_root_is_counted_once():
    m, geo, frame = flat(oblique=True, roll=np.radians(35))
    t = medial.trace(m, geo, prune="wall", recenter=True)
    before = medial.trace(m, geo, prune="wall")
    P0 = np.concatenate([np.array(b["points"]) for b in before.branches])
    P1 = np.concatenate([np.array(b["points"]) for b in t.branches])
    roots = [b for b in t.branches if b["parent"] < 0]
    root_moved = not np.allclose(roots[0]["points"][0], before.branches[0]["points"][0])
    shared = 1 if len(roots) == 2 and root_moved else 0                 # listed in both branches
    assert t.stats["recentered_points"] == int((np.linalg.norm(P1 - P0, axis=1) > 0).sum()) - shared
    assert t.stats["deepest_point"] == before.branches[0]["points"][0]


def test_names_are_matched_as_totalsegmentator_spells_them():
    """A DICOM SEG written by TotalSegmentator uses display names (the round-10 review found its
    esophagus traced with the vessel settings, and its lobes not found)."""
    from thalweg.centerlines import tube_settings
    from thalweg.store import StructureRef, canonical_name, matching
    assert canonical_name("Esophagus") == "esophagus"
    assert canonical_name("Small Intestine") == "small_bowel"
    assert canonical_name("Middle lobe of right lung") == "lung_middle_lobe_right"
    assert tube_settings("Esophagus") == ("wall", True) and tube_settings("Small Intestine") == ("wall", True)
    assert tube_settings("label_35") == ("length", False)
    refs = [StructureRef("Esophagus", 0, 35, "x"), StructureRef("esophagus", 1, 15, "x"),
            StructureRef("Left Upper lobe of lung", 0, 13, "x")]
    assert matching(refs, "esophagus") == [refs[1]]                       # an exact name wins
    assert matching(refs, "ESOPHAGUS") == refs[:2]
    assert matching(refs, "lung_upper_lobe_left") == [refs[2]]
    assert matching(refs, "esophagus", part=0) == [refs[0]]


def test_rerooting_merges_the_deepest_point_when_it_lies_along_one_path(monkeypatch):
    """The round-10 review found the tracer's deepest point left as a joint after re-rooting,
    splitting the pulmonary trunk and the trachea into two edges each."""
    from thalweg import centerlines
    from thalweg.centerlines import centerline_graph
    monkeypatch.setattr(centerlines, "open_store", lambda s: s)

    class Ref:
        scheme, part, label_value = "test", None, None

    class Store:
        path = "memory"
        structures = []

    segs = [((0.0, 0.0, 0.0), (20.0, 0.0, 0.0), 2.0, 4.0), ((20.0, 0.0, 0.0), (40.0, 0.0, 0.0), 4.0, 2.0)]
    m, geo = PS.field_of(PS.chain_distance(segs), (-4, -8, -8), (44, 8, 8), 0.7)
    deep = centerline_graph(Store(), "t", margin=(m, geo, Ref()), root="deepest")
    assert len(deep.edges) == 2                                  # two branches leave the deepest point
    g = centerline_graph(Store(), "t", margin=(m, geo, Ref()))
    assert len(g.edges) == 1 and not any(nd.kind == "joint" for nd in g.nodes)
    assert abs(g.edges[0].length_mm - sum(e.length_mm for e in deep.edges)) < 1e-9
    st = g.structures[0].statistics
    assert st["edge_count"] == 1 and st["joint_count"] == 0
    assert st["deepest_point"] == deep.structures[0].statistics["deepest_point"]
    P = g.edge_points(g.edges[0])
    assert np.linalg.norm(P[0] - np.array(g.nodes[g.edges[0].start_node].position)) < 1e-9
    assert np.linalg.norm(P[-1] - np.array(g.nodes[g.edges[0].end_node].position)) < 1e-9
