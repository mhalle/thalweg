"""vtkvmtkCenterlineBranchExtractor (+ PolyBallLine, CenterlineSphereDistance) against vmtk's output.

The oracle stage ``extract`` is vmtkbranchextractor run on the ``attributes`` output; ids and cell
structure must match exactly, coordinates and point arrays to 1e-9. The pruned / deduplicated
tube evaluation and the backward touching-sphere search (which the oracle does not exercise) are
checked against literal scalar readings of the C++.
"""
import hashlib
import math
import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import pytest

from thalweg.vmtk import VMTK_FLAGS, Centerlines
from thalweg.vmtk import branches as B
from thalweg.vmtk import polyball as PB
from thalweg.vmtk.sphere_distance import find_touching_sphere_center

R = "MaximumInscribedSphereRadius"
INT_ARRAYS = ("GroupIds", "CenterlineIds", "TractIds", "Blanking")
FIXTURES = Path(__file__).parent / "fixtures" / "vmtk_oracle"
LOOPBACK = FIXTURES / "loopback"


@pytest.fixture(scope="module")
def extracted(vmtk_oracle):
    inp = Centerlines.from_npz(vmtk_oracle["attributes"])
    out = B.extract_branches(inp, **{f: True for f in VMTK_FLAGS["extract_branches"]})
    return inp, out, Centerlines.from_npz(vmtk_oracle["extract"])


def test_extract_matches_oracle(extracted):
    _, out, ref = extracted
    assert out.n_points == ref.n_points and out.n_cells == ref.n_cells
    assert all(np.array_equal(a, b) for a, b in zip(out.cells, ref.cells))
    for k in INT_ARRAYS:
        assert out.cell_data[k].dtype == ref.cell_data[k].dtype
        assert np.array_equal(out.cell_data[k], ref.cell_data[k]), k
    assert set(out.cell_data) == set(ref.cell_data)
    np.testing.assert_allclose(out.points, ref.points, rtol=0, atol=1e-9)
    assert set(out.point_data) == set(ref.point_data)
    for k, v in ref.point_data.items():
        assert out.point_data[k].shape == v.shape
        np.testing.assert_allclose(out.point_data[k], v, rtol=0, atol=1e-9, err_msg=k)   # InterpolateEdge


def test_input_not_mutated(phantom_oracle):
    inp = Centerlines.from_npz(phantom_oracle["attributes"])
    before = inp.copy()
    B.extract_branches(inp)
    assert np.array_equal(inp.points, before.points)
    assert all(np.array_equal(a, b) for a, b in zip(inp.cells, before.cells))
    assert set(inp.cell_data) == set(before.cell_data)


def test_corrected_steps_keep_the_grouping(phantom_oracle):
    """The squared-length step count only changes where the split points land."""
    inp = Centerlines.from_npz(phantom_oracle["attributes"])
    ref = Centerlines.from_npz(phantom_oracle["extract"])
    out = B.extract_branches(inp)
    assert out.n_cells == ref.n_cells
    for k in INT_ARRAYS:
        assert np.array_equal(out.cell_data[k], ref.cell_data[k])


# -- the pruned evaluations equal the brute-force ones -------------------------------------------

def _brute_inside(cl, pts, exclude):
    out = np.zeros((len(pts), cl.n_cells), bool)
    for j in range(cl.n_cells):
        if j == exclude:
            continue
        v = PB.evaluate_function(pts, PB.tube_segments(cl, R, [j])).value
        out[:, j] = v <= 0.0
    return out


def test_inside_matches_brute_force(phantom_oracle):
    cl = Centerlines.from_npz(phantom_oracle["attributes"])
    tubes = B._Tubes(cl, R)
    for i in range(cl.n_cells):
        pts = cl.cell_points(i)
        assert np.array_equal(tubes.inside(pts, cl.n_cells, i), _brute_inside(cl, pts, i))


def test_mutual_in_tube_matches_literal_loop(phantom_oracle):
    tr = Centerlines.from_npz(phantom_oracle["extract"])
    got = B._mutual_in_tube(tr, R).dense()
    want = np.full(got.shape, np.inf)
    for i in range(tr.n_cells):
        own = PB.tube_segments(tr, R, [i])
        x = tr.cell_points(i)[:1]
        for j in range(tr.n_cells):
            st = PB.evaluate_function(x, PB.tube_segments(tr, R, [j]))
            cv = PB.evaluate_function(st.center, own).value
            if st.value[0] < -1e-12:
                want[i, j] = cv[0]
    assert np.array_equal(got, want)
    assert (got < -1e-12).any()


def _literal_grouping(tr, n_input, rule):
    """PointInTubeGroupTracts read literally (every k, every j, every tube evaluated on its own):
    ``rule="last"`` is the C++'s per-j min bookkeeping (the last candidate wins), ``"min"`` the minimum
    it means to keep (first on a tie), ``"first"`` the first candidate in cell order (a wrong rule,
    kept to show the fixtures tell them apart)."""
    cid, blank = tr.cell_data["CenterlineIds"], tr.cell_data["Blanking"]
    group = list(range(tr.n_cells))
    for i in range(tr.n_cells):
        own = PB.tube_segments(tr, R, [i])
        cg = group[i]
        for k in range(n_input):
            if k == cid[i]:
                continue
            if any(j != i and cid[j] == k and group[j] == cg for j in range(tr.n_cells)):
                continue
            same = -1
            min_cv = 1e32
            for j in range(tr.n_cells):
                if j == i or cid[j] != k or blank[i] != blank[j] or group[j] == cg:
                    continue
                if rule == "last":
                    min_cv = 1e32                  # declared inside the j loop in the C++
                st = PB.evaluate_function(tr.cell_points(i)[:1], PB.tube_segments(tr, R, [j]))
                cv = PB.evaluate_function(st.center, own).value[0]
                if st.value[0] < -1e-12 and cv < -1e-12 and cv < min_cv:
                    min_cv = cv
                    same = group[j]
                    if rule == "first":
                        break
            if same != -1:
                group = [cg if g == same else g for g in group]
    return group


@pytest.mark.parametrize("vmtk_last_tract", [True, False])
def test_grouping_replays_the_literal_loops(frozen_oracle, vmtk_last_tract):
    """The port's grouping against the literal loops: with ``vmtk_last_tract`` the last candidate,
    else the minimum. On the tracts before grouping (the hairpin fixture has two candidates of
    different groups there, the loopback fixture three)."""
    inp = Centerlines.from_npz(frozen_oracle["attributes"])
    tr = B._split_tracts(inp, R, "CenterlineIds", "TractIds", "Blanking", True, True)
    want = _literal_grouping(tr, inp.n_cells, "last" if vmtk_last_tract else "min")
    got = B._point_in_tube_group_tracts(tr, inp.n_cells, R, "CenterlineIds", "Blanking", vmtk_last_tract)
    assert got.tolist() == want


def test_loopback_first_deepest_and_last_candidates_differ():
    """The default grouping rule, pinned without case data. Centerline 1 passes over the root and
    then ends beside it, so the trunk of centerline 0 (tract 0) has three candidates on centerline 1:
    its trunk (tract 3, first in cell order), the pass (5, the deepest: the trunk widens upward) and
    the looping tail (7, last and shallowest). The three rules pick three different tracts and group
    three different ways; the default is the deepest."""
    inp = Centerlines.from_npz(LOOPBACK / "attributes.npz")
    tr = B._split_tracts(inp, R, "CenterlineIds", "TractIds", "Blanking", False, False)
    assert tr.cell_data["CenterlineIds"].tolist() == [0, 0, 0, 1, 1, 1, 1, 1]
    assert tr.cell_data["Blanking"].tolist() == [0, 1, 0, 0, 1, 0, 1, 0]
    value = B._mutual_in_tube(tr, R).dense()
    cand = [j for j in (3, 5, 7) if value[0, j] < -1e-12]
    assert cand == [3, 5, 7]
    assert value[0, 5] < value[0, 3] < value[0, 7]                       # deepest, first, last
    want = {rule: _literal_grouping(tr, 2, rule) for rule in ("first", "min", "last")}
    assert want == {"first": [0, 1, 2, 0, 1, 5, 6, 7], "min": [3, 1, 2, 3, 1, 3, 6, 7],
                    "last": [3, 1, 2, 3, 1, 5, 6, 3]}
    got = B._point_in_tube_group_tracts(tr, 2, R, "CenterlineIds", "Blanking")
    assert got.tolist() == want["min"]
    vm = B._point_in_tube_group_tracts(tr, 2, R, "CenterlineIds", "Blanking", vmtk_last_tract=True)
    assert vm.tolist() == want["last"]
    out = B.extract_branches(inp)
    assert out.cell_data["GroupIds"].tolist() == [2, 0, 1, 2, 3, 4]      # the pass joins the trunk
    ref = Centerlines.from_npz(LOOPBACK / "extract.npz")                  # vmtk: the tail does
    assert ref.cell_data["GroupIds"].tolist() == [2, 0, 1, 2, 2]


@pytest.mark.data
def test_default_extract_on_the_case(case_oracle):
    """The corrected default (every flag off) on the case, pinned: 637 tracts in 139 groups, and a
    digest of the cells and the CenterlineIds / TractIds / Blanking / GroupIds arrays. (A
    first-candidate grouping rule gives 587 tracts; vmtk's flags give 637 tracts, 12182 points.)"""
    out = B.extract_branches(Centerlines.from_npz(case_oracle["attributes"]))
    assert (out.n_cells, out.n_points) == (637, 12234)
    assert len(np.unique(out.cell_data["GroupIds"])) == 139
    h = hashlib.sha256()
    h.update(np.array([len(c) for c in out.cells], np.int64).tobytes())
    h.update(np.concatenate(out.cells).astype(np.int64).tobytes())
    for k in ("CenterlineIds", "TractIds", "Blanking", "GroupIds"):
        h.update(np.asarray(out.cell_data[k], np.int32).tobytes())
    assert h.hexdigest() == "4ed91abbea16da638eda68abf91464afc4b2fb92d43b4cd37f6463af40bcec32"


@pytest.mark.data
@pytest.mark.slow
def test_whole_tree_extract_budget(vessels_data):
    """A whole tree, source to every tip: the C3N-00704 airways (134 centerlines, ~82k points; the
    shared trunk repeated in every one). Once 17 s and 3.5 GB (a dense tract x tract array and
    O(tips^2) tube-exit searches); now ~2 s and ~0.2 GB. Budget: 30 s under tracemalloc and a
    traced peak under 0.5 GB."""
    from thalweg.adapters import to_vmtk
    from thalweg.centerlines import centerline_graph
    store = vessels_data / "runs" / "C3N-00704_ctpa0625.lung_vessels.duckn.zip"
    if not store.exists():
        pytest.skip(f"no {store.name}")
    cl = to_vmtk(centerline_graph(store, "lung_airways"), "lung_airways").centerlines
    assert cl.n_cells > 100
    tracemalloc.start()
    try:
        t = time.perf_counter()
        out = B.extract_branches(cl)
        seconds = time.perf_counter() - t
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    print(f"airways: {cl.n_cells} centerlines -> {out.n_cells} tracts, {seconds:.1f} s, "
          f"traced peak {peak / 1e9:.2f} GB")
    assert set(out.cell_data["CenterlineIds"].tolist()) == set(range(cl.n_cells))
    assert seconds < 30.0
    assert peak < 0.5e9


# -- vtkvmtkCenterlineSphereDistance, both directions, against a scalar reading -------------------

def _scalar_touching(cl, cell, sub, pc, forward, vmtk_steps):
    ids, P, r = cl.cells[cell], cl.points, cl.point_data[R]

    def sph(c, rad, p):
        return (c[0] - p[0]) * (c[0] - p[0]) + (c[1] - p[1]) * (c[1] - p[1]) + (c[2] - p[2]) * (c[2] - p[2]) \
            - rad * rad
    subs = list(range(sub)) if forward else list(range(len(ids) - 2, sub, -1))
    p0, p1 = P[ids[sub]], P[ids[sub + 1]]
    x = [p0[k] + pc * (p1[k] - p0[k]) for k in range(3)]
    ts = -1
    for s in subs:
        a, b = sph(P[ids[s]], r[ids[s]], x), sph(P[ids[s + 1]], r[ids[s + 1]], x)
        if (a > 0.0 and b <= 0.0) if forward else (a <= 0.0 and b > 0.0):
            ts = s
            break
    if ts == -1:
        a, b = sph(P[ids[sub]], r[ids[sub]], x), sph(P[ids[sub + 1]], r[ids[sub + 1]], x)
        if a <= 0.0 and b <= 0.0:
            return -1, 0.0
        ts = sub
    c0, c1, r0, r1 = P[ids[ts]], P[ids[ts + 1]], r[ids[ts]], r[ids[ts + 1]]
    e = [c0[k] - c1[k] for k in range(3)]
    d2 = e[0] * e[0] + e[1] * e[1] + e[2] * e[2]
    n = int(math.ceil(min((d2 if vmtk_steps else math.sqrt(d2)) / (1e-6 * (r0 + r1) / 2.0), 1e5)))
    step, cur, tp = 1.0 / n, 0.0, 0.0
    for _ in range(n):
        s0 = [c0[k] + cur * (c1[k] - c0[k]) for k in range(3)]
        s1 = [c0[k] + (cur + step) * (c1[k] - c0[k]) for k in range(3)]
        a = sph(s0, r0 + cur * (r1 - r0), x)
        b = sph(s1, r0 + (cur + step) * (r1 - r0), x)
        if forward and a > 0.0 and b <= 0.0:
            tp = cur
            break
        if not forward and a <= 0.0 and b > 0.0:
            tp = cur + step
            break
        cur += step
    if ts == sub and ((forward and tp > pc) or (not forward and tp < pc)):
        return -1, 0.0
    return ts, tp


@pytest.mark.parametrize("forward", [True, False])
def test_touching_sphere_matches_scalar(phantom_oracle, forward):
    cl = Centerlines.from_npz(phantom_oracle["attributes"])
    for cell, sub, pc, vmtk_steps in [(0, 150, 0.3, True), (3, 60, 0.75, False)]:
        got = find_touching_sphere_center(cl, R, cell, sub, pc, forward, vmtk_steps)
        want = _scalar_touching(cl, cell, sub, pc, forward, vmtk_steps)
        assert got[0] == want[0] and got[1] == want[1]
        assert got[0] != -1


# -- MergeTracts and its last-tract defect --------------------------------------------------------

def _tracts(groups):
    """One centerline cut into len(groups) two-point tracts along x (shared end coordinates)."""
    n = len(groups)
    pts = np.concatenate([[[i, 0, 0], [i + 1, 0, 0]] for i in range(n)]).astype(float)
    cells = [np.array([2 * i, 2 * i + 1]) for i in range(n)]
    cd = {"GroupIds": np.array(groups, np.int32), "CenterlineIds": np.zeros(n, np.int32),
          "TractIds": np.arange(n, dtype=np.int32), "Blanking": np.zeros(n, np.int32)}
    return Centerlines(pts, cells, {R: np.ones(2 * n)}, cd)


def test_merge_tracts_joins_runs():
    out = B._merge_tracts(_tracts([0, 1, 0, 2]), "GroupIds", "CenterlineIds", "TractIds", False)
    assert out.cell_data["GroupIds"].tolist() == [0, 2]
    assert [c.tolist() for c in out.cells] == [[0, 1, 3, 5], [6, 7]]


def test_merge_tracts_last_tract_defect_flag():
    out = B._merge_tracts(_tracts([0, 1, 0]), "GroupIds", "CenterlineIds", "TractIds", False)
    assert [c.tolist() for c in out.cells] == [[0, 1, 3, 5]]
    vm = B._merge_tracts(_tracts([0, 1, 0]), "GroupIds", "CenterlineIds", "TractIds", True)
    assert [c.tolist() for c in vm.cells] == [[0, 1, 3], [4, 5]]
    assert vm.cell_data["GroupIds"].tolist() == [0, 0]


def test_last_tract_defect_on_the_hairpin():
    """Daughter b curves back to end beside the root: the root (cell 0's first point) is inside both
    of centerline 1's candidate tracts - its trunk, and b. vmtk keeps the LAST (b) and so groups b,
    and then all of centerline 1, with the trunk; the intended minimum keeps the trunk."""
    hairpin = Path(__file__).parent / "fixtures" / "vmtk_oracle" / "hairpin"
    inp = Centerlines.from_npz(hairpin / "attributes.npz")
    tr = B._split_tracts(inp, R, "CenterlineIds", "TractIds", "Blanking", True, True)
    assert tr.cell_data["CenterlineIds"].tolist() == [0, 0, 0, 1, 1, 1]
    assert tr.cell_data["Blanking"].tolist() == [0, 1, 0, 0, 1, 0]
    value = B._mutual_in_tube(tr, R).dense()
    assert np.all(value[0, [3, 5]] < -1e-12) and value[0, 3] < value[0, 5]     # trunk deeper than b
    fixed = B._point_in_tube_group_tracts(tr, 2, R, "CenterlineIds", "Blanking")
    vmtk = B._point_in_tube_group_tracts(tr, 2, R, "CenterlineIds", "Blanking", vmtk_last_tract=True)
    assert fixed[0] == fixed[3] and fixed[1] == fixed[4] and len(set(fixed.tolist())) == 4
    assert vmtk[0] == vmtk[3] == vmtk[5] and vmtk[1] == vmtk[4]     # MergeTracts then swallows tract 4
    out = B.extract_branches(inp)
    assert out.cell_data["GroupIds"].tolist() == [0, 1, 2, 0, 1, 3]
    ref = Centerlines.from_npz(hairpin / "extract.npz")                      # vmtk: centerline 1 is one group
    assert ref.cell_data["GroupIds"].tolist() == [2, 0, 1, 2, 2]


def test_unsplit_centerline_is_float32_too():
    """A centerline with no splitting point goes through vtkPoints::DeepCopy into a float vtkPoints."""
    single = Path(__file__).parent / "fixtures" / "vmtk_oracle" / "single_line"
    inp = Centerlines.from_npz(single / "attributes.npz")
    ref = Centerlines.from_npz(single / "extract.npz")
    vm = B.extract_branches(inp, vmtk_float32=True)
    assert np.array_equal(vm.points, ref.points)
    assert np.array_equal(vm.points, inp.points.astype(np.float32).astype(np.float64))
    fixed = B.extract_branches(inp)
    assert np.array_equal(fixed.points, inp.points) and not np.array_equal(fixed.points, vm.points)


def _y_lines(radius):
    x = np.linspace(0, 10, 34)
    line = np.stack([x, 0 * x, 0 * x], 1)
    side = np.r_[line[:17], line[16] + np.cumsum(np.tile([[0, 0.3, 0]], (15, 1)), 0)]
    return Centerlines.from_lines([line, side], [np.full(34, radius), np.full(32, radius)])


def test_zero_radius_is_capped_not_an_overflow():
    """vmtk's tube-exit search has no step cap: (int)ceil(L / 0). The port caps it at 1e5 steps."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = B.extract_branches(_y_lines(0.0))
    assert out.n_cells == 6 and out.cell_data["Blanking"].tolist() == [0, 1, 0, 0, 1, 0]


def test_non_finite_radius_is_an_error():
    cl = _y_lines(1.0)
    cl.point_data[R][33] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        B.extract_branches(cl)
