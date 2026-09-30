"""vtkvmtkCenterlineBranchExtractor: split centerlines into tracts and group the tracts into branches.

Ports ``vtkvmtkCenterlineBranchExtractor`` as the ``vmtkbranchextractor`` script drives it, with
everything on its call path, line for line:

- vtkvmtkCenterlineBranchExtractor.cxx: ``ComputeCenterlineSplitting`` (where each centerline
  leaves every other centerline's tube, and the point one inscribed sphere upstream of it) and
  ``GroupTracts`` (the base class's);
- vtkvmtkCenterlineSplittingAndGroupingFilter.cxx: ``RequestData``, ``SplitCenterline``,
  ``PointInTubeGroupTracts`` (the default ``GroupingMode``), ``MergeTracts``,
  ``MakeGroupIdsAdjacent``, ``MakeTractIdsAdjacent``;
- vtkvmtkCenterlineSphereDistance.cxx (:mod:`.sphere_distance`) and vtkvmtkPolyBallLine.cxx
  (:mod:`.polyball`); ``vtkAppendPolyData`` and ``vtkDataSetAttributes::InterpolateEdge`` (VTK) for
  how the pieces are joined and the split points get their point data.

Public: :func:`extract_branches` (the filter) and :func:`compute_centerline_splitting` (one cell's
splitting points, for inspection).

Output, as vmtk's: one cell per tract (after merging), cell arrays ``CenterlineIds``, ``TractIds``,
``Blanking`` and ``GroupIds`` (int32, plus any input cell arrays, copied from the source cell);
every input point array carried over, linearly interpolated at the inserted split points
(``(1 - t) * a + t * b``, VTK's ``InterpolateEdge``, :func:`._vtk.interpolate_edge`; vmtk's
defective ``InterpolateTuple`` is not on this path, so there is no ``vmtk_interp`` here). Field data
is dropped (``vtkAppendPolyData`` does not carry it).

vmtk defects, each behind a flag (default: corrected; ``True`` reproduces vmtk, for oracle
comparison):

- ``vmtk_float32``: ``SplitCenterline`` stores every split centerline's points in a default
  ``vtkPoints`` - VTK_FLOAT - so vmtk's output coordinates (and the tubes the grouping step
  evaluates on them) are float32-rounded; a centerline with no splitting point is deep-copied
  into the same float ``vtkPoints`` (``vtkPoints::DeepCopy`` converts to the destination's type),
  so it is float32 too. The default keeps double coordinates throughout: on the oracles the
  grouping is the same and the points move by up to 1.9e-6 mm (phantom) / 7.6e-6 mm (case) - the
  rounding itself; the point data are unchanged.
- ``vmtk_steps``: both linear searches (the tube exit here, the touching sphere in
  :mod:`.sphere_distance`) count their steps from the SQUARED segment length
  (``vtkMath::Distance2BetweenPoints``) while sizing them as a fraction of the radius, so the
  search resolution is off by a factor of the segment length (coarser on segments shorter than 1).
  On the oracles (0.3 mm spacing) the grouping is the same and only the split points move: by up
  to 0.089 mm on the phantom (abscissas likewise; every other point unchanged) and 0.062 mm on the
  case, where 62 of the 637 tracts also differ by one point (a moved split point falls on the
  other side of the 1e-6 mm^2 insertion test): 12182 points with the flag, 12234 without.
- ``vmtk_merge``: ``MergeTracts`` joins consecutive same-group tracts of a centerline, but when
  the run of same-group tracts reaches the centerline's LAST tract, that tract is emitted as a
  second cell of the same group instead of being appended to the merged one. A run can only reach
  the last tract when a daughter was grouped with an upstream tract, so it never triggers on the
  phantom or case oracles; it does on the ``hairpin`` fixture together with ``vmtk_last_tract``
  (vmtk: 5 cells, the corrected merge: 4).
- ``vmtk_last_tract``: ``PointInTubeGroupTracts`` means to take, among the tracts of another
  centerline that pass the mutual in-tube test, the one whose ``centerlineTubeValue`` (this
  tract's tube at that tract's closest center) is smallest - it keeps ``minTubeValue`` /
  ``minCenterlineTubeValue`` for that - but declares both inside the loop over those tracts, so the
  comparison is always against 1e32 and the LAST passing tract in cell order wins. The default
  takes the minimum (the first one on a tie). On the case oracle 25 relabel decisions have more
  than one candidate (always in distinct groups), and the last one is also the deepest in every
  one, so no group changes; the phantom and the other small fixtures never have two. It decides the
  ``hairpin`` fixture: a daughter that curves back to end beside the root of the tree holds the
  root in its tube, is the last candidate for the other centerline's trunk, and vmtk groups it
  with the trunk (that whole centerline collapses into the trunk group, its bifurcation tract
  included); the default groups trunk with trunk (4 groups, as the geometry has). The ``loopback``
  fixture pins the rule itself: three candidates for one decision, and the first, the deepest and
  the last are three different tracts (a first-candidate rule differs from both; on the case it
  gives 587 tracts instead of 637).

Scale. Source-to-tip centerlines repeat every shared segment once per tip, and a tree of N tips has
~N^2 crossings, so the literal loops are quadratic twice over. Nothing here changes a result: work
is shared only where the result provably depends on equal inputs alone. Tube segments are
deduplicated (:class:`_Tubes`); the tube-exit search runs once per distinct crossing segment,
against every cell at once; the touching-sphere walk once per distinct cell prefix; the grouping
test once per pair of distinct tracts (:class:`_MutualInTube`, sparse). The C3N-00704 arteries (537
tips, 268k points) take ~16 s and ~0.4 GB (the dense version was killed at 6.9 GB), the airways
(134 tips) ~2 s and ~0.2 GB instead of 17 s and 3.5 GB.

Design choices kept as they are (not defects): only a crossing from inside (``<= 0``) to outside
(``> 0``) a tube counts ("divergent networks"); the split lands at the end of the sub-step where it
occurs; grouping tests only a tract's FIRST point.

Degenerate radii. A non-finite radius raises ``ValueError`` (vmtk runs, but every comparison with
NaN is false, so its tubes and splits are meaningless). The tube-exit search has no step cap in
vmtk: ``(int)ceil(length / (1e-2 * mean radius))`` is unbounded as the radius goes to 0 (a zero
radius makes it ``(int)inf``, then up to 2^31 tube evaluations). It is capped here at 1e5 steps,
the cap vmtk's own touching-sphere search uses; it binds only where the mean radius of a crossing
segment is below 1e-3 of its length (of its squared length with ``vmtk_steps``): 3e-4 mm at 0.3 mm
spacing (9e-5 mm with ``vmtk_steps``), where vmtk takes more than 1e5 steps per crossing, or never
finishes (radius 0).
"""
from __future__ import annotations

import math

import numpy as np

from ._vtk import (VTK_VMTK_DOUBLE_TOL, VTK_VMTK_FLOAT_TOL, VTK_VMTK_LARGE_DOUBLE, interpolate_edge,
                   round_float32)
from .centerlines import Centerlines
from .polyball import TubeSegments, candidate_pairs, segment_values, tube_segments
from .sphere_distance import find_n_touching_sphere_center, running_pcoords
from .utilities import get_centerline_cell_ids

MAX_TUBE_STEPS = 1e5          # the tube-exit search's step cap (vmtk has none; see the module docstring)

# -- helpers ----------------------------------------------------------------------------------

def _segment_starts(seg_cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(cells that own segments, index of each one's first segment) for a segment table."""
    if len(seg_cell) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    first = np.r_[True, seg_cell[1:] != seg_cell[:-1]]
    starts = np.flatnonzero(first)
    return seg_cell[starts], starts


# -- splitting --------------------------------------------------------------------------------

class _Tubes:
    """Every cell's tube segments, deduplicated: in a tree, centerlines share their prefixes point
    for point, so one (point, segment) value serves every cell holding that segment.

    Three shared computations, each exact (a cache keyed on everything the result depends on):

    - :meth:`inside`: which cells' tubes hold each of some points (the sign of every cell's
      ``EvaluateFunction``), evaluated once per distinct segment; the distinct segments are
      grouped by the set of cells that hold them ("owner classes": the whole shared trunk is one
      class), so a hit expands to cells once per class, not once per segment;
    - :meth:`exit_pcoords`: the tube-exit search on one segment of a line, against every cell at
      once - it depends only on that segment's end points and radii and on the other cell's tube,
      so every centerline through the same segment shares it (a tree of N tips has ~N^2 crossings
      but only as many distinct crossing segments as it has bifurcations, give or take);
    - :meth:`prefix_id`: one id per distinct cell prefix (its segments 0..s), so the touching-sphere
      walk (which reads only points 0..s+1 of its cell) is shared by centerlines with the same
      prefix."""

    def __init__(self, cl: Centerlines, radius: str):
        self.seg = tube_segments(cl, radius)
        self.n_cells = cl.n_cells
        S = len(self.seg)
        self.cell_start = np.zeros(cl.n_cells + 1, np.int64)
        np.add.at(self.cell_start, self.seg.cell + 1, 1)
        self.cell_start = np.cumsum(self.cell_start)
        self._exits: dict = {}
        self._prefix: np.ndarray | None = None
        if S == 0:
            self.useg = self.seg
            self.inv = np.zeros(0, np.int64)
            self.cls = self.cls_ptr = np.zeros(1, np.int64)
            self.cls_cells = np.zeros(0, np.int64)
            return
        rows = np.concatenate([self.seg.p0, self.seg.p1, self.seg.r0[:, None], self.seg.r1[:, None]], axis=1)
        _, first, inv = np.unique(rows, axis=0, return_index=True, return_inverse=True)
        # renumber the distinct segments in order of first appearance, so that consecutive ones are
        # neighbors along a line (candidate_pairs bundles consecutive segments)
        rank = np.empty(len(first), np.int64)
        rank[np.argsort(first, kind="stable")] = np.arange(len(first))
        inv = rank[inv.reshape(-1)]
        first = np.sort(first)
        self.useg = self.seg.take(first)                   # one row per distinct segment
        self.inv = inv                                     # segment -> distinct segment
        order = np.argsort(inv, kind="stable")             # segments grouped by distinct segment
        ptr = np.r_[0, np.cumsum(np.bincount(inv, minlength=len(first)))]
        # owner classes: distinct segments held by the same list of cells
        owners = self.seg.cell[order]
        classes: dict = {}
        cls = np.empty(len(first), np.int64)
        for u in range(len(first)):
            cls[u] = classes.setdefault(owners[ptr[u]:ptr[u + 1]].tobytes(), len(classes))
        self.cls = cls
        members = [np.frombuffer(k, np.int64) for k in classes]
        self.cls_ptr = np.r_[0, np.cumsum([len(m) for m in members])]
        self.cls_cells = np.concatenate(members)

    def cell_segments(self, j: int) -> TubeSegments:
        return self.seg.take(slice(self.cell_start[j], self.cell_start[j + 1]))

    def inside(self, pts: np.ndarray, n_cells: int, exclude: int, chunk: int = 2048) -> np.ndarray:
        """inside[k, j]: cell j's tube holds point k (EvaluateFunction over cell j <= 0), j != exclude.
        Only the sign is used, and every pair left out by candidate_pairs has a value > 0. The
        points go ``chunk`` at a time (each point's row is independent)."""
        out = np.zeros((len(pts), n_cells), dtype=bool)
        if len(self.useg) == 0:
            return out
        ncls = len(self.cls_ptr) - 1
        for q0 in range(0, len(pts), chunk):
            x = pts[q0:q0 + chunk]
            qi, ui = candidate_pairs(x, self.useg)
            v = segment_values(x[qi], self.useg.take(ui), pairwise=True)
            hit = v <= 0.0
            key = np.unique(qi[hit] * ncls + self.cls[ui[hit]])          # (point, owner class)
            q, c = key // ncls, key % ncls
            cnt = self.cls_ptr[c + 1] - self.cls_ptr[c]
            qq = np.repeat(q, cnt)
            base = np.repeat(self.cls_ptr[c] - np.r_[0, np.cumsum(cnt)[:-1]], cnt)
            cells = self.cls_cells[base + np.arange(len(qq))]
            keep = cells != exclude
            out[qq[keep] + q0, cells[keep]] = True
        return out

    def exit_pcoords(self, p0, p1, r0: float, r1: float, vmtk_steps: bool) -> np.ndarray:
        """ComputeCenterlineSplitting's tube-exit search on the segment (p0, r0) -> (p1, r1) of a line,
        against every cell's tube: element j is the pcoord where the search leaves cell j's tube (the
        end of the first sub-step that goes from inside, ``<= 0``, to outside, ``> 0``; the last
        sub-step's end if none). Cached on exactly its inputs.

        The C++ takes the minimum of cell j's segment values at each sub-step point and tests its
        sign; the minimum is <= 0 exactly when some segment's value is, which is :meth:`inside`."""
        key = (p0.tobytes(), p1.tobytes(), float(r0), float(r1), bool(vmtk_steps))
        pcs = self._exits.get(key)
        if pcs is not None:
            return pcs
        e = p0 - p1
        d2 = e[0] * e[0] + e[1] * e[1] + e[2] * e[2]
        seg_len = d2 if vmtk_steps else math.sqrt(d2)
        step_size = 1e-2 * (r0 + r1) / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.float64(seg_len) / np.float64(step_size)
        ratio = MAX_TUBE_STEPS if ratio > MAX_TUBE_STEPS else ratio    # vmtk: no cap (see above)
        n_steps = int(math.ceil(ratio)) if math.isfinite(ratio) else 0
        if n_steps > 0:
            pstep = 1.0 / float(n_steps)
            cp = running_pcoords(n_steps, pstep)
            sub = np.stack([p0[d] + cp * (p1[d] - p0[d]) for d in range(3)], axis=1)
            ins = self.inside(sub, self.n_cells, -1)                     # (n_steps + 1, cells)
            leave = ins[:-1] & ~ins[1:]
            pcs = np.where(leave.any(axis=0), cp[leave.argmax(axis=0) + 1], cp[n_steps])
        else:
            pcs = np.zeros(self.n_cells)
        self._exits[key] = pcs
        return pcs

    def prefix_id(self, cell: int, sub: int) -> int:
        """An id shared exactly by the cells whose segments 0..sub are the same distinct segments
        (so whose points 0..sub+1 and radii are equal)."""
        if self._prefix is None:
            trie: dict = {}
            pid = np.empty(len(self.inv), np.int64)
            inv = self.inv.tolist()
            for c in range(self.n_cells):
                prev = -1
                for s in range(self.cell_start[c], self.cell_start[c + 1]):
                    prev = trie.setdefault((prev, inv[s]), len(trie))
                    pid[s] = prev
            self._prefix = pid
        return int(self._prefix[self.cell_start[cell] + sub])


def compute_centerline_splitting(cl: Centerlines, cell_id: int, radius: str = "MaximumInscribedSphereRadius",
                                 vmtk_steps: bool = False, _tubes: _Tubes | None = None,
                                 _cache: dict | None = None):
    """vtkvmtkCenterlineBranchExtractor::ComputeCenterlineSplitting for one cell.

    Returns ``(sub_ids, pcoords, tract_blanking)``: the splitting points along the cell, sorted,
    and one blanking flag per tract (``len(sub_ids) + 1``)."""
    ids = cl.cells[cell_id]
    n = len(ids)
    pts = cl.points[ids]
    rad_arr = np.asarray(cl.point_data[radius], dtype=np.float64)
    rad_all = rad_arr.reshape(len(cl.points), -1)[:, 0]
    rad = rad_all[ids]
    tubes = _tubes if _tubes is not None else _Tubes(cl, radius)

    # every other cell's tube at every point of this one; a crossing is inside -> outside, and at
    # each one the search for where along the segment the line leaves that tube
    inter_sub: list[int] = []
    inter_pc: list[float] = []
    if n >= 2:
        inside = tubes.inside(pts, cl.n_cells, cell_id)
        cross = inside[:-1] & ~inside[1:]                              # (P-1, n_cells)
        ks, cs = np.nonzero(cross)
        if len(ks):
            pcs = np.empty(len(ks))
            for k in np.unique(ks):
                m = ks == k
                pcs[m] = tubes.exit_pcoords(pts[k], pts[k + 1], rad[k], rad[k + 1], vmtk_steps)[cs[m]]
            # vmtk inserts each intersection into a list it keeps sorted by (sub id, pcoord) (the
            # first slot with prev < new <= next, else the front); by induction the list stays sorted
            # and holds every intersection, so it IS the sorted list (equal entries are equal values)
            o = np.lexsort((pcs, ks))
            inter_sub = ks[o].tolist()
            inter_pc = pcs[o].tolist()

    # for each intersection, the point one sphere upstream (touching forward). The walk reads only
    # the cell's points 0..s+1 and their radii, so it is shared by cells with the same prefix (a
    # one-component radius array: the tube rows then hold exactly the radii the walk reads)
    shared = rad_arr.size == len(cl.points)
    touch_sub: list[int] = []
    touch_pc: list[float] = []
    for s, pc in zip(inter_sub, inter_pc):
        key = (tubes.prefix_id(cell_id, s) if shared else ("cell", cell_id), s, pc, vmtk_steps)
        if _cache is not None and key in _cache:
            ts, tp = _cache[key]
        else:
            ts, tp = find_n_touching_sphere_center(cl, radius, cell_id, s, pc, 1, True, vmtk_steps)
            if _cache is not None:
                _cache[key] = (ts, tp)
        if ts == -1:
            ts, tp = 0, 0.0
        touch_sub.append(ts)
        touch_pc.append(tp)

    def not_after(a_sub, a_pc, b_sub, b_pc):          # a <= b along the cell
        return a_sub < b_sub or (a_sub == b_sub and a_pc <= b_pc)

    split_sub: list[int] = []
    split_pc: list[float] = []
    blanking = [0]
    prev = -1
    nt = len(touch_sub)
    for i in range(nt):
        if i > 0 and not_after(touch_sub[i], touch_pc[i], inter_sub[prev], inter_pc[prev]):
            continue
        split_sub.append(touch_sub[i])
        split_pc.append(touch_pc[i])
        blanking.append(1)
        mx = i
        if i < nt - 1:
            for j in range(i + 1, nt):
                if not_after(touch_sub[j], touch_pc[j], inter_sub[mx], inter_pc[mx]):
                    if (inter_sub[j] > inter_sub[mx]
                            or (inter_sub[j] == inter_sub[mx] and inter_pc[j] >= inter_pc[mx])):
                        mx = j
                else:
                    break
        split_sub.append(inter_sub[mx])
        split_pc.append(inter_pc[mx])
        blanking.append(0)
        prev = mx
    return split_sub, split_pc, blanking


def _split_centerline(cl: Centerlines, cell_id: int, sub_ids, pcoords, blanking, vmtk_float32: bool):
    """vtkvmtkCenterlineSplittingAndGroupingFilter::SplitCenterline for one cell.

    Returns (points (M, 3), point sources (M, 3): [id0, id1, t] with id1 = -1 for a plain copy,
    cells as local point-id arrays, tract ids, blanking flags)."""
    ids = cl.cells[cell_id]
    n = len(ids)
    pts = cl.points[ids]
    nsp = len(sub_ids)
    if nsp == 0:
        src = np.stack([ids.astype(np.float64), np.full(n, -1.0), np.zeros(n)], axis=1)
        # vtkPoints::DeepCopy into the default (VTK_FLOAT) vtkPoints: vtkDataArray::DeepCopy converts
        # to the destination's type, so the unsplit centerline is float32 too
        return (round_float32(pts) if vmtk_float32 else pts.copy()), src, [np.arange(n)], [0], [0]
    out_pts, out_src, cells, tracts, blanks = [], [], [], [], []
    for i in range(nsp + 1):
        lower = 0 if i == 0 else sub_ids[i - 1] + 1
        higher = n - 1 if i == nsp else sub_ids[i]
        cell = []
        if i > 0:
            s = sub_ids[i - 1]
            t = pcoords[i - 1]
            p0, p1 = pts[s], pts[s + 1]
            q = np.array([p0[k] + t * (p1[k] - p0[k]) for k in range(3)])
            e = q - pts[lower]
            if e[0] * e[0] + e[1] * e[1] + e[2] * e[2] > VTK_VMTK_FLOAT_TOL:
                cell.append(len(out_pts))
                out_pts.append(q)
                out_src.append((ids[s], ids[s + 1], t))
        for j in range(lower, higher + 1):
            cell.append(len(out_pts))
            out_pts.append(pts[j])
            out_src.append((ids[j], -1, 0.0))
        if i < nsp:
            s = sub_ids[i]
            t = pcoords[i]
            p0, p1 = pts[s], pts[s + 1]
            q = np.array([p0[k] + t * (p1[k] - p0[k]) for k in range(3)])
            e = q - pts[higher]
            if e[0] * e[0] + e[1] * e[1] + e[2] * e[2] > VTK_VMTK_FLOAT_TOL:
                cell.append(len(out_pts))
                out_pts.append(q)
                out_src.append((ids[s], ids[s + 1], t))
        cells.append(np.asarray(cell, np.int64))
        tracts.append(i)
        blanks.append(blanking[i])
    P = np.array(out_pts, dtype=np.float64).reshape(-1, 3)
    if vmtk_float32:
        P = round_float32(P)                                          # a default (VTK_FLOAT) vtkPoints
    return P, np.array(out_src, dtype=np.float64).reshape(-1, 3), cells, tracts, blanks


def _interpolate_point_data(a: np.ndarray, src: np.ndarray) -> np.ndarray:
    """Copy (``id1 == -1``) or ``InterpolateEdge`` each output row from the input array ``a``."""
    i0 = src[:, 0].astype(np.int64)
    i1 = src[:, 1].astype(np.int64)
    out = a[i0].copy()
    m = i1 >= 0
    if m.any():
        out[m] = interpolate_edge(a, i0[m], i1[m], src[m, 2])
    return out


# -- grouping ---------------------------------------------------------------------------------

class _MutualInTube:
    """PointInTubeGroupTracts' geometric test for every pair of tracts, sparse.

    ``value(i, j)`` is ``centerlineTubeValue`` - tract i's tube evaluated at the center of tract j's
    tube closest to i's first point (``LastPolyBallCenter``) - where that first point is inside j's
    tube (``tubeValue < -1e-12``), and +inf elsewhere; the pair passes the mutual in-tube test where
    ``value < -1e-12``.

    The value depends on the two tracts' geometry only (their points and radii), and source-to-tip
    centerlines repeat the shared tracts once per tip (the trunk's once per centerline), so it is
    computed once per pair of distinct tracts ("tract classes": identical points and radii, compared
    exactly) and looked up for every pair of tracts. Stored: ``tract_class`` (T,) (-1 for a tract with
    no points), ``class_tracts`` (tracts of each class, in cell order), and the class pairs whose
    value is finite, ``(ci, cj, cv)`` sorted by (ci, cj)."""

    def __init__(self, tract_class, class_tracts, ci, cj, cv):
        self.tract_class = tract_class
        self.class_tracts = class_tracts
        self.ci, self.cj, self.cv = ci, cj, cv
        self.row_ptr = np.searchsorted(ci, np.arange(len(class_tracts) + 1))
        self._cand: dict = {}

    def candidates(self, c: int):
        """(tracts j, values) with ``value(i, j) < -1e-12`` for any tract i of class c, j in cell order."""
        got = self._cand.get(c)
        if got is None:
            a, b = self.row_ptr[c], self.row_ptr[c + 1]
            ok = self.cv[a:b] < -VTK_VMTK_DOUBLE_TOL
            cjs, cvs = self.cj[a:b][ok], self.cv[a:b][ok]
            if len(cjs):
                j = np.concatenate([self.class_tracts[k] for k in cjs])
                v = np.repeat(cvs, [len(self.class_tracts[k]) for k in cjs])
                o = np.argsort(j, kind="stable")
                got = (j[o], v[o])
            else:
                got = (np.zeros(0, np.int64), np.zeros(0))
            self._cand[c] = got
        return got

    def dense(self) -> np.ndarray:
        """The (T, T) value matrix (for inspection and tests; the grouping never builds it)."""
        T = len(self.tract_class)
        out = np.full((T, T), np.inf)
        for a, b, v in zip(self.ci, self.cj, self.cv):
            out[np.ix_(self.class_tracts[a], self.class_tracts[b])] = v
        return out


def _mutual_in_tube(tr: Centerlines, radius: str, chunk: int = 256) -> _MutualInTube:
    """Evaluate :class:`_MutualInTube` on the tract classes.

    Only pairs candidate_pairs keeps are evaluated: a dropped segment's value is > 0, so it can be
    neither a minimum below -1e-12 nor that minimum's first occurrence. The first points go ``chunk``
    at a time (every (point, tract) group lies within one chunk)."""
    T = tr.n_cells
    rad = np.asarray(tr.point_data[radius], dtype=np.float64).reshape(-1)     # as tube_segments reads it
    tract_class = np.full(T, -1, np.int64)
    keys: dict = {}
    for i in range(T):
        ids = tr.cells[i]
        if len(ids):
            tract_class[i] = keys.setdefault((tr.points[ids].tobytes(), rad[ids].tobytes()), len(keys))
    TC = len(keys)
    rep = np.full(TC, -1, np.int64)
    members: list[list[int]] = [[] for _ in range(TC)]
    for i in range(T):
        c = tract_class[i]
        if c >= 0:
            members[c].append(i)
            if rep[c] < 0:
                rep[c] = i
    class_tracts = [np.asarray(m, np.int64) for m in members]
    empty = np.zeros(0, np.int64)
    seg = tube_segments(tr, radius, rep)                  # the representatives' segments, in class order
    if len(seg) == 0 or TC == 0:
        return _MutualInTube(tract_class, class_tracts, empty, empty, np.zeros(0))
    seg_cls = tract_class[seg.cell]
    owners, starts = _segment_starts(seg_cls)
    ends = np.r_[starts[1:], len(seg)]
    seg_by_class = {int(c): seg.take(slice(s, e)) for c, s, e in zip(owners, starts, ends)}
    x_all = tr.points[[tr.cells[i][0] for i in rep]]
    out_i, out_j, out_v = [], [], []
    for q0 in range(0, TC, chunk):
        x = x_all[q0:q0 + chunk]
        qi, si = candidate_pairs(x, seg)                 # sorted by (point, segment) = (point, class, sub)
        if len(qi) == 0:
            continue
        vals, (c0, c1, c2, _c3, _t) = segment_values(x[qi], seg.take(si), return_center=True, pairwise=True)
        key = qi * TC + seg_cls[si]
        gs = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])  # one group per (point, tract j)
        size = np.diff(np.r_[gs, len(key)])
        mins = np.minimum.reduceat(vals, gs)
        pos = np.arange(len(vals))
        first = np.minimum.reduceat(np.where(vals == np.repeat(mins, size), pos, len(vals)), gs)
        good = mins < -VTK_VMTK_DOUBLE_TOL
        gi = qi[gs[good]] + q0                           # class of tract i
        gj = seg_cls[si[gs[good]]]                       # class of tract j
        f = first[good]
        centers = np.stack([c0[f], c1[f], c2[f]], axis=1)
        for i in np.unique(gi):
            own = seg_by_class.get(int(i))
            if own is None:
                continue
            m = gi == i
            cv = np.minimum(segment_values(centers[m], own).min(axis=1), VTK_VMTK_LARGE_DOUBLE)
            out_i.append(np.full(int(m.sum()), i, np.int64))
            out_j.append(gj[m])
            out_v.append(cv)
    if not out_i:
        return _MutualInTube(tract_class, class_tracts, empty, empty, np.zeros(0))
    ci, cj, cv = np.concatenate(out_i), np.concatenate(out_j), np.concatenate(out_v)
    o = np.lexsort((cj, ci))
    return _MutualInTube(tract_class, class_tracts, ci[o], cj[o], cv[o])


def _point_in_tube_group_tracts(tr: Centerlines, n_input_cells: int, radius: str, centerline_ids: str,
                                blanking: str, vmtk_last_tract: bool = False) -> np.ndarray:
    """vtkvmtkCenterlineSplittingAndGroupingFilter::PointInTubeGroupTracts -> group ids (int32).

    The geometric test does not depend on the groups, so it is evaluated once for every pair; the
    loops then replay vmtk's order: only (tract, centerline) pairs with a candidate can relabel, and
    the others are skipped without being visited. Among the candidates the one with the smallest
    ``centerlineTubeValue`` wins (the first on a tie); ``vmtk_last_tract=True``: the last one in cell
    order (see the module docstring).

    "Centerline k already has a tract in this group" is kept as a set of centerlines per group,
    merged on each relabel (every tract of the relabeled group joins the current one), instead of
    scanning centerline k's tracts."""
    T = tr.n_cells
    cl_ids = np.asarray(tr.cell_data[centerline_ids]).astype(np.int64)
    blank = np.asarray(tr.cell_data[blanking]).astype(np.int64)
    group = np.arange(T, dtype=np.int64)
    lines_of = {g: {int(cl_ids[g])} for g in range(T)}       # group -> centerlines with a tract in it
    mt = _mutual_in_tube(tr, radius)
    for i in range(T):
        c = mt.tract_class[i]
        if c < 0:
            continue
        cg = group[i]
        cid = cl_ids[i]
        j, v = mt.candidates(int(c))
        ok = (blank[j] == blank[i]) & (cl_ids[j] != cid) & (cl_ids[j] < n_input_cells) & (cl_ids[j] >= 0)
        j, v = j[ok], v[ok]
        if len(j) == 0:
            continue
        k = cl_ids[j]
        if vmtk_last_tract:
            o = np.lexsort((j, k))                          # by centerline, then cell order
            pick = np.r_[k[o][1:] != k[o][:-1], True]       # the last candidate of each centerline
        else:
            o = np.lexsort((j, v, k))                       # by centerline, value, cell order
            pick = np.r_[True, k[o][1:] != k[o][:-1]]       # the deepest: first minimum
        for kk, jj in zip(k[o][pick].tolist(), j[o][pick].tolist()):
            if kk in lines_of[cg]:                         # centerline k already has a tract in this group
                continue
            same = int(group[jj])
            group[group == same] = cg
            a, b = lines_of.pop(same), lines_of[cg]
            if len(a) > len(b):
                a |= b
                lines_of[cg] = a
            else:
                b |= a
    return group.astype(np.int32)


def _merge_tracts(tr: Centerlines, group_ids: str, centerline_ids: str, tract_ids: str,
                  vmtk_merge: bool) -> Centerlines:
    """vtkvmtkCenterlineSplittingAndGroupingFilter::MergeTracts (points and point data unchanged)."""
    T = tr.n_cells
    groups = tr.cell_data[group_ids]            # modified in place, as vmtk does
    present = set(int(c) for c in tr.cell_data[centerline_ids])
    cells: list[np.ndarray] = []
    rows: list[int] = []
    for i in range(T):
        if i not in present:
            continue
        cc = get_centerline_cell_ids(tr, i, centerline_ids, tract_ids)
        nct = len(cc)
        for j in range(nct):
            g = groups[cc[j]]
            for k in range(nct - 1, j, -1):
                if g == groups[cc[k]]:
                    for m in range(j + 1, k):
                        groups[cc[m]] = g
        prev = -1
        first = 0
        merged: list[int] = []
        for j in range(nct):
            tc = cc[j]
            tp = tr.cells[tc]
            g = int(groups[tc])
            last = j == nct - 1
            if g != prev or (last and vmtk_merge):
                if prev != -1:
                    cells.append(np.asarray(merged, np.int64))
                    rows.append(first)
                first = tc
                merged = list(tp)
                if last:
                    cells.append(np.asarray(merged, np.int64))
                    rows.append(first)
            else:
                merged.extend(tp[1:])
                if last:
                    cells.append(np.asarray(merged, np.int64))
                    rows.append(first)
            prev = g
    rows_a = np.asarray(rows, np.int64)
    cd = {k: v[rows_a] for k, v in tr.cell_data.items()}
    return Centerlines(tr.points, cells, dict(tr.point_data), cd)


def _make_group_ids_adjacent(groups: np.ndarray) -> np.ndarray:
    """MakeGroupIdsAdjacent: vmtk's repeated "smallest id >= current -> current" relabeling, which
    for the non-negative ids it sees is the rank of each id among the distinct ids."""
    _, inv = np.unique(groups, return_inverse=True)
    return inv.reshape(groups.shape).astype(groups.dtype)


def _make_tract_ids_adjacent(cl_ids: np.ndarray, tract: np.ndarray) -> np.ndarray:
    """MakeTractIdsAdjacent: per centerline, tract ids renumbered 0.. in increasing order."""
    out = tract.copy()
    for c in dict.fromkeys(int(x) for x in cl_ids):
        m = cl_ids == c
        _, inv = np.unique(tract[m], return_inverse=True)
        out[m] = inv.reshape(-1).astype(tract.dtype)
    return out


def _split_tracts(cl: Centerlines, radius: str, centerline_ids: str, tract_ids: str, blanking: str,
                  vmtk_steps: bool, vmtk_float32: bool) -> Centerlines:
    """RequestData up to the grouping: every input cell split (ComputeCenterlineSplitting +
    SplitCenterline) and the pieces appended (vtkAppendPolyData): the tracts, before grouping."""
    cache: dict = {}
    tubes = _Tubes(cl, radius)
    all_pts, all_src, cells, cl_col, tract_col, blank_col, src_cell = [], [], [], [], [], [], []
    off = 0
    for i in range(cl.n_cells):
        sub, pc, bl = compute_centerline_splitting(cl, i, radius, vmtk_steps, tubes, cache)
        P, src, pc_cells, tids, blanks = _split_centerline(cl, i, sub, pc, bl, vmtk_float32)
        all_pts.append(P)
        all_src.append(src)
        for c, t, b in zip(pc_cells, tids, blanks):
            cells.append(c + off)
            cl_col.append(i)
            tract_col.append(t)
            blank_col.append(b)
            src_cell.append(i)
        off += len(P)
    points = np.concatenate(all_pts) if all_pts else np.zeros((0, 3))
    src = np.concatenate(all_src) if all_src else np.zeros((0, 3))
    pd = {k: _interpolate_point_data(np.asarray(v), src) for k, v in cl.point_data.items()}
    sc = np.asarray(src_cell, np.int64)
    cd = {k: np.asarray(v)[sc] for k, v in cl.cell_data.items()}
    cd[centerline_ids] = np.asarray(cl_col, np.int32)
    cd[tract_ids] = np.asarray(tract_col, np.int32)
    cd[blanking] = np.asarray(blank_col, np.int32)
    return Centerlines(points, cells, pd, cd)


# -- the filter -------------------------------------------------------------------------------

def extract_branches(cl: Centerlines, radius: str = "MaximumInscribedSphereRadius",
                     group_ids: str = "GroupIds", centerline_ids: str = "CenterlineIds",
                     tract_ids: str = "TractIds", blanking: str = "Blanking", vmtk_steps: bool = False,
                     vmtk_merge: bool = False, vmtk_float32: bool = False,
                     vmtk_last_tract: bool = False) -> Centerlines:
    """vmtkbranchextractor: split each centerline where it leaves the other centerlines' tubes (and
    one sphere upstream, blanking the bifurcation tract in between), group the tracts by mutual
    tube containment, merge, and renumber groups and tracts. Returns a new :class:`Centerlines`.

    ``vmtk_steps``, ``vmtk_merge``, ``vmtk_float32`` and ``vmtk_last_tract`` reproduce vmtk (see the
    module docstring)."""
    if radius not in cl.point_data:
        raise KeyError(f"radius array {radius!r} not in point data")
    bad = ~np.isfinite(np.asarray(cl.point_data[radius], dtype=np.float64))
    if bad.any():
        raise ValueError(f"radius array {radius!r} has {int(bad.sum())} non-finite value(s) (first at point "
                         f"{int(np.flatnonzero(bad.reshape(len(bad), -1).any(axis=1))[0])}); vmtk's result "
                         "is meaningless there")

    tracts = _split_tracts(cl, radius, centerline_ids, tract_ids, blanking, vmtk_steps, vmtk_float32)
    tracts.cell_data[group_ids] = _point_in_tube_group_tracts(tracts, cl.n_cells, radius, centerline_ids,
                                                              blanking, vmtk_last_tract)
    out = _merge_tracts(tracts, group_ids, centerline_ids, tract_ids, vmtk_merge)
    out.cell_data[group_ids] = _make_group_ids_adjacent(out.cell_data[group_ids])
    out.cell_data[tract_ids] = _make_tract_ids_adjacent(out.cell_data[centerline_ids].astype(np.int64),
                                                        out.cell_data[tract_ids])
    return out
