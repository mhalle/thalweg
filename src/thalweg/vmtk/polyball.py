"""vtkvmtkPolyBallLine and vtkvmtkMath::EvaluateSphereFunction, vectorized over query points.

``vtkvmtkPolyBallLine`` (vtkVmtk/ComputationalGeometry/vtkvmtkPolyBallLine.cxx) is vmtk's tube
function: the envelope of spheres swept along polyline segments with linearly varying radius. For a
point ``x`` and a segment (p0, r0) -> (p1, r1) it projects in the "complex" metric (the radius as a
fourth, negative-signature coordinate), clamps ``t`` to [0, 1] with a 1e-12 snap, and returns
``|x - c(t)|^2 - r(t)^2``; the function value is the minimum over the segments of the selected cells
(first minimum wins; segments with ``|den| < 1e-12`` are skipped; 1e32 if none is left), and the
minimizing segment's center is kept as ``LastPolyBallCenter``.

Public:

- :func:`tube_segments`: the segment table (p0, p1, r0, r1, owning cell, sub id) of some cells, in
  vmtk's iteration order;
- :func:`segment_values`: the per-segment function values (Q, S) for Q query points, with the same
  arithmetic, operation order and snapping as the C++ (``return_center=True`` also gives the
  per-segment centers, radii and ``t``);
- :func:`candidate_pairs`: a conservative pruning (point, segment) -> "value may be <= 0", so a
  sign test over many points and segments evaluates only the pairs that can matter;
- :func:`evaluate_function`: ``EvaluateFunction`` for Q points over a segment table - the value and
  the ``LastPolyBall*`` state (cell, sub id, pcoord, center, radius) per point. With
  ``tube_segments(cl, None, cells)`` (``UseRadiusInformation`` off) it is the plain closest point
  on the cells' polylines that the offset filter (:mod:`.offset`) asks for;
- :func:`sphere_function`: ``vtkvmtkMath::EvaluateSphereFunction`` (vtkVmtk/Common/vtkvmtkMath.cxx),
  broadcasting.

Vectorizing changes nothing: each (point, segment) value is computed with the C++'s operations in
the C++'s order, and a first-occurrence argmin is the C++'s strict-``<`` scan.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._vtk import VTK_VMTK_DOUBLE_TOL, VTK_VMTK_LARGE_DOUBLE
from .centerlines import Centerlines

PAIR_CHUNK = 1 << 21        # candidate_pairs' working set: about this many (point, bundle) distances


@dataclass
class TubeSegments:
    """The segments of some cells, in the order vtkvmtkPolyBallLine visits them."""

    p0: np.ndarray      # (S, 3)
    p1: np.ndarray      # (S, 3)
    r0: np.ndarray      # (S,)
    r1: np.ndarray      # (S,)
    cell: np.ndarray    # (S,) owning cell id
    sub: np.ndarray     # (S,) segment index within its cell

    def __len__(self) -> int:
        return len(self.r0)

    def take(self, mask) -> "TubeSegments":
        return TubeSegments(self.p0[mask], self.p1[mask], self.r0[mask], self.r1[mask],
                            self.cell[mask], self.sub[mask])


def tube_segments(cl: Centerlines, radius: str, cell_ids=None) -> TubeSegments:
    """Segments of ``cell_ids`` (default: every cell), in order: cells as listed, segments along each.
    ``radius=None`` is ``UseRadiusInformation`` off (all radii 0)."""
    if cell_ids is None:
        cell_ids = range(cl.n_cells)
    rad = None if radius is None else np.asarray(cl.point_data[radius], dtype=np.float64).reshape(-1)
    ids0, ids1, cells, subs = [], [], [], []
    for c in cell_ids:
        ids = cl.cells[c]
        n = len(ids) - 1
        if n <= 0:
            continue
        ids0.append(ids[:-1])
        ids1.append(ids[1:])
        cells.append(np.full(n, c, np.int64))
        subs.append(np.arange(n, dtype=np.int64))
    if not ids0:
        z = np.zeros((0, 3))
        e = np.zeros(0)
        ei = np.zeros(0, np.int64)
        return TubeSegments(z, z.copy(), e, e.copy(), ei, ei.copy())
    i0 = np.concatenate(ids0)
    i1 = np.concatenate(ids1)
    r0 = rad[i0] if rad is not None else np.zeros(len(i0))
    r1 = rad[i1] if rad is not None else np.zeros(len(i1))
    return TubeSegments(cl.points[i0], cl.points[i1], r0, r1, np.concatenate(cells), np.concatenate(subs))


def segment_values(x, seg: TubeSegments, return_center: bool = False, pairwise: bool = False):
    """Per-segment polyball values for query points ``x`` (Q, 3): (Q, S), +inf where the C++ skips
    the segment (``|den| < 1e-12``). With ``return_center``: also (cx, cy, cz, cr, t), each (Q, S).
    ``pairwise=True``: point q against segment q only (Q == S), each output (Q,)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1, 3)
    if pairwise:
        X0, X1, X2 = x[:, 0], x[:, 1], x[:, 2]
    else:
        X0, X1, X2 = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    a0, a1 = seg.p0, seg.p1
    v00 = a1[:, 0] - a0[:, 0]
    v01 = a1[:, 1] - a0[:, 1]
    v02 = a1[:, 2] - a0[:, 2]
    v03 = seg.r1 - seg.r0
    v10 = X0 - a0[:, 0]
    v11 = X1 - a0[:, 1]
    v12 = X2 - a0[:, 2]
    v13 = 0.0 - seg.r0
    num = v00 * v10 + v01 * v11 + v02 * v12 - v03 * v13
    den = v00 * v00 + v01 * v01 + v02 * v02 - v03 * v03
    skip = np.abs(den) < VTK_VMTK_DOUBLE_TOL
    with np.errstate(divide="ignore", invalid="ignore"):
        t = num / den
    lo = t < VTK_VMTK_DOUBLE_TOL
    hi = ~lo & (1.0 - t < VTK_VMTK_DOUBLE_TOL)
    t = np.where(lo, 0.0, np.where(hi, 1.0, t))
    c0 = np.where(lo, a0[:, 0], np.where(hi, a1[:, 0], a0[:, 0] + t * v00))
    c1 = np.where(lo, a0[:, 1], np.where(hi, a1[:, 1], a0[:, 1] + t * v01))
    c2 = np.where(lo, a0[:, 2], np.where(hi, a1[:, 2], a0[:, 2] + t * v02))
    c3 = np.where(lo, seg.r0, np.where(hi, seg.r1, seg.r0 + t * v03))
    d0 = X0 - c0
    d1 = X1 - c1
    d2 = X2 - c2
    val = d0 * d0 + d1 * d1 + d2 * d2 - c3 * c3
    val = np.where(skip, np.inf, val)
    if return_center:
        return val, (c0, c1, c2, c3, t)
    return val


def candidate_pairs(x, seg: TubeSegments, block: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """(point index, segment index) pairs whose polyball value may be <= 0; every other pair's
    value is certainly > 0 (a pruning that cannot change any sign test). Sorted by (point, segment).

    Consecutive segments are bundled ``block`` at a time into a ball holding all their endpoints
    (center ``C``, radius ``rho``) with ``R`` the largest ``|radius|`` among them. Every center
    ``c(t)`` of those segments is within ``rho`` of ``C`` and every ``r(t)`` within ``R``, so the value
    is at least ``(|x - C| - rho)^2 - R^2``; a pair is dropped only when ``|x - C|`` exceeds
    ``bound = rho + R`` by a relative 1e-6 (+1e-9), far above the rounding of the value itself.

    The points are bundled the same way (``block`` consecutive query points in a ball of center
    ``Cq``, radius ``rq``) to skip whole (point bundle, segment bundle) blocks first: a point bundle
    can hold a kept pair only if ``|Cq - C| <= bound + rq`` (triangle inequality; tested with the same
    relative margin), and every pair in a surviving block is then tested exactly as above - so the
    pairs are the same as a test of every (point, segment bundle) pair."""
    x = np.asarray(x, dtype=np.float64).reshape(-1, 3)
    S = len(seg)
    Q = len(x)
    if S == 0 or Q == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    C, rho = _bundle(seg.p0, seg.p1, block)
    nb = len(C)
    rr = np.maximum(np.abs(seg.r0), np.abs(seg.r1))
    R = np.concatenate([rr, np.repeat(rr[-1:], nb * block - S)]).reshape(nb, block).max(axis=1)
    bound = (rho + R) * (1.0 + 1e-6) + 1e-9
    Cq, rq = _bundle(x, x, block)
    # coarse: (point bundle, segment bundle) blocks, the point bundles in chunks so the (chunk,
    # bundles, 3) difference stays near ``PAIR_CHUNK`` values
    step = max(1, PAIR_CHUNK // nb)
    qs, bs = [], []
    for q0 in range(0, len(Cq), step):
        d = np.sqrt(((Cq[q0:q0 + step, None, :] - C[None, :, :]) ** 2).sum(-1))
        qb, bi = np.nonzero(d <= (bound[None, :] + rq[q0:q0 + step, None]) * (1.0 + 1e-6) + 1e-9)
        qs.append(qb + q0)
        bs.append(bi)
    qb = np.concatenate(qs)
    bi = np.concatenate(bs)
    # fine: every point of a surviving block against its segment bundle, the original test
    qi = (qb[:, None] * block + np.arange(block)[None, :]).reshape(-1)
    bi = np.repeat(bi, block)
    ok = qi < Q
    qi, bi = qi[ok], bi[ok]
    d = np.sqrt(((x[qi] - C[bi]) ** 2).sum(-1))
    ok = d <= bound[bi]
    qi, bi = qi[ok], bi[ok]
    o = np.lexsort((bi, qi))
    qi, bi = qi[o], bi[o]
    si = (bi[:, None] * block + np.arange(block)[None, :]).reshape(-1)
    qi = np.repeat(qi, block)
    keep = si < S
    return qi[keep], si[keep]


def _bundle(a: np.ndarray, b: np.ndarray, block: int) -> tuple[np.ndarray, np.ndarray]:
    """Balls holding ``block`` consecutive rows of ``a`` and ``b`` (padded with the last row): the
    bounding box center and the largest distance from it."""
    n = len(a)
    nb = -(-n // block)
    pad = nb * block - n
    e = np.concatenate([a, np.repeat(a[-1:], pad, 0)]).reshape(nb, block, 3)
    f = np.concatenate([b, np.repeat(b[-1:], pad, 0)]).reshape(nb, block, 3)
    both = np.concatenate([e, f], axis=1)                               # (nb, 2*block, 3)
    C = 0.5 * (both.min(axis=1) + both.max(axis=1))
    rho = np.sqrt(((both - C[:, None, :]) ** 2).sum(-1)).max(axis=1)
    return C, rho


@dataclass
class PolyBallState:
    """``EvaluateFunction``'s value and ``LastPolyBall*`` state, one row per query point."""

    value: np.ndarray       # (Q,)
    cell: np.ndarray        # (Q,) LastPolyBallCellId (-1 if no segment)
    sub: np.ndarray         # (Q,) LastPolyBallCellSubId
    pcoord: np.ndarray      # (Q,) LastPolyBallCellPCoord
    center: np.ndarray      # (Q, 3) LastPolyBallCenter
    radius: np.ndarray      # (Q,) LastPolyBallCenterRadius


def evaluate_function(x, seg: TubeSegments) -> PolyBallState:
    """vtkvmtkPolyBallLine::EvaluateFunction at each query point over one segment table."""
    x = np.asarray(x, dtype=np.float64).reshape(-1, 3)
    q = len(x)
    out = PolyBallState(np.full(q, VTK_VMTK_LARGE_DOUBLE), np.full(q, -1, np.int64),
                        np.full(q, -1, np.int64), np.zeros(q), np.zeros((q, 3)), np.zeros(q))
    if len(seg) == 0 or q == 0:
        return out
    val, (c0, c1, c2, c3, t) = segment_values(x, seg, return_center=True)
    k = np.argmin(val, axis=1)                         # first occurrence = strict-< scan
    rows = np.arange(q)
    best = val[rows, k]
    ok = best < VTK_VMTK_LARGE_DOUBLE
    out.value[ok] = best[ok]
    kk = k[ok]
    rr = rows[ok]
    out.cell[ok] = seg.cell[kk]
    out.sub[ok] = seg.sub[kk]
    out.pcoord[ok] = t[rr, kk]
    out.center[ok] = np.stack([c0[rr, kk], c1[rr, kk], c2[rr, kk]], axis=1)
    out.radius[ok] = c3[rr, kk]
    return out


def sphere_function(center, radius, point):
    """vtkvmtkMath::EvaluateSphereFunction: |center - point|^2 - radius^2, summed x, y, z in order."""
    center = np.asarray(center, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    d0 = center[..., 0] - point[..., 0]
    d1 = center[..., 1] - point[..., 1]
    d2 = center[..., 2] - point[..., 2]
    return d0 * d0 + d1 * d1 + d2 * d2 - radius * radius
