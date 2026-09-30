"""vmtkbranchmetrics: a point's coordinates along and around its branch.

``vmtkbranchmetrics`` gives every surface point of a branch group two numbers
(``vtkvmtkPolyDataCenterlineAbscissaMetricFilter``, ``...AngularMetricFilter``, both built on
``vtkvmtkPolyDataCenterlineMetricFilter``):

- ``AbscissaMetric``: the abscissa of the centerline nearest the point;
- ``AngularMetric``: the angle around the centerline, from the centerline's normal to the point,
  in [-pi, pi].

Both are point-wise formulas, so they need no surface: :func:`branch_metrics` evaluates them at any
points that carry a group. For a point of group g, every centerline cell of g gives a closest point
(the polyball search with ``UseRadiusInformation`` off: plain Euclidean), and the closest point,
the segment's direction, the normal and the abscissa there are averaged over those cells with
weights radius^2. The abscissa uses each cell together with the blanked cells next to it on the
same centerline (``IncludeBifurcations`` on), the angle the cell alone. The angle is between the
point's offset from the averaged center and the averaged normal, both projected on the plane normal
to the averaged tangent, negative where tangent . (offset x normal) < 0.

``vmtk_interp``: vmtk interpolates the radius, the abscissa and the normal along a segment with
``vtkvmtkCenterlineUtilities::InterpolateTuple``, which returns the segment's END value whatever
the position (see :mod:`.utilities`). The default interpolates; ``vmtk_interp=True`` reproduces
vmtk, for oracle comparison only.

The input is the split centerlines after ``offset_attributes`` (abscissas and normals in vmtk's
offset convention), as vmtk's own chain feeds it.
"""
from __future__ import annotations

import numpy as np

from .centerlines import (ABSCISSAS, BLANKING, CENTERLINE_IDS, GROUP_IDS, NORMALS, RADIUS, TRACT_IDS,
                          Centerlines)
from .polyball import evaluate_function, tube_segments


def group_cell_sets(cl: Centerlines, group: int, include_bifurcations: bool) -> list[list[int]]:
    """One set per cell of ``group``: the cell, and with ``include_bifurcations`` the blanked cells
    of other groups next to it (tract id +-1) on its centerline."""
    gid = np.asarray(cl.cell_data[GROUP_IDS]).reshape(-1)
    blank = np.asarray(cl.cell_data[BLANKING]).reshape(-1)
    line = np.asarray(cl.cell_data[CENTERLINE_IDS]).reshape(-1)
    tract = np.asarray(cl.cell_data[TRACT_IDS]).reshape(-1)
    sets = []
    for i in np.nonzero(gid == group)[0]:
        adj = []
        if include_bifurcations:
            adj = np.nonzero((blank == 1) & (gid != group) & (line == line[i])
                             & (np.abs(tract - tract[i]) == 1))[0].tolist()
        sets.append([int(i), *adj])
    return sets


def _average(x: np.ndarray, cl: Centerlines, group: int, include_bifurcations: bool, vmtk_interp: bool):
    """Radius^2-weighted abscissa, center, tangent and normal over the cell sets of ``group``."""
    rad = np.asarray(cl.point_data[RADIUS], float).reshape(-1)
    absc = np.asarray(cl.point_data[ABSCISSAS], float).reshape(-1)
    nrm = np.asarray(cl.point_data[NORMALS], float).reshape(-1, 3)
    n = len(x)
    W, A = np.zeros(n), np.zeros(n)
    C, T, N = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3))
    for cells in group_cell_sets(cl, group, include_bifurcations):
        st = evaluate_function(x, tube_segments(cl, None, cells))
        ok = st.cell >= 0
        for u in np.unique(st.cell[ok]):
            s = ok & (st.cell == u)
            ids = cl.cells[u]
            t = st.pcoord[s]
            a0, a1 = ids[st.sub[s]], ids[st.sub[s] + 1]
            if vmtk_interp:                                   # the END point's value, whatever t
                r, ab, nn = rad[a1], absc[a1], nrm[a1]
            else:
                r = rad[a0] * (1 - t) + rad[a1] * t
                ab = absc[a0] * (1 - t) + absc[a1] * t
                nn = nrm[a0] * (1 - t)[:, None] + nrm[a1] * t[:, None]
            w = r * r
            p0, p1 = cl.points[a0], cl.points[a1]
            C[s] += w[:, None] * (p0 * (1 - t)[:, None] + p1 * t[:, None])
            T[s] += w[:, None] * (p1 - p0)
            N[s] += w[:, None] * nn
            A[s] += w * ab
            W[s] += w
    with np.errstate(invalid="ignore", divide="ignore"):
        return A / W, C / W[:, None], T, N


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return v / n


def branch_metrics(x, groups, cl: Centerlines, vmtk_interp: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """``(AbscissaMetric, AngularMetric)`` for points ``x`` (Q, 3) whose branch groups are ``groups``
    (Q,) - from :func:`thalweg.vmtk.partition.lowest_label`, or a clipped surface's ``GroupIds``.
    NaN for a point whose group has no cell, or whose offset or normal has no component across the
    tangent (the point is on the centerline). A group that exists only as blanked cells still gets
    values (vmtk's filters do not look at blanking for the group's own cells)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1, 3)
    groups = np.asarray(groups).reshape(-1)
    abscissa = np.full(len(x), np.nan)
    angle = np.full(len(x), np.nan)
    have = set(np.asarray(cl.cell_data[GROUP_IDS]).reshape(-1).tolist())
    for g in np.unique(groups):
        if int(g) not in have:
            continue
        s = groups == g
        xs = x[s]
        abscissa[s] = _average(xs, cl, int(g), True, vmtk_interp)[0]
        _, C, T, N = _average(xs, cl, int(g), False, vmtk_interp)
        T, N = _unit(T), _unit(N)
        pos = xs - C
        pos = _unit(pos - (pos * T).sum(1, keepdims=True) * T)
        pn = _unit(N - (N * T).sum(1, keepdims=True) * T)
        with np.errstate(invalid="ignore"):
            a = np.arccos(np.clip((pos * pn).sum(1), -1.0, 1.0))
            angle[s] = np.where((T * np.cross(pos, pn)).sum(1) < 0, -a, a)
    return abscissa, angle
