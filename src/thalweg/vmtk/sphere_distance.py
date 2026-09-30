"""vtkvmtkCenterlineSphereDistance: walk along a centerline to the point one inscribed sphere away.

Ports the two static functions of vtkvmtkCenterlineSphereDistance
(vtkVmtk/ComputationalGeometry/vtkvmtkCenterlineSphereDistance.cxx), line for line:

- :func:`find_touching_sphere_center`: for the point at (cell, sub id, pcoord), the location along
  the same cell where the sphere (center on the line, radius interpolated linearly) starts to hold
  the point: with ``forward=True`` scanning the segments before it from the cell's start (the first
  one whose start sphere misses the point and whose end sphere holds it), with ``forward=False``
  the segments after it from the cell's end (mirrored); then a linear search within that segment;
- :func:`find_n_touching_sphere_center`: that step repeated ``n`` times (``(-1, 0.0)`` once it
  runs off the line).

Both return ``(touching_sub_id, touching_pcoord)``; ``touching_sub_id == -1`` means none.

:func:`running_pcoords` is vmtk's running parametric coordinate (``c += step``), shared with the
branch extractor's tube-exit search.

One implementation serves every filter that walks spheres: the branch extractor
(:mod:`.branches`), the bifurcation vectors (:mod:`.vectors`) and the branch geometry
(:mod:`.branch_geometry`).

**The squared step length defect** (``vmtk_steps``). The in-segment linear search sizes its steps
as ``1e-6 * mean radius`` but counts them from ``vtkMath::Distance2BetweenPoints`` - the SQUARED
segment length - so the resolution is not what it says (finer on segments longer than 1, coarser on
shorter ones; the count is capped at 1e5 either way). ``vmtk_steps=False`` (default) counts steps
from the segment length; ``vmtk_steps=True`` reproduces vmtk, for oracle comparison. A zero mean
radius gives an infinite ratio (1e5 steps) and a NaN one no search (pcoord 0), as the arm64 build
does. A cell with fewer than two points raises ``ValueError`` (the C++ reads past its end).

The in-segment search is vectorized: vmtk's running pcoord ``c += step`` is ``np.cumsum`` (a
sequential accumulate, the same additions in the same order), and ``c + step`` at step i IS the
running value at step i + 1, so one array of sphere values holds both of vmtk's per-step values.
"""
from __future__ import annotations

import math

import numpy as np

from .centerlines import Centerlines
from .polyball import sphere_function

MAX_NUMBER_OF_STEPS = 1e5


def running_pcoords(n: int, step: float) -> np.ndarray:
    """vmtk's running pcoord at steps 0..n (0, step, step+step, ...), accumulated as the C++ does."""
    out = np.empty(n + 1)
    out[0] = 0.0
    if n:
        out[1:] = np.cumsum(np.full(n, step))
    return out


def find_touching_sphere_center(cl: Centerlines, radius: str, cell_id: int, sub_id: int, pcoord: float,
                                forward: bool = True, vmtk_steps: bool = False) -> tuple[int, float]:
    """vtkvmtkCenterlineSphereDistance::FindTouchingSphereCenter -> (touching sub id, pcoord).

    The touching point is where the centerline's inscribed sphere (center on the line, radius
    interpolated linearly along the segment) passes through the query point at (sub_id, pcoord):
    ``forward=True`` scans segments 0 .. sub_id-1 from the start for the first one whose start
    sphere excludes the point and whose end sphere contains it; ``forward=False`` scans segments
    n-2 .. sub_id+1 from the end for the first one whose start sphere contains it and whose end
    sphere excludes it. Without such a segment the query segment itself is tried, unless both its
    end spheres contain the point (-> -1). That segment is then searched linearly in pcoord (see the
    module docstring for the step); a hit on the query segment on the wrong side of ``pcoord``
    gives -1."""
    ids = cl.cells[cell_id]
    n = len(ids)
    if n < 2:
        raise ValueError(f"cell {cell_id} has {n} point(s); FindTouchingSphereCenter needs a line")
    pts = cl.points[ids]
    rad = np.asarray(cl.point_data[radius], dtype=np.float64).reshape(len(cl.points), -1)[ids, 0]
    if forward:
        sub_ids = list(range(0, sub_id))
    else:
        sub_ids = list(range(n - 2, sub_id, -1))

    p0, p1 = pts[sub_id], pts[sub_id + 1]
    point = np.array([p0[k] + pcoord * (p1[k] - p0[k]) for k in range(3)])

    touching = -1
    if sub_ids:
        # sphere value of every center involved (a center's value does not depend on which end it is)
        sd = sphere_function(pts, rad, point)
        s = np.asarray(sub_ids)
        a, b = sd[s], sd[s + 1]
        hit = (a > 0.0) & (b <= 0.0) if forward else (a <= 0.0) & (b > 0.0)
        w = np.flatnonzero(hit)
        if len(w):
            touching = int(s[w[0]])

    if touching == -1:
        sd0 = sphere_function(pts[sub_id], rad[sub_id], point)
        sd1 = sphere_function(pts[sub_id + 1], rad[sub_id + 1], point)
        if sd0 <= 0.0 and sd1 <= 0.0:
            return -1, 0.0
        touching = sub_id

    # linear search within (touching, touching + 1), step proportional to the sphere radius
    c0, c1 = pts[touching], pts[touching + 1]
    r0, r1 = rad[touching], rad[touching + 1]
    e = c0 - c1
    d2 = e[0] * e[0] + e[1] * e[1] + e[2] * e[2]
    seg_len = d2 if vmtk_steps else math.sqrt(d2)
    step_size = 1e-6 * (r0 + r1) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.float64(seg_len) / np.float64(step_size)
    m = MAX_NUMBER_OF_STEPS if MAX_NUMBER_OF_STEPS < ratio else ratio    # std::min: NaN stays NaN
    n_steps = int(math.ceil(m)) if math.isfinite(m) else 0              # (int)ceil(NaN): 0 on arm64
    pstep = 1.0 / float(n_steps) if n_steps > 0 else 0.0

    touching_pcoord = 0.0
    if n_steps > 0:
        cp = running_pcoords(n_steps, pstep)
        centers = np.stack([c0[k] + cp * (c1[k] - c0[k]) for k in range(3)], axis=1)
        radii = r0 + cp * (r1 - r0)
        sd = sphere_function(centers, radii, point)
        a, b = sd[:-1], sd[1:]
        hit = (a > 0.0) & (b <= 0.0) if forward else (a <= 0.0) & (b > 0.0)
        w = np.flatnonzero(hit)
        if len(w):
            i = int(w[0])
            touching_pcoord = float(cp[i]) if forward else float(cp[i + 1])

    if forward:
        if touching == sub_id and touching_pcoord > pcoord:
            return -1, 0.0
    else:
        if touching == sub_id and touching_pcoord < pcoord:
            return -1, 0.0
    return touching, touching_pcoord


def find_n_touching_sphere_center(cl: Centerlines, radius: str, cell_id: int, sub_id: int, pcoord: float,
                                  n: int, forward: bool = True,
                                  vmtk_steps: bool = False) -> tuple[int, float]:
    """vtkvmtkCenterlineSphereDistance::FindNTouchingSphereCenter: ``n`` touching steps."""
    if n == 0:
        return sub_id, pcoord
    s0, c0 = sub_id, pcoord
    s1, c1 = -1, 0.0
    for _ in range(n):
        s1, c1 = find_touching_sphere_center(cl, radius, cell_id, s0, c0, forward, vmtk_steps)
        if s1 == -1:
            break
        s0, c0 = s1, c1
    return s1, c1
