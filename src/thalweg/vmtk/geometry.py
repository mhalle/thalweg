"""vtkvmtkCenterlineGeometry in numpy: length, curvature, torsion, tortuosity and Frenet frames.

Ports ``vtkvmtkCenterlineGeometry`` (vtkVmtk/ComputationalGeometry/vtkvmtkCenterlineGeometry.cxx;
script ``vmtkcenterlinegeometry``). Public:

- :func:`centerline_geometry` - the filter: per point ``Curvature``, ``Torsion``, ``FrenetTangent``,
  ``FrenetNormal``, ``FrenetBinormal``; per cell ``Length`` and ``Tortuosity`` (the script's names);
- :func:`line_curvature`, :func:`line_torsion`, :func:`line_frenet` - the static per-polyline
  helpers (``ComputeLineCurvature`` / ``ComputeLineTorsion`` return the weighted mean too, as the
  branch geometry filter uses it).

vmtk's discrete derivatives at interior point j, with h0 = |p[j] - p[j-1]|, h1 = |p[j+1] - p[j]|:
``x' = (p[j+1] - p[j-1]) / (h0 + h1)``, ``x'' = ((p[j+1] - p[j]) / h1 - (p[j] - p[j-1]) / h0) /
((h0 + h1) / 2)``, curvature ``|x' x x''| / |x'|^3``; torsion (j = 2 .. n-3) uses
``x''' = (x''[j+1] - x''[j-1]) / (h0 + h1)`` and ``(x' x x'') . x''' / |x' x x''|^2``. A point with
h0 or h1 below 1e-12 is skipped (left 0). Frenet frames: T = x'/|x'|, B = (x' x x'')/|..|,
N = B x T normalized; the end points copy their neighbor's frame. Mean curvature / torsion are
weighted by (h0 + h1) / 2.

**The single-precision defect** (see :mod:`.smoothing`): the filter copies each line into a
``vtkPoints::New()`` (VTK_FLOAT) before any computation, so vmtk's geometry - Length and Tortuosity
included - is computed from float32-rounded points even when the input is double. With 0.3 mm
spacing that rounding (~8e-6 mm at 100 mm coordinates) is 3e-5 of the spacing; the second and
third differences divide it by h^2 and h^3, so vmtk's torsion is dominated by it. Every function here
works in float64 by default; ``vmtk_float32=True`` reproduces vmtk. Measured: Curvature up to
6.8e-5 per mm on the phantom, median 3.5e-5 and 99.9 % 2.5e-4 per mm on the case oracle (1.0 per mm
at a 1023 per mm cusp); Torsion up to 0.36 per mm on the phantom, median 6.6e-4 per mm on the case,
where on nearly straight runs (|x' x x''| ~ 0, 10 % of its points) torsion is ill-conditioned in
either precision (up to 1e15 per mm) and the two differ by as much; Length 1.2e-6 / 1.7e-5 mm,
Tortuosity 4e-8 / 2e-7.

**Residual against vmtk.** vmtk 1.5.2's arm64 build contracts ``a * b + c`` into fused multiply-adds
(visible in the disassembly of ``ComputeLineCurvature``: every ``vtkMath::Norm``, ``Cross`` and
``Dot`` is fmadd/fnmul chains). numpy has no fused multiply-add, so the port rounds those products
separately; results then differ from vmtk in the last bits (curvature <= 6e-16 relative; replaying
the disassembled fma chain with libm's ``fma`` reproduces vmtk's curvature bit for bit on all 10,973
interior points of the case oracle). Torsion amplifies that where ``|x' x x''|`` is tiny (nearly
straight or nearly coincident points): <= 1.5e-12 relative; see tests/test_vmtk_geometry.py.
"""
from __future__ import annotations

import numpy as np

from ._vtk import (VTK_VMTK_DOUBLE_TOL, cross_rows, dot_rows, norm_rows, normalized_rows,
                   polyline_length, round_float32)
from .centerlines import Centerlines
from .smoothing import smooth_line


def _derivatives(p: np.ndarray):
    """The interior (j = 1 .. n-2) quantities every vmtk geometry loop computes, in its order:
    (valid, h0 + h1, x', x''). Rows where h0 or h1 < 1e-12 are invalid (vmtk ``continue``s)."""
    p0, p1, p2 = p[:-2], p[1:-1], p[2:]
    norm0 = norm_rows(p1 - p0)
    norm1 = norm_rows(p2 - p1)
    valid = ~((norm0 < VTK_VMTK_DOUBLE_TOL) | (norm1 < VTK_VMTK_DOUBLE_TOL))
    with np.errstate(divide="ignore", invalid="ignore"):
        s = norm0 + norm1
        xp = (p2 - p0) / s[:, None]
        xpp = (p2 - p1) / norm1[:, None] - (p1 - p0) / norm0[:, None]
        xpp = xpp / (s / 2.0)[:, None]
    return valid, s, xp, xpp


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """vmtk's running ``average += value * weight; weightSum += weight`` then divide if > 0."""
    average, weight_sum = 0.0, 0.0
    for v, w in zip(values.tolist(), weights.tolist()):
        average += v * w
        weight_sum += w
    if weight_sum > 0.0:
        average /= weight_sum
    return average


def line_curvature(points: np.ndarray, vmtk_float32: bool = False) -> tuple[np.ndarray, float]:
    """vtkvmtkCenterlineGeometry::ComputeLineCurvature: (per-point curvature, weighted mean)."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if vmtk_float32:
        p = round_float32(p)
    n = len(p)
    curvature = np.zeros(n)
    if n < 3:
        return curvature, 0.0
    valid, s, xp, xpp = _derivatives(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = norm_rows(cross_rows(xp, xpp)) / np.power(norm_rows(xp), 3.0)
    curvature[1:-1] = np.where(valid, c, 0.0)
    return curvature, _weighted_mean(c[valid], (s / 2.0)[valid])


def line_torsion(points: np.ndarray, vmtk_float32: bool = False) -> tuple[np.ndarray, float]:
    """vtkvmtkCenterlineGeometry::ComputeLineTorsion: (per-point torsion, weighted mean)."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if vmtk_float32:
        p = round_float32(p)
    n = len(p)
    torsion = np.zeros(n)
    if n < 5:
        return torsion, 0.0
    valid, s, xp, xpp = _derivatives(p)
    xps = np.zeros((n, 3))
    xpps = np.zeros((n, 3))
    xps[1:-1][valid] = xp[valid]
    xpps[1:-1][valid] = xpp[valid]
    j = np.arange(2, n - 2)
    ok = valid[j - 1]
    sj = s[j - 1]
    cross = cross_rows(xps[j], xpps[j])
    with np.errstate(divide="ignore", invalid="ignore"):
        xppp = (xpps[j + 1] - xpps[j - 1]) / sj[:, None]
        cn = norm_rows(cross)
        t = np.where(cn > 0, dot_rows(cross, xppp) / np.power(cn, 2.0), 0.0)
    torsion[2:-2] = np.where(ok, t, 0.0)
    return torsion, _weighted_mean(t[ok], (sj / 2.0)[ok])


def line_frenet(points: np.ndarray, vmtk_float32: bool = False):
    """vtkvmtkCenterlineGeometry::ComputeLineFrenetReferenceSystem: (tangent, normal, binormal),
    each (n, 3); invalid interior points stay 0 and the ends copy points 1 and n-2."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if vmtk_float32:
        p = round_float32(p)
    n = len(p)
    tangent, normal, binormal = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3))
    if n < 2:
        return tangent, normal, binormal      # vmtk reads tuple 1 of a 1-tuple array here (UB)
    if n >= 3:
        valid, _, xp, xpp = _derivatives(p)
        t = normalized_rows(xp)
        b = normalized_rows(cross_rows(xp, xpp))
        nn = normalized_rows(cross_rows(b, t))
        tangent[1:-1][valid] = t[valid]
        normal[1:-1][valid] = nn[valid]
        binormal[1:-1][valid] = b[valid]
    for a in (tangent, normal, binormal):
        a[0] = a[1]
        a[n - 1] = a[n - 2]
    return tangent, normal, binormal


def centerline_geometry(cl: Centerlines, line_smoothing: bool = False, iterations: int = 100,
                        factor: float = 0.1, output_smoothed_lines: bool = False,
                        vmtk_float32: bool = False, length: str = "Length", curvature: str = "Curvature",
                        torsion: str = "Torsion", tortuosity: str = "Tortuosity",
                        frenet_tangent: str = "FrenetTangent", frenet_normal: str = "FrenetNormal",
                        frenet_binormal: str = "FrenetBinormal") -> Centerlines:
    """vtkvmtkCenterlineGeometry on every line cell: a copy of ``cl`` with the geometry arrays added.

    ``line_smoothing`` first runs :func:`.smoothing.smooth_line` (``iterations``, ``factor``) on each
    line; ``output_smoothed_lines`` also writes those points into the output. Defaults are the
    vmtk script's (100 iterations, factor 0.1; the C++ class defaults to 0.01). Points in no cell
    keep zeros; a point shared by two cells gets the later cell's values. Tortuosity is
    ``length / |p[-1] - p[0]| - 1`` (inf for a closed line, as in vmtk).
    """
    out = cl.copy()
    npts, ncells = cl.n_points, cl.n_cells
    length_a, tortuosity_a = np.zeros(ncells), np.zeros(ncells)
    curvature_a, torsion_a = np.zeros(npts), np.zeros(npts)
    tangent_a, normal_a, binormal_a = np.zeros((npts, 3)), np.zeros((npts, 3)), np.zeros((npts, 3))
    for i in range(ncells):
        ids = cl.cells[i]
        p = cl.points[ids]
        if vmtk_float32:
            p = round_float32(p)
        if line_smoothing:
            p = smooth_line(p, iterations, factor, vmtk_float32)
        c, _ = line_curvature(p)
        t, _ = line_torsion(p)
        ft, fn, fb = line_frenet(p)
        if output_smoothed_lines:
            out.points[ids] = p
        curvature_a[ids] = c
        torsion_a[ids] = t
        tangent_a[ids] = ft
        normal_a[ids] = fn
        binormal_a[ids] = fb
        L = polyline_length(p) if len(p) else 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            d = p[-1] - p[0] if len(p) else np.zeros(3)
            tort = np.float64(L) / np.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) - 1.0
        length_a[i] = L
        tortuosity_a[i] = tort
    out.cell_data[length] = length_a
    out.point_data[curvature] = curvature_a
    out.point_data[torsion] = torsion_a
    out.cell_data[tortuosity] = tortuosity_a
    out.point_data[frenet_tangent] = tangent_a
    out.point_data[frenet_normal] = normal_a
    out.point_data[frenet_binormal] = binormal_a
    return out
