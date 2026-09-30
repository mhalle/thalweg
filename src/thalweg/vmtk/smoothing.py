"""vtkvmtkCenterlineSmoothing in numpy: vmtk's in-place Laplacian relaxation of each polyline.

Ports ``vtkvmtkCenterlineSmoothing`` (vtkVmtk/ComputationalGeometry/vtkvmtkCenterlineSmoothing.cxx;
script ``vmtkcenterlinesmoothing``). Public:

- :func:`smooth_centerlines` - the filter: every line cell's points relaxed, all arrays copied;
- :func:`smooth_line` / :func:`smooth_lines` - ``SmoothLine`` on one polyline / on many at once (the
  helper ``vtkvmtkCenterlineGeometry`` and ``vtkvmtkCenterlineBranchGeometry`` call).

``SmoothLine`` is Gauss-Seidel, not Jacobi: for each iteration it walks j = 1 .. n-2 and moves
point j by ``factor * (0.5 * (p[j-1] + p[j+1]) - p[j])`` IN PLACE, so p[j-1] is already this
iteration's value. End points never move. The port keeps that order exactly; it is vectorized
over a wavefront (all (iteration i, point j) with the same 2i + j are independent) and over lines,
which changes no arithmetic.

**The single-precision defect.** ``SmoothLine`` works in ``vtkPoints::New()``, whose default type is
VTK_FLOAT, and ``vtkPoints::DeepCopy`` converts into the destination's type: the double input is
rounded to float32 on entry and every relaxed point is rounded to float32 when stored back. The
output of vmtk's smoothing is therefore float32-exact (~8e-6 mm at 100 mm coordinates), whatever
the input precision. :func:`smooth_line` works in float64 by default; ``vmtk_float32=True``
reproduces vmtk's rounding (bit-exact against vmtk 1.5.2 on both oracles), for oracle comparison.
Measured (100 iterations, factor 0.1): the smoothed points differ by up to 9.9e-5 mm (phantom) and
1.8e-4 mm (case oracle) - the per-iteration rounding accumulates to ~10x a single rounding.
"""
from __future__ import annotations

import numpy as np

from ._vtk import round_float32
from .centerlines import Centerlines


def smooth_lines(lines: list[np.ndarray], iterations: int = 100, factor: float = 0.1,
                 vmtk_float32: bool = False) -> list[np.ndarray]:
    """``vtkvmtkCenterlineSmoothing::SmoothLine`` on each (n_i, 3) polyline; new arrays."""
    lines = [np.asarray(L, dtype=np.float64).reshape(-1, 3) for L in lines]
    if vmtk_float32:
        lines = [round_float32(L) for L in lines]
    out = [L.copy() for L in lines]
    lens = np.array([len(L) for L in lines], dtype=np.int64)
    if not lines or iterations <= 0 or lens.max(initial=0) < 3:
        return out
    m = int(lens.max())
    buf = np.zeros((len(lines), m, 3))
    for k, L in enumerate(lines):
        buf[k, :len(L)] = L
    iters = np.arange(iterations)
    # wavefront t = 2i + j: (i, j) reads (i, j-1) written at t-1 and (i-1, j), (i-1, j+1) written
    # at t-2 and t-1; no two updates of one step touch neighboring j, so gather-then-scatter is exact
    for t in range(1, 2 * (iterations - 1) + (m - 2) + 1):
        j = t - 2 * iters
        j = j[(j >= 1) & (j <= m - 2)]
        if len(j) == 0:
            continue
        rows, cols = np.nonzero(j[None, :] <= (lens[:, None] - 2))
        if len(rows) == 0:
            continue
        jj = j[cols]
        p0, p1, p2 = buf[rows, jj - 1], buf[rows, jj], buf[rows, jj + 1]
        p1 = p1 + factor * (0.5 * (p0 + p2) - p1)
        buf[rows, jj] = round_float32(p1) if vmtk_float32 else p1
    return [buf[k, :n].copy() for k, n in enumerate(lens)]


def smooth_line(points: np.ndarray, iterations: int = 100, factor: float = 0.1,
                vmtk_float32: bool = False) -> np.ndarray:
    """``SmoothLine`` on one (n, 3) polyline: ``iterations`` in-place relaxation sweeps."""
    return smooth_lines([points], iterations, factor, vmtk_float32)[0]


def smooth_centerlines(cl: Centerlines, iterations: int = 100, factor: float = 0.1,
                       vmtk_float32: bool = False) -> Centerlines:
    """vtkvmtkCenterlineSmoothing: a copy of ``cl`` whose line points are relaxed cell by cell.

    Each cell is smoothed from the INPUT points, then written back in cell order (a point shared by
    two cells keeps the later cell's value, as vmtk's ``SetPoint`` loop does). All arrays are copied
    unchanged. The vmtk script's defaults are 100 iterations and factor 0.1 (the C++ class defaults
    to 0.01).
    """
    out = cl.copy()
    smoothed = smooth_lines([cl.cell_points(k) for k in range(cl.n_cells)], iterations, factor,
                            vmtk_float32)
    for k, S in enumerate(smoothed):
        out.points[cl.cells[k]] = S
    return out
