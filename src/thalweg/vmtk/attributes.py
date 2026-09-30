"""vtkvmtkCenterlineAttributesFilter in numpy: abscissas and parallel-transport normals.

Ports ``vtkvmtkCenterlineAttributesFilter`` (vmtk master 66b068a,
``vtkVmtk/ComputationalGeometry/vtkvmtkCenterlineAttributesFilter.cxx``), the filter behind
``vmtkCenterlineAttributes``, with the VTK 9.4 helpers it calls (``vtkMath::Perpendiculars``,
``vtkMath::Normalize``, ``vtkMatrix4x4::MatrixFromRotation`` via ``vtkTransform::RotateWXYZ``,
``vtkLinearTransform`` point transform; all in :mod:`._vtk`) followed operation for operation.
Public: :func:`centerline_attributes`.

- **Abscissas** (float64, one per point): per cell, cumulative Euclidean length from the cell's
  first point. A point shared by several cells keeps the value of the LAST cell that visits it.
- **ParallelTransportNormals** (float64, (N, 3)): per cell, the first normal is
  ``vtkMath::Perpendiculars`` of the first non-degenerate tangent; each interior normal is the
  previous one rotated by the turning angle about ``t0 x t1``, projected off ``t1`` and normalized;
  the last point repeats the second-to-last normal.

One-point cells get abscissa 0 and no normal (a zero normal, unless another cell writes it).

**The two-point cell defect** (``vmtk_two_point_cells=True`` reproduces it). ``vtkPolyData`` types a
two-point line cell ``VTK_LINE`` (a ``vtkLine``) and every other line cell ``VTK_POLY_LINE``; the
filter reaches its cells through ``vtkPolyLine::SafeDownCast``, which rejects a ``vtkLine``, so a
two-point centerline gets neither abscissas nor normals (its points stay 0 unless another cell
writes them). A two-point line is a centerline like any other - vmtk's other filters accept
``VTK_LINE`` and ``VTK_POLY_LINE`` alike - so the default processes it: abscissas 0 and its length,
the first point's normal (``Perpendiculars`` of the chord) at both points (the last point repeats
its neighbor's normal, as on longer cells). No oracle has a two-point input cell, so the flag
changes nothing there; it matters for re-running the filter on split output (the case oracle's
extract has two-point tracts).

Differences from vmtk, all confined to inputs vmtk mishandles (none occurs in the oracles):

- vmtk zeroes only component 0 of the normals array (``FillComponent(0, 0.0)``); components 1-2 of
  points no processed cell reaches are uninitialized memory. Here they are 0.
- vmtk's search for the first non-degenerate tangent reads past the end of a cell whose points
  all coincide; here the search stops at the cell's last point (the normals then come out NaN,
  as a zero tangent through ``Perpendiculars`` gives).

Residual differences from the compiled vmtk/VTK are floating-point contraction (FMA) in the arm64
build (e.g. the abscissa's ``Distance2BetweenPoints`` is ``fma(dz, dz, fma(dx, dx, dy * dy))``;
emulating that makes the abscissas bit-identical); they are at the 1e-15 level.
"""
from __future__ import annotations

import math

import numpy as np

from ._vtk import VTK_VMTK_DOUBLE_TOL, norm, normalized, perpendiculars, rotate_point
from .centerlines import ABSCISSAS, NORMALS, Centerlines


def _acos(x: float) -> float:
    """C's acos: NaN (not a ValueError) outside [-1, 1]."""
    return math.acos(x) if -1.0 <= x <= 1.0 else math.nan


def _is_polyline(cell: np.ndarray) -> bool:
    """vtkPolyLine::SafeDownCast(input->GetCell(k)) succeeds: every line cell except 2-point ones."""
    return len(cell) != 2


def compute_abscissas(cl: Centerlines, vmtk_two_point_cells: bool = False) -> np.ndarray:
    """vtkvmtkCenterlineAttributesFilter::ComputeAbscissas."""
    out = np.zeros(cl.n_points, dtype=np.float64)
    pts = cl.points.tolist()
    for cell in cl.cells:
        if vmtk_two_point_cells and not _is_polyline(cell):
            continue
        ids = cell.tolist()
        abscissa = 0.0
        out[ids[0]] = abscissa
        for i in range(1, len(ids)):
            p0, p1 = pts[ids[i - 1]], pts[ids[i]]
            d2 = ((p0[0] - p1[0]) * (p0[0] - p1[0]) + (p0[1] - p1[1]) * (p0[1] - p1[1])
                  + (p0[2] - p1[2]) * (p0[2] - p1[2]))
            abscissa += math.sqrt(d2)
            out[ids[i]] = abscissa
    return out


def compute_parallel_transport_normals(cl: Centerlines, vmtk_two_point_cells: bool = False) -> np.ndarray:
    """vtkvmtkCenterlineAttributesFilter::ComputeParallelTransportNormals."""
    out = np.zeros((cl.n_points, 3), dtype=np.float64)
    pts = cl.points.tolist()
    for cell in cl.cells:
        if vmtk_two_point_cells and not _is_polyline(cell):
            continue
        ids = cell.tolist()
        n = len(ids)
        if n < 2:
            continue
        t0 = [0.0, 0.0, 0.0]
        p0 = pts[ids[0]]
        next_id = 1
        while norm(t0) < VTK_VMTK_DOUBLE_TOL and next_id < n:
            p1 = pts[ids[next_id]]
            t0 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]]
            next_id += 1
        t0 = normalized(t0)
        n0 = perpendiculars(t0)[0]
        out[ids[0]] = n0
        n1 = n0
        for i in range(1, n - 1):
            p0, p1, p2 = pts[ids[i - 1]], pts[ids[i]], pts[ids[i + 1]]
            t0 = normalized([p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]])
            t1 = normalized([p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]])
            dot = t0[0] * t1[0] + t0[1] * t1[1] + t0[2] * t1[2]
            if 1 - dot < VTK_VMTK_DOUBLE_TOL:
                theta = 0.0
            else:
                theta = _acos(dot) * 180.0 / 3.141592653589793
            v = [t0[1] * t1[2] - t0[2] * t1[1],
                 t0[2] * t1[0] - t0[0] * t1[2],
                 t0[0] * t1[1] - t0[1] * t1[0]]
            n1 = rotate_point(theta, v, n0)
            dot = t1[0] * n1[0] + t1[1] * n1[1] + t1[2] * n1[2]
            n1 = normalized([n1[0] - dot * t1[0], n1[1] - dot * t1[1], n1[2] - dot * t1[2]])
            out[ids[i]] = n1
            n0 = n1
        # SetTuple(id2, n1): id2 is the cell's last point (on a two-point cell, which vmtk never gets
        # here, the loop does not run: n1 is still the first normal)
        out[ids[-1]] = n1
    return out


def centerline_attributes(cl: Centerlines, abscissas: str = ABSCISSAS, normals: str = NORMALS,
                          vmtk_two_point_cells: bool = False) -> Centerlines:
    """vmtkCenterlineAttributes: a copy of ``cl`` with the ``abscissas`` and parallel-transport
    ``normals`` point arrays added (replacing same-named arrays). ``vmtk_two_point_cells=True``
    skips two-point cells, as vmtk does (see the module docstring)."""
    out = cl.copy()
    out.point_data[abscissas] = compute_abscissas(cl, vmtk_two_point_cells)
    out.point_data[normals] = compute_parallel_transport_normals(cl, vmtk_two_point_cells)
    return out
