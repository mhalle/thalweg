"""vmtkCenterlineResampling in numpy: vtkCleanPolyData, then vtkSplineFilter with a cardinal spline.

``vmtkCenterlineResampling`` delegates to ``vmtkLineResampling`` (vmtk master 66b068a,
``vmtkScripts/vmtklineresampling.py``), which is VTK only:

1. ``vtkCleanPolyData`` with its defaults (``Filters/Core/vtkCleanPolyData.cxx``): exact point
   merging (tolerance 0, so a ``vtkMergePoints`` locator: coordinates equal in all three
   components), new ids in first-seen order along the cells; a merged point carries the point
   data of the LAST input point merged into it; consecutive repeated ids inside a line are
   dropped; a line left with one point becomes a vertex (``ConvertLinesToPoints``), one left with
   none disappears.
2. if ``length == 0``: ``length = cleaned.GetLength() / 100`` (the cleaned bounding-box diagonal).
3. ``vtkSplineFilter`` with ``SetSubdivideToLength()`` and ``SetLength(length)``, everything else
   default (``Filters/General/vtkSplineFilter.cxx``): per line, the chord-length parameter
   ``t = s / L`` in [0, 1], three independent ``vtkCardinalSpline`` s (x, y, z;
   ``Common/ComputationalGeometry/vtkCardinalSpline.cxx``) with VTK's default end constraints -
   ``LeftConstraint = RightConstraint = 1`` with value 0, i.e. ZERO first derivative at both ends
   (so the resampled line eases in and out of its end points) - evaluated at ``i / n`` for
   ``n = int(L / length)`` (at least 1) and ``i = 0..n``. The end points are reproduced to
   rounding only (a knot is evaluated on the cubic piece to its left); the spacing is ``L / n``
   in the parameter, so up to twice ``length`` for short lines.
   Point arrays are linearly interpolated along the INPUT polyline at the same ``t``
   (``vtkDataSetAttributes::InterpolateEdge``: ``a * (1 - tc) + b * tc``, integers rounded half
   away from zero); the segment parameters are held in a ``vtkFloatArray`` (float32), exactly as
   VTK does. A ``TCoords`` point array (float32, ``(t, 0)``) is added
   (``GenerateTCoords = VTK_TCOORDS_FROM_NORMALIZED_LENGTH``). Cell arrays are copied per line.
   Output points are never shared: each line gets ``n + 1`` new points.

Public: :func:`resample_centerlines`, plus :func:`clean_lines` and :class:`CardinalSpline` (the two
VTK pieces, reusable; they live in :mod:`._vtk` with ``vtkSplineFilter``). Field data is not carried
(neither VTK filter passes it). The output points keep the input's precision, as
``vtkSplineFilter`` does (``OutputPointsPrecision`` DEFAULT): vmtk's own input is double here, so
there is no float32 rounding to reproduce.

**vtkSplineFilter cell-data defect** (``vmtk_cell_data=True`` reproduces it): the filter copies
the cell data of input cell ``inCellId``, its index among the LINES, but ``vtkPolyData`` numbers
vertex cells before lines. Whenever the clean step turned a degenerate line (all points
coincident, or a one-point cell) into a vertex, every line reads a shifted row (the first lines
get the vertices' data). Default: each line keeps its own cell data. Without degenerate lines the
two agree (so the oracles cannot see it; it was checked against vmtk 1.5.2 on a degenerate layout,
see tests/test_vmtk_resampling.py).

Residual differences from the compiled VTK are floating-point contraction (FMA) in the arm64 build,
e.g. ``vtkMath::Distance2BetweenPoints`` evaluated as ``fma(dz, dz, fma(dx, dx, dy * dy))``;
they are at the 1e-15 level.
"""
from __future__ import annotations

import numpy as np

from ._vtk import CardinalSpline, bounds_length, clean_lines, spline_filter
from .centerlines import Centerlines

__all__ = ["resample_centerlines", "clean_lines", "CardinalSpline"]


def resample_centerlines(cl: Centerlines, length: float = 0.0, vmtk_cell_data: bool = False) -> Centerlines:
    """vmtkCenterlineResampling: each line resampled by a cardinal spline to ``length`` spacing
    (``length == 0``: 1/100 of the bounding-box diagonal). See the module docstring."""
    points, lines, verts, pd, cd = clean_lines(cl)
    if length == 0.0:
        length = bounds_length(points) / 100.0
    if len(points) < 1 or len(lines) < 1:
        return Centerlines(np.zeros((0, 3)), [], {}, {}, {})
    return spline_filter(points, lines, len(verts), pd, cd, length, vmtk_cell_data)
