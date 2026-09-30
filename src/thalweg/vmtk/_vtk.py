"""Ports of the VTK routines vmtk's filters call (and of vtkvmtkMath's), shared by the filter ports.

One implementation of each, followed operation for operation (VTK 9.4, vmtk master 66b068a), so a
port that calls one reads like the C++ it follows and gets the same bits:

- constants: ``vtkvmtkConstants.h`` and the VTK ones the filters test against;
- C arithmetic: :func:`div` (IEEE division of doubles: x / 0 is +-inf or NaN, never an exception);
- ``vtkPoints`` storage: :func:`round_float32` (what a default, VTK_FLOAT ``vtkPoints`` keeps);
- ``vtkMath``: :func:`dot`, :func:`cross`, :func:`norm`, :func:`normalize` (in place) /
  :func:`normalized` (a copy), :func:`distance2` (``Distance2BetweenPoints``),
  :func:`perpendiculars`; and the same arithmetic row-wise over (..., 3) arrays, for the vectorized
  loops: :func:`dot_rows`, :func:`cross_rows`, :func:`norm_rows`, :func:`normalized_rows` (bit-identical
  to the scalar ones, row for row); :func:`polyline_length` (the running sum of segment lengths);
- ``vtkMatrix4x4`` / ``vtkTransform``: :func:`matrix_from_rotation` (``RotateWXYZ``),
  :func:`invert_4x4`, :func:`rotate_point` (``RotateWXYZ`` + ``TransformPoint``),
  :func:`rotate_normal` (``RotateWXYZ`` + ``TransformNormal``);
- ``vtkvmtkMath``: :func:`angle_between_normals`;
- ``vtkLine`` / ``vtkPolyLine`` cell queries: :func:`line_distance_to_line`,
  :func:`line_evaluate_position`, :func:`polyline_evaluate_position`;
- ``vtkDataSetAttributes::InterpolateEdge``: :func:`interpolate_edge`;
- ``vtkCleanPolyData`` (lines, defaults): :func:`clean_lines`; ``vtkDataSet::GetLength``:
  :func:`bounds_length`;
- ``vtkCardinalSpline``: :class:`CardinalSpline`; ``vtkSplineFilter`` (subdivide to length):
  :func:`spline_filter`.

**vtkSplineFilter cell-data defect** (``vmtk_cell_data=True`` in :func:`spline_filter` reproduces it;
it reaches ``vmtkCenterlineResampling`` and ``vtkvmtkMergeCenterlines``): the filter copies the cell
data of input cell ``inCellId``, its index among the LINES, but ``vtkPolyData`` numbers vertex cells
before lines. Whenever the clean step turned a degenerate line (all points coincident, or a one-point
cell) into a vertex, every line reads a shifted row (the first lines get the vertices' data).
Correct: each line keeps its own cell data. Without degenerate lines (every oracle) the two agree.

Residual differences from the compiled VTK are floating-point contraction (FMA) in the arm64 build,
e.g. ``Distance2BetweenPoints`` evaluated as ``fma(dz, dz, fma(dx, dx, dy * dy))``; they are at the
1e-15 level.
"""
from __future__ import annotations

import math

import numpy as np

from .centerlines import Centerlines

VTK_VMTK_DOUBLE_TOL = 1e-12        # vtkvmtkConstants.h
VTK_VMTK_FLOAT_TOL = 1.0e-6
VTK_VMTK_LARGE_DOUBLE = 1.0e+32
VTK_TOL = 1e-05                    # vtkLine / vtkMath tolerance used by DistanceToLine
VTK_DOUBLE_MAX = 1.0e+299          # vtkType.h
VTK_DOUBLE_MIN = -1.0e+299
VTK_INT_MAX = 2147483647
VTK_PI = 3.141592653589793


# -- C arithmetic -----------------------------------------------------------------------------

def div(a, b) -> float:
    """``a / b`` on doubles as C does it: a zero divisor gives +-inf or NaN, not ZeroDivisionError."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.float64(a) / np.float64(b))


# -- vtkPoints storage ------------------------------------------------------------------------

def round_float32(a) -> np.ndarray:
    """Float64 values rounded to float32 and back: what a default (VTK_FLOAT) ``vtkPoints`` stores."""
    return np.asarray(a, dtype=np.float64).astype(np.float32).astype(np.float64)


# -- vtkMath (scalar, in the C++ operation order) ---------------------------------------------

def dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b) -> list[float]:
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def norm(v) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def normalize(v: list) -> float:
    """vtkMath::Normalize, in place: divide by the norm unless it is 0; returns the norm."""
    den = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if den != 0.0:
        v[0] /= den
        v[1] /= den
        v[2] /= den
    return den


def normalized(v) -> list[float]:
    """vtkMath::Normalize on a copy."""
    out = [v[0], v[1], v[2]]
    normalize(out)
    return out


def distance2(p, q) -> float:
    """vtkMath::Distance2BetweenPoints."""
    return (p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]) + (p[2] - q[2]) * (p[2] - q[2])


def perpendiculars(v1) -> tuple[list[float], list[float]]:
    """vtkMath::Perpendiculars(v1, v2, v3, theta=0) -> (v2, v3). A zero ``v1`` gives NaNs, as in C."""
    v1sq, v2sq, v3sq = v1[0] * v1[0], v1[1] * v1[1], v1[2] * v1[2]
    r = math.sqrt(v1sq + v2sq + v3sq)
    if v1sq > v2sq and v1sq > v3sq:
        d1, d2, d3 = 0, 1, 2
    elif v2sq > v3sq:
        d1, d2, d3 = 1, 2, 0
    else:
        d1, d2, d3 = 2, 0, 1
    with np.errstate(divide="ignore", invalid="ignore"):
        a, b, c = np.float64(v1[d1]) / r, np.float64(v1[d2]) / r, np.float64(v1[d3]) / r
        tmp = np.sqrt(a * a + c * c)
        v2 = [0.0, 0.0, 0.0]
        v3 = [0.0, 0.0, 0.0]
        v2[d1], v2[d2], v2[d3] = float(c / tmp), 0.0, float(-a / tmp)
        v3[d1], v3[d2], v3[d3] = float(-a * b / tmp), float(tmp), float(-b * c / tmp)
    return v2, v3


# -- vtkMath, row-wise (the same operations in the same order, over (..., 3) arrays) ------------

def norm_rows(v: np.ndarray) -> np.ndarray:
    """vtkMath::Norm, row-wise."""
    return np.sqrt(v[..., 0] * v[..., 0] + v[..., 1] * v[..., 1] + v[..., 2] * v[..., 2])


def cross_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """vtkMath::Cross, row-wise."""
    return np.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                     a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                     a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], axis=-1)


def dot_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """vtkMath::Dot, row-wise."""
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def normalized_rows(v: np.ndarray) -> np.ndarray:
    """vtkMath::Normalize, row-wise: rows of norm 0 are left unchanged."""
    den = norm_rows(v)
    safe = np.where(den != 0.0, den, 1.0)
    return np.where((den != 0.0)[..., None], v / safe[..., None], v)


def polyline_length(p: np.ndarray) -> float:
    """A polyline's length as vmtk sums it: ``length += sqrt(Distance2BetweenPoints(p[j-1], p[j]))``,
    in order (the squares, their sum and the running sum in the C++ order)."""
    d = p[1:] - p[:-1]
    seg = np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2])
    length = 0.0
    for x in seg.tolist():
        length += x
    return length


# -- vtkMatrix4x4 / vtkTransform ----------------------------------------------------------------

def matrix_from_rotation(angle_deg: float, x: float, y: float, z: float) -> list[float]:
    """vtkMatrix4x4::MatrixFromRotation (what ``vtkTransform::RotateWXYZ`` builds): the 16 elements,
    row-major (``m[4 * row + col]``)."""
    m = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    if angle_deg == 0.0 or (x == 0.0 and y == 0.0 and z == 0.0):
        return m
    angle = angle_deg * 0.017453292519943295
    w = math.cos(0.5 * angle)
    f = math.sin(0.5 * angle) / math.sqrt(x * x + y * y + z * z)
    x *= f
    y *= f
    z *= f
    ww, wx, wy, wz = w * w, w * x, w * y, w * z
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    s = ww - xx - yy - zz
    m[0] = xx * 2 + s
    m[4] = (xy + wz) * 2
    m[8] = (xz - wy) * 2
    m[1] = (xy - wz) * 2
    m[5] = yy * 2 + s
    m[9] = (yz + wx) * 2
    m[2] = (xz + wy) * 2
    m[6] = (yz - wx) * 2
    m[10] = zz * 2 + s
    return m


def _det2(a, b, c, d) -> float:
    return a * d - b * c


def _det3(a1, a2, a3, b1, b2, b3, c1, c2, c3) -> float:
    """vtkMath::Determinant3x3 (the nine-scalar form)."""
    return a1 * _det2(b2, b3, c2, c3) - b1 * _det2(a2, a3, c2, c3) + c1 * _det2(a2, a3, b2, b3)


def invert_4x4(e: list) -> list[float]:
    """vtkMatrix4x4::Invert: adjoint / determinant (a zero determinant leaves the matrix as is)."""
    a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3, a4, b4, c4, d4 = e
    det = (a1 * _det3(b2, b3, b4, c2, c3, c4, d2, d3, d4) - b1 * _det3(a2, a3, a4, c2, c3, c4, d2, d3, d4)
           + c1 * _det3(a2, a3, a4, b2, b3, b4, d2, d3, d4) - d1 * _det3(a2, a3, a4, b2, b3, b4, c2, c3, c4))
    if det == 0.0:
        return list(e)
    o = [0.0] * 16
    o[0] = _det3(b2, b3, b4, c2, c3, c4, d2, d3, d4)
    o[4] = -_det3(a2, a3, a4, c2, c3, c4, d2, d3, d4)
    o[8] = _det3(a2, a3, a4, b2, b3, b4, d2, d3, d4)
    o[12] = -_det3(a2, a3, a4, b2, b3, b4, c2, c3, c4)
    o[1] = -_det3(b1, b3, b4, c1, c3, c4, d1, d3, d4)
    o[5] = _det3(a1, a3, a4, c1, c3, c4, d1, d3, d4)
    o[9] = -_det3(a1, a3, a4, b1, b3, b4, d1, d3, d4)
    o[13] = _det3(a1, a3, a4, b1, b3, b4, c1, c3, c4)
    o[2] = _det3(b1, b2, b4, c1, c2, c4, d1, d2, d4)
    o[6] = -_det3(a1, a2, a4, c1, c2, c4, d1, d2, d4)
    o[10] = _det3(a1, a2, a4, b1, b2, b4, d1, d2, d4)
    o[14] = -_det3(a1, a2, a4, b1, b2, b4, c1, c2, c4)
    o[3] = -_det3(b1, b2, b3, c1, c2, c3, d1, d2, d3)
    o[7] = _det3(a1, a2, a3, c1, c2, c3, d1, d2, d3)
    o[11] = -_det3(a1, a2, a3, b1, b2, b3, d1, d2, d3)
    o[15] = _det3(a1, a2, a3, b1, b2, b3, c1, c2, c3)
    return [v / det for v in o]


def rotate_point(angle_deg: float, axis, p) -> list[float]:
    """``vtkTransform::RotateWXYZ(angle_deg, axis)`` then ``TransformPoint(p)``
    (vtkLinearTransformPoint; the translation column is 0)."""
    m = matrix_from_rotation(angle_deg, axis[0], axis[1], axis[2])
    return [m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3],
            m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7],
            m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11]]


def rotate_normal(angle_deg: float, axis, normal) -> list[float]:
    """``vtkTransform::RotateWXYZ(angle_deg, axis)`` then ``TransformNormal(normal)``: the normal times
    the transposed inverse of the rotation matrix, normalized."""
    inv = invert_4x4(matrix_from_rotation(angle_deg, axis[0], axis[1], axis[2]))
    # transposed: m[i][j] = inv[j][i]
    out = [inv[0] * normal[0] + inv[4] * normal[1] + inv[8] * normal[2],
           inv[1] * normal[0] + inv[5] * normal[1] + inv[9] * normal[2],
           inv[2] * normal[0] + inv[6] * normal[1] + inv[10] * normal[2]]
    normalize(out)
    return out


# -- vtkvmtkMath ------------------------------------------------------------------------------

def angle_between_normals(n0, n1) -> float:
    """vtkvmtkMath::AngleBetweenNormals: 2 atan2(|n0 - n1|, |n0 + n1|)."""
    s = [n0[0] + n1[0], n0[1] + n1[1], n0[2] + n1[2]]
    d = [n0[0] - n1[0], n0[1] - n1[1], n0[2] - n1[2]]
    return 2.0 * math.atan2(norm(d), norm(s))


# -- vtkLine / vtkPolyLine cell queries ---------------------------------------------------------

def line_distance_to_line(x, p1, p2) -> tuple[float, float, np.ndarray]:
    """vtkLine::DistanceToLine (VTK 9.4): (squared distance, t, closest point). t is NOT clamped;
    the closest point is. A degenerate segment gives t = +-VTK_DOUBLE_MAX/MIN, as VTK does."""
    p21 = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
    num = p21[0] * (x[0] - p1[0]) + p21[1] * (x[1] - p1[1]) + p21[2] * (x[2] - p1[2])
    if num == 0.0:
        t, closest = 0.0, p1
    else:
        denom = p21[0] * p21[0] + p21[1] * p21[1] + p21[2] * p21[2]
        tol = VTK_TOL * num
        if tol < 0.0:
            tol = -tol
        if denom < tol:                              # numerically bad
            if num > 0:
                closest, t = p2, VTK_DOUBLE_MAX
            else:
                closest, t = p1, VTK_DOUBLE_MIN
        else:
            t = num / denom
            if t < 0.0:
                closest = p1
            elif t > 1.0:
                closest = p2
            else:
                closest = (p1[0] + t * p21[0], p1[1] + t * p21[1], p1[2] + t * p21[2])
    d = (closest[0] - x[0], closest[1] - x[1], closest[2] - x[2])
    return d[0] * d[0] + d[1] * d[1] + d[2] * d[2], t, np.array(closest, dtype=np.float64)


def line_evaluate_position(x, p1, p2) -> tuple[int, np.ndarray, float, float]:
    """vtkLine::EvaluatePosition: (status, closest, pcoord, dist2). status 1 if 0 <= t <= 1 else 0."""
    dist2, t, closest = line_distance_to_line(x, p1, p2)
    return (1 if 0.0 <= t <= 1.0 else 0), closest, t, dist2


def polyline_evaluate_position(x, pts: np.ndarray) -> tuple[int, np.ndarray, int, float, float]:
    """vtkPolyLine::EvaluatePosition: (status, closest, sub_id, pcoord, min_dist2) over segments."""
    x = np.asarray(x, dtype=np.float64)
    best = VTK_DOUBLE_MAX
    status_out, sub, pc_out, closest_out = 0, -1, 0.0, np.zeros(3)
    for i in range(len(pts) - 1):
        status, closest, pc, dist2 = line_evaluate_position(x, pts[i], pts[i + 1])
        if status != -1 and (dist2 < best or (dist2 == best and status_out == 0)):
            status_out, closest_out, best, sub, pc_out = status, closest, dist2, i, pc
    return status_out, closest_out, sub, pc_out, best


# -- vtkDataSetAttributes::InterpolateEdge ------------------------------------------------------

def interpolate_edge(a: np.ndarray, i0, i1, t) -> np.ndarray:
    """Rows ``a[i0] * (1 - t) + a[i1] * t`` in ``a``'s dtype, vectorized over ``i0``, ``i1``, ``t``
    (vtkGenericDataArray::InterpolateTuple with vtkMath::RoundDoubleToIntegralIfNecessary: integers
    clamped to their range and rounded half away from zero, NaN -> 0; float32 clamped to its range)."""
    a = np.asarray(a)
    i0 = np.asarray(i0, dtype=np.int64)
    i1 = np.asarray(i1, dtype=np.int64)
    t = np.asarray(t, dtype=np.float64).reshape((-1,) + (1,) * (a.ndim - 1))
    val = a[i0].astype(np.float64) * (1.0 - t) + a[i1].astype(np.float64) * t
    if np.issubdtype(a.dtype, np.integer):
        info = np.iinfo(a.dtype)
        with np.errstate(invalid="ignore"):
            val = np.clip(val, float(info.min), float(info.max))
            val = np.where(val >= 0.0, val + 0.5, val - 0.5)
            val = np.where(np.isnan(val), 0.0, np.trunc(val))
    elif a.dtype == np.float32:
        fmax = float(np.finfo(np.float32).max)
        val = np.where(np.isnan(val), val, np.clip(val, -fmax, fmax))
    return val.astype(a.dtype)


# -- vtkCleanPolyData (lines only, default settings) -------------------------------------------

def clean_lines(cl: Centerlines) -> tuple[np.ndarray, list[list[int]], list[int], dict[str, np.ndarray],
                                           dict[str, np.ndarray]]:
    """vtkCleanPolyData on a lines-only polydata with default settings: exact point merging
    (tolerance 0, a ``vtkMergePoints`` locator: coordinates equal in all three components), new ids
    in first-seen order along the cells; a merged point carries the point data of the LAST input
    point merged into it (every visit copies, VTK 9.4 without ghost points); consecutive repeated
    ids inside a line are dropped; a line left with one point becomes a vertex
    (``ConvertLinesToPoints``), one left with none disappears.

    Returns ``(points, lines, vertices, point_data, cell_data)``: the merged points, the kept
    lines (lists of new ids), the vertex cells made from degenerate lines (their new point id),
    the merged point arrays, and the cell arrays in VTK's output order - vertex cells first, then
    lines (``cell_data[name][k]`` is output cell ``k``).
    """
    pts = cl.points.tolist()
    index: dict[tuple, int] = {}
    new_points: list = []
    source: list[int] = []                    # new id -> input id whose data it carries (the last)
    lines: list[list[int]] = []
    line_cells: list[int] = []
    verts: list[int] = []
    vert_cells: list[int] = []
    for in_cell, cell in enumerate(cl.cells):
        ids = cell.tolist()
        updated: list[int] = []
        for i, pid in enumerate(ids):
            x = pts[pid]
            key = (x[0], x[1], x[2])
            pt_id = index.get(key)
            if pt_id is None:
                pt_id = len(new_points)
                index[key] = pt_id
                new_points.append(x)
                source.append(pid)
            else:
                source[pt_id] = pid           # IsPrimaryPoint (no ghosts): every visit copies
            if i == 0 or pt_id != updated[-1]:
                updated.append(pt_id)
        if len(updated) >= 2:
            lines.append(updated)
            line_cells.append(in_cell)
        elif len(updated) == 1:               # npts == 1 or ConvertLinesToPoints: a vertex
            verts.append(updated[0])
            vert_cells.append(in_cell)
    points = np.array(new_points, dtype=np.float64).reshape(-1, 3)
    src = np.array(source, dtype=np.int64)
    point_data = {k: v[src] for k, v in cl.point_data.items()}
    order = np.array(vert_cells + line_cells, dtype=np.int64)
    cell_data = {k: v[order] for k, v in cl.cell_data.items()}
    return points, lines, verts, point_data, cell_data


def bounds_length(points: np.ndarray) -> float:
    """vtkDataSet::GetLength: the bounding-box diagonal."""
    if len(points) == 0:
        return 0.0
    lo, hi = points.min(axis=0), points.max(axis=0)
    total = 0.0
    for i in range(3):
        diff = float(hi[i]) - float(lo[i])
        total += diff * diff
    return math.sqrt(total)


# -- vtkCardinalSpline (open) -------------------------------------------------------------------

class CardinalSpline:
    """vtkCardinalSpline, open, with vtkSpline's defaults: left and right constraint 1 with value 0
    (prescribed zero first derivative at both ends), no parametric range, no clamping."""

    def __init__(self, left_constraint: int = 1, left_value: float = 0.0,
                 right_constraint: int = 1, right_value: float = 0.0):
        self.left_constraint, self.left_value = left_constraint, left_value
        self.right_constraint, self.right_value = right_constraint, right_value
        self._nodes: dict[float, float] = {}
        self._fit = None

    def add_point(self, t: float, x: float) -> None:
        """vtkPiecewiseFunction::AddPoint: a node at the same t is replaced."""
        self._nodes.pop(t, None)
        self._nodes[t] = x
        self._fit = None

    def _compute(self):
        nodes = sorted(self._nodes.items())
        xs = [n[0] for n in nodes]
        ys = [n[1] for n in nodes]
        size = len(xs)
        if size < 2:
            self._fit = (size, xs, None)
            return
        coef = [[0.0, 0.0, 0.0, 0.0] for _ in range(size)]
        work = [0.0] * size
        x, y = xs, ys
        lc, rc = self.left_constraint, self.right_constraint
        if lc == 0:
            coef[0][1], coef[0][2] = 1.0, 0.0
            work[0] = y[1] - y[0]                                   # ComputeLeftDerivative
        elif lc == 1:
            coef[0][1], coef[0][2] = 1.0, 0.0
            work[0] = self.left_value
        elif lc == 2:
            coef[0][1], coef[0][2] = 2.0, 1.0
            work[0] = 3.0 * ((y[1] - y[0]) / (x[1] - x[0])) - 0.5 * (x[1] - x[0]) * self.left_value
        elif lc == 3:
            lv = self.left_value
            coef[0][1] = 2.0
            coef[0][2] = 4.0 * ((0.5 + lv) / (2.0 + lv))
            work[0] = 6.0 * ((1.0 + lv) / (2.0 + lv)) * ((y[1] - y[0]) / (x[1] - x[0]))
        for k in range(1, size - 1):
            xlk = x[k] - x[k - 1]
            xlkp = x[k + 1] - x[k]
            coef[k][0] = xlkp
            coef[k][1] = 2.0 * (xlkp + xlk)
            coef[k][2] = xlk
            work[k] = 3.0 * (((xlkp * (y[k] - y[k - 1])) / xlk) + ((xlk * (y[k + 1] - y[k])) / xlkp))
        n1 = size - 1
        if rc == 0:
            coef[n1][0], coef[n1][1] = 0.0, 1.0
            work[n1] = y[n1] - y[n1 - 1]                            # ComputeRightDerivative
        elif rc == 1:
            coef[n1][0], coef[n1][1] = 0.0, 1.0
            work[n1] = self.right_value
        elif rc == 2:
            coef[n1][0], coef[n1][1] = 1.0, 2.0
            work[n1] = (3.0 * ((y[n1] - y[n1 - 1]) / (x[n1] - x[n1 - 1]))
                        + 0.5 * (x[n1] - x[n1 - 1]) * self.right_value)
        elif rc == 3:
            rv = self.right_value
            coef[n1][0] = 4.0 * ((0.5 + rv) / (2.0 + rv))
            coef[n1][1] = 2.0
            work[n1] = 6.0 * ((1.0 + rv) / (2.0 + rv)) * ((y[n1] - y[n1 - 1]) / (x[n1] - x[n1 - 1]))
        coef[0][2] = coef[0][2] / coef[0][1]
        work[0] = work[0] / coef[0][1]
        coef[n1][2] = 0.0
        for k in range(1, size):
            coef[k][1] = coef[k][1] - (coef[k][0] * coef[k - 1][2])
            coef[k][2] = coef[k][2] / coef[k][1]
            work[k] = (work[k] - (coef[k][0] * work[k - 1])) / coef[k][1]
        for k in range(size - 2, -1, -1):
            work[k] = work[k] - (coef[k][2] * work[k + 1])
        b = 0.0
        for k in range(size - 1):
            b = x[k + 1] - x[k]
            coef[k][0] = y[k]
            coef[k][1] = work[k]
            coef[k][2] = (3.0 * (y[k + 1] - y[k])) / (b * b) - (work[k + 1] + 2.0 * work[k]) / b
            coef[k][3] = (2.0 * (y[k] - y[k + 1])) / (b * b * b) + (work[k + 1] + work[k]) / (b * b)
        coef[n1][0] = y[n1]
        coef[n1][1] = work[n1]
        coef[n1][2] = coef[n1 - 1][2] + 3.0 * coef[n1 - 1][3] * b
        coef[n1][3] = coef[n1 - 1][3]
        self._fit = (size, xs, coef)

    def _find_index(self, size: int, t: float) -> int:
        """vtkSpline::FindIndex: bisection; a t on a knot falls in the interval to its left."""
        iv = self._fit[1]
        index = 0
        if size > 2:
            right = size - 1
            center = right - size // 2
            while True:
                if iv[index] <= t <= iv[center]:
                    right = center
                else:
                    index = center
                if index + 1 == right:
                    break
                center = index + (right - index) // 2
        return index

    def evaluate(self, t: float) -> float:
        if self._fit is None:
            self._compute()
        size, iv, coef = self._fit
        if size < 2:
            return 0.0
        if t < iv[0]:
            t = iv[0]
        if t > iv[size - 1]:
            t = iv[size - 1]
        index = self._find_index(size, t)
        t = t - iv[index]
        c = coef[index]
        return t * (t * (t * c[3] + c[2]) + c[1]) + c[0]


# -- vtkSplineFilter (SubdivideToLength, default spline, TCoords from normalized length) -------

def spline_filter(points: np.ndarray, lines: list, n_verts: int, point_data: dict[str, np.ndarray],
                  cell_data: dict[str, np.ndarray], length: float,
                  vmtk_cell_data: bool = False) -> Centerlines:
    """vtkSplineFilter with ``SetSubdivideToLength()`` and ``SetLength(length)``, everything else
    default, on :func:`clean_lines` output (``cell_data`` rows: ``n_verts`` vertices, then lines).

    Per line: the chord-length parameter ``t = s / L`` in [0, 1] (held in a float32
    ``vtkFloatArray``, as VTK does), three independent :class:`CardinalSpline` s (x, y, z) evaluated
    at ``i / n`` for ``n = int(L / length)`` (at least 1) and ``i = 0..n``; the end points are
    reproduced to rounding only (a knot is evaluated on the cubic piece to its left). Point arrays
    are interpolated along the INPUT polyline at the same ``t`` (:func:`interpolate_edge`), a
    float32 ``TCoords`` point array ``(t, 0)`` is added, cell arrays are copied per line (see the
    module docstring for ``vmtk_cell_data``). Output points are never shared: each line gets
    ``n + 1`` new points. The output points keep the input's precision (``OutputPointsPrecision``
    DEFAULT): round them to float32 when the input was a float32 ``vtkPoints``.
    """
    length = min(max(float(length), 0.0000001), 1.0e+299)       # vtkSetClampMacro(Length, ...)
    out_pts: list = []
    out_cells: list[np.ndarray] = []
    e0: list[int] = []
    e1: list[int] = []
    et: list[float] = []
    tcoords: list[float] = []
    cell_rows: list[int] = []
    tcoord_map: list[float] = []              # vtkFloatArray TCoordMap: Reset() keeps stale values
    pts = points.tolist()
    offset = 0
    for in_cell_id, ids in enumerate(lines):
        npts = len(ids)
        if npts < 2:
            continue
        # GeneratePoints: the polyline length
        length_total = 0.0
        x_prev = pts[ids[0]]
        for i in range(1, npts):
            x = pts[ids[i]]
            length_total += math.sqrt(distance2(x, x_prev))
            x_prev = x
        if length_total <= 0.0:
            continue
        splines = (CardinalSpline(), CardinalSpline(), CardinalSpline())
        x_prev = pts[ids[0]]
        run = 0.0
        for i in range(npts):
            x = pts[ids[i]]
            dist = math.sqrt(distance2(x, x_prev))
            if i > 0 and dist == 0:
                continue                      # unreachable after the clean step (no repeated ids)
            run += dist
            t = run / length_total
            if i < len(tcoord_map):
                tcoord_map[i] = float(np.float32(t))
            else:
                tcoord_map.extend([0.0] * (i - len(tcoord_map)))
                tcoord_map.append(float(np.float32(t)))
            for a in range(3):
                splines[a].add_point(t, x[a])
            x_prev = x
        num_divs = int(length_total / length)
        if num_divs > VTK_INT_MAX:
            num_divs = VTK_INT_MAX
        num_divs = 1 if num_divs < 1 else num_divs
        num_new = num_divs + 1
        idx = 0
        t_lo, t_hi = tcoord_map[0], tcoord_map[1]
        for i in range(num_new):
            t = float(i) / num_divs
            out_pts.append([splines[0].evaluate(t), splines[1].evaluate(t), splines[2].evaluate(t)])
            while t > t_hi and idx < npts - 2:
                idx += 1
                t_lo, t_hi = tcoord_map[idx], tcoord_map[idx + 1]
            e0.append(ids[idx])
            e1.append(ids[idx + 1])
            et.append((t - t_lo) / (t_hi - t_lo))
            tcoords.append(t)
        out_cells.append(np.arange(offset, offset + num_new))
        cell_rows.append(in_cell_id if vmtk_cell_data else n_verts + in_cell_id)
        offset += num_new

    out_pd = {k: interpolate_edge(v, e0, e1, et).reshape((-1,) + v.shape[1:]) for k, v in point_data.items()}
    tc_arr = np.zeros((len(tcoords), 2), dtype=np.float32)
    tc_arr[:, 0] = np.array(tcoords, dtype=np.float64).astype(np.float32)
    out_pd["TCoords"] = tc_arr
    rows = np.array(cell_rows, dtype=np.int64)
    out_cd = {k: v[rows] for k, v in cell_data.items()}
    return Centerlines(np.array(out_pts, dtype=np.float64).reshape(-1, 3), out_cells, out_pd, out_cd, {})
