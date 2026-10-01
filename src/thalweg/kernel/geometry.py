"""Path geometry, thalweg's own: one smoothing spline per path, and what is read from it.

vmtk's centerline geometry (``thalweg.vmtk.geometry``, kept for compatibility) differentiates the
sampled polyline itself, which on 0.3 mm samples of a voxel-derived path amplifies the sampling
noise: curvature is dominated by the step pattern and torsion is unstable wherever curvature is
small. Here a path is fitted once (:class:`SmoothPath`): a cubic smoothing spline whose per-point
weights 1 / (0.15 r + 0.05) hold the deviation to ~15 % of the local radius (one global budget let
a 19 mm trunk's allowance cut distal corners - research/vessels/straighten.py). From it:

- :meth:`SmoothPath.stations`: points every ``step`` mm of arc length with parallel-transport
  frames, for cross-sections (reproduces straighten.py's stations exactly);
- :meth:`SmoothPath.geometry`: the spline's analytic curvature k = |r' x r''| / |r'|^3 and torsion
  t = (r' x r'') . r''' / |r' x r''|^2 (1/mm; NaN where k < ``min_curvature``, where the osculating
  plane is undefined), and per path:
  - ``distance_metric`` (DM): arc length / chord; 1 for a straight path;
  - ``sum_of_angles`` (SOAM, Bullitt et al. 2003): total turning per mm, the in-plane angle between
    successive tangents combined with the torsional angle between successive binormals. A
    binormal flip (> 90 degrees: the path changes the side it bends to) is an inflection, not
    torsion, and is not counted as torsional turning;
  - ``inflection_count``: those flips, between stations where the curvature is defined; and the
    inflection count metric ICM = (inflections + 1) * DM.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_CURVATURE = 0.01            # 1/mm: a radius of curvature of 10 cm counts as straight


def transport_frames(C: np.ndarray, initial=None):
    """Tangents (np.gradient of the points) and parallel-transport normals n1, n2 = T x n1.

    ``initial``: n1 at the first station (projected onto the normal plane); default T0 x z (or
    T0 x x when T0 is within ~25 degrees of z), the reference's choice."""
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    n1 = np.zeros_like(T)
    if initial is None:
        a0 = np.array([0, 0, 1.0]) if abs(T[0, 2]) < 0.9 else np.array([1.0, 0, 0])
        n1[0] = np.cross(T[0], a0)
    else:
        v = np.asarray(initial, float)
        n1[0] = v - (v @ T[0]) * T[0]
    n1[0] /= np.linalg.norm(n1[0])
    for i in range(1, len(T)):
        v = n1[i - 1] - (n1[i - 1] @ T[i]) * T[i]
        n1[i] = v / np.linalg.norm(v)
    return T, n1, np.cross(T, n1)


@dataclass
class Stations:
    """``centers`` (n, 3), ``tangents``, ``n1``, ``n2`` (unit, n1 x n2 = tangent direction),
    ``s`` arc length (mm) along the smoothed line, ``radius`` interpolated from the input."""

    centers: np.ndarray
    tangents: np.ndarray
    n1: np.ndarray
    n2: np.ndarray
    s: np.ndarray
    radius: np.ndarray


@dataclass
class PathGeometry:
    s: np.ndarray               # arc length of each station, mm
    points: np.ndarray          # smoothed station points
    curvature: np.ndarray       # 1/mm
    torsion: np.ndarray         # 1/mm, NaN where curvature < min_curvature
    length: float
    chord: float
    distance_metric: float
    sum_of_angles: float        # rad/mm
    inflection_count: int
    inflection_count_metric: float


class SmoothPath:
    """One path's smoothing spline (see the module docstring), fitted once."""

    def __init__(self, points, radius, relative: float = 0.15, floor: float = 0.05, samples: int = 4000):
        from scipy.interpolate import splev, splprep
        self.points = np.asarray(points, float)
        self.radius = np.asarray(radius, float)
        keep = np.r_[True, np.linalg.norm(np.diff(self.points, axis=0), axis=1) > 1e-6]
        self.k = min(3, int(keep.sum()) - 1)
        if self.k < 1:
            raise ValueError("a path needs at least two distinct points")
        self.tck, _ = splprep(self.points[keep].T, w=1.0 / (relative * self.radius[keep] + floor),
                              s=float(keep.sum()), k=self.k)
        self.u = np.linspace(0, 1, samples)
        self.fine = np.array(splev(self.u, self.tck)).T
        self.arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(self.fine, axis=0), axis=1))]
        self.length = float(self.arc[-1])

    def along(self, s) -> np.ndarray:
        """Arc lengths on the spline -> arc lengths along the input polyline (mm): the spline's
        parameter is the polyline's normalized arc length, so a station's position along the edge
        is reported in the edge's own length (the spline itself can run a little longer or
        shorter)."""
        poly = float(np.sum(np.linalg.norm(np.diff(self.points, axis=0), axis=1)))
        return np.interp(np.asarray(s, float), self.arc, self.u) * poly

    def stations(self, step: float = 0.5) -> Stations:
        """Stations every ``step`` mm from the start (none on a path shorter than two steps' worth
        of samples: a frame needs a tangent from at least two points)."""
        s = np.arange(0, self.length, step)
        if len(s) < 2:
            z = np.zeros((0, 3))
            return Stations(z, z, z, z, np.zeros(0), np.zeros(0))
        C = np.stack([np.interp(s, self.arc, self.fine[:, a]) for a in range(3)], 1)
        r = np.interp(s, np.r_[0, np.cumsum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))],
                      self.radius)
        T, n1, n2 = transport_frames(C)
        return Stations(C, T, n1, n2, s, r)

    def geometry(self, step: float = 0.5, min_curvature: float = MIN_CURVATURE, start: float = 0.0,
                 stop: float | None = None) -> PathGeometry:
        """Curvature, torsion and tortuosity over arc lengths [start, stop] (default: the whole path).
        A traced path bends sharply where it enters a junction's ball or a tube's rounded end; a caller
        measuring a branch's own shape excludes about one radius at each end."""
        from scipy.interpolate import splev
        stop = self.length if stop is None else min(stop, self.length)
        start = max(0.0, min(start, stop))
        span = stop - start
        grid = np.arange(0, span, step)
        s = (start + (np.r_[grid, span] if span - grid[-1] > 1e-9 else grid) if span > step
             else start + np.array([0.0, span]))
        us = np.interp(s, self.arc, self.u)
        d1 = np.array(splev(us, self.tck, der=1)).T
        d2 = np.array(splev(us, self.tck, der=2)).T if self.k >= 2 else np.zeros_like(d1)
        d3 = np.array(splev(us, self.tck, der=3)).T if self.k >= 3 else np.zeros_like(d1)
        c12 = np.cross(d1, d2)
        n12 = np.linalg.norm(c12, axis=1)
        sp = np.linalg.norm(d1, axis=1)
        curvature = n12 / np.maximum(sp ** 3, 1e-300)
        defined = curvature >= min_curvature
        torsion = np.where(defined, (c12 * d3).sum(1) / np.maximum(n12 ** 2, 1e-300), np.nan)
        P = np.array(splev(us, self.tck)).T
        chord = float(np.linalg.norm(P[-1] - P[0]))
        length = span
        dm = length / chord if chord > 0 else float("inf")
        T = d1 / sp[:, None]
        ip = np.arccos(np.clip((T[1:] * T[:-1]).sum(1), -1, 1))
        B = c12 / np.maximum(n12, 1e-300)[:, None]
        both = defined[1:] & defined[:-1]
        cosb = np.clip((B[1:] * B[:-1]).sum(1), -1, 1)
        tp = np.where(both & (cosb >= 0), np.arccos(cosb), 0.0)          # a flip is an inflection
        soam = float(np.sqrt(ip ** 2 + tp ** 2).sum() / length) if length > 0 else 0.0
        N = np.cross(B, T)
        idx = np.nonzero(defined)[0]
        infl = int(((N[idx[1:]] * N[idx[:-1]]).sum(1) < 0).sum()) if len(idx) > 1 else 0
        return PathGeometry(s=s, points=P, curvature=curvature, torsion=torsion, length=length, chord=chord,
                            distance_metric=dm, sum_of_angles=soam, inflection_count=infl,
                            inflection_count_metric=(infl + 1) * dm)


def path_geometry(points, radius, step: float = 0.5, relative: float = 0.15, floor: float = 0.05,
                  samples: int = 4000, min_curvature: float = MIN_CURVATURE) -> PathGeometry:
    """Curvature, torsion and tortuosity of one path (see the module docstring)."""
    return SmoothPath(points, radius, relative, floor, samples).geometry(step, min_curvature)


def direction_at(points: np.ndarray, distance: float, start: float = 0.0) -> np.ndarray:
    """Unit chord direction along a path from arc length ``start`` to ``start + distance`` (clipped to
    the path's length): how a branch leaves a junction when ``start`` clears the junction's ball."""
    p = np.asarray(points, float)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    a = min(start, s[-1])
    b = min(start + distance, s[-1])
    pa = np.array([np.interp(a, s, p[:, c]) for c in range(3)])
    pb = np.array([np.interp(b, s, p[:, c]) for c in range(3)])
    v = pb - pa
    n = np.linalg.norm(v)
    return v / n if n > 0 else v
