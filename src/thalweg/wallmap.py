"""Wall maps: a branch's wall unrolled, r(s, angle), by ray casting in the field.

vmtk maps a surface onto each branch (``vmtkbranchmetrics``, ``vmtkbranchmapping``,
``vmtkbranchpatching``: abscissa and angle per surface point, then a raster). Here there is no
surface: from every station along a branch, rays run outward at every angle to the first zero
crossing of the margin (:func:`thalweg.kernel.rays.first_crossing`). The distances are the map - a
raster from the start, ``radius_mm[station, angle]``. Anything defined at the wall maps the same
way: sample it at ``WallMap.wall_points()``.

Two conventions:

- :func:`edge_wall_map`: thalweg's. Stations every ``step`` mm on the edge's smoothed path
  (:func:`thalweg.kernel.sections.stations`), arc length from the edge's start, angle from the
  station's parallel-transport normal n1 toward n2 (a frame carried along the edge without twist;
  its zero is arbitrary per edge);
- :func:`group_wall_map`: vmtk's. Stations at the points of a branch group's longest cell,
  abscissas and normals in vmtk's offset convention (zero at the root bifurcation, the normal
  tied to the bifurcation's plane), angle as ``vmtkbranchmetrics`` measures it. Against vmtk's own
  ``DistanceToCenterlines`` at the same (group, abscissa, angle), the ray-cast radius differs by
  a median 0.02-0.03 mm (``tests/test_wallmap.py``).

**Ostia.** Where a side branch leaves, the ray runs into it: far past the branch's own wall, or to
no crossing at all. :func:`ostium` marks those rays (no crossing, or more than ``factor`` times the
station's median radius). They are openings, not wall.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import ThalwegError
from .graph import TubeGraph
from .kernel import sections
from .kernel.rays import first_crossing
from .vmtk.centerlines import ABSCISSAS, BLANKING, GROUP_IDS, NORMALS, RADIUS, Centerlines


@dataclass
class WallMap:
    """``radius_mm[i, j]``: the wall's distance from station i at angle j (NaN: no crossing)."""

    arc_length_mm: np.ndarray        # (S,) thalweg: from the edge's start; vmtk: the offset abscissa
    angle_rad: np.ndarray            # (A,) in [-pi, pi)
    radius_mm: np.ndarray            # (S, A)
    centers: np.ndarray              # (S, 3) the stations
    directions: np.ndarray           # (S, A, 3) unit ray directions

    def wall_points(self) -> np.ndarray:
        """(S, A, 3) the wall points the rays found (NaN where a ray found none)."""
        return self.centers[:, None, :] + self.radius_mm[:, :, None] * self.directions


def angles(count: int = 72) -> np.ndarray:
    """``count`` equally spaced angles in [-pi, pi)."""
    return -np.pi + 2.0 * np.pi * np.arange(count) / count


def _cast(m, geometry, centers, n, b, radius, s, angle, reach_factor, reach_extra, ray_step) -> WallMap:
    dirs = np.cos(angle)[None, :, None] * n[:, None, :] + np.sin(angle)[None, :, None] * b[:, None, :]
    reach = reach_factor * float(np.max(radius)) + reach_extra
    origins = np.repeat(centers[:, None, :], len(angle), axis=1)
    r = first_crossing(m, geometry, origins.reshape(-1, 3), dirs.reshape(-1, 3), reach, ray_step)
    return WallMap(np.asarray(s, float), angle, r.reshape(len(centers), len(angle)), centers, dirs)


def edge_wall_map(graph: TubeGraph, edge: int, m: np.ndarray, geometry, step: float = 0.5,
                  angle_count: int = 72, reach_factor: float = 3.0, reach_extra: float = 2.0,
                  ray_step: float = 0.05) -> WallMap:
    """The wall map of one edge in thalweg's convention (see the module docstring). Rays search to
    ``reach_factor`` times the edge's largest traced radius plus ``reach_extra`` mm."""
    p, r = graph.edge_points(edge), graph.edge_radius(edge)
    if len(p) < 4:
        raise ThalwegError(f"edge {edge} has {len(p)} samples; a wall map needs at least 4")
    st = sections.stations(p, np.where(r > 0, r, 0.5), step=step)
    return _cast(m, geometry, st.centers, st.n1, st.n2, st.radius, st.s, angles(angle_count),
                 reach_factor, reach_extra, ray_step)


def group_wall_map(cl: Centerlines, group: int, m: np.ndarray, geometry, angle_count: int = 72,
                   smooth: float = 0.6, reach_factor: float = 3.0, reach_extra: float = 2.0,
                   ray_step: float = 0.05) -> WallMap:
    """The wall map of one vmtk branch group, in vmtk's coordinates. ``cl``: the split centerlines
    after ``offset_attributes`` (:attr:`thalweg.branching.Branching.offset`). The station line is
    the group's longest non-blanked cell, smoothed over ``smooth`` mm of arc length for the ray
    frame (0: the raw polyline, whose 0.3 mm segments jitter the tangent); the ray at angle phi
    runs along cos(phi) N + sin(phi) (N x T)."""
    from scipy import ndimage as ndi
    gid = np.asarray(cl.cell_data[GROUP_IDS]).reshape(-1)
    blank = np.asarray(cl.cell_data[BLANKING]).reshape(-1)
    cand = np.nonzero((gid == group) & (blank == 0))[0]
    if not len(cand):
        raise ThalwegError(f"group {group} has no non-blanked cell")
    ids = cl.cells[max(cand, key=lambda i: len(cl.cells[i]))]
    gap = np.r_[np.inf, np.linalg.norm(np.diff(cl.points[ids], axis=0), axis=1)]
    ids = ids[gap > 0.05]                                  # near-duplicate points give no tangent
    if len(ids) < 4:
        raise ThalwegError(f"group {group}'s longest cell has {len(ids)} distinct points; a wall map needs 4")
    C = cl.points[ids]
    if smooth > 0:
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
        C = ndi.gaussian_filter1d(C, smooth / np.median(np.diff(arc)), axis=0, mode="nearest")
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    N = np.asarray(cl.point_data[NORMALS], float)[ids]
    N = N - (N * T).sum(1, keepdims=True) * T
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    radius = np.asarray(cl.point_data[RADIUS], float).reshape(-1)[ids]
    s = np.asarray(cl.point_data[ABSCISSAS], float).reshape(-1)[ids]
    return _cast(m, geometry, C, N, np.cross(N, T), radius, s, angles(angle_count),
                 reach_factor, reach_extra, ray_step)


def ostium(wall: WallMap, factor: float = 1.8) -> np.ndarray:
    """(S, A) bool: rays that found no wall, or one more than ``factor`` times their station's
    median radius away - they left along a side branch."""
    r = wall.radius_mm
    some = np.isfinite(r).any(axis=1, keepdims=True)                  # a station with no crossing at all
    med = np.nanmedian(np.where(some, r, 0.0), axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        return ~np.isfinite(r) | (r > factor * med)


def wall_maps(graph: TubeGraph, structure: str, m: np.ndarray, geometry, min_length_mm: float = 3.0,
              **kw) -> dict[int, WallMap]:
    """The wall map of every edge of the structure at least ``min_length_mm`` long (and with at
    least 4 samples), by edge id. ``kw`` go to :func:`edge_wall_map`."""
    out = {}
    for e in graph.structure_edges(structure):
        if e.length_mm >= min_length_mm and e.point_range[1] - e.point_range[0] >= 4:
            out[e.id] = edge_wall_map(graph, e.id, m, geometry, **kw)
    return out


def write_wall_maps(maps: dict[int, WallMap], path, structure: str = "") -> None:
    """One ``.npz``: ``structure``, ``edges`` (the ids), and per edge ``edge_<id>_arc_length_mm``
    (S,), ``edge_<id>_radius_mm`` (S, A) and ``edge_<id>_center_mm`` (S, 3); ``angle_rad`` (A,) is
    shared (every map must have the same angles). World coordinates are LPS, as in the graph."""
    if not maps:
        raise ThalwegError("no wall maps to write")
    first = next(iter(maps.values())).angle_rad
    arrays = dict(structure=np.array(structure), edges=np.array(sorted(maps), np.int64), angle_rad=first)
    for k, w in maps.items():
        if len(w.angle_rad) != len(first) or not np.array_equal(w.angle_rad, first):
            raise ThalwegError("wall maps with different angles cannot share a file")
        arrays[f"edge_{k}_arc_length_mm"] = w.arc_length_mm
        arrays[f"edge_{k}_radius_mm"] = w.radius_mm.astype(np.float32)
        arrays[f"edge_{k}_center_mm"] = w.centers
    np.savez_compressed(path, **arrays)
