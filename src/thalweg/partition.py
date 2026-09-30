"""The branch partition: which branch every point of a structure belongs to, read from the field.

vmtk's ``vmtkbranchclipper`` cuts a SURFACE into branches. The rule underneath is point-wise (a
point belongs to the branch whose tube function is lowest there, :mod:`thalweg.vmtk.partition`), so
it needs no surface: here it labels any points, and in particular every lattice point inside a
structure - the partition as a label volume, with a volume per branch.

Two sets of tubes can do the labeling:

- :func:`edge_tubes`: the graph's own edges, each a tube of its traced radii, labeled with the
  edge id. Every edge takes part (there are no blanked bifurcation regions), so the junction
  volume is shared out among the edges that meet there. This is thalweg's partition.
- :func:`group_tubes`: vmtk's branch groups from :func:`thalweg.branching.vmtk_branching`, labeled
  with the group id, bifurcation cells left out - what vmtkBranchClipper does.

Then:

- :func:`label_points`: label and tube value for any world points. Exact, and fast: a spatial
  index finds, per point, the only segments that can have the lowest value (see below);
- :func:`label_field`: an int32 volume on the field's grid, the label at every lattice point
  inside the structure's traced piece, -1 elsewhere (outside, and in pieces the tracer dropped);
- :func:`label_volumes`: mm^3 per label (lattice points times the voxel volume);
- :func:`volume_rows`: ``volume_mm3`` on branch-table rows.

**Why the pruning is exact.** A segment's value at x is ``|x - c|^2 - r^2`` for a center c on the
segment and its radius r there, so it is at least ``dist(x, segment)^2 - R^2`` with R the largest
radius in the segment's class. The value U at the segment whose midpoint is nearest x is an upper
bound on the minimum. A segment that beats U must therefore lie within ``sqrt(U + R^2)`` of x;
segments are grouped into radius classes so that the many thin ones are searched in small balls.
Every segment that can win is evaluated, with vmtk's arithmetic.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .errors import ThalwegError
from .graph import TubeGraph
from .vmtk import partition as _vp
from .vmtk.partition import LabeledTubes
from .vmtk.polyball import TubeSegments, segment_values

RADIUS_CLASSES = (0.75, 1.5, 3.0, 6.0, 12.0)            # class edges, mm
POINT_CHUNK = 100_000


def edge_tubes(graph: TubeGraph, structure: str) -> LabeledTubes:
    """Each edge of the structure as a tube of its traced radii, labeled with the edge id. A sample
    without a radius (the refinement found no inside point) takes 0: it still attracts the points
    nearest its centerline."""
    p0, p1, r0, r1, cell, sub = [], [], [], [], [], []
    for e in graph.structure_edges(structure):
        p = graph.edge_points(e)
        r = np.maximum(graph.edge_radius(e), 0.0)
        n = len(p) - 1
        p0.append(p[:-1]), p1.append(p[1:]), r0.append(r[:-1]), r1.append(r[1:])
        cell.append(np.full(n, e.id, np.int64)), sub.append(np.arange(n, dtype=np.int64))
    if not p0:
        raise ThalwegError(f"{structure!r} has no edges to partition by")
    seg = TubeSegments(*(np.concatenate(a) for a in (p0, p1, r0, r1, cell, sub)))
    return LabeledTubes(seg, seg.cell.copy())


def group_tubes(branching) -> LabeledTubes:
    """vmtk's non-blanked branch groups (a :class:`thalweg.branching.Branching`), labeled by group."""
    return _vp.group_tubes(branching.split)


def _candidates(x: np.ndarray, tubes: LabeledTubes) -> tuple[np.ndarray, np.ndarray]:
    """(point, segment) pairs holding, for every point, each segment that can have the lowest value."""
    seg = tubes.segments
    mid = 0.5 * (seg.p0 + seg.p1)
    half = 0.5 * np.linalg.norm(seg.p1 - seg.p0, axis=1)
    rmax = np.maximum(np.abs(seg.r0), np.abs(seg.r1))
    near = cKDTree(mid).query(x, workers=-1)[1]
    upper = segment_values(x, seg.take(near), pairwise=True)       # a bound on the minimum
    cls = np.searchsorted(RADIUS_CLASSES, rmax)
    ps, ss = [np.arange(len(x))], [near]
    for c in np.unique(cls):
        idx = np.nonzero(cls == c)[0]
        R, h = float(rmax[idx].max()), float(half[idx].max())
        reach2 = upper + R * R
        sel = np.nonzero(~(reach2 < 0))[0]                         # inf (an unusable nearest) stays in
        if not len(sel):
            continue
        reach = np.sqrt(np.maximum(reach2[sel], 0.0))
        span = float(np.linalg.norm(mid[idx].max(0) - mid[idx].min(0))) + 1.0
        reach = np.where(np.isfinite(reach), reach, span + np.linalg.norm(x[sel] - mid[idx].mean(0), axis=1))
        reach = (reach + h) * (1.0 + 1e-9) + 1e-9
        hits = cKDTree(mid[idx]).query_ball_point(x[sel], reach, workers=-1)
        n = np.fromiter((len(v) for v in hits), np.int64, len(hits))
        if n.sum():
            ps.append(np.repeat(sel, n))
            ss.append(idx[np.concatenate([np.asarray(v, np.int64) for v in hits if v])])
    return np.concatenate(ps), np.concatenate(ss)


def label_points(x, tubes: LabeledTubes, exhaustive: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """``(label, value)`` for world points ``x``: the tube with the lowest value at each, and that
    value (negative inside the tube). ``exhaustive=True`` evaluates every (point, segment) pair -
    the same answer, for checking."""
    x = np.asarray(x, dtype=np.float64).reshape(-1, 3)
    if exhaustive or len(x) == 0 or len(tubes) == 0:
        return _vp.lowest_label(x, tubes)
    label = np.empty(len(x), np.int64)
    value = np.empty(len(x))
    for a in range(0, len(x), POINT_CHUNK):
        xx = x[a:a + POINT_CHUNK]
        label[a:a + POINT_CHUNK], value[a:a + POINT_CHUNK] = _vp.lowest_label(
            xx, tubes, pairs=_candidates(xx, tubes))
    return label, value


def traced_mask(m: np.ndarray, geometry, points: np.ndarray) -> np.ndarray:
    """The lattice points with m > 0 in pieces (26-connected) that hold a centerline sample: the
    structure's traced piece, without the fragments the tracer dropped."""
    lab, _ = ndimage.label(m > 0, structure=np.ones((3, 3, 3), bool))
    d = np.asarray(geometry.directions, float)
    idx = np.rint((np.asarray(points, float) - np.asarray(geometry.origin, float)) @ np.linalg.inv(d))
    idx = idx.astype(np.int64)
    ok = ((idx >= 0) & (idx < np.asarray(m.shape))).all(1)
    keep = np.unique(lab[tuple(idx[ok].T)])
    return np.isin(lab, keep[keep > 0])


def label_field(m: np.ndarray, geometry, tubes: LabeledTubes, points: np.ndarray | None = None) -> np.ndarray:
    """The partition as a volume on the field's grid: int32, the label at every lattice point with
    m > 0, -1 elsewhere. With ``points`` (the structure's centerline samples), pieces of the field
    that hold none of them - what the tracer dropped - are left at -1."""
    inside = traced_mask(m, geometry, points) if points is not None else m > 0
    idx = np.argwhere(inside)
    world = np.asarray(geometry.origin, float) + idx @ np.asarray(geometry.directions, float)
    out = np.full(m.shape, -1, np.int32)
    out[tuple(idx.T)] = label_points(world, tubes)[0]
    return out


def label_volumes(labels: np.ndarray, geometry) -> dict[int, float]:
    """mm^3 per label: its lattice points times the voxel volume."""
    voxel = abs(float(np.linalg.det(np.asarray(geometry.directions, float))))
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    return {int(k): float(c) * voxel for k, c in zip(ids, counts)}


def volume_rows(rows: list[dict], volumes: dict[int, float]) -> None:
    """Add ``volume_mm3`` (the edge's share of the structure's volume; 0.0 if no lattice point is
    nearest it) to branch-table rows of one structure, in place."""
    for r in rows:
        r["volume_mm3"] = volumes.get(r["edge"], 0.0)
