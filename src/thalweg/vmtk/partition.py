"""vtkvmtkPolyDataCenterlineGroupsClipper's labeling rule: which branch group a point belongs to.

``vmtkbranchclipper`` cuts a surface into branches. Underneath the cutting is a point-wise rule:
a point belongs to the non-blanked centerline group whose tube function
(:mod:`.polyball`, ``vtkvmtkPolyBallLine``) is lowest there; blanked cells - the bifurcation
regions - take no part. This module is that rule and nothing else: no surface, no clipping. It
labels any points - a surface's vertices, or every lattice point inside a structure.

Public:

- :func:`group_tubes`: the non-blanked cells' segments and each segment's group;
- :func:`lowest_label`: for each query point, the label of the segment with the lowest tube value
  and that value. Ties go to the lowest label. Dense by default; with ``pairs`` (point index,
  segment index) only those pairs are evaluated - a caller that can prove the other pairs cannot
  win (``thalweg.partition`` does, with a spatial index) gets the same labels for a fraction of
  the work.

Checked against vmtk 1.5.2 on the C3N-00704 subtree, with vmtk's centerlines and with ours: at
each of the 21,221 vertices of vmtkBranchClipper's input surface that reach its output, the label
is vmtk's. The clipper also inserts points along its cuts (about 7,000); those lie where two
groups' values tie, and there the label is either neighbor's: about 3,200 differ, by a value gap
of 5e-8 mm^2 (vmtk's centerlines) and 1.5e-4 mm^2 (ours) in the median and 0.34 and 0.58 mm^2 at
most (``tests/test_partition.py``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._vtk import VTK_VMTK_LARGE_DOUBLE
from .centerlines import BLANKING, GROUP_IDS, RADIUS, Centerlines
from .polyball import TubeSegments, segment_values, tube_segments

DENSE_CHUNK = 1 << 22        # (points x segments) values evaluated at once in the dense path


@dataclass
class LabeledTubes:
    """Tube segments and the label (a group id, an edge id) each one carries."""

    segments: TubeSegments
    label: np.ndarray            # (S,) int64

    def __len__(self) -> int:
        return len(self.label)


def group_tubes(cl: Centerlines, radius: str = RADIUS, group_ids: str = GROUP_IDS,
                blanking: str = BLANKING) -> LabeledTubes:
    """The segments of the split centerlines' non-blanked cells, each labeled with its cell's group."""
    blank = np.asarray(cl.cell_data[blanking]).reshape(-1)
    group = np.asarray(cl.cell_data[group_ids]).reshape(-1)
    cells = [k for k in range(cl.n_cells) if blank[k] != 1]
    seg = tube_segments(cl, radius, cells)
    return LabeledTubes(seg, group[seg.cell].astype(np.int64))


def _reduce(point: np.ndarray, value: np.ndarray, label: np.ndarray, n: int):
    """Per point: the lowest value among its pairs and that pair's label (ties: the lowest label)."""
    out_label = np.full(n, -1, np.int64)
    out_value = np.full(n, VTK_VMTK_LARGE_DOUBLE)
    ok = np.isfinite(value)
    point, value, label = point[ok], value[ok], label[ok]
    if len(point):
        o = np.lexsort((label, value, point))
        point, value, label = point[o], value[o], label[o]
        first = np.concatenate([[True], point[1:] != point[:-1]])
        out_label[point[first]] = label[first]
        out_value[point[first]] = value[first]
    return out_label, out_value


def lowest_label(x, tubes: LabeledTubes, pairs: tuple[np.ndarray, np.ndarray] | None = None):
    """``(label, value)`` per query point: the label of the segment whose tube value is lowest, and
    the value (negative inside that tube). -1 and vmtk's large double where no segment is usable."""
    x = np.asarray(x, dtype=np.float64).reshape(-1, 3)
    n, s = len(x), len(tubes)
    if n == 0 or s == 0:
        return np.full(n, -1, np.int64), np.full(n, VTK_VMTK_LARGE_DOUBLE)
    if pairs is not None:
        pi, si = (np.asarray(a, np.int64) for a in pairs)
        val = segment_values(x[pi], tubes.segments.take(si), pairwise=True)
        return _reduce(pi, val, tubes.label[si], n)
    out_label = np.full(n, -1, np.int64)
    out_value = np.full(n, VTK_VMTK_LARGE_DOUBLE)
    order = np.argsort(tubes.label, kind="stable")           # ties then fall to the lowest label
    seg, lab = tubes.segments.take(order), tubes.label[order]
    step = max(1, DENSE_CHUNK // s)
    for a in range(0, n, step):
        val = segment_values(x[a:a + step], seg)
        val = np.where(np.isfinite(val), val, np.inf)        # a NaN radius makes no tube
        k = np.argmin(val, axis=1)                           # first occurrence
        best = val[np.arange(len(k)), k]
        ok = np.isfinite(best)
        out_label[a:a + step][ok] = lab[k[ok]]
        out_value[a:a + step][ok] = best[ok]
    return out_label, out_value
