"""vtkvmtkCenterlineUtilities in numpy.

Every function follows its C++ original line for line (vmtk master 66b068a); names are snake_case
versions of vmtk's. The VTK cell queries they lean on (``vtkPolyLine::EvaluatePosition`` and
below) are in :mod:`._vtk`. Ids are plain ints and lists; "unique" inserts
keep first-seen order, as ``vtkIdList::InsertUniqueId`` does.

**The InterpolateTuple defect.** ``vtkvmtkCenterlineUtilities::InterpolateTuple`` fetches both
end tuples with ``vtkDataArray::GetTuple(id)``, which returns one shared internal buffer: the
second call overwrites the first, so every "interpolated" value is the segment's END value
(radius, abscissa, normal), whatever ``pcoord`` is. :func:`interpolate_tuple` interpolates
correctly by default; ``vmtk_interp=True`` reproduces vmtk's output, for oracle comparison only.
"""
from __future__ import annotations

import numpy as np

from ._vtk import VTK_VMTK_DOUBLE_TOL, polyline_evaluate_position
from .centerlines import Centerlines


def _insert_unique(lst: list, v) -> None:
    if v not in lst:
        lst.append(v)


def get_max_group_id(cl: Centerlines, group_ids: str = "GroupIds") -> int:
    g = cl.cell_data[group_ids]
    return int(max((int(x) for x in g), default=-1))


def get_groups_id_list(cl: Centerlines, group_ids: str = "GroupIds", blanking: str | None = None,
                       blanked: int = 0) -> list[int]:
    """Group ids in first-seen cell order; with ``blanking``, only cells whose blanking == ``blanked``."""
    g = cl.cell_data[group_ids]
    b = cl.cell_data[blanking] if blanking else None
    out: list[int] = []
    for i in range(cl.n_cells):
        if b is not None and int(b[i]) != blanked:
            continue
        _insert_unique(out, int(g[i]))
    return out


def get_non_blanked_groups_id_list(cl, group_ids="GroupIds", blanking="Blanking") -> list[int]:
    return get_groups_id_list(cl, group_ids, blanking, 0)


def get_blanked_groups_id_list(cl, group_ids="GroupIds", blanking="Blanking") -> list[int]:
    return get_groups_id_list(cl, group_ids, blanking, 1)


def get_group_cell_ids(cl: Centerlines, group_id: int, group_ids: str = "GroupIds") -> list[int]:
    g = cl.cell_data[group_ids]
    return [i for i in range(cl.n_cells) if int(g[i]) == group_id]


def get_group_unique_cell_ids(cl: Centerlines, group_id: int, group_ids: str = "GroupIds") -> list[int]:
    """The group's cells, dropping any whose first AND last points coincide (squared distance <
    1e-12) with an already kept cell's."""
    g = cl.cell_data[group_ids]
    out: list[int] = []
    for i in range(cl.n_cells):
        if int(g[i]) != group_id:
            continue
        pi = cl.cell_points(i)
        dup = False
        for j in out:
            pj = cl.cell_points(j)
            if (np.sum((pj[0] - pi[0]) ** 2) < VTK_VMTK_DOUBLE_TOL
                    and np.sum((pj[-1] - pi[-1]) ** 2) < VTK_VMTK_DOUBLE_TOL):
                dup = True
                break
        if not dup:
            out.append(i)
    return out


def get_centerline_cell_ids(cl: Centerlines, centerline_id: int, centerline_ids: str = "CenterlineIds",
                            tract_ids: str | None = None) -> list[int]:
    """Cells of one centerline; with ``tract_ids``, bubble-sorted by tract id (vmtk's stable swap)."""
    c = cl.cell_data[centerline_ids]
    cells = [i for i in range(cl.n_cells) if int(c[i]) == centerline_id]
    if tract_ids is None:
        return cells
    t = cl.cell_data[tract_ids]
    tr = [int(t[i]) for i in cells]
    done = False
    while not done:
        done = True
        for i in range(len(cells) - 1):
            if tr[i] > tr[i + 1]:
                tr[i], tr[i + 1] = tr[i + 1], tr[i]
                cells[i], cells[i + 1] = cells[i + 1], cells[i]
                done = False
    return cells


def is_cell_blanked(cl: Centerlines, cell_id: int, blanking: str = "Blanking") -> int:
    return 1 if int(cl.cell_data[blanking][cell_id]) == 1 else 0


def is_group_blanked(cl: Centerlines, group_id: int, group_ids: str = "GroupIds",
                     blanking: str = "Blanking") -> int:
    g = cl.cell_data[group_ids]
    for i in range(cl.n_cells):
        if int(g[i]) == group_id:
            return is_cell_blanked(cl, i, blanking)
    return -1


def find_adjacent_centerline_group_ids(cl: Centerlines, group_id: int, group_ids: str = "GroupIds",
                                       centerline_ids: str = "CenterlineIds",
                                       tract_ids: str = "TractIds") -> tuple[list[int], list[int]]:
    """(upstream, downstream) group ids: along every centerline through ``group_id``, the groups of
    the tracts just before and just after (vmtk assumes adjacent tract ids).

    vmtk skips cells that are not VTK_LINE / VTK_POLY_LINE; every cell of a centerline polydata is
    one of those (``vtkPolyData`` types a 2-point line cell VTK_LINE and any other, one point
    included, VTK_POLY_LINE), so no cell is skipped here."""
    g, c, t = cl.cell_data[group_ids], cl.cell_data[centerline_ids], cl.cell_data[tract_ids]
    up: list[int] = []
    down: list[int] = []
    for i in range(cl.n_cells):
        if int(g[i]) != group_id:
            continue
        ci, ti = int(c[i]), int(t[i])
        for j in range(cl.n_cells):
            gj = int(g[j])
            if gj == group_id or int(c[j]) != ci:
                continue
            if int(t[j]) == ti - 1:
                _insert_unique(up, gj)
            if int(t[j]) == ti + 1:
                _insert_unique(down, gj)
    return up, down


def interpolate_point(cl: Centerlines, cell_id: int, sub_id: int, pcoord: float) -> np.ndarray:
    p = cl.cell_points(cell_id)
    return (1.0 - pcoord) * p[sub_id] + pcoord * p[sub_id + 1]


def interpolate_tuple(cl: Centerlines, name: str, cell_id: int, sub_id: int, pcoord: float,
                      vmtk_interp: bool = False) -> np.ndarray:
    """A point array linearly interpolated at (cell, sub_id, pcoord). ``vmtk_interp=True`` gives
    vmtk's result instead: the value at ``sub_id + 1`` (see the module docstring)."""
    a = cl.point_data[name]
    ids = cl.cells[cell_id]
    t0 = np.atleast_1d(np.asarray(a[ids[sub_id]], dtype=np.float64))
    t1 = np.atleast_1d(np.asarray(a[ids[sub_id + 1]], dtype=np.float64))
    if vmtk_interp:
        t0 = t1
    return (1.0 - pcoord) * t0 + pcoord * t1


def find_merging_points(cl: Centerlines, tolerance: float = 1e-8) -> np.ndarray:
    """Points where a cell starts or stops coinciding with a later cell (vmtk's FindMergingPoints,
    including its carried ``previousMergeStatus`` across the inner loop)."""
    out = []
    for i in range(cl.n_cells):
        pts_i = cl.cell_points(i)
        prev = -1
        for j, p in enumerate(pts_i):
            for k in range(i + 1, cl.n_cells):
                _, _, _, _, d2 = polyline_evaluate_position(p, cl.cell_points(k))
                status = 1 if d2 < tolerance * tolerance else 0
                if j > 0 and status != prev:
                    out.append(p.copy())
                    prev = status
    return np.array(out).reshape(-1, 3)
