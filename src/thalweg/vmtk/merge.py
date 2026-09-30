"""vtkvmtkMergeCenterlines in numpy: one polyline per branch group, joined at the bifurcation origins.

Ports ``vtkvmtkMergeCenterlines.cxx`` (vmtk master 66b068a; the class behind the
``vmtkCenterlineMerge`` script) line for line, with the VTK 9.4 filters it runs first, privately
(all three in :mod:`._vtk` / :mod:`.frames`, shared with :mod:`.resampling`):

- ``vtkCleanPolyData`` (defaults: exact point merging, consecutive duplicate ids dropped, a line
  reduced to one point becomes a vertex; point data copied at EVERY occurrence of a merged point,
  so the last occurrence wins, as VTK 9.4 does without ghost points);
- ``vtkSplineFilter`` with its default ``vtkCardinalSpline`` (open, first-derivative constraints of
  0 at both ends, knots at normalized polyline length), subdivided to length: ``int(length / step)``
  equal parameter steps per line, point data interpolated linearly along the input segment the
  parameter falls in, and a float32 ``TCoords`` array ``(t, 0)``;
- ``vtkvmtkCenterlineBifurcationReferenceSystems`` (:mod:`.frames`) for the bifurcation points.

Public: :func:`merge_centerlines`. Each non-blanked group of the resampled lines becomes one cell:
its unique cells, truncated to the shortest, averaged point by point with squared-radius weights;
point and cell data are copied from the group's first unique cell. With ``merge_blanked`` each
bifurcation's origin is added as a point (point data copied from the first upstream cell's last
resampled point, or the first downstream cell's first) and prepended / appended to the adjacent
merged cells. ``length=0`` means 1 % of the cleaned lines' bounding-box diagonal.

Defect flags (default: correct; ``True`` reproduces vmtk, for oracle comparison):

- ``vmtk_float32``: vmtk's input (a vmtk-produced polydata) holds float32 points; the clean and
  spline filters keep that type, and the merged points and the bifurcation origins go into default
  (VTK_FLOAT) ``vtkPoints``. True rounds at each of those places. The default keeps float64
  throughout; with every other flag on, its points differ from vmtk's by up to 3.2e-6 mm (phantom)
  and 7.0e-6 mm (case oracle; largest coordinate difference), the point data by <= 1.4e-14 (the
  FMA residual).
- ``vmtk_cell_data``: ``vtkSplineFilter`` indexes the cell data by line number although a cleaned
  polydata lists vertex cells first, so a line that collapses to a vertex shifts the cell data of
  the lines after it (see :mod:`._vtk`). Not triggered by vmtk's split centerlines (every oracle).

Kept (unreachable, or undefined in vmtk): ``vtkSplineFilter``'s point-to-parameter map is reset
without clearing, so a zero-length segment would read a stale parameter (the clean step removes
every such segment). A bifurcation with neither an upstream nor a downstream merged cell gets a
point but no point data in vmtk (arrays one tuple short); here its point data is NaN / 0.
"""
from __future__ import annotations

import numpy as np

from ._vtk import bounds_length, clean_lines, div, round_float32, spline_filter
from .centerlines import BLANKING, CENTERLINE_IDS, GROUP_IDS, RADIUS, TRACT_IDS, Centerlines
from .frames import bifurcation_reference_systems
from .utilities import (find_adjacent_centerline_group_ids, get_blanked_groups_id_list,
                        get_group_unique_cell_ids, get_max_group_id, get_non_blanked_groups_id_list)


# -- vtkvmtkMergeCenterlines ---------------------------------------------------------------------

def merge_centerlines(cl: Centerlines, length: float = 0.0, merge_blanked: bool = True, radius: str = RADIUS,
                      group_ids: str = GROUP_IDS, centerline_ids: str = CENTERLINE_IDS,
                      tract_ids: str = TRACT_IDS, blanking: str = BLANKING, vmtk_float32: bool = False,
                      vmtk_cell_data: bool = False) -> Centerlines:
    """Merge split centerlines into one polyline per group (vmtkCenterlineMerge).

    ``length`` is the resampling step (0: 1 % of the bounding-box diagonal). Returns a new
    :class:`Centerlines` whose point data are the input's plus ``TCoords`` (float32, (N, 2)) and whose
    cell data are the input's, one row per merged cell.
    """
    for name, where in ((radius, cl.point_data), (group_ids, cl.cell_data), (centerline_ids, cl.cell_data),
                        (tract_ids, cl.cell_data), (blanking, cl.cell_data)):
        if name not in where:
            raise KeyError(f"array {name!r} does not exist")
    if vmtk_float32:                              # the input's float32 vtkPoints
        cl = Centerlines(round_float32(cl.points), cl.cells, cl.point_data, cl.cell_data)
    points, lines, verts, pd, cd = clean_lines(cl)
    step = float(length)
    if step < 1e-12:
        step = 0.01 * bounds_length(points)
    res = spline_filter(points, lines, len(verts), pd, cd, step, vmtk_cell_data)
    if vmtk_float32:                              # vtkSplineFilter keeps the input's point type
        res.points = round_float32(res.points)

    rad = res.point_data[radius]
    rad = rad if rad.ndim == 1 else rad[:, 0]
    out_pts: list = []
    out_lines: list[list[int]] = []
    pd_rows: list[int] = []                       # resampled point whose data each output point copies
    cd_rows: list[int] = []
    max_group = get_max_group_id(res, group_ids)
    group_to_cell = [-1] * max(max_group, 0)

    for g in get_non_blanked_groups_id_list(res, group_ids, blanking):
        cells = get_group_unique_cell_ids(res, g, group_ids)
        n_merged = 0
        for j, c in enumerate(cells):
            nc = len(res.cells[c])
            if j == 0 or nc < n_merged:
                n_merged = nc
        merged_cell = len(out_lines)
        if g >= len(group_to_cell):
            group_to_cell.extend([-1] * (g + 1 - len(group_to_cell)))
        group_to_cell[g] = merged_cell
        line: list[int] = []
        for k in range(n_merged):
            mp = [0.0, 0.0, 0.0]
            wsum = 0.0
            for c in cells:
                pid = int(res.cells[c][k])
                r = float(rad[pid])
                w = r * r
                p = res.points[pid]
                mp[0] += w * float(p[0])
                mp[1] += w * float(p[1])
                mp[2] += w * float(p[2])
                wsum += w
            mp = [div(mp[0], wsum), div(mp[1], wsum), div(mp[2], wsum)]     # zero radii: NaN, as vmtk
            line.append(len(out_pts))
            out_pts.append(mp)
            pd_rows.append(int(res.cells[cells[0]][k]))
        out_lines.append(line)
        cd_rows.append(cells[0])

    if merge_blanked:
        frames = bifurcation_reference_systems(res, radius=radius, group_ids=group_ids, blanking=blanking,
                                               vmtk_float32=vmtk_float32)
        fg = frames.point_data[group_ids]
        extra = [[-1, -1] for _ in out_lines]
        for g in get_blanked_groups_id_list(res, group_ids, blanking):
            rs_id = next((i for i in range(len(fg)) if int(fg[i]) == g), -1)
            if rs_id == -1:
                raise ValueError(f"bifurcation group {g} has no reference system (vmtk reads point -1 here)")
            up, down = find_adjacent_centerline_group_ids(res, g, group_ids, centerline_ids, tract_ids)
            bif_id = len(out_pts)
            out_pts.append([float(v) for v in frames.points[rs_id]])
            source = -1
            for u in up:
                mc = group_to_cell[u] if u < len(group_to_cell) else -1
                if mc == -1:
                    continue
                if source == -1:
                    c0 = get_group_unique_cell_ids(res, u, group_ids)[0]
                    source = int(res.cells[c0][-1])
                extra[mc][1] = bif_id
            for d in down:
                mc = group_to_cell[d] if d < len(group_to_cell) else -1
                if mc == -1:
                    continue
                if source == -1:
                    c0 = get_group_unique_cell_ids(res, d, group_ids)[0]
                    source = int(res.cells[c0][0])
                extra[mc][0] = bif_id
            pd_rows.append(source)                # -1: vmtk copies no data for this point
        out_lines = [([e[0]] if e[0] != -1 else []) + line + ([e[1]] if e[1] != -1 else [])
                     for line, e in zip(out_lines, extra)]

    rows = np.array(pd_rows, dtype=np.int64)
    new_pd = {}
    for k, a in res.point_data.items():
        v = a[np.maximum(rows, 0)].copy()
        if (rows < 0).any():
            v[rows < 0] = np.nan if np.issubdtype(v.dtype, np.floating) else 0
        new_pd[k] = v
    new_cd = {k: v[np.array(cd_rows, dtype=np.int64)] for k, v in res.cell_data.items()}
    pts = np.array(out_pts, dtype=np.float64).reshape(-1, 3)
    if vmtk_float32:                              # the merged polydata's default vtkPoints
        pts = round_float32(pts)
    return Centerlines(pts, [np.array(L, dtype=np.int64) for L in out_lines], new_pd, new_cd)
