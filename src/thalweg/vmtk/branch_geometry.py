"""vtkvmtkCenterlineBranchGeometry in numpy: one length/curvature/torsion/tortuosity per branch group.

Ports ``vtkvmtkCenterlineBranchGeometry`` (vtkVmtk/ComputationalGeometry/
vtkvmtkCenterlineBranchGeometry.cxx; script ``vmtkbranchgeometry``); the touching-sphere walk it
subsamples with is :func:`.sphere_distance.find_touching_sphere_center`. Public:

- :func:`branch_geometry` - the filter, on split centerlines (``GroupIds``, ``Blanking``): one vertex
  (at the origin, as vmtk writes it) per non-blanked group, in first-seen order, with point arrays
  ``Length``, ``Curvature``, ``Torsion``, ``Tortuosity`` and ``GroupIds`` (float64, as vmtk's);
- :func:`sphere_subsample_line`, :func:`subsample_line` - the two subsamplings.

Per group, over all its cells: Length is the mean cell length; Curvature and Torsion the mean over
cells of :func:`.geometry.line_curvature` / :func:`.geometry.line_torsion`'s weighted means, computed
on each cell's points subsampled one touching sphere at a time (the default, and all the vmtk script
can do: it exposes no subsampling switch); Tortuosity is Length over the distance between the
radius^2-weighted means of the cells' first and last points, minus 1.

Defect flags (default: correct; ``True``: vmtk's result, for oracle comparison):

- ``vmtk_float32`` - the subsampled points are stored in a ``vtkPoints::New()`` (VTK_FLOAT), and
  without subsampling the cell is copied into one; curvature and torsion are then computed from
  float32-rounded points (see :mod:`.smoothing`). Length and Tortuosity read the double input.
  Effect: Curvature up to 3.0e-7 / 6.9e-5 per mm, Torsion up to 2.2e-5 / 1.6e-3 per mm (phantom /
  case oracle; relative 4e-4 and 1.2e-2 on the case).
- ``vmtk_steps`` - the touching-sphere search counts its steps from the SQUARED segment length
  (:mod:`.sphere_distance`); it is the search's quantization, which moves the subsampled points
  and so the second and third differences: Curvature up to 8.3e-7 / 4.2e-5 per mm, Torsion up to
  1.1e-5 / 3.2e-3 per mm (phantom / case oracle).
- ``vmtk_discard_smoothing`` - with ``line_smoothing``, vmtk smooths each cell and then, when sphere
  subsampling is on, overwrites the smoothed points with a subsampling of the RAW input cell, so
  ``LineSmoothing`` has no effect on anything. Correct: the touching spheres are walked along the
  smoothed points (same radii). With the script's smoothing (100 iterations, factor 0.1) the
  corrected group curvature is 0.93 of vmtk's (median; 0.50-0.96) on the phantom and 0.56 (5-95 %:
  0.12-0.79) on the case oracle; without ``line_smoothing`` the flag changes nothing.

Not flagged (unreachable from the vmtk script): ``SubsampleLine`` (``sphere_subsampling=False``)
accumulates the distance from the last kept point to every skipped point rather than the arc
length - kept as written.

Degenerate radii: a group whose cells all have a zero radius at their first (or last) point gives
Tortuosity NaN (the radius^2-weighted mean is 0 / 0), as vmtk's does. Sphere subsampling along a
segment whose two end radii sum to <= 0 raises ``ValueError``: the touching sphere of a zero radius
is its own center, so the walk advances 1e-5 of the segment per sphere (vmtk's step cap), 1e5 walk
steps of 1e5 search steps each, and with a negative sum it never advances at all (vmtk loops
forever).

**Residual against vmtk**: as in :mod:`.geometry`, vmtk's arm64 build fuses multiply-adds that numpy
rounds separately; last-bit differences only (see tests/test_vmtk_branch_geometry.py).
"""
from __future__ import annotations

import math

import numpy as np

from ._vtk import distance2, polyline_length, round_float32
from .centerlines import RADIUS, Centerlines
from .geometry import line_curvature, line_torsion
from .smoothing import smooth_line
from .sphere_distance import find_touching_sphere_center
from .utilities import get_non_blanked_groups_id_list


def sphere_subsample_line(cl: Centerlines, cell_id: int, radius: str = RADIUS, vmtk_float32: bool = False,
                          vmtk_steps: bool = False) -> np.ndarray:
    """vtkvmtkCenterlineBranchGeometry::SphereSubsampleLine: the cell's first point, then each
    touching sphere center walking toward the end (``forward=False``), then its last point."""
    pts = cl.cell_points(cell_id)
    rad_all = np.asarray(cl.point_data[radius], dtype=np.float64).reshape(len(cl.points), -1)
    rad = rad_all[cl.cells[cell_id], 0]
    if len(rad) >= 2 and not np.all(rad[:-1] + rad[1:] > 0.0):
        k = int(np.flatnonzero(~(rad[:-1] + rad[1:] > 0.0))[0])
        raise ValueError(f"cell {cell_id}: segment {k} has a mean radius <= 0; the touching-sphere walk "
                         "cannot advance along it (vmtk crawls 1e-5 of the segment per sphere)")
    out = [pts[0]]
    if len(pts) >= 2:                  # vmtk reads point 1 of a one-point cell here (UB)
        sub_id, pcoord = 0, 0.0
        while True:
            t_sub, t_pc = find_touching_sphere_center(cl, radius, cell_id, sub_id, pcoord, forward=False,
                                                      vmtk_steps=vmtk_steps)
            if t_sub == -1:
                break
            q0, q1 = pts[t_sub], pts[t_sub + 1]
            out.append(np.array([(1.0 - t_pc) * q0[k] + t_pc * q1[k] for k in range(3)]))
            sub_id, pcoord = t_sub, t_pc
    out.append(pts[-1])
    out = np.array(out)
    return round_float32(out) if vmtk_float32 else out


def subsample_line(points: np.ndarray, min_spacing: float, vmtk_float32: bool = False) -> np.ndarray:
    """vtkvmtkCenterlineBranchGeometry::SubsampleLine (not reachable from the vmtk script)."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    spacing = 0.0
    point0 = pts[0]
    out = [pts[0]]
    for j in range(1, n - 1):
        point1 = pts[j]
        spacing += math.sqrt(distance2(point0, point1))
        if spacing < min_spacing:
            continue
        out.append(point1)
        point0 = pts[j]
        spacing = 0.0
    point1 = pts[n - 1]
    spacing = math.sqrt(distance2(point0, point1))
    if spacing < min_spacing:
        out[-1] = point1
    else:
        out.append(point1)
    out = np.array(out)
    return round_float32(out) if vmtk_float32 else out


def branch_geometry(cl: Centerlines, radius: str = RADIUS, group_ids: str = "GroupIds",
                    blanking: str = "Blanking", line_smoothing: bool = False, iterations: int = 100,
                    factor: float = 0.1, line_subsampling: bool = True, sphere_subsampling: bool = True,
                    min_subsampling_spacing: float = 0.1, vmtk_float32: bool = False,
                    vmtk_steps: bool = False, vmtk_discard_smoothing: bool = False, length: str = "Length",
                    curvature: str = "Curvature", torsion: str = "Torsion",
                    tortuosity: str = "Tortuosity") -> Centerlines:
    """vtkvmtkCenterlineBranchGeometry: one vertex per non-blanked group with its geometry.

    Defaults are the vmtk script's (smoothing 100 iterations, factor 0.1; the C++ class defaults to
    10 and 0.01; subsampling on, by touching spheres, which the script cannot change).
    """
    gid = cl.cell_data[group_ids]
    rad = cl.point_data[radius]
    groups = get_non_blanked_groups_id_list(cl, group_ids, blanking)
    cells_of = {g: [i for i in range(cl.n_cells) if int(gid[i]) == g] for g in groups}

    def line_points(i: int) -> np.ndarray:
        """The points ComputeGroupCurvature / ComputeGroupTorsion measure for cell i."""
        raw = cl.cell_points(i)
        p = round_float32(raw) if vmtk_float32 else raw
        if line_smoothing:
            p = smooth_line(p, iterations, factor, vmtk_float32)
        if line_subsampling:
            if sphere_subsampling:
                if vmtk_discard_smoothing or not line_smoothing:
                    p = sphere_subsample_line(cl, i, radius, vmtk_float32, vmtk_steps)
                else:                  # walk the spheres along the smoothed points, same radii
                    one = Centerlines.from_lines([p], [rad[cl.cells[i]]], radius)
                    p = sphere_subsample_line(one, 0, radius, vmtk_float32, vmtk_steps)
            else:
                p = subsample_line(p, min_subsampling_spacing, vmtk_float32)
        return p

    out_length, out_curvature, out_torsion, out_tortuosity = [], [], [], []
    for g in groups:
        cells = cells_of[g]
        # ComputeGroupLength
        group_length, weight_sum = 0.0, 0.0
        for i in cells:
            group_length += polyline_length(cl.cell_points(i))
            weight_sum += 1.0
        group_length /= weight_sum
        # ComputeGroupCurvature / ComputeGroupTorsion (the same per-cell points for both)
        group_curvature, group_torsion, weight_sum = 0.0, 0.0, 0.0
        for i in cells:
            p = line_points(i)
            group_curvature += line_curvature(p)[1]
            group_torsion += line_torsion(p)[1]
            weight_sum += 1.0
        group_curvature /= weight_sum
        group_torsion /= weight_sum
        # ComputeGroupTortuosity
        first_avg, last_avg = np.zeros(3), np.zeros(3)
        first_w, last_w = 0.0, 0.0
        for i in cells:
            ids = cl.cells[i]
            first, last = cl.points[ids[0]], cl.points[ids[-1]]
            rf, rl = float(rad[ids[0]]), float(rad[ids[-1]])
            for k in range(3):
                first_avg[k] += rf * rf * first[k]
                last_avg[k] += rl * rl * last[k]
            first_w += rf * rf
            last_w += rl * rl
        with np.errstate(divide="ignore", invalid="ignore"):     # zero end radii: 0 / 0, NaN as in vmtk
            first_avg = first_avg / first_w
            last_avg = last_avg / last_w
            dd = first_avg - last_avg
            tort = np.float64(group_length) / np.sqrt(dd[0] * dd[0] + dd[1] * dd[1] + dd[2] * dd[2]) - 1.0
        out_length.append(group_length)
        out_curvature.append(group_curvature)
        out_torsion.append(group_torsion)
        out_tortuosity.append(float(tort))

    ng = len(groups)
    return Centerlines(np.zeros((ng, 3)), [np.array([k]) for k in range(ng)],
                       {length: np.array(out_length, dtype=np.float64),
                        curvature: np.array(out_curvature, dtype=np.float64),
                        torsion: np.array(out_torsion, dtype=np.float64),
                        tortuosity: np.array(out_tortuosity, dtype=np.float64),
                        group_ids: np.array(groups, dtype=np.float64)})
