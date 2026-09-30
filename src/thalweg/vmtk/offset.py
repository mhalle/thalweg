"""vtkvmtkCenterlineReferenceSystemAttributesOffset in numpy: abscissas and normals relative to a frame.

Ports ``vtkvmtkCenterlineReferenceSystemAttributesOffset.cxx`` (vmtk master 66b068a; the class
behind the ``vmtkCenterlineOffsetAttributes`` script) line for line, with the pieces of VTK and
vmtk it calls: ``vtkvmtkPolyBallLine::EvaluateFunction`` (no radius; :mod:`.polyball`), ``vtkvmtkMath::
AngleBetweenNormals``, and ``vtkTransform::RotateWXYZ`` + ``TransformNormal`` (quaternion ->
matrix, then the inverse transpose by adjoint / determinant, then a normalize; :mod:`._vtk`).
Public: :func:`offset_attributes`.

For each cell of the reference group (in cell order): the closest point of that cell to the
reference origin gives an abscissa and a parallel-transport normal; every point of the cell's
centerline gets its abscissa shifted so that closest point is 0, and its normal rotated about the
local tangent by the angle that takes that normal onto the frame normal (projected normal to the
tangent).

``vmtk_interp`` (default False): the closest point's abscissa and normal are read with
``vtkvmtkCenterlineUtilities::InterpolateTuple``, which returns the segment END value (see
:mod:`.utilities`). False interpolates correctly; True reproduces vmtk. The error is up to one
segment of abscissa and one segment's turn of normal, for every point of the covered centerlines:
over every frame of the oracles, offset abscissas move by up to 0.28 mm (phantom, 3 frames) /
0.29 mm (case, 44 frames) and offset normals by up to 0.094 / 12.4 degrees (case: median 0.023,
90th percentile 1.08 degrees over the covered points).

Behaviors kept from vmtk, not defects: ``reference_group_id=-1`` takes the first frame's group; a
group with no frame logs vmtk's "Invalid ReferenceGroupId" error and returns an unchanged copy
(vmtk has already deep-copied its input then); points on centerlines that never cross the reference
group are left uninitialized by vmtk (``SetNumberOfTuples`` without a fill) - here they are NaN.

Validation (tests/test_vmtk_offset.py): every oracle's ``offset`` / ``offset_keep`` stage is vmtk run
with ReferenceGroupId -1 - the first frame's group, a bifurcation (``root_group`` in oracle.json) -
so the valid path is what is compared, against every covered point (with ``vmtk_interp=True``,
within 1e-9; in practice abscissas exact and normals within 5e-16): phantom, case, and the small
fixtures. The invalid-group path is tested on its own (one warning, the input returned unchanged).
A fixture without a bifurcation has no frame; there vmtk logs "ReferenceSystems empty" and outputs
nothing, and the port raises ``ValueError``.
"""
from __future__ import annotations

import warnings

import numpy as np

from ._vtk import VTK_PI, angle_between_normals, cross, dot, normalize, rotate_normal
from .centerlines import ABSCISSAS, CENTERLINE_IDS, GROUP_IDS, NORMALS, Centerlines, ReferenceSystems
from .polyball import evaluate_function, tube_segments
from .utilities import get_centerline_cell_ids, get_group_cell_ids, interpolate_tuple


def offset_attributes(cl: Centerlines, reference_systems: ReferenceSystems, reference_group_id: int = -1,
                      replace_attributes: bool = True, abscissas: str = ABSCISSAS, normals: str = NORMALS,
                      group_ids: str = GROUP_IDS, centerline_ids: str = CENTERLINE_IDS,
                      reference_systems_normal: str = "Normal", reference_systems_group_ids: str = GROUP_IDS,
                      offset_abscissas: str = "OffsetAbscissas", offset_normals: str = "OffsetNormals",
                      vmtk_interp: bool = False) -> Centerlines:
    """Abscissas and normals offset to one bifurcation frame (vmtkCenterlineOffsetAttributes).

    With ``replace_attributes`` the results replace ``abscissas`` / ``normals`` in place (same array
    position); otherwise they are added as ``offset_abscissas`` / ``offset_normals``. Returns a new
    :class:`Centerlines`; the input is not modified.
    """
    if reference_systems.points.shape[0] == 0:
        raise ValueError("ReferenceSystems empty")
    out = cl.copy()
    abs_in = np.asarray(cl.point_data[abscissas], dtype=np.float64).reshape(-1)
    nrm_in = np.asarray(cl.point_data[normals], dtype=np.float64).reshape(-1, 3)
    cids = cl.cell_data[centerline_ids]
    rs_groups = reference_systems.point_data[reference_systems_group_ids]
    if reference_group_id == -1:
        reference_group_id = int(rs_groups[0])
    rs_point = -1
    for i in range(len(rs_groups)):
        if int(rs_groups[i]) == reference_group_id:
            rs_point = i
            break
    if rs_point == -1:
        warnings.warn("vtkvmtkCenterlineReferenceSystemAttributesOffset: Invalid ReferenceGroupId "
                      f"({reference_group_id}); returning the input unchanged, as vmtk does", stacklevel=2)
        return out
    origin = [float(v) for v in reference_systems.points[rs_point]]
    rs_normals = np.asarray(reference_systems.point_data[reference_systems_normal])
    rs_normal = [float(v) for v in rs_normals[rs_point]]

    n = cl.n_points
    off_abs = np.full(n, np.nan)
    off_nrm = np.full((n, 3), np.nan)
    pts_all = cl.points
    for cell_id in get_group_cell_ids(cl, reference_group_id, group_ids):
        centerline_id = int(cids[cell_id])
        # vtkvmtkPolyBallLine::EvaluateFunction on this cell alone, UseRadiusInformation off
        state = evaluate_function(origin, tube_segments(cl, None, [cell_id]))
        sub, pcoord = int(state.sub[0]), float(state.pcoord[0])
        if sub == -1:
            raise ValueError(f"cell {cell_id} of the reference group has no segment "
                             "(vmtk reads cell -1 here)")
        abscissa = float(interpolate_tuple(cl, abscissas, cell_id, sub, pcoord, vmtk_interp)[0])
        abscissa_offset = -abscissa
        normal = [float(v) for v in interpolate_tuple(cl, normals, cell_id, sub, pcoord, vmtk_interp)]
        normalize(normal)
        cp = pts_all[cl.cells[cell_id]]
        p0, p1 = cp[sub], cp[sub + 1]
        tangent = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]]
        nd = dot(tangent, normal)
        tangent = [tangent[0] - nd * normal[0], tangent[1] - nd * normal[1], tangent[2] - nd * normal[2]]
        normalize(tangent)
        rd = dot(tangent, rs_normal)
        proj = [rs_normal[0] - rd * tangent[0], rs_normal[1] - rd * tangent[1],
                rs_normal[2] - rd * tangent[2]]
        normalize(proj)
        angle = angle_between_normals(normal, proj)
        if dot(tangent, cross(proj, normal)) < 0.0:
            angle *= -1.0
        angle_offset_deg = -angle / VTK_PI * 180.0

        for branch_cell in get_centerline_cell_ids(cl, centerline_id, centerline_ids):
            ids = cl.cells[branch_cell]
            bp = pts_all[ids]
            npts = len(ids)
            for k in range(npts):
                pid = int(ids[k])
                off_abs[pid] = abs_in[pid] + abscissa_offset
                bn = [float(v) for v in nrm_in[pid]]
                p = bp[k]
                bt = [0.0, 0.0, 0.0]
                if k > 0:
                    q = bp[k - 1]
                    bt[0] += p[0] - q[0]
                    bt[1] += p[1] - q[1]
                    bt[2] += p[2] - q[2]
                if k < npts - 1:
                    q = bp[k + 1]
                    bt[0] += q[0] - p[0]
                    bt[1] += q[1] - p[1]
                    bt[2] += q[2] - p[2]
                bd = dot(bt, bn)
                bt = [bt[0] - bd * bn[0], bt[1] - bd * bn[1], bt[2] - bd * bn[2]]
                normalize(bt)
                off_nrm[pid] = rotate_normal(angle_offset_deg, bt, bn)

    if replace_attributes:
        out.point_data[abscissas] = off_abs
        out.point_data[normals] = off_nrm
    else:
        out.point_data[offset_abscissas] = off_abs
        out.point_data[offset_normals] = off_nrm
    return out
