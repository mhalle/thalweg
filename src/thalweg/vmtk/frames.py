"""vtkvmtkCenterlineBifurcationReferenceSystems in numpy: one reference system per bifurcation.

Ports ``vtkvmtkCenterlineBifurcationReferenceSystems.cxx`` (vmtk master 66b068a; the class behind
the ``vmtkBifurcationReferenceSystems`` script) line for line, with the VTK and vmtk helpers it
calls (``vtkTriangle::ComputeNormal``, ``vtkMath::Normalize``/``Perpendiculars``,
``vtkvmtkMath::Cotangent``). Public: :func:`bifurcation_reference_systems`.

For every blanked group (a bifurcation), in first-seen cell order: the group's unique cells give
(start, end) point pairs; the origin is their squared-radius-weighted mean; the normal is the
consistently oriented sum of the cotangent-weighted normals of the polygons spanned by each pair of
cells; the up normal is the squared-downstream-radius-weighted sum of the cells' start-to-end
vectors projected onto the bifurcation plane. A group with a single unique cell gets the cell
direction as up normal and ``vtkMath::Perpendiculars`` of it as normal.

**vmtk_float32** (default False; True reproduces vmtk): vmtk collects each cell's end points in a
default ``vtkPoints`` (VTK_FLOAT) and stores the origins in another, so its end points and origins
are float32-rounded. On vmtk's own input (float32 points from the branch extractor) the first
rounding is a no-op and only the origins' rounding shows: the default origins differ from vmtk's by
up to 7.9e-7 mm (phantom) and 3.8e-6 mm (case oracle), the normals are bit-identical either way.
The normals are float64, the group ids int32.
"""
from __future__ import annotations

import math

import numpy as np

from ._vtk import (VTK_VMTK_DOUBLE_TOL, VTK_VMTK_LARGE_DOUBLE, distance2, div, dot, normalize,
                   perpendiculars, round_float32)
from .centerlines import GROUP_IDS, RADIUS, Centerlines, ReferenceSystems
from .utilities import get_blanked_groups_id_list, get_group_unique_cell_ids


def _triangle_normal(v1, v2, v3) -> list:
    """vtkTriangle::ComputeNormal."""
    ax, ay, az = v3[0] - v2[0], v3[1] - v2[1], v3[2] - v2[2]
    bx, by, bz = v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]
    n = [ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx]
    length = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
    if length != 0.0:
        n[0] /= length
        n[1] /= length
        n[2] /= length
    return n


def _cotangent(p0, p1, p2) -> float:
    """vtkvmtkMath::Cotangent: cotangent of the angle at ``p1``."""
    n0 = ((p0[0] - p1[0]) * (p0[0] - p1[0]) + (p0[1] - p1[1]) * (p0[1] - p1[1])
          + (p0[2] - p1[2]) * (p0[2] - p1[2]))
    n1 = ((p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1])
          + (p2[2] - p1[2]) * (p2[2] - p1[2]))
    dot = ((p0[0] - p1[0]) * (p2[0] - p1[0]) + (p0[1] - p1[1]) * (p2[1] - p1[1])
           + (p0[2] - p1[2]) * (p2[2] - p1[2]))
    arg = n0 * n1 - dot * dot
    cross_norm = math.sqrt(arg) if arg >= 0.0 else math.nan
    if cross_norm < VTK_VMTK_DOUBLE_TOL:
        return 0.0 if abs(dot) < VTK_VMTK_DOUBLE_TOL else VTK_VMTK_LARGE_DOUBLE
    if abs(dot) < VTK_VMTK_DOUBLE_TOL:
        return 0.0
    return dot / cross_norm


def _group_reference_system(cl: Centerlines, group_id: int, radius: str, group_ids: str, vmtk_float32: bool):
    """ComputeGroupReferenceSystem: (origin, normal, up_normal) or None when the group has no cell."""
    rad = cl.point_data[radius]
    rad = rad if rad.ndim == 1 else rad[:, 0]
    bif_points: list[list[float]] = []            # a default vtkPoints: float32 in vmtk
    bif_radii: list[float] = []
    for cell_id in get_group_unique_cell_ids(cl, group_id, group_ids):
        ids = cl.cells[cell_id]                   # every Centerlines cell is a (poly)line, as vmtk checks
        p0, p1 = cl.points[ids[0]], cl.points[ids[-1]]
        if vmtk_float32:
            p0, p1 = round_float32(p0), round_float32(p1)
        p0, p1 = [float(x) for x in p0], [float(x) for x in p1]
        skip = False
        for j in range(0, len(bif_points), 2):
            if (distance2(p0, bif_points[j]) < VTK_VMTK_DOUBLE_TOL
                    and distance2(p1, bif_points[j + 1]) < VTK_VMTK_DOUBLE_TOL):
                skip = True
                break
        if skip:
            continue
        bif_points.append(p0)
        bif_radii.append(float(rad[ids[0]]))
        bif_points.append(p1)
        bif_radii.append(float(rad[ids[-1]]))

    n_bif = len(bif_points)
    origin = [0.0, 0.0, 0.0]
    weight_sum = 0.0
    for i in range(n_bif):
        point = bif_points[i]
        weight = bif_radii[i] * bif_radii[i]
        origin[0] += weight * point[0]
        origin[1] += weight * point[1]
        origin[2] += weight * point[2]
        weight_sum += weight
    if n_bif:                                     # all radii 0: 0 / 0, NaN as in vmtk
        origin = [div(origin[0], weight_sum), div(origin[1], weight_sum), div(origin[2], weight_sum)]
    else:                                         # vmtk divides 0 by 0 here; nothing is emitted below
        origin = [math.nan, math.nan, math.nan]

    normals: list[list[float]] = []
    for i in range(0, n_bif, 2):
        for j in range(i + 2, n_bif, 2):
            if distance2(bif_points[i], bif_points[j]) < VTK_VMTK_DOUBLE_TOL:
                verts = [bif_points[i], bif_points[i + 1], bif_points[j + 1]]
            else:
                verts = [bif_points[i], bif_points[i + 1], bif_points[j + 1], bif_points[j]]
            npoly = len(verts)
            poly_normal = [0.0, 0.0, 0.0]
            for n in range(npoly):
                vp0 = verts[(n - 1 + npoly) % npoly]
                vp1 = verts[n]
                vp2 = verts[(n + 1) % npoly]
                vn = _triangle_normal(vp0, vp1, vp2)
                cot0 = _cotangent(vp0, vp1, origin)
                cot1 = _cotangent(origin, vp1, vp2)
                weight = div(cot0 + cot1, distance2(origin, vp1))
                for k in range(3):
                    poly_normal[k] += weight * vn[k]
            normalize(poly_normal)
            normals.append(poly_normal)

    if not normals:
        if n_bif > 1:
            p1, p0 = bif_points[1], bif_points[0]
            up = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]]
            normalize(up)
            normal, _ = perpendiculars(up)
            return origin, normal, up
        return None

    positive = normals[0]
    for i in range(1, len(normals)):
        o = normals[i]
        if dot(positive, o) < 0.0:
            normals[i] = [o[0] * -1.0, o[1] * -1.0, o[2] * -1.0]
    normal = [0.0, 0.0, 0.0]
    for o in normals:
        for k in range(3):
            normal[k] += o[k]
    normalize(normal)

    up = [0.0, 0.0, 0.0]
    for i in range(0, n_bif, 2):
        p1, p0 = bif_points[i + 1], bif_points[i]
        r1 = bif_radii[i + 1]
        weight = r1 * r1                          # squared radius of the downstream end
        vec = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]]
        vd = dot(vec, normal)
        proj = [vec[0] - vd * normal[0], vec[1] - vd * normal[1], vec[2] - vd * normal[2]]
        normalize(proj)
        for k in range(3):
            up[k] += proj[k] * weight
    normalize(up)
    return origin, normal, up


def bifurcation_reference_systems(cl: Centerlines, radius: str = RADIUS, group_ids: str = GROUP_IDS,
                                  blanking: str = "Blanking", normal: str = "Normal",
                                  up_normal: str = "UpNormal",
                                  vmtk_float32: bool = False) -> ReferenceSystems:
    """One reference system per blanked group of split centerlines (vmtkBifurcationReferenceSystems).

    Output points are the origins, with point arrays ``normal`` and ``up_normal`` (float64, (M, 3))
    and ``group_ids`` (int32), in first-seen blanked-group order. ``vmtk_float32``: see the module
    docstring.
    """
    origins, normals, ups, gids = [], [], [], []
    for g in get_blanked_groups_id_list(cl, group_ids, blanking):
        rs = _group_reference_system(cl, g, radius, group_ids, vmtk_float32)
        if rs is None:
            continue
        o, n, u = rs
        origins.append(o)
        normals.append(n)
        ups.append(u)
        gids.append(g)
    pts = np.array(origins, dtype=np.float64).reshape(-1, 3)
    if vmtk_float32:                              # the output's default vtkPoints
        pts = round_float32(pts)
    return ReferenceSystems(pts, {normal: np.array(normals, dtype=np.float64).reshape(-1, 3),
                                  up_normal: np.array(ups, dtype=np.float64).reshape(-1, 3),
                                  group_ids: np.array(gids, dtype=np.int32)})
