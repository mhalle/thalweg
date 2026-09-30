"""vtkvmtkCenterlineBifurcationVectors (script ``vmtkbifurcationvectors``) in numpy.

Ports ``vtkvmtkCenterlineBifurcationVectors.cxx`` (vmtk master 66b068a) and
``vtkvmtkReferenceSystemUtilities::GetReferenceSystemPointId``, on the shared ports of the query it
leans on, ``vtkvmtkCenterlineSphereDistance::FindTouchingSphereCenter`` (:mod:`.sphere_distance`,
also used by the branch extractor and vtkvmtkCenterlineBranchGeometry), and of
``vtkvmtkMath::AngleBetweenNormals`` (:mod:`._vtk`).

Public: :func:`bifurcation_vectors` (the filter); :func:`find_touching_sphere_center` is
re-exported from :mod:`.sphere_distance`.

**What the filter computes.** Input: split centerlines (vmtkbranchextractor: GroupIds, CenterlineIds,
TractIds, Blanking cell arrays, a radius point array) and the bifurcation reference systems
(vmtkbifurcationreferencesystems: one origin per blanked group with ``Normal``, ``UpNormal``,
``GroupIds``). For every blanked (bifurcation) group, in first-seen cell order, and for every group
adjacent to it along some centerline - upstream groups first, then downstream, each in first-seen
order - one output point carrying:

- ``BifurcationVectors``: over the adjacent group's unique cells, the radius^2-weighted mean of the
  cell's end at the bifurcation (last point of an upstream branch, first point of a downstream one)
  and the radius^2-weighted mean of its *touching point* (the point where the branch centerline
  leaves the maximal inscribed sphere of that end point, see :func:`find_touching_sphere_center`;
  the touching radius comes from InterpolateTuple). The vector joins the two means in the flow
  direction: touching -> end for the parent (pointing INTO the bifurcation), end -> touching for a
  daughter (pointing AWAY from it). Its length is about one radius; ``normalize=True`` makes it unit.
- the output point: the vector's tail (touching mean for upstream, end mean for downstream);
- ``OutOfPlaneBifurcationVectors`` = (v . N) N and ``InPlaneBifurcationVectors`` = v - (v . N) N,
  the split of v along the reference system's plane normal N;
- ``InPlaneBifurcationVectorAngles`` / ``OutOfPlaneBifurcationVectorAngles`` (defined below);
- ``BifurcationVectorsOrientation`` 0 = upstream (parent), 1 = downstream (daughter); ``GroupIds``
  the adjacent group; ``BifurcationGroupIds`` the bifurcation group. Integer arrays are int32
  (vtkIntArray); the output cells are one-point vertices.

The output points are the vectors' tails; every vector and angle is double, as in vmtk.

**The angles, precisely.** The reference system of a bifurcation
(vtkvmtkCenterlineBifurcationReferenceSystems) has N = ``Normal``, the unit normal of the bifurcation
plane (a cotangent-weighted mean of the normals of the polygons spanned by pairs of the bifurcation
group's centerline pieces; its SIGN is arbitrary - it follows the winding of the first pair of pieces,
so it may point either way), and U = ``UpNormal``, the unit in-plane direction of the bifurcation
group's pieces from their upstream to their downstream ends (each piece's end-minus-start chord
projected on the plane, normalized, weighted by the downstream end radius^2) - i.e. the local
"straight-on" flow direction through the bifurcation. With v the bifurcation vector and p its
in-plane component, both normalized:

- in-plane angle = the unsigned angle between p and U, 2 atan2(|p - U|, |p + U|) in [0, pi], made
  NEGATIVE when (p x U) . N < 0, i.e. when p is reached from U by a counterclockwise (right-handed)
  turn about N. Positive: p lies clockwise of U seen from the +N side. So it is a signed angle in
  (-pi, pi] measured in the bifurcation plane FROM U TO p with clockwise-about-N positive; its sign
  flips with N's arbitrary orientation. Because every vector points along the flow, the parent's
  angle is near 0 and a daughter's is its deviation from going straight on - NOT the angle between
  parent and daughter as vessels leaving the branch point (that is pi minus the difference).
- out-of-plane angle = pi/2 - angle(v, N) = the elevation of v above the bifurcation plane, in
  [-pi/2, pi/2], positive on the +N side (again sign follows N).

Both are radians. A zero in-plane component (v parallel to N) gives angle(0, U) = pi/2, not an error.

**Defect flags.**

- ``vmtk_interp``: the touching-point radius (the weight of the touching point) goes through
  ``vtkvmtkCenterlineUtilities::InterpolateTuple``, which returns the segment END value
  (see :mod:`thalweg.vmtk.utilities`). False interpolates; True reproduces vmtk. The touching
  radius only weights the touching point, so the effect is second order: tails and vectors move by
  up to 1.4e-4 / 3.4e-3 mm, angles by up to 2.2e-6 / 7.8e-4 rad (phantom / case oracle).
- ``vmtk_fallback``: when no touching point is found on the adjacent cell (the whole cell lies
  inside the end point's sphere), vmtk falls back to the cell's LAST point whatever the orientation.
  That is the far end of a downstream branch, but for an upstream branch it is the end point
  itself, so the cell adds a zero vector with full weight (a parent that short gives v = 0 and
  meaningless angles: in-plane pi/2, out-of-plane 0). The default follows the definition instead:
  the touching-sphere walk continues upstream along the SAME centerline, through the preceding
  tracts in tract-id order (blanked or not), until it leaves the sphere; the touching point is
  then on a preceding tract, one inscribed sphere upstream of the end point. If the walk reaches
  the centerline's start still inside the sphere there is no touching point: that row's vector,
  its components, both angles and its tail (the output point) are NaN. A downstream branch shorter
  than its sphere keeps vmtk's fallback (its far end) in both modes: downstream the continuation
  depends on which centerline the group's cell belongs to, so "one sphere downstream" is not a
  property of the branch. True reproduces vmtk. The case oracle hits it for 6 of its 138 vectors
  (6 parent groups, 12 unique cells of 0.07-1.90 mm and 2-8 points, the longest cell of each group
  0.34-1.90 mm; end radius 1.67-1.98 mm): vmtk reports v = 0 there, the default |v| = 1.61-2.09 mm
  (about one radius, as for every other parent), in-plane angles -0.08..0.33 rad instead of pi/2,
  out-of-plane -0.40..0.02 rad instead of 0. The phantom never hits it; the ``short_trunk``
  fixture (a 1.5 mm trunk of radius 2 from the centerlines' start) gives NaN.
- ``vmtk_steps``: FindTouchingSphereCenter's in-segment linear search sizes its steps as 1e-6 of
  the mean radius but counts them from the SQUARED segment length (``Distance2BetweenPoints``),
  capped at 1e5, so the pcoord resolution is 1/ceil(min(L^2 / (1e-6 r), 1e5)) - unit-dependent
  (see :mod:`.sphere_distance`). False counts from the length; True reproduces vmtk. The effect
  is the search's quantization: tails move by up to 1.9e-6 / 7.6e-6 mm and the angles by up to
  5e-8 / 7e-7 rad (phantom / case oracle).
- ``vmtk_float32``: vmtk stores the output points (the tails) in a default ``vtkPoints``
  (VTK_FLOAT). False keeps them double; True rounds as vmtk does. The default tails differ by up
  to 7.4e-7 / 3.8e-6 mm (phantom / case oracle); every vector and angle is unchanged.

Degenerate inputs where the C++ reads out of bounds (an adjacent cell with fewer than 2 points, a
bifurcation group with no reference system) raise ``ValueError`` here.
"""
from __future__ import annotations

import math

import numpy as np

from ._vtk import angle_between_normals, cross, dot, normalized, round_float32
from .centerlines import RADIUS, Centerlines, ReferenceSystems
from .sphere_distance import find_touching_sphere_center
from .utilities import (find_adjacent_centerline_group_ids, get_blanked_groups_id_list,
                        get_centerline_cell_ids, get_group_unique_cell_ids, interpolate_point,
                        interpolate_tuple)

__all__ = ["bifurcation_vectors", "find_touching_sphere_center", "UPSTREAM", "DOWNSTREAM"]

UPSTREAM = 0      # VTK_VMTK_UPSTREAM_ORIENTATION
DOWNSTREAM = 1    # VTK_VMTK_DOWNSTREAM_ORIENTATION


# -- vtkvmtkCenterlineBifurcationVectors ------------------------------------------------------

def _reference_system_point_id(rs: ReferenceSystems, group_ids: str, group_id: int) -> int:
    g = rs.point_data[group_ids]
    for i in range(len(rs.points)):
        if int(g[i]) == group_id:
            return i
    return -1


def _upstream_line(cl: Centerlines, cell_id: int, radius_name: str, centerline_ids: str,
                   tract_ids: str) -> Centerlines:
    """One cell: the tracts of ``cell_id``'s centerline before it (tract-id order), then the cell
    itself, joined where consecutive tracts share their end coordinates; radius carried along."""
    cid = int(cl.cell_data[centerline_ids][cell_id])
    tid = int(cl.cell_data[tract_ids][cell_id])
    tracts = cl.cell_data[tract_ids]
    chain = [c for c in get_centerline_cell_ids(cl, cid, centerline_ids, tract_ids) if int(tracts[c]) < tid]
    ids: list[int] = []
    for c in chain + [cell_id]:
        cell = [int(i) for i in cl.cells[c]]
        if ids and cell and np.array_equal(cl.points[ids[-1]], cl.points[cell[0]]):
            cell = cell[1:]
        ids.extend(cell)
    radius = np.asarray(cl.point_data[radius_name], dtype=np.float64).reshape(len(cl.points), -1)[:, 0]
    return Centerlines.from_lines([cl.points[ids]], [radius[ids]], radius_name)


def _compute_bifurcation_vectors(cl, bifurcation_group_id, radius_name, group_ids, centerline_ids, tract_ids,
                                 normalize, vmtk_interp, vmtk_fallback, vmtk_steps):
    radius = np.asarray(cl.point_data[radius_name], dtype=np.float64).reshape(len(cl.points), -1)[:, 0]
    up, down = find_adjacent_centerline_group_ids(cl, bifurcation_group_id, group_ids, centerline_ids,
                                                  tract_ids)
    vec_groups = list(up) + list(down)
    vec_orient = [UPSTREAM] * len(up) + [DOWNSTREAM] * len(down)

    vectors, origins = [], []
    for group, orientation in zip(vec_groups, vec_orient):
        avg_last = [0.0, 0.0, 0.0]
        avg_touch = [0.0, 0.0, 0.0]
        last_w = 0.0
        touch_w = 0.0
        for cell_id in get_group_unique_cell_ids(cl, group, group_ids):
            ids = cl.cells[cell_id]
            n = len(ids)
            if n < 2:
                raise ValueError(f"cell {cell_id} (group {group}) has {n} point(s)")
            if orientation == UPSTREAM:
                sub_id, pc, forward = n - 2, 1.0, True
                last = cl.points[ids[n - 1]]
                last_r = float(radius[ids[n - 1]])
            else:
                sub_id, pc, forward = 0, 0.0, False
                last = cl.points[ids[0]]
                last_r = float(radius[ids[0]])
            t_sub, t_pc = find_touching_sphere_center(cl, radius_name, cell_id, sub_id, pc, forward,
                                                      vmtk_steps)
            walk, walk_cell = cl, cell_id
            if t_sub == -1:
                if vmtk_fallback or orientation == DOWNSTREAM:
                    t_sub, t_pc = n - 2, 1.0
                else:                                   # continue upstream along the centerline
                    walk = _upstream_line(cl, cell_id, radius_name, centerline_ids, tract_ids)
                    walk_cell = 0
                    m = len(walk.points)
                    t_sub, t_pc = find_touching_sphere_center(walk, radius_name, 0, m - 2, 1.0, True,
                                                              vmtk_steps)

            w = last_r * last_r
            for k in range(3):
                avg_last[k] += w * last[k]
            last_w += w
            if t_sub == -1:                             # the whole centerline is inside the sphere
                touch, touch_r = np.full(3, np.nan), math.nan
            else:
                touch = interpolate_point(walk, walk_cell, t_sub, t_pc)
                touch_r = float(interpolate_tuple(walk, radius_name, walk_cell, t_sub, t_pc, vmtk_interp)[0])
            w = touch_r * touch_r
            for k in range(3):
                avg_touch[k] += w * touch[k]
            touch_w += w
        with np.errstate(divide="ignore", invalid="ignore"):
            avg_last = [float(np.float64(a) / np.float64(last_w)) for a in avg_last]
            avg_touch = [float(np.float64(a) / np.float64(touch_w)) for a in avg_touch]

        if orientation == UPSTREAM:
            v = [avg_last[k] - avg_touch[k] for k in range(3)]
            origin = avg_touch
        else:
            v = [avg_touch[k] - avg_last[k] for k in range(3)]
            origin = avg_last
        if normalize:
            v = normalized(v)
        vectors.append(v)
        origins.append(origin)
    return vec_groups, vec_orient, vectors, origins


def bifurcation_vectors(cl: Centerlines, reference_systems: ReferenceSystems, normalize: bool = False,
                        vmtk_interp: bool = False, vmtk_fallback: bool = False, vmtk_steps: bool = False,
                        vmtk_float32: bool = False, radius_name: str = RADIUS,
                        group_ids: str = "GroupIds", centerline_ids: str = "CenterlineIds",
                        tract_ids: str = "TractIds", blanking: str = "Blanking",
                        reference_group_ids: str = "GroupIds", normal: str = "Normal",
                        up_normal: str = "UpNormal") -> Centerlines:
    """vmtkbifurcationvectors: one vertex per (bifurcation, adjacent branch), see the module docstring.

    ``cl`` is the split centerlines, ``reference_systems`` the bifurcation reference systems. The
    output's points are the vectors' tails, its cells one-point vertices, and its point arrays
    ``BifurcationVectors``, ``InPlaneBifurcationVectors``, ``OutOfPlaneBifurcationVectors``,
    ``InPlaneBifurcationVectorAngles``, ``OutOfPlaneBifurcationVectorAngles`` (radians),
    ``BifurcationVectorsOrientation`` (0 up, 1 down), ``<group_ids>`` and ``BifurcationGroupIds``.
    A parent that lies inside its end point's sphere all the way back to its centerline's start has
    no touching point: its row (tail, vectors, angles) is NaN (see ``vmtk_fallback``).
    """
    rs_normal = np.asarray(reference_systems.point_data[normal], dtype=np.float64).reshape(-1, 3)
    rs_up = np.asarray(reference_systems.point_data[up_normal], dtype=np.float64).reshape(-1, 3)

    out_pts, out_v, out_in, out_out, out_ain, out_aout = [], [], [], [], [], []
    out_orient, out_group, out_bif = [], [], []
    for bif in get_blanked_groups_id_list(cl, group_ids, blanking):
        groups, orients, vectors, origins = _compute_bifurcation_vectors(
            cl, bif, radius_name, group_ids, centerline_ids, tract_ids, normalize, vmtk_interp, vmtk_fallback,
            vmtk_steps)
        if not groups:
            continue
        rid = _reference_system_point_id(reference_systems, reference_group_ids, bif)
        if rid == -1:
            raise ValueError(f"no reference system for bifurcation group {bif}")
        nrm = [float(x) for x in rs_normal[rid]]
        upn = [float(x) for x in rs_up[rid]]
        for group, orient, v, origin in zip(groups, orients, vectors, origins):
            # ComputeBifurcationVectorComponents
            vd = dot(v, nrm)
            oop = [vd * nrm[0], vd * nrm[1], vd * nrm[2]]
            inp = [v[0] - oop[0], v[1] - oop[1], v[2] - oop[2]]
            # ComputeBifurcationVectorAngles
            vn = normalized(v)
            pn = normalized(inp)
            a_in = angle_between_normals(pn, upn)
            if dot(cross(pn, upn), nrm) < 0.0:
                a_in *= -1.0
            a_out = math.pi / 2.0 - angle_between_normals(vn, nrm)

            out_pts.append(origin)
            out_v.append(v)
            out_in.append(inp)
            out_out.append(oop)
            out_ain.append(a_in)
            out_aout.append(a_out)
            out_orient.append(orient)
            out_group.append(group)
            out_bif.append(bif)

    m = len(out_pts)

    def vec3(a):
        return np.array(a, dtype=np.float64).reshape(m, 3)

    pd = {
        "BifurcationVectors": vec3(out_v),
        "InPlaneBifurcationVectors": vec3(out_in),
        "OutOfPlaneBifurcationVectors": vec3(out_out),
        "InPlaneBifurcationVectorAngles": np.array(out_ain, dtype=np.float64),
        "OutOfPlaneBifurcationVectorAngles": np.array(out_aout, dtype=np.float64),
        "BifurcationVectorsOrientation": np.array(out_orient, dtype=np.int32),
        group_ids: np.array(out_group, dtype=np.int32),
        "BifurcationGroupIds": np.array(out_bif, dtype=np.int32),
    }
    pts = vec3(out_pts)
    if vmtk_float32:                              # the output's default vtkPoints
        pts = round_float32(pts)
    return Centerlines(pts, [np.array([i]) for i in range(m)], pd)
