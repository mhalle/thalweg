"""vmtkbifurcationsections: a section of every branch a set distance from each bifurcation.

``vtkvmtkPolyDataBifurcationSections`` places, for each bifurcation (blanked) group, a section on
each branch group next to it - upstream (the parent) and downstream (the daughters). Per branch
group, every centerline cell of the group walks ``number_of_distance_spheres`` touching spheres
away from its bifurcation end (:func:`.sphere_distance.find_n_touching_sphere_center`); the
section's point and normal are the radius^2-weighted means of those cells' touching points and
segment directions. vmtk then cuts the SURFACE with that plane and measures the polygon.

:func:`bifurcation_section_planes` is the placement - centerline-only, and vmtk's to rounding on the
phantom oracle. :func:`section_area` and :func:`section_shape` are vmtk's measures of a section
polygon (area; the two calipers through the center along the nearest and farthest boundary points,
and their ratio), for any polygon: :mod:`thalweg.branching` applies them to sections of the field.

``vmtk_interp``: vmtk interpolates the touching point's radius with ``InterpolateTuple``, which
returns the segment's END value (see :mod:`.utilities`); the default interpolates.
"""
from __future__ import annotations

import numpy as np

from .centerlines import BLANKING, CENTERLINE_IDS, GROUP_IDS, RADIUS, TRACT_IDS, Centerlines
from .sphere_distance import find_n_touching_sphere_center
from .utilities import (find_adjacent_centerline_group_ids, get_blanked_groups_id_list,
                        get_group_unique_cell_ids, interpolate_point, interpolate_tuple)
from .vectors import DOWNSTREAM, UPSTREAM


def bifurcation_section_planes(cl: Centerlines, number_of_distance_spheres: int = 1, radius: str = RADIUS,
                               group_ids: str = GROUP_IDS, centerline_ids: str = CENTERLINE_IDS,
                               tract_ids: str = TRACT_IDS, blanking: str = BLANKING,
                               vmtk_interp: bool = False, vmtk_steps: bool = False) -> list[dict]:
    """One dict per section vmtk would try to cut, in vmtk's order: ``group``, ``bifurcation_group``,
    ``orientation`` (``UPSTREAM`` 0, ``DOWNSTREAM`` 1), ``point`` and ``normal`` (unit), and
    ``radius`` (the radius^2-weighted mean of the touching points' radii; not vmtk's). A group
    none of whose cells can walk the spheres (it is shorter than that) gets no section."""
    out = []
    for bif in get_blanked_groups_id_list(cl, group_ids, blanking):
        up, down = find_adjacent_centerline_group_ids(cl, bif, group_ids, centerline_ids, tract_ids)
        for group, orientation in [(g, UPSTREAM) for g in up] + [(g, DOWNSTREAM) for g in down]:
            p_sum, t_sum, w_sum, r_sum, any_point = np.zeros(3), np.zeros(3), 0.0, 0.0, False
            for cell_id in get_group_unique_cell_ids(cl, group, group_ids):
                n = len(cl.cells[cell_id])
                if n < 2:
                    continue
                if orientation == UPSTREAM:
                    sub, pc, forward = n - 2, 1.0, True
                else:
                    sub, pc, forward = 0, 0.0, False
                t_sub, t_pc = find_n_touching_sphere_center(cl, radius, cell_id, sub, pc,
                                                            number_of_distance_spheres, forward, vmtk_steps)
                if t_sub == -1:
                    continue
                any_point = True
                point = interpolate_point(cl, cell_id, t_sub, t_pc)
                p0, p1 = cl.cell_points(cell_id)[t_sub], cl.cell_points(cell_id)[t_sub + 1]
                tangent = p1 - p0
                norm = np.linalg.norm(tangent)
                tangent = tangent / norm if norm > 0 else tangent
                r = float(interpolate_tuple(cl, radius, cell_id, t_sub, t_pc, vmtk_interp)[0])
                w = r * r
                p_sum += w * point
                t_sum += w * tangent
                w_sum += w
                r_sum += w * r
            if not any_point:
                continue
            point = p_sum / w_sum
            normal = t_sum / w_sum
            normal = normal / np.linalg.norm(normal)
            out.append(dict(group=int(group), bifurcation_group=int(bif), orientation=int(orientation),
                            point=point, normal=normal, radius=r_sum / w_sum if w_sum > 0 else 0.0))
    return out


def section_area(polygon, normal) -> float:
    """The area of a planar polygon (k, 3) with the given unit normal (vmtk ear-cuts it and sums
    the triangles; for a simple polygon that is this)."""
    P = np.asarray(polygon, float)
    return float(abs(np.cross(P, np.roll(P, -1, axis=0)).sum(0) @ np.asarray(normal, float)) / 2.0)


def _intersect(a, b, c, d):
    """vtkLine::Intersection in the plane of the four points: (hit, u) for segment a-b against c-d."""
    u_dir, v_dir, w = b - a, d - c, a - c
    A = np.array([[u_dir @ u_dir, -(u_dir @ v_dir)], [-(u_dir @ v_dir), v_dir @ v_dir]])
    rhs = np.array([-(u_dir @ w), v_dir @ w])
    det = np.linalg.det(A)
    if abs(det) < 1e-12 * max(1.0, abs(A).max()) ** 2:
        return False, 0.0
    u, v = np.linalg.solve(A, rhs)
    return (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0), float(u)


def section_shape(polygon, center) -> tuple[float, float, float]:
    """vmtk's (MinSize, MaxSize, Shape) of a section polygon about ``center``
    (vtkvmtkPolyDataBranchSections::ComputeBranchSectionShape): find the boundary points nearest
    and farthest from the center; extend each direction through the center to the far side of
    the polygon; MinSize and MaxSize are those two calipers (near side + far side), Shape their
    ratio. A convex section's MinSize is at most its narrowest width, not equal to it."""
    P = np.asarray(polygon, float)
    c = np.asarray(center, float)
    if len(P) < 3:
        return 0.0, 0.0, 0.0
    d = np.linalg.norm(P - c, axis=1)
    i_min, i_max = int(np.argmin(d)), int(np.argmax(d))
    d_min, d_max = float(d[i_min]), float(d[i_max])
    sizes = []
    for base, i in ((d_min, i_min), (d_max, i_max)):
        direction = (P[i] - c) / np.linalg.norm(P[i] - c)
        opposite = c - 2.0 * d_max * direction
        far = 0.0
        for k in range(len(P)):
            hit, u = _intersect(opposite, c, P[k], P[(k + 1) % len(P)])
            if hit:
                far = max(far, float(np.linalg.norm((1.0 - u) * opposite + u * c - c)))
        sizes.append(base + far)
    return sizes[0], sizes[1], (sizes[0] / sizes[1] if sizes[1] > 0 else 0.0)


