"""Per-branch measurements: the branch table, and the station profile it summarizes.

A structure must be a tree rooted at its root with edges pointing away from it
(:meth:`TubeGraph.tree`; a cycle is an error, not a hang). For every edge (a branch between two
nodes) :func:`branch_table` reports:

- **topology**: its end kinds; ``generation`` (the tracer's attachment depth, see the format);
  ``bifurcation_depth`` (junctions between the root and its start); ``strahler_order`` (a tip's
  edge 1; a parent takes the highest child order, plus one when two or more children share it;
  an edge ending at a ``truncated`` end - the structure runs on beyond the field - has no order,
  and parents ignore it);
- **size**: length along the traced path, chord, the traced radius (mean, min, max), and from
  field cross-sections every ``step`` mm the section area at the model's boundary with the
  model's own interval (``area_low_mm2`` at margin +2 logits, ``area_high_mm2`` at -2), the
  equivalent diameter, the minimum and maximum caliper widths and the aspect ratio: medians over
  the stations whose center lies inside the structure (``station_count``; stations whose center
  falls outside are counted apart and left out of every median);
- **shape**, from thalweg's spline (:mod:`thalweg.kernel.geometry`): distance metric,
  sum-of-angles metric, inflection count metric, curvature (mean, max), mean absolute torsion,
  over the edge's interior - one radius clear of each end, where the traced path hooks into the
  junction's ball or the tube's rounded end (``chord_mm`` stays the whole edge's). On an edge
  shorter than max(4 x its mean radius, 3 mm), or with under 2 mm of interior, these are not
  reported (None, ``shape_reliable`` False): a spline over a few samples measures its own wiggle;
- **branching** at the start junction: ``deflection_deg``, the angle between this branch's
  initial direction and the parent's continuation through the junction (0 = straight on), and
  ``sibling_angle_deg``, the angle to the nearest-in-angle sibling. A direction is a chord along
  the path from arc length r_J (the junction radius: outside the junction's ball, where every
  branch still points into the junction) over max(2 r_J, 3 mm). The parent's chord runs
  upstream from the junction and continues through earlier edges when the parent is shorter
  (near-coincident junctions leave stubs of a fraction of a millimeter). ``angle_reliable`` is
  False when a chord could not reach its full length; an angle is None when a path does not
  reach 0.5 mm beyond the junction's ball at all (no direction is defined there).

Everything here follows the tree's root. By default that is the structure's inlet
(:func:`thalweg.centerlines.inlet`: the pulmonary trunk's end for the arteries, the trachea for
the airways), so depth, order and angles run from where the tree enters. With
``root="deepest"`` (the research reference) the root lies inside the trunk: the inlet then counts
as a tip with Strahler order 1, and a root that lands on a bifurcation leaves that bifurcation
without a junction node (no angles there).

- **wall** (a lumen whose wall the model labels, e.g. ``lung_airways`` with ``lung_airways_wall``;
  pass the lumen-or-wall margin as ``outer``): at each station the outer contour's area minus the
  lumen's is the wall area; ``wall_area_percent`` = 100 x wall / outer area (the clinical WA%);
  ``wall_thickness_mm`` = wall area / the mean of the two perimeters; ``internal_perimeter_mm`` is
  the lumen's perimeter (for :func:`pi10`). Medians over the stations where the outer contour
  closes; ``wall_station_count`` counts them. The model's wall is a shell about two voxels thick
  (1.1-1.45 mm on a 0.7 mm grid, at every airway order), so thin walls are at its resolution.

Sections near a junction cut through the neighboring branches too, so stations within the
junction's radius + 1 mm of a junction end are left out (an edge shorter than that has none).
"""
from __future__ import annotations

import numpy as np

from .graph import Tree, TubeGraph
from .kernel import sections as S
from .kernel.geometry import SmoothPath, direction_at

LOW, HIGH = 2.0, -2.0            # margin levels for the interval: the small and the large area


def strahler(graph: TubeGraph, structure: str, tree: Tree | None = None) -> dict[int, int | None]:
    """Strahler order per edge (see the module docstring; truncated ends have no order)."""
    tree = tree or graph.tree(structure)
    kind = {nd.id: nd.kind for nd in graph.nodes}
    order: dict[int, int | None] = {}
    for eid in reversed(tree.order):                                  # children before parents
        e = graph.edges[eid]
        kids = [order[k] for k in tree.children.get(e.end_node, [])]
        known = [o for o in kids if o is not None]
        if not kids:
            order[eid] = None if kind[e.end_node] == "truncated" else 1
        elif not known:
            order[eid] = None
        else:
            top = max(known)
            order[eid] = top + 1 if known.count(top) >= 2 else top
    return order


def bifurcation_depth(graph: TubeGraph, structure: str, tree: Tree | None = None) -> dict[int, int]:
    """Junctions between the root and each edge's start node."""
    tree = tree or graph.tree(structure)
    kind = {nd.id: nd.kind for nd in graph.nodes}
    depth: dict[int, int] = {}
    for eid in tree.order:                                            # parents before children
        e = graph.edges[eid]
        up = tree.parent.get(e.start_node)
        depth[eid] = 0 if up is None else depth[up] + (1 if kind[e.start_node] == "junction" else 0)
    return depth


MIN_CHORD_MM = 0.5


def chord_length(radius: float) -> float:
    return max(2.0 * radius, 3.0)


def _upstream_path(graph: TubeGraph, tree: Tree, eid: int, need: float) -> np.ndarray:
    """The path from edge ``eid``'s end back toward the root, through as many edges as it takes to
    be ``need`` mm long (or up to the root)."""
    parts = [graph.edge_points(eid)[::-1]]
    total = float(np.sum(np.linalg.norm(np.diff(parts[0], axis=0), axis=1)))
    e = graph.edges[eid]
    while total < need and e.start_node in tree.parent:
        e = graph.edges[tree.parent[e.start_node]]
        p = graph.edge_points(e)[::-1]
        parts.append(p[1:])
        total += float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
    return np.concatenate(parts)


def _length(p: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def _angle(a, b) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def branch_table(graph: TubeGraph, structure: str, margin: np.ndarray | None = None, geometry=None,
                 step: float = 1.0, max_pixels: int = 256, stations_out: list | None = None,
                 outer: np.ndarray | None = None) -> list[dict]:
    """One row per edge of ``structure`` (see the module docstring). ``margin``/``geometry``: the
    structure's field; without them the section columns are left out. ``stations_out``: a list
    that receives one row per section station (the profile). ``outer``: for a lumen with a
    labeled wall, the margin of lumen-or-wall (the max of the two classes' margins, same grid):
    adds the wall columns (see the module docstring)."""
    tree = graph.tree(structure)
    nodes = {nd.id: nd for nd in graph.nodes}
    order = strahler(graph, structure, tree)
    depth = bifurcation_depth(graph, structure, tree)
    rows = []
    for eid in sorted(tree.order):
        e = graph.edges[eid]
        p, r = graph.edge_points(e), graph.edge_radius(e)
        pos = r[r > 0]
        rr = np.where(r > 0, r, 0.5)
        a_kind, b_kind = nodes[e.start_node].kind, nodes[e.end_node].kind
        row = dict(structure=structure, edge=e.id, start_node=e.start_node, end_node=e.end_node,
                   start_kind=a_kind, end_kind=b_kind, generation=e.generation,
                   bifurcation_depth=depth[e.id], strahler_order=order[e.id],
                   provenance_method=e.provenance.method, length_mm=e.length_mm,
                   radius_mean_mm=float(pos.mean()) if len(pos) else None,
                   radius_min_mm=float(pos.min()) if len(pos) else None,
                   radius_max_mm=float(pos.max()) if len(pos) else None)
        try:
            path = SmoothPath(p, rr)
        except (ValueError, TypeError):
            path = None
        r0, r1 = float(rr[0]), float(rr[-1])
        interior = (path.length - r0 - r1) if path is not None else 0.0
        reliable = (path is not None and e.length_mm >= max(4.0 * (row["radius_mean_mm"] or 0.5), 3.0)
                    and interior >= 2.0)
        row["chord_mm"] = float(np.linalg.norm(p[-1] - p[0]))
        row["shape_reliable"] = bool(reliable)
        if reliable:
            g = path.geometry(step=min(step, 0.5), start=r0, stop=path.length - r1)
            row.update(distance_metric=g.distance_metric, sum_of_angles_rad_per_mm=g.sum_of_angles,
                       inflection_count_metric=g.inflection_count_metric,
                       curvature_mean_per_mm=float(np.mean(g.curvature)),
                       curvature_max_per_mm=float(np.max(g.curvature)),
                       torsion_mean_absolute_per_mm=(float(np.nanmean(np.abs(g.torsion)))
                                                     if np.isfinite(g.torsion).any() else None))
        else:
            row.update(distance_metric=None, sum_of_angles_rad_per_mm=None, inflection_count_metric=None,
                       curvature_mean_per_mm=None, curvature_max_per_mm=None,
                       torsion_mean_absolute_per_mm=None)
        row.update(_branching(graph, tree, e))
        if margin is not None:
            row.update(_sections(path, p, r, e, a_kind, b_kind, margin, geometry, step, max_pixels, outer,
                                 stations_out))
        rows.append(row)
    return rows


def _branching(graph: TubeGraph, tree: Tree, e) -> dict:
    parent = tree.parent.get(e.start_node)
    if parent is None:
        return dict(deflection_deg=None, sibling_angle_deg=None, angle_reliable=None)
    rj = max(float(graph.edge_radius(e)[0]), 0.5)
    need = rj + chord_length(rj)
    mine_p = graph.edge_points(e)
    up = _upstream_path(graph, tree, parent, need)

    def chord(p):
        """The direction leaving the junction along p, or None where p does not get MIN_CHORD_MM
        beyond the junction's ball (no direction is defined there; a zero vector would read 90)."""
        return direction_at(p, chord_length(rj), start=rj) if _length(p) >= rj + MIN_CHORD_MM else None

    mine = chord(mine_p)
    back = chord(up)
    ok = _length(mine_p) >= need and _length(up) >= need
    sib = []
    for o in tree.children.get(e.start_node, []):
        if o == e.id:
            continue
        op = graph.edge_points(o)
        other = chord(op)
        if mine is not None and other is not None:
            sib.append(_angle(mine, other))
        ok = ok and _length(op) >= need
    return dict(deflection_deg=_angle(mine, -back) if mine is not None and back is not None else None,
                sibling_angle_deg=min(sib) if sib else None, angle_reliable=bool(ok))


SECTION_KEYS = ("area_mm2", "area_low_mm2", "area_high_mm2", "equivalent_diameter_mm", "min_feret_mm",
                "max_feret_mm", "aspect_ratio")


WALL_KEYS = ("wall_area_mm2", "wall_area_percent", "wall_thickness_mm", "internal_perimeter_mm")


def _wall(img_outer, g, lumen: dict) -> dict:
    """The wall around one lumen section: outer contour (lumen or wall) minus the lumen."""
    o = S.describe(img_outer, g, 0.0)
    if not o["area"] or not o["closed"] or o["area"] < lumen["area"]:
        return {k: None for k in WALL_KEYS}
    wa = o["area"] - lumen["area"]
    return dict(wall_area_mm2=wa, wall_area_percent=100.0 * wa / o["area"],
                wall_thickness_mm=wa / (0.5 * (lumen["perimeter"] + o["perimeter"])),
                internal_perimeter_mm=lumen["perimeter"])


def _sections(path, p, r, e, a_kind, b_kind, margin, geometry, step, max_pixels, outer, stations_out) -> dict:
    keys = SECTION_KEYS + (WALL_KEYS if outer is not None else ())
    empty = {k: None for k in keys} | {"station_count": 0, "stations_center_outside": 0}
    if path is None:
        return empty
    st = path.stations(step)
    guard_a = (r[0] + 1.0) if a_kind in ("junction", "root") else 0.0
    guard_b = (r[-1] + 1.0) if b_kind == "junction" else 0.0
    L = st.s[-1] if len(st.s) else 0.0
    vals = {k: [] for k in keys}
    outside = 0
    for i in range(len(st.s)):
        if st.s[i] < guard_a or st.s[i] > L - guard_b:
            continue
        half = S.half_width(st.radius[i])
        pixel = max(S.PIXEL, 2 * half / max_pixels)
        img, g = S.section_image(margin, geometry, st.centers[i], st.n1[i], st.n2[i], half, pixel)
        d = S.describe(img, g, 0.0)
        if d["area"] == 0.0:
            outside += 1
            continue
        areas = S.pixel_areas(img, pixel, (HIGH, 0.0, LOW))
        rec = dict(area_mm2=d["area"], area_low_mm2=float(areas[2]), area_high_mm2=float(areas[0]),
                   equivalent_diameter_mm=d["equivalent_diameter"], min_feret_mm=d["min_feret"],
                   max_feret_mm=d["max_feret"], aspect_ratio=d["aspect_ratio"])
        if outer is not None:
            img_o, _ = S.section_image(outer, geometry, st.centers[i], st.n1[i], st.n2[i], half, pixel)
            rec.update(_wall(img_o, g, d))
        for k in keys:
            if rec[k] is not None:
                vals[k].append(rec[k])
        if stations_out is not None:
            stations_out.append(dict(structure=e.structure, edge=e.id, arc_length_mm=float(st.s[i]),
                                     position_x_mm=float(st.centers[i, 0]),
                                     position_y_mm=float(st.centers[i, 1]),
                                     position_z_mm=float(st.centers[i, 2]),
                                     traced_radius_mm=float(st.radius[i]),
                                     contour_closed=d["closed"], perimeter_mm=d["perimeter"],
                                     centroid_offset_mm=d["centroid_offset"], **rec))
    out = {k: (float(np.median(v)) if v else None) for k, v in vals.items()}
    out["station_count"] = len(vals["area_mm2"])
    if outer is not None:
        out["wall_station_count"] = len(vals["wall_area_mm2"])
    out["stations_center_outside"] = outside
    return out


def pi10(stations: list[dict]) -> dict:
    """Pi10: the square root of wall area (mm) predicted at an internal perimeter of 10 mm, from a
    least-squares line of sqrt(wall area) on internal perimeter over every measured station - the
    airway-wall summary of Nakano et al. (2005). Also the fit's slope and station count."""
    pts = [(s_["internal_perimeter_mm"], np.sqrt(s_["wall_area_mm2"])) for s_ in stations
           if s_.get("wall_area_mm2") is not None and s_.get("internal_perimeter_mm")]
    if len(pts) < 10:
        return dict(pi10_mm=None, slope=None, stations=len(pts))
    x, y = np.asarray(pts, float).T
    b, a = np.polyfit(x, y, 1)
    return dict(pi10_mm=float(a + 10.0 * b), slope=float(b), stations=len(pts),
                internal_perimeter_range_mm=[float(x.min()), float(x.max())])


def write_table(rows: list[dict], path) -> None:
    """Rows -> Parquet (needs pyarrow: the ``tables`` extra).

    The table's columns are the union of every row's keys, in first-seen order, with None where a
    row lacks one: rows of different structures carry different columns (only airways have wall
    and pairing columns), and pyarrow would otherwise take the schema from the first row alone."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:                     # pragma: no cover
        raise ImportError("writing Parquet needs pyarrow: pip install 'thalweg[tables]'") from e
    keys: dict[str, None] = {}
    for r in rows:
        keys.update(dict.fromkeys(r))
    pq.write_table(pa.Table.from_pylist([{k: r.get(k) for k in keys} for r in rows]), path)
