"""Centerlines of named structures in a ranked store, as a tube graph.

:func:`centerline_graph` decodes one structure's margin, traces its seed-free centerline tree
(:func:`thalweg.kernel.medial.trace`) and writes it into a :class:`~thalweg.graph.TubeGraph`:
nodes (root, junctions, tips; a tip where the structure runs off the field's grid is marked
``truncated``), one edge per segment between nodes, and the shared point table. :func:`combine`
puts several structures' graphs into one document.

:func:`inlet` and :func:`reroot` root the tree at its inlet (the default; the tracer's own root is
its deepest point), :func:`radius_interval` adds the model's +-2-logit radius bounds as point
columns, and :func:`check_source` refuses a field the graph was not traced from.
"""
from __future__ import annotations

import numpy as np

from . import __version__
from .graph import Edge, Grid, Node, Points, Provenance, Source, Structure, TubeGraph
from .kernel import medial
from .kernel.field import to_index
from .errors import ThalwegError
from .store import FieldStore, open_store

TRUNCATION_VOXELS = 1.5


def _on_boundary(m: np.ndarray, geometry, point, radius: float) -> bool:
    """Does the structure reach the grid's edge next to ``point``? True when the point lies within
    TRUNCATION_VOXELS of a face of the array and the margin is positive on that face within
    ``radius`` + 1 voxel of the point's projection onto it."""
    i = to_index(geometry, point)
    shape = np.array(m.shape)
    spacing = np.linalg.norm(np.asarray(geometry.directions, float), axis=1)
    for a in range(3):
        for face in (0, shape[a] - 1):
            if abs(i[a] - face) > TRUNCATION_VOXELS:
                continue
            reach = (max(radius, 0.0) / spacing + 1.0)
            lo = np.floor(i - reach).astype(int).clip(0, shape - 1)
            hi = np.ceil(i + reach).astype(int).clip(0, shape - 1)
            sl = [slice(lo[k], hi[k] + 1) for k in range(3)]
            sl[a] = slice(int(face), int(face) + 1)
            if (m[tuple(sl)] > 0).any():
                return True
    return False


def graph_from_tree(tree: medial.MedialTree, name: str, m: np.ndarray, geometry, source: Source,
                    parameters: dict, node_offset: int = 0, edge_offset: int = 0,
                    point_offset: int = 0) -> tuple[list[Node], list[Edge], list, list, Structure]:
    """The traced tree as graph pieces (ids offset so several structures share one document).

    Two things the tracer's segments leave loose are tied here, so the document meets the format's
    invariants: a child branch starts at its own refined point, near but not at the junction node
    on its parent, so the node's position is prepended to its polyline (the connecting sample
    takes the child's first radius and adds its length); and node kinds follow degree (a tip or a
    truncated end has degree 1, a junction 3 or more, a degree-2 node is a joint, the root keeps
    its kind)."""
    method = parameters.get("connectivity", "field")
    deg = {nd["id"]: 0 for nd in tree.nodes}
    for sg in tree.segments:
        deg[sg["a"]] += 1
        deg[sg["b"]] += 1
    radius_at = {}
    for sg in tree.segments:
        radius_at.setdefault(sg["b"], sg["radius"][-1])
    nodes = []
    for nd in tree.nodes:
        kind = nd["kind"]
        attrs = {}
        if kind != "root":
            kind = "tip" if deg[nd["id"]] == 1 else ("joint" if deg[nd["id"]] == 2 else "junction")
        if kind in ("tip", "root") and _on_boundary(m, geometry, nd["point"], radius_at.get(nd["id"], 0.0)):
            if kind == "tip":
                kind = "truncated"
            else:
                attrs["on_grid_boundary"] = True
        nodes.append(Node(id=nd["id"] + node_offset, kind=kind, position=tuple(nd["point"]), structure=name,
                          attributes=attrs))
    gen = {b["id"]: b["generation"] for b in tree.branches}
    edges, pos, rad = [], [], []
    at = point_offset
    connected = 0
    for sg in tree.segments:
        pts = [tuple(p) for p in sg["points"]]
        rr = [float(r) for r in sg["radius"]]
        length = float(sg["length_mm"])
        start = tuple(tree.nodes[sg["a"]]["point"])
        if max(abs(a - b) for a, b in zip(pts[0], start)) > 0:
            gap = float(np.linalg.norm(np.subtract(pts[0], start)))
            pts.insert(0, start)
            rr.insert(0, rr[0])
            length += gap
            connected += 1
        n = len(pts)
        edges.append(Edge(id=sg["id"] + edge_offset, structure=name, start_node=sg["a"] + node_offset,
                          end_node=sg["b"] + node_offset, point_range=(at, at + n), length_mm=length,
                          provenance=Provenance(method=method), branch=sg["branch"],
                          generation=gen[sg["branch"]]))
        pos.extend(pts)
        rad.extend(rr)
        at += n
    roots = [nd.id for nd in nodes if nd.kind == "root"]
    stats = dict(tree.stats)
    stats.update(edges=len(edges), tips=sum(nd.kind == "tip" for nd in nodes),
                 truncated_ends=sum(nd.kind == "truncated" for nd in nodes),
                 junctions=sum(nd.kind == "junction" for nd in nodes),
                 joints=sum(nd.kind == "joint" for nd in nodes),
                 edges_joined_to_their_start_node=connected,
                 deepest_point=[float(v) for v in nodes[roots[0] - node_offset].position] if roots else None,
                 length_mm=round(float(sum(e.length_mm for e in edges)), 3))
    structure = Structure(name=name, source=source, roots=roots, method="thalweg.trace",
                          parameters=parameters, statistics=stats)
    return nodes, edges, pos, rad, structure


def centerline_graph(store: FieldStore | str, name: str, *, part: int | None = None, graph: str = "field",
                     scale: float = medial.SCALE, const: float = medial.CONST, eps: float = medial.EPS,
                     ridge_passes: int = 4, prune: str = "length", root: str = "inlet", margin=None,
                     log=None) -> TubeGraph:
    """Trace structure ``name`` of a ranked store into a one-structure :class:`TubeGraph`.

    ``graph``: connectivity, ``"field"`` (decided by the interpolant) or ``"voxel"`` (the argmax
    labelmap, 26-connected; comparison only). ``ridge_passes`` and ``prune``: see
    :func:`thalweg.kernel.medial.trace` (``ridge_passes=1`` reproduces the research reference; the
    default is 4). ``root``: ``"inlet"`` (the default) re-roots the traced tree at its inlet
    (:func:`inlet`); ``"deepest"`` keeps the tracer's root, its deepest point (the research
    reference). ``margin``: an already decoded ``(margin, geometry, ref)`` from
    :meth:`FieldStore.margin`, to skip the decode (``thalweg.case.Case`` keeps them). Kernel errors
    come back as ThalwegError."""
    if root not in ("inlet", "deepest"):
        raise ThalwegError(f"root must be 'inlet' or 'deepest'; got {root!r}")
    st = open_store(store) if not isinstance(store, FieldStore) else store
    m, geometry, ref = margin if margin is not None else st.margin(name, part)
    mask = st.labelmap_mask(ref) if graph == "voxel" else None
    try:
        tree = medial.trace(m, geometry, graph=graph, scale=scale, const=const, eps=eps, mask=mask,
                            ridge_passes=ridge_passes, prune=prune, log=log)
    except ValueError as e:
        raise ThalwegError(f"{name}: {e}") from e
    source = Source(store=str(st.path), labeling_scheme=ref.scheme, part=ref.part,
                    label_value=ref.label_value,
                    grid=Grid(shape=tuple(int(s) for s in m.shape),
                              directions=tuple(tuple(map(float, r)) for r in geometry.directions),
                              origin=tuple(map(float, geometry.origin))))
    params = dict(connectivity=graph, cover_scale=scale, cover_constant_mm=const, cost_epsilon_mm=eps,
                  ridge_passes=ridge_passes, prune=prune, root=root)
    nodes, edges, pos, rad, structure = graph_from_tree(tree, name, m, geometry, source, params)
    g = TubeGraph(created_by=f"thalweg {__version__}", structures=[structure], nodes=nodes, edges=edges,
                  points=Points(position=pos, radius=rad))
    if root == "inlet" and g.edges:
        n = inlet(g, name)
        deg = g.degree()
        widths = [end_width(g, nd.id) for nd in g.nodes if deg[nd.id] == 1]
        g = reroot(g, name, n)
        st = g.structures[0]
        stats = dict(st.statistics, inlet_end_width_mm=round(end_width(g, n), 4),
                     widest_end_width_mm=round(max(widths), 4) if widths else None)
        g = g.model_copy(update={"structures": [st.model_copy(update={"statistics": stats})]})
    return g


def combine(graphs: list[TubeGraph]) -> TubeGraph:
    """Several documents as one: ids and point ranges renumbered, structures kept in order."""
    nodes, edges, pos, rad, structures = [], [], [], [], []
    cols: dict[str, list] = {}
    for g in graphs:
        n0, e0, p0 = len(nodes), len(edges), len(pos)
        nmap = {nd.id: n0 + i for i, nd in enumerate(g.nodes)}
        for nd in g.nodes:
            nodes.append(nd.model_copy(update={"id": nmap[nd.id]}))
        for i, e in enumerate(g.edges):
            edges.append(e.model_copy(update={"id": e0 + i, "start_node": nmap[e.start_node],
                                              "end_node": nmap[e.end_node],
                                              "point_range": (e.point_range[0] + p0, e.point_range[1] + p0)}))
        for k in set(cols) | set(g.points.columns):
            cols.setdefault(k, [None] * p0).extend(g.points.columns.get(k, [None] * len(g.points.position)))
        pos.extend(g.points.position)
        rad.extend(g.points.radius)
        for s in g.structures:
            if any(s.name == t.name for t in structures):
                raise ThalwegError(f"structure {s.name!r} appears twice")
            structures.append(s.model_copy(update={"roots": [nmap[r] for r in s.roots]}))
    return TubeGraph(created_by=f"thalweg {__version__}", structures=structures, nodes=nodes, edges=edges,
                     points=Points(position=pos, radius=rad, columns=cols))


def check_source(structure, geometry, ref) -> None:
    """Refuse a margin that is not the one ``structure`` was traced from: the graph records its
    field's grid and label value, and measuring or exporting against another store's field gives
    numbers with no meaning (a ThalwegError, not a silent result)."""
    src = structure.source
    if src.label_value is not None and ref is not None and src.label_value != ref.label_value:
        raise ThalwegError(f"{structure.name}: the graph was traced from label value {src.label_value}, "
                           f"this store has it as {ref.label_value}")
    if src.grid is None:
        return
    same = (tuple(src.grid.shape) == tuple(int(v) for v in geometry.shape)
            and np.allclose(np.asarray(src.grid.origin), np.asarray(geometry.origin, float), atol=1e-4)
            and np.allclose(np.asarray(src.grid.directions), np.asarray(geometry.directions, float),
                            atol=1e-6))
    if not same:
        raise ThalwegError(f"{structure.name}: the store's field is not the one the graph was traced from "
                           f"(graph grid {tuple(src.grid.shape)} at "
                           f"{tuple(round(v, 3) for v in src.grid.origin)}, "
                           f"store grid {tuple(geometry.shape)} at "
                           f"{tuple(round(float(v), 3) for v in geometry.origin)})")


def end_width(graph: TubeGraph, node: int) -> float:
    """How wide the tube is that ends at ``node``: the 75th percentile of the traced radius along the
    node's terminal edge. Not the radius at the end itself - a tube narrows into its rounded end,
    and where a crop cuts a wide trunk (the pulmonary trunk leaving a lung crop) its last
    millimeters can read narrower than a side twig's."""
    for e in graph.edges:
        if node in (e.start_node, e.end_node):
            r = graph.edge_radius(e)
            r = r[r > 0]
            return float(np.percentile(r, 75)) if len(r) else 0.0
    return 0.0


OFF_FIELD_WIDTH_SHARE = 0.5


def inlet(graph: TubeGraph, structure: str) -> int:
    """The structure's inlet: where the tree enters, the node a rooted tree should start from.

    Among its ends (degree-1 nodes): the widest of those that run off the field (``truncated``, or
    a degree-1 root ``on_grid_boundary``) - a trachea or a pulmonary trunk leaving the crop -
    provided it is at least ``OFF_FIELD_WIDTH_SHARE`` (half) as wide as the widest end of all;
    otherwise the widest end. (A field of view that cuts thin peripheral branches must not root the
    tree at one of them.) Width is :func:`end_width`. The tracer's own root is its deepest point,
    which on a lung artery tree lies inside the pulmonary trunk, not at an end."""
    deg = graph.degree()
    ends = [nd for nd in graph.nodes if nd.structure == structure and deg[nd.id] == 1]
    if not ends:
        roots = graph.structure(structure).roots
        if not roots:
            raise ThalwegError(f"{structure!r} has no root")
        return roots[0]
    width = {nd.id: end_width(graph, nd.id) for nd in ends}
    widest = max(ends, key=lambda nd: width[nd.id])
    off = [nd for nd in ends if nd.kind == "truncated" or nd.attributes.get("on_grid_boundary")]
    if off:
        best = max(off, key=lambda nd: width[nd.id])
        if width[best.id] >= OFF_FIELD_WIDTH_SHARE * width[widest.id]:
            return best.id
    return widest.id


def reroot(graph: TubeGraph, structure: str, node: int) -> TubeGraph:
    """The structure re-rooted at ``node``: edges on the path from the old root to it are reversed
    (their samples and point columns reversed in place), so every edge points away from the new
    root. The new root keeps a record of what it was (``attributes.end_kind``: ``tip`` or
    ``truncated``); the old root takes the kind its degree gives and loses that record. Counts in
    the structure's statistics are recomputed; ``deepest_point`` (the tracer's start) is kept."""
    t = graph.tree(structure)
    if node == t.root:
        return graph
    if graph.nodes[node].structure != structure:
        raise ThalwegError(f"node {node} is not in {structure!r}")
    path = []                                              # edges from the old root down to node
    n = node
    while n != t.root:
        eid = t.parent[n]
        path.append(eid)
        n = graph.edges[eid].start_node
    flip = set(path)
    pos = list(graph.points.position)
    rad = list(graph.points.radius)
    cols = {k: list(v) for k, v in graph.points.columns.items()}
    edges = []
    for e in graph.edges:
        if e.id in flip:
            a, b = e.point_range
            pos[a:b] = pos[a:b][::-1]
            rad[a:b] = rad[a:b][::-1]
            for v in cols.values():
                v[a:b] = v[a:b][::-1]
            e = e.model_copy(update={"start_node": e.end_node, "end_node": e.start_node})
        edges.append(e)
    deg = graph.degree()
    nodes = []
    for nd in graph.nodes:
        if nd.id == node:
            attrs = dict(nd.attributes, end_kind=nd.kind)
            if nd.kind == "truncated":
                attrs["on_grid_boundary"] = True
            nd = nd.model_copy(update={"kind": "root", "attributes": attrs})
        elif nd.id == t.root:
            kind = "tip" if deg[nd.id] == 1 else ("joint" if deg[nd.id] == 2 else "junction")
            if kind == "tip" and nd.attributes.get("on_grid_boundary"):
                kind = "truncated"
            attrs = {k: v for k, v in nd.attributes.items() if k not in ("on_grid_boundary", "end_kind")}
            nd = nd.model_copy(update={"kind": kind, "attributes": attrs})
        nodes.append(nd)
    structures = []
    for s in graph.structures:
        if s.name == structure:
            mine = [nd for nd in nodes if nd.structure == structure]
            stats = dict(s.statistics, tips=sum(nd.kind == "tip" for nd in mine),
                         truncated_ends=sum(nd.kind == "truncated" for nd in mine),
                         junctions=sum(nd.kind == "junction" for nd in mine),
                         joints=sum(nd.kind == "joint" for nd in mine))
            s = s.model_copy(update={"roots": [node], "statistics": stats})
        structures.append(s)
    return graph.model_copy(update={"nodes": nodes, "edges": edges, "structures": structures,
                                    "points": Points(position=pos, radius=rad, columns=cols)})


INTERVAL_LOGITS = 2.0


def radius_interval(graph: TubeGraph, structure: str, m: np.ndarray, geometry,
                    logits: float = INTERVAL_LOGITS) -> TubeGraph:
    """The graph with the model's own interval on each sample's radius, as two point columns:

    - ``radius_lower_mm``: the distance from the sample to the surface where the margin is
      +``logits`` (the structure drawn more strictly); 0 where the sample itself is below that
      level (a thin branch the stricter surface does not contain);
    - ``radius_upper_mm``: the distance to the surface where the margin is -``logits``.

    The traced ``radius`` is the distance to the margin's zero set from the same sample, so
    lower <= radius <= upper wherever the refinement found an inside point (the bounds are clamped
    to it where the surfaces' point sampling would miss by a hair). Only this structure's samples
    are filled; others keep what they had (None if nothing)."""
    from scipy.spatial import cKDTree

    from .kernel.field import crossings, sample
    P = graph.positions()
    n = len(P)
    rows = np.zeros(n, bool)
    rows[graph.point_rows(structure)] = True
    cols = dict(graph.points.columns)
    lower = list(cols.get("radius_lower_mm", [None] * n))
    upper = list(cols.get("radius_upper_mm", [None] * n))
    if rows.any():
        Q = P[rows]
        here = sample(m, geometry, Q, cval=-8.0)
        out = []
        for level in (logits, -logits):
            X = crossings(m - np.float32(level), geometry)
            out.append(cKDTree(X).query(Q, workers=-1)[0] if len(X) else np.zeros(len(Q)))
        lo = np.where(here > logits, out[0], 0.0)
        hi = np.where(here > -logits, out[1], 0.0)
        r = graph.radii()[rows]
        found = r > 0                               # each surface is a sampled point set: in ~0.3 % of
        lo = np.where(found, np.minimum(lo, r), lo)  # samples a bound misses the radius by a hair
        hi = np.where(found, np.maximum(hi, r), hi)
        for i, a, b in zip(np.nonzero(rows)[0], lo, hi):
            lower[i], upper[i] = round(float(a), 4), round(float(b), 4)
    cols["radius_lower_mm"], cols["radius_upper_mm"] = lower, upper
    return graph.model_copy(update={"points": graph.points.model_copy(update={"columns": cols})})
