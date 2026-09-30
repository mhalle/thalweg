"""Centerlines of named structures in a ranked store, as a tube graph.

:func:`centerline_graph` decodes one structure's margin, traces its seed-free centerline tree
(:func:`thalweg.kernel.medial.trace`) and writes it into a :class:`~thalweg.graph.TubeGraph`:
nodes (root, junctions, tips; a tip where the structure runs off the field's grid is marked
``truncated``), one edge per segment between nodes, and the shared point table. :func:`combine`
puts several structures' graphs into one document.
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
                 length_mm=round(float(sum(e.length_mm for e in edges)), 3))
    structure = Structure(name=name, source=source, roots=roots, method="thalweg.trace",
                          parameters=parameters, statistics=stats)
    return nodes, edges, pos, rad, structure


def centerline_graph(store: FieldStore | str, name: str, *, part: int | None = None, graph: str = "field",
                     scale: float = medial.SCALE, const: float = medial.CONST, eps: float = medial.EPS,
                     ridge_passes: int = 4, prune: str = "length", margin=None, log=None) -> TubeGraph:
    """Trace structure ``name`` of a ranked store into a one-structure :class:`TubeGraph`.

    ``graph``: connectivity, ``"field"`` (decided by the interpolant) or ``"voxel"`` (the argmax
    labelmap, 26-connected; comparison only). ``ridge_passes`` and ``prune``: see
    :func:`thalweg.kernel.medial.trace` (``ridge_passes=1`` reproduces the research reference; the
    default is 4). ``margin``: an
    already decoded ``(margin, geometry, ref)`` from :meth:`FieldStore.margin`, to skip the decode
    (``thalweg.case.Case`` keeps them). Kernel errors come back as ThalwegError."""
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
                  ridge_passes=ridge_passes, prune=prune)
    nodes, edges, pos, rad, structure = graph_from_tree(tree, name, m, geometry, source, params)
    return TubeGraph(created_by=f"thalweg {__version__}", structures=[structure], nodes=nodes, edges=edges,
                     points=Points(position=pos, radius=rad))


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
