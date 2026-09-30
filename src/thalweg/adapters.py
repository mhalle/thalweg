"""Between thalweg's graph and vmtk's centerline convention.

vmtk describes a tree as one polyline per target, each running from the one source to its target
with the shared stretch repeated, radius in ``MaximumInscribedSphereRadius``. thalweg's graph
holds each stretch once, as edges between nodes. :func:`to_vmtk` writes the graph's
source-to-tip paths in vmtk's convention, the recipe the research comparisons used
(``research/vessels/vmtk_prep.py`` + ``vmtk_branch.py``):

1. every edge on a path is densified to ``densify`` mm by arc length (its own vertices kept at
   the ends), the radius interpolated the same way;
2. the edges of a path are concatenated (junction points repeated, as the reference did) and the
   whole path resampled to ``step`` mm from the source, so paths that share a stretch share its
   samples exactly;
3. each resampled point takes the radius of the nearest densified point of the subtree.

``source`` may be any node or a (edge, trim) pair: the path starts ``trim`` mm into that edge
(the reference started one radius plus 1 mm past a junction). Targets are the tips (and
truncated ends) downstream of the source.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .errors import ThalwegError
from .graph import TubeGraph
from .vmtk.centerlines import RADIUS, Centerlines


@dataclass
class VmtkPaths:
    """What :func:`to_vmtk` built: the centerlines, and for each cell the tip node and the graph
    edges (in path order) it runs through."""

    centerlines: Centerlines
    targets: list[int]
    edges: list[list[int]]


def densify(p: np.ndarray, r: np.ndarray | None = None, step: float = 0.1):
    """Resample a polyline every ``step`` mm of arc length from its first point (the last sample
    falls at or before the end)."""
    p = np.asarray(p, float)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    t = np.arange(0, s[-1] + 1e-9, step)
    q = np.stack([np.interp(t, s, p[:, a]) for a in range(3)], 1)
    return (q, np.interp(t, s, np.asarray(r, float))) if r is not None else q


def _downstream(graph: TubeGraph, structure: str) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for e in graph.edges:
        if e.structure == structure:
            out.setdefault(e.start_node, []).append(e.id)
    return out


def to_vmtk(graph: TubeGraph, structure: str, source: int | tuple[int, float] | None = None,
            step: float = 0.3, densify_step: float = 0.1, min_points: int = 4) -> VmtkPaths:
    """The source-to-tip paths of one structure as vmtk centerlines (see the module docstring).

    ``source``: a node id (default: the structure's root) or ``(edge id, trim mm)``. Paths with
    fewer than ``min_points`` densified points are dropped, as the reference did (``len > 3``).
    """
    if not graph.tree(structure).order:                    # a tree, edges away from the root, or ThalwegError
        raise ThalwegError(f"{structure!r} has no edges (a compact structure traced as its root alone)")
    out_of = _downstream(graph, structure)
    edges = {e.id: e for e in graph.edges}
    trim_edge, trim = None, 0.0
    if source is None:
        roots = graph.structure(structure).roots
        if len(roots) != 1:
            raise ValueError(f"{structure!r} has {len(roots)} roots; name a source")
        first = out_of.get(roots[0], [])
    elif isinstance(source, tuple):
        trim_edge, trim = int(source[0]), float(source[1])
        first = [trim_edge]
    else:
        first = out_of.get(int(source), [])

    def pts(eid):
        p, r = graph.edge_points(eid), graph.edge_radius(eid)
        if eid == trim_edge:
            c = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
            k0 = int(np.argmax(c >= trim))
            p, r = p[k0:], r[k0:]
        return p, r

    # every edge of the subtree, densified once. The visiting order is the reference's (a stack,
    # children pushed in order and popped last-first): cell order decides vmtk's group numbering,
    # and the nearest-radius lookup breaks exact ties by it.
    sub, stack, parent = [], list(first), {}
    while stack:
        j = stack.pop()
        sub.append(j)
        for k in out_of.get(edges[j].end_node, []):
            parent[k] = j
            stack.append(k)
    dense = {j: densify(*pts(j), step=densify_step) for j in sub}
    tree = cKDTree(np.concatenate([dense[j][0] for j in sub]))
    rad = np.concatenate([dense[j][1] for j in sub])

    # one path per end edge (no edge leaves its end node), in visiting order
    paths = []
    for j in sub:
        if not out_of.get(edges[j].end_node):
            chain = [j]
            while chain[-1] in parent:
                chain.append(parent[chain[-1]])
            paths.append(chain[::-1])
    lines, radii, targets, chains = [], [], [], []
    for chain in paths:
        p = np.concatenate([dense[j][0] for j in chain])
        if len(p) <= min_points - 1:
            continue
        q = densify(p, step=step)
        lines.append(q)
        radii.append(rad[tree.query(q)[1]])
        targets.append(edges[chain[-1]].end_node)
        chains.append(chain)
    return VmtkPaths(Centerlines.from_lines(lines, radii, RADIUS), targets, chains)
