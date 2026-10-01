"""The seed-free centerline tree of one structure, read from its margin field (TEASAR-style).

Ported from ``research/vessels/centerline.py`` (the reference: its output JSON must be
reproduced). Four steps, over one margin ``m`` (positive inside) on an oriented grid:

1. **Distance.** Every interior lattice point's distance to the margin's sub-voxel zero crossings
   (a KD tree over :func:`~thalweg.kernel.field.crossings`: exact to the crossing sampling and
   unbounded - a stored narrow-band distance cannot see the medial axis of anything wide).
2. **Paths.** Dijkstra from the root (the deepest point) over the lattice graph whose joins the
   FIELD decides (``graph="field"``: faces, face saddles, cell interiors -
   :mod:`~thalweg.kernel.topology`); ``graph="voxel"`` is the 26-connected labelmap graph, kept
   for comparison. Edge cost = length * mean(1 / (d + eps)^2): vmtk's integral of ds / R,
   sharpened, on the lattice instead of the Voronoi diagram.
3. **Branches without seeds.** Repeatedly take the uncovered point farthest from the root along
   the tree (path length), walk its minimal path back to the skeleton so far, and cover every
   point within ``scale * d + const`` mm of the new path. Terminal branches shorter than
   2 * (radius at their junction) + 1 mm are pruned as spurs.
4. **Radius.** Each path point moves to the largest inscribed ball within +-0.5 mm (inside the
   margin); that ball's radius is the point's radius. In field mode, a chord between consecutive
   points that leaves the structure (a join that curves around a face saddle) gets its midpoint
   inserted and moved to the local ridge, up to four rounds.

The tree is then split at every child's junction into segments between nodes (root, junctions,
tips), vmtk's branch-splitting shape. Positions are quantized to 1 um (0.001 mm) before the split,
as the reference does; radii likewise.

Only the largest connected component is traced (the reference's behavior); ``stats`` reports the
others so a caller can see what was dropped. A structure too compact to have a branch (a blob no
wider than its own cover) comes back as its root alone.

Three options depart from the reference (docs/validation.md §5 has the measurements; defaults
decided 2026-09-30, ``recenter`` added after):

- ``ridge_passes``: coarse-to-fine ridge refinement (:func:`~thalweg.kernel.field.inscribed_radius`).
  The default is 4, which removes the one-pass refinement's quantization (a radius 0.03-0.07 mm
  small on oblique vessels, ~0.1 mm on grid-aligned phantoms) and matches vmtk, at 1.85x the trace
  time. ``ridge_passes=1`` is the research reference, which the reproduction tests pin;
- ``prune="wall"`` (the default stays ``"length"``, the reference rule): after refinement, also
  drop terminal branches whose tip does not reach beyond the parent's wall by max(its own
  radius, 1 mm), the wall measured by casting rays through the
  field from the parent's axis, clear of the junction. The length rule assumes round lumens: in a
  flattened one the junction's inscribed radius is half the depth, and side lobes across the
  width survive as branches (an elliptic tube of 3 x 1.2 mm traced as 24 ends, an esophagus as 13).
  Branches leaving the root are tested against its first branch, unless they continue its axis
  backward (:func:`prune_by_wall`);
- ``recenter`` (default off here; the pipeline turns it on for flat tubes): after pruning, move
  every point to the area centroid of its cross-section (:mod:`.recenter`), holding radius + 1 mm
  around every node. The path then lies on a flattened lumen's axis (2:1 to 3:1 elliptic
  tubes: 0.26-1.14 mm median off it -> 0.002 mm), where the tracer wanders across the width; a
  round tube barely moves. Pair it with ``prune="wall"``: the
  length rule's side lobes hold the path at their junctions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from .field import crossings, edge_lengths, inscribed_radius, sample, to_world
from . import recenter as _rc
from .topology import OFFSETS, components, field_edges, voxel_edges

SCALE, CONST, EPS = 1.5, 1.0, 0.1
STATION_COS = np.cos(np.radians(30.0))     # wall-pruning stations must run within 30 deg of the junction


@dataclass
class MedialTree:
    """The traced tree. ``branches``: TEASAR branches, each junction -> tip (dicts: id, parent,
    generation, points, radius, lattice_points, length_mm). ``nodes`` (id, kind, point) and
    ``segments`` (id, branch, a, b, points, radius, length_mm): the same tree split between nodes.
    ``root``: the root's world position. ``stats``: counts for reporting."""

    root: np.ndarray
    branches: list[dict]
    nodes: list[dict]
    segments: list[dict]
    stats: dict = field(default_factory=dict)


def _path_length(world, path) -> float:
    return float(np.sum(np.linalg.norm(np.diff(world[path], axis=0), axis=1)))


def trace(m: np.ndarray, geometry, graph: str = "field", scale: float = SCALE, const: float = CONST,
          eps: float = EPS, mask: np.ndarray | None = None, ridge_passes: int = 4, prune: str = "length",
          recenter: bool = False, log=None) -> MedialTree:
    """The centerline tree of ``{m > 0}``'s largest component (see the module docstring).

    ``m``: the margin, float32, positive inside, on ``geometry`` (origin + direction rows).
    ``mask``: voxel mode only - the lattice points to connect (default ``m > 0``; the reference
    used the argmax labelmap, which also counts exact ties m == 0 won by this class).
    ``log``: optional callable taking one message string per step. ``ridge_passes``, ``prune`` and
    ``recenter``: see the module docstring.
    """
    if prune not in ("length", "wall"):
        raise ValueError(f"prune must be 'length' or 'wall'; got {prune!r}")
    if int(ridge_passes) < 1:
        raise ValueError(f"ridge_passes must be at least 1; got {ridge_passes!r}")
    say = log or (lambda msg: None)
    if graph == "field":
        idx_all, r_all, c_all, k_all, tun = field_edges(m)
    elif graph == "voxel":
        idx_all, r_all, c_all, k_all = voxel_edges(m > 0 if mask is None else mask, 26)
        tun = 0
    else:
        raise ValueError(f"graph must be 'field' or 'voxel'; got {graph!r}")
    if len(idx_all) == 0:
        raise ValueError("the structure is empty: no lattice point has m > 0")
    ncomp, comp = components(len(idx_all), r_all, c_all)
    sizes = np.bincount(comp)
    main = np.argmax(sizes)
    keep_node = comp == main
    new_id = -np.ones(len(idx_all), np.int64)
    new_id[keep_node] = np.arange(keep_node.sum())
    e_ok = keep_node[r_all]                                             # both ends share a component
    idx = idx_all[keep_node]
    rows, cols = new_id[r_all[e_ok]], new_id[c_all[e_ok]]
    lens = edge_lengths(geometry, OFFSETS)[k_all[e_ok]]
    world = to_world(geometry, idx)
    X = crossings(m, geometry)
    xtree = cKDTree(X)
    d = xtree.query(world, workers=-1)[0]
    say(f"[{graph} graph] main component {len(idx)} of {len(idx_all)} nodes ({ncomp} components), "
        f"{len(X)} crossings, max distance {d.max():.1f} mm, {tun} cell-interior joins")
    w = 1.0 / (d + eps) ** 2
    cost = lens * 0.5 * (w[rows] + w[cols])
    N = len(idx)
    Gc = csr_matrix((np.r_[cost, cost], (np.r_[rows, cols], np.r_[cols, rows])), shape=(N, N))
    Gl = csr_matrix((np.r_[lens, lens], (np.r_[rows, cols], np.r_[cols, rows])), shape=(N, N))
    root = int(np.argmax(d))
    _, pred = dijkstra(Gc, indices=root, return_predecessors=True)
    daf = dijkstra(Gl, indices=root)
    say(f"graph {len(rows)} edges; root at {np.round(world[root], 1)} (d = {d[root]:.1f} mm)")

    ntree = cKDTree(world)
    valid = np.isfinite(daf)
    in_skel = np.zeros(N, bool)
    in_skel[root] = True
    branch_of = -np.ones(N, np.int64)
    branches = []                                                       # (nodes tip -> junction, parent)

    def cover(nodes):
        radius = np.round(scale * d[nodes] + const, 1)
        for r in np.unique(radius):
            sel = nodes[radius == r]
            for hit in ntree.query_ball_point(world[sel], r):
                valid[hit] = False

    cover(np.array([root]))
    score = np.where(valid, daf, -1.0)
    while True:
        t = int(np.argmax(score))
        if score[t] < 0:
            break
        path = [t]
        while not in_skel[path[-1]]:
            p = pred[path[-1]]
            if p < 0:
                break
            path.append(p)
        path = np.array(path)
        junction = path[-1]
        parent = int(branch_of[junction]) if branch_of[junction] >= 0 else -1
        newp = path[:-1] if in_skel[junction] else path
        in_skel[newp] = True
        branch_of[newp] = len(branches)
        branches.append((path, parent))
        cover(newp)
        valid[t] = False
        score[~valid] = -1.0
    say(f"TEASAR: {len(branches)} raw branches")

    # spur pruning: terminal branches shorter than 2 R(junction) + 1 mm
    children = {i: 0 for i in range(len(branches))}
    for _, par in branches:
        if par >= 0:
            children[par] += 1
    alive = [i for i, (path, par) in enumerate(branches)
             if not (children[i] == 0 and par >= 0 and _path_length(world, path) < 2 * d[path[-1]] + 1.0)]
    say(f"pruned {len(branches) - len(alive)} spurs, {len(alive)} branches remain")

    # ridge refinement and radius on the kept branch points
    def inside(q):
        return sample(m, geometry, q) > 0

    # every branch at once: the refinement is point-wise, so one batched call per step gives the
    # same numbers as one call per branch (the reference's loop), at a fraction of the calls
    stats = dict(traced_lattice_points=int(N), lattice_points=int(len(idx_all)), components=int(ncomp),
                 component_sizes=sorted((int(s) for s in sizes), reverse=True), zero_crossings=int(len(X)),
                 cell_interior_joins=int(tun), branches_before_pruning=len(branches),
                 max_distance_mm=float(d.max()))
    if not alive:                                                       # nothing beyond the root's cover
        say("no branch: the structure is its root's cover")
        r0, q0 = inscribed_radius(world[[root]], xtree, inside, passes=ridge_passes)
        node = dict(id=0, kind="root", point=[float(v) for v in np.round(q0[0], 3)])
        return MedialTree(root=np.round(world[root], 3), branches=[], nodes=[node], segments=[],
                          stats=dict(stats, branches=0))
    lattice = [world[branches[i][0]][::-1] for i in alive]              # junction -> tip
    cuts = np.cumsum([len(p) for p in lattice])[:-1]
    r_all, q_all = inscribed_radius(np.concatenate(lattice), xtree, inside, passes=ridge_passes)
    qs, rs = np.split(q_all, cuts), np.split(r_all, cuts)
    if graph == "field":
        # a join the field makes can curve around a face saddle, so the chord between two joined
        # points may clip the outside: insert the midpoint of any chord that leaves the structure,
        # move it to the local ridge (inside by construction), repeat up to four rounds
        tt = np.linspace(0, 1, 11)[1:-1, None]
        for _ in range(4):
            chords = [(b, k) for b in range(len(qs)) for k in range(len(qs[b]) - 1)]
            if not chords:
                break
            a = np.array([qs[b][k] for b, k in chords])
            z = np.array([qs[b][k + 1] for b, k in chords])
            probe = (a[:, None, :] + tt[None] * (z - a)[:, None, :]).reshape(-1, 3)
            leaves = (sample(m, geometry, probe) <= 0).reshape(len(chords), -1).any(1)
            if not leaves.any():
                break
            hit = np.nonzero(leaves)[0]
            rm, qm = inscribed_radius(0.5 * (a[hit] + z[hit]), xtree, inside, step=0.35, n=7,
                                      passes=ridge_passes)
            per = {}
            for h, j in enumerate(hit):
                b, k = chords[j]
                per.setdefault(b, []).append((k + 1, h))
            for b, ins in per.items():
                where_ = [w for w, _ in ins]
                qs[b] = np.insert(qs[b], where_, qm[[h for _, h in ins]], axis=0)
                rs[b] = np.insert(rs[b], where_, rm[[h for _, h in ins]])
    out, gen = [], {}
    remap = {old: new for new, old in enumerate(alive)}
    for n_, i in enumerate(alive):
        path, par = branches[i]
        while par >= 0 and par not in remap:                    # a pruned parent cannot occur; guard anyway
            par = branches[par][1]
        pts, q, r = lattice[n_], qs[n_], rs[n_]
        gen[i] = 0 if par < 0 else gen[par] + 1
        out.append(dict(id=remap[i], parent=remap.get(par, -1), generation=gen[i],
                        points=np.round(q, 3).tolist(), radius=np.round(r, 3).tolist(),
                        lattice_points=np.round(pts, 3).tolist(),
                        length_mm=round(_path_length(world, path), 2)))
    say("radius refined")
    if prune == "wall":
        out, dropped = prune_by_wall(out, m, geometry)
        say(f"wall pruning dropped {dropped} terminal branches")
        stats["pruned_by_wall"] = dropped
    if recenter:
        if out:                                         # where the tracer started, before any move
            stats["deepest_point"] = [float(v) for v in out[0]["points"][0]]
        out, n_moved = recenter_branches(out, m, geometry, xtree)
        say(f"recentered {n_moved} points")
        stats["recentered_points"] = n_moved
    nodes, segments = split(out)
    stats["branches"] = len(out)
    return MedialTree(root=np.round(world[root], 3), branches=out, nodes=nodes, segments=segments,
                      stats=stats)


def _junction_index(parent: dict, child: dict) -> int:
    """Where ``child`` leaves ``parent``: the parent's point nearest the child's first point."""
    return int(np.argmin(np.linalg.norm(np.array(parent["points"]) - np.array(child["points"][0]), axis=1)))


def recenter_branches(branches: list[dict], m, geometry, xtree) -> tuple[list[dict], int]:
    """Each branch's points moved to their sections' area centroids (:mod:`.recenter`), holding
    radius + 1 mm around both ends and every child's junction on it, so the nodes stay where they
    are. A root that only two branches leave is a point along one path (the deepest point,
    mid-tube), not a node to hold: those two are recentered as one path through it.
    Returns (the branches, the number of points moved)."""
    at = {b["id"]: [] for b in branches}
    for b in branches:
        if b["parent"] >= 0:
            at[b["parent"]].append(_junction_index(branches[b["parent"]], b))
    paths = [np.array(b["points"], float) for b in branches]
    radii = [np.array(b["radius"], float) for b in branches]
    holds = [_rc.end_holds(p, r, at[b["id"]]) for p, r, b in zip(paths, radii, branches)]
    roots = [b["id"] for b in branches if b["parent"] < 0]
    through = len(roots) == 2 and min(len(paths[i]) for i in roots) > 1
    if through:                                         # one path: reversed second branch, then the first
        a, b = roots
        nb = len(paths[b])
        paths[a] = np.concatenate([paths[b][::-1], paths[a][1:]])
        radii[a] = np.concatenate([radii[b][::-1], radii[a][1:]])
        joins = [nb - 1 + k for k in at[a]] + [nb - 1 - k for k in at[b]]
        holds[a] = _rc.end_holds(paths[a], radii[a], joins)
        paths[b], radii[b], holds[b] = paths[b][:1], radii[b][:1], np.ones(1, bool)
    P, R, moved = _rc.recenter(m, geometry, paths, radii, holds, xtree)
    if through:
        both = moved[a]
        P[b], R[b], moved[b] = P[a][:nb][::-1], R[a][:nb][::-1], both[:nb][::-1].copy()
        P[a], R[a], moved[a] = P[a][nb - 1:], R[a][nb - 1:], both[nb - 1:].copy()
        moved[a][0] = False                             # the shared root point, counted once (in b)
    out = [dict(b_, points=np.round(p, 3).tolist(), radius=np.round(r, 3).tolist())
           for b_, p, r in zip(branches, P, R)]
    return out, int(sum(int(v.sum()) for v in moved))


def split(branches: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split every TEASAR branch at its children's junctions: segments between nodes (the root,
    junctions, tips). ``branches`` must be in creation order (parents first), as :func:`trace`
    returns them."""
    cuts = {b["id"]: [] for b in branches}
    for b in branches:
        if b["parent"] >= 0:
            cuts[b["parent"]].append((_junction_index(branches[b["parent"]], b), b["id"]))
    nodes, segments = [], []

    def node_at(p, kind):
        nodes.append(dict(id=len(nodes), kind=kind, point=list(map(float, p))))
        return len(nodes) - 1

    start_node = {}
    for b in branches:
        p, r = np.array(b["points"]), np.array(b["radius"])
        if b["parent"] >= 0:
            first = start_node[b["id"]]
        else:                                                   # every parentless branch leaves the one root
            if "root" not in start_node:
                start_node["root"] = node_at(p[0], "root")
            first = start_node["root"]
        last_k = len(p) - 1
        ks = sorted(set(k for k, _ in cuts[b["id"]] if 0 < k < last_k))
        at = {0: first}
        a = first
        for k0, k1 in zip([0] + ks, ks + [last_k]):
            z = node_at(p[k1], "junction" if k1 < last_k else "tip")
            at[k1] = z
            seg = p[k0:k1 + 1]
            segments.append(dict(id=len(segments), branch=b["id"], a=a, b=z, points=np.round(seg, 3).tolist(),
                                 radius=r[k0:k1 + 1].tolist(),
                                 length_mm=float(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1)))))
            a = z
        for k, child in cuts[b["id"]]:
            k = min(max(k, 0), last_k)
            start_node[child] = at[k]
            if k == last_k:
                nodes[at[k]]["kind"] = "junction"                 # a child continues from the tip
    return nodes, segments


def _ray_to_wall(m, geometry, origin, u, step: float = 0.1, max_mm: float = 60.0) -> float:
    t = np.arange(0, max_mm, step)
    v = sample(m, geometry, origin[None] + t[:, None] * u[None])
    out = np.nonzero(v <= 0)[0]
    return float(t[out[0]]) if len(out) else max_mm


def protrusion(m, geometry, parent_points, k: int, tip, spacing: float, stations=(-4, -3, 3, 4),
               skip_turns: bool = False):
    """How far ``tip`` lies beyond the parent's wall, measured across the parent's axis at point k.

    The direction u is the tip's offset from the junction with the parent's tangent removed; the
    wall is the median ray length from the parent's axis along u at a few stations ``spacing`` mm
    apart on both sides of the junction (clear of the junction itself). Returns (protrusion, the
    tip's offset across the axis, the wall distance); a tip straight along the parent returns
    +inf (nothing to prune). With ``skip_turns``, a station where the parent's direction (the
    chord over two spacings either way) is more than 30° from that direction at the junction is
    skipped: a ray across the junction's axis would run down the turned parent and read no wall.
    With no station left there is nothing to measure against, and the branch stays (+inf).
    :func:`prune_by_wall` asks for it at the root, where all the stations lie on one side, a few
    root radii down the first branch."""
    P = np.asarray(parent_points, float)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    a, b = P[max(k - 3, 0)], P[min(k + 3, len(P) - 1)]
    T = (b - a) / (np.linalg.norm(b - a) + 1e-12)
    v = np.asarray(tip, float) - P[k]
    u = v - (v @ T) * T
    du = float(np.linalg.norm(u))
    if du < 1e-6:
        return np.inf, 0.0, 0.0
    u /= du
    def at(sj):
        return np.array([np.interp(sj, s, P[:, c]) for c in range(3)])

    def chord(sj):                          # the parent's direction over two spacings either way
        t = at(min(sj + 2 * spacing, s[-1])) - at(max(sj - 2 * spacing, 0.0))
        return t / max(float(np.linalg.norm(t)), 1e-12)

    here = chord(s[k])
    walls = []
    for st in stations:
        sj = s[k] + st * spacing
        if 0 <= sj <= s[-1]:
            if skip_turns and abs(chord(sj) @ here) < STATION_COS:
                continue                    # the parent turns there: a ray along u would run down it
            walls.append(_ray_to_wall(m, geometry, at(sj), u))
    if not walls:
        return np.inf, du, 0.0
    w = float(np.median(walls))
    return du - w, du, w


def prune_by_wall(branches: list[dict], m, geometry, relative: float = 1.0, floor: float = 1.0):
    """Drop terminal branches whose tip does not reach beyond the parent's wall by
    max(``relative`` x the branch's median radius, ``floor`` mm), repeatedly (a parent left terminal
    is tested in turn). Returns (the kept branches renumbered, parents first; how many were dropped).

    A branch leaving the root has no parent: it is tested against the root's first branch (the
    longest), at its start - unless its tip lies behind that start, along the first branch's axis,
    by more than both the wall distance and its offset across the axis. It then continues the axis
    the other way (the root is the deepest point, often mid-tube, and in the pulmonary arteries the
    trunk leaves it backward), where a lobe across a flat lumen reaches no farther back than about
    the lumen's half-width, and mostly sideways."""
    alive = {b["id"] for b in branches}
    first = next((b["id"] for b in branches if b["parent"] < 0), None)
    changed = True
    while changed:
        changed = False
        kids = {i: 0 for i in alive}
        for b in branches:
            if b["id"] in alive and b["parent"] in alive:
                kids[b["parent"]] += 1
        for b in branches:
            i = b["id"]
            if i not in alive or i == first or kids[i] > 0:
                continue
            par = branches[b["parent"] if b["parent"] >= 0 else first]
            P = np.array(par["points"])
            tip = np.array(b["points"][-1])
            k = _junction_index(par, b) if b["parent"] >= 0 else 0
            r = np.array(b["radius"])
            r_tip = float(np.median(r[r > 0])) if (r > 0).any() else 0.5
            rj = max(float(np.array(par["radius"])[k]), 0.5)
            p, across, wall = protrusion(m, geometry, P, k, tip, spacing=rj, skip_turns=b["parent"] < 0)
            need = max(relative * r_tip, floor)
            if b["parent"] < 0:
                ahead = P[min(3, len(P) - 1)] - P[0]
                n = float(np.linalg.norm(ahead))
                if n > 0 and -((tip - P[0]) @ ahead) / n > max(wall, across):
                    continue
            if p < need:
                alive.discard(i)
                changed = True
    keep = [b for b in branches if b["id"] in alive]
    remap = {b["id"]: n for n, b in enumerate(keep)}
    out = [dict(b, id=remap[b["id"]], parent=remap.get(b["parent"], -1)) for b in keep]
    return out, len(branches) - len(keep)
