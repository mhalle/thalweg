"""Seed-free centerline graph of one vessel class from a ranked store (TEASAR-style minimal paths).

    python bench/vessels/centerline.py RUN [lung_arteries|lung_veins]

1. Distance: every interior voxel's distance to the margin's sub-voxel zero crossings (a KD tree
   over the crossings, so exact to the crossing sampling and unbounded - the stored 2-voxel
   `distance` layer cannot see the medial axis of anything wider than ~4 voxels).
2. Paths: Dijkstra from the root (the deepest point, the trunk) over the lattice graph whose
   joins the FIELD decides (GRAPH=field, default: faces, face saddles, cell interiors - _topo.py);
   GRAPH=voxel keeps the old 26-connected labelmap graph for comparison. Over it,
   edge cost = length * mean(1 / (d + eps)^2) - vmtk's integral of ds / R, sharpened, on the
   voxel grid instead of the Voronoi diagram.
3. Branches without seeds: repeatedly take the uncovered voxel farthest from the root along the
   tree (path length), walk the minimal path back to the skeleton so far, and cover every voxel
   within SCALE * d + CONST mm of the new path. Terminal branches shorter than 2 * (radius at
   their junction) + 1 mm are pruned as spurs.
4. Radius: each path point is moved to the largest inscribed ball within +-0.5 mm (inside the
   margin), whose radius is the point's radius.
Writes DATA/<RUN>_<class>_centerlines.json: branches (world points, radius, parent, generation).
"""
import json, sys, time
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from _data import DATA
from _field import fine_field, crossings, sample, inscribed_radius
from cases import store
from _topo import field_edges, voxel_edges, components, OFFSETS

SCALE, CONST, EPS = 1.5, 1.0, 0.1
import os
GRAPH = os.environ.get("GRAPH", "field")          # "field": the interpolant decides connectivity;
                                                   # "voxel": the old 26-connected labelmap graph
run = sys.argv[1] if len(sys.argv) > 1 else "C3N-00704_ctpa0625"
cls = sys.argv[2] if len(sys.argv) > 2 else "lung_arteries"
t0 = time.time()
def tick(msg): print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

margins, labels, grid, clip = fine_field(store(run))
m = margins[cls]
t_decoded = time.time() - t0
tick("decoded")
code = {"lung_arteries": 3, "lung_veins": 4}[cls]
# nodes are the same either way (a lattice point is inside iff its class wins, i.e. m > 0); the
# graph mode decides which neighbors are joined
if GRAPH == "field":
    idx_all, r_all, c_all, k_all, tun = field_edges(m)
else:
    idx_all, r_all, c_all, k_all = voxel_edges(labels == code, 26); tun = 0
ncomp, comp = components(len(idx_all), r_all, c_all)
main = np.argmax(np.bincount(comp))
keep_node = comp == main
new_id = -np.ones(len(idx_all), np.int64); new_id[keep_node] = np.arange(keep_node.sum())
e_ok = keep_node[r_all]                                             # both ends share a component
idx = idx_all[keep_node]
rows, cols = new_id[r_all[e_ok]], new_id[c_all[e_ok]]
lens = np.array([np.linalg.norm(o @ grid.dirs) for o in OFFSETS])[k_all[e_ok]]
mask = np.zeros(labels.shape, bool); mask[tuple(idx.T)] = True
world = grid.to_world(idx)
X = crossings(m, grid)
xtree = cKDTree(X)
d = xtree.query(world)[0]
tick(f"{cls} [{GRAPH} graph]: main component {len(idx)} of {len(idx_all)} nodes ({ncomp} components), "
     f"{len(X)} crossings, max distance {d.max():.1f} mm, {tun} cell-interior joins")
w = 1.0 / (d + EPS) ** 2
cost = lens * 0.5 * (w[rows] + w[cols])
N = len(idx)
Gc = csr_matrix((np.r_[cost, cost], (np.r_[rows, cols], np.r_[cols, rows])), shape=(N, N))
Gl = csr_matrix((np.r_[lens, lens], (np.r_[rows, cols], np.r_[cols, rows])), shape=(N, N))
root = int(np.argmax(d))
_, pred = dijkstra(Gc, indices=root, return_predecessors=True)
daf = dijkstra(Gl, indices=root)
tick(f"graph {len(rows)} edges; root at {np.round(world[root], 1)} (d = {d[root]:.1f} mm); "
     f"farthest path length {daf[np.isfinite(daf)].max():.0f} mm")

ntree = cKDTree(world)
valid = np.isfinite(daf)
in_skel = np.zeros(N, bool); in_skel[root] = True
branch_of = -np.ones(N, np.int64)
branches = []                                                       # (nodes target->junction, parent branch)


def cover(nodes):
    for r in np.unique(np.round(SCALE * d[nodes] + CONST, 1)):
        sel = nodes[np.round(SCALE * d[nodes] + CONST, 1) == r]
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
    in_skel[newp] = True; branch_of[newp] = len(branches)
    branches.append((path, parent))
    cover(newp)
    valid[t] = False
    score[~valid] = -1.0
tick(f"TEASAR: {len(branches)} raw branches")

# spur pruning: terminal branches shorter than 2 R(junction) + 1 mm
def plen(path):
    return float(np.sum(np.linalg.norm(np.diff(world[path], axis=0), axis=1)))
children = {i: 0 for i in range(len(branches))}
for _, par in branches:
    if par >= 0: children[par] += 1
alive = []
for i, (path, par) in enumerate(branches):
    if children[i] == 0 and par >= 0 and plen(path) < 2 * d[path[-1]] + 1.0:
        continue
    alive.append(i)
tick(f"pruned {len(branches) - len(alive)} spurs, {len(alive)} branches remain")

# ridge refinement and radius on the kept branch points
inside = lambda q: sample(m, grid, q) > 0
out, gen = [], {}
remap = {old: new for new, old in enumerate(alive)}
for i in alive:
    path, par = branches[i]
    while par >= 0 and par not in remap:                        # a pruned parent cannot occur, guard anyway
        par = branches[par][1]
    pts = world[path][::-1]                                      # junction -> tip
    r, q = inscribed_radius(pts, xtree, inside)
    if GRAPH == "field":
        # a join the field makes can curve around a face saddle, so a straight chord between two
        # joined points may clip the outside: insert the midpoint of any chord that leaves the
        # vessel, move it to the local ridge (inside by construction), repeat
        for _ in range(4):
            mids, where_ = [], []
            for k in range(len(q) - 1):
                t = np.linspace(0, 1, 11)[1:-1, None]
                if (sample(m, grid, q[k] + t * (q[k + 1] - q[k])) <= 0).any():
                    mids.append(0.5 * (q[k] + q[k + 1])); where_.append(k + 1)
            if not mids:
                break
            rm, qm = inscribed_radius(np.array(mids), xtree, inside, step=0.35, n=7)
            q = np.insert(q, where_, qm, axis=0); r = np.insert(r, where_, rm)
    gen[i] = 0 if par < 0 else gen[par] + 1
    out.append(dict(id=remap[i], parent=remap.get(par, -1), generation=gen[i],
                    points=np.round(q, 3).tolist(), radius=np.round(r, 3).tolist(),
                    lattice_points=np.round(pts, 3).tolist(),
                    length_mm=round(plen(path), 2)))
tick("radius refined")

# The graph. A TEASAR branch runs from its junction on the parent to its OWN tip, and its
# children attach along it; split every branch at its children's junctions, so segments run
# between nodes (the root, junctions, tips) - vmtk's branch splitting has the same shape.
cuts = {b["id"]: [] for b in out}
for b in out:
    if b["parent"] >= 0:
        par = out[b["parent"]]
        k = int(np.argmin(np.linalg.norm(np.array(par["points"]) - np.array(b["points"][0]), axis=1)))
        cuts[b["parent"]].append((k, b["id"]))
nodes, segments = [], []
def node_at(p, kind):
    nodes.append(dict(id=len(nodes), kind=kind, point=list(map(float, p)))); return len(nodes) - 1
start_node = {}
for b in out:                                                   # ids are in creation order: parents first
    p, r = np.array(b["points"]), np.array(b["radius"])
    if b["parent"] >= 0:
        first = start_node[b["id"]]
    else:                                                       # every parentless branch leaves the one root
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
            nodes[at[k]]["kind"] = "junction"                     # a child continues from the tip
deg = np.bincount([s["a"] for s in segments] + [s["b"] for s in segments], minlength=len(nodes))
kinds = np.array([n["kind"] for n in nodes])
L = np.array([s["length_mm"] for s in segments]); R = np.concatenate([b["radius"] for b in out])
print(f"\n{run} {cls}: {len(out)} TEASAR branches -> graph of {len(segments)} segments, "
      f"{(kinds == 'tip').sum()} tips, {(kinds == 'junction').sum()} junction nodes "
      f"(degree 3: {((kinds == 'junction') & (deg == 3)).sum()}, >3: {((kinds == 'junction') & (deg > 3)).sum()})")
print(f"  total centerline {L.sum()/10:.1f} cm; segment length median {np.median(L):.1f} mm; "
      f"radius along the centerlines p10/50/90: {np.percentile(R[R > 0], [10, 50, 90]).round(2)} mm")
gens = np.bincount([b["generation"] for b in out])
print("  branches by attachment depth:", " ".join(f"{g}:{c}" for g, c in enumerate(gens) if c))
# the labelmap skeleton's own topology, for contrast: endpoints and junction voxels
sk = skeletonize(mask).astype(np.uint8)
nb = ndi.convolve(sk, np.ones((3, 3, 3), np.uint8), mode="constant") - 1
print(f"  labelmap skeleton of the same component: {int(((nb == 1) & (sk == 1)).sum())} endpoints, "
      f"{int(((nb >= 3) & (sk == 1)).sum())} junction voxels")
dest = DATA / f"{run}_{cls}_centerlines{'' if GRAPH == 'field' else '_voxel'}.json"
json.dump(dict(run=run, cls=cls, graph=GRAPH, root=world[root].round(3).tolist(), scale=SCALE, const=CONST,
               branches=out, nodes=nodes, segments=segments), open(dest, "w"))
tick(f"wrote {dest}")
print("TIMING " + __import__("json").dumps({k: round(float(v), 4) for k, v in ({"decode": t_decoded, "centerlines_whole_tree": time.time() - t0 - t_decoded, "branches": len(alive)}).items()}))
