"""Cut one arterial subtree out of the field for vmtk, with seeds from our own centerline graph.

    python bench/vessels/vmtk_prep.py [RUN] [MIN_TIPS] [MAX_TIPS]

Uses the segment graph written by centerline.py. Picks the segment in the patient's LEFT lung
(LPS +x of the trunk) whose downstream subtree has the most tips within [MIN_TIPS, MAX_TIPS].
The region is the union of balls (radius + 1.5 mm) around the subtree's centerline, from one
radius past the segment's start; the margin is floored outside it, and marching cubes of the
margin at 0 gives the closed surface vmtk sees - the field's own zero set, never smoothed.
Seeds: the source on the first segment, every downstream tip as a target. For each target our
own path from the source is stored, densified to 0.1 mm, for the route-by-route comparison.
Writes DATA/<RUN>_vmtk_input.npz.
"""
import json, sys
import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
from _data import DATA
from _field import fine_field
from cases import store

run = sys.argv[1] if len(sys.argv) > 1 else "C3N-00704_ctpa0625"
lo_tips, hi_tips = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (15, 60)
G = json.load(open(DATA / f"{run}_lung_arteries_centerlines.json"))
N, S = G["nodes"], G["segments"]
out_of = {}                                                    # node -> segments leaving it (away from the root)
for s in S:
    out_of.setdefault(s["a"], []).append(s["id"])
def downstream(sid):
    res, stack = [], [sid]
    while stack:
        j = stack.pop(); res.append(j); stack.extend(out_of.get(S[j]["b"], []))
    return res
root_x = G["root"][0]
best = None
for s in S:
    ds = downstream(s["id"])
    tips = [j for j in ds if N[S[j]["b"]]["kind"] == "tip"]
    if lo_tips <= len(tips) <= hi_tips and np.mean(np.array(s["points"])[:, 0]) > root_x + 15:
        if best is None or len(tips) > len(best[2]):
            best = (s["id"], ds, tips)
s0, ds, tips = best
print(f"{run}: subtree below segment {s0}: {len(ds)} segments, {len(tips)} tips")


def dense(p, r=None, step=0.1):
    p = np.array(p); s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    t = np.arange(0, s[-1] + 1e-9, step)
    q = np.stack([np.interp(t, s, p[:, a]) for a in range(3)], 1)
    return (q, np.interp(t, s, np.array(r))) if r is not None else q


# the first segment starts one radius (at least 1 mm) + 1 mm past its junction
p0 = np.array(S[s0]["points"]); c0 = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p0, axis=0), axis=1))]
k0 = int(np.argmax(c0 >= max(S[s0]["radius"][0], 1.0) + 1.0))
seg_pts = {j: (np.array(S[j]["points"])[k0:] if j == s0 else np.array(S[j]["points"])) for j in ds}
seg_r = {j: (np.array(S[j]["radius"])[k0:] if j == s0 else np.array(S[j]["radius"])) for j in ds}
src = seg_pts[s0][0]

import time
_t = time.time()
margins, labels, grid, clip = fine_field(store(run))
t_decode = time.time() - _t
m = margins["lung_arteries"]
P = np.concatenate([seg_pts[j] for j in ds]); R = np.concatenate([seg_r[j] for j in ds])
lo = np.floor(grid.to_index(P).min(0) - 12).astype(int).clip(0)
hi = np.ceil(grid.to_index(P).max(0) + 12).astype(int).clip(max=np.array(m.shape) - 1)
sub = m[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
ii = np.stack(np.meshgrid(*[np.arange(a, b + 1) for a, b in zip(lo, hi)], indexing="ij"), -1).reshape(-1, 3)
dist, k = cKDTree(P).query(grid.to_world(ii))
region = (dist < R[k].clip(0.5) + 1.5).reshape(sub.shape)
f = np.pad(np.where(region, sub, -clip), 1, constant_values=-clip)
_t = time.time()
verts, faces, _, _ = marching_cubes(f, level=0.0)
t_mc = time.time() - _t
world = grid.to_world(verts - 1 + lo)

parent_seg = {}
for j in ds:
    for c in out_of.get(S[j]["b"], []):
        parent_seg[c] = j
targets, paths = [], []
for t in tips:
    chain, j = [], t
    while True:
        chain.append(j)
        if j == s0: break
        j = parent_seg[j]
    paths.append(np.concatenate([dense(seg_pts[j]) for j in reversed(chain)]))
    targets.append(seg_pts[t][-1])
dp = [dense(seg_pts[j], seg_r[j]) for j in ds]
np.savez(DATA / f"{run}_vmtk_input.npz", verts=world, faces=faces, source=src, targets=np.array(targets),
         region_field=f[1:-1, 1:-1, 1:-1], region_lo=lo, clip=clip,
         ours_points=np.concatenate([q for q, _ in dp]), ours_radius=np.concatenate([r for _, r in dp]),
         ours_segment=np.concatenate([np.full(len(q), j) for j, (q, _) in zip(ds, dp)]),
         path_points=np.concatenate(paths), path_len=np.array([len(p) for p in paths]))
print(f"surface {len(world)} vertices / {len(faces)} triangles; source {src.round(1)}; {len(targets)} targets "
      f"-> {DATA / f'{run}_vmtk_input.npz'}")
print("TIMING " + __import__("json").dumps({k: round(float(v), 4) for k, v in ({"decode": t_decode, "surface_for_vmtk": t_mc}).items()}))
