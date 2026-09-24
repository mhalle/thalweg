"""vmtk's branch partition without a surface: its tube function evaluated anywhere in the field.

    python bench/vessels/branch_partition.py [RUN]          (after vmtk_branch.py)

vmtkBranchClipper labels SURFACE points with the non-bifurcation centerline group whose tube
function is lowest (vtkvmtkPolyDataCenterlineGroupsClipper), and the tube function is a point-wise
formula (vtkvmtkPolyBallLine::EvaluateFunction): per centerline segment, the closest point in the
Minkowski sense (4-vectors (x, r), dot product x.y - r_x r_y), t clamped to [0, 1], value
|x - c(t)|^2 - r(t)^2, minimized over segments; blanked (bifurcation) cells take no part.

Here that formula is ported (torch) and evaluated
1. at the clipper's own output points - it must reproduce vmtk's labels, no surface operations;
2. over the vessel VOLUME (every lattice point inside the field) - the partition as a label field;
3. for both centerline sets (vmtk's and ours), compared with each other at the same points,
   with the bifurcation reference systems vmtk computed from each set (centerline-only).
"""
import sys, time
import numpy as np
import torch
from scipy.spatial import cKDTree
from _data import DATA
from _field import fine_field
from cases import store

run = sys.argv[1] if len(sys.argv) > 1 else "C3N-00704_ctpa0625"
dev = "mps" if torch.backends.mps.is_available() else "cpu"


def load(name):
    B = np.load(DATA / f"{run}_vmtk_branch_{name}.npz")
    cells = np.split(B["cell_ids"], np.cumsum(B["cell_len"])[:-1])
    return B, cells


def segments(B, cells):
    """Non-blanked centerline segments: p0, p1, r0, r1, group per segment."""
    P, R = B["split_points"], B["split_radius"]
    p0, p1, r0, r1, g = [], [], [], [], []
    for c, gid, bl in zip(cells, B["cell_group"], B["cell_blank"]):
        if bl == 1 or len(c) < 2:
            continue
        p0.append(P[c[:-1]]); p1.append(P[c[1:]]); r0.append(R[c[:-1]]); r1.append(R[c[1:]])
        g.append(np.full(len(c) - 1, gid))
    return tuple(np.concatenate(a) for a in (p0, p1, r0, r1, g))


def tube_argmin(x, seg, chunk=2048):
    """For points x (N,3): the group of the lowest tube value, and that value (vmtk's PolyBallLine)."""
    p0, p1, r0, r1 = (torch.tensor(a, dtype=torch.float32, device=dev) for a in seg[:4])
    g = torch.tensor(seg[4], dtype=torch.long, device=dev)
    groups = torch.unique(g); gi = torch.searchsorted(groups, g)
    v0 = p1 - p0; dr = r1 - r0
    den = (v0 * v0).sum(1) - dr * dr
    ok = den.abs() > 1e-12
    out_g = np.empty(len(x), int); out_v = np.empty(len(x), np.float32)
    X = torch.tensor(x, dtype=torch.float32, device=dev)
    for a in range(0, len(x), chunk):
        xx = X[a:a + chunk]
        v1 = xx[:, None, :] - p0[None]                                  # (n, S, 3)
        num = (v1 * v0[None]).sum(-1) + dr[None] * r0[None]              # complex dot: - (dr)(0 - r0)
        t = (num / torch.where(ok, den, torch.ones_like(den))[None]).clamp(0, 1)
        c = p0[None] + t[..., None] * v0[None]
        rc = r0[None] + t * dr[None]
        val = ((xx[:, None, :] - c) ** 2).sum(-1) - rc * rc
        val = torch.where(ok[None], val, torch.full_like(val, float("inf")))
        per_group = torch.full((len(xx), len(groups)), float("inf"), device=dev)
        per_group.scatter_reduce_(1, gi[None].expand(len(xx), -1), val, reduce="amin")
        v, k = per_group.min(1)
        out_g[a:a + chunk] = groups[k].cpu().numpy(); out_v[a:a + chunk] = v.cpu().numpy()
    return out_g, out_v


Z = np.load(DATA / f"{run}_vmtk_input.npz")
res = {}
for name in ("vmtk", "ours"):
    B, cells = load(name)
    seg = segments(B, cells)
    t0 = time.time()
    lab, _ = tube_argmin(B["clip_points"], seg)
    dt = time.time() - t0
    agree = (lab == B["clip_group"]).mean()
    print(f"[{name} centerlines] field-native tube labels vs vmtkBranchClipper at its {len(lab)} output points: "
          f"{agree:.2%} identical ({dt:.1f} s here vs {B['seconds'][1]:.0f} s for the clipper)")
    res[name] = (B, cells, seg)
    print("TIMING " + __import__("json").dumps({k: round(float(v), 4) for k, v in ({f"partition_surface_{name}": dt}).items()}))

# the partition as a VOLUME: every lattice point inside the artery field in the subtree's region
margins, labels, grid, clip = fine_field(store(run))
m = margins["lung_arteries"]
lo = np.floor(grid.to_index(Z["verts"]).min(0)).astype(int) - 1
hi = np.ceil(grid.to_index(Z["verts"]).max(0)).astype(int) + 1
sub = m[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
inside = np.argwhere(sub > 0) + lo
pts = grid.to_world(inside)
near = cKDTree(Z["ours_points"]).query(pts)[0] < 4.0                  # this subtree's vessels only
pts = pts[near]
t0 = time.time()
vol_lab, vol_val = tube_argmin(pts, res["ours"][2])
t_vol = time.time() - t0
print(f"[ours] volume partition: {len(pts)} lattice points inside the subtree labeled in {time.time() - t0:.1f} s; "
      f"{len(np.unique(vol_lab))} groups; {np.mean(vol_val < 0):.1%} of the points lie inside their group's tube")

# ours vs vmtk: map each of our groups to vmtk's by centerline overlap, compare labels at the same points
Bv, cv, segv = res["vmtk"]; Bo, co, sego = res["ours"]
tv = cKDTree(0.5 * (segv[0] + segv[1]))
votes = {}
for mid, g in zip(0.5 * (sego[0] + sego[1]), sego[4]):
    votes.setdefault(g, []).append(segv[4][tv.query(mid)[1]])
mapping = {g: np.bincount(v).argmax() for g, v in votes.items()}
query = Z["verts"]
lab_v, _ = tube_argmin(query, segv)
lab_o, _ = tube_argmin(query, sego)
same = np.array([mapping.get(a, -1) == b for a, b in zip(lab_o, lab_v)])
print(f"ours vs vmtk centerlines: {len(mapping)} of our groups map to {len(set(mapping.values()))} vmtk groups; "
      f"the wall points labeled the same group: {same.mean():.1%}")
# where they differ: near bifurcations? distance from each differing point to the nearest vmtk bifurcation origin
if (~same).any() and len(Bv["ref_origin"]):
    dref = cKDTree(Bv["ref_origin"]).query(query)[0]
    print(f"   differing points: median {np.median(dref[~same]):.1f} mm from a bifurcation origin "
          f"(agreeing points: {np.median(dref[same]):.1f} mm)")

# bifurcation frames: vmtk's reference systems from each centerline set, matched by origin
Ov, Oo = Bv["ref_origin"], Bo["ref_origin"]
d, j = cKDTree(Ov).query(Oo)
ok = d < 5.0
ang = lambda a, b: np.degrees(np.arccos(np.clip(np.abs((a * b).sum(1)), 0, 1)))
print(f"bifurcation frames: {ok.sum()} of our {len(Oo)} matched within 5 mm of vmtk's {len(Ov)}; origin distance "
      f"median {np.median(d[ok]):.2f} mm, p90 {np.percentile(d[ok], 90):.2f}; normal angle median "
      f"{np.median(ang(Bo['ref_normal'][ok], Bv['ref_normal'][j[ok]])):.1f} deg, up-normal "
      f"{np.median(ang(Bo['ref_upnormal'][ok], Bv['ref_upnormal'][j[ok]])):.1f} deg")
np.savez(DATA / f"{run}_branch_partition.npz", points=pts, group=vol_lab, value=vol_val)
print("TIMING " + __import__("json").dumps({k: round(float(v), 4) for k, v in ({"partition_volume": t_vol}).items()}))
