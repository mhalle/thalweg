"""Vascular catchments as a rankfield: prototype on C3N-00704's pulmonary arteries.

    cd explorations/catchments && ../../../haversack/.venv/bin/python catchments.py [RUN]

Every lung voxel is ranked by its distance to the arterial branches, so the winner is the
branch that supplies it, rank 2 the territory next door, and the gap d2 - d1 how far behind it
is. That is a rankfield: classes = branch groups, "logits" = -distance / TAU.

- Groups: the tree trimmed at Strahler order 3. A segment of order >= 3 is its own class; a
  twig (order 1-2) belongs to its nearest ancestor of order >= 3. Distances are taken to EVERY
  point of the full tree and minimized within each group, so the twigs still extend their
  parent's reach into the periphery.
- Walls: a voxel only sees sites in its own lobe (lobes from the store's crop layer,
  total_fast at 3 mm, labels 10-14), so no territory crosses a fissure.
- Distance: Euclidean to the centerline (sites every 0.5 mm). Prototype choice; the tube
  function and flow weighting are the obvious alternatives.
- Grid: 1.5 mm, axis-aligned in LPS, over the lungs. Class 0 = outside the lungs.

Then: a real rankfield.RankField from the rankings; haversack's distance and junction layers
from its planes (the numpy references in ranked_build, as haversack says to use locally);
the one plane rankfield lacks, d1 (distance to the supplying vessel); coarser levels by
rf.decode_groups, checked against a direct recomputation; and the anatomical test that
pulmonary veins run between arterial territories.
"""
import json, sys, time, zlib
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import rankfield as rf
from rankfield.code import byte_of_gap, levels, rank_dtype
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "vessels"))
import _field                                                          # noqa: E402
from _data import DATA                                                 # noqa: E402
from cases import store                                                # noqa: E402
from haversack import ranked_build                                     # noqa: E402

from _catch import run, H, STEP, TAU, DEPTH, TRIM, LOBES, T, tick, tree, ancestor_at, nearest_groups, SUPPLY, WALLS, supply_mask, wall_of  # noqa: E402

t0 = time.time()
# ---- lobes from the crop layer (3 mm total_fast), sampled onto a 1.5 mm grid over the lungs
root = _field.open_store(Path(store(run)), "r").root
code0 = _field.parts_of(root)[0].field
geo = code0.geometry
D0, o0 = np.asarray(geo.directions, float), np.asarray(geo.origin, float)
lobe_m = np.stack([rf.margin(code0, code0.labels.index(v)).astype(np.float32) for v in LOBES])
lung3 = np.argwhere(lobe_m.max(0) > 0) @ D0 + o0
b0 = lung3.min(0) - 3; b1 = lung3.max(0) + 3
axes = [np.arange(b0[a], b1[a], H) for a in range(3)]
shape = tuple(len(a) for a in axes)
X = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
q = ((X - o0) @ np.linalg.inv(D0)).T
lm = np.stack([ndi.map_coordinates(m, q, order=1, mode="constant", cval=-8.0) for m in lobe_m])
lobe = np.where(lm.max(0) > 0, lm.argmax(0) + 1, 0)                   # 0 = outside, 1..5 = LOBES order
lobe = wall_of(lobe)                                                  # the walls in force (WALLS)
t0 = tick("lobes", t0)

# ---- arterial sites, grouped at Strahler >= TRIM, each assigned to the lobe it runs in
S, order, parent, sites, site_seg = tree("lung_arteries")
keep = supply_mask(order, site_seg); sites, site_seg = sites[keep], site_seg[keep]
anc = ancestor_at(order, parent, TRIM)
group_seg = sorted(set(anc[s_] for s_ in np.unique(site_seg)))         # class c+1 <-> group_seg[c]
cls_of = {g: c for c, g in enumerate(group_seg)}
site_cls = np.array([cls_of[anc[s]] for s in site_seg])
qs = ((sites - o0) @ np.linalg.inv(D0)).T
sm = np.stack([ndi.map_coordinates(m, qs, order=1, mode="constant", cval=-8.0) for m in lobe_m])
site_lobe = wall_of(np.where(sm.max(0) > 0, sm.argmax(0) + 1, 0))
K = len(group_seg)
print(f"supply: {SUPPLY}, walls: {WALLS}")
print(f"{run}: arteries {len(S)} segments -> {K} classes at Strahler >= {TRIM} (+ 'outside'); "
      f"{len(sites)} sites every {STEP} mm, {np.mean(site_lobe > 0):.0%} inside a lobe; "
      f"grid {shape} at {H} mm, {np.mean(lobe > 0):.1%} lung ({np.sum(lobe > 0) * H**3 / 1e3:.0f} mL)")

# ---- rank every lung voxel by distance to the groups, lobe by lobe
N = len(X)
Gr = np.full((N, DEPTH), -1); Dr = np.full((N, DEPTH), np.inf); Pr = np.full((N, DEPTH), -1)
for L in range(1, 6):
    vox = np.nonzero(lobe == L)[0]; own = np.nonzero(site_lobe == L)[0]
    if len(vox) == 0 or len(own) == 0:
        continue
    dep = min(DEPTH, len(np.unique(site_cls[own])))                   # a small lobe may hold fewer groups
    g, d, p = nearest_groups(X[vox], sites[own], site_cls[own], dep)
    Gr[vox, :dep], Dr[vox, :dep], Pr[vox, :dep] = g, d, own[p]
t0 = tick("rank", t0)

# ---- the rankfield: classes 0 = outside, 1..K = groups; logits -d/TAU
meta = {"mode": "ranked", "version": "0.4", "classes": K + 1, "depth": DEPTH, "clip": 32.0,
        "gap_curve": "log", "gap_range": 64.0, "gap_origin": 0.5, "keep": "nearest", "exhaustive": False,
        "units": f"logit = -distance / {TAU} mm"}
inside = lobe > 0
ranks = np.zeros((DEPTH, N), rank_dtype(K + 1))
ranks[0, ~inside] = 1                                                 # 'outside' wins outside
ranks[:, inside] = np.where(Gr[inside] >= 0, Gr[inside] + 2, 0).T      # group c -> class c+1 -> stored c+2; 0 absent
gap = (Dr - Dr[:, :1]) / TAU
support = np.zeros((DEPTH - 1, N), np.uint8)
byte = np.clip(np.round(byte_of_gap(np.where(np.isfinite(gap), gap, 0), meta)), 1, 255)
support[:, inside] = np.where(np.isfinite(gap[inside, 1:]), byte[inside, 1:], 0).T.astype(np.uint8)
ranks = ranks.reshape((DEPTH,) + shape); support = support.reshape((DEPTH - 1,) + shape)
code = rf.RankField(ranks=ranks, support=support, tail=None, meta=meta, labels=list(range(K + 1)))
d1 = np.where(inside, Dr[:, 0], 0).reshape(shape).astype(np.float32)  # the plane rankfield lacks
lut = levels(meta)
t0 = tick("encode", t0)

# ---- haversack's distance and junction layers, from the planes alone
TRUNC = 20.0
dist_q = ranked_build.distance_field(ranks, support, meta["clip"], (H, H, H), TRUNC, levels=lut)
dist = np.where(dist_q > 0, (1 - dist_q / 255.0) * TRUNC, np.inf)
t0 = tick("distance", t0)
junction, pair = ranked_build.junction_field(ranks, support, meta["clip"], (H, H, H), TRUNC, levels=lut)
t0 = tick("junction", t0)

# check the distance layer against the geometry: near a boundary between point sites the
# territory edge is the bisector of the two nearest sites, (dj^2 - d1^2) / (2 |pj - p1|)
deep = inside & (ndi.distance_transform_edt(inside.reshape(shape)).ravel() * H > 4)
bis = np.min([(Dr[:, j] ** 2 - Dr[:, 0] ** 2) / (2 * np.maximum(np.linalg.norm(sites[Pr[:, j]] - sites[Pr[:, 0]], axis=1), 1e-6))
              for j in range(1, DEPTH)], axis=0)
chk = deep & (bis < 5) & np.isfinite(dist.ravel())
e = dist.ravel()[chk] - bis[chk]
print(f"distance layer vs the bisector distance, {chk.sum()} voxels within 5 mm of a boundary (4 mm from the pleura): "
      f"median {np.median(e):+.3f} mm, |.| median {np.median(np.abs(e)):.3f}, p90 {np.percentile(np.abs(e), 90):.3f}")
print(f"junction layer: {np.sum(junction > 0)} voxels ({np.mean(junction > 0):.2%}) around triple lines")

# ---- hierarchy: coarser levels as groups of classes, against a direct recomputation
lev = {}
for k in (4, 5):
    anc_k = ancestor_at(order, parent, k)
    top = sorted(set(anc_k[g] for g in group_seg))
    members = [[0]] + [[cls_of[g] + 1 for g in group_seg if anc_k[g] == t] for t in top]
    m = rf.decode_groups(code, members).cpu().numpy().reshape(len(members), -1)
    win = m.argmax(0)
    # direct: sites relabeled to the level-k groups, nearest two
    site_k = np.array([top.index(anc_k[group_seg[c]]) for c in site_cls])
    Gd = np.full(N, -1); gapd = np.full(N, np.inf)
    for L in range(1, 6):
        vox = np.nonzero(lobe == L)[0]; own = np.nonzero(site_lobe == L)[0]
        if len(vox) == 0: continue
        if len(np.unique(site_k[own])) < 2:
            Gd[vox] = site_k[own][0]; continue
        g, d, _ = nearest_groups(X[vox], sites[own], site_k[own], 2)
        Gd[vox] = g[:, 0]; gapd[vox] = d[:, 1] - d[:, 0]
    same = (win[inside] - 1) == Gd[inside]
    mm = m[win, np.arange(N)][inside] * TAU                             # decoded group margin, mm of gap
    exact = gapd[inside] < 3.0                                          # near a level-k boundary
    err = np.abs(mm[exact] - gapd[inside][exact])
    print(f"level Strahler >= {k}: {len(top)} regions; decode_groups winner = direct recomputation at {same.mean():.4%} of lung voxels; "
          f"group margin vs direct gap within 3 mm of a boundary: |err| median {np.median(err):.4f} mm, "
          f"p99 {np.percentile(err, 99):.3f} ({np.mean(err > 0.05):.2%} off by > 0.05 mm: the first out-of-group class was not in the top {DEPTH})")
    lev[k] = (win.reshape(shape), gapd, top, site_k, anc_k)
t0 = tick("levels", t0)

# ---- anatomy: do pulmonary veins run between arterial territories?
Sv, order_v, _, vsites, vseg = tree("lung_veins")
qv = ((vsites - o0) @ np.linalg.inv(D0)).T
vl = np.stack([ndi.map_coordinates(m, qv, order=1, mode="constant", cval=-8.0) for m in lobe_m])
vlobe = wall_of(np.where(vl.max(0) > 0, vl.argmax(0) + 1, 0))
rng = np.random.default_rng(0)
lung_idx = np.nonzero(inside)[0]
rnd = X[rng.choice(lung_idx, 40000)] + rng.uniform(-H / 2, H / 2, (40000, 3))
rq = ((rnd - o0) @ np.linalg.inv(D0)).T
rl = np.stack([ndi.map_coordinates(m, rq, order=1, mode="constant", cval=-8.0) for m in lobe_m])
rlobe = wall_of(np.where(rl.max(0) > 0, rl.argmax(0) + 1, 0))
small_vein = np.array([order_v[s] <= 2 for s in vseg])


def gaps_at(P, P_lobe, site_lab):
    """(gap d2 - d1, d1) per point: to the nearest two groups, and to the nearest artery."""
    out = np.full(len(P), np.nan); near = np.full(len(P), np.nan)
    for L in range(1, 6):
        v = np.nonzero(P_lobe == L)[0]; own = np.nonzero(site_lobe == L)[0]
        if len(v) == 0 or len(np.unique(site_lab[own])) < 2: continue
        _, d, _ = nearest_groups(P[v], sites[own], site_lab[own], 2)
        out[v] = d[:, 1] - d[:, 0]; near[v] = d[:, 0]
    return out, near


def matched_median(g_r, d_r, d_v, width=0.5):
    """Median of the random points' gaps, reweighted to the veins' distance-from-artery histogram."""
    bins = np.arange(0, max(d_r.max(), d_v.max()) + width, width)
    hv, _ = np.histogram(d_v, bins); hr, _ = np.histogram(d_r, bins)
    w = (hv / max(hv.sum(), 1)) / np.maximum(hr / hr.sum(), 1e-12)
    wi = w[np.clip(np.digitize(d_r, bins) - 1, 0, len(w) - 1)] * (hr[np.clip(np.digitize(d_r, bins) - 1, 0, len(hr) - 1)] > 0)
    o = np.argsort(g_r); c = np.cumsum(wi[o])
    within1 = np.sum(wi * (g_r < 1)) / wi.sum()
    return g_r[o][np.searchsorted(c, c[-1] / 2)], within1


print("pulmonary veins vs random lung points: gap d2 - d1 to the nearest two arterial territories (mm; 0 = on the boundary)")
for name, lab in ((f"Strahler >= {TRIM} ({K} regions)", site_cls), ("Strahler >= 4", lev[4][3]), ("Strahler >= 5", lev[5][3])):
    (gv, nv), (gr, nr) = gaps_at(vsites, vlobe, lab), gaps_at(rnd, rlobe, lab)
    ok_v = np.isfinite(gv) & (vlobe > 0); ok_r = np.isfinite(gr)
    gs = gv[ok_v & small_vein]
    mmed, mw1 = matched_median(gr[ok_r], nr[ok_r], nv[ok_v])
    print(f"  {name:28s} veins median {np.median(gv[ok_v]):5.2f} (small veins, order <= 2: {np.median(gs):5.2f}), "
          f"random {np.median(gr[ok_r]):5.2f}, random matched to the veins' distance from an artery {mmed:5.2f};  "
          f"within 1 mm of a boundary: veins {np.mean(gv[ok_v] < 1):.0%}, random {np.mean(gr[ok_r] < 1):.0%}, matched {mw1:.0%}")
    if lab is site_cls:
        print(f"    (distance from the nearest artery: veins median {np.median(nv[ok_v]):.2f} mm, random {np.median(nr[ok_r]):.2f} mm)")
t0 = tick("veins", t0)

# ---- sizes: raw and deflated (a store would use zstd; zlib is a fair stand-in for the ratio)
def sz(a):
    return a.nbytes / 1e6, len(zlib.compress(np.ascontiguousarray(a).tobytes(), 6)) / 1e6
planes = {"ranks": ranks, "support": support, "distance": dist_q, "junction": junction,
          "d1 (uint8, 0.1 mm)": np.clip(np.round(d1 * 10), 0, 255).astype(np.uint8)}
print("planes, MB raw -> deflated: " + ", ".join(f"{k} {a:.1f} -> {b:.2f}" for k, (a, b) in ((k, sz(v)) for k, v in planes.items())))
print("seconds: " + ", ".join(f"{k} {v:.1f}" for k, v in T.items()))
np.savez_compressed(DATA / f"{run}_catchments_{SUPPLY}_{WALLS}.npz", ranks=ranks, support=support, distance=dist_q, junction=junction,
                    pair=pair, d1=d1, lobe=lobe.reshape(shape), origin=b0, spacing=H, group_segment=np.array(group_seg))

# ---- figure: a coronal slice, territories at Strahler >= 5 and >= 3, veins near the slice
yk = int(np.argmax(inside.reshape(shape).sum((0, 2))))
fig, axs = plt.subplots(1, 3, figsize=(18, 7))
cm = np.random.default_rng(3).uniform(0.25, 1.0, (K + 2, 3))
ins = inside.reshape(shape)[:, yk, :].T
for a_, (title, lab) in zip(axs[:2], ((f"Strahler >= 5: {len(lev[5][2])} regions", lev[5][0][:, yk, :]),
                                       (f"Strahler >= {TRIM}: {K} regions", ranks[0][:, yk, :].astype(int) - 1))):
    img = np.where(ins[..., None], cm[np.clip(lab.T, 0, K + 1)], 1.0)
    edge = ins & ((np.diff(lab.T, axis=0, prepend=lab.T[:1]) != 0) | (np.diff(lab.T, axis=1, prepend=lab.T[:, :1]) != 0))
    img[edge] = 0.15
    a_.imshow(img, origin="lower", extent=(axes[0][0], axes[0][-1], axes[2][0], axes[2][-1]))
    near = np.abs(vsites[:, 1] - axes[1][yk]) < 2.0
    a_.plot(vsites[near, 0], vsites[near, 2], ".", ms=1.5, color="#1f4e9c", label="vein centerlines (within 2 mm)")
    a_.set_title(title, fontsize=10); a_.set_xlabel("x (LPS, mm)"); a_.set_ylabel("z (mm)")
axs[0].legend(loc="lower left", fontsize=8, markerscale=6)
dd = np.where(ins, np.minimum(dist[:, yk, :].T, TRUNC), np.nan)
im = axs[2].imshow(dd, origin="lower", cmap="magma", extent=(axes[0][0], axes[0][-1], axes[2][0], axes[2][-1]))
axs[2].set_title("distance layer: mm to the nearest territory boundary (Strahler >= 3)", fontsize=10)
plt.colorbar(im, ax=axs[2], fraction=0.046)
fig.suptitle(f"{run}: pulmonary arterial catchments as a rankfield (coronal slice y = {axes[1][yk]:.0f} mm)")
plt.tight_layout(); plt.savefig(DATA / f"{run}_catchments_{SUPPLY}_{WALLS}.png", dpi=110)
print(f"wrote {DATA / f'{run}_catchments_{SUPPLY}_{WALLS}.png'}")
