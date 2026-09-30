"""Named bronchopulmonary segments from the airway tree, and their catchments.

    cd explorations/catchments && ../../../haversack/.venv/bin/python segments.py [RUN]

The airway tree is thalweg's centerlines of `lung_airways` (explorations/_thalweg.py, traced in
memory). A rule-based prototype, not an atlas matcher:

1. Lobar bronchi. Walking only through SIGNIFICANT children (>= 3 tips), which steps over
   zero-length segments and single-tip side twigs: trachea -> carina -> the main bronchi (left =
   larger x in LPS). Below each main bronchus, the 3 (right) or 2 (left) largest subtrees,
   assigned by POSITION, not branching order: the highest is the upper lobe; on the right the more
   anterior of the other two is the middle lobe. Branching order failed here: the graph put the
   middle lobe's takeoff above the upper lobe's, a few millimeters apart.
2. Segments within each lobe. Anatomy defines a segment as a subtree of one segmental bronchus, one
   of the first real branches below the lobar bronchus. So: split the lobe's largest significant
   subtree into its significant children, repeatedly, until the lobe has as many subtrees as
   it has segments (large common trunks such as the basal trunk split first). Then name the
   subtrees by POSITION WITHIN THE LOBE (tip centroid minus the lobe's tip centroid), one name each,
   by optimal assignment against template directions. Measuring direction from the lobar origin
   instead fails: the origin is at the hilum, so every tip of a lobe points lateral. Right: RUL B1
   apical, B2 posterior, B3 anterior; RML B4 lateral, B5 medial; RLL B6 superior, B7 medial basal,
   B8 anterior basal, B9 lateral basal, B10 posterior basal. Left: LUL B1+2 apicoposterior, B3
   anterior, B4 superior lingular, B5 inferior lingular; LLL B6 superior, B7+8 anteromedial basal,
   B9 lateral basal, B10 posterior basal. 18 segments, the count anatomy texts give (10 + 8).
   Side twigs that belong to no chosen subtree take the segment of their nearest labeled sibling.
3. Catchments: every lung voxel to its nearest named airway source (terminal branches, Strahler
   <= 2), walls = left/right lung. Checks: lobe volumes against the store's own lobe map; how the
   arterial territories fall into the segments; veins against the segment boundaries.
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import rankfield as rf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _catch import run, LOBES, tree, ancestor_at, nearest_groups, supply_mask, wall_of, DATA
from _thalweg import centerlines, crop_field                            # noqa: E402

PURE, SIG, H = 0.75, 3, 1.5
U = lambda v: np.asarray(v, float) / np.linalg.norm(v)
# Where each segment sits WITHIN its lobe, relative to the lobe's center. LPS: +x = patient left,
# +y = posterior, +z = superior; right lung lateral = -x, left lung lateral = +x.
TEMPLATES = {
    "RUL": {"RB1 apical": U([0, 0, 1]), "RB2 posterior": U([0, 1, -0.2]), "RB3 anterior": U([0, -1, -0.2])},
    "RML": {"RB4 lateral": U([-1, 0.3, 0]), "RB5 medial": U([1, -0.3, 0])},
    "RLL": {"RB6 superior": U([0, 0.5, 1]), "RB7 medial basal": U([1, -0.2, -0.6]), "RB8 anterior basal": U([-0.2, -1, -0.5]),
            "RB9 lateral basal": U([-1, 0, -0.5]), "RB10 posterior basal": U([0, 1, -0.6])},
    "LUL": {"LB1+2 apicoposterior": U([0, 0.5, 1]), "LB3 anterior": U([0, -1, 0.3]),
            "LB4 superior lingular": U([0.6, -0.6, -0.6]), "LB5 inferior lingular": U([-0.2, -0.6, -1])},
    "LLL": {"LB6 superior": U([0, 0.5, 1]), "LB7+8 anteromedial basal": U([-0.5, -1, -0.5]),
            "LB9 lateral basal": U([1, 0, -0.5]), "LB10 posterior basal": U([0, 1, -0.6])},
}
LOBE_NAME = {"RUL": "right upper", "RML": "right middle", "RLL": "right lower", "LUL": "left upper", "LLL": "left lower"}

# ---- the airway graph
G = centerlines(run, "lung_airways")
S = G["segments"]; NODE = {n["id"]: np.array(n["point"], float) for n in G["nodes"]}
kids = defaultdict(list)
for s in S:
    kids[s["a"]].append(s["id"])
_tips = {}


def tips(sid):
    if sid not in _tips:
        ch = kids.get(S[sid]["b"], [])
        _tips[sid] = [S[sid]["b"]] if not ch else [t for c in ch for t in tips(c)]
    return _tips[sid]


centroid = lambda sids: np.mean([NODE[t] for s in sids for t in tips(s)], 0)


def follow(sid, sig=SIG):
    """Walk down a bronchus through single significant children; return (chain, significant children at the split)."""
    chain = [sid]
    while True:
        sc = [c for c in kids.get(S[chain[-1]]["b"], []) if len(tips(c)) >= sig]
        if len(sc) != 1:
            return chain, sc
        chain.append(sc[0])


root = min((s for s in S if s["a"] == next(n["id"] for n in G["nodes"] if n["kind"] == "root")), key=lambda s: -len(tips(s["id"])))
trachea, main = follow(root["id"])
assert len(main) >= 2, "no carina found"
main = sorted(main, key=lambda c: centroid([c])[0])
rmb, lmb = main[0], main[-1]                                          # smallest x = patient right
def split_into(head, n):
    """The n largest significant subtrees below a bronchus, splitting the largest first - so the
    result does not depend on the ORDER of nearby branch points, which the centerline graph can
    get wrong over a few millimeters (on C3N-00704 it puts the middle lobe's takeoff above the
    upper lobe's)."""
    parts = [head]
    while len(parts) < n:
        big = max(parts, key=lambda c: len(tips(c)))
        _, sc = follow(big)
        if len(sc) < 2:
            break
        parts.remove(big); parts.extend(sc)
    return parts


# lobes by POSITION: right - the highest subtree is the upper lobe, the more anterior of the other two
# the middle lobe; left - the higher of two is the upper lobe
r = sorted(split_into(rmb, 3), key=lambda c: -centroid([c])[2])
lobes = {"RUL": [r[0]]}
rest = sorted(r[1:], key=lambda c: centroid([c])[1])
lobes["RML"], lobes["RLL"] = [rest[0]], rest[1:]
l_ = sorted(split_into(lmb, 2), key=lambda c: -centroid([c])[2])
lobes["LUL"], lobes["LLL"] = [l_[0]], l_[1:]
origin = {k: NODE[S[v[0]]["a"]] for k, v in lobes.items()}

# ---- segments: the first real branches below each lobar bronchus, named by position in the lobe
from scipy.optimize import linear_sum_assignment
label = {}                                                            # airway segment id -> (lobe, segment or None)
split_count = defaultdict(int)
confidence = {}


def subtree(sid):
    out, st = [], [sid]
    while st:
        j = st.pop(); out.append(j); st.extend(kids.get(S[j]["b"], []))
    return out


for lobe, heads in lobes.items():
    names = list(TEMPLATES[lobe]); M = np.stack([TEMPLATES[lobe][n] for n in names])
    for h in heads:                                                   # the lobar chains carry the lobe, not a segment
        for j in subtree(h):
            label[j] = (lobe, None)
    # a lobe whose airways the model follows only briefly has segmental branches of 1-2 tips:
    # relax the significance threshold for that lobe only, and say so
    for sig in (SIG, 2, 1):
        parts = list(heads)
        while len(parts) < len(names):
            big = max(parts, key=lambda c: len(tips(c)))
            chain, sc = follow(big, sig)
            if len(sc) < 2:
                break
            parts.remove(big); parts.extend(sc)
        if len(parts) >= 2:
            break
    if sig != SIG:
        print(f"  {LOBE_NAME[lobe]}: significance threshold relaxed to {sig} tip(s) to find {len(parts)} segmental branches")
    if len(parts) < 2:
        print(f"  {LOBE_NAME[lobe]}: no branching below the lobar bronchus - its segments cannot be named")
        continue
    center = centroid(parts)
    V = np.stack([U(centroid([c]) - center) for c in parts])
    cost = -(V @ M.T)
    ri, ci = linear_sum_assignment(cost)
    for r, c in zip(ri, ci):
        for j in subtree(parts[r]):
            label[j] = (lobe, names[c])
        split_count[names[c]] += 1
        confidence[names[c]] = float(-cost[r, c])
    # side twigs that hang off the chains: the segment of the nearest labeled tip in the lobe
    named_tips = [(NODE[S[j]["b"]], label[j][1]) for h in heads for j in subtree(h)
                  if label[j][1] and not kids.get(S[j]["b"])]
    if named_tips:
        tr = cKDTree(np.stack([p for p, _ in named_tips]))
        for h in heads:
            for j in subtree(h):
                if label[j][1] is None and not kids.get(S[j]["b"]):
                    label[j] = (lobe, named_tips[tr.query(NODE[S[j]["b"]])[1]][1])
seg_names = [n for lobe in TEMPLATES for n in TEMPLATES[lobe]]
tips_of = defaultdict(int)
for sid, (lobe, n) in label.items():
    if n is not None and not kids.get(S[sid]["b"]):
        tips_of[n] += 1
print(f"{run}: airway tree {len(S)} segments, {sum(len(tips(h)) for v in lobes.values() for h in v)} tips below the lobar bronchi")
for lobe in TEMPLATES:
    got = [n for n in TEMPLATES[lobe] if tips_of[n]]
    print(f"  {LOBE_NAME[lobe]:12s} ({sum(len(tips(h)) for h in lobes[lobe]):3d} tips): " +
          ", ".join(f"{n.split()[0]} {tips_of[n]} tips/{split_count[n]} br" for n in TEMPLATES[lobe] if tips_of[n]) +
          ("" if len(got) == len(TEMPLATES[lobe]) else f"   MISSING: {[n for n in TEMPLATES[lobe] if not tips_of[n]]}"))
print("  naming confidence (cosine between a subtree's position in its lobe and its template): " +
      ", ".join(f"{n.split()[0]} {confidence[n]:.2f}" for n in [k for lb in TEMPLATES for k in TEMPLATES[lb]] if n in confidence))

# ---- catchments of the named segments
code0 = crop_field(run)
D0, o0 = np.asarray(code0.geometry.directions, float), np.asarray(code0.geometry.origin, float)
lobe_m = np.stack([rf.margin(code0, code0.labels.index(v)).astype(np.float32) for v in LOBES])


def lobe_at(P):
    q = ((P - o0) @ np.linalg.inv(D0)).T
    m = np.stack([ndi.map_coordinates(x, q, order=1, mode="constant", cval=-8.0) for x in lobe_m])
    return np.where(m.max(0) > 0, m.argmax(0) + 1, 0)


lung3 = np.argwhere(lobe_m.max(0) > 0) @ D0 + o0
b0 = lung3.min(0) - 3
axes = [np.arange(b0[a], lung3.max(0)[a] + 3, H) for a in range(3)]
shape = tuple(len(a) for a in axes)
X = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
store_lobe = lobe_at(X); wall = wall_of(store_lobe)
L = np.nonzero(wall > 0)[0]; XL, WL = X[L], wall[L]

_, aorder, aparent, asites, aseg = tree("lung_airways")
keep = supply_mask(aorder, aseg) & np.array([s in label and label[s][1] is not None for s in aseg])
sites, sseg = asites[keep], aseg[keep]
slab = np.array([seg_names.index(label[s][1]) for s in sseg])
swall = wall_of(lobe_at(sites))
seg_of = np.full(len(XL), -1); gap = np.full(len(XL), np.nan)
for w in (1, 2):
    v = np.nonzero(WL == w)[0]; own = np.nonzero(swall == w)[0]
    if len(v) and len(own):
        g, d, _ = nearest_groups(XL[v], sites[own], slab[own], 2)
        seg_of[v] = g[:, 0]; gap[v] = d[:, 1] - d[:, 0]
vol = np.bincount(seg_of[seg_of >= 0], minlength=len(seg_names)) * H**3 / 1e3
lobe_of_seg = {n: lobe for lobe in TEMPLATES for n in TEMPLATES[lobe]}
print("segment catchment volumes (mL): " + ", ".join(f"{n.split()[0]} {vol[i]:.0f}" for i, n in enumerate(seg_names)))
store_names = list(LOBES.values())
key = {"LUL": "left upper", "LLL": "left lower", "RUL": "right upper", "RML": "right middle", "RLL": "right lower"}
print("lobes from the airway segments vs the store's own lobe map (3 mm total_fast), mL and Dice:")
for lobe in TEMPLATES:
    mine = np.isin(seg_of, [seg_names.index(n) for n in TEMPLATES[lobe]])
    theirs = store_lobe[L] == store_names.index(key[lobe]) + 1
    dice = 2 * (mine & theirs).sum() / max(mine.sum() + theirs.sum(), 1)
    print(f"  {LOBE_NAME[lobe]:12s} airways {mine.sum() * H**3 / 1e3:5.0f}   store {theirs.sum() * H**3 / 1e3:5.0f}   Dice {dice:.2f}")

# ---- arterial territories inside the named segments
_, order, parent, art_sites, art_seg = tree("lung_arteries")
ak = supply_mask(order, art_seg); art_sites, art_seg = art_sites[ak], art_seg[ak]
aw = wall_of(lobe_at(art_sites))
for k in (4, 5):
    anc = ancestor_at(order, parent, k)
    lab = np.array([anc[s] for s in art_seg])
    part = np.full(len(XL), -1)
    for w in (1, 2):
        v = np.nonzero(WL == w)[0]; own = np.nonzero(aw == w)[0]
        part[v] = lab[own][cKDTree(art_sites[own]).query(XL[v])[1]]
    ok = (part >= 0) & (seg_of >= 0)
    groups = np.unique(part[ok])
    pur = [np.bincount(seg_of[ok][part[ok] == gid]).max() / np.sum(part[ok] == gid) for gid in groups]
    sizes = np.array([np.sum(part[ok] == gid) for gid in groups])
    print(f"arterial Strahler >= {k} ({len(groups)} territories): volume-weighted share of each territory inside ONE named "
          f"segment {np.sum(np.array(pur) * sizes) / sizes.sum():.0%} (median territory {np.median(pur):.0%})")

# ---- veins against the named segment boundaries (the anatomy's statement), matched on distance from an airway
Sv, _, _, vsites, _ = tree("lung_veins")
vw = wall_of(lobe_at(vsites))
rng = np.random.default_rng(0)
rnd = XL[rng.choice(len(XL), 40000)] + rng.uniform(-H / 2, H / 2, (40000, 3)); rw = wall_of(lobe_at(rnd))


def gaps(P, Pw):
    gp = np.full(len(P), np.nan); d1 = np.full(len(P), np.nan)
    for w in (1, 2):
        v = np.nonzero(Pw == w)[0]; own = np.nonzero(swall == w)[0]
        if len(v) and len(np.unique(slab[own])) > 1:
            _, d, _ = nearest_groups(P[v], sites[own], slab[own], 2)
            gp[v] = d[:, 1] - d[:, 0]; d1[v] = d[:, 0]
    return gp, d1


gv, dv = gaps(vsites, vw); gr, dr = gaps(rnd, rw)
ov, orr = np.isfinite(gv), np.isfinite(gr)
bins = np.arange(0, max(np.nanmax(dr), np.nanmax(dv)) + 0.5, 0.5)
hv, _ = np.histogram(dv[ov], bins); hr, _ = np.histogram(dr[orr], bins)
wt = (hv / hv.sum()) / np.maximum(hr / hr.sum(), 1e-12)
ix = np.clip(np.digitize(dr[orr], bins) - 1, 0, len(wt) - 1); wi = wt[ix] * (hr[ix] > 0)
o = np.argsort(gr[orr]); c = np.cumsum(wi[o]); mmed = gr[orr][o][np.searchsorted(c, c[-1] / 2)]
print(f"veins vs NAMED SEGMENT boundaries: gap median veins {np.median(gv[ov]):.2f} mm, random matched on distance from an "
      f"airway {mmed:.2f} mm; within 2 mm: veins {np.mean(gv[ov] < 2):.0%}, matched {np.sum(wi * (gr[orr] < 2)) / wi.sum():.0%}")

np.savez_compressed(DATA / f"{run}_segments.npz", names=np.array(seg_names), lung_index=L, seg_of=seg_of, gap=gap,
                    origin=b0, spacing=H, shape=np.array(shape),
                    airway_label=np.array([[sid, seg_names.index(n) if n else -1] for sid, (_, n) in label.items()]))

# ---- figure: coronal and two sagittal slices at 0.5 mm, named segments
FIG_H = 0.5
pal = {"RB1": "#e6194b", "RB2": "#f58231", "RB3": "#ffe119", "RB4": "#3cb44b", "RB5": "#46f0f0", "RB6": "#4363d8",
       "RB7": "#911eb4", "RB8": "#f032e6", "RB9": "#bcf60c", "RB10": "#fabebe",
       "LB1+2": "#e6194b", "LB3": "#ffe119", "LB4": "#3cb44b", "LB5": "#46f0f0", "LB6": "#4363d8",
       "LB7+8": "#f032e6", "LB9": "#bcf60c", "LB10": "#fabebe"}
col = np.array([matplotlib.colors.to_rgb(pal[n.split()[0]]) for n in seg_names])


def slice_labels(P):
    wl = wall_of(lobe_at(P)); out = np.full(len(P), -1)
    for w in (1, 2):
        v = np.nonzero(wl == w)[0]; own = np.nonzero(swall == w)[0]
        if len(v) and len(own):
            out[v] = slab[own][cKDTree(sites[own]).query(P[v])[1]]
    return out


x_right = XL[WL == 2][:, 0].mean(); x_left = XL[WL == 1][:, 0].mean()     # wall 1 = left lung, 2 = right
views = [("coronal", 1, 90.0), ("sagittal, right lung", 0, x_right), ("sagittal, left lung", 0, x_left)]
fig, axs = plt.subplots(1, 3, figsize=(27, 10))
for ax, (title, axis, pos) in zip(axs, views):
    if axis == 1:
        u = np.arange(axes[0][0], axes[0][-1], FIG_H); v = np.arange(axes[2][0], axes[2][-1], FIG_H)
        UU, VV = np.meshgrid(u, v); P = np.stack([UU.ravel(), np.full(UU.size, pos), VV.ravel()], 1)
        xl = "x (LPS, mm): patient right <-> left"
    else:
        u = np.arange(axes[1][0], axes[1][-1], FIG_H); v = np.arange(axes[2][0], axes[2][-1], FIG_H)
        UU, VV = np.meshgrid(u, v); P = np.stack([np.full(UU.size, pos), UU.ravel(), VV.ravel()], 1)
        xl = "y (LPS, mm): anterior <-> posterior"
    lab = slice_labels(P).reshape(UU.shape)
    img = np.where((lab >= 0)[..., None], col[np.clip(lab, 0, None)], 1.0)
    edge = (lab >= 0) & (ndi.maximum_filter(lab, 3) != ndi.minimum_filter(lab, 3))
    img[edge] = 0.15
    ax.imshow(img, origin="lower", extent=(u[0], u[-1], v[0], v[-1]), interpolation="nearest")
    for i in np.unique(lab[lab >= 0]):
        yy, xx = np.nonzero(lab == i)
        if len(xx) > 400:
            ax.text(u[int(np.median(xx))], v[int(np.median(yy))], seg_names[i].split()[0], ha="center", va="center",
                    fontsize=11, weight="bold")
    ax.set_title(f"{title} ({'y' if axis == 1 else 'x'} = {pos:.0f} mm)", fontsize=13); ax.set_xlabel(xl); ax.set_ylabel("z (mm)")
fig.suptitle(f"{run}: bronchopulmonary segments named from the airway tree, catchments of their terminal airways "
             "(prototype, direction templates)", fontsize=15)
plt.tight_layout(); out = DATA / f"{run}_segments.png"; plt.savefig(out, dpi=110)
print(f"wrote {out}")
