"""Segment names carried from the airways onto the arteries, and segment catchments from the arteries.

    cd explorations/catchments && ../../../haversack/.venv/bin/python artery_segments.py [RUN]

After segments.py (named airway segments, DATA/<run>_segments.npz). A segmental bronchus has its
own segmental artery running beside it, and the model follows the arteries much farther than the
airways (537 arterial tips against 134 airway tips on C3N-00704; 10 airway tips in the right upper
lobe). So:

1. Pairing: an arterial centerline point (every 0.5 mm) is paired with the nearest NAMED airway
   point if it lies within PAIR_MM and the two run parallel (|cos| >= PAIR_COS).
2. Naming, top down: an arterial subtree whose paired points agree (>= PURE on one segment, and at
   least MIN_PAIRED points) takes that segment as a whole; otherwise its children are decided one
   by one. A subtree with no pairing at all is left for step 3.
3. Inheritance along the tree: every arterial twig still unnamed takes the name of the nearest
   named branch point ALONG THE TREE (multi-source shortest paths over the artery graph), not
   through space - a branch off the right upper lobe artery inherits a right-upper name even where
   a middle-lobe vessel lies closer.
4. Catchments from the named arterial twigs (Strahler <= 2), walls left/right lung; compared with
   the airway-only segments of segments.py: lobe volumes, agreement per segment, and whether veins
   lie on the segment boundaries (anatomy: they are intersegmental).
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
import rankfield as rf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _catch import run, LOBES, tree, nearest_groups, supply_mask, wall_of, DATA
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "vessels"))
import _field                                                          # noqa: E402
from cases import store                                                # noqa: E402

PAIR_MM, PAIR_COS, PURE, MIN_PAIRED, H = 8.0, 0.7, 0.8, 10, 1.5
SEG = np.load(DATA / f"{run}_segments.npz")
names = [str(n) for n in SEG["names"]]
LOBE_OF = {n: ("RUL" if n.split()[0] in ("RB1", "RB2", "RB3") else "RML" if n.split()[0] in ("RB4", "RB5") else
               "RLL" if n.startswith("RB") else "LUL" if n.split()[0] in ("LB1+2", "LB3", "LB4", "LB5") else "LLL")
           for n in names}
LOBE_NAME = {"RUL": "right upper", "RML": "right middle", "RLL": "right lower", "LUL": "left upper", "LLL": "left lower"}


def dense_with_tangents(cls):
    S, order, parent, pts, seg = tree(cls)
    tan = np.zeros_like(pts)
    for s_ in np.unique(seg):
        i = np.nonzero(seg == s_)[0]
        if len(i) > 1:
            t = np.gradient(pts[i], axis=0)
            tan[i] = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    return S, order, parent, pts, seg, tan


# ---- named airway points
_, _, _, apts, aseg, atan = dense_with_tangents("lung_airways")
alab_of = {int(sid): int(k) for sid, k in SEG["airway_label"]}
alab = np.array([alab_of.get(int(s_), -1) for s_ in aseg])
named = alab >= 0
atree = cKDTree(apts[named]); alab_n, atan_n = alab[named], atan[named]

# ---- arterial points, paired with named airways
S, order, parent, pts, seg, tan = dense_with_tangents("lung_arteries")
d, j = atree.query(pts)
paired = (d <= PAIR_MM) & (np.abs((tan * atan_n[j]).sum(1)) >= PAIR_COS)
plab = np.where(paired, alab_n[j], -1)
print(f"{run}: {len(pts)} arterial points, {paired.mean():.1%} paired with a named airway "
      f"(within {PAIR_MM} mm, |cos| >= {PAIR_COS})")

# ---- top-down naming
kids = defaultdict(list)
for s in S:
    kids[s["a"]].append(s["id"])
pts_of = defaultdict(list)
for i, s_ in enumerate(seg):
    pts_of[int(s_)].append(i)
votes_cache = {}


def votes(sid):
    if sid not in votes_cache:
        v = np.bincount(plab[pts_of[sid]][plab[pts_of[sid]] >= 0], minlength=len(names)).astype(float)
        for c in kids.get(S[sid]["b"], []):
            v += votes(c)
        votes_cache[sid] = v
    return votes_cache[sid]


name_of = {}                                                          # arterial segment -> segment index
how = {}


def assign(sid):
    v = votes(sid)
    if v.sum() == 0:
        return                                                        # no airway partner anywhere below
    top = int(v.argmax())
    ch = kids.get(S[sid]["b"], [])
    if v.sum() >= MIN_PAIRED and v[top] / v.sum() >= PURE or not ch:
        st = [sid]
        while st:
            k = st.pop(); name_of[k] = top; how[k] = "paired"; st.extend(kids.get(S[k]["b"], []))
    else:
        for c in ch:
            assign(c)


root = [s["id"] for s in S if s["a"] not in {t["b"] for t in S}]
for r in root:
    assign(r)

# ---- inheritance along the tree for the rest: nearest named branch point by path length
nodes = sorted({s["a"] for s in S} | {s["b"] for s in S}); nid = {n: i for i, n in enumerate(nodes)}
w = np.array([max(s["length_mm"], 1e-3) for s in S])
A = csr_matrix((np.r_[w, w], (np.r_[[nid[s["a"]] for s in S], [nid[s["b"]] for s in S]],
                               np.r_[[nid[s["b"]] for s in S], [nid[s["a"]] for s in S]])), shape=(len(nodes),) * 2)
src_nodes, src_lab = [], []
for sid, k in name_of.items():
    src_nodes += [nid[S[sid]["a"]], nid[S[sid]["b"]]]; src_lab += [k, k]
src_nodes = np.array(src_nodes); src_lab = np.array(src_lab)
dist, _, sources = dijkstra(A, directed=False, indices=src_nodes, min_only=True, return_predecessors=True)
lab_at_node = {n: src_lab[np.nonzero(src_nodes == sources[nid[n]])[0][0]] for n in nodes if sources[nid[n]] >= 0}
for s in S:
    if s["id"] not in name_of and s["a"] in lab_at_node:
        name_of[s["id"]] = int(lab_at_node[s["a"]]); how[s["id"]] = "inherited"

twig = supply_mask(order, seg)
length_by = defaultdict(float)
for s in S:
    if order[s["id"]] <= 2:
        length_by[how.get(s["id"], "unnamed")] += s["length_mm"]
tot = sum(length_by.values())
print("arterial twig length (Strahler <= 2) named by: " + ", ".join(f"{k} {v / tot:.0%}" for k, v in sorted(length_by.items())))
per_seg = defaultdict(lambda: [0.0, 0.0])
for s in S:
    if order[s["id"]] <= 2 and s["id"] in name_of:
        per_seg[names[name_of[s["id"]]]][0 if how[s["id"]] == "paired" else 1] += s["length_mm"] / 10
print("per segment, arterial twig length cm (paired + inherited): " +
      ", ".join(f"{n.split()[0]} {per_seg[n][0]:.0f}+{per_seg[n][1]:.0f}" for n in names))

# ---- lung grid (as segments.py) and catchments of the named arterial twigs
code0 = _field.parts_of(_field.open_store(Path(store(run)), "r").root)[0].field
D0, o0 = np.asarray(code0.geometry.directions, float), np.asarray(code0.geometry.origin, float)
lobe_m = np.stack([rf.margin(code0, code0.labels.index(v)).astype(np.float32) for v in LOBES])


def lobe_at(P):
    q = ((P - o0) @ np.linalg.inv(D0)).T
    m = np.stack([ndi.map_coordinates(x, q, order=1, mode="constant", cval=-8.0) for x in lobe_m])
    return np.where(m.max(0) > 0, m.argmax(0) + 1, 0)


lung3 = np.argwhere(lobe_m.max(0) > 0) @ D0 + o0
b0 = lung3.min(0) - 3
axes = [np.arange(b0[a], lung3.max(0)[a] + 3, H) for a in range(3)]
X = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
store_lobe = lobe_at(X); wall = wall_of(store_lobe)
L = np.nonzero(wall > 0)[0]; XL, WL = X[L], wall[L]
assert np.array_equal(L, SEG["lung_index"]), "the lung grid differs from segments.py's"

ok = twig & np.array([s_ in name_of for s_ in seg])
sites, slab = pts[ok], np.array([name_of[s_] for s_ in seg[ok]])
swall = wall_of(lobe_at(sites))
seg_of = np.full(len(XL), -1); gap = np.full(len(XL), np.nan)
for wl in (1, 2):
    v = np.nonzero(WL == wl)[0]; own = np.nonzero(swall == wl)[0]
    if len(v) and len(own):
        g, dd, _ = nearest_groups(XL[v], sites[own], slab[own], 2)
        seg_of[v] = g[:, 0]; gap[v] = dd[:, 1] - dd[:, 0]
air_of = SEG["seg_of"]
vol = np.bincount(seg_of[seg_of >= 0], minlength=len(names)) * H**3 / 1e3
vol_air = np.bincount(air_of[air_of >= 0], minlength=len(names)) * H**3 / 1e3
print("segment volumes mL, from arteries (from airways): " +
      ", ".join(f"{n.split()[0]} {vol[i]:.0f} ({vol_air[i]:.0f})" for i, n in enumerate(names)))
store_names = list(LOBES.values())
print("lobes, mL (airways only) and Dice with the store's lobe map:")
for lobe in LOBE_NAME:
    idx = [i for i, n in enumerate(names) if LOBE_OF[n] == lobe]
    mine, air = np.isin(seg_of, idx), np.isin(air_of, idx)
    theirs = store_lobe[L] == store_names.index(LOBE_NAME[lobe]) + 1
    dc = lambda a, b: 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)
    print(f"  {LOBE_NAME[lobe]:12s} arteries {mine.sum() * H**3 / 1e3:5.0f} ({air.sum() * H**3 / 1e3:5.0f})   "
          f"Dice store {dc(mine, theirs):.2f} ({dc(air, theirs):.2f})   Dice arteries vs airways {dc(mine, air):.2f}")
both = (seg_of >= 0) & (air_of >= 0)
print(f"voxels given the same segment by the arteries and the airways: {np.mean(seg_of[both] == air_of[both]):.0%}")

# ---- veins against the segment boundaries, matched on distance from the nearest artery
_, _, _, vsites, _ = tree("lung_veins")
vw = wall_of(lobe_at(vsites))
rng = np.random.default_rng(0)
rnd = XL[rng.choice(len(XL), 40000)] + rng.uniform(-H / 2, H / 2, (40000, 3)); rw = wall_of(lobe_at(rnd))


def gaps(P, Pw):
    gp = np.full(len(P), np.nan); d1 = np.full(len(P), np.nan)
    for wl in (1, 2):
        v = np.nonzero(Pw == wl)[0]; own = np.nonzero(swall == wl)[0]
        if len(v) and len(np.unique(slab[own])) > 1:
            _, dd, _ = nearest_groups(P[v], sites[own], slab[own], 2)
            gp[v] = dd[:, 1] - dd[:, 0]; d1[v] = dd[:, 0]
    return gp, d1


gv, dv = gaps(vsites, vw); gr, dr = gaps(rnd, rw)
ov, orr = np.isfinite(gv), np.isfinite(gr)
bins = np.arange(0, max(np.nanmax(dr), np.nanmax(dv)) + 0.5, 0.5)
hv, _ = np.histogram(dv[ov], bins); hr, _ = np.histogram(dr[orr], bins)
wt = (hv / hv.sum()) / np.maximum(hr / hr.sum(), 1e-12)
ix = np.clip(np.digitize(dr[orr], bins) - 1, 0, len(wt) - 1); wi = wt[ix] * (hr[ix] > 0)
o = np.argsort(gr[orr]); c = np.cumsum(wi[o]); mmed = gr[orr][o][np.searchsorted(c, c[-1] / 2)]
print(f"veins vs the segment boundaries (arterial sources): gap median veins {np.median(gv[ov]):.2f} mm, random matched on "
      f"distance from an artery {mmed:.2f} mm; within 2 mm: veins {np.mean(gv[ov] < 2):.0%}, matched "
      f"{np.sum(wi * (gr[orr] < 2)) / wi.sum():.0%}")
np.savez_compressed(DATA / f"{run}_artery_segments.npz", names=np.array(names), lung_index=L, seg_of=seg_of, gap=gap,
                    artery_label=np.array([[k, v, how[k] == "paired"] for k, v in name_of.items()]))

# ---- figure: the same three views as segments.py, arteries (top) against airways (bottom)
FIG_H = 0.5
pal = {"RB1": "#e6194b", "RB2": "#f58231", "RB3": "#ffe119", "RB4": "#3cb44b", "RB5": "#46f0f0", "RB6": "#4363d8",
       "RB7": "#911eb4", "RB8": "#f032e6", "RB9": "#bcf60c", "RB10": "#fabebe",
       "LB1+2": "#e6194b", "LB3": "#ffe119", "LB4": "#3cb44b", "LB5": "#46f0f0", "LB6": "#4363d8",
       "LB7+8": "#f032e6", "LB9": "#bcf60c", "LB10": "#fabebe"}
col = np.array([matplotlib.colors.to_rgb(pal[n.split()[0]]) for n in names])
_, _, _, air_sites_all, air_seg_all = tree("lung_airways")
_, aorder, _, _, _ = tree("lung_airways")
akeep = supply_mask(aorder, air_seg_all) & (np.array([alab_of.get(int(s_), -1) for s_ in air_seg_all]) >= 0)
air_sites, air_slab = air_sites_all[akeep], np.array([alab_of[int(s_)] for s_ in air_seg_all[akeep]])
air_swall = wall_of(lobe_at(air_sites))


def slice_labels(P, S_, L_, W_):
    wl = wall_of(lobe_at(P)); out = np.full(len(P), -1)
    for w_ in (1, 2):
        v = np.nonzero(wl == w_)[0]; own = np.nonzero(W_ == w_)[0]
        if len(v) and len(own):
            out[v] = L_[own][cKDTree(S_[own]).query(P[v])[1]]
    return out


x_right = XL[WL == 2][:, 0].mean(); x_left = XL[WL == 1][:, 0].mean()
views = [("coronal", 1, 90.0), ("sagittal, right lung", 0, x_right), ("sagittal, left lung", 0, x_left)]
fig, axs = plt.subplots(2, 3, figsize=(27, 19))
for row, (tag, S_, L_, W_) in enumerate((("named from the ARTERIES (airway names carried by pairing)", sites, slab, swall),
                                          ("named from the AIRWAYS alone (segments.py)", air_sites, air_slab, air_swall))):
    for ax, (title, axis, pos) in zip(axs[row], views):
        if axis == 1:
            u = np.arange(axes[0][0], axes[0][-1], FIG_H); vv = np.arange(axes[2][0], axes[2][-1], FIG_H)
            UU, VV = np.meshgrid(u, vv); P = np.stack([UU.ravel(), np.full(UU.size, pos), VV.ravel()], 1)
            xl = "x (LPS, mm): patient right <-> left"
        else:
            u = np.arange(axes[1][0], axes[1][-1], FIG_H); vv = np.arange(axes[2][0], axes[2][-1], FIG_H)
            UU, VV = np.meshgrid(u, vv); P = np.stack([np.full(UU.size, pos), UU.ravel(), VV.ravel()], 1)
            xl = "y (LPS, mm): anterior <-> posterior"
        lab = slice_labels(P, S_, L_, W_).reshape(UU.shape)
        img = np.where((lab >= 0)[..., None], col[np.clip(lab, 0, None)], 1.0)
        img[(lab >= 0) & (ndi.maximum_filter(lab, 3) != ndi.minimum_filter(lab, 3))] = 0.15
        ax.imshow(img, origin="lower", extent=(u[0], u[-1], vv[0], vv[-1]), interpolation="nearest")
        for i in np.unique(lab[lab >= 0]):
            yy, xx = np.nonzero(lab == i)
            if len(xx) > 400:
                ax.text(u[int(np.median(xx))], vv[int(np.median(yy))], names[i].split()[0], ha="center", va="center",
                        fontsize=11, weight="bold")
        ax.set_title(f"{title} ({'y' if axis == 1 else 'x'} = {pos:.0f} mm)\n{tag}", fontsize=12)
        ax.set_xlabel(xl); ax.set_ylabel("z (mm)")
fig.suptitle(f"{run}: bronchopulmonary segments, carried from the airways onto the arteries (top) vs airways alone (bottom)",
             fontsize=16)
plt.tight_layout(); out = DATA / f"{run}_artery_segments.png"; plt.savefig(out, dpi=100)
print(f"wrote {out}")
