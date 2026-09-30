"""Airway catchments against arterial catchments: do the two trees divide the lungs the same way?

    cd explorations/catchments && ../../../haversack/.venv/bin/python airway_catchments.py [RUN]

After airway_centerlines.py. Anatomy says a bronchopulmonary segment is "supplied by a tertiary
bronchus and its own segmental artery", with veins along its edges - so territories drawn from the
airway tree and from the arterial tree should coincide, and veins should lie on both sets of
boundaries. Same rules for both trees (_catch.py): classes = Strahler >= k subtrees, sources = the
terminal branches (Strahler <= 2), walls = left/right lung, straight-line distance, 1.5 mm grid.

1. Partition agreement at matched region counts: adjusted Rand index (ARI) and normalized mutual
   information (NMI), against a spatial null - partitions from the same number of random seeds
   with the same walls - because any two partitions of similar grain agree somewhat by chance.
2. Veins vs the airway territory boundaries, matched on distance from the nearest airway.
3. Bronchoarterial pairing: arteries run with the bronchi, veins do not.
"""
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import rankfield as rf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _catch import run, LOBES, tree, ancestor_at, nearest_groups, supply_mask, wall_of, DATA
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "vessels"))
import _field                                                          # noqa: E402
from cases import store                                                # noqa: E402

H = 1.5
code0 = _field.parts_of(_field.open_store(Path(store(run)), "r").root)[0].field
D0, o0 = np.asarray(code0.geometry.directions, float), np.asarray(code0.geometry.origin, float)
lobe_m = np.stack([rf.margin(code0, code0.labels.index(v)).astype(np.float32) for v in LOBES])


def wall_at(P):
    q = ((P - o0) @ np.linalg.inv(D0)).T
    m = np.stack([ndi.map_coordinates(x, q, order=1, mode="constant", cval=-8.0) for x in lobe_m])
    return wall_of(np.where(m.max(0) > 0, m.argmax(0) + 1, 0))


lung3 = np.argwhere(lobe_m.max(0) > 0) @ D0 + o0
b0 = lung3.min(0) - 3
axes = [np.arange(b0[a], lung3.max(0)[a] + 3, H) for a in range(3)]
X = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
wall = wall_at(X)
L = np.nonzero(wall > 0)[0]
XL, WL = X[L], wall[L]


def load(cls):
    S, order, parent, sites, seg = tree(cls)
    keep = supply_mask(order, seg)
    return dict(S=S, order=order, parent=parent, all_sites=sites, all_seg=seg,
                sites=sites[keep], seg=seg[keep], wall=wall_at(sites[keep]))


def partition(T, k):
    """Lung voxels -> territory at Strahler >= k (nearest source's group), and the region count."""
    anc = ancestor_at(T["order"], T["parent"], k)
    lab = np.array([anc[s] for s in T["seg"]])
    out = np.full(len(XL), -1)
    for w in (1, 2):
        v = np.nonzero(WL == w)[0]; own = np.nonzero(T["wall"] == w)[0]
        if len(v) and len(own):
            out[v] = lab[own][cKDTree(T["sites"][own]).query(XL[v])[1]]
    return out, len(np.unique(out[out >= 0]))


def random_partition(n, rng):
    out = np.full(len(XL), -1)
    for w in (1, 2):
        v = np.nonzero(WL == w)[0]
        k = max(1, int(round(n * len(v) / len(XL))))
        seeds = XL[rng.choice(v, k, replace=False)]
        out[v] = cKDTree(seeds).query(XL[v])[1] + (0 if w == 1 else 100000)
    return out


def ari_nmi(a, b):
    ok = (a >= 0) & (b >= 0); a, b = a[ok], b[ok]
    _, ai = np.unique(a, return_inverse=True); _, bi = np.unique(b, return_inverse=True)
    C = np.zeros((ai.max() + 1, bi.max() + 1)); np.add.at(C, (ai, bi), 1)
    n = C.sum(); c2 = lambda x: x * (x - 1) / 2
    sij, sa, sb = c2(C).sum(), c2(C.sum(1)).sum(), c2(C.sum(0)).sum()
    exp = sa * sb / c2(n)
    ari = (sij - exp) / (0.5 * (sa + sb) - exp)
    P = C / n; pa, pb = P.sum(1), P.sum(0); nz = P > 0
    mi = (P[nz] * np.log(P[nz] / np.outer(pa, pb)[nz])).sum()
    ha, hb = -(pa * np.log(pa)).sum(), -(pb * np.log(pb)).sum()
    return ari, mi / np.sqrt(ha * hb)


art, air = load("lung_arteries"), load("lung_airways")
print(f"{run}: arteries {len(art['S'])} segments ({len(art['sites'])} source sites); airways {len(air['S'])} segments "
      f"({len(air['sites'])} source sites); lungs {len(XL) * H**3 / 1e3:.0f} mL at {H} mm, walls left/right")
lv_art = {k: partition(art, k) for k in range(3, 8)}
lv_air = {k: partition(air, k) for k in range(2, 8)}
print("regions per level: arteries " + ", ".join(f"S>={k}: {n}" for k, (_, n) in lv_art.items() if n > 1)
      + " | airways " + ", ".join(f"S>={k}: {n}" for k, (_, n) in lv_air.items() if n > 1))

rng = np.random.default_rng(0)
print("partition agreement at matched counts (ARI, NMI; null = random seeds with the same count and walls, mean of 5):")
rows = []
for ka, (pa, na) in lv_art.items():
    if na < 4: continue
    kb = min(lv_air, key=lambda k: abs(np.log(max(lv_air[k][1], 1) / na)))
    pb, nb = lv_air[kb]
    if nb < 4: continue
    ari, nmi = ari_nmi(pa, pb)
    nulls = np.array([ari_nmi(pa, random_partition(nb, rng)) for _ in range(5)])
    rows.append((ka, na, kb, nb, ari, nmi, nulls[:, 0].mean(), nulls[:, 1].mean()))
    print(f"  arteries S>={ka} ({na:3d}) vs airways S>={kb} ({nb:3d}): ARI {ari:.3f} (null {nulls[:, 0].mean():.3f}), "
          f"NMI {nmi:.3f} (null {nulls[:, 1].mean():.3f})")

# veins on the AIRWAY territory boundaries, matched on distance from the nearest airway
Sv, order_v, _, vsites, vseg = tree("lung_veins")
vwall = wall_at(vsites)
rnd = XL[rng.choice(len(XL), 40000)] + rng.uniform(-H / 2, H / 2, (40000, 3))
rwall = wall_at(rnd)


def gap_and_d1(P, Pw, T, k):
    anc = ancestor_at(T["order"], T["parent"], k)
    lab = np.array([anc[s] for s in T["seg"]])
    gap = np.full(len(P), np.nan); d1 = np.full(len(P), np.nan)
    for w in (1, 2):
        v = np.nonzero(Pw == w)[0]; own = np.nonzero(T["wall"] == w)[0]
        if len(v) == 0 or len(np.unique(lab[own])) < 2: continue
        _, d, _ = nearest_groups(P[v], T["sites"][own], lab[own], 2)
        gap[v] = d[:, 1] - d[:, 0]; d1[v] = d[:, 0]
    return gap, d1


def matched(g_r, d_r, d_v, width=0.5):
    bins = np.arange(0, max(d_r.max(), d_v.max()) + width, width)
    hv, _ = np.histogram(d_v, bins); hr, _ = np.histogram(d_r, bins)
    w = (hv / hv.sum()) / np.maximum(hr / hr.sum(), 1e-12)
    ix = np.clip(np.digitize(d_r, bins) - 1, 0, len(w) - 1)
    wi = w[ix] * (hr[ix] > 0)
    o = np.argsort(g_r); c = np.cumsum(wi[o])
    return g_r[o][np.searchsorted(c, c[-1] / 2)], np.sum(wi * (g_r < 1)) / wi.sum()


print("veins vs AIRWAY territory boundaries (gap d2 - d1, mm; random points matched on distance from the nearest airway):")
for k in sorted(lv_air):
    if not 4 <= lv_air[k][1] <= 150: continue
    gv, dv = gap_and_d1(vsites, vwall, air, k); gr, dr = gap_and_d1(rnd, rwall, air, k)
    ov, orr = np.isfinite(gv), np.isfinite(gr)
    mm, w1 = matched(gr[orr], dr[orr], dv[ov])
    print(f"  airways S>={k} ({lv_air[k][1]:3d} regions): veins median {np.median(gv[ov]):5.2f}, matched random {mm:5.2f}, "
          f"unmatched {np.median(gr[orr]):5.2f}; within 1 mm: veins {np.mean(gv[ov] < 1):.0%}, matched {w1:.0%}")

# bronchoarterial pairing: distance from vessel centerline points to the nearest airway centerline
atr = cKDTree(air["all_sites"])
da = atr.query(art["all_sites"])[0]; dv_ = atr.query(vsites)[0]
reach = 15.0
print(f"bronchoarterial pairing, vessel points within {reach:.0f} mm of an airway: distance to the nearest airway centerline "
      f"median arteries {np.median(da[da < reach]):.2f} mm vs veins {np.median(dv_[dv_ < reach]):.2f} mm; within 3 mm: "
      f"arteries {np.mean(da[da < reach] < 3):.0%}, veins {np.mean(dv_[dv_ < reach] < 3):.0%}")

# figure: a coronal slice, airway territories at the level matching arteries S>=5, arterial boundaries over them
ka, na, kb, nb = next((r[:4] for r in rows if r[0] == 5), rows[-1][:4])
pa_full = np.full(len(X), -1); pa_full[L] = lv_art[ka][0]
pb_full = np.full(len(X), -1); pb_full[L] = lv_air[kb][0]
shape = tuple(len(a) for a in axes)
yk = int(np.argmax((wall.reshape(shape) > 0).sum((0, 2))))
A = pa_full.reshape(shape)[:, yk, :].T; B = pb_full.reshape(shape)[:, yk, :].T
ins = B >= 0
_, bi = np.unique(B, return_inverse=True); bi = bi.reshape(B.shape)
cm = np.random.default_rng(5).uniform(0.4, 1.0, (bi.max() + 1, 3))
img = np.where(ins[..., None], cm[bi], 1.0)
edgeA = ins & (ndi.maximum_filter(A, 3) != ndi.minimum_filter(A, 3))
edgeB = ins & (ndi.maximum_filter(B, 3) != ndi.minimum_filter(B, 3))
img[edgeB] = (0.35, 0.35, 0.35)
img[edgeA] = (0.75, 0.05, 0.05)
fig, ax = plt.subplots(figsize=(13, 9))
ext = (axes[0][0], axes[0][-1], axes[2][0], axes[2][-1])
ax.imshow(img, origin="lower", extent=ext, interpolation="nearest")
near = np.abs(air["all_sites"][:, 1] - axes[1][yk]) < 2.0
ax.plot(air["all_sites"][near, 0], air["all_sites"][near, 2], ".", ms=2, color="#1b3a6b", label="airway centerlines (within 2 mm)")
ax.plot([], [], color=(0.75, 0.05, 0.05), label=f"arterial territory boundaries (S>={ka}, {na})")
ax.plot([], [], color=(0.35, 0.35, 0.35), label=f"airway territory boundaries (S>={kb}, {nb}); fill = airway territory")
ax.legend(loc="lower left", fontsize=9, markerscale=5)
ax.set_title(f"{run}: airway vs arterial territories, coronal slice y = {axes[1][yk]:.0f} mm", fontsize=12)
ax.set_xlabel("x (LPS, mm): patient right <-> left"); ax.set_ylabel("z (mm)")
plt.tight_layout(); out = DATA / f"{run}_airway_vs_arterial_catchments.png"; plt.savefig(out, dpi=140)
print(f"wrote {out}")
