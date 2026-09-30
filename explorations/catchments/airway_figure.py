"""Airway vs arterial territories on one coronal slice, evaluated exactly at 0.25 mm.

    cd explorations/catchments && ../../../haversack/.venv/bin/python airway_figure.py [RUN] [Y_MM] [ART_K] [AIR_K]

Same rules as airway_catchments.py (sources = Strahler <= 2 branches, walls = left/right lung);
defaults compare arteries at Strahler >= 5 with airways at Strahler >= 4, the pair that agreed best
(ARI 0.70 vs 0.37 for random partitions of the same grain).
"""
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import rankfield as rf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _catch import run, LOBES, tree, ancestor_at, supply_mask, wall_of, DATA
from _thalweg import crop_field                                         # noqa: E402

RES = 0.25
y = float(sys.argv[2]) if len(sys.argv) > 2 else 90.0
KA = int(sys.argv[3]) if len(sys.argv) > 3 else 5
KB = int(sys.argv[4]) if len(sys.argv) > 4 else 4
code0 = crop_field(run)
D0, o0 = np.asarray(code0.geometry.directions, float), np.asarray(code0.geometry.origin, float)
lobe_m = np.stack([rf.margin(code0, code0.labels.index(v)).astype(np.float32) for v in LOBES])


def wall_at(P):
    q = ((P - o0) @ np.linalg.inv(D0)).T
    m = np.stack([ndi.map_coordinates(x, q, order=1, mode="constant", cval=-8.0) for x in lobe_m])
    return wall_of(np.where(m.max(0) > 0, m.argmax(0) + 1, 0))


lung = np.argwhere(lobe_m.max(0) > 0) @ D0 + o0
xs = np.arange(lung[:, 0].min() - 3, lung[:, 0].max() + 3, RES)
zs = np.arange(lung[:, 2].min() - 3, lung[:, 2].max() + 3, RES)
XX, ZZ = np.meshgrid(xs, zs)
P = np.stack([XX.ravel(), np.full(XX.size, y), ZZ.ravel()], 1)
W = wall_at(P)


def territories(cls, k):
    S, order, parent, sites, seg = tree(cls)
    keep = supply_mask(order, seg); sites, seg = sites[keep], seg[keep]
    sw = wall_at(sites)
    anc = ancestor_at(order, parent, k)
    lab = np.array([anc[s] for s in seg])
    out = np.full(len(P), -1)
    for w in (1, 2):
        v = np.nonzero(W == w)[0]; own = np.nonzero(sw == w)[0]
        if len(v) and len(own):
            out[v] = lab[own][cKDTree(sites[own]).query(P[v])[1]]
    return out.reshape(XX.shape), len(np.unique(lab)), tree(cls)[3]


A, na, _ = territories("lung_arteries", KA)
B, nb, air_pts = territories("lung_airways", KB)
ins = B >= 0
_, bi = np.unique(B, return_inverse=True); bi = bi.reshape(B.shape)
img = np.where(ins[..., None], np.random.default_rng(5).uniform(0.45, 1.0, (bi.max() + 1, 3))[bi], 1.0)
# interior boundaries only: a pixel whose neighborhood holds two different territories, and no
# outside pixel (the lung surface is not a boundary between territories)
inner = ins & (ndi.minimum_filter(np.where(ins, 0, -1), 5) == 0)
edgeB = inner & (ndi.maximum_filter(B, 3) != ndi.minimum_filter(B, 3))
edgeA = inner & (ndi.maximum_filter(A, 5) != ndi.minimum_filter(A, 5))
img[edgeB] = (0.2, 0.2, 0.2)
img[edgeA & ~edgeB] = (0.8, 0.05, 0.05)
img[edgeA & edgeB] = (0.45, 0.0, 0.45)
fig, ax = plt.subplots(figsize=(16, 12))
ext = (xs[0], xs[-1], zs[0], zs[-1])
ax.imshow(img, origin="lower", extent=ext, interpolation="nearest")
near = np.abs(air_pts[:, 1] - y) < 2.0
ax.plot(air_pts[near, 0], air_pts[near, 2], ".", ms=3, color="#1b3a6b", label="airway centerlines (within 2 mm)")
ax.plot([], [], color=(0.8, 0.05, 0.05), lw=3, label=f"arterial territory boundaries (Strahler >= {KA})")
ax.plot([], [], color=(0.2, 0.2, 0.2), lw=3, label=f"airway territory boundaries (Strahler >= {KB}); fill = airway territory")
ax.plot([], [], color=(0.45, 0.0, 0.45), lw=3, label="both, within 0.5 mm")
ax.contour(xs, zs, ins.astype(float), [0.5], colors="0.55", linewidths=0.8)
ax.legend(loc="lower left", fontsize=11, markerscale=4)
ax.set_title(f"{run}: airway vs arterial territories, coronal slice y = {y:.0f} mm, evaluated at {RES} mm", fontsize=14)
ax.set_xlabel("x (LPS, mm): patient right <-> left"); ax.set_ylabel("z (mm)")
plt.tight_layout()
out = DATA / f"{run}_airway_vs_arterial_catchments_hires.png"
plt.savefig(out, dpi=150)
# how close the two sets of interior boundaries run, against a random rotation of the arterial
# labels' boundary (same boundary length, shifted 25 mm along x within the lung) as a null
dA = ndi.distance_transform_edt(~(inner & (ndi.maximum_filter(A, 3) != ndi.minimum_filter(A, 3)))) * RES
near_b = dA[edgeB]
shift = np.roll(dA, int(25 / RES), axis=1)[edgeB]
print(f"wrote {out}")
print("airway interior boundary pixels within d of an arterial interior boundary (null: arterial boundaries shifted 25 mm):")
for d in (1, 2, 3, 5):
    print(f"  d = {d} mm: {np.mean(near_b <= d):.0%} (null {np.mean(shift <= d):.0%})")
print(f"  median distance {np.median(near_b):.1f} mm (null {np.median(shift):.1f} mm)")
