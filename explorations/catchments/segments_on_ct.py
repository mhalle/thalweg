"""The artery-named segments over the CT: is the small right upper lobe anatomy or labeling?

    cd explorations/catchments && ../../../haversack/.venv/bin/python segments_on_ct.py [RUN] [Z_AXIAL]

After artery_segments.py. Lung window (-1000..200 HU) on coronal y = 90 mm, the right-lung sagittal
x = -46 mm and one axial slice through the upper lobe, sampled from the original DICOM in world
mm at 0.5 mm; lobe boundaries thick, segment boundaries thin, from the artery-named territories.
"""
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import rankfield as rf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _catch import run, LOBES, tree, supply_mask, wall_of, DATA
from _thalweg import crop_field, ct as ct_of                            # noqa: E402

Z_AX = float(sys.argv[2]) if len(sys.argv) > 2 else -60.0
RES = 0.5
A = np.load(DATA / f"{run}_artery_segments.npz"); names = [str(n) for n in A["names"]]
lab = {int(k): int(v) for k, v, _ in A["artery_label"]}
lobe_of = lambda n: ("RUL" if n.split()[0] in ("RB1", "RB2", "RB3") else "RML" if n.split()[0] in ("RB4", "RB5") else
                     "RLL" if n.startswith("RB") else "LUL" if n.split()[0] in ("LB1+2", "LB3", "LB4", "LB5") else "LLL")
lobe_idx = {k: i for i, k in enumerate(["RUL", "RML", "RLL", "LUL", "LLL"])}
seg_lobe = np.array([lobe_idx[lobe_of(n)] for n in names])

code0 = crop_field(run)
D0, o0 = np.asarray(code0.geometry.directions, float), np.asarray(code0.geometry.origin, float)
lobe_m = np.stack([rf.margin(code0, code0.labels.index(v)).astype(np.float32) for v in LOBES])


def wall_at(P):
    q = ((P - o0) @ np.linalg.inv(D0)).T
    m = np.stack([ndi.map_coordinates(x, q, order=1, mode="constant", cval=-8.0) for x in lobe_m])
    return wall_of(np.where(m.max(0) > 0, m.argmax(0) + 1, 0))


S, order, parent, pts, seg = tree("lung_arteries")
ok = supply_mask(order, seg) & np.array([s_ in lab for s_ in seg])
sites, slab = pts[ok], np.array([lab[s_] for s_ in seg[ok]]); swall = wall_at(sites)


def labels(P):
    wl = wall_at(P); out = np.full(len(P), -1)
    for w in (1, 2):
        v = np.nonzero(wl == w)[0]; own = np.nonzero(swall == w)[0]
        if len(v) and len(own):
            out[v] = slab[own][cKDTree(sites[own]).query(P[v])[1]]
    return out


ct = ct_of(run)                                                       # $VESSELS_DICOM, via research/vessels/cases.py
lung = np.argwhere(lobe_m.max(0) > 0) @ D0 + o0
lo, hi = lung.min(0) - 5, lung.max(0) + 5
views = [("coronal y = 90 mm", "xz", 90.0), ("sagittal x = -46 mm (right lung)", "yz", -46.0),
         (f"axial z = {Z_AX:.0f} mm", "xy", Z_AX)]
fig, axs = plt.subplots(1, 3, figsize=(27, 10))
for ax, (title, plane, pos) in zip(axs, views):
    a, b = {"xz": (0, 2), "yz": (1, 2), "xy": (0, 1)}[plane]
    c = 3 - a - b
    u = np.arange(lo[a], hi[a], RES); v = np.arange(lo[b], hi[b], RES)
    UU, VV = np.meshgrid(u, v)
    P = np.zeros((UU.size, 3)); P[:, a] = UU.ravel(); P[:, b] = VV.ravel(); P[:, c] = pos
    hu = ct(P).reshape(UU.shape)
    L = labels(P).reshape(UU.shape)
    Lo = np.where(L >= 0, seg_lobe[np.clip(L, 0, None)], -1)
    ax.imshow(np.clip((hu + 1000) / 1200, 0, 1), cmap="gray", origin="lower", extent=(u[0], u[-1], v[0], v[-1]))
    inside = L >= 0
    segedge = inside & (ndi.maximum_filter(L, 3) != ndi.minimum_filter(L, 3))
    lobedge = inside & (ndi.maximum_filter(Lo, 5) != ndi.minimum_filter(Lo, 5))
    ov = np.zeros(L.shape + (4,))
    ov[segedge] = (1.0, 0.85, 0.0, 0.8)
    ov[lobedge] = (1.0, 0.1, 0.1, 1.0)
    ax.imshow(ov, origin="lower", extent=(u[0], u[-1], v[0], v[-1]))
    for i in np.unique(L[L >= 0]):
        yy, xx = np.nonzero(L == i)
        if len(xx) > 1500:
            ax.text(u[int(np.median(xx))], v[int(np.median(yy))], names[i].split()[0], color="cyan", ha="center",
                    va="center", fontsize=11, weight="bold")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel({"xz": "x (LPS): patient right <-> left", "yz": "y (LPS): anterior <-> posterior",
                   "xy": "x (LPS): patient right <-> left"}[plane])
    ax.set_ylabel({"xz": "z (mm)", "yz": "z (mm)", "xy": "y (LPS): anterior (down) <-> posterior (up)"}[plane])
fig.suptitle(f"{run}: CT (lung window) with the artery-named segments; lobe boundaries red, segment boundaries yellow",
             fontsize=15)
plt.tight_layout(); out = DATA / f"{run}_segments_on_ct.png"; plt.savefig(out, dpi=100)
print(f"wrote {out}")
