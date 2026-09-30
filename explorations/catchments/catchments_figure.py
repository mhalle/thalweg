"""The catchment figure at high resolution: one coronal slice evaluated directly at 0.25 mm.

    cd explorations/catchments && ../../../haversack/.venv/bin/python catchments_figure.py [RUN] [Y_MM]

Territories are geometry (nearest branch group within the lobe), so a slice can be computed at
any resolution rather than read from the 1.5 mm store. Same definitions as catchments.py:
groups at Strahler >= 5 and >= 3, lobes as walls (the crop layer's lobe margins, trilinear, so
the lobe edges are smooth), Euclidean distance to the centerline. The right panel is the
distance to the nearest boundary BETWEEN territories, from the two nearest sites of different
groups (the bisector, min over the next three groups; the store's distance layer matched it to
0.16 mm median at 1.5 mm). Unlike the store's layer it does not count the lung surface.
"""
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
import rankfield as rf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _catch import run, TRIM, LOBES, tree, ancestor_at, nearest_groups, DATA, SUPPLY, WALLS, supply_mask, wall_of
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "vessels"))
import _field                                                          # noqa: E402
from cases import store                                                # noqa: E402

RES = 0.25
code0 = _field.parts_of(_field.open_store(Path(store(run)), "r").root)[0].field
D0, o0 = np.asarray(code0.geometry.directions, float), np.asarray(code0.geometry.origin, float)
lobe_m = np.stack([rf.margin(code0, code0.labels.index(v)).astype(np.float32) for v in LOBES])


def lobe_at(P):
    q = ((P - o0) @ np.linalg.inv(D0)).T
    m = np.stack([ndi.map_coordinates(x, q, order=1, mode="constant", cval=-8.0) for x in lobe_m])
    return np.where(m.max(0) > 0, m.argmax(0) + 1, 0)


lung = np.argwhere(lobe_m.max(0) > 0) @ D0 + o0
y = float(sys.argv[2]) if len(sys.argv) > 2 else 90.0
xs = np.arange(lung[:, 0].min() - 3, lung[:, 0].max() + 3, RES)
zs = np.arange(lung[:, 2].min() - 3, lung[:, 2].max() + 3, RES)
XX, ZZ = np.meshgrid(xs, zs)                                           # image rows = z, columns = x
P = np.stack([XX.ravel(), np.full(XX.size, y), ZZ.ravel()], 1)
lob = wall_of(lobe_at(P))

S, order, parent, sites, site_seg = tree("lung_arteries")
keep = supply_mask(order, site_seg); sites, site_seg = sites[keep], site_seg[keep]
slob = wall_of(lobe_at(sites))
labels = {}
for k in (5, TRIM):
    anc = ancestor_at(order, parent, k)
    top = sorted(set(anc[s_] for s_ in np.unique(site_seg))); idx = {g: i for i, g in enumerate(top)}
    lab = np.array([idx[anc[s]] for s in site_seg])
    L = np.full(len(P), -1); dist = np.full(len(P), np.nan)
    for lo in range(1, 6):
        v = np.nonzero(lob == lo)[0]; own = np.nonzero(slob == lo)[0]
        if len(v) == 0 or len(own) == 0:
            continue
        dep = min(4, len(np.unique(lab[own])))
        g, d, p = nearest_groups(P[v], sites[own], lab[own], dep)
        L[v] = g[:, 0]
        if dep > 1:
            p = own[p]
            bis = [(d[:, j] ** 2 - d[:, 0] ** 2) / (2 * np.maximum(np.linalg.norm(sites[p[:, j]] - sites[p[:, 0]], axis=1), 1e-6))
                   for j in range(1, dep)]
            dist[v] = np.min(bis, axis=0)
    labels[k] = (L.reshape(XX.shape), dist.reshape(XX.shape), len(top))
    print(f"Strahler >= {k}: {len(top)} regions, slice {XX.shape[1]} x {XX.shape[0]} at {RES} mm")

Sv, _, _, vsites, _ = tree("lung_veins")
near = np.abs(vsites[:, 1] - y) < 1.5
ins = (lob > 0).reshape(XX.shape)
ext = (xs[0], xs[-1], zs[0], zs[-1])
fig, axs = plt.subplots(1, 3, figsize=(24, 9.5))
for a_, k in zip(axs[:2], (5, TRIM)):
    L, _, n = labels[k]
    cm = np.random.default_rng(3 + k).uniform(0.35, 1.0, (n + 1, 3))
    img = np.where(ins[..., None], cm[np.clip(L, 0, n)], 1.0)
    edge = ins & (ndi.maximum_filter(L, 3) != ndi.minimum_filter(L, 3))
    img[edge] = 0.12
    rim = ins & ~ndi.binary_erosion(ins, iterations=2)
    img[rim] = 0.35
    a_.imshow(img, origin="lower", extent=ext, interpolation="nearest")
    a_.plot(vsites[near, 0], vsites[near, 2], ".", ms=2.2, color="#1f4e9c", label="vein centerlines (within 1.5 mm)")
    a_.set_title(f"Strahler >= {k}: {n} arterial territories", fontsize=13)
    a_.set_xlabel("x (LPS, mm): patient right <-> left"); a_.set_ylabel("z (mm)")
axs[0].legend(loc="lower left", fontsize=10, markerscale=5)
_, dist, _ = labels[TRIM]
im = axs[2].imshow(np.where(ins, np.minimum(dist, 12), np.nan), origin="lower", extent=ext, cmap="magma",
                   interpolation="nearest")
axs[2].plot(vsites[near, 0], vsites[near, 2], ".", ms=1.6, color="#7fd4ff")
axs[2].set_title(f"mm to the nearest boundary between territories (Strahler >= {TRIM})\n"
                 "lung surface not counted; veins in blue", fontsize=13)
plt.colorbar(im, ax=axs[2], fraction=0.046)
fig.suptitle(f"{run}: pulmonary arterial catchments, coronal slice y = {y:.0f} mm, evaluated at {RES} mm "
             f"(supply: {SUPPLY}, walls: {WALLS})", fontsize=15)
plt.tight_layout()
out = DATA / f"{run}_catchments_hires_{SUPPLY}_{WALLS}.png"
plt.savefig(out, dpi=150)
print(f"wrote {out}")
