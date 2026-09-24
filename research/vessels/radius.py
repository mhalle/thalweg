"""(a) CT alignment check, (b) small fragments: breaks, speckle or artery/vein swaps,
(c) sub-voxel radius from the margin's own zero crossings vs the labelmap EDT."""
import time
from pathlib import Path
import numpy as np
from _data import DATA
CASE = __import__('sys').argv[1] if len(__import__('sys').argv) > 1 else 'idc-torso1'
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

C = np.load(DATA / f"{CASE}_cache.npz")
lab, m_art, m_vein, ct = C["labels"], C["m_art"], C["m_vein"], C["ct"]
sp, clip = C["spacing"], float(C["clip"])
S26 = np.ones((3, 3, 3), bool)
NAMES = {3: "arteries", 4: "veins"}
t0 = time.time()

print("== (a) CT alignment: HU by depth inside the vessel (margin level), and at the skeleton")
for c, m in ((3, m_art), (4, m_vein)):
    mk = lab == c
    sk = skeletonize(mk).astype(bool)
    rows = [f"m in [0,2): {np.median(ct[(m >= 0) & (m < 2)]):5.0f}",
            f"[2,6): {np.median(ct[(m >= 2) & (m < 6)]):5.0f}",
            f"sat (>=7.9): {np.median(ct[m >= 7.9]):5.0f}",
            f"skeleton: {np.median(ct[sk]):5.0f}",
            f"shell outside m in (-2,0): {np.median(ct[(m > -2) & (m < 0)]):5.0f}"]
    print(f"{NAMES[c]:9s} median HU  " + "  ".join(rows))

print("\n== (b) small fragments: gap to the main tree of the SAME class, and contact with the OTHER class")
for c in (3, 4):
    other = 7 - c
    mk = lab == c
    cc, k = ndi.label(mk, S26)
    sizes = np.bincount(cc.ravel()); sizes[0] = 0
    main = cc == sizes.argmax()
    d_main = ndi.distance_transform_edt(~main, sampling=sp)
    om = lab == other
    rows = []
    for i in np.argsort(sizes)[::-1][1:]:
        if sizes[i] == 0: break
        f = cc == i
        shell = ndi.binary_dilation(f, S26) & ~f
        gap = d_main[f].min()
        touch_other = (shell & om).sum() / shell.sum()
        rows.append((sizes[i], gap, touch_other))
    rows = np.array(rows)
    print(f"{NAMES[c]}: {len(rows)} fragments besides the main tree, {rows[:,0].sum():.0f} voxels")
    enclave = rows[:, 2] > 0.25
    near = (rows[:, 1] <= 2.1) & ~enclave
    far = ~enclave & ~near
    for nm, sel in (("touching the other class (>25% of rim): swap candidates", enclave),
                    ("within 2.1 mm of own tree: break candidates", near),
                    ("isolated", far)):
        print(f"   {nm:55s} n={sel.sum():3d} voxels={rows[sel,0].sum():6.0f} "
              f"largest={rows[sel,0].max() if sel.any() else 0:.0f}")

print("\n== (c) sub-voxel radius: crossings of m=0 on grid edges, t = m_a/(m_a-m_b)")
res = {}
for c, m in ((3, m_art), (4, m_vein)):
    pts = []
    for a in range(3):
        s0 = [slice(None)] * 3; s1 = [slice(None)] * 3
        s0[a] = slice(0, -1); s1[a] = slice(1, None)
        ma, mb = m[tuple(s0)], m[tuple(s1)]
        flip = (ma > 0) != (mb > 0)
        idx = np.argwhere(flip).astype(np.float32)
        t = ma[flip] / (ma[flip] - mb[flip])
        idx[:, a] += t
        pts.append(idx * sp)                 # array-index mm (axis-aligned grid, so this is metric)
    pts = np.concatenate(pts)
    tree = cKDTree(pts)
    mk = lab == c
    sk = np.argwhere(skeletonize(mk).astype(bool)).astype(np.float32)
    edt = ndi.distance_transform_edt(mk, sampling=sp)[tuple(sk.T.astype(int))]
    # ridge refinement by brute force: the largest inscribed-ball radius within +-0.5 voxel
    g = np.linspace(-0.5, 0.5, 5)
    offs = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
    r0 = tree.query(sk * sp)[0]
    best = r0.copy()
    for o in offs:
        q = (sk + o) * sp
        # a candidate must stay inside: sign of m by trilinear interpolation
        inside = ndi.map_coordinates(m, (sk + o).T, order=1) > 0
        r = tree.query(q)[0]
        best = np.where(inside & (r > best), r, best)
    res[c] = (edt, r0, best)
    print(f"{NAMES[c]:9s} crossings {len(pts):8d}  skeleton pts {len(sk):6d}")
    print(f"          EDT (labelmap)    p10/50/90 {np.percentile(edt,[10,50,90]).round(2)} mm")
    print(f"          crossings @voxel  p10/50/90 {np.percentile(r0,[10,50,90]).round(2)} mm")
    print(f"          crossings, ridge  p10/50/90 {np.percentile(best,[10,50,90]).round(2)} mm   "
          f"EDT - ridge median {np.median(edt-best):+.2f} mm")
    rv = best / sp[1]                       # radius in in-plane voxels
    bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 1e9]
    h = np.histogram(rv, bins)[0] / len(rv)
    lbl = ["<0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", ">=3"]
    print(f"          radius in voxels (0.703 mm): " + "  ".join(f"{l}: {v:.1%}" for l, v in zip(lbl, h)))
    print(f"          -> diameter < 1.4 mm (radius < 1 voxel): {(rv < 1).mean():.1%} of centerline length")
print(f"[{time.time()-t0:.1f}s]")
np.savez(DATA / f"{CASE}_radius.npz", **{f"{NAMES[c]}_{n}": v for c in res for n, v in zip(("edt", "r0", "ridge"), res[c])})
