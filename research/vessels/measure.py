"""Step-4 measurements on the idc-torso1 lung_vessels store (0.703 x 0.703 x 1.0 mm)."""
import time
from pathlib import Path
import numpy as np
from _data import DATA
CASE = __import__('sys').argv[1] if len(__import__('sys').argv) > 1 else 'idc-torso1'
from scipy import ndimage as ndi
from skimage.measure import euler_number
from skimage.morphology import skeletonize

C = np.load(DATA / f"{CASE}_cache.npz")
lab, m_art, m_vein, m_sel, ct = C["labels"], C["m_art"], C["m_vein"], C["m_sel"], C["ct"]
sp, clip = C["spacing"], float(C["clip"])
vox_ml = float(np.prod(sp)) / 1000
NAMES = {1: "airways", 2: "airway_wall", 3: "arteries", 4: "veins"}
S26 = np.ones((3, 3, 3), bool)
t0 = time.time()
def tick(msg): print(f"  [{time.time()-t0:5.1f}s] {msg}", flush=True)

print("== 1. volumes, HU, components (26-conn)")
for c, n in NAMES.items():
    mk = lab == c
    cc, k = ndi.label(mk, S26)
    sizes = np.bincount(cc.ravel())[1:]
    big = np.sort(sizes)[::-1]
    print(f"{n:12s} {mk.sum()*vox_ml:7.1f} mL  HU median {np.median(ct[mk]):6.0f}  "
          f"components {k:5d}  largest {big[0]/sizes.sum():.1%}  top3 {big[:3].tolist()}  "
          f"<50 vox: {(sizes<50).sum()}")

print("\n== 2. outside the body (CT < -500 HU region connected to the volume border = outside air)")
air = ct < -500
bcc, _ = ndi.label(air)
border = np.unique(np.concatenate([bcc[[0, -1]].ravel(), bcc[:, [0, -1]].ravel(), bcc[:, :, [0, -1]].ravel()]))
outside = np.isin(bcc, border[border > 0])
body = ~outside
for c, n in NAMES.items():
    mk = lab == c
    print(f"{n:12s} outside body: {(mk & outside).sum()/max(mk.sum(),1):.2%}  "
          f"at HU < -800: {(mk & (ct < -800)).sum()/max(mk.sum(),1):.2%}")
tick("body")

print("\n== 3. margin steepness at the surface: k = |grad m| (logits/mm), band = clip/k")
for c, m in ((3, m_art), (4, m_vein)):
    g = np.gradient(m, *sp)
    gm = np.sqrt(sum(x * x for x in g))
    # surface voxels: sign change to a face neighbor, and not saturated on either side
    surf = np.zeros(m.shape, bool)
    for a in range(3):
        s0 = [slice(None)] * 3; s1 = [slice(None)] * 3
        s0[a] = slice(0, -1); s1[a] = slice(1, None)
        flip = (m[tuple(s0)] > 0) != (m[tuple(s1)] > 0)
        surf[tuple(s0)] |= flip; surf[tuple(s1)] |= flip
    surf &= np.abs(m) < clip * 0.95
    k = gm[surf]
    q = np.percentile(k, [10, 50, 90])
    print(f"{NAMES[c]:12s} surface voxels {surf.sum():8d}  k p10/50/90 = {q.round(2)} logits/mm  "
          f"band clip/k (median) = {clip/q[1]:.2f} mm = {clip/q[1]/sp[1]:.2f} in-plane voxels")
    inside = m > 0
    print(f"{'':12s} interior voxels saturated (m >= clip*0.99): {(m[inside] >= clip*0.99).mean():.1%}")
tick("steepness")

print("\n== 4. caliber along the tree (skeleton of each class, radius = anisotropic EDT at skeleton)")
cal = {}
for c in (3, 4):
    mk = lab == c
    edt = ndi.distance_transform_edt(mk, sampling=sp)
    sk = skeletonize(mk)
    r = edt[sk.astype(bool)]
    cal[c] = r
    bins = [0, 0.7, 1.05, 1.4, 2.1, 3.5, 1e9]
    h = np.histogram(r, bins)[0] / len(r)
    lbl = ["<=0.7", "0.7-1.05", "1.05-1.4", "1.4-2.1", "2.1-3.5", ">3.5"]
    print(f"{NAMES[c]:12s} skeleton voxels {len(r):7d}  radius mm p10/50/90 {np.percentile(r,[10,50,90]).round(2)}")
    print(f"{'':12s} " + "  ".join(f"r {l}: {v:.1%}" for l, v in zip(lbl, h)))
tick("caliber")

print("\n== 5. loops: b1 from Euler number (b1 = b0 + b2 - chi); artery, vein, and their binary union")
def betti(mk):
    b0 = ndi.label(mk, S26)[1]
    b2 = ndi.label(~mk, np.array([[[0,0,0],[0,1,0],[0,0,0]],[[0,1,0],[1,1,1],[0,1,0]],[[0,0,0],[0,1,0],[0,0,0]]], bool))[1] - 1
    chi = euler_number(mk, connectivity=3)
    return b0, b0 + b2 - chi, b2
res = {}
for n, mk in (("arteries", lab == 3), ("veins", lab == 4), ("art|vein union", (lab == 3) | (lab == 4))):
    b0, b1, b2 = betti(mk)
    res[n] = (b0, b1)
    print(f"{n:16s} b0 {b0:5d}  b1 {b1:5d}  b2 {b2:4d}")
extra = res["art|vein union"][1] - res["arteries"][1] - res["veins"][1]
print(f"loops created by merging arteries and veins into one lumen: {extra}")
a, v = lab == 3, lab == 4
faces = sum(int(((np.take(a, range(0, a.shape[ax]-1), ax) & np.take(v, range(1, a.shape[ax]), ax)) |
                 (np.take(v, range(0, a.shape[ax]-1), ax) & np.take(a, range(1, a.shape[ax]), ax))).sum()) for ax in range(3))
print(f"artery|vein touching voxel faces: {faces}")
tick("topology")

print("\n== 6. superlevel persistence of the vessel-union margin: components (>= 20 vox) vs level")
for lvl in (0.0, -0.5, -1.0, -2.0, -4.0, -7.9):
    cc, k = ndi.label(m_sel > lvl, S26)
    sizes = np.bincount(cc.ravel())[1:]
    print(f"m_sel > {lvl:5.1f}: {k:5d} components, {(sizes>=20).sum():4d} with >= 20 voxels, "
          f"largest holds {sizes.max()/sizes.sum():.1%}")
for c, m in ((3, m_art), (4, m_vein)):
    out = []
    for lvl in (0.0, -1.0, -2.0, -4.0):
        cc, k = ndi.label(m > lvl, S26)
        sizes = np.bincount(cc.ravel())[1:]
        out.append(f"{lvl:4.1f}: {k} ({(sizes>=20).sum()} >=20)")
    print(f"{NAMES[c]:12s} " + " | ".join(out))
tick("persistence")
np.savez(DATA / f"{CASE}_caliber.npz", art=cal[3], vein=cal[4])
