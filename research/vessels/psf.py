"""Is the model's thin-vessel radius the anatomy or the scanner's blur?

1. Edge-spread fit on LARGE vessels (ridge radius > 4 mm, wall flat on the PSF scale):
   HU(s) = bg + D * Phi((s0 - s)/sigma), s = signed distance to the model boundary (+ outside).
   Gives the blur sigma and where the model's boundary sits relative to the 50 % level (s0).
2. Per centerline point: the center contrast f = (HU_c - bg)/D of a disc of radius a blurred by
   a Gaussian of sigma is 1 - exp(-a^2 / 2 sigma^2), so a = sigma sqrt(-2 ln(1 - f)).
   Compare that image radius with the model's (ridge) radius.
The PSF is anisotropic (2 mm slices vs 0.65 mm pixels); sigma here is an orientation average.
"""
from pathlib import Path
import numpy as np
from _data import DATA
CASE = __import__('sys').argv[1] if len(__import__('sys').argv) > 1 else 'idc-torso1'
from scipy import ndimage as ndi
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from scipy.special import ndtr
from skimage.morphology import skeletonize

H = DATA
C = np.load(H / f"{CASE}_cache.npz"); R = np.load(H / f"{CASE}_radius.npz")
lab, ct, sp = C["labels"], C["ct"].astype(np.float32), C["spacing"]
FIELD = {3: ("arteries", C["m_art"]), 4: ("veins", C["m_vein"])}

def crossings(m):
    pts = []
    for a in range(3):
        s0 = [slice(None)] * 3; s1 = [slice(None)] * 3
        s0[a] = slice(0, -1); s1[a] = slice(1, None)
        ma, mb = m[tuple(s0)], m[tuple(s1)]
        flip = (ma > 0) != (mb > 0)
        idx = np.argwhere(flip).astype(np.float32)
        idx[:, a] += ma[flip] / (ma[flip] - mb[flip])
        pts.append(idx * sp)
    return np.concatenate(pts)

esf = lambda s, bg, D, s0, sig: bg + D * ndtr((s0 - s) / sig)

for c, (name, m) in FIELD.items():
    mk = lab == c
    sk = np.argwhere(skeletonize(mk).astype(bool))
    r_model = R[f"{name}_ridge"]
    skt = cKDTree(sk * sp)
    ctree = cKDTree(crossings(m))
    # band voxels around the boundary, attributed to their nearest centerline point
    band = np.argwhere(ndi.binary_dilation(mk, iterations=4) & ~ndi.binary_erosion(mk, iterations=6))
    d, _ = ctree.query(band * sp)
    s = np.where(m[tuple(band.T)] > 0, -d, d)               # + outside
    _, near = skt.query(band * sp)
    rn = r_model[near]
    big = (rn > 4.0) & (np.abs(s) < 4.0)
    hu = ct[tuple(band.T)]
    p, _ = curve_fit(esf, s[big], hu[big], p0=(-800, 900, 0.0, 0.8))
    bg, D, s0, sig = p
    blood = bg + D
    print(f"{name}: ESF on {big.sum()} band voxels of vessels with r > 4 mm")
    print(f"   bg {bg:.0f} HU, lumen {blood:.0f} HU, sigma {sig:.2f} mm (FWHM {2.355*sig:.2f}), "
          f"50% level sits {s0:+.2f} mm from the model boundary (+ = outside)")
    # lung background for thin vessels is the parenchyma, not the hilar fit: measure it
    lung_bg = np.median(ct[(lab == 0) & (ct < -500) & ndi.binary_dilation(mk, iterations=3)])
    hc = ct[tuple(sk.T)]
    f = np.clip((hc - lung_bg) / (blood - lung_bg), 1e-4, 0.97)
    a_img = sig * np.sqrt(-2 * np.log(1 - f))
    print(f"   thin-vessel background (parenchyma next to vessels) {lung_bg:.0f} HU")
    for lo, hi in ((0, 1.25), (1.25, 1.5), (1.5, 1.75), (1.75, 2.0), (2.0, 2.5), (2.5, 3.0)):
        q = (r_model >= lo) & (r_model < hi)
        if q.sum() < 30: continue
        print(f"   model r {lo:.2f}-{hi:.2f} mm (n={q.sum():5d}): center HU {np.median(hc[q]):5.0f}  "
              f"image radius a = {np.median(a_img[q]):.2f} mm  (IQR {np.percentile(a_img[q],25):.2f}-{np.percentile(a_img[q],75):.2f})")
    # what a model boundary at the half-max contour of a blurred thin disc would read, a -> 0
    print(f"   half-max contour of a vanishing tube: r = sigma*sqrt(2 ln 2) = {sig*np.sqrt(2*np.log(2)):.2f} mm; "
          f"model r p5 = {np.percentile(r_model,5):.2f} mm")
