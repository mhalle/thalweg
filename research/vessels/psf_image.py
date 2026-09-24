"""Scanner blur from the image alone, and vessel radius from the image with that blur.

    python bench/vessels/psf_image.py [PATIENT ...]        (after thick_compare.py)

1. PSF. Wall points of LARGE vessels (reference ridge radius > 4 mm, lung outside) come from
   the thin reference run: the margin's zero crossings, normals from its gradient. At each,
   every variant's own image is sampled along the normal (cubic spline, native grid) and fitted
   with lo + (hi - lo) * Phi((s0 - s) / sigma), s0 FREE per profile - the model's boundary only
   says where to look, never where the edge is (psf.py fixed it to the model boundary and so
   folded the model's scatter into sigma). With a Gaussian PSF of covariance
   diag(sxy^2, sxy^2, sz^2) about the slice axis k, a profile along n measures
   sigma_n^2 = sxy^2 (1 - (n.k)^2) + sz^2 (n.k)^2; sxy and sz are the least-squares solution.
   Check: the synthetic 2 mm slab series must show sz^2 grown by 2^2/12 = 0.333 mm^2.
2. Radius. At each centerline point the PSF is projected on the plane normal to the vessel
   (two sigmas), and the center contrast f = (HU_c - bg) / (lumen - bg) of a disc of radius a
   under that blur is inverted for a. HU_c is averaged +-1 mm along the axis; bg is the local
   parenchyma on a ring 3 mm outside the wall; lumen is the variant's median center HU over
   vessels with r > 4 mm. The same anatomy seen at four thicknesses should give one radius.
"""
import sys, time
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import curve_fit
from scipy.special import ndtr
from scipy.spatial import cKDTree
from _data import DATA
from _field import fine_field, crossings, Image, Grid
from cases import LADDERS, store, image

NAMES = {3: "arteries", 4: "veins"}
MARGIN = {3: "lung_arteries", 4: "lung_veins"}
BINS = [0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 99]
S = np.arange(-4.0, 4.0001, 0.1)
rng = np.random.default_rng(0)


class SplineImage(Image):
    def __init__(self, path):
        super().__init__(path)
        self.coef = ndi.spline_filter(self.arr, order=3, output=np.float32)

    def cubic(self, pts):
        return ndi.map_coordinates(self.coef, self.grid.to_index(pts).T, order=3, prefilter=False,
                                   mode="nearest")


def esf(s, lo, hi, s0, sig):
    return lo + (hi - lo) * ndtr((s0 - s) / sig)


def fit_profiles(img, P, N):
    prof = img.cubic((P[:, None, :] + S[None, :, None] * N[:, None, :]).reshape(-1, 3)).reshape(len(P), len(S))
    out = np.full(len(P), np.nan)
    for i, y in enumerate(prof):
        try:
            p, _ = curve_fit(esf, S, y, p0=(y[-10:].mean(), y[:10].mean(), 0.0, 0.8),
                             bounds=((-1100, -600, -2.5, 0.05), (400, 2000, 2.5, 4.0)), maxfev=400)
        except Exception:
            continue
        resid = y - esf(S, *p)
        step = p[1] - p[0]
        if step > 250 and np.sqrt(np.mean(resid ** 2)) < 0.08 * step:
            out[i] = p[3]
    return out


def solve_psf(sig, c2):
    ok = np.isfinite(sig)
    A = np.stack([1 - c2[ok], c2[ok]], 1)
    s2 = sig[ok] ** 2
    for _ in range(3):                                     # iteratively trimmed least squares
        x, *_ = np.linalg.lstsq(A, s2, rcond=None)
        r = s2 - A @ x
        keep = np.abs(r) < 2.5 * np.median(np.abs(r)) / 0.6745
        A, s2 = A[keep], s2[keep]
    return np.sqrt(np.clip(x, 1e-4, None)), ok.sum(), len(s2)


# f(a; s1, s2): probability mass of a disc of radius a under N(0, diag(s1^2, s2^2)), tabulated
# on u = a / s1 and q = s2 / s1 >= 1 by integrating the Gaussian over the disc in polar form.
U = np.linspace(0.0, 8.0, 321); Q = np.linspace(1.0, 8.0, 57)
th = np.linspace(0, 2 * np.pi, 721)[:-1]
def _mass(u, q):
    # in units of s1: density exp(-rho^2 / 2g(th)) / (2 pi q), g = 1 / (cos^2 + sin^2 / q^2);
    # the radial integral of rho * exp(-rho^2 / 2g) to u is g (1 - exp(-u^2 / 2g))
    g = 1.0 / (np.cos(th) ** 2 + (np.sin(th) / q) ** 2)
    return np.mean(g / q * (1 - np.exp(-(u ** 2) / (2 * g))))
TABLE = np.array([[_mass(u, q) for q in Q] for u in U])     # (len U, len Q), increasing in u
# self-test against Monte Carlo at two anisotropies, and the isotropic closed form
for _q, _u in ((3.0, 1.5), (1.5, 2.0)):
    _x = rng.standard_normal((400000, 2)) * [1.0, _q]
    assert abs(TABLE[np.searchsorted(U, _u), np.searchsorted(Q, _q)] - np.mean(np.hypot(*_x.T) < _u)) < 0.005
assert np.allclose(TABLE[:, 0], 1 - np.exp(-U ** 2 / 2), atol=1e-9)


def invert(f, s1, s2):
    q = np.clip(s2 / s1, 1.0, Q[-1])
    j = np.interp(q, Q, np.arange(len(Q)))
    j0 = np.floor(j).astype(int).clip(0, len(Q) - 2); w = j - j0
    col = TABLE[:, j0] * (1 - w) + TABLE[:, j0 + 1] * w           # (len U, n)
    a = np.array([np.interp(fi, col[:, i], U) for i, fi in enumerate(f)])
    return a * s1


def plane_sigmas(T, k, sxy, sz):
    """The PSF covariance projected on the plane normal to each tangent: its two sigmas."""
    Sig = sxy ** 2 * np.eye(3) + (sz ** 2 - sxy ** 2) * np.outer(k, k)
    out = np.zeros((len(T), 2))
    for i, t in enumerate(T):
        e1 = np.cross(t, k if abs(t @ k) < 0.9 else np.array([1.0, 0, 0])); e1 /= np.linalg.norm(e1)
        e2 = np.cross(t, e1)
        E = np.stack([e1, e2], 1)
        out[i] = np.sqrt(np.sort(np.linalg.eigvalsh(E.T @ Sig @ E)))
    return out


for patient in (sys.argv[1:] or LADDERS):
    t0 = time.time()
    ladder = LADDERS[patient]
    L = np.load(DATA / f"{patient}_ladder.npz")
    margins, labels, grid, _ = fine_field(store(ladder[0][0]))
    # wall points of large vessels with lung outside
    walls, normals = [], []
    for c in (3, 4):
        m = margins[MARGIN[c]]
        ctr, rr = L[f"{NAMES[c]}/points"], L[f"{NAMES[c]}/ref_r"]
        big = cKDTree(ctr[rr > 4.0])
        X = crossings(m, grid)
        d, j = big.query(X)
        near = X[d < rr[rr > 4.0][np.minimum(j, (rr > 4.0).sum() - 1)] + 1.0]
        g = np.stack(np.gradient(m), 0)
        gi = np.stack([ndi.map_coordinates(g[a], grid.to_index(near).T, order=1) for a in range(3)], 1)
        gw = gi @ np.linalg.inv(grid.dirs).T                     # d m / d world
        n = -gw / np.linalg.norm(gw, axis=1, keepdims=True).clip(1e-6)   # outward
        walls.append(near); normals.append(n)
    P = np.concatenate(walls); N = np.concatenate(normals)
    thin = SplineImage(image(ladder[0][1]))
    ok = (thin.cubic(P + 3.0 * N) < -600) & (thin.cubic(P - 1.5 * N) > 0)
    P, N = P[ok], N[ok]
    pick = rng.choice(len(P), min(4000, len(P)), replace=False)
    P, N = P[pick], N[pick]
    print(f"== {patient}: {len(P)} large-vessel wall profiles with lung outside ({time.time()-t0:.0f} s)", flush=True)

    psf = {}
    for run, src, thk, note in ladder:
        img = thin if run == ladder[0][0] else SplineImage(image(src))
        k = img.slice_axis / np.linalg.norm(img.slice_axis)
        sig = fit_profiles(img, P, N)
        (sxy, sz), nfit, nkeep = solve_psf(sig, (N @ k) ** 2)
        psf[run] = (sxy, sz, k, img)
        print(f"   {run:28s} {thk:5.3f} mm  sigma_xy {sxy:.2f}  sigma_z {sz:.2f} mm  "
              f"(sz^2 {sz**2:.3f}; {nfit} fits, {nkeep} kept)  ({time.time()-t0:.0f} s)", flush=True)
    ref = ladder[0][0]
    for run, _, thk, _ in ladder:
        if thk == 2.0 and "slab" in run:
            print(f"   check: slab sz^2 - thin sz^2 = {psf[run][1]**2 - psf[ref][1]**2:+.3f} mm^2 (boxcar 2 mm adds 0.333)")

    out = {}
    for c in (3, 4):
        ctr, rr, T = L[f"{NAMES[c]}/points"], L[f"{NAMES[c]}/ref_r"], L[f"{NAMES[c]}/tangent"]
        # ring 3 mm outside the wall, in the normal plane, for the local parenchyma
        e1 = np.cross(T, [0, 0, 1.0]); bad = np.linalg.norm(e1, axis=1) < 0.1
        e1[bad] = np.cross(T[bad], [1.0, 0, 0]); e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = np.cross(T, e1)
        ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        for run, src, thk, note in ladder:
            sxy, sz, k, img = psf[run]
            hc = np.mean([img(ctr + s * T) for s in np.linspace(-1, 1, 5)], 0)
            ring = np.stack([img(ctr + (rr + 3.0)[:, None] * (np.cos(a) * e1 + np.sin(a) * e2)) for a in ang], 1)
            ring = np.where(ring < -500, ring, np.nan)
            bg = np.nanmedian(ring, 1)
            lumen = np.median(hc[rr > 4.0])
            f = np.clip((hc - bg) / (lumen - bg), 1e-3, 0.98)
            s12 = plane_sigmas(T, k, sxy, sz)
            a = invert(f, s12[:, 0], s12[:, 1])
            a[~np.isfinite(bg) | (np.isfinite(ring).sum(1) < 4)] = np.nan
            out[f"{NAMES[c]}/a:{run}"] = a
        print(f"\n{patient} {NAMES[c]}: IMAGE radius at the reference points, median mm (model's thin-run radius in the 2nd column)")
        print(f"{'ref r bin':>11} {'n':>6} {'model':>6} " + " ".join(f"{t:>8}" for _, _, t, _ in ladder))
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            s = (rr >= lo) & (rr < hi)
            if s.sum() < 30: continue
            cells = [f"{np.nanmedian(out[f'{NAMES[c]}/a:{run}'][s]):8.2f}" for run, _, _, _ in ladder]
            print(f"{lo:5.2f}-{hi:<5.2f} {s.sum():6d} {np.median(rr[s]):6.2f} " + " ".join(cells))
    np.savez(DATA / f"{patient}_psf.npz", **out,
             **{f"psf:{run}": np.array([v[0], v[1]]) for run, v in psf.items()})
    print(f"wrote {DATA / f'{patient}_psf.npz'} ({time.time()-t0:.0f} s)\n", flush=True)
