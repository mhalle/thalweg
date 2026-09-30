"""Mean curvature of the wall, read from the field: the curvature of the margin's zero level set.

The surface is the margin's zero set, so its mean curvature is the level set's,
H = -1/2 div(grad m / |grad m|) (m > 0 inside; the outward normal is -grad m / |grad m|), with
vtkCurvatures' convention: H = (k1 + k2) / 2, positive where the wall is convex, so a tube of
radius r has H = 1 / (2 r) and a sphere 1 / r.

A ranked store's margin is steep (about 10 logit/mm) and clipped (at +-8), so it is flat within
about 0.75 mm of the wall on either side; finite differences reach the plateau and read the
curvature low. :func:`mean_curvature` instead fits a quadric to the lattice samples within
``radius`` mm of each point that are NOT clipped, weighted by a Gaussian of half that radius, and
takes the curvature of the fitted quadric's level set at the point
(research/vessels/curvature.py: 1.02-1.04 x the truth on oblique tube and sphere phantoms, and
about six times less spread than vmtk's mesh curvature across reconstructions of one scan).

The fit smooths over its 2.8 mm window: where the curvature changes within it - a saddle, a
bifurcation's crotch - it reads the window's average (the inner equator of a torus of radii 3 and
2, H = -0.25, reads -0.09). ``clip`` must be the store's own clip value: samples at it are left
out, and a wrong value lets plateau samples into the fit.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .field import to_index

QUADRIC_RADIUS = 2.8
MIN_SAMPLES = 12
CHUNK = 2048


def mean_curvature(m: np.ndarray, geometry, points, clip: float = 8.0,
                   radius: float = QUADRIC_RADIUS) -> np.ndarray:
    """Mean curvature (1/mm, vtkCurvatures' sign) of the margin's level set through each world point
    (see the module docstring). NaN where fewer than 12 unclipped samples lie within ``radius``."""
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    out = np.full(len(P), np.nan)
    if not len(P):
        return out
    idx = to_index(geometry, P)
    d = np.asarray(geometry.directions, float)
    reach = int(np.ceil(radius / np.min(np.linalg.norm(d, axis=1)))) + 1
    lo = np.maximum(np.floor(idx.min(0)).astype(int) - reach, 0)
    hi = np.minimum(np.ceil(idx.max(0)).astype(int) + reach + 1, m.shape)
    f = m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    L = np.argwhere(np.abs(f) < clip - 1e-3)
    if not len(L):
        return out
    W = np.asarray(geometry.origin, float) + (L + lo) @ d
    vals = f[tuple(L.T)].astype(np.float64)
    tree = cKDTree(W)
    for a in range(0, len(P), CHUNK):
        pts = P[a:a + CHUNK]
        nbs = tree.query_ball_point(pts, radius, workers=-1)
        n = np.fromiter((len(v) for v in nbs), np.int64, len(nbs))
        ok = n >= MIN_SAMPLES
        if not ok.any():
            continue
        k = int(n[ok].max())
        rows = np.nonzero(ok)[0]
        ids = np.zeros((len(rows), k), np.int64)
        mask = np.zeros((len(rows), k), bool)
        for r, i in enumerate(rows):
            ids[r, :n[i]] = nbs[i]
            mask[r, :n[i]] = True
        x = W[ids] - pts[rows, None, :]                                        # (R, k, 3)
        w = np.exp(-(x ** 2).sum(2) / (radius / 2) ** 2) * mask                # squared weights
        A = np.concatenate([np.ones(x.shape[:2] + (1,)), x, x ** 2, x[..., [0]] * x[..., [1]],
                            x[..., [0]] * x[..., [2]], x[..., [1]] * x[..., [2]]], axis=2)     # (R, k, 10)
        AtA = np.einsum("rki,rk,rkj->rij", A, w, A)
        Atb = np.einsum("rki,rk,rk->ri", A, w, vals[ids])
        c = np.linalg.solve(AtA + 1e-12 * np.eye(10), Atb[..., None])[..., 0]
        g = c[:, 1:4]
        H = np.stack([np.stack([2 * c[:, 4], c[:, 7], c[:, 8]], 1),
                      np.stack([c[:, 7], 2 * c[:, 5], c[:, 9]], 1),
                      np.stack([c[:, 8], c[:, 9], 2 * c[:, 6]], 1)], 1)
        g2 = (g * g).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            hk = -0.5 * (np.trace(H, axis1=1, axis2=2) * g2 - np.einsum("ri,rij,rj->r", g, H, g)) / g2 ** 1.5
        out[a + rows] = hk
    return out
