"""Analytic tube phantoms: margin fields whose truth is known.

A tube is a polyline with a radius per vertex; the phantom's signed distance is the union (max)
over segments of r(t) - |x - c(t)| (a capsule chain, cone frusta joined by spheres), and the
margin is that distance times a logit slope, clipped like a real ranked store (~10.6 logit/mm,
+-8). Everything is in world mm on a rankfield Geometry.
"""
import numpy as np
from rankfield.geometry import Geometry

SLOPE, CLIP = 10.6, 8.0


def capsule_distance(points, a, b, ra, rb):
    """Signed distance (positive inside, approximately for ra != rb) to the frustum a-b."""
    ab = b - a
    t = np.clip(((points - a) @ ab) / (ab @ ab), 0, 1)
    c = a + t[:, None] * ab
    return (ra + t * (rb - ra)) - np.linalg.norm(points - c, axis=1)


def tube_field(polylines, radii, spacing=0.7, pad=4.0):
    """(margin float32, Geometry) of the union of the tubes (lists of vertices and radii)."""
    allp = np.concatenate(polylines)
    rmax = max(float(np.max(r)) for r in radii)
    lo = allp.min(0) - rmax - pad
    hi = allp.max(0) + rmax + pad
    shape = tuple(int(np.ceil((b - a) / spacing)) + 1 for a, b in zip(lo, hi))
    geo = Geometry(shape=shape, directions=((spacing, 0, 0), (0, spacing, 0), (0, 0, spacing)),
                   origin=tuple(lo))
    idx = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1).reshape(-1, 3)
    X = lo + idx * spacing
    d = np.full(len(X), -np.inf)
    for P, R in zip(polylines, radii):
        for k in range(len(P) - 1):
            d = np.maximum(d, capsule_distance(X, P[k], P[k + 1], R[k], R[k + 1]))
    m = np.clip(d * SLOPE, -CLIP, CLIP).astype(np.float32).reshape(shape)
    return m, geo


def y_tree():
    """A trunk (radius 3 mm, 30 mm) splitting into two daughters (2.2 and 1.8 mm, 25 mm)."""
    top = np.array([0.0, 0.0, 30.0])
    trunk = np.array([[0.0, 0.0, 0.0], top])
    left = np.array([top, top + 25.0 * np.array([-0.6, 0.0, 0.8])])
    right = np.array([top, top + 25.0 * np.array([0.6, 0.3, 0.74])])
    return [trunk, left, right], [np.array([3.0, 3.0]), np.array([2.2, 2.2]), np.array([1.8, 1.8])]
