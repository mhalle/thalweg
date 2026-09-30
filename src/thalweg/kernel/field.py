"""A field on an oriented grid, read in world millimeters.

The grid is rankfield's (duckn's, NRRD's) placement: ``origin`` is sample (0, 0, 0) and
``directions[i]`` the world step along array axis ``i`` (its length is the spacing), so
``world = origin + index @ directions``. Anything with ``origin`` and ``directions``
(``rankfield.geometry.Geometry``) works; everything speaks LPS mm.

- :func:`crossings`: the sub-voxel points where a margin crosses zero along grid edges;
- :func:`sample`: trilinear (or other spline order) samples at world points;
- :func:`inscribed_radius`: the ridge-refined inscribed-ball radius near given points.

Ported from ``research/vessels/_field.py`` without change of arithmetic.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def _dirs(geometry) -> np.ndarray:
    return np.asarray(geometry.directions, dtype=float)


def to_world(geometry, index) -> np.ndarray:
    """Continuous array-order index (..., 3) -> world (..., 3), mm."""
    return np.asarray(index, float) @ _dirs(geometry) + np.asarray(geometry.origin, float)


def to_index(geometry, points) -> np.ndarray:
    """World (..., 3), mm -> continuous array-order index (..., 3)."""
    return (np.asarray(points, float) - np.asarray(geometry.origin, float)) @ np.linalg.inv(_dirs(geometry))


def edge_lengths(geometry, offsets) -> np.ndarray:
    """World length, mm, of each integer lattice offset (rows of ``offsets``)."""
    return np.array([np.linalg.norm(np.asarray(o) @ _dirs(geometry)) for o in offsets])


def crossings(m: np.ndarray, geometry) -> np.ndarray:
    """World points where ``m`` crosses zero along grid edges, at t = m_a / (m_a - m_b)."""
    pts = []
    for a in range(3):
        s0 = [slice(None)] * 3
        s1 = [slice(None)] * 3
        s0[a] = slice(0, -1)
        s1[a] = slice(1, None)
        ma, mb = m[tuple(s0)], m[tuple(s1)]
        flip = (ma > 0) != (mb > 0)
        idx = np.argwhere(flip).astype(np.float64)
        idx[:, a] += ma[flip] / (ma[flip] - mb[flip])
        pts.append(to_world(geometry, idx))
    return np.concatenate(pts)


def sample(arr: np.ndarray, geometry, points, order: int = 1, cval: float = 0.0) -> np.ndarray:
    """Spline samples (trilinear for ``order=1``) of ``arr`` at world points; ``cval`` outside."""
    return ndi.map_coordinates(arr, to_index(geometry, points).T, order=order, mode="constant", cval=cval)


def inscribed_radius(points, tree, inside, step: float = 0.5, n: int = 5, passes: int = 1):
    """Largest inscribed-ball radius near each point.

    Over an ``n``^3 grid of offsets within +-``step`` mm, the distance to the nearest zero crossing
    (``tree``: a KD tree of :func:`crossings`), among offsets ``inside`` says are inside; returns
    (radius, the refined point). A point with no inside offset keeps radius -1 and its position.

    ``passes`` > 1 refines coarse to fine: each further pass searches a grid four times finer
    around the best point so far. One pass (the research reference) quantizes the refined point to
    the grid's step/2 spacing, and a point off the axis by d has an inscribed radius smaller by d:
    on oblique vessels the radius reads 0.03-0.07 mm small (~0.1 mm on grid-aligned phantoms). Four
    passes leave 0.007-0.016 mm (docs/validation.md §4).
    """
    points = np.asarray(points, dtype=float)
    best = np.full(len(points), -1.0)
    where = points.copy()
    center = points.copy()
    for k in range(max(1, int(passes))):
        s = step / 4 ** k
        g = np.linspace(-s, s, n)
        offs = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
        for o in offs:
            q = center + o
            ok = inside(q)
            r = tree.query(q, workers=-1)[0]                     # parallel; identical results
            take = ok & (r > best)
            best[take] = r[take]
            where[take] = q[take]
        center = where.copy()
    return best, where
