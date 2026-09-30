"""Rays through the field: how far from a point, in a direction, the structure's wall lies.

:func:`first_crossing` marches each ray in steps of ``step`` mm through the margin's trilinear
interpolant and returns the distance to the first point where it is <= 0, placed between the last
inside sample and the first outside one by linear interpolation of the two values. A ray that
starts outside (margin <= 0 at its first sample) or never leaves within ``reach`` has no crossing
(NaN). The wall is thus located to a small fraction of ``step``, and a wall thinner than ``step``
can be stepped over; the default 0.05 mm is far below any grid this runs on.
"""
from __future__ import annotations

import numpy as np

from .field import sample

RAY_CHUNK = 1 << 22                 # samples (rays x steps) evaluated at once


def first_crossing(m: np.ndarray, geometry, origins, directions, reach, step: float = 0.05) -> np.ndarray:
    """Distance (mm) from each origin along its unit direction to the margin's first zero crossing.

    ``origins``, ``directions``: (N, 3); ``reach``: the farthest distance searched, a number or (N,).
    NaN where the ray starts outside the structure or does not leave it within ``reach``."""
    o = np.asarray(origins, float).reshape(-1, 3)
    d = np.asarray(directions, float).reshape(-1, 3)
    reach = np.broadcast_to(np.asarray(reach, float), (len(o),))
    out = np.full(len(o), np.nan)
    if not len(o):
        return out
    steps = np.arange(step, float(reach.max()) + step, step)
    k = len(steps)
    per = max(1, RAY_CHUNK // k)
    for a in range(0, len(o), per):
        oo, dd = o[a:a + per], d[a:a + per]
        q = oo[:, None, :] + steps[None, :, None] * dd[:, None, :]
        v = sample(m, geometry, q.reshape(-1, 3), cval=-1.0).reshape(len(oo), k)
        outside = (v <= 0) | (steps[None, :] > reach[a:a + per, None])
        first = np.where(outside.any(1), outside.argmax(1), -1)
        ok = first >= 1
        j = np.clip(first, 1, k - 1)
        rows = np.arange(len(oo))
        va, vb = v[rows, j - 1], v[rows, j]
        r = steps[j - 1] + step * va / np.maximum(va - vb, 1e-12)
        ok &= (vb <= 0) & (steps[j] <= reach[a:a + per] + step)       # left the structure, within reach
        out[a:a + per] = np.where(ok, r, np.nan)
    return out
