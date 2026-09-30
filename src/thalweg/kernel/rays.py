"""Rays through the field: how far from a point, in a direction, the structure's wall lies.

:func:`first_crossing` samples each ray at its origin and then every ``step`` mm through the
margin's trilinear interpolant, and returns the distance to the first point where it is <= 0,
placed between the last inside sample and the first outside one by linear interpolation of the two
values. A ray whose origin is outside (margin <= 0 there), that does not leave within its
``reach``, or that runs off the grid while still inside (a structure cut by the field of view has
no wall there) has no crossing (NaN). The wall is located to a small fraction of ``step``; a wall
thinner than ``step`` can be stepped over, and the default 0.05 mm is far below any grid this runs
on. Every ray is marched to the largest ``reach`` given, so one very long ray costs all of them.
"""
from __future__ import annotations

import numpy as np

from .field import sample

RAY_CHUNK = 1 << 22                 # samples (rays x steps) evaluated at once


def first_crossing(m: np.ndarray, geometry, origins, directions, reach, step: float = 0.05) -> np.ndarray:
    """Distance (mm) from each origin along its direction to the margin's first zero crossing.

    ``origins``, ``directions``: (N, 3); directions are normalized here. ``reach``: the farthest
    distance searched, a positive number or (N,). NaN where the ray starts outside the structure
    (or off the grid) or does not leave it within its ``reach``."""
    o = np.asarray(origins, float).reshape(-1, 3)
    d = np.asarray(directions, float).reshape(-1, 3)
    length = np.linalg.norm(d, axis=1, keepdims=True)
    if len(d) and not (length > 0).all():
        raise ValueError("a ray direction has zero length")
    d = d / np.where(length > 0, length, 1.0)
    reach = np.broadcast_to(np.asarray(reach, float), (len(o),))
    if len(o) and not (np.isfinite(reach).all() and (reach > 0).all()):
        raise ValueError("reach must be positive and finite")
    if not step > 0:
        raise ValueError(f"step must be positive; got {step!r}")
    out = np.full(len(o), np.nan)
    if not len(o):
        return out
    steps = np.arange(0.0, float(reach.max()) + step, step)             # the origin, then every step
    k = len(steps)
    per = max(1, RAY_CHUNK // k)
    for a in range(0, len(o), per):
        oo, dd, rr = o[a:a + per], d[a:a + per], reach[a:a + per]
        q = oo[:, None, :] + steps[None, :, None] * dd[:, None, :]
        v = sample(m, geometry, q.reshape(-1, 3), cval=np.nan).reshape(len(oo), k)
        off = ~np.isfinite(v)                                           # past the grid's edge
        stop = (v <= 0) | off
        first = np.where(stop.any(1), stop.argmax(1), -1)
        first = np.where((first >= 0) & off[np.arange(len(oo)), np.maximum(first, 0)], -1, first)
        j = np.clip(first, 1, k - 1)
        rows = np.arange(len(oo))
        va, vb = v[rows, j - 1], v[rows, j]
        r = steps[j - 1] + step * va / np.maximum(va - vb, 1e-12)
        ok = (first >= 1) & (r <= rr)                                   # started inside, left within reach
        out[a:a + per] = np.where(ok, r, np.nan)
    return out
