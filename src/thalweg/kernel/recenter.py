"""Move centerline points to the area centroid of their cross-sections.

The tracer's path follows the cheapest route through the inscribed-ball distance, and in a tube
that is not round that distance is nearly flat across the width: a flattened lumen's path wanders
from side to side (elliptic tubes of 2.5:1 and 3:1: 0.3-0.4 mm median, 2-3 mm at the 95th
percentile off the axis). The ridge refinement cannot fix it - it moves a point at most 0.5 mm, to
the largest inscribed ball, and across a flat lumen every ball is about as large.

:func:`recenter` moves each point to the area centroid of the structure's section on the plane
normal to the path there, and repeats (the tangent changes as the path straightens; three rounds
settle it). The centroid is the axis's own definition for a section, round or not, and for a round
tube it is where the point already is. The tangent is the chord over one radius, or over the
section's own half-width where that is wider (``WIDE``): over one radius a flattened lumen's
lateral wander tilts neighboring planes into each other across its width.

A section is not trusted, and its point takes the shift interpolated along the path from its
neighbors, when:

- it stays open at a window of ``MAX_HALF_RADII`` radii;
- it is the junction's merged lumen: another path's axis crosses the plane inside it, or another
  path's tube claims part of its rim (a rim point whose tube function |x - c|^2 - r^2 is lower for
  that path, the partition's rule) - this extends the hold at a shallow junction, where the
  branches stay merged for several radii;
- it reaches past the path's end (its half-width exceeds the arc length to either end: a pouch or
  a cap, not a tube);
- it crosses the plane of the nearest point ``NEIGHBOR_MM`` along either way: the planes fold
  inside the lumen (a bend tight for the section's size), and the centroid is not the axis.

The caller's ``holds`` do not move (the tracer holds radius + 1 mm around every node), and the
total shift ramps up from them at most ``RAMP`` mm per mm of path, so the path leaves a node
without a step. A moved point must stay inside the structure (a C-shaped section's centroid may
not).

The radius is then re-measured at each moved point: the distance to the nearest zero crossing,
the inscribed ball centered there (not the largest one nearby, which would undo the move).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .field import sample
from .sections import _inside, _polygon_area_centroid, center_contour, section_image

ITERATIONS = 3
HOLD_MM = 1.0                     # hold the points within (radius + HOLD_MM) of a node
MAX_HALF_RADII = 8.0              # the widest window, in radii (half-width)
MIN_RADIUS = 0.5                  # mm: a floor for the window size
SECTION_PIXELS = 41               # samples across a section window
SECTION_CHUNK = 512               # sections sampled per call
NEIGHBOR_MM = 0.25                # sections must not cross the planes of the points this far along
SETTLED_MM = 0.01                 # a point whose neighborhood moved less than this last round stays
RAMP = 0.5                        # the largest shift per mm of path from a held point (27 degrees)
RIM_POINTS = 32                   # rim samples tested against the other paths' tubes
WIDE = 1.5                        # a section reaching beyond this many radii takes a longer tangent


def _arc(P: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]


def tangents(P: np.ndarray, reach: np.ndarray) -> np.ndarray:
    """Unit tangents along a polyline: the chord between the points ``reach`` (at least 1 mm)
    behind and ahead in arc length, clamped to the ends."""
    s = _arc(P)
    h = np.maximum(reach, 1.0)
    a, b = np.clip(s - h, 0, s[-1]), np.clip(s + h, 0, s[-1])
    A = np.stack([np.interp(a, s, P[:, k]) for k in range(3)], 1)
    B = np.stack([np.interp(b, s, P[:, k]) for k in range(3)], 1)
    T = B - A
    n = np.linalg.norm(T, axis=1, keepdims=True)
    return np.where(n > 0, T / np.where(n > 0, n, 1.0), np.array([0.0, 0.0, 1.0]))


def end_holds(P: np.ndarray, radius: np.ndarray, at=(), hold_mm: float = HOLD_MM) -> np.ndarray:
    """Points within (radius + ``hold_mm``) of the path's two ends and of the points ``at`` (indices
    of junctions along it), in arc length; the radius is the one at that end or junction."""
    s = _arc(P)
    r = np.maximum(np.asarray(radius, float), 0.0)
    held = (s <= r[0] + hold_mm) | (s >= s[-1] - r[-1] - hold_mm)
    for k in at:
        held |= np.abs(s - s[k]) <= r[k] + hold_mm
    return held


def _axis_crossings(p, t, n1, n2, reach, segs, tree, own):
    """In-plane (u, v) of the other paths' axes where they cross the plane (p, t) within ``reach``."""
    p0, p1, path = segs
    idx = np.asarray(tree.query_ball_point(p, reach), np.int64)
    idx = idx[path[idx] != own]
    if not len(idx):
        return np.zeros((0, 2))
    d0, d1 = (p0[idx] - p) @ t, (p1[idx] - p) @ t
    cross = (d0 * d1 <= 0) & (d0 != d1)
    if not cross.any():
        return np.zeros((0, 2))
    i, a, b = idx[cross], d0[cross], d1[cross]
    x = p0[i] + (a / (a - b))[:, None] * (p1[i] - p0[i]) - p
    return np.stack([x @ n1, x @ n2], 1)


def _tube_values(x: np.ndarray, a: np.ndarray, b: np.ndarray, ra: np.ndarray, rb: np.ndarray) -> np.ndarray:
    """The polyball tube function |x - c|^2 - r^2 of points ``x`` (Q, 3) against segments a-b with
    radii ra-rb (S), c the segment's nearest point and r interpolated there: the minimum over the
    segments, per point."""
    if not len(a):
        return np.full(len(x), np.inf)
    d = b - a
    dd = np.maximum((d * d).sum(1), 1e-300)
    t = np.clip(((x[:, None] - a[None]) * d[None]).sum(2) / dd[None], 0.0, 1.0)
    c = a[None] + t[..., None] * d[None]
    r = ra[None] + t * (rb - ra)[None]
    return (((x[:, None] - c) ** 2).sum(2) - r * r).min(1)


def _frames(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.eye(3)[np.argmin(np.abs(T), axis=1)]
    n1 = np.cross(T, a)
    n1 /= np.linalg.norm(n1, axis=1, keepdims=True)
    return n1, np.cross(T, n1)


def _closed(xy, g, pixel) -> bool:
    return bool(xy.min() > g[0] + 0.5 * pixel and xy.max() < g[-1] - 0.5 * pixel)


def _sections(m, geometry, P, T, r) -> list:
    """For each point, the closed section contour around it on the plane normal to its tangent, in
    (n1, n2) mm, with the frame and the window's half-width - or None if the center is outside or
    the contour stays open at the widest window. The first window (2r + 1 mm, ``SECTION_PIXELS``
    across) is sampled for every point in one call; a wider one, at the same pixel size, only where
    that stays open."""
    out = [None] * len(P)
    if not len(P):
        return out
    n1, n2 = _frames(T)
    half = 2.0 * r + 1.0
    unit = np.linspace(-1.0, 1.0, SECTION_PIXELS)
    U, V = np.meshgrid(unit, unit, indexing="ij")
    imgs = np.empty((len(P), *U.shape), np.float32)
    for a in range(0, len(P), SECTION_CHUNK):
        b = min(a + SECTION_CHUNK, len(P))
        X = (P[a:b, None, None] + half[a:b, None, None, None] * (U[None, ..., None] * n1[a:b, None, None]
                                                                 + V[None, ..., None] * n2[a:b, None, None]))
        imgs[a:b] = sample(m, geometry, X.reshape(-1, 3), cval=-8.0).reshape(b - a, *U.shape)
    for k in range(len(P)):
        g = unit * half[k]
        pixel = float(g[1] - g[0])
        got = center_contour(imgs[k], g, 0.0)
        if got is None:
            continue
        if _closed(got[0], g, pixel):
            out[k] = (got[0][:-1], n1[k], n2[k], half[k])
            continue
        h = 2.0 * half[k]
        while h <= MAX_HALF_RADII * r[k] + 1.0:              # wider windows, one at a time
            img, g = section_image(m, geometry, P[k], n1[k], n2[k], h, pixel)
            got = center_contour(img, g, 0.0)
            if got is None:
                break
            if _closed(got[0], g, float(g[1] - g[0])):
                out[k] = (got[0][:-1], n1[k], n2[k], h)
                break
            h *= 2.0
    return out


def _claimed(p, xy, n1, n2, own, segs, tree, rmax, half_seg, rim_points: int = RIM_POINTS) -> bool:
    """Does another path's tube claim part of the section's rim - a rim point whose tube function
    is lower for another path than for its own (the partition rule)? Then the section runs into
    that path's lumen: a junction's merged region, beyond the hold.

    Only segments near the section can: the own path passes through ``p``, so at a rim point within
    e of it the own value is at most e^2, and a segment beating it lies within e + rmax of the
    point, 2 e + rmax (+ half a segment, for its midpoint) of ``p``."""
    p0, p1, r0, r1, lab = segs
    e = float(np.linalg.norm(xy, axis=1).max())
    idx = np.asarray(tree.query_ball_point(p, 2 * e + rmax + half_seg), np.int64)
    if not len(idx):
        return False
    other = lab[idx] != own
    if not other.any():
        return False
    uv = xy[:: max(1, len(xy) // rim_points)]
    rim = p + uv[:, :1] * n1 + uv[:, 1:] * n2
    mine, them = idx[~other], idx[other]
    v_own = _tube_values(rim, p0[mine], p1[mine], r0[mine], r1[mine])
    return bool((_tube_values(rim, p0[them], p1[them], r0[them], r1[them]) < v_own).any())


def recenter(m: np.ndarray, geometry, paths: list[np.ndarray], radii: list[np.ndarray],
             holds: list[np.ndarray], distance: cKDTree, iterations: int = ITERATIONS):
    """Recenter several paths of one structure together (see the module docstring).

    ``paths``: (n_i, 3) world points; ``radii``: their inscribed radii; ``holds``: boolean masks of
    points to keep; ``distance``: a KD tree of the field's zero crossings, for the radius. Returns
    (paths, radii, moved masks): held and rejected points keep their position and radius."""
    P = [np.array(p, float) for p in paths]
    R = [np.array(r, float) for r in radii]
    start = [p.copy() for p in P]
    moved = [np.zeros(len(p), bool) for p in P]
    last = [np.full(len(p), np.inf) for p in P]          # each point's shift in the previous round
    for _ in range(int(iterations)):
        p0 = np.concatenate([p[:-1] for p in P if len(p) > 1] or [np.zeros((0, 3))])
        p1 = np.concatenate([p[1:] for p in P if len(p) > 1] or [np.zeros((0, 3))])
        lab = np.concatenate([np.full(len(p) - 1, i) for i, p in enumerate(P) if len(p) > 1]
                             or [np.zeros(0, int)])
        r0 = np.concatenate([np.maximum(r[:-1], 0) for p, r in zip(P, R) if len(p) > 1] or [np.zeros(0)])
        r1 = np.concatenate([np.maximum(r[1:], 0) for p, r in zip(P, R) if len(p) > 1] or [np.zeros(0)])
        rmax = float(max(r0.max(), r1.max())) if len(r0) else 0.0
        half_seg = 0.5 * np.linalg.norm(p1 - p0, axis=1).max() if len(p0) else 0.0
        tree = cKDTree(0.5 * (p0 + p1)) if len(p0) else None
        shift = 0.0
        for i, p in enumerate(P):
            if len(p) < 3:
                continue
            # the tangent over the section's own extent: over one radius, a flattened lumen's
            # lateral wander tilts the planes of neighbors into each other across its width
            r_ = np.maximum(R[i], MIN_RADIUS)
            reach = r_.copy()
            T = tangents(p, reach)
            s_ = _arc(p)
            # a point whose neighborhood (two tangent reaches) did not move last round has settled
            busy = np.nonzero(last[i] > SETTLED_MM)[0]
            near = np.zeros(len(p), bool)
            if len(busy):
                lo = np.searchsorted(s_, s_[busy] - 2 * np.maximum(r_[busy], 1.0))
                hi = np.searchsorted(s_, s_[busy] + 2 * np.maximum(r_[busy], 1.0), side="right")
                cover = np.zeros(len(p) + 1, int)
                np.add.at(cover, lo, 1)
                np.add.at(cover, hi, -1)
                near = np.cumsum(cover)[:-1] > 0
            ks = np.nonzero(~holds[i] & near)[0]
            got = _sections(m, geometry, p[ks], T[ks], r_[ks])
            found = {int(k): g for k, g in zip(ks, got) if g is not None}
            wide = np.array([k for k, g in found.items()
                             if np.linalg.norm(g[0], axis=1).max() > WIDE * r_[k]], np.int64)
            if len(wide):
                for k in wide:
                    reach[k] = np.linalg.norm(found[k][0], axis=1).max()
                T = tangents(p, reach)
                for k, g in zip(wide, _sections(m, geometry, p[wide], T[wide], r_[wide])):
                    if g is None:
                        del found[int(k)]
                    else:
                        found[int(k)] = g
            for k in list(found):
                xy, n1, n2, half = found[k]
                if tree is not None:
                    hits = _axis_crossings(p[k], T[k], n1, n2, half + half_seg, (p0, p1, lab), tree, i)
                    if any(_inside(xy, h) for h in hits) or _claimed(
                            p[k], xy, n1, n2, i, (p0, p1, r0, r1, lab), tree, rmax, half_seg):
                        del found[k]
                        continue
                found[k] = (xy, n1, n2)
            shiftk = np.full((len(p), 3), np.nan)
            shiftk[holds[i] | ~near] = 0.0
            behind = np.searchsorted(s_, s_ - NEIGHBOR_MM, side="right") - 1     # the nearest points at
            ahead = np.searchsorted(s_, s_ + NEIGHBOR_MM, side="left")          # least NEIGHBOR_MM away
            for k, (xy, n1, n2) in found.items():
                if np.linalg.norm(xy, axis=1).max() > min(s_[k], s_[-1] - s_[k]):
                    continue                              # the plane reaches past the path's end
                rim = p[k] + xy[:, :1] * n1 + xy[:, 1:] * n2
                j = behind[k]
                if j >= 0 and ((rim - p[j]) @ T[j] <= 0).any():
                    continue                              # the section crosses its neighbor's plane
                j = ahead[k]
                if j < len(p) and ((rim - p[j]) @ T[j] >= 0).any():
                    continue
                _, c = _polygon_area_centroid(xy)
                shiftk[k] = c[0] * n1 + c[1] * n2
            known = ~np.isnan(shiftk[:, 0])
            if not known.any():
                continue
            for a in range(3):                            # the others: interpolated along the path
                shiftk[~known, a] = np.interp(s_[~known], s_[known], shiftk[known, a])
            Q = p + shiftk
            if holds[i].any():                            # ramp up from the held points, no step
                sh = s_[holds[i]]
                total = Q - start[i]
                size = np.linalg.norm(total, axis=1)
                cap = RAMP * np.abs(s_[:, None] - sh[None]).min(1)
                over = size > cap
                Q[over] = start[i][over] + total[over] * (cap[over] / size[over])[:, None]
                shiftk = Q - p
            go = np.linalg.norm(shiftk, axis=1) > 0
            go[go] = sample(m, geometry, Q[go]) > 0
            Q[~go] = p[~go]
            if go.any():
                shift = max(shift, float(np.linalg.norm(Q[go] - p[go], axis=1).max()))
            moved[i] |= go
            last[i] = np.linalg.norm(Q - p, axis=1)
            P[i] = Q
        if shift < 1e-3:
            break
    for i in range(len(P)):
        if moved[i].any():
            R[i][moved[i]] = distance.query(P[i][moved[i]], workers=-1)[0]
    return P, R, moved
