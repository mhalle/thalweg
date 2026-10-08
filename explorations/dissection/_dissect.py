"""Shared pieces of the dissection prototype: a dual-lumen aortic phantom and the field.

The phantom is an aortic arch-like tube split into a true and a false lumen by an intimal
flap. The flap is a thin background wall where it is intact, so true and false do not touch;
at an entry tear the wall is missing and the two lumens meet at the plane (what a
two-lumen `aortic_dissection` labelmap shows). Branch stubs leave the outer wall from a
known side, so branch feed has ground truth.

    cd explorations/dissection && ../../../haversack/.venv/bin/python dissection.py

Everything is LPS millimeters on a ZYX grid, like the research scripts.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

SPACING = 0.75                      # mm, isotropic; the 1.2 mm flap is 1-2 voxels
R_OUT = 11.0                        # mm, outer aortic wall
FLAP_T = 1.2                        # mm, intimal flap thickness where intact
TEAR_R = 2.4                        # mm, entry-tear window radius in the flap plane
GRADIENT = 4.0                      # logits per mm of margin

# Ground truth. A tear is a disk of missing flap, centered on the flap plane (v = 0)
# at centerline station s and offset u along +N (u = 0 is the centerline itself).
TEARS = (
    {"s": 38.0, "u": 0.0, "r": TEAR_R},
    {"s": 92.0, "u": 4.0, "r": TEAR_R * 0.85},
)
# Branches leave the outer wall along direction (cos phi, sin phi) in the (N, B) frame.
# phi in (-pi, 0) is the true side (v < 0); phi in (0, pi) the false side; a "both" branch
# straddles the flap (phi near 0 or pi) so its stub is split by v = 0.
BRANCHES = (
    {"name": "brachiocephalic", "s": 22.0, "phi": -0.9, "len": 16.0, "r": 3.2, "feed": "true"},
    {"name": "celiac", "s": 58.0, "phi": 1.2, "len": 14.0, "r": 2.8, "feed": "false"},
    {"name": "sma", "s": 72.0, "phi": -1.1, "len": 14.0, "r": 2.6, "feed": "true"},
    {"name": "renal_left", "s": 100.0, "phi": 0.0, "len": 12.0, "r": 2.4, "feed": "both"},
)


def centerline(n=800):
    """Arch-like centerline and a parallel-transported frame. Returns s, P, T, N, B."""
    t = np.linspace(0, 1, n)
    p0 = np.array([0.0, -20.0, -55.0])
    p1 = np.array([0.0, -40.0, 35.0])
    p2 = np.array([0.0, 20.0, 50.0])
    p3 = np.array([0.0, 55.0, -55.0])
    P = ((1 - t) ** 3)[:, None] * p0 + (3 * (1 - t) ** 2 * t)[:, None] * p1 \
        + (3 * (1 - t) * t ** 2)[:, None] * p2 + (t ** 3)[:, None] * p3
    d = np.gradient(P, axis=0)
    T = d / np.linalg.norm(d, axis=1, keepdims=True)
    ref = np.array([1.0, 0.0, 0.0])                     # +N starts along LPS x (right)
    N0 = ref - (ref @ T[0]) * T[0]
    N0 /= np.linalg.norm(N0)
    frames = [N0]
    for i in range(1, n):
        v = frames[-1] - (frames[-1] @ T[i]) * T[i]
        nv = np.linalg.norm(v)
        frames.append(v / nv if nv > 1e-9 else frames[-1])
    N = np.stack(frames)
    B = np.cross(T, N)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    return s, P, T, N, B


def grid_shape(P, margin=20.0):
    """ZYX shape and world origin_xyz of voxel (0, 0, 0) covering the phantom at SPACING."""
    lo = P.min(0) - margin
    hi = P.max(0) + margin
    shape = tuple(int(np.ceil((hi[k] - lo[k]) / SPACING)) + 1 for k in (2, 1, 0))  # Z,Y,X
    return shape, lo


def section_coords(P, N, B, pts, s_cl, tree=None):
    """World points -> (s, r, u, v, phi, i). u along +N, v along +B; phi = atan2(v, u)."""
    if tree is None:
        tree = cKDTree(P)
    _, i = tree.query(pts)
    d = pts - P[i]
    u = np.einsum("ij,ij->i", d, N[i])
    v = np.einsum("ij,ij->i", d, B[i])
    r = np.hypot(u, v)
    phi = np.arctan2(v, u)
    return s_cl[i], r, u, v, phi, i


def phantom_labels(shape, origin_xyz=None):
    """Label map 0 wall/background, 1 true lumen, 2 false lumen; plus ground truth.

    The flap is the wall ``|v| < FLAP_T/2`` inside the tube, minus the tear windows.
    True lumen is ``v < -FLAP_T/2``, false ``v > +FLAP_T/2``. Inside a tear window the
    wall is gone and the labels meet at ``v = 0``, so true and false become one
    continuous domain carrying two labels - the case the field must make sense of.
    """
    h = float(SPACING)
    s_cl, P, T, N, B = centerline()
    tree = cKDTree(P)
    origin = np.zeros(3) if origin_xyz is None else np.asarray(origin_xyz, float)
    # voxel (iz, iy, ix) -> world (origin + (ix, iy, iz) * h) in LPS
    zs = np.arange(shape[0]) * h + origin[2]
    ys = np.arange(shape[1]) * h + origin[1]
    xs = np.arange(shape[2]) * h + origin[0]
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], 1)       # LPS
    s, r, u, v, phi, _ = section_coords(P, N, B, pts, s_cl, tree)

    inside = r < R_OUT
    tear = np.zeros(len(s), bool)
    for t in TEARS:
        tear |= ((s - t["s"]) ** 2 + (u - t["u"]) ** 2) < t["r"] ** 2

    wall = inside & (np.abs(v) < FLAP_T / 2) & ~tear
    true = inside & ((v < -FLAP_T / 2) | (tear & (v < 0)))
    false = inside & ((v > FLAP_T / 2) | (tear & (v >= 0)))

    labels = np.zeros(len(s), np.uint8)
    labels[true] = 1
    labels[false] = 2

    branch_gt = []
    for b in BRANCHES:
        i0 = int(np.argmin(np.abs(s_cl - b["s"])))
        axis = np.cos(b["phi"]) * N[i0] + np.sin(b["phi"]) * B[i0]
        axis = axis / np.linalg.norm(axis)
        base = P[i0] + axis * (R_OUT - 1.5)
        tip = base + axis * b["len"]
        ab = tip - base
        tpar = np.clip(np.einsum("ij,j->i", pts - base, ab) / (ab @ ab), 0, 1)
        closest = base + tpar[:, None] * ab
        dist = np.linalg.norm(pts - closest, axis=1)
        in_stub = dist < b["r"]
        # keep the stub outside the main tube, plus its ostium
        in_stub &= (r > R_OUT - 2.0) | (np.linalg.norm(pts - base, axis=1) < b["r"] + 3.0)
        if b["feed"] == "true":
            labels[in_stub] = 1
        elif b["feed"] == "false":
            labels[in_stub] = 2
        else:
            labels[in_stub & (v < 0)] = 1
            labels[in_stub & (v >= 0)] = 2
        branch_gt.append({**b, "base": base.tolist(), "tip": tip.tolist(), "axis": axis.tolist()})

    gt = {"tears": [{k: float(t[k]) for k in ("s", "u", "r")} for t in TEARS],
          "branches": branch_gt, "r_out": R_OUT, "flap_t": FLAP_T, "spacing": h,
          "shape": list(shape), "origin_xyz": origin.tolist(), "n_voxels": int(len(s)),
          "tear_labels": tear.reshape(shape).astype(np.uint8),
          "wall_labels": wall.reshape(shape).astype(np.uint8)}
    return labels.reshape(shape), gt, (s, r, u, v)


def margins_from_labels(labels, n_classes=3, gradient=GRADIENT):
    """Per-class margins from a label map: ``-(gradient/2) * signed distance`` (mm).

    Channel 0 is background. Signed distance is positive outside the class.
    """
    half = gradient / 2.0
    out = np.zeros((n_classes,) + labels.shape, np.float32)
    for c in range(n_classes):
        m = labels == c
        if not m.any():
            out[c] = -half * 20.0
            continue
        d = ndi.distance_transform_edt(~m) - ndi.distance_transform_edt(m)
        out[c] = (-half * d).astype(np.float32)
    return out


def encode_field(logits):
    """logits (K, Z, Y, X) -> (code, margins (K, ...)). rankfield encoder on CPU."""
    import torch
    import rankfield as rf
    code = rf.encode(torch.from_numpy(np.ascontiguousarray(logits)), depth=6, clip=8.0)
    margins = np.stack([rf.margin(code, c).astype(np.float32) for c in range(code.classes)])
    return code, margins
