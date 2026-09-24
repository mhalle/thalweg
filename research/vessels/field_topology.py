"""Topology of the vessel trees three ways: voxels (26- and 6-connected) and the field.

    python bench/vessels/field_topology.py [RUN ...]

The labelmap's topology depends on a connectivity convention: 26-connectivity joins voxels that
touch only at an edge or corner (merging nearby vessels into false loops), 6-connectivity
refuses those joins (breaking thin vessels that run diagonally). The field decides each such
corner by its own interpolated values: the margin sampled on a grid SUPER times finer (trilinear)
and thresholded at 0, taken 6-connected, converges to the topology of the interpolant itself.

Scored against anatomy: the pulmonary arterial tree is one tree (b0 -> 1 large component, b1 -> 0
loops); the veins are a few trees (they join only at the left atrium, which the task does not
segment), also loop-free. b1 from the Euler characteristic: b1 = b0 + b2 - chi.
"""
import sys, time
import numpy as np
import rankfield as rf
from scipy import ndimage as ndi
from skimage.measure import euler_number
from haversack.ranked_store import open_store
from haversack.ranked_restore import parts_of
from _field import CLASSES, _value
from cases import store
from _topo import field_edges, components, surface_loops, check_against_graph

import os
SUPER = int(os.environ.get("SUPER", "2"))
CROP = os.environ.get("CROP")          # "z0:z1" native slices, for the convergence check
MIN_VOX = 20                    # components counted as "large" (native voxels, ~10 mm^3)
S6 = ndi.generate_binary_structure(3, 1)
S26 = np.ones((3, 3, 3), bool)


def fields(run):
    root = open_store(store(run), "r").root
    segs = root.attrs.asdict()["duckn"]["extensions"]["seg"]["segments"]
    parts = parts_of(root); k = len(parts) - 1
    value_of = {s["name"]: _value(s) for s in segs
                if _value(s) is not None and not s.get("members") and int(s.get("layer", 0)) == k}
    code = parts[k].field
    ch = {n: code.labels.index(value_of[n]) for n in CLASSES}
    a, v = ch["lung_arteries"], ch["lung_veins"]
    win = np.argmax(np.stack([rf.deficit(code, c) for c in range(code.classes)]), axis=0)
    g = rf.decode_groups(code, [[a], [v], [a, v]], device="cpu").numpy().astype(np.float32)
    return {"arteries": (win == a, g[0]), "veins": (win == v, g[1]), "art|vein union": ((win == a) | (win == v), g[2])}


def betti(mask, fg6):
    """(b0, large b0, b1, b2) with 6- (fg6) or 26-connected foreground, complementary background."""
    lab, b0 = ndi.label(mask, S6 if fg6 else S26)
    sizes = np.bincount(lab.ravel())[1:]
    b2 = ndi.label(~mask, S26 if fg6 else S6)[1] - 1
    chi = euler_number(mask, connectivity=1 if fg6 else 3)
    return b0, sizes, b0 + b2 - chi, b2


def supersample(m, box, f):
    """The margin's trilinear interpolant on a grid f times finer, inside box, thresholded at 0."""
    lo, hi = box
    axes = [np.arange(l, h - 1 + 1e-9, 1.0 / f) for l, h in zip(lo, hi)]
    out = np.zeros([len(a) for a in axes], bool)
    for i0 in range(0, len(axes[0]), 32):
        zz = axes[0][i0:i0 + 32]
        Z, Y, X = np.meshgrid(zz, axes[1], axes[2], indexing="ij")
        out[i0:i0 + len(zz)] = ndi.map_coordinates(m, [Z, Y, X], order=1, mode="nearest") > 0
    return out


for run in (sys.argv[1:] or ["C3N-00704_ctpa0625", "MSB-02664_ctape0625_v013", "DEMO:idc-torso1"]):
    t0 = time.time()
    F = fields(run)
    print(f"\n== {run}")
    print(f"{'selection':16s} {'method':22s} {'b0':>6s} {'b0>=20vox':>10s} {'largest':>8s} {'b1 loops':>9s} {'b2':>4s}")
    for name, (mask, m) in F.items():
        if CROP:
            z0, z1 = map(int, CROP.split(":")); keep = np.zeros_like(mask); keep[z0:z1] = True
            mask = mask & keep; m = np.where(keep, m, -8.0)
        nz = np.argwhere(mask)
        lo = np.maximum(nz.min(0) - 2, 0); hi = np.minimum(nz.max(0) + 3, mask.shape)
        sub = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        rows = []
        for label, fg6, arr, vox in (("voxels, 26-connected", False, sub, 1),
                                     ("voxels, 6-connected", True, sub, 1),
                                     (f"field (x{SUPER}), 6-conn.", True, supersample(m, (lo, hi), SUPER), SUPER ** 3)):
            b0, sizes, b1, b2 = betti(arr, fg6)
            big = (sizes >= MIN_VOX * vox).sum()
            rows.append((label, b0, big, sizes.max() / sizes.sum(), b1, b2))
        # the field on the native grid: saddle-rule graph for pieces, surface genus for loops
        box = (lo, hi)
        msub = m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        idx, r_, c_, _, tun = field_edges(msub)
        nc, lab = components(len(idx), r_, c_)
        sizes = np.bincount(lab)
        loops, nsurf, slab, anchor, _ = surface_loops(msub)
        ok, cav, problems = check_against_graph(msub.shape, idx, lab, slab, anchor)
        rows.append(("field, native grid", nc, (sizes >= MIN_VOX).sum(), sizes.max() / sizes.sum(), loops, cav))
        agree = (f"agree ({nsurf} surfaces = {nc} components + {cav} cavities)" if ok
                 else f"DISAGREE: {problems[:3]}")
        for label, b0, big, lg, b1, b2 in rows:
            print(f"{name:16s} {label:22s} {b0:6d} {big:10d} {lg:8.1%} {b1:9d} {b2:4d}")
        print(f"{'':16s} (native field: {tun} cell-interior joins; surface vs graph: {agree})")
    print(f"({time.time() - t0:.0f} s)", flush=True)
