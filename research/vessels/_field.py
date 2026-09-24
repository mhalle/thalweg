"""Shared pieces for the vessel benches: a store's fine part in world coordinates, the
margin's sub-voxel zero crossings, images sampled in world coordinates, and the
ridge-refined inscribed radius.

Everything here speaks LPS millimeters. A grid maps index i (array order) to world
``origin + i @ dirs`` (rows of ``dirs`` = array axes); ``to_index`` inverts it.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import rankfield as rf
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from haversack.ranked_store import open_store
from haversack.ranked_restore import parts_of

CLASSES = ("lung_airways", "lung_airways_wall", "lung_arteries", "lung_veins")


@dataclass
class Grid:
    origin: np.ndarray
    dirs: np.ndarray            # (3, 3) rows = array axes, mm per index step, LPS

    @property
    def spacing(self):
        return np.linalg.norm(self.dirs, axis=1)

    def to_world(self, idx):
        return np.asarray(idx, float) @ self.dirs + self.origin

    def to_index(self, p):
        return (np.asarray(p, float) - self.origin) @ np.linalg.inv(self.dirs)


def _value(s):
    v = s.get("label_value", s.get("label_values"))
    return v[0] if isinstance(v, list) else v


def fine_field(store):
    """(margins {name: float32 array}, labels uint8 in 0..4 by CLASSES order, Grid, clip)."""
    root = open_store(Path(store), "r").root
    segs = root.attrs.asdict()["duckn"]["extensions"]["seg"]["segments"]
    parts = parts_of(root)
    k = len(parts) - 1
    value_of = {s["name"]: _value(s) for s in segs
                if _value(s) is not None and not s.get("members") and int(s.get("layer", 0)) == k}
    code = parts[k].field
    ch = {n: code.labels.index(value_of[n]) for n in CLASSES}
    geo = code.geometry
    grid = Grid(np.asarray(geo.origin, float), np.asarray(geo.directions, float))
    margins = {n: rf.margin(code, ch[n]).astype(np.float32) for n in ("lung_arteries", "lung_veins")}
    win = np.argmax(np.stack([rf.deficit(code, c) for c in range(code.classes)]), axis=0)
    labels = np.zeros(win.shape, np.uint8)
    for i, n in enumerate(CLASSES, 1):
        labels[win == ch[n]] = i
    return margins, labels, grid, float(code.meta["clip"])


def crossings(m, grid):
    """World points where the margin crosses zero along grid edges, t = m_a / (m_a - m_b)."""
    pts = []
    for a in range(3):
        s0 = [slice(None)] * 3; s1 = [slice(None)] * 3
        s0[a] = slice(0, -1); s1[a] = slice(1, None)
        ma, mb = m[tuple(s0)], m[tuple(s1)]
        flip = (ma > 0) != (mb > 0)
        idx = np.argwhere(flip).astype(np.float64)
        idx[:, a] += ma[flip] / (ma[flip] - mb[flip])
        pts.append(grid.to_world(idx))
    return np.concatenate(pts)


def sample(arr, grid, pts, order=1, cval=0.0):
    """Trilinear (order=1) samples of an array on ``grid`` at world points."""
    return ndi.map_coordinates(arr, grid.to_index(pts).T, order=order, mode="constant", cval=cval)


class Image:
    """A CT (DICOM directory or NIfTI) in its own native geometry, sampled in world mm."""
    def __init__(self, path):
        path = Path(path)
        if path.is_dir():
            r = sitk.ImageSeriesReader(); r.SetFileNames(r.GetGDCMSeriesFileNames(str(path)))
            img = r.Execute()
        else:
            img = sitk.ReadImage(str(path))
        self.arr = sitk.GetArrayFromImage(img).astype(np.float32)     # (k, j, i)
        sp = np.array(img.GetSpacing()); d = np.array(img.GetDirection()).reshape(3, 3)
        # array axis 0 is sitk index k (third), so rows of dirs = columns of d, reversed
        dirs = (d * sp[None, :]).T[::-1]
        self.grid = Grid(np.array(img.GetOrigin()), dirs)
        # the slice axis: the index axis most aligned with world S (every case here is axial).
        # Not "the third index": a NIfTI may store its axes permuted (the demo's i runs along z).
        self.slice_axis = d[:, int(np.argmax(np.abs(d[2])))]

    def __call__(self, pts, order=1):
        return sample(self.arr, self.grid, pts, order=order, cval=-1024.0)


def inscribed_radius(points, tree, inside, step=0.5, n=5):
    """Largest inscribed-ball radius near each point: max over a (n^3) offset grid of +-step mm of
    the distance to the nearest crossing, among offsets the margin says are inside."""
    g = np.linspace(-step, step, n)
    offs = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    best = np.full(len(points), -1.0); where = points.copy()
    for o in offs:
        q = points + o
        ok = inside(q)
        r = tree.query(q)[0]
        take = ok & (r > best)
        best[take] = r[take]; where[take] = q[take]
    return best, where
