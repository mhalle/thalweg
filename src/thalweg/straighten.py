"""A straightened view along a path (vmtk's ``vmtkimagecurvedmpr``): the field resampled on the
normal planes of a smoothed centerline, stacked into a volume whose third axis is arc length.

:func:`straighten` takes any path (points and radii) and any volume on a grid - the structure's
margin, or an image on the same kind of grid - and samples it on square normal sections every
``step`` mm, each carried along the path by parallel transport (:func:`thalweg.kernel.sections.stations`),
so the view does not twist. :func:`path_to` gives the path from a structure's root to one of its
nodes. Ported from ``research/vessels/straighten.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import ThalwegError
from .graph import TubeGraph
from .kernel import sections as S
from .kernel.field import sample


@dataclass
class Straightened:
    """``image[i]``: the section at station i, axis 0 along ``n1[i]``, axis 1 along ``n2[i]``, both
    over ``coords`` (mm from the path); ``arc_length_mm[i]`` along the smoothed path."""

    image: np.ndarray                # (S, K, K) float32
    coords: np.ndarray               # (K,)
    arc_length_mm: np.ndarray        # (S,)
    centers: np.ndarray              # (S, 3)
    n1: np.ndarray                   # (S, 3)
    n2: np.ndarray                   # (S, 3)
    radius_mm: np.ndarray            # (S,) the path's radius, interpolated


def straighten(volume: np.ndarray, geometry, points, radius, step: float = 0.5,
               half_width: float | None = None, pixel: float = 0.1,
               cval: float | None = None) -> Straightened:
    """The straightened view (see the module docstring). ``half_width``: the section's half size
    (default 1.6 x the path's largest radius + 2 mm, as the branch table's sections); ``cval``: the
    value outside the grid (default the volume's minimum)."""
    points = np.asarray(points, float)
    radius = np.asarray(radius, float)
    if len(points) < 4:
        raise ThalwegError(f"a path of {len(points)} points is too short to straighten")
    st = S.stations(points, np.where(radius > 0, radius, 0.5), step=step)
    half = S.half_width(float(np.max(st.radius))) if half_width is None else float(half_width)
    g = np.arange(-half, half + 1e-9, pixel)
    U, V = np.meshgrid(g, g, indexing="ij")
    fill = float(np.min(volume)) if cval is None else float(cval)
    img = np.empty((len(st.centers), len(g), len(g)), np.float32)
    for i in range(len(st.centers)):
        P = st.centers[i] + U[..., None] * st.n1[i] + V[..., None] * st.n2[i]
        img[i] = sample(volume, geometry, P.reshape(-1, 3), cval=fill).reshape(U.shape)
    return Straightened(img, g, st.s, st.centers, st.n1, st.n2, st.radius)


def path_to(graph: TubeGraph, structure: str, node: int) -> tuple[np.ndarray, np.ndarray]:
    """``(points, radii)`` of the structure's centerline from its root to ``node``, edge by edge
    (each junction sample once)."""
    t = graph.tree(structure)
    chain = []
    n = node
    while n != t.root:
        if n not in t.parent:
            raise ThalwegError(f"node {node} is not in {structure!r}")
        e = t.parent[n]
        chain.append(e)
        n = graph.edges[e].start_node
    if not chain:
        raise ThalwegError(f"node {node} is the root; there is no path to straighten")
    pts, rad = [], []
    for k, e in enumerate(reversed(chain)):
        p, r = graph.edge_points(e), graph.edge_radius(e)
        pts.append(p if k == 0 else p[1:])
        rad.append(r if k == 0 else r[1:])
    return np.concatenate(pts), np.concatenate(rad)
