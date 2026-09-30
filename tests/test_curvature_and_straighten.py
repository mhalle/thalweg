"""Wall curvature from the field, and the straightened view along a path."""
import numpy as np
import pytest
from rankfield.geometry import Geometry

from thalweg.export import zero_set
from thalweg.kernel.curvature import mean_curvature
from thalweg.straighten import path_to, straighten


def _oblique(kind, r, spacing=(0.7, 0.62, 0.66), slope=10.0):
    """A tube (axis oblique to the grid) or a sphere of radius r, as a clipped margin."""
    sp = np.array(spacing)
    shape = tuple(np.ceil(np.array([30.0, 30, 30]) / sp).astype(int))
    geo = Geometry(shape=shape, directions=tuple(map(tuple, np.diag(sp))), origin=(0.0, 0, 0))
    X = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1).reshape(-1, 3) * sp
    ctr = np.array([15.0, 15, 15]) + 0.123
    ax = np.array([1.0, 0.62, 0.37])
    ax /= np.linalg.norm(ax)
    v = X - ctr
    d = np.linalg.norm(v - (v @ ax)[:, None] * ax, axis=1) if kind == "tube" else np.linalg.norm(v, axis=1)
    m = np.clip(slope * (r - d), -8, 8).reshape(shape).astype(np.float32)
    return m, geo, ctr, ax


@pytest.mark.parametrize("kind,r", [("tube", 0.75), ("tube", 1.5), ("tube", 3.0),
                                    ("sphere", 2.0), ("sphere", 4.0)])
def test_mean_curvature_of_tubes_and_spheres(kind, r):
    """At the zero set's vertices: 1 / (2 r) for a tube, 1 / r for a sphere, within 5 % in the
    median and 7 % at the 10th and 90th percentiles (the fit reads 2-4 % high)."""
    m, geo, ctr, ax = _oblique(kind, r)
    V, _ = zero_set(m, geo)
    if kind == "tube":
        V = V[np.abs((V - ctr) @ ax) < 8]                              # away from the box
    h = mean_curvature(m, geo, V) / (1 / (2 * r) if kind == "tube" else 1 / r)
    assert np.isfinite(h).all() and abs(np.median(h) - 1) < 0.05
    assert np.percentile(h, 10) > 0.93 and np.percentile(h, 90) < 1.07


def test_mean_curvature_sign_and_gaps():
    """A cavity (the margin negated) is concave: negative curvature. A point far from any unclipped
    sample has none."""
    m, geo, ctr, _ = _oblique("sphere", 3.0)
    V, _ = zero_set(m, geo)
    assert np.median(mean_curvature(-m, geo, V[:50])) < 0
    assert np.isnan(mean_curvature(m, geo, [[2.0, 2.0, 2.0]])[0])        # the clipped outside, far away
    assert len(mean_curvature(m, geo, np.zeros((0, 3)))) == 0


def test_straightening_a_curved_tube():
    """A tube of radius 2 bent through a quarter circle: every section of the straightened view is
    the tube's disk, centered, of area pi r^2; arc length runs along the bend."""
    t = np.linspace(0, np.pi / 2, 200)
    axis = np.stack([20 * np.cos(t), 20 * np.sin(t), 0 * t], 1) + [2.0, 2.0, 5.0]
    h = 0.4
    shape = (60, 60, 26)
    geo = Geometry(shape=shape, directions=((h, 0, 0), (0, h, 0), (0, 0, h)), origin=(0.0, 0, 0))
    X = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1).reshape(-1, 3) * h
    from scipy.spatial import cKDTree
    d = cKDTree(axis).query(X)[0]
    m = np.clip(10 * (2.0 - d), -8, 8).reshape(shape).astype(np.float32)
    s = straighten(m, geo, axis, np.full(len(axis), 2.0), step=1.0, pixel=0.1)
    k = len(s.coords)
    assert s.image.shape == (len(s.arc_length_mm), k, k) and s.image.dtype == np.float32
    assert abs(s.arc_length_mm[-1] - 10 * np.pi) < 0.6
    inner = slice(3, -3)
    area = (s.image[inner] > 0).sum((1, 2)) * 0.01
    assert np.abs(area / (np.pi * 4) - 1).max() < 0.03
    assert (s.image[inner, k // 2, k // 2] > 7).all()                  # the center is on the axis
    twist = np.abs((s.n1[1:] * s.n1[:-1]).sum(1))
    assert twist.min() > 0.99                                          # carried along without twisting


def test_path_to_a_node():
    from thalweg.centerlines import graph_from_tree
    from thalweg.errors import ThalwegError
    from thalweg.graph import Points, Source, TubeGraph
    from thalweg.kernel import medial
    from phantoms import tube_field, y_tree
    m, geo = tube_field(*y_tree())
    T = medial.trace(m, geo)
    nodes, edges, pos, rad, st = graph_from_tree(T, "y", m, geo, Source(), {"connectivity": "field"})
    g = TubeGraph(structures=[st], nodes=nodes, edges=edges, points=Points(position=pos, radius=rad))
    tip = next(nd.id for nd in g.nodes if nd.kind == "tip")
    P, R = path_to(g, "y", tip)
    assert np.allclose(P[0], g.nodes[st.roots[0]].position) and np.allclose(P[-1], g.nodes[tip].position)
    assert len(P) == len(R) and (np.linalg.norm(np.diff(P, axis=0), axis=1) > 0).all()
    with pytest.raises(ThalwegError):
        path_to(g, "y", st.roots[0])
    view = straighten(m, geo, P, R, step=1.0)
    assert (view.image[2:-2, len(view.coords) // 2, len(view.coords) // 2] > 0).all()
