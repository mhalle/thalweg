"""thalweg's own path geometry on analytic curves."""
import numpy as np

from thalweg.kernel.geometry import direction_at, path_geometry


def helix(a=10.0, b=3.0, turns=2, n=600, noise=0.0, seed=0):
    t = np.linspace(0, 2 * np.pi * turns, n)
    p = np.stack([a * np.cos(t), a * np.sin(t), b * t], 1)
    if noise:
        p += np.random.default_rng(seed).normal(0, noise, p.shape)
    return p


def test_helix_curvature_and_torsion():
    a, b = 10.0, 3.0
    g = path_geometry(helix(a, b), np.full(600, 1.0))
    k_true, t_true = a / (a * a + b * b), b / (a * a + b * b)
    mid = slice(10, -10)
    # the smoothing may move the path ~15 % of the radius (0.2 mm here) inward on a 10 mm helix:
    # curvature reads ~1.6 % high; that is the price of the noise suppression tested below
    assert abs(np.median(g.curvature[mid]) - k_true) / k_true < 0.03
    assert abs(np.nanmedian(g.torsion[mid]) - t_true) / t_true < 0.02


def test_noise_does_not_inflate_curvature():
    """0.1 mm jitter on 0.3 mm-ish samples: the spline holds curvature near truth (a polyline
    derivative would read several times higher)."""
    a, b = 10.0, 3.0
    g = path_geometry(helix(a, b, noise=0.1), np.full(600, 1.5))
    k_true = a / (a * a + b * b)
    assert abs(np.median(g.curvature[10:-10]) - k_true) / k_true < 0.15


def test_straight_line_metrics():
    p = np.stack([np.linspace(0, 20, 100), np.zeros(100), np.zeros(100)], 1)
    g = path_geometry(p, np.full(100, 1.0))
    assert abs(g.distance_metric - 1) < 1e-6 and g.inflection_count == 0 and g.sum_of_angles < 1e-3
    assert np.allclose(direction_at(p, 5.0), [1, 0, 0])


def test_s_curve_has_one_inflection():
    t = np.linspace(-np.pi, np.pi, 400)
    p = np.stack([t * 10, 5 * np.sin(t), np.zeros_like(t)], 1)
    g = path_geometry(p, np.full(400, 1.0))
    assert g.inflection_count == 1 and g.distance_metric > 1.05
