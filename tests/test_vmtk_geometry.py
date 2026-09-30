"""vtkvmtkCenterlineGeometry port against vmtk 1.5.2 (stages ``geometry`` and ``geometry_smoothed``).

Achieved (vmtk_float32=True): Length and Tortuosity bit-exact; Curvature <= 4e-15 (6e-16 relative),
Frenet vectors <= 4e-16, Torsion <= 5e-10 absolute (at |torsion| ~ 2e6 on the case's raw polyline),
relative <= 1.5e-12 where |torsion| > 1 and absolute <= 3.2e-12 where it is <= 1. The residual is
vmtk's fused multiply-adds (arm64 build), which numpy rounds separately.
"""
import math

import numpy as np
import pytest

from thalweg.vmtk import Centerlines
from thalweg.vmtk.geometry import centerline_geometry, line_curvature, line_torsion

POINT = ["Curvature", "Torsion", "FrenetTangent", "FrenetNormal", "FrenetBinormal"]
CELL = ["Length", "Tortuosity"]


@pytest.mark.parametrize("stage, smoothing", [("geometry", False), ("geometry_smoothed", True)])
def test_geometry_matches_oracle(vmtk_oracle, stage, smoothing):
    inp = Centerlines.from_npz(vmtk_oracle["input"])
    G = Centerlines.from_npz(vmtk_oracle[stage])
    got = centerline_geometry(inp, line_smoothing=smoothing, iterations=100, factor=0.1, vmtk_float32=True)
    assert got.n_points == G.n_points and got.n_cells == G.n_cells
    assert all(np.array_equal(a, b) for a, b in zip(got.cells, G.cells))
    assert np.array_equal(got.points, G.points)             # OutputSmoothedLines 0: input points kept
    assert sorted(got.point_data) == sorted(G.point_data) and sorted(got.cell_data) == sorted(G.cell_data)
    for k in CELL:
        assert np.array_equal(got.cell_data[k], G.cell_data[k]), k
    for k in POINT:
        a, b = got.point_data[k], G.point_data[k]
        assert a.shape == b.shape and a.dtype == b.dtype
        scale = np.maximum(np.abs(b), 1.0)
        if k != "Torsion":                                  # torsion reaches 2e6: relative only
            assert np.abs(a - b).max() <= 1e-9, (k, np.abs(a - b).max())
        rtol = 1e-10 if k == "Torsion" else 1e-12          # torsion: worst 3e-12 (cancellation)
        assert np.all(np.abs(a - b) <= rtol * scale), k


def test_output_smoothed_lines_writes_smoothed_points(phantom_oracle):
    inp = Centerlines.from_npz(phantom_oracle["input"])
    got = centerline_geometry(inp, line_smoothing=True, output_smoothed_lines=True, vmtk_float32=True)
    assert np.array_equal(got.points, Centerlines.from_npz(phantom_oracle["smoothing"]).points)


def test_circle_and_helix_in_float64():
    t = np.linspace(0, 2.0, 401)
    circle = np.stack([5 * np.cos(t), 5 * np.sin(t), 0 * t], 1)
    k, mean = line_curvature(circle)
    assert np.allclose(k[1:-1], 0.2, rtol=1e-4) and k[0] == 0 == k[-1] and abs(mean - 0.2) < 1e-4
    a, b = 3.0, 1.0                                        # helix: torsion b / (a^2 + b^2)
    helix = np.stack([a * np.cos(t), a * np.sin(t), b * t], 1)
    tau, _ = line_torsion(helix)
    assert np.allclose(tau[2:-2], b / (a * a + b * b), rtol=1e-3) and np.all(tau[:2] == 0)


def test_short_and_closed_lines():
    lines = [np.array([[0, 0, 0], [1, 0, 0.0]]), np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0.0]])]
    with np.errstate(divide="ignore"):
        got = centerline_geometry(Centerlines.from_lines(lines))
    assert got.cell_data["Length"][0] == 1.0 and got.cell_data["Tortuosity"][0] == 0.0
    assert np.isinf(got.cell_data["Tortuosity"][1])         # closed: vmtk divides by 0 too
    assert np.all(got.point_data["FrenetTangent"][:2] == 0)  # two points: no interior frame to copy


def test_row_wise_vtk_math_is_the_scalar_math():
    """The row-wise vtkMath ports the vectorized filters use give the scalar ports' bits, row for row."""
    from thalweg.vmtk import _vtk as K
    rng = np.random.default_rng(3)
    a = rng.normal(size=(200, 3)) * 10.0 ** rng.integers(-8, 8, size=(200, 1))
    b = rng.normal(size=(200, 3))
    a[:3] = 0.0                                               # zero rows: Normalize leaves them
    assert np.array_equal(K.norm_rows(a), [K.norm(x) for x in a])
    assert np.array_equal(K.dot_rows(a, b), [K.dot(x, y) for x, y in zip(a, b)])
    assert np.array_equal(K.cross_rows(a, b), [K.cross(x, y) for x, y in zip(a, b)])
    assert np.array_equal(K.normalized_rows(a), [K.normalized(x) for x in a])
    p = np.cumsum(b, axis=0)
    total = 0.0
    for j in range(1, len(p)):
        total += math.sqrt(K.distance2(p[j - 1], p[j]))
    assert K.polyline_length(p) == total
