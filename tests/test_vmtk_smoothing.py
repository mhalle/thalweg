"""vtkvmtkCenterlineSmoothing port against vmtk 1.5.2 (oracle stage ``smoothing``: 100 iterations, 0.1)."""
import numpy as np

from thalweg.vmtk import Centerlines
from thalweg.vmtk.smoothing import smooth_centerlines, smooth_line


def _structure_equal(a: Centerlines, b: Centerlines):
    assert a.n_points == b.n_points and a.n_cells == b.n_cells
    assert all(np.array_equal(x, y) for x, y in zip(a.cells, b.cells))


def test_smoothing_matches_oracle_bit_for_bit(vmtk_oracle):
    inp = Centerlines.from_npz(vmtk_oracle["input"])
    S = Centerlines.from_npz(vmtk_oracle["smoothing"])
    got = smooth_centerlines(inp, iterations=100, factor=0.1, vmtk_float32=True)
    _structure_equal(got, S)
    assert np.array_equal(got.points, S.points)             # float32 storage reproduced exactly
    assert sorted(got.point_data) == sorted(S.point_data) and not S.cell_data
    for k, v in S.point_data.items():
        assert np.array_equal(got.point_data[k], v)
    assert np.array_equal(inp.points, Centerlines.from_npz(vmtk_oracle["input"]).points)   # input untouched


def test_float64_default_differs_only_by_vmtk_rounding(vmtk_oracle):
    inp = Centerlines.from_npz(vmtk_oracle["input"])
    S = Centerlines.from_npz(vmtk_oracle["smoothing"])
    d = np.abs(smooth_centerlines(inp).points - S.points).max()
    assert 0 < d < 1e-3                                   # the float32 defect, ~1e-4 mm after 100 sweeps


def test_gauss_seidel_order_and_fixed_ends():
    p = np.array([[0, 0, 0], [1, 1, 0], [2, 0, 0], [3, 1, 0.0]])
    q = smooth_line(p, iterations=1, factor=0.5)
    p1 = p[1] + 0.5 * (0.5 * (p[0] + p[2]) - p[1])
    p2 = p[2] + 0.5 * (0.5 * (p1 + p[3]) - p[2])          # uses the already moved p1
    assert np.array_equal(q, np.array([p[0], p1, p2, p[3]]))
    straight = np.linspace([0, 0, 0], [3, 0, 0], 7)
    assert np.allclose(smooth_line(straight, 50, 0.3), straight, atol=1e-15)
