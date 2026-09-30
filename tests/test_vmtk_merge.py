"""vtkvmtkMergeCenterlines port against vmtk (extract -> merge, merge_resampled_unblanked)."""
import numpy as np
import pytest

from thalweg.vmtk import Centerlines
from thalweg.vmtk._vtk import CardinalSpline
from thalweg.vmtk.merge import merge_centerlines


@pytest.mark.parametrize("stage,kw", [("merge", dict(length=0.0, merge_blanked=True)),
                                      ("merge_resampled_unblanked", dict(length=0.5, merge_blanked=False))])
def test_merge_matches_oracle(vmtk_oracle, stage, kw):
    E = Centerlines.from_npz(vmtk_oracle["extract"])
    Ref = Centerlines.from_npz(vmtk_oracle[stage])
    R = merge_centerlines(E, **kw, vmtk_float32=True, vmtk_cell_data=True)
    assert R.n_points == Ref.n_points and R.n_cells == Ref.n_cells
    assert all(np.array_equal(a, b) for a, b in zip(R.cells, Ref.cells))
    np.testing.assert_allclose(R.points, Ref.points, rtol=0, atol=1e-9)    # NaN at zero radii, as vmtk
    assert list(R.point_data) == list(Ref.point_data)
    for k, v in Ref.point_data.items():
        assert R.point_data[k].dtype == v.dtype and R.point_data[k].shape == v.shape, k
        np.testing.assert_allclose(R.point_data[k], v, rtol=0, atol=1e-9, err_msg=k)
    assert list(R.cell_data) == list(Ref.cell_data)
    for k, v in Ref.cell_data.items():
        assert R.cell_data[k].dtype == v.dtype
        np.testing.assert_array_equal(R.cell_data[k], v)


def test_cardinal_spline_interpolates_knots_with_zero_end_slopes():
    x = [0.0, 0.3, 0.55, 1.0]
    y = [1.0, 2.0, -1.0, 0.5]
    s = CardinalSpline()
    for xi, yi in zip(x, y):
        s.add_point(xi, yi)
    for xi, yi in zip(x, y):
        assert abs(s.evaluate(xi) - yi) < 1e-14
    h = 1e-7
    assert abs((s.evaluate(h) - y[0]) / h) < 1e-5
    assert abs((y[-1] - s.evaluate(1.0 - h)) / h) < 1e-5


def test_zero_radius_gives_nan_points_not_an_exception():
    """All weights 0: vmtk divides 0 by 0 and emits NaN points (frames: NaN origins); so does the port."""
    import warnings

    from thalweg.vmtk import extract_branches
    from thalweg.vmtk.frames import bifurcation_reference_systems
    x = np.linspace(0, 10, 34)
    line = np.stack([x, 0 * x, 0 * x], 1)
    side = np.r_[line[:17], line[16] + np.cumsum(np.tile([[0, 0.3, 0]], (15, 1)), 0)]
    E = extract_branches(Centerlines.from_lines([line, side], [np.zeros(34), np.zeros(32)]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        F = bifurcation_reference_systems(E)
        M = merge_centerlines(E)
    assert np.isnan(F.points).all()
    assert M.n_cells > 0 and np.isnan(M.points).all()
