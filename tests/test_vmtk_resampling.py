"""vmtkCenterlineResampling port (vtkCleanPolyData + vtkSplineFilter) against vmtk (``input`` ->
``resampling``, Length 0.5)."""
import numpy as np

from thalweg.vmtk.centerlines import Centerlines
from thalweg.vmtk.resampling import CardinalSpline, clean_lines, resample_centerlines

TOL = 1e-9


def test_resampling_matches_oracle(vmtk_oracle):
    cl = Centerlines.from_npz(vmtk_oracle["input"])
    ref = Centerlines.from_npz(vmtk_oracle["resampling"])
    out = resample_centerlines(cl, 0.5, vmtk_cell_data=True)
    assert out.n_points == ref.n_points and out.n_cells == ref.n_cells
    assert all(np.array_equal(a, b) for a, b in zip(out.cells, ref.cells))
    assert np.abs(out.points - ref.points).max() <= TOL
    assert list(out.point_data) == list(ref.point_data)
    for name, want in ref.point_data.items():
        got = out.point_data[name]
        assert got.dtype == want.dtype and got.shape == want.shape, name
        assert np.abs(got.astype(np.float64) - want.astype(np.float64)).max() <= TOL, name
    assert np.array_equal(out.point_data["TCoords"], ref.point_data["TCoords"])
    assert out.cell_data.keys() == ref.cell_data.keys() and not out.field_data


def test_clean_merges_exact_duplicates_last_data_wins():
    p = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0.0]])
    cl = Centerlines(p, [np.arange(4), np.array([4, 5, 6])],
                     {"r": np.arange(7, dtype=np.float64)})
    pts, lines, verts, pd, cd = clean_lines(cl)
    assert lines == [[0, 1, 2], [0, 1, 3]] and verts == []
    assert np.array_equal(pts, [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]])
    assert np.array_equal(pd["r"], [4.0, 5.0, 3.0, 6.0])


def test_cell_data_defect_flag():
    # as vmtk 1.5.2 / VTK 9.6.2 does on the same cell layout (checked with vmtk): a collapsed line and
    # a one-point cell become vertices, numbered before the lines, and vtkSplineFilter reads the
    # cell data by line index
    rng = np.random.default_rng(0)
    L0 = np.cumsum(rng.normal(size=(12, 3)) * 0.4 + [0.5, 0, 0], axis=0)
    lines = [L0, np.repeat(L0[3:4], 3, axis=0), L0[:9] + [0, 0, 1], np.array([[5.0, 5, 5]]),
             L0[:7] + [0, 1, 0]]
    cl = Centerlines.from_lines(lines)
    cl.cell_data["CenterlineIds"] = np.arange(5, dtype=np.int32) * 10
    assert resample_centerlines(cl, 0.3).cell_data["CenterlineIds"].tolist() == [0, 20, 40]
    vtk = resample_centerlines(cl, 0.3, vmtk_cell_data=True)
    assert vtk.cell_data["CenterlineIds"].tolist() == [10, 30, 0]


def test_cardinal_spline_has_zero_end_slopes():
    s = CardinalSpline()
    for t, x in ((0.0, 0.0), (0.3, 1.0), (1.0, 2.0)):
        s.add_point(t, x)
    h = 1e-7
    assert abs((s.evaluate(h) - s.evaluate(0.0)) / h) < 1e-5
    assert abs((s.evaluate(1.0) - s.evaluate(1.0 - h)) / h) < 1e-5
    # a knot is evaluated on the piece to its left, so knot values come back to rounding only
    assert abs(s.evaluate(0.3) - 1.0) < 1e-12 and abs(s.evaluate(1.0) - 2.0) < 1e-12
