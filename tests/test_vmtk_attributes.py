"""vtkvmtkCenterlineAttributesFilter port against vmtk (stage ``input`` -> ``attributes``)."""
import numpy as np

from thalweg.vmtk.attributes import centerline_attributes
from thalweg.vmtk.centerlines import Centerlines

TOL = 1e-9


def test_attributes_match_oracle(vmtk_oracle):
    cl = Centerlines.from_npz(vmtk_oracle["input"])
    ref = Centerlines.from_npz(vmtk_oracle["attributes"])
    out = centerline_attributes(cl, vmtk_two_point_cells=True)
    assert out.n_points == ref.n_points and out.n_cells == ref.n_cells
    assert all(np.array_equal(a, b) for a, b in zip(out.cells, ref.cells))
    assert np.array_equal(out.points, ref.points)
    assert list(out.point_data) == list(ref.point_data)
    for name, want in ref.point_data.items():
        got = out.point_data[name]
        assert got.dtype == want.dtype and got.shape == want.shape, name
        assert np.abs(got - want).max() <= TOL, name
    assert "Abscissas" not in cl.point_data                     # the input is not mutated


def _two_point_input():
    p = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [5, 5, 5], [6, 5, 5.0]])
    return Centerlines(p, [np.arange(4), np.array([4, 5]), np.array([0, 2, 3])])


def test_vmtk_skips_two_point_cells_and_shared_points_keep_the_last_cell():
    # vtkPolyData types a 2-point line as vtkLine, which the filter's vtkPolyLine cast rejects
    out = centerline_attributes(_two_point_input(), abscissas="A", normals="N", vmtk_two_point_cells=True)
    a, n = out.point_data["A"], out.point_data["N"]
    assert np.array_equal(a[4:], [0.0, 0.0]) and np.array_equal(n[4:], np.zeros((2, 3)))
    assert a[1] == 1.0 and a[2] == np.sqrt(2.0) and a[3] == np.sqrt(2.0) + 1.0   # cell 2 wrote last
    assert np.allclose(np.linalg.norm(n[:4], axis=1), 1.0)
    assert np.array_equal(n[3], n[2])                           # the last point repeats its neighbor


def test_two_point_cells_are_centerlines_by_default():
    cl = _two_point_input()
    out = centerline_attributes(cl, abscissas="A", normals="N")
    vm = centerline_attributes(cl, abscissas="A", normals="N", vmtk_two_point_cells=True)
    a, n = out.point_data["A"], out.point_data["N"]
    assert np.array_equal(a[4:], [0.0, 1.0])
    assert np.allclose(np.linalg.norm(n[4:], axis=1), 1.0) and np.array_equal(n[5], n[4])
    assert abs(np.dot(n[4], [1.0, 0, 0])) < 1e-15                   # normal to the chord
    assert np.array_equal(a[:4], vm.point_data["A"][:4]) and np.array_equal(n[:4], vm.point_data["N"][:4])
