"""vtkvmtkCenterlineBifurcationReferenceSystems port against vmtk's own frames (extract -> frames)."""
import numpy as np

from thalweg.vmtk import Centerlines, ReferenceSystems
from thalweg.vmtk._vtk import perpendiculars
from thalweg.vmtk.frames import bifurcation_reference_systems


def test_frames_match_oracle(vmtk_oracle):
    E = Centerlines.from_npz(vmtk_oracle["extract"])
    F = ReferenceSystems.from_npz(vmtk_oracle["frames"])
    R = bifurcation_reference_systems(E, vmtk_float32=True)
    assert R.points.shape == F.points.shape
    assert R.point_data["GroupIds"].dtype == np.int32
    np.testing.assert_array_equal(R.point_data["GroupIds"], F.point_data["GroupIds"])
    # float32 origins: bit-identical in practice; NaN where every end radius is 0, as in vmtk
    np.testing.assert_allclose(R.points, F.points, rtol=0, atol=1e-9)
    for k in ("Normal", "UpNormal"):
        assert R.point_data[k].shape == F.point_data[k].shape
        np.testing.assert_allclose(R.point_data[k], F.point_data[k], rtol=0, atol=1e-9)


def test_frames_are_orthonormal(vmtk_oracle):
    R = bifurcation_reference_systems(Centerlines.from_npz(vmtk_oracle["extract"]))
    n, u = R.point_data["Normal"], R.point_data["UpNormal"]
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(u, axis=1), 1.0, atol=1e-12)
    assert np.all(np.abs(np.einsum("ij,ij->i", n, u)) < 1e-12)


def test_single_cell_group_uses_perpendiculars():
    """A blanked group with one unique cell: up normal = the cell direction, normal = Perpendiculars."""
    line = np.array([[0.0, 0, 0], [0.5, 0.25, 1.0], [1.0, 0.5, 2.0]])   # float32-exact, as vmtk stores them
    cl = Centerlines.from_lines([line], [np.array([1.0, 1.0, 1.0])])
    cl.cell_data = {"GroupIds": np.array([0], np.int32), "Blanking": np.array([1], np.int32)}
    R = bifurcation_reference_systems(cl)
    up = (line[-1] - line[0]) / np.linalg.norm(line[-1] - line[0])
    np.testing.assert_allclose(R.point_data["UpNormal"][0], up, atol=1e-15)
    np.testing.assert_allclose(R.point_data["Normal"][0], perpendiculars(list(up))[0], atol=0)
    assert abs(R.point_data["Normal"][0] @ up) < 1e-15
