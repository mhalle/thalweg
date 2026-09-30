"""vtkvmtkCenterlineBifurcationVectors port against vmtk (extract + frames -> bifurcation_vectors)."""
import math

import numpy as np
import pytest

from thalweg.vmtk import VMTK_FLAGS, Centerlines, ReferenceSystems
from thalweg.vmtk.sphere_distance import find_touching_sphere_center
from thalweg.vmtk.vectors import bifurcation_vectors

FLOATS = ["BifurcationVectors", "InPlaneBifurcationVectors", "OutOfPlaneBifurcationVectors",
          "InPlaneBifurcationVectorAngles", "OutOfPlaneBifurcationVectorAngles"]
INTS = ["BifurcationVectorsOrientation", "GroupIds", "BifurcationGroupIds"]


@pytest.fixture(scope="module")
def inputs(vmtk_oracle):
    return (Centerlines.from_npz(vmtk_oracle["extract"]), ReferenceSystems.from_npz(vmtk_oracle["frames"]),
            Centerlines.from_npz(vmtk_oracle["bifurcation_vectors"]))


def test_matches_vmtk(inputs):
    E, F, V = inputs
    B = bifurcation_vectors(E, F, **{f: True for f in VMTK_FLAGS["bifurcation_vectors"]})
    assert B.n_points == V.n_points and B.n_cells == V.n_cells
    assert all(np.array_equal(b, o) for b, o in zip(B.cells, V.cells))
    assert set(B.point_data) == set(V.point_data)
    assert np.array_equal(B.points, V.points)                 # both float32-rounded
    for k in INTS:
        assert B.point_data[k].dtype == np.int32 and np.array_equal(B.point_data[k], V.point_data[k]), k
    for k in FLOATS:
        np.testing.assert_allclose(B.point_data[k], V.point_data[k], rtol=0, atol=1e-9, err_msg=k)


def test_inputs_not_mutated(inputs):
    E, F, _ = inputs
    before = (E.points.copy(), {k: v.copy() for k, v in E.point_data.items()}, F.points.copy())
    bifurcation_vectors(E, F)
    assert np.array_equal(E.points, before[0]) and np.array_equal(F.points, before[2])
    assert all(np.array_equal(E.point_data[k], v) for k, v in before[1].items())


def test_fallback_defect(inputs):
    """vmtk's upstream fallback gives zero vectors for parents shorter than their radius; ours walks on
    upstream (a nonzero vector) or, off the centerlines' start, gives NaN - never a zero vector."""
    E, F, V = inputs
    B = bifurcation_vectors(E, F)
    norm = np.linalg.norm(B.point_data["BifurcationVectors"], axis=1)
    assert not np.any(norm == 0.0)
    zero = np.linalg.norm(V.point_data["BifurcationVectors"], axis=1) == 0.0
    assert np.all(V.point_data["BifurcationVectorsOrientation"][zero] == 0)
    assert np.all(zero[np.isnan(norm)])                      # NaN only where vmtk fell back


def test_normalize(inputs):
    E, F, _ = inputs
    A = bifurcation_vectors(E, F)
    B = bifurcation_vectors(E, F, normalize=True)
    np.testing.assert_allclose(np.linalg.norm(B.point_data["BifurcationVectors"], axis=1),
                               np.where(np.isnan(A.point_data["BifurcationVectors"][:, 0]), np.nan, 1.0),
                               rtol=0, atol=1e-12)
    for k in ("InPlaneBifurcationVectorAngles", "OutOfPlaneBifurcationVectorAngles"):
        np.testing.assert_allclose(A.point_data[k], B.point_data[k], rtol=0, atol=1e-12)


def _y_bifurcation():
    """Parent along +x to the origin, a bifurcation piece per centerline, then daughter A along
    (1, 1, 0) and daughter B along (1, -1, 0.3); radius 0.95 everywhere (off the 0.1 sampling, so no
    sphere passes exactly through a sample). Reference system N = +z, U = +x."""
    s = np.arange(0, 10.0001, 0.1)[:, None]
    parent = np.array([-10.0, 0, 0]) + s * np.array([1.0, 0, 0])
    da, db = np.array([1.0, 1, 0]) / math.sqrt(2), np.array([1.0, -1, 0.3]) / math.sqrt(2.09)
    t = np.arange(0, 1.0001, 0.1)[:, None]
    lines = [parent, t * da * 2, 2 * da + s * da,       # centerline 0: groups 0, 1, 2
             parent, t * db * 2, 2 * db + s * db]       # centerline 1: groups 0, 1, 3
    cl = Centerlines.from_lines(lines, [np.full(len(L), 0.95) for L in lines])
    cl.cell_data = {"GroupIds": np.array([0, 1, 2, 0, 1, 3]), "CenterlineIds": np.array([0, 0, 0, 1, 1, 1]),
                    "TractIds": np.array([0, 1, 2, 0, 1, 2]), "Blanking": np.array([0, 1, 0, 0, 1, 0])}
    rs = ReferenceSystems(np.zeros((1, 3)), {"Normal": np.array([[0.0, 0, 1]]),
                                             "UpNormal": np.array([[1.0, 0, 0]]), "GroupIds": np.array([1])})
    return cl, rs, da, db


def test_angle_conventions():
    cl, rs, da, db = _y_bifurcation()
    B = bifurcation_vectors(cl, rs, normalize=True)
    assert B.point_data["GroupIds"].tolist() == [0, 2, 3]
    assert B.point_data["BifurcationVectorsOrientation"].tolist() == [0, 1, 1]
    v = B.point_data["BifurcationVectors"]
    assert np.allclose(v[0], [1, 0, 0]) and np.allclose(v[1], da) and np.allclose(v[2], db)
    a_in = B.point_data["InPlaneBifurcationVectorAngles"]
    a_out = B.point_data["OutOfPlaneBifurcationVectorAngles"]
    # +y is a counterclockwise turn from U about N: negative; -y is clockwise: positive
    assert np.allclose(a_in, [0.0, -math.pi / 4, math.pi / 4])
    assert np.allclose(a_out, [0.0, 0.0, math.asin(0.3 / math.sqrt(2.09))])
    # the daughter's tail is its first point; the parent's is its touching point, one radius back
    assert np.allclose(B.points, [[-0.95, 0, 0], 2 * da, 2 * db], atol=1e-5)


def test_touching_sphere_on_a_straight_line():
    x = np.arange(0, 5.01, 0.5)
    cl = Centerlines.from_lines([np.c_[x, 0 * x, 0 * x]], [np.full(11, 0.9)])
    sub, pc = find_touching_sphere_center(cl, "MaximumInscribedSphereRadius", 0, 9, 1.0, True)
    assert sub == 8 and abs(pc - 0.2) < 2e-5             # x = 4.1, one radius before x = 5
    sub, pc = find_touching_sphere_center(cl, "MaximumInscribedSphereRadius", 0, 0, 0.0, False)
    assert sub == 1 and abs(pc - 0.8) < 2e-5             # x = 0.9, one radius after x = 0
    short = Centerlines.from_lines([np.array([[0.0, 0, 0], [0.5, 0, 0]])], [np.ones(2)])
    assert find_touching_sphere_center(short, "MaximumInscribedSphereRadius", 0, 0, 1.0, True) == (-1, 0.0)


def _short_parent_after_trunk():
    """A Y whose parent (group 2, 0.5 mm) is shorter than its radius (0.95) but follows a trunk
    (group 0) and an upstream bifurcation piece (group 1, blanked): no touching point on the parent
    itself, so the walk continues into group 1. Two centerlines, daughters as in _y_bifurcation."""
    def xs(a, b, n):
        return np.c_[np.linspace(a, b, n), np.zeros((n, 2))]
    s = np.arange(0, 10.0001, 0.1)[:, None]
    t = np.arange(0, 1.0001, 0.1)[:, None]
    da, db = np.array([1.0, 1, 0]) / math.sqrt(2), np.array([1.0, -1, 0.3]) / math.sqrt(2.09)
    stem = [xs(-10, -2, 81), xs(-2, -0.5, 16), xs(-0.5, 0, 6)]
    lines = stem + [t * da * 2, 2 * da + s * da] + stem + [t * db * 2, 2 * db + s * db]
    cl = Centerlines.from_lines(lines, [np.full(len(L), 0.95) for L in lines])
    cl.cell_data = {"GroupIds": np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 5]),
                    "CenterlineIds": np.repeat([0, 1], 5), "TractIds": np.tile(np.arange(5), 2),
                    "Blanking": np.tile([0, 1, 0, 1, 0], 2)}
    rs = ReferenceSystems(np.array([[-1.2, 0, 0], [0.0, 0, 0]]),
                          {"Normal": np.array([[0.0, 0, 1], [0.0, 0, 1]]),
                           "UpNormal": np.array([[1.0, 0, 0], [1.0, 0, 0]]), "GroupIds": np.array([1, 3])})
    return cl, rs


@pytest.mark.parametrize("vmtk_interp", [False, True])
@pytest.mark.parametrize("vmtk_steps", [False, True])
def test_short_parent_walks_into_the_preceding_tracts(vmtk_steps, vmtk_interp):
    """The default fallback: one inscribed sphere upstream along the centerline, not the cell's start
    (0.5 mm back) and not vmtk's end point (v = 0)."""
    cl, rs = _short_parent_after_trunk()
    B = bifurcation_vectors(cl, rs, vmtk_steps=vmtk_steps, vmtk_interp=vmtk_interp)
    row = int(np.flatnonzero((B.point_data["BifurcationGroupIds"] == 3) & (B.point_data["GroupIds"] == 2))[0])
    assert B.point_data["BifurcationVectorsOrientation"][row] == 0
    np.testing.assert_allclose(B.point_data["BifurcationVectors"][row], [0.95, 0, 0], atol=1e-5)
    np.testing.assert_allclose(B.points[row], [-0.95, 0, 0], atol=1e-5)       # on the blanked tract
    assert abs(B.point_data["InPlaneBifurcationVectorAngles"][row]) < 1e-12
    assert abs(B.point_data["OutOfPlaneBifurcationVectorAngles"][row]) < 1e-12
    vm = bifurcation_vectors(cl, rs, vmtk_fallback=True, vmtk_steps=vmtk_steps, vmtk_interp=vmtk_interp)
    assert np.array_equal(vm.point_data["BifurcationVectors"][row], np.zeros(3))
    assert np.array_equal(vm.points[row], [0.0, 0, 0])
    others = np.arange(B.n_points) != row                  # every other row is the same in both modes
    for k in ("BifurcationVectors", "InPlaneBifurcationVectorAngles", "OutOfPlaneBifurcationVectorAngles"):
        assert np.array_equal(B.point_data[k][others], vm.point_data[k][others]), k


def test_short_parent_at_the_centerline_start_is_nan():
    """A parent inside its sphere all the way back to the centerlines' start: no touching point, so
    that row is NaN (vector, components, angles and tail); the daughters are unaffected."""
    s = np.arange(0, 10.0001, 0.1)[:, None]
    parent = np.array([-0.5, 0, 0]) + np.arange(0, 0.5001, 0.1)[:, None] * np.array([1.0, 0, 0])
    da, db = np.array([1.0, 1, 0]) / math.sqrt(2), np.array([1.0, -1, 0.3]) / math.sqrt(2.09)
    t = np.arange(0, 1.0001, 0.1)[:, None]
    lines = [parent, t * da * 2, 2 * da + s * da, parent, t * db * 2, 2 * db + s * db]
    cl = Centerlines.from_lines(lines, [np.full(len(L), 0.95) for L in lines])
    cl.cell_data = {"GroupIds": np.array([0, 1, 2, 0, 1, 3]), "CenterlineIds": np.array([0, 0, 0, 1, 1, 1]),
                    "TractIds": np.array([0, 1, 2, 0, 1, 2]), "Blanking": np.array([0, 1, 0, 0, 1, 0])}
    rs = ReferenceSystems(np.zeros((1, 3)), {"Normal": np.array([[0.0, 0, 1]]),
                                             "UpNormal": np.array([[1.0, 0, 0]]), "GroupIds": np.array([1])})
    B = bifurcation_vectors(cl, rs)
    assert B.point_data["GroupIds"].tolist() == [0, 2, 3]
    for k in ("BifurcationVectors", "InPlaneBifurcationVectors", "OutOfPlaneBifurcationVectors",
              "InPlaneBifurcationVectorAngles", "OutOfPlaneBifurcationVectorAngles"):
        assert np.isnan(B.point_data[k][0]).all() and np.isfinite(B.point_data[k][1:]).all(), k
    assert np.isnan(B.points[0]).all() and np.isfinite(B.points[1:]).all()
    vm = bifurcation_vectors(cl, rs, vmtk_fallback=True)
    assert np.array_equal(vm.point_data["BifurcationVectors"][0], np.zeros(3))
