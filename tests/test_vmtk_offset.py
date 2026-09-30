"""vtkvmtkCenterlineReferenceSystemAttributesOffset port against vmtk (extract + frames -> offset).

The oracles' offset stages are vmtk's own defaults: ``offset`` with ReferenceGroupId -1 (the first
frame's group, a bifurcation; recorded as ``root_group`` in oracle.json) and ReplaceAttributes on,
``offset_keep`` with that group and ReplaceAttributes off. So the VALID path is what they test,
against every covered point: vmtk leaves the points of centerlines that never cross the reference
group uninitialized (``SetNumberOfTuples`` without a fill), the port NaN, and those are compared
for coverage only. A fixture without a bifurcation (``single_line``) has no frames and no offset
stage (vmtk logs "ReferenceSystems empty" and outputs nothing; the port raises). The invalid-group
path (a group with no frame) is pinned by ``test_invalid_reference_group_returns_the_input``.
"""
import warnings

import numpy as np
import pytest

from thalweg.vmtk import Centerlines, ReferenceSystems
from thalweg.vmtk import polyball as PB
from thalweg.vmtk import utilities as U
from thalweg.vmtk._vtk import rotate_normal
from thalweg.vmtk.frames import bifurcation_reference_systems
from thalweg.vmtk.offset import offset_attributes


def _covered(E: Centerlines, group: int) -> np.ndarray:
    """Points of every centerline that crosses ``group`` (the ones the filter writes)."""
    cids = {int(E.cell_data["CenterlineIds"][c]) for c in U.get_group_cell_ids(E, group)}
    out = np.zeros(E.n_points, bool)
    for c in range(E.n_cells):
        if int(E.cell_data["CenterlineIds"][c]) in cids:
            out[E.cells[c]] = True
    return out


def _assert_same(R: Centerlines, Ref: Centerlines, covered: np.ndarray, changed=()):
    """R equals vmtk's Ref to 1e-9; the ``changed`` arrays only on the ``covered`` points."""
    assert R.n_points == Ref.n_points and R.n_cells == Ref.n_cells
    assert all(np.array_equal(a, b) for a, b in zip(R.cells, Ref.cells))
    np.testing.assert_allclose(R.points, Ref.points, rtol=0, atol=1e-9)
    assert list(R.point_data) == list(Ref.point_data)
    for k in Ref.point_data:
        a, b = R.point_data[k], Ref.point_data[k]
        if k in changed:
            a, b = a[covered], b[covered]
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9, err_msg=k)
    for k in Ref.cell_data:
        np.testing.assert_array_equal(R.cell_data[k], Ref.cell_data[k])


def _frames(vmtk_oracle) -> ReferenceSystems:
    F = ReferenceSystems.from_npz(vmtk_oracle["frames"])
    if len(F.points) == 0:
        pytest.skip("no bifurcation: no reference system")
    return F


@pytest.mark.parametrize("stage,replace", [("offset", True), ("offset_keep", False)])
def test_offset_matches_oracle(vmtk_oracle, stage, replace):
    E = Centerlines.from_npz(vmtk_oracle["extract"])
    F = _frames(vmtk_oracle)
    Ref = Centerlines.from_npz(vmtk_oracle[stage])
    group = vmtk_oracle.meta["root_group"]
    assert group == int(F.point_data["GroupIds"][0])         # -1: the first frame's group, a bifurcation
    with warnings.catch_warnings():
        warnings.simplefilter("error")                          # the valid path warns nothing
        R = offset_attributes(E, F, group, replace_attributes=replace, vmtk_interp=True)
    covered = _covered(E, group)
    assert covered.any()
    changed = ("Abscissas", "ParallelTransportNormals") if replace else ("OffsetAbscissas", "OffsetNormals")
    for k in changed:                                           # uncovered: NaN here, uninitialized in vmtk
        assert np.all(np.isnan(R.point_data[k][~covered])) and not np.isnan(R.point_data[k][covered]).any()
    _assert_same(R, Ref, covered, changed)


def test_invalid_reference_group_returns_the_input(frozen_oracle):
    """A group with no frame (here a branch): vmtk logs "Invalid ReferenceGroupId" and returns its
    deep copy of the input; the port warns once and returns an unchanged copy."""
    E = Centerlines.from_npz(frozen_oracle["extract"])
    F = _frames(frozen_oracle)
    branch = U.get_non_blanked_groups_id_list(E)[0]
    assert branch not in F.point_data["GroupIds"].tolist()
    for replace in (True, False):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            R = offset_attributes(E, F, branch, replace_attributes=replace)
        assert len(w) == 1 and "Invalid ReferenceGroupId" in str(w[0].message)
        assert R is not E and list(R.point_data) == list(E.point_data)
        assert np.array_equal(R.points, E.points)
        assert all(np.array_equal(R.point_data[k], v) for k, v in E.point_data.items())
        assert all(np.array_equal(R.cell_data[k], v) for k, v in E.cell_data.items())


@pytest.mark.parametrize("vmtk_interp", [True, False])
def test_offset_chained_from_own_frames(vmtk_oracle, vmtk_interp):
    """Own frames -> offset at the first bifurcation: the reference cells' closest point is at abscissa 0."""
    E = Centerlines.from_npz(vmtk_oracle["extract"])
    F = bifurcation_reference_systems(E)
    if len(F.points) == 0:
        pytest.skip("no bifurcation: no reference system")
    g = int(F.point_data["GroupIds"][0])
    R = offset_attributes(E, F, -1, replace_attributes=False, vmtk_interp=vmtk_interp)
    assert list(R.point_data)[-2:] == ["OffsetAbscissas", "OffsetNormals"]
    np.testing.assert_array_equal(R.point_data["Abscissas"], E.point_data["Abscissas"])
    covered = ~np.isnan(R.point_data["OffsetAbscissas"])
    np.testing.assert_array_equal(covered, _covered(E, g))
    nrm = R.point_data["OffsetNormals"][covered]
    np.testing.assert_allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-12)
    origin = F.points[0]
    for c in U.get_group_cell_ids(E, g):
        state = PB.evaluate_function(origin, PB.tube_segments(E, None, [c]))     # no radius: closest point
        sub, pc = int(state.sub[0]), float(state.pcoord[0])
        a = U.interpolate_tuple(R, "OffsetAbscissas", c, sub, pc, vmtk_interp)[0]
        assert abs(a) < 1e-12


def test_offset_replace_keeps_array_order(vmtk_oracle):
    E = Centerlines.from_npz(vmtk_oracle["extract"])
    F = _frames(vmtk_oracle)
    R = offset_attributes(E, F, -1, replace_attributes=True, vmtk_interp=True)
    assert list(R.point_data) == list(E.point_data)
    assert not np.array_equal(R.point_data["Abscissas"], E.point_data["Abscissas"])
    assert E.point_data["Abscissas"] is not R.point_data["Abscissas"]


def test_rotate_normal_is_a_rotation():
    axis = np.array([0.2, -0.5, 0.8])
    axis /= np.linalg.norm(axis)
    n = np.cross(axis, [1.0, 0, 0])
    n /= np.linalg.norm(n)
    r = np.array(rotate_normal(90.0, list(axis), list(n)))
    np.testing.assert_allclose(r, np.cross(axis, n), atol=1e-15)
    np.testing.assert_allclose(rotate_normal(0.0, list(axis), list(n)), n, atol=0)
