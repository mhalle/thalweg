"""vtkvmtkCenterlineBranchGeometry port against vmtk 1.5.2 (stage ``branch_geometry`` on ``extract``).

Achieved (vmtk_float32=True): structure and Length exact; Curvature, Torsion <= 5e-16, Tortuosity
<= 1.2e-13 (fused multiply-adds in vmtk's build). The touching-sphere subsampling is reproduced step
for step: any different step would move curvature by ~1e-6.
"""
import warnings

import numpy as np
import pytest

from thalweg.vmtk import Centerlines
from thalweg.vmtk import utilities as U
from thalweg.vmtk.branch_geometry import branch_geometry, sphere_subsample_line, subsample_line


def test_branch_geometry_matches_oracle(vmtk_oracle):
    E = Centerlines.from_npz(vmtk_oracle["extract"])
    B = Centerlines.from_npz(vmtk_oracle["branch_geometry"])
    got = branch_geometry(E, vmtk_float32=True, vmtk_steps=True, vmtk_discard_smoothing=True)
    assert got.n_points == B.n_points and got.n_cells == B.n_cells
    assert all(np.array_equal(a, b) for a, b in zip(got.cells, B.cells))
    assert np.array_equal(got.points, B.points)
    assert sorted(got.point_data) == sorted(B.point_data) and not B.cell_data
    assert np.array_equal(got.point_data["GroupIds"], B.point_data["GroupIds"])
    assert got.point_data["GroupIds"].tolist() == U.get_non_blanked_groups_id_list(E)
    assert np.array_equal(got.point_data["Length"], B.point_data["Length"])
    for k in ("Curvature", "Torsion", "Tortuosity"):
        a, b = got.point_data[k], B.point_data[k]
        assert np.array_equal(np.isnan(a), np.isnan(b)), k        # NaN: zero end radii (degenerate_radii)
        ok = ~np.isnan(b)
        scale = np.maximum(np.abs(b[ok]), 1.0) if k == "Torsion" else 1.0   # torsion: relative where huge
        assert np.all(np.abs(a[ok] - b[ok]) <= 1e-9 * scale), (k, np.abs(a[ok] - b[ok]).max())


def test_line_smoothing_is_discarded_by_vmtk(phantom_oracle):
    E = Centerlines.from_npz(phantom_oracle["extract"])
    plain = branch_geometry(E, vmtk_float32=True, vmtk_steps=True)
    vmtk = branch_geometry(E, line_smoothing=True, vmtk_float32=True, vmtk_steps=True,
                           vmtk_discard_smoothing=True)
    fixed = branch_geometry(E, line_smoothing=True)
    for k in plain.point_data:                             # checked against vmtk: LineSmoothing 1 == 0
        assert np.array_equal(plain.point_data[k], vmtk.point_data[k])
    assert not np.allclose(plain.point_data["Curvature"], fixed.point_data["Curvature"], rtol=1e-3)
    assert np.array_equal(plain.point_data["Length"], fixed.point_data["Length"])


def test_sphere_subsample_steps_one_radius():
    x = np.linspace(0, 10, 101)                           # 0.1 spacing, radius 1: a point every 1.0
    line = Centerlines.from_lines([np.stack([x, 0 * x, 0 * x], 1)], [np.ones(101)])
    q = sphere_subsample_line(line, 0)
    assert np.allclose(np.diff(q[:, 0])[:-1], 1.0, atol=1e-5) and q[-1, 0] == 10.0


def test_default_is_close_to_vmtk(phantom_oracle):
    E = Centerlines.from_npz(phantom_oracle["extract"])
    B = Centerlines.from_npz(phantom_oracle["branch_geometry"])
    got = branch_geometry(E)                              # float64, step count from the true length
    for k in ("Curvature", "Torsion"):
        assert np.abs(got.point_data[k] - B.point_data[k]).max() < 1e-2, k


def test_subsample_line_keeps_ends():
    pts = np.stack([np.linspace(0, 1, 11), np.zeros(11), np.zeros(11)], 1)
    q = subsample_line(pts, 0.35)
    assert np.array_equal(q[0], pts[0]) and np.array_equal(q[-1], pts[-1]) and 2 < len(q) < 11


def test_zero_end_radii_give_nan_tortuosity_silently():
    x = np.linspace(0, 10, 34)
    line = Centerlines.from_lines([np.stack([x, np.sin(x), 0 * x], 1)], [np.r_[0.0, np.ones(33)]])
    line.cell_data = {"GroupIds": np.array([0]), "Blanking": np.array([0])}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        g = branch_geometry(line)
    assert np.isnan(g.point_data["Tortuosity"][0]) and np.isfinite(g.point_data["Curvature"][0])


def test_zero_radius_segment_cannot_be_sphere_subsampled():
    x = np.linspace(0, 10, 34)
    line = Centerlines.from_lines([np.stack([x, 0 * x, 0 * x], 1)], [np.r_[np.ones(20), np.zeros(14)]])
    with pytest.raises(ValueError, match="mean radius <= 0"):
        sphere_subsample_line(line, 0)
    line.point_data["MaximumInscribedSphereRadius"][20] = 0.5         # a zero TIP is fine (vmtk too)
    line.point_data["MaximumInscribedSphereRadius"][21:] = 0.5
    line.point_data["MaximumInscribedSphereRadius"][-1] = 0.0
    assert len(sphere_subsample_line(line, 0)) > 2
