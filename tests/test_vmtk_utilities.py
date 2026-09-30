"""vtkvmtkCenterlineUtilities ports against vmtk's split centerlines (ids only: exact)."""
import numpy as np

from thalweg.vmtk import Centerlines
from thalweg.vmtk import utilities as U


def test_group_lists_match_oracle_groups(vmtk_oracle):
    E = Centerlines.from_npz(vmtk_oracle["extract"])
    g, b = E.cell_data["GroupIds"], E.cell_data["Blanking"]
    assert sorted(U.get_non_blanked_groups_id_list(E)) == sorted(set(g[b == 0].tolist()))
    assert sorted(U.get_blanked_groups_id_list(E)) == sorted(set(g[b == 1].tolist()))
    assert U.get_max_group_id(E) == int(g.max())


def test_adjacent_groups_are_bifurcation_neighbors(vmtk_oracle):
    """Each bifurcation's neighbors are the groups just before / after it along its centerlines (a
    re-entering daughter or a hairpin can put one group on both sides, or several upstream)."""
    E = Centerlines.from_npz(vmtk_oracle["extract"])
    g, c, t = (E.cell_data[k].tolist() for k in ("GroupIds", "CenterlineIds", "TractIds"))
    for bif in U.get_blanked_groups_id_list(E):
        up, down = U.find_adjacent_centerline_group_ids(E, bif)
        mine = [(c[i], t[i]) for i in range(E.n_cells) if g[i] == bif]
        want_up = {g[j] for j in range(E.n_cells) for ci, ti in mine if c[j] == ci and t[j] == ti - 1}
        want_down = {g[j] for j in range(E.n_cells) for ci, ti in mine if c[j] == ci and t[j] == ti + 1}
        assert set(up) == want_up - {bif} and set(down) == want_down - {bif}
        assert len(up) == len(set(up)) and len(down) == len(set(down))
        assert len(up) >= 1 and len(down) >= 1
        assert all(U.is_group_blanked(E, x) == 0 for x in up + down)


def test_adjacent_groups_include_one_point_cells():
    """vtkPolyData types a one-point line cell VTK_POLY_LINE, so FindAdjacentCenterlineGroupIds takes
    it (a bifurcation group whose only cell collapsed to one point still has neighbors)."""
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0.0]])
    cl = Centerlines(pts, [np.array([0, 1]), np.array([2]), np.array([3, 4])])
    cl.cell_data = {"GroupIds": np.array([0, 1, 2]), "CenterlineIds": np.zeros(3, int),
                    "TractIds": np.arange(3), "Blanking": np.array([0, 1, 0])}
    assert U.find_adjacent_centerline_group_ids(cl, 1) == ([0], [2])


def test_interpolate_tuple_defect_flag():
    cl = Centerlines.from_lines([np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0.0]])], [np.array([1.0, 2.0, 4.0])])
    assert U.interpolate_tuple(cl, "MaximumInscribedSphereRadius", 0, 1, 0.25)[0] == 2.5
    assert U.interpolate_tuple(cl, "MaximumInscribedSphereRadius", 0, 1, 0.25, vmtk_interp=True)[0] == 4.0


def test_polyline_evaluate_position():
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0.0]])
    status, closest, sub, pc, d2 = U.polyline_evaluate_position(np.array([1.2, 0.5, 0.0]), pts)
    assert status == 1 and sub == 1 and abs(pc - 0.5) < 1e-15 and abs(d2 - 0.04) < 1e-15
