"""Graph -> vmtk convention, against the exact polylines the saved vmtk runs were fed (data)."""
import numpy as np
import pytest

from cases import reference_source
from thalweg.adapters import densify, to_vmtk
from thalweg.vmtk import Centerlines


def test_densify_keeps_start_and_step():
    p = np.array([[0, 0, 0], [1.0, 0, 0], [1.0, 1.0, 0]])
    q = densify(p, step=0.3)
    assert np.allclose(q[0], p[0])
    # every sample on the polyline, 0.3 mm of arc apart, the last within one step of the end
    on = np.minimum(np.abs(q[:, 1]) + np.abs(q[:, 2]), np.abs(q[:, 0] - 1.0) + np.abs(q[:, 2]))
    assert on.max() < 1e-12
    s = np.where(q[:, 1] == 0, q[:, 0], 1.0 + q[:, 1])
    assert np.allclose(np.diff(s), 0.3) and 2.0 - s[-1] < 0.3


@pytest.mark.data
@pytest.mark.slow
def test_subtree_matches_the_oracle_input(vessels_data, case_oracle):
    """The C3N-00704 left-lung subtree: our graph, written in vmtk's convention with the reference's
    source (the segment research/vessels/vmtk_prep.py chose, trimmed one radius + 1 mm), must give
    the oracle's input polylines and radii exactly."""
    from thalweg.centerlines import centerline_graph
    g = centerline_graph(vessels_data / "runs" / "C3N-00704_ctpa0625.lung_vessels.duckn.zip", "lung_arteries",
                         ridge_passes=1, root="deepest")        # the graph vmtk was fed
    paths = to_vmtk(g, "lung_arteries", source=reference_source(g))
    ref = Centerlines.from_npz(case_oracle["input"])
    assert paths.centerlines.n_cells == ref.n_cells == 51
    assert [len(c) for c in paths.centerlines.cells] == [len(c) for c in ref.cells]
    assert np.array_equal(paths.centerlines.points, ref.points)
    assert np.array_equal(paths.centerlines.point_data["MaximumInscribedSphereRadius"],
                          ref.point_data["MaximumInscribedSphereRadius"])
