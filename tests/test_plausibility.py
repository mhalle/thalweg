"""Artery/vein plausibility against the heart classes (data)."""
import pytest

from thalweg.plausibility import check


@pytest.mark.data
@pytest.mark.slow
@pytest.mark.parametrize("run", ["MSB-02664_ctape0625_v013", "C3N-00704_ctpa0625"])
def test_real_trees_are_plausible_and_a_swap_is_not(vessels_data, run):
    """Named store and misnamed store (crop classes as label_<value>): the veins run through the
    pulmonary vein class and the arteries do not; calling them by each other's name fails."""
    from thalweg.centerlines import centerline_graph, combine
    from thalweg.store import open_store
    st = open_store(vessels_data / "runs" / f"{run}.lung_vessels.duckn.zip")
    g = combine([centerline_graph(st, "lung_arteries"), centerline_graph(st, "lung_veins")])
    c = check(g, st)
    assert c["plausible"] is True and c["reasons"] == []
    assert c["vein_length_share_inside_pulmonary_vein"] > 0.03
    assert c["artery_length_share_inside_pulmonary_vein"] < 0.01
    swapped = check(g, st, arteries="lung_veins", veins="lung_arteries")
    assert swapped["plausible"] is False and swapped["reasons"]


def test_needs_both_trees(phantom_oracle):
    from thalweg.graph import TubeGraph
    c = check(TubeGraph(), None)
    assert c["plausible"] is None and "needs both" in c["reason"]
