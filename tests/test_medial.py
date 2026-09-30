"""The seed-free tracer: on an analytic Y tube (always), and against the research reference (data)."""
import json

import numpy as np
import pytest

from thalweg.kernel import medial
from phantoms import tube_field, y_tree


@pytest.fixture(scope="module")
def y_trace():
    lines, radii = y_tree()
    m, geo = tube_field(lines, radii)
    return medial.trace(m, geo), lines, radii


def test_y_topology(y_trace):
    T, lines, _ = y_trace
    kinds = [n["kind"] for n in T.nodes]
    assert kinds.count("root") == 1 and kinds.count("junction") == 1
    # the root is the deepest point; on a constant-radius trunk that is its base end, so the three
    # true ends are the root and two tips
    ends = np.array([n["point"] for n in T.nodes if n["kind"] in ("root", "tip")])
    assert len(ends) == 3
    truth = np.array([lines[0][0], lines[1][-1], lines[2][-1]])
    d = np.linalg.norm(truth[:, None] - ends[None], axis=2).min(1)
    # each true end has a traced end within its radius + a voxel (the ball reaches the rounded cap)
    assert (d < np.array([3.0, 2.2, 1.8]) + 0.7).all(), d
    junction = np.array([n["point"] for n in T.nodes if n["kind"] == "junction"][0])
    assert np.linalg.norm(junction - lines[0][1]) < 3.0


def test_y_radius_is_the_tube_radius(y_trace):
    T, lines, radii = y_trace
    from phantoms import capsule_distance
    ends = np.array([lines[0][0], lines[0][1], lines[1][-1], lines[2][-1]])
    errs = []
    for s in T.segments:
        p, r = np.array(s["points"]), np.array(s["radius"])
        truth = np.max([capsule_distance(p, L[0], L[1], R[0], R[1]) for L, R in zip(lines, radii)], axis=0)
        far = np.linalg.norm(p[:, None] - ends[None], axis=2).min(1) > 5.0     # off the caps and the junction
        errs.append(np.abs(r[far] - truth[far]))
    errs = np.concatenate(errs)
    assert len(errs) > 50
    assert np.median(errs) < 0.05 and errs.max() < 0.25, (np.median(errs), errs.max())


def test_y_points_stay_inside(y_trace):
    T, lines, radii = y_trace
    m, geo = tube_field(lines, radii)
    from thalweg.kernel.field import sample
    for s in T.segments:
        assert (sample(m, geo, np.array(s["points"])) > 0).all()


def test_rejects_empty_field():
    m, geo = tube_field(*y_tree())
    with pytest.raises(ValueError):
        medial.trace(np.full_like(m, -8.0), geo)


@pytest.mark.data
@pytest.mark.slow
@pytest.mark.parametrize("name", ["lung_arteries", "lung_veins", "lung_airways"])
def test_reproduces_research_reference(vessels_data, name):
    from thalweg.store import open_store
    ref = vessels_data / f"C3N-00704_ctpa0625_{name}_centerlines.json"
    if not ref.exists():
        pytest.skip(f"no reference {ref}")
    R = json.loads(ref.read_text())
    m, geo, _ = open_store(vessels_data / "runs" / "C3N-00704_ctpa0625.lung_vessels.duckn.zip").margin(name)
    T = medial.trace(m, geo, ridge_passes=1)                      # the research reference
    assert T.root.tolist() == R["root"]
    assert T.branches == R["branches"]
    assert T.nodes == R["nodes"]
    assert T.segments == R["segments"]


def test_compact_structures_come_back_as_a_root():
    """A sphere no wider than its own cover has no branch: the result is its root alone."""
    from rankfield.geometry import Geometry
    sp = 0.7
    shape = (15, 15, 15)
    geo = Geometry(shape=shape, directions=((sp, 0, 0), (0, sp, 0), (0, 0, sp)), origin=(0.0, 0.0, 0.0))
    X = np.stack(np.meshgrid(*[np.arange(s) * sp for s in shape], indexing="ij"), -1) - 7 * sp
    m = np.clip((1.5 - np.linalg.norm(X, axis=-1)) * 10.6, -8, 8).astype(np.float32)
    T = medial.trace(m, geo)
    assert T.branches == [] and T.segments == [] and [n["kind"] for n in T.nodes] == ["root"]
    single = np.full(shape, -8.0, np.float32)
    single[7, 7, 7] = 1.0
    T = medial.trace(single, geo)
    assert len(T.nodes) == 1


@pytest.mark.data
@pytest.mark.slow
def test_voxel_mode_reproduces_the_reference_with_the_labelmap(vessels_data):
    """The research voxel graph used the argmax labelmap: with it, MSB-02664 matches exactly."""
    from thalweg.store import open_store
    ref = vessels_data / "MSB-02664_ctape0625_v013_lung_arteries_centerlines_voxel.json"
    if not ref.exists():
        pytest.skip("no voxel reference")
    R = json.loads(ref.read_text())
    st = open_store(vessels_data / "runs" / "MSB-02664_ctape0625_v013.lung_vessels.duckn.zip")
    m, geo, r = st.margin("lung_arteries")
    T = medial.trace(m, geo, graph="voxel", mask=st.labelmap_mask(r), ridge_passes=1)
    assert T.segments == R["segments"] and T.nodes == R["nodes"]
