"""Labelmaps and signed distance images as input (degraded mode, the `volumes` extra)."""
import os
from pathlib import Path

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

from click.testing import CliRunner  # noqa: E402

from thalweg.cli import main  # noqa: E402
from thalweg.errors import ThalwegError  # noqa: E402
from thalweg.kernel.field import to_world  # noqa: E402
from thalweg.store import open_store  # noqa: E402
from thalweg.volume import SLOPE, VolumeStore, signed_distance  # noqa: E402
from phantoms import tube_field, y_tree  # noqa: E402


def _image(array_xyz, spacing, origin, direction=None, pixel=None):
    """A SimpleITK image whose voxel (i, j, k) is array_xyz[i, j, k]."""
    img = sitk.GetImageFromArray(np.transpose(array_xyz, (2, 1, 0)))
    img.SetSpacing(tuple(map(float, spacing)))
    img.SetOrigin(tuple(map(float, origin)))
    if direction is not None:
        img.SetDirection(tuple(np.asarray(direction, float).ravel()))
    return img


def _y_labelmap(tmp_path, value=3, suffix=".nii.gz"):
    m, geo = tube_field(*y_tree())
    sp = np.linalg.norm(np.asarray(geo.directions), axis=1)
    img = _image((m > 0).astype(np.uint8) * value, sp, geo.origin)
    path = tmp_path / f"y{suffix}"
    sitk.WriteImage(img, str(path))
    return path, m, geo


def test_geometry_is_simpleitks_on_an_oblique_grid(tmp_path):
    """Direction cosines, anisotropic spacing and an origin: every voxel's world point is the one
    SimpleITK computes, exactly."""
    rot = np.array([[0.0, -1, 0], [0.6, 0, -0.8], [0.8, 0, 0.6]])
    arr = np.zeros((6, 7, 8), np.int16)
    arr[2:4, 3:5, 1:7] = 2
    img = _image(arr, (0.5, 0.8, 1.2), (-10.0, 4.0, 7.5), rot)
    path = tmp_path / "o.nrrd"
    sitk.WriteImage(img, str(path))
    st = open_store(path)
    geo = st.geometry(0)
    idx = np.stack(np.meshgrid(*[np.arange(n) for n in geo.shape], indexing="ij"), -1).reshape(-1, 3)
    world = to_world(geo, idx)
    for a, w in zip(idx[::37], world[::37]):
        assert np.allclose(w, img.TransformIndexToPhysicalPoint(tuple(int(v) for v in a[::-1])), atol=1e-9)
    assert st.labelmap_mask(st.ref("label_2")).sum() == 2 * 2 * 6


def test_a_labelmap_traces_like_its_structure(tmp_path):
    """The Y phantom written as a labelmap: the graph has its three branches, from a degraded field
    whose wall slope is SLOPE, and the radii read a staircase's (0.15-0.3 mm small on 0.7 mm voxels)."""
    from thalweg.centerlines import centerline_graph
    path, m, geo = _y_labelmap(tmp_path)
    st = open_store(path)
    assert isinstance(st, VolumeStore) and st.names == ["label_3"] and st.kind == "labelmap"
    margin, geo2, ref = st.margin("label_3")
    assert ((margin > 0) == (m > 0).transpose(2, 1, 0)).all()            # the store's axes are (z, y, x)
    assert margin.max() == 8.0 and ref.scheme == "degraded:labelmap"
    g = centerline_graph(st, "label_3")
    assert len(g.edges) == 3 and g.structures[0].source.labeling_scheme == "degraded:labelmap"
    radii = sorted(float(np.median(g.edge_radius(e))) for e in g.edges)
    assert all(0.1 < t - r < 0.35 for r, t in zip(radii, (1.8, 2.2, 3.0)))


def test_signed_distance_places_the_wall_half_a_voxel_out():
    from rankfield.geometry import Geometry
    geo = Geometry(shape=(9, 9, 9), directions=((1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0)), origin=(0.0, 0, 0))
    mask = np.zeros((9, 9, 9), bool)
    mask[:, :, :4] = True                                               # a slab: z < 3.5
    d = signed_distance(mask, geo)
    assert np.allclose(d[4, 4, :], [3.5, 2.5, 1.5, 0.5, -0.5, -1.5, -2.5, -3.5, -4.5])
    assert (signed_distance(np.zeros_like(mask), geo) == -np.inf).all()


def test_slicer_segment_names_and_a_distance_image(tmp_path):
    arr = np.zeros((10, 10, 10), np.uint8)
    arr[2:5, 2:5, :] = 1
    arr[6:9, 6:9, :] = 2
    img = _image(arr, (1, 1, 1), (0, 0, 0))
    for k, (name, value) in enumerate([("artery", 1), ("vein", 2)]):
        img.SetMetaData(f"Segment{k}_Name", name)
        img.SetMetaData(f"Segment{k}_LabelValue", str(value))
        img.SetMetaData(f"Segment{k}_Layer", "0")
    path = tmp_path / "v.seg.nrrd"
    sitk.WriteImage(img, str(path))
    st = open_store(path)
    assert st.names == ["artery", "vein"] and st.ref("vein").label_value == 2
    assert VolumeStore(path, names={2: "pulmonary_vein"}).names == ["artery", "pulmonary_vein"]
    # a distance image, negative inside (ITK) and positive inside
    x = np.arange(20.0)
    sdf = np.abs(x[:, None, None] - 10.0) * np.ones((20, 20, 20)) - 3.0            # a slab of half-width 3
    sitk.WriteImage(_image(sdf.astype(np.float32), (1, 1, 1), (0, 0, 0)), str(tmp_path / "slab.nii.gz"))
    st = open_store(tmp_path / "slab.nii.gz")
    assert st.kind == "sdf" and st.names == ["slab"]
    m, _, ref = st.margin("slab")
    assert m[5, 5, 10] == 8.0 and m[5, 5, 0] == -8.0 and m[5, 5, 13] == 0.0         # the wall at x = 13
    assert st.margin("slab")[0] is m and SLOPE == 10.0                           # decoded once
    flipped = VolumeStore(tmp_path / "slab.nii.gz", sdf_inside="positive").margin("slab")[0]
    assert flipped[5, 5, 10] == -8.0 and ref.scheme == "degraded:sdf"


def test_bad_inputs_are_clear_errors(tmp_path):
    with pytest.raises(ThalwegError, match="no image"):
        open_store(tmp_path / "missing.nii.gz")
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((4, 4), np.uint8)), str(tmp_path / "flat.nrrd"))
    with pytest.raises(ThalwegError, match="3-D"):
        open_store(tmp_path / "flat.nrrd")
    path, _, _ = _y_labelmap(tmp_path)
    with pytest.raises(ThalwegError, match="has no structure"):
        open_store(path).margin("lung_arteries")
    with pytest.raises(ThalwegError):
        VolumeStore(path, sdf_inside="up")


def test_the_cli_runs_on_a_labelmap(tmp_path):
    path, _, _ = _y_labelmap(tmp_path, suffix=".nrrd")
    graph = tmp_path / "y.thalweg.json.gz"
    r = CliRunner().invoke(main, ["centerlines", str(path), "-s", "label_3", "-o", str(graph), "-q"])
    assert r.exit_code == 0, r.output + str(r.exception)
    r = CliRunner().invoke(main, ["run", str(path), "-s", "label_3", "-o", str(tmp_path / "out"), "-q",
                                  "--step", "2"])
    assert r.exit_code == 0, r.output + str(r.exception)
    r = CliRunner().invoke(main, ["export", str(graph), str(path), "-s", "label_3", "--mesh",
                                  str(tmp_path / "y.vtp"), "--cap-kinds", "tip,root"])
    assert r.exit_code == 0, r.output + str(r.exception)


def _seg_case(tmp_path, fractional=False):
    pytest.importorskip("highdicom")
    from dicom_seg import ct_series, world_points, write_seg
    from phantoms import capsule_distance
    rd = np.array([0.8, 0.6, 0.0])
    cd = np.array([-0.36, 0.48, 0.8])
    cd -= (cd @ rd) * rd
    cd /= np.linalg.norm(cd)
    shape, sp, org = (40, 50, 45), (0.8, 0.6, 0.7), np.array([-12.0, -14.0, -3.0])
    W = world_points(shape, sp, org, rd, cd).reshape(-1, 3)
    trunk = capsule_distance(W, np.zeros(3), np.array([0.0, 0, 20]), 3.0, 3.0).reshape(shape)
    side = capsule_distance(W, np.array([-2.0, 0, 10]), np.array([8.0, 5, 15]), 1.5, 1.5).reshape(shape)
    if fractional:
        masks = [np.clip(0.5 + trunk / 0.7, 0, 1), np.clip(0.5 + side / 0.7, 0, 1)]
    else:
        masks = [trunk > 0, side > 0]
    path = write_seg(tmp_path / "seg.dcm", ct_series(shape, sp, org, rd, cd), masks,
                     ["trunk", "side branch"], fractional=fractional)
    return path, trunk > 0, side > 0


@pytest.mark.parametrize("fractional", [False, True])
def test_a_dicom_seg_on_an_oblique_series(tmp_path, fractional):
    """A two-segment SEG (overlapping segments) of oblique, anisotropic CT slices: each segment is a
    structure named by its label; its voxels lie where the truth says, in world space; the two
    overlap as drawn; a fractional segment counts from 0.5."""
    from phantoms import capsule_distance
    path, trunk, side = _seg_case(tmp_path, fractional)
    st = open_store(path)
    assert isinstance(st, VolumeStore) and st.kind == "dicom-seg" and st.names == ["side branch", "trunk"]
    assert st.ref("trunk").label_value == 1 and st.labeling_scheme == "degraded:dicom-seg"
    m, geo, _ = st.margin("trunk")
    inside = to_world(geo, np.argwhere(m > 0))
    assert len(inside) == trunk.sum()                                   # every segmented voxel, no other
    assert (capsule_distance(inside, np.zeros(3), np.array([0.0, 0, 20]), 3.0, 3.0) > -1e-6).all()
    both = st.labelmap_mask(st.ref("trunk")) & st.labelmap_mask(st.ref("side branch"))
    assert both.sum() == (trunk & side).sum() > 0
    edge = m[:3].max(), m[-3:].max(), m[:, :3].max(), m[:, -3:].max()
    assert max(edge) < 0                                                 # the padding is outside


def test_the_cli_runs_on_a_dicom_seg_and_refuses_other_dicom(tmp_path):
    path, _, _ = _seg_case(tmp_path)
    graph = tmp_path / "t.thalweg.json.gz"
    r = CliRunner().invoke(main, ["centerlines", str(path), "-s", "trunk", "-o", str(graph), "-q"])
    assert r.exit_code == 0, r.output + str(r.exception)
    from thalweg.graph import TubeGraph
    g = TubeGraph.read(graph)
    P = g.positions()
    long = max(g.edges, key=lambda e: e.length_mm)
    assert long.length_mm > 10 and np.median(np.hypot(P[:, 0], P[:, 1])) < 0.4    # on the trunk's axis
    from dicom_seg import ct_series
    ct = ct_series((2, 8, 8), (1, 1, 1), (0, 0, 0), (1, 0, 0), (0, 1, 0))[0]
    ct.save_as(str(tmp_path / "ct.dcm"))
    with pytest.raises(ThalwegError, match="not a Segmentation"):
        open_store(tmp_path / "ct.dcm")


IDC_SEG = Path(os.environ.get("IDC_SEG_DATA", Path.home() / "tmp/data/idc_seg"))


@pytest.mark.data
@pytest.mark.slow
def test_a_real_idc_seg():
    """NLST patient 217076's TotalSegmentator SEG from IDC (80 binary segments, 5,695 frames): the
    segments are named by their labels, left organs lie at +x (LPS) and right ones at -x, and the
    aorta traces as one long tube."""
    pytest.importorskip("highdicom")
    files = sorted(IDC_SEG.rglob("*.dcm")) if IDC_SEG.exists() else []
    if not files:
        pytest.skip(f"no IDC SEG under {IDC_SEG}")
    from thalweg.centerlines import centerline_graph
    st = open_store(files[0])
    assert st.kind == "dicom-seg" and len(st.names) == 80 and "Pulmonary artery" in st.names
    geo = st.geometry(0)

    def centroid(name):
        return to_world(geo, np.argwhere(st.margin(name)[0] > 0)).mean(0)
    assert centroid("Spleen")[0] > 30 > -30 > centroid("Liver")[0]
    g = centerline_graph(st, "Aorta")
    assert sum(e.length_mm for e in g.edges) > 300 and sum(n.kind == "tip" for n in g.nodes) == 1
