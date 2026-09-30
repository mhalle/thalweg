"""Stations and sections: analytic tubes (always) and the research reference (data)."""
import numpy as np
import pytest

from thalweg.kernel import sections as S
from phantoms import SLOPE, tube_field


def straight_tube(radius=2.0, length=30.0):
    line = np.array([[0.0, 0.0, 0.0], [length, 0.0, 0.0]])
    return tube_field([line], [np.array([radius, radius])], spacing=0.5)


def test_transport_frames_are_orthonormal_and_untwisted():
    t = np.linspace(0, 4 * np.pi, 400)
    C = np.stack([10 * np.cos(t), 10 * np.sin(t), 3 * t], 1)               # a helix
    T, n1, n2 = S.transport_frames(C)
    assert np.allclose((T * n1).sum(1), 0, atol=1e-12) and np.allclose(np.linalg.norm(n2, axis=1), 1)
    # parallel transport: n1 changes only along T (its derivative has no n2 component)
    dn1 = np.diff(n1, axis=0)
    assert np.abs((dn1 * n2[1:]).sum(1)).max() < 1e-3


def test_round_section_describes_a_circle():
    m, geo = straight_tube(2.0)
    img, g = S.section_image(m, geo, np.array([15.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]),
                             S.half_width(2.0))
    d = S.describe(img, g, 0.0)
    # the trilinear interpolant of a round field on a 0.5 mm grid reads ~1 % small (chords); not the contour
    assert d["closed"] and abs(d["area"] - np.pi * 4) / (np.pi * 4) < 0.02
    assert abs(d["equivalent_diameter"] - 4.0) < 0.04 and abs(d["min_feret"] - d["max_feret"]) < 0.05
    assert d["aspect_ratio"] > 0.99 and d["centroid_offset"] < 0.02
    # the model's interval: +-2 logits is +-2/SLOPE mm of radius
    A = S.pixel_areas(img)
    r = np.sqrt(A / np.pi)
    assert abs((r[0] - r[4]) - 4.0 / SLOPE) < 0.03


def test_flat_section_aspect_ratio():
    """An elliptic lumen (a flattened tube): Feret widths are its axes, the radius alone is not."""
    from rankfield.geometry import Geometry
    sp = 0.25
    shape = (40, 60, 40)
    geo = Geometry(shape=shape, directions=((sp, 0, 0), (0, sp, 0), (0, 0, sp)), origin=(-5.0, -7.5, -5.0))
    idx = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1)
    X = np.array(geo.origin) + idx * sp
    a, b = 3.0, 1.5                                               # semi-axes along y and z; tube along x
    d = (1 - np.sqrt((X[..., 1] / a) ** 2 + (X[..., 2] / b) ** 2)) * b
    m = np.clip(d * SLOPE, -8, 8).astype(np.float32)
    img, g = S.section_image(m, geo, np.zeros(3), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), 5.0)
    dd = S.describe(img, g)
    assert abs(dd["max_feret"] - 2 * a) < 0.1 and abs(dd["min_feret"] - 2 * b) < 0.1
    assert abs(dd["area"] - np.pi * a * b) / (np.pi * a * b) < 0.02
    assert abs(dd["aspect_ratio"] - b / a) < 0.02


def test_center_outside_gives_empty():
    m, geo = straight_tube(2.0)
    img, g = S.section_image(m, geo, np.array([15.0, 5.0, 0]), np.array([0, 1.0, 0]),
                             np.array([0, 0, 1.0]), 3.0)
    assert S.describe(img, g)["area"] == 0.0 and S.pixel_areas(img)[2] == 0.0


@pytest.mark.data
@pytest.mark.slow
def test_reproduces_research_pixel_areas(vessels_data):
    """straighten.py's areas at its own stations (saved centerline and radius), exactly."""
    ref = vessels_data / "C3N-00704_straight.npz"
    if not ref.exists():
        pytest.skip("no straighten.py output")
    z = np.load(ref)
    from thalweg.store import open_store
    store = vessels_data / "runs" / "C3N-00704_ctpa0625.lung_vessels.duckn.zip"
    m, geo, _ = open_store(store).margin("lung_arteries")
    C, r = z["centerline"], z["r_path"]
    T, n1, n2 = S.transport_frames(C)
    idx = np.arange(0, len(C), 7)                                  # every 7th station keeps this quick
    A = np.array([S.pixel_areas(S.section_image(m, geo, C[i], n1[i], n2[i], S.half_width(r[i]))[0])
                  for i in idx])
    # pixel counts equal; 0.1 * 0.1 != 0.01 in float
    assert np.abs(A - z["A_thin"][idx]).max() < 1e-9


@pytest.mark.data
@pytest.mark.slow
def test_stations_reproduce_straighten(vessels_data):
    """straighten.py's path (root -> the farthest left-lung tip of the research graph) through
    S.stations gives its saved station centers and radii exactly."""
    import json
    ref = vessels_data / "C3N-00704_straight.npz"
    G = vessels_data / "C3N-00704_ctpa0625_lung_arteries_centerlines.json"
    if not (ref.exists() and G.exists()):
        pytest.skip("no straighten.py output")
    z = np.load(ref)
    G = json.loads(G.read_text())
    N, Sg = G["nodes"], G["segments"]
    into = {s["b"]: s["id"] for s in Sg}

    def chain_to(node):
        out = []
        while node in into:
            s = Sg[into[node]]
            out.append(s["id"])
            node = s["a"]
        return out[::-1]

    best, best_len = None, 0
    for n in N:
        if n["kind"] == "tip" and n["point"][0] >= G["root"][0] + 30:
            ch = chain_to(n["id"])
            L = sum(Sg[j]["length_mm"] for j in ch)
            if L > best_len:
                best, best_len = ch, L
    pts = np.concatenate([np.array(Sg[j]["points"])[(1 if i else 0):] for i, j in enumerate(best)])
    rad = np.concatenate([np.array(Sg[j]["radius"])[(1 if i else 0):] for i, j in enumerate(best)])
    st = S.stations(pts, rad)
    if len(st.centers) != len(z["centerline"]):
        pytest.skip("straighten.py's output predates the current research graph")
    assert np.array_equal(st.centers, z["centerline"]) and np.array_equal(st.radius, z["r_path"])
