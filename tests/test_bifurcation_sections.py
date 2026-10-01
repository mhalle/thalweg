"""vmtkbifurcationsections: the placement against vmtk's own output on the phantom, vmtk's polygon
measures, and the sections cut from the field against the ones vmtk cut from the surface."""
from types import SimpleNamespace

import numpy as np
import pytest
from rankfield.geometry import Geometry
from scipy.spatial import cKDTree

from thalweg.branching import bifurcation_sections
from thalweg.vmtk import Centerlines
from thalweg.vmtk.sections import bifurcation_section_planes, section_area, section_shape

R = "MaximumInscribedSphereRadius"


@pytest.fixture(scope="module")
def phantom(phantom_oracle):
    E = Centerlines.from_npz(phantom_oracle["extract"])
    sections = {n: phantom_oracle[f"bifurcation_sections_{n}"] for n in (1, 2)}
    return E, sections


@pytest.mark.parametrize("n", [1, 2])
def test_planes_are_vmtks(phantom, n):
    """With vmtk's defects on, the section points and normals are vmtk's to rounding; the default
    (interpolated radius, true step count) moves the points by at most 0.2 micrometers here."""
    E, sections = phantom
    z = sections[n]
    exact = bifurcation_section_planes(E, n, vmtk_interp=True, vmtk_steps=True)
    assert [p["group"] for p in exact] == z["cd__BifurcationSectionGroupIds"].tolist()
    assert [p["bifurcation_group"] for p in exact] == z["cd__BifurcationSectionBifurcationGroupIds"].tolist()
    assert [p["orientation"] for p in exact] == z["cd__BifurcationSectionOrientation"].tolist()
    assert np.abs(np.array([p["point"] for p in exact]) - z["cd__BifurcationSectionPoint"]).max() < 1e-10
    assert np.abs(np.array([p["normal"] for p in exact]) - z["cd__BifurcationSectionNormal"]).max() < 1e-12
    ours = bifurcation_section_planes(E, n)
    assert np.abs(np.array([p["point"] for p in ours]) - z["cd__BifurcationSectionPoint"]).max() < 2e-4
    assert all(p["radius"] > 0.5 for p in ours)


@pytest.mark.parametrize("n", [1, 2])
def test_polygon_measures_are_vmtks(phantom, n):
    """vmtk's area, MinSize, MaxSize and Shape of its own section polygons, recomputed."""
    _, sections = phantom
    z = sections[n]
    polys = np.split(z["points"], np.cumsum(z["cell_len"])[:-1])
    for k, poly in enumerate(polys):
        c, nn = z["cd__BifurcationSectionPoint"][k], z["cd__BifurcationSectionNormal"][k]
        assert section_area(poly, nn) == pytest.approx(z["cd__BifurcationSectionArea"][k], abs=1e-6)
        got = section_shape(poly, c)
        want = (z["cd__BifurcationSectionMinSize"][k], z["cd__BifurcationSectionMaxSize"][k],
                z["cd__BifurcationSectionShape"][k])
        assert np.allclose(got, want, atol=1e-10)


def test_shape_of_a_circle_and_an_ellipse():
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    circle = np.c_[2 * np.cos(t), 2 * np.sin(t), 0 * t]
    lo, hi, shape = section_shape(circle, np.zeros(3))
    assert lo == pytest.approx(4.0, abs=1e-3) and hi == pytest.approx(4.0, abs=1e-3)
    assert shape == pytest.approx(1.0, abs=1e-3)
    assert section_area(circle, [0, 0, 1]) == pytest.approx(4 * np.pi, rel=1e-3)
    ellipse = np.c_[3 * np.cos(t), 1 * np.sin(t), 0 * t]
    lo, hi, shape = section_shape(ellipse, np.zeros(3))
    assert lo == pytest.approx(2.0, abs=1e-3) and hi == pytest.approx(6.0, abs=1e-3)
    assert shape == pytest.approx(1 / 3, abs=1e-3)
    assert section_shape(circle[:2], np.zeros(3)) == (0.0, 0.0, 0.0)


def _sphere_union(lines, h=0.35, pad=3.0):
    """The oracle's field (tests/oracle/vmtk_centerline_oracle.py ``sphere_union``), rebuilt."""
    P = lines.points
    r = np.asarray(lines.point_data[R]).reshape(-1)
    lo = P.min(0) - r.max() - pad
    hi = P.max(0) + r.max() + pad
    shape = tuple(int(np.ceil((b - a) / h)) + 1 for a, b in zip(lo, hi))
    X = lo + h * np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1).reshape(-1, 3)
    d, i = cKDTree(P).query(X, k=32, workers=-1)
    f = (r[i] - d).max(1).reshape(shape).astype(np.float32)
    return f, Geometry(shape=shape, directions=((h, 0, 0), (0, h, 0), (0, 0, h)), origin=tuple(lo))


@pytest.mark.slow
def test_field_sections_match_vmtks_surface_sections(phantom, phantom_oracle):
    """The same field vmtk meshed and clipped: section areas within 0.3 %, calipers within 0.03 mm,
    except where the plane also crosses a neighboring branch - vmtk cuts only its own group's
    surface (an open polygon), the field's contour would take in the neighbor, and both say the
    section is not closed (and ours reports no size for it)."""
    E, sections = phantom
    f, geo = _sphere_union(Centerlines.from_npz(phantom_oracle["input"]))
    for n in (1, 2):
        z = sections[n]
        rows = bifurcation_sections(SimpleNamespace(split=E), f, geo, n, vmtk_compatible=True)
        assert len(rows) == len(z["cd__BifurcationSectionArea"])
        for k, row in enumerate(rows):
            assert row["distance_spheres"] == n and row["orientation"] in ("upstream", "downstream")
            if not row["contour_closed"]:
                assert z["cd__BifurcationSectionClosed"][k] == 0 and row["area_mm2"] is None
                continue
            assert row["area_lower_mm2"] < row["area_mm2"] < row["area_upper_mm2"]
            assert row["area_mm2"] == pytest.approx(z["cd__BifurcationSectionArea"][k], rel=3e-3)
            assert row["min_size_mm"] == pytest.approx(z["cd__BifurcationSectionMinSize"][k], abs=0.03)
            assert row["max_size_mm"] == pytest.approx(z["cd__BifurcationSectionMaxSize"][k], abs=0.03)
        assert sum(r["contour_closed"] for r in rows) >= 9


def test_a_walk_that_leaves_the_group_gives_no_section(phantom):
    """Fifty spheres from a bifurcation is farther than most groups reach: those sections are
    skipped, not placed at a bogus point."""
    E, _ = phantom
    one = bifurcation_section_planes(E, 1)
    far = bifurcation_section_planes(E, 50)
    assert len(far) < len(one) and all(np.isfinite(p["point"]).all() for p in far)


def test_sections_of_the_y_phantom_from_its_graph():
    """vmtk's branching of the Y phantom's graph, sections cut from its field: the trunk's is about
    pi 3^2, each daughter's pi r^2 for its radius, all closed and nearly round."""
    from thalweg.branching import vmtk_branching
    from thalweg.centerlines import graph_from_tree
    from thalweg.graph import Points, Source, TubeGraph
    from thalweg.kernel import medial
    from phantoms import tube_field, y_tree
    m, geo = tube_field(*y_tree())
    T = medial.trace(m, geo)
    nodes, edges, pos, rad, st = graph_from_tree(T, "y", m, geo, Source(), {"connectivity": "field"})
    g = TubeGraph(structures=[st], nodes=nodes, edges=edges, points=Points(position=pos, radius=rad))
    rows = bifurcation_sections(vmtk_branching(g, "y", step=0.3), m, geo)
    assert len(rows) == 3 and [r["orientation"] for r in rows].count("upstream") == 1
    areas = sorted(r["area_mm2"] for r in rows)
    want = sorted(np.pi * np.array([3.0, 2.2, 1.8]) ** 2)
    assert all(r["contour_closed"] and r["shape"] > 0.9 for r in rows)
    assert np.allclose(areas, want, rtol=0.08)
    for r in rows:
        normal = np.array([r["normal_x"], r["normal_y"], r["normal_z"]])
        assert abs(np.linalg.norm(normal) - 1) < 1e-9
        assert r["min_size_mm"] <= r["max_size_mm"]
        assert r["area_lower_mm2"] < r["area_mm2"] < r["area_upper_mm2"]
