"""Export: VTP files, and the surface with named caps on analytic phantoms."""
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from rankfield.geometry import Geometry

from thalweg.centerlines import graph_from_tree
from thalweg.errors import ThalwegError
from thalweg.export import (Cut, Skipped, _refined_into, boundaries, capped_surface, end_cuts, ends,
                            mesh_defects, surface, wall_slope, write_vtp_centerlines, write_vtp_mesh,
                            zero_set)
from thalweg.graph import Points, Source, TubeGraph
from thalweg.kernel import medial
from thalweg.vmtk import Centerlines
from phantoms import SLOPE, tube_field, y_tree


def _graph(m, geo, name):
    T = medial.trace(m, geo)
    nodes, edges, pos, rad, s = graph_from_tree(T, name, m, geo, Source(), {"graph": "field"})
    return TubeGraph(structures=[s], nodes=nodes, edges=edges, points=Points(position=pos, radius=rad))


def _mirrored(m, geo):
    """The same field on a left-handed grid: axis 0 reversed, its direction negated (det < 0)."""
    D = np.asarray(geo.directions, float)
    origin = np.asarray(geo.origin, float) + (m.shape[0] - 1) * D[0]
    D2 = D.copy()
    D2[0] = -D[0]
    return m[::-1].copy(), Geometry(shape=m.shape, directions=tuple(map(tuple, D2)), origin=tuple(origin))


def _good(mesh):
    d = mesh_defects(mesh.vertices, mesh.faces)
    assert (d["boundary_edges"], d["nonmanifold_edges"], d["repeated_directed_edges"],
            d["nonmanifold_vertices"]) == (0, 0, 0, 0), d
    assert d["signed_volume_mm3"] > 0, d
    return d


def _normals(mesh, k):
    """Twice-area-weighted normals of boundary k's faces."""
    f = mesh.faces[mesh.boundary == k]
    V = mesh.vertices
    return np.cross(V[f[:, 1]] - V[f[:, 0]], V[f[:, 2]] - V[f[:, 0]])


@pytest.fixture(scope="module")
def y():
    m, geo = tube_field(*y_tree())
    return m, geo, _graph(m, geo, "y")


@pytest.fixture(scope="module")
def tree2():
    """Two generations below the trunk, one branch running off the grid (a truncated end)."""
    top = np.array([0, 0, 30.0])
    a = top + 20 * np.array([-0.6, 0, 0.8])
    b = top + 18 * np.array([0.6, 0, 0.8])
    P = [np.array([[0, 0, 0.0], top]), np.array([top, a]), np.array([top, b]),
         np.array([b, b + 15 * np.array([0.0, 0.6, 0.8])]), np.array([b, b + 30 * np.array([1.0, 0.0, 0.2])])]
    R = [np.array([3.0, 3.0]), np.array([2.2, 2.0]), np.array([2.4, 2.2]), np.array([1.6, 1.5]),
         np.array([1.8, 1.6])]
    m, geo = tube_field(P, R)
    m = m[:int((b[0] + 18 - geo.origin[0]) / 0.7)].copy()
    geo = Geometry(shape=m.shape, directions=geo.directions, origin=geo.origin)
    return m, geo, _graph(m, geo, "t")


def test_wall_slope_is_the_phantom_slope(y):
    m, geo, _ = y
    assert abs(wall_slope(m, geo) - SLOPE) / SLOPE < 0.05


@pytest.mark.parametrize("mirror", [False, True], ids=["right-handed", "left-handed"])
def test_wall_points_out_on_either_handedness(y, mirror):
    """skimage winds inward on a right-handed index->world map; the wall must point out on both."""
    m, geo, _ = y
    if mirror:
        m, geo = _mirrored(m, geo)
    assert (np.linalg.det(np.asarray(geo.directions)) < 0) == mirror
    field_volume = (m > 0).sum() * abs(np.linalg.det(np.asarray(geo.directions)))
    for refine in (1, 2):
        V, F = zero_set(m, geo, refine)
        d = mesh_defects(V, F)
        assert d["repeated_directed_edges"] == 0 and d["boundary_edges"] == 0
        assert abs(d["signed_volume_mm3"] / field_volume - 1) < 0.03, (refine, d, field_volume)


@pytest.mark.parametrize("mirror", [False, True], ids=["right-handed", "left-handed"])
def test_capped_surface_is_closed_with_flat_named_outward_caps(y, mirror, tmp_path):
    m, geo, g = y
    if mirror:
        m, geo = _mirrored(m, geo)
        g = _graph(m, geo, "y")
    cuts, skipped = end_cuts(g, "y", kinds=("tip", "root"))
    assert len(cuts) == 3 and not skipped                            # the root is the trunk's base end here
    for refine in (1, 2):
        mesh = surface(m, geo, cuts, graph=g, refine=refine)
        _good(mesh)
        assert [c.name for c in mesh.caps] == [c.name for c in cuts] and not mesh.skipped
        for k, c in enumerate(cuts, 1):
            n2 = _normals(mesh, k)
            assert len(n2) > 10, k
            v = mesh.vertices[np.unique(mesh.faces[mesh.boundary == k])]
            assert np.abs((v - c.center) @ c.normal).max() < 1e-3          # flat, on the plane
            assert (n2 @ c.normal > 0).all()                               # every cap face faces out
    assert mesh.names[0] == "wall" and all(n.startswith("y ") for n in mesh.names[1:])
    p = write_vtp_mesh(mesh, tmp_path / "y.vtp")
    piece = ET.parse(p).getroot().find("PolyData/Piece")
    assert int(piece.get("NumberOfPolys")) == len(mesh.faces)
    arr = {a.get("Name"): a for a in piece.iter("DataArray")}
    assert arr["BoundaryId"].get("type") == "Int32"


def test_cap_area_matches_the_section(y):
    """Each cap is the tube's cross-section there: area ~ pi r^2 of the phantom radius."""
    m, geo, g = y
    cuts, _ = end_cuts(g, "y", kinds=("tip",))
    mesh = surface(m, geo, cuts, refine=2)
    assert len(mesh.caps) == 2
    for k, c in enumerate(mesh.caps, 1):
        a = 0.5 * np.linalg.norm(_normals(mesh, k), axis=1).sum()
        true_r = 2.2 if c.center[0] < 0 else 1.8
        axis = (np.array([-0.6, 0.0, 0.8]) if c.center[0] < 0
                else np.array([0.6, 0.3, 0.74]) / np.linalg.norm([0.6, 0.3, 0.74]))
        expect = np.pi * true_r ** 2 / abs(axis @ c.normal)            # an oblique cut is an ellipse
        assert abs(a - expect) / expect < 0.03, (k, a, expect)


def test_refined_field_is_the_trilinear_interpolant():
    from scipy import ndimage as ndi
    a = np.random.default_rng(0).normal(size=(7, 9, 6)).astype(np.float32)
    lo, hi = np.array([1, 0, 2]), np.array([6, 7, 5])
    sub = a[1:7, 0:8, 2:6]
    for r in (1, 2, 3):
        out = _refined_into(a, lo, hi, r, 8.0)
        shape = tuple((s - 1) * r + 1 for s in sub.shape)
        assert out.shape == tuple(s + 2 for s in shape) and out.dtype == np.float32
        g = np.stack(np.meshgrid(*[np.arange(s) / r for s in shape], indexing="ij"))
        ref = ndi.map_coordinates(sub, g.reshape(3, -1), order=1).reshape(shape)
        assert np.abs(out[1:-1, 1:-1, 1:-1] - ref).max() < 1e-6
        assert (out[0] == -8).all() and (out[:, :, -1] == -8).all()


def _thin_quantized():
    """A thin Y whose margin, rounded to 1/16 logit like a ranked store's, has exact zeros that
    make marching cubes at level 0 leave holes (at refine 1 and 2)."""
    top = np.array([0, 0, 20.0])
    P = [np.array([[0, 0, 0.0], top]),
         np.array([top, [14.500837989936324, -0.6751753180845762, 16.222456369614125]]),
         np.array([top, [11.38361550780609, -8.88554479384731, 15.942859173387042]])]
    R = [np.array([1.2, 1.0]), np.array([0.8, 0.45]), np.array([0.7, 0.4])]
    m, geo = tube_field(P, R)
    return (np.round(m * 16) / 16).astype(np.float32), geo


@pytest.mark.parametrize("refine", [1, 2])
def test_quantized_field_gives_a_closed_manifold_surface(refine):
    from skimage.measure import marching_cubes
    m, geo = _thin_quantized()
    assert (m == 0).any()
    lo, hi = np.zeros(3, int), np.array(m.shape) - 1
    _, faces, _, _ = marching_cubes(_refined_into(m, lo, hi, refine, 8.0), level=0.0, method="lewiner",
                                    allow_degenerate=False)
    assert mesh_defects(np.zeros((faces.max() + 1, 3)), faces)["boundary_edges"] > 0   # the defect at level 0
    g = _graph(m, geo, "q")
    mesh = capped_surface(g, "q", m, geo, refine=refine)
    _good(mesh)
    assert mesh.caps


def test_mesh_defects_sees_each_defect():
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    F = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])      # a tetrahedron, outward
    d = mesh_defects(V, F)
    assert not any(d[k] for k in ("boundary_edges", "nonmanifold_edges", "repeated_directed_edges",
                                  "nonmanifold_vertices")) and abs(d["signed_volume_mm3"] - 1 / 6) < 1e-12
    assert mesh_defects(V, F[:3])["boundary_edges"] == 3
    assert mesh_defects(V, F[:, ::-1])["signed_volume_mm3"] < 0
    flipped = F.copy()
    flipped[0] = flipped[0, ::-1]
    assert mesh_defects(V, flipped)["repeated_directed_edges"] > 0
    two = np.vstack([V, -V[1:]])                                    # a mirrored tetrahedron sharing vertex 0
    F2 = np.vstack([F, np.where(F == 0, 0, F + 3)[:, ::-1]])        # the mirror flips it: rewind
    d2 = mesh_defects(two, F2)
    assert d2["boundary_edges"] == 0 and d2["nonmanifold_vertices"] == 1


def test_ends_count_the_boundary_root_as_truncated(y):
    m, geo, _ = y
    k = int((5.0 - geo.origin[2]) / 0.7)                        # the trunk runs off the bottom of the grid
    m2 = m[:, :, k:].copy()
    geo2 = Geometry(shape=m2.shape, directions=geo.directions,
                    origin=tuple(np.asarray(geo.origin) + [0, 0, k * 0.7]))
    g = _graph(m2, geo2, "y")
    root = g.nodes[g.structure("y").roots[0]]
    assert root.attributes.get("on_grid_boundary")
    assert (root.id, "truncated") in ends(g, "y")
    mesh = capped_surface(g, "y", m2, geo2)
    _good(mesh)
    assert f"y truncated {root.id}" in mesh.names and not mesh.skipped


def test_every_end_is_capped_or_reported(tree2):
    m, geo, g = tree2
    kinds = ("tip", "truncated", "root")
    got = ends(g, "t", kinds)
    assert {k for _, k in got} == set(kinds)
    mesh = capped_surface(g, "t", m, geo, kinds=kinds)
    _good(mesh)
    assert len(mesh.caps) + len(mesh.skipped) == len(got)
    assert sorted([c.node for c in mesh.caps] + [s.node for s in mesh.skipped]) == sorted(n for n, _ in got)
    assert len(ends(g, "t")) == len(got) - 1                         # the default leaves the root out
    with pytest.raises(ThalwegError):
        ends(g, "t", ("tip", "junction"))


def test_two_close_tips_are_reported_not_dropped():
    P = [np.array([[0, 0, 0.0], [0, 0, 30]]), np.array([[0, 0, 30.0], [-1, 0, 34]]),
         np.array([[0, 0, 30.0], [1, 0, 34]])]
    m, geo = tube_field(P, [np.array([2.0, 2.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0])], spacing=0.5)
    g = _graph(m, geo, "t")
    cuts, skipped = end_cuts(g, "t")
    tips = [n for n, k in ends(g, "t") if k == "tip"]
    assert len(tips) == 2 and not cuts and sorted(s.node for s in skipped) == sorted(tips)
    assert all("too short" in s.reason for s in skipped)
    mesh = capped_surface(g, "t", m, geo)
    _good(mesh)
    assert not mesh.caps and len(mesh.skipped) == 2


def test_min_radius_reads_the_radius_at_the_cut(y):
    """The tips' last samples sit in the rounded ends (radius << the tube's); the rule reads the cut."""
    _, _, g = y
    cuts, skipped = end_cuts(g, "y", kinds=("tip",), min_radius=2.0)
    assert [round(c.radius) for c in cuts] == [2]                   # the 2.2 mm daughter stays
    assert len(skipped) == 1 and "below 2.0 mm" in skipped[0].reason   # the 1.8 mm one does not


def test_a_cut_outside_the_structure_is_reported(y):
    m, geo, g = y
    cuts, skipped = end_cuts(g, "y", margin=np.full_like(m, -1.0), geometry=geo)
    assert not cuts and len(skipped) == 2 and all("outside the structure" in s.reason for s in skipped)


def test_a_cut_whose_far_side_stays_attached_is_undone():
    """A ring with a tail: a plane across the ring leaves the far side attached around the loop, so
    capping it would put a double wall inside; the cut is undone and reported, the tail's is made."""
    th = np.linspace(0, 2 * np.pi, 25)
    ring = np.stack([12 * np.cos(th), 12 * np.sin(th), np.zeros_like(th)], 1)
    tail = np.array([[12.0, 0, 0], [30.0, 0, 0]])
    m, geo = tube_field([ring, tail], [np.full(25, 2.0), np.array([2.0, 2.0])])
    cuts = [Cut(np.array([-12.0, 0, 0]), np.array([0, 1.0, 0]), 2.0, "ring"),
            Cut(np.array([24.0, 0, 0]), np.array([1.0, 0, 0]), 2.0, "tail")]
    for keep_main in (False, True):
        mesh = surface(m, geo, cuts, keep_main=keep_main)
        assert [c.name for c in mesh.caps] == ["tail"]
        assert [s.name for s in mesh.skipped] == ["ring"] and "double wall" in mesh.skipped[0].reason
    _good(mesh)


def test_boundaries_sidecar_names_only_existing_caps(tree2, tmp_path):
    m, geo, g = tree2
    mesh = capped_surface(g, "t", m, geo, kinds=("tip", "truncated", "root"))
    mesh.skipped.append(Skipped("t tip 99", "a test skip", 99, "tip"))
    write_vtp_mesh(mesh, tmp_path / "t.vtp")
    doc = json.loads((tmp_path / "t.vtp.boundaries.json").read_text())
    assert doc == json.loads(json.dumps(boundaries(mesh)))
    assert doc["space"] == "LPS" and doc["units"] == "mm"
    rows = doc["boundaries"]
    assert [r["id"] for r in rows] == list(range(len(mesh.caps) + 1)) and rows[0]["name"] == "wall"
    assert sorted(np.unique(mesh.boundary).tolist()) == [r["id"] for r in rows]
    for r, c in zip(rows[1:], mesh.caps):
        assert r["name"] == c.name and r["node"] == c.node and r["cap_kind"] == c.kind
        n = np.asarray(r["normal"])
        assert abs(np.linalg.norm(n) - 1) < 1e-5 and r["area_mm2"] > 0 and r["inscribed_radius_mm"] > 0
        assert abs((np.asarray(r["centroid"]) - r["center"]) @ n) < 1e-4       # on the plane
    assert doc["skipped"][-1] == {"name": "t tip 99", "cap_kind": "tip", "node": 99, "reason": "a test skip"}
    assert not {s["name"] for s in doc["skipped"]} & {r["name"] for r in rows}


def test_centerlines_vtp(tmp_path, phantom_oracle):
    cl = Centerlines.from_npz(phantom_oracle["extract"])
    p = write_vtp_centerlines(cl, tmp_path / "c.vtp")
    piece = ET.parse(p).getroot().find("PolyData/Piece")
    assert int(piece.get("NumberOfLines")) == cl.n_cells and int(piece.get("NumberOfPoints")) == cl.n_points
    arrays = {a.get("Name"): a.get("type") for a in piece.iter("DataArray")}
    assert {"GroupIds", "Blanking", "MaximumInscribedSphereRadius"} <= set(arrays)
    for name in ("GroupIds", "Blanking", "CenterlineIds", "TractIds"):   # vmtk reads these as vtkIntArray
        assert arrays[name] == "Int32", (name, arrays[name])
    assert arrays["MaximumInscribedSphereRadius"] == "Float64"


def test_int_arrays_out_of_int32_range_are_refused(tmp_path):
    cl = Centerlines(np.zeros((2, 3)), [np.array([0, 1])], cell_data={"GroupIds": np.array([2 ** 40])})
    with pytest.raises(ThalwegError):
        write_vtp_centerlines(cl, tmp_path / "c.vtp")


def test_swc_is_a_tree_of_the_graph(y, tmp_path):
    from thalweg.export import write_swc
    _, _, g = y
    p = write_swc(g, "y", tmp_path / "y.swc")
    rows = [line.split() for line in p.read_text().splitlines() if not line.startswith("#")]
    ids = [int(r[0]) for r in rows]
    parents = [int(r[6]) for r in rows]
    assert ids == list(range(1, len(rows) + 1)) and parents[0] == -1
    assert all(0 < q < i for i, q in zip(ids[1:], parents[1:]))            # parents come first
    # every graph sample appears once, junctions shared: points = table rows - (edges - 0) duplicated starts
    assert len(rows) == len(g.points.position) - len(g.edges) + 1


def test_swc_of_a_multigeneration_tree_with_a_truncated_end(tree2, tmp_path):
    from thalweg.export import write_swc
    _, _, g = tree2
    assert any(nd.kind == "truncated" for nd in g.nodes) and max(e.tracer_generation for e in g.edges) >= 1
    text = write_swc(g, "t", tmp_path / "t.swc").read_text()
    rows = np.array([[float(v) for v in ln.split()] for ln in text.splitlines() if not ln.startswith("#")])
    ids, parents = rows[:, 0].astype(int), rows[:, 6].astype(int)
    assert ids.tolist() == list(range(1, len(rows) + 1)) and parents[0] == -1
    assert (parents[1:] < ids[1:]).all()
    assert len(rows) == len(g.points.position) - len(g.edges) + 1
    children = np.bincount(parents[1:], minlength=len(rows) + 1)[1:]
    deg = g.degree()
    for nd in g.nodes:                                                   # each node once, with its degree
        k = int(np.argmin(np.linalg.norm(rows[:, 2:5] - nd.position, axis=1)))
        assert np.linalg.norm(rows[k, 2:5] - nd.position) < 1e-3
        assert children[k] + (parents[k] != -1) == deg[nd.id], (nd.id, nd.kind)
    assert (rows[:, 5] >= 0).all()


def test_markups_follow_the_slicer_schema(y, tmp_path):
    """The structure markups-schema-v1.0.3 requires (the C3N arteries' file was validated against
    the schema itself with jsonschema when this was written)."""
    from thalweg.export import MARKUPS_SCHEMA, write_slicer_markups
    _, _, g = y
    doc = json.loads(write_slicer_markups(g, "y", tmp_path / "y.mrk.json", every=3).read_text())
    assert set(doc) == {"@schema", "markups"} and doc["@schema"] == MARKUPS_SCHEMA
    assert len(doc["markups"]) == len(g.edges)
    for mk, e in zip(doc["markups"], g.edges):
        assert mk["type"] == "Curve" and mk["coordinateSystem"] == "LPS" and mk["coordinateUnits"] == "mm"
        assert isinstance(mk["name"], str)
        ids = [c["id"] for c in mk["controlPoints"]]
        assert all(isinstance(i, str) for i in ids) and len(set(ids)) == len(ids)
        P = np.array([c["position"] for c in mk["controlPoints"]])
        assert P.shape[1] == 3 and np.isfinite(P).all()
        ep = g.edge_points(e)
        assert np.allclose(P[0], ep[0]) and np.allclose(P[-1], ep[-1])        # both ends kept
