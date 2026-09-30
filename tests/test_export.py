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


def test_boundary_reference_systems_of_the_y(y):
    """Each cap's ring: barycenter on the tube's axis, mean radius the tube's, normal the cut's; the
    ring runs counterclockwise about that normal; vmtk's vertex average is the other definition."""
    from thalweg.export import boundary_reference_system, cap_ring
    m, geo, g = y
    mesh = capped_surface(g, "y", m, geo, kinds=("tip", "root"))
    true_radius = {"root": 3.0}
    for k, c in enumerate(mesh.caps, 1):
        ref = boundary_reference_system(mesh, k)
        assert np.linalg.norm(ref["barycenter"] - c.center) < 0.1 and np.allclose(ref["normal"], c.normal)
        assert abs((ref["barycenter"] - c.center) @ c.normal) < 1e-6           # in the cap's plane
        assert abs(ref["mean_radius_mm"] - true_radius.get(c.kind, c.radius)) < 0.08
        assert abs(ref["perimeter_mm"] - 2 * np.pi * ref["mean_radius_mm"]) < 0.15
        P = mesh.vertices[cap_ring(mesh, k)] - ref["barycenter"]
        turn = np.cross(P, np.roll(P, -1, axis=0)) @ c.normal
        assert (turn > 0).all()                                                  # counterclockwise
        v = boundary_reference_system(mesh, k, vmtk_vertex_mean=True)
        ring = mesh.vertices[cap_ring(mesh, k)]
        assert np.allclose(v["barycenter"], ring.mean(0))
        assert v["mean_radius_mm"] == pytest.approx(np.linalg.norm(ring - ring.mean(0), axis=1).mean())
    with pytest.raises(ThalwegError):
        cap_ring(mesh, 0)                                                        # the wall is no disk


def test_flow_extensions_are_tubes_that_end_in_a_named_flat_circle(y, tmp_path):
    from thalweg.export import boundary_reference_system, flow_extensions
    m, geo, g = y
    mesh = capped_surface(g, "y", m, geo, kinds=("tip", "root"))
    refs = {k: boundary_reference_system(mesh, k) for k in range(1, len(mesh.caps) + 1)}
    ext = flow_extensions(mesh, ratio=4.0, transition=0.25)
    _good(ext)
    assert ext.names == mesh.names and len(ext.caps) == len(mesh.caps)
    vol = lambda msh: mesh_defects(msh.vertices, msh.faces)["signed_volume_mm3"]   # noqa: E731
    added = sum(np.pi * r["mean_radius_mm"] ** 2 * 4.0 * r["mean_radius_mm"] for r in refs.values())
    assert abs((vol(ext) - vol(mesh)) / added - 1.0) < 0.03                 # the tubes' volume
    for k, c in enumerate(ext.caps, 1):
        r, e = refs[k], ext.extensions[k]
        L, R = e["length_mm"], e["radius_mm"]
        assert L == pytest.approx(4.0 * r["mean_radius_mm"]) and R == pytest.approx(r["mean_radius_mm"])
        assert np.allclose(c.center, r["barycenter"] + L * r["normal"]) and np.allclose(c.normal, r["normal"])
        v = ext.vertices[np.unique(ext.faces[ext.boundary == k])]
        assert np.abs((v - c.center) @ c.normal).max() < 1e-9 and (_normals(ext, k) @ c.normal > 0).all()
        end = boundary_reference_system(ext, k)
        assert np.allclose(end["barycenter"], c.center, atol=1e-2)
        assert 0 <= R - end["mean_radius_mm"] < 0.01 * R           # a polygon inscribed in the circle
        # every new wall vertex past the transition lies on the cylinder; none lies beyond the end
        new = ext.vertices[len(mesh.vertices):]
        s = (new - r["barycenter"]) @ r["normal"]
        rad = np.linalg.norm((new - r["barycenter"]) - s[:, None] * r["normal"], axis=1)
        mine = (s > -1e-9) & (s <= L + 1e-9) & (rad < R + 1.0)
        straight = mine & (s >= 0.25 * L) & (rad > 0.5 * R)
        assert straight.sum() > 50 and np.abs(rad[straight] - R).max() < 1e-9
        blend = mine & (s > 0) & (s < 0.25 * L)
        assert np.abs(rad[blend] - R).max() < 0.2                           # a nearly round ring, morphing
    write_vtp_mesh(ext, tmp_path / "e.vtp")
    rows = json.loads((tmp_path / "e.vtp.boundaries.json").read_text())["boundaries"]
    for row, c in zip(rows[1:], ext.caps):
        k = row["id"]
        assert row["extension_length_mm"] == pytest.approx(ext.extensions[k]["length_mm"], abs=1e-5)
        assert row["extension_radius_mm"] == pytest.approx(ext.extensions[k]["radius_mm"], abs=1e-5)
        assert np.allclose(row["extension_start_barycenter"], refs[k]["barycenter"], atol=1e-5)
        assert np.allclose(row["ring_barycenter"], c.center, atol=1e-2)
        assert row["ring_mean_radius_mm"] == pytest.approx(ext.extensions[k]["radius_mm"], rel=0.01)
    plain = json.loads(json.dumps(boundaries(mesh)))["boundaries"][1]
    assert "extension_length_mm" not in plain and "ring_mean_radius_mm" in plain


def test_flow_extensions_of_some_caps_and_bad_arguments(y):
    from thalweg.export import flow_extensions
    m, geo, g = y
    mesh = capped_surface(g, "y", m, geo, kinds=("tip", "root"))
    one = flow_extensions(mesh, caps=[2])
    _good(one)
    assert set(one.extensions) == {2} and np.allclose(one.caps[0].center, mesh.caps[0].center)
    assert (one.boundary == 1).sum() == (mesh.boundary == 1).sum()
    sharp = flow_extensions(mesh, transition=0.0)                             # a circle from the first layer
    _good(sharp)
    for bad in (dict(ratio=0.0), dict(transition=1.5), dict(caps=[9])):
        with pytest.raises(ThalwegError):
            flow_extensions(mesh, **bad)


@pytest.mark.data
def test_rings_and_extensions_against_vmtk(vessels_data):
    """The C3N-00704 subtree opened at the nine cuts vmtk was given (research/vessels/flow_ext.py).
    With vmtk's vertex average the ring barycenters and mean radii are vmtk's; the default,
    length-weighted, is within 0.3 mm and 0.05 mm. vmtk's extension vertices in the straight part
    lie on our cylinders to a median 0.03 mm."""
    from scipy.spatial import cKDTree
    from thalweg.export import Cut, boundary_reference_system, flow_extensions
    from thalweg.store import open_store
    names = ("vmtk_input", "flowext_outlets", "vmtk_flowext")
    files = [vessels_data / f"C3N-00704_ctpa0625_{n}.npz" for n in names]
    if not all(f.exists() for f in files):
        pytest.skip("the flow extension reference is not there (research/vessels/vmtk_flowext.py)")
    Z, outlets, V = (np.load(f) for f in files)
    _, geo, _ = open_store(vessels_data / "runs" / "C3N-00704_ctpa0625.lung_vessels.duckn.zip").margin(
        "lung_arteries")
    F = Z["region_field"].astype(np.float32)
    origin = np.asarray(geo.origin, float) + Z["region_lo"] @ np.asarray(geo.directions, float)
    region = Geometry(shape=F.shape, directions=geo.directions, origin=tuple(origin))
    cuts = [Cut(np.asarray(c, float), np.asarray(n, float) / np.linalg.norm(n), float(r), f"cut {i}")
            for i, (c, n, r) in enumerate(zip(outlets["center"], outlets["normal"], outlets["radius"]))]
    mesh = surface(F, region, cuts)
    assert len(mesh.caps) == 9 and not mesh.skipped
    tree = cKDTree(V["ring_barycenter"])
    exact = 0
    for k in range(1, 10):
        v = boundary_reference_system(mesh, k, vmtk_vertex_mean=True)
        d, j = tree.query(v["barycenter"])
        assert d < 0.03 and abs(V["ring_mean_radius"][j] - v["mean_radius_mm"]) < 0.03
        exact += d < 1e-4 and abs(V["ring_mean_radius"][j] - v["mean_radius_mm"]) < 1e-4
        ref = boundary_reference_system(mesh, k)
        assert tree.query(ref["barycenter"])[0] < 0.3
        assert abs(V["ring_mean_radius"][j] - ref["mean_radius_mm"]) < 0.05
    assert exact >= 8
    ext = flow_extensions(mesh, ratio=float(outlets["ratio"]), transition=float(outlets["transition"]))
    _good(ext)
    X = V["verts"].astype(float)
    own = np.argmin(np.stack([np.linalg.norm(X - c.center, axis=1) for c in mesh.caps], 1), 1)
    diffs = []
    for k, c in enumerate(mesh.caps):
        e = ext.extensions[k + 1]
        x = X[own == k] - e["ring_barycenter"]
        s = x @ c.normal
        sel = (s >= 0.25 * e["length_mm"]) & (s < e["length_mm"] - 0.3)
        diffs.append(np.linalg.norm(x[sel] - s[sel, None] * c.normal, axis=1) - e["radius_mm"])
    d = np.concatenate(diffs)
    assert len(d) > 10000 and np.median(np.abs(d)) < 0.05 and np.percentile(np.abs(d), 90) < 0.12
    # no extension of these nine runs back into the vessel
    from thalweg.export import extension_collisions
    assert sum(v > 0 for v in extension_collisions(ext, F, region).values()) <= 1


def _prism(ring_xy, height=6.0, center=None):
    """A closed prism by hand: ``ring_xy`` (counterclockwise) swept from z = 0 to ``height``; its top
    is cap 1 (a fan from ``center``, default the vertex mean), everything else wall."""
    from thalweg.export import Mesh
    ring = np.asarray(ring_xy, float)
    m = len(ring)
    c = ring.mean(0) if center is None else np.asarray(center, float)
    V = np.concatenate([np.c_[ring, np.zeros(m)], np.c_[ring, np.full(m, height)],
                        [[*c, 0.0]], [[*c, height]]])
    i = np.arange(m)
    j = (i + 1) % m
    side = np.concatenate([np.stack([i, j, m + j], 1), np.stack([i, m + j, m + i], 1)])
    bottom = np.stack([np.full(m, 2 * m), j, i], 1)
    top = np.stack([np.full(m, 2 * m + 1), m + i, m + j], 1)
    F = np.concatenate([side, bottom, top])
    B = np.r_[np.zeros(3 * m, np.int64), np.ones(m, np.int64)]
    cut = Cut(np.array([*c, height]), np.array([0.0, 0, 1]), 1.0, "p tip 1", kind="tip")
    mesh = Mesh(V, F, B, ["wall", cut.name], [], [cut])
    _good(mesh)
    return mesh


def _outline(corners, per_edge=6):
    """The corners' polygon with ``per_edge`` points on every edge."""
    c = np.asarray(corners, float)
    return np.concatenate([np.linspace(a, b, per_edge, endpoint=False) for a, b in zip(c, np.roll(c, -1, 0))])


def test_the_ring_system_weighs_by_length_and_the_sidecar_reports_the_ring():
    """A right triangle with 40 extra vertices on one side: the ring's barycenter is its perimeter's
    centroid, (1.5, 1.0), and its mean radius the perimeter's mean distance from there, whatever the
    vertex spacing; the area centroid (4/3, 1) is a different point, and the sidecar has both."""
    from thalweg.export import boundary_reference_system
    ring = np.concatenate([np.linspace([0.0, 0], [4.0, 0], 40, endpoint=False), [[4.0, 0], [0.0, 3]]])
    mesh = _prism(ring, center=(1.0, 1.0))
    ref = boundary_reference_system(mesh, 1)
    assert np.allclose(ref["barycenter"], [1.5, 1.0, 6.0]) and ref["perimeter_mm"] == pytest.approx(12.0)
    corners = np.array([(0.0, 0), (4, 0), (0, 3)])
    exact, t = 0.0, (np.arange(4000) + 0.5) / 4000                     # along each edge, by its length
    for p, q in zip(corners, np.roll(corners, -1, 0)):
        exact += np.linalg.norm(p + t[:, None] * (q - p) - [1.5, 1.0], axis=1).mean() * np.linalg.norm(q - p)
    exact /= 12.0
    assert ref["mean_radius_mm"] == pytest.approx(exact, abs=1e-3)
    plain = boundary_reference_system(mesh, 1, vmtk_vertex_mean=True)  # pulled to the crowded side
    assert np.linalg.norm(plain["barycenter"] - ref["barycenter"]) > 0.8
    assert abs(plain["mean_radius_mm"] - exact) > 0.2
    row = boundaries(mesh)["boundaries"][1]
    assert np.allclose(row["ring_barycenter"], [1.5, 1.0, 6.0])
    assert np.allclose(row["centroid"], [4 / 3, 1.0, 6.0])
    assert row["ring_mean_radius_mm"] == pytest.approx(ref["mean_radius_mm"], abs=1e-6)


def test_the_morph_is_a_smoothstep_over_the_transition_and_layers_follow_the_ring_spacing():
    from thalweg.export import boundary_reference_system, flow_extensions
    mesh = _prism(_outline([(-3, -1), (3, -1), (3, 1), (-3, 1)], 8))         # a 6 x 2 rectangle, 32 vertices
    ref = boundary_reference_system(mesh, 1)
    R, L = ref["mean_radius_mm"], 8.0 * ref["mean_radius_mm"]
    ext = flow_extensions(mesh, ratio=8.0, transition=0.5)
    _good(ext)
    e = ext.extensions[1]
    a, z = e["vertices"]
    layers = int(np.ceil(L / (ref["perimeter_mm"] / 32)))
    assert z - a == layers * 32 + 1 and e["transition"] == 0.5
    new = ext.vertices[a:z - 1].reshape(layers, 32, 3)
    s = new[:, 0, 2] - 6.0
    assert np.allclose(s, L * np.arange(1, layers + 1) / layers)
    assert np.allclose(new[..., 2] - 6.0, s[:, None])
    P = mesh.vertices[cap_ring_of(mesh)]
    t = np.clip(s / (0.5 * L), 0, 1)
    w = t * t * (3 - 2 * t)
    # every layer is (1 - w) ring + w circle point, with w the smoothstep: the circle part is at R
    part = (new[1:, :, :2] - (1 - w[1:, None, None]) * P[None, :, :2]) / w[1:, None, None]
    assert np.allclose(np.linalg.norm(part, axis=2), R)
    assert 0 < w[1] < t[1] < 1                                          # not a linear blend
    straight = s >= 0.5 * L - 1e-9
    assert np.allclose(np.linalg.norm(new[straight][..., :2], axis=2), R) and not np.allclose(
        np.linalg.norm(new[~straight][-1][:, :2], axis=1), R)
    gaps = np.linalg.norm(np.diff(np.vstack([new[-1], new[-1][:1]]), axis=0), axis=1)
    assert np.allclose(gaps, gaps.mean())                               # evenly spaced once round
    sharp = flow_extensions(mesh, ratio=8.0, transition=0.0)
    a, z = sharp.extensions[1]["vertices"]
    assert np.allclose(np.linalg.norm(sharp.vertices[a:z - 1, :2], axis=1), R)      # round at once
    two = flow_extensions(mesh, ratio=0.01)                             # never fewer than two layers
    a, z = two.extensions[1]["vertices"]
    assert z - a == 2 * 32 + 1


def cap_ring_of(mesh, k=1):
    from thalweg.export import cap_ring
    return cap_ring(mesh, k)


@pytest.mark.parametrize("corners", [
    [(0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6)],                                   # an L
    [(0, 0), (6, 0), (6, 2), (2, 2), (2, 4), (6, 4), (6, 6), (0, 6)],                    # a C
])
def test_a_ring_that_is_not_star_shaped_does_not_fold(corners):
    """Rings whose barycenter does not see every vertex: the tube's faces all face away from its
    axis once it is round, the end cap faces out, and the circle's vertices go round once in order."""
    from thalweg.export import flow_extensions
    mesh = _prism(_outline(corners, 5))
    ext = flow_extensions(mesh, ratio=6.0)
    _good(ext)
    e = ext.extensions[1]
    a, z = e["vertices"]
    m = len(cap_ring_of(mesh))
    last = ext.vertices[z - 1 - m:z - 1] - ext.vertices[z - 1]
    turn = np.cross(last, np.roll(last, -1, axis=0))[:, 2]
    assert (turn > 0).all()
    assert np.arctan2(turn, (last * np.roll(last, -1, 0)).sum(1)).sum() == pytest.approx(2 * np.pi)
    f = ext.faces[(ext.boundary == 0) & (ext.faces >= a).all(1)]
    tri = ext.vertices[f]
    mid = tri.mean(1)
    round_part = mid[:, 2] - 6.0 > e["transition"] * e["length_mm"] + 0.5
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    out = ((mid - e["ring_barycenter"])[:, :2] * normal[:, :2]).sum(1)
    assert round_part.sum() > 100 and (out[round_part] > 0).all()
    assert (_normals(ext, 1)[:, 2] > 0).all()
    added = mesh_defects(ext.vertices, ext.faces)["signed_volume_mm3"] - mesh_defects(
        mesh.vertices, mesh.faces)["signed_volume_mm3"]
    assert 0.8 < added / (np.pi * e["radius_mm"] ** 2 * e["length_mm"]) < 1.3


def test_extension_bookkeeping_and_refusals():
    from thalweg.export import Mesh, cap_ring, flow_extensions
    mesh = _prism(_outline([(-2, -2), (2, -2), (2, 2), (-2, 2)], 4))
    ext = flow_extensions(mesh, caps=[1, 1])                           # a repeated id is one extension
    _good(ext)
    assert np.allclose(ext.extensions[1]["cut_center"], [0, 0, 6.0]) and ext.caps[0].center[2] > 6.0
    with pytest.raises(ThalwegError, match="already has"):
        flow_extensions(ext)
    for bad in ([1.5], [0], [2]):
        with pytest.raises(ThalwegError):
            flow_extensions(mesh, caps=bad)
    with pytest.raises(ThalwegError, match="no cap"):
        cap_ring(mesh, 5)
    # a wall face missing: refused with check, returned without
    broken = Mesh(mesh.vertices, mesh.faces[1:], mesh.boundary[1:], mesh.names, [], mesh.caps)
    with pytest.raises(ThalwegError, match="not closed"):
        flow_extensions(broken)
    assert len(flow_extensions(broken, check=False).faces) > len(broken.faces)
    # a cap pinched at one vertex (two triangles sharing only a corner) is not a disk
    V = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
    pinched = Mesh(V, np.array([[0, 1, 2], [0, 3, 4]]), np.array([1, 1]), ["wall", "c"], [],
                   [Cut(V[0], np.array([0.0, 0, 1]), 1.0, "c")])
    with pytest.raises(ThalwegError, match="not a disk"):
        cap_ring(pinched, 1)


def test_extensions_keep_earlier_ones_and_collisions_are_counted(y):
    """Extending one cap, then another, keeps both records. An extension pointed into the structure
    (the root's normal reversed) is counted as running into it; the honest ones are not."""
    from dataclasses import replace
    from thalweg.export import extension_collisions, flow_extensions
    m, geo, g = y
    mesh = capped_surface(g, "y", m, geo, kinds=("tip", "root"))
    ext = flow_extensions(flow_extensions(mesh, caps=[1]), caps=[2, 3])
    _good(ext)
    assert sorted(ext.extensions) == [1, 2, 3]
    hits = extension_collisions(ext, m, geo)
    assert hits == {1: 0, 2: 0, 3: 0} and ext.extensions[1]["vertices_inside_structure"] == 0
    rows = json.loads(json.dumps(boundaries(ext)))["boundaries"]
    assert all(r["extension_vertices_inside_structure"] == 0 for r in rows[1:])
    assert all({"extension_transition", "cut_center", "extension_radius_mm"} <= set(r) for r in rows[1:])
    # a tube of the daughters' size laid along the trunk's axis lies inside the trunk
    k = 2
    e = dict(ext.extensions[k])
    a, z = e["vertices"]
    inside = ext.vertices.copy()
    n = ext.caps[k - 1].normal
    s = (inside[a:z] - e["ring_barycenter"]) @ n
    rad = (inside[a:z] - e["ring_barycenter"]) - s[:, None] * n
    axis = np.array([0.0, 0, 1])
    ortho = np.cross(axis, [1.0, 0, 0])
    frame = np.stack([np.cross(ortho, axis), ortho], 0)
    inside[a:z] = (np.array([0.0, 0, 8]) + s[:, None] * axis * 0.8
                   + 0.3 * (rad @ rad[:2].T / 4)[:, :1] * frame[0])
    moved = replace(ext, vertices=inside, extensions={k: dict(e, ring_barycenter=np.array([0.0, 0, 8]))},
                    caps=[replace(c, normal=axis) if i == k - 1 else c for i, c in enumerate(ext.caps)])
    assert extension_collisions(moved, m, geo)[k] > 50


def test_a_ring_with_crowded_vertices_does_not_drag_thin_triangles_along_the_tube():
    """60 of a square ring's 63 vertices on one side: next to the ring the triangles are as thin as
    the ring's own edges, but once the tube is round they are all well shaped."""
    from thalweg.export import flow_extensions
    crowded = np.linspace([-2.0, -2], [2.0, -2], 60, endpoint=False)
    mesh = _prism(np.concatenate([crowded, [[2.0, -2], [2, 2], [-2, 2]]]))
    ext = flow_extensions(mesh)
    _good(ext)
    e = ext.extensions[1]
    a, _ = e["vertices"]
    f = ext.faces[(ext.faces >= a).all(1) & (ext.boundary == 0)]
    tri = ext.vertices[f]
    u, v, w = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], tri[:, 2] - tri[:, 1]
    quality = 2 * np.sqrt(3) * np.linalg.norm(np.cross(u, v), axis=1) / ((u * u) + (v * v) + (w * w)).sum(1)
    round_part = tri.mean(1)[:, 2] - 6.0 > e["transition"] * e["length_mm"]
    assert round_part.sum() > 1000 and quality[round_part].min() > 0.5


def _c_ring(outer=3.0, inner=2.6, gap_deg=60.0, n=40):
    """A C-shaped section (an annulus with a gap), counterclockwise: its barycenter is outside it."""
    half = np.radians(gap_deg) / 2
    t = np.linspace(half, 2 * np.pi - half, n)
    return np.concatenate([np.c_[outer * np.cos(t), outer * np.sin(t)],
                           np.c_[inner * np.cos(t[::-1]), inner * np.sin(t[::-1])]])


@pytest.mark.parametrize("gap", [60.0, 10.0])
def test_a_ring_the_blend_would_cross_is_extruded_unchanged(gap):
    """A thin C (and a hook, 350 degrees round): blending it toward a circle passes some layers
    through themselves, so the extension is a straight prism of the ring, ending in the ring's own
    shape, every layer a simple polygon; the sidecar says so."""
    from thalweg.export import _simple, flow_extensions
    mesh = _prism(_c_ring(gap_deg=gap), center=(-2.8, 0.0))              # the fan from inside the C
    ext = flow_extensions(mesh)
    _good(ext)
    e = ext.extensions[1]
    assert e["end_shape"] == "ring"
    a, z = e["vertices"]
    m = len(cap_ring_of(mesh))
    layers = ext.vertices[a:z].reshape(-1, m, 3)
    ring = mesh.vertices[cap_ring_of(mesh)]
    assert np.allclose(layers[..., :2], ring[None, :, :2])              # a straight prism
    assert all(_simple(q[:, :2]) for q in layers)
    assert (_normals(ext, 1)[:, 2] > 0).all() and len(np.unique(ext.faces[ext.boundary == 1])) == m
    assert boundaries(ext)["boundaries"][1]["extension_end_shape"] == "ring"


def test_simple_polygon_test():
    from thalweg.export import _simple
    square = np.array([[0.0, 0], [1, 0], [1, 1], [0, 1]])
    bowtie = np.array([[0.0, 0], [1, 1], [1, 0], [0, 1]])
    assert _simple(square) and not _simple(bowtie) and _simple(_c_ring())
    touching = np.array([[0.0, 0], [2, 0], [2, 2], [1, 0], [0, 2]])            # a vertex on an edge
    assert not _simple(touching)


def test_the_circle_is_turned_to_meet_the_ring():
    """A square ring with its vertices started at an arbitrary corner: every vertex's angle on the
    final circle is within one vertex spacing of its angle on the ring - the circle is turned to
    the ring, not started at an arbitrary angle."""
    from thalweg.export import flow_extensions
    ring = np.roll(_outline([(-2, -2), (2, -2), (2, 2), (-2, 2)], 6), 7, axis=0)
    mesh = _prism(ring)
    ext = flow_extensions(mesh)
    e = ext.extensions[1]
    a, z = e["vertices"]
    m = len(ring)
    last = ext.vertices[z - 1 - m:z - 1, :2]
    order = cap_ring_of(mesh)
    start = np.arctan2(*mesh.vertices[order][:, [1, 0]].T)
    end = np.arctan2(last[:, 1], last[:, 0])
    gap = np.abs(np.angle(np.exp(1j * (end - start))))
    assert gap.max() < 2 * np.pi / m + 0.2
