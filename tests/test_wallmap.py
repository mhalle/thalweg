"""Rays in the field, wall maps in thalweg's and vmtk's conventions, and vmtk's branch metrics."""
import numpy as np
import pytest

from thalweg.centerlines import graph_from_tree
from thalweg.graph import Points, Source, TubeGraph
from thalweg.kernel import medial
from thalweg.kernel.rays import first_crossing
from thalweg.vmtk.centerlines import Centerlines
from thalweg.vmtk.metrics import branch_metrics, group_cell_sets
from thalweg.wallmap import angles, edge_wall_map, ostium, wall_maps, write_wall_maps
from phantoms import tube_field


def _graph(m, geo, name="t"):
    T = medial.trace(m, geo)
    nodes, edges, pos, rad, s = graph_from_tree(T, name, m, geo, Source(), {"connectivity": "field"})
    return TubeGraph(structures=[s], nodes=nodes, edges=edges, points=Points(position=pos, radius=rad))


def test_rays_find_the_wall_of_a_tube():
    m, geo = tube_field([np.array([[0.0, 0, 0], [0, 0, 30]])], [np.array([2.0, 2.0])])
    a = angles(36)
    d = np.stack([np.cos(a), np.sin(a), 0 * a], 1)
    r = first_crossing(m, geo, np.tile([0.0, 0, 15], (36, 1)), d, 8.0)
    assert np.abs(r - 2.0).max() < 0.05
    r = first_crossing(m, geo, np.tile([0.5, 0, 15], (2, 1)), [[1.0, 0, 0], [-1.0, 0, 0]], 8.0)
    assert abs(r[0] - 1.5) < 0.05 and abs(r[1] - 2.5) < 0.05              # off the axis
    along = [[0.0, 0, 15]], [[0.0, 0, 1]]
    assert np.isnan(first_crossing(m, geo, *along, 8.0)[0])               # no wall within reach
    assert abs(first_crossing(m, geo, *along, 40.0)[0] - 17.0) < 0.1      # the rounded end, 15 + 2
    assert np.isnan(first_crossing(m, geo, [[5.0, 0, 15]], [[1.0, 0, 0]], 8.0)[0])   # starts outside
    per_ray = first_crossing(m, geo, np.tile([0.0, 0, 15], (2, 1)), d[:2], [8.0, 1.0])
    assert abs(per_ray[0] - 2.0) < 0.05 and np.isnan(per_ray[1])          # its own reach


def test_edge_wall_map_of_a_tube_and_a_y(tmp_path):
    m, geo = tube_field([np.array([[0.0, 0, 0], [0, 0, 40]])], [np.array([2.5, 2.5])])
    g = _graph(m, geo)
    e = max(g.edges, key=lambda x: x.length_mm)
    w = edge_wall_map(g, e.id, m, geo, step=1.0, angle_count=24)
    assert w.radius_mm.shape == (len(w.arc_length_mm), 24) and len(w.angle_rad) == 24
    assert w.angle_rad[0] == -np.pi and np.all(np.diff(w.arc_length_mm) > 0)
    assert w.arc_length_mm[-1] <= e.length_mm + 1e-9          # positions along the edge itself
    mid = (w.arc_length_mm > 6) & (w.arc_length_mm < w.arc_length_mm[-1] - 6)
    # the traced line lies within 0.1 mm of the axis, so the radius varies that much around it;
    # its mean over the angles, and the wall points themselves, are the cylinder's
    assert np.abs(w.radius_mm[mid] - 2.5).max() < 0.15 and not ostium(w)[mid].any()
    assert np.abs(w.radius_mm[mid].mean(1) - 2.5).max() < 0.03
    pts = w.wall_points()[mid]
    assert np.abs(np.hypot(pts[..., 0], pts[..., 1]) - 2.5).max() < 0.05     # on the cylinder
    # a side branch leaving a main tube at a right angle: the main tube's maps have an opening
    # toward it (+x) beside the junction, and wall everywhere else
    lines = [np.array([[0.0, 0, 0], [0, 0, 40]]), np.array([[0.0, 0, 20], [18.0, 0, 20]])]
    m, geo = tube_field(lines, [np.array([3.0, 3.0]), np.array([1.5, 1.5])])
    g = _graph(m, geo, "y")
    maps = wall_maps(g, "y", m, geo, step=0.5)
    assert set(maps) == {e.id for e in g.edges if e.length_mm >= 3.0}
    main = [k for k in maps if np.nanmedian(maps[k].radius_mm) > 2.5]
    assert len(main) == 2
    opened = 0
    for k in main:
        w, o = maps[k], ostium(maps[k])
        z = w.centers[:, 2]
        away = np.abs(z - 20) > 3.5
        assert not o[away].any() and abs(np.nanmedian(w.radius_mm[away]) - 3.0) < 0.1
        assert (w.directions[o][:, 0] > 0.9).all()                    # opened rays point down the branch
        opened += int(o.sum())
    assert opened > 0
    write_wall_maps(maps, tmp_path / "w.npz", "y")
    z = np.load(tmp_path / "w.npz")
    k = int(z["edges"][0])
    assert str(z["structure"]) == "y" and z[f"edge_{k}_radius_mm"].shape == maps[k].radius_mm.shape
    assert z[f"edge_{k}_center_mm"].shape == (len(maps[k].arc_length_mm), 3) and len(z["angle_rad"]) == 72


def _straight(n=21, radius=2.0):
    """One centerline along +z split into three cells (group 0, a blanked group 1, group 2), with
    abscissas = z and the normal +x: the metrics of a point are then its z and its azimuth."""
    z = np.linspace(0, 30, n)
    P = np.stack([0 * z, 0 * z, z], 1)
    cells = [np.arange(0, 8), np.arange(7, 14), np.arange(13, n)]
    return Centerlines(P, cells, {"MaximumInscribedSphereRadius": np.full(n, radius), "Abscissas": z.copy(),
                                  "ParallelTransportNormals": np.tile([1.0, 0, 0], (n, 1))},
                       {"GroupIds": np.array([0, 1, 2]), "Blanking": np.array([0, 1, 0]),
                        "CenterlineIds": np.zeros(3, int), "TractIds": np.array([0, 1, 2])})


def test_branch_metrics_are_position_along_and_angle_around():
    cl = _straight()
    assert group_cell_sets(cl, 0, True) == [[0, 1]] and group_cell_sets(cl, 0, False) == [[0]]
    x = np.array([[3.0, 0, 4.2], [0, 3.0, 4.2], [-3.0, 0, 4.2], [0, -3.0, 4.2], [2.0, 2.0, 25.3]])
    a, phi = branch_metrics(x, [0, 0, 0, 0, 2], cl)
    assert np.allclose(a, [4.2, 4.2, 4.2, 4.2, 25.3])
    assert np.allclose(np.abs(phi[[0, 2]]), [0.0, np.pi]) and np.allclose(np.abs(phi[[1, 3]]), np.pi / 2)
    assert phi[1] == -phi[3] and abs(abs(phi[4]) - np.pi / 4) < 1e-12        # signed, one way around
    # past the group's own cell the abscissa follows the bifurcation cell beside it
    a_bif, _ = branch_metrics([[3.0, 0, 12.0]], [0], cl)
    assert abs(a_bif[0] - 12.0) < 1e-12
    # vmtk's defect reads the segment's end value: the abscissa jumps to the next sample (1.5 mm apart)
    a_vmtk, _ = branch_metrics(x[:1], [0], cl, vmtk_interp=True)
    assert abs(a_vmtk[0] - 4.5) < 1e-12
    # a group the centerlines do not have, and a point on the axis
    a, phi = branch_metrics([[3.0, 0, 4.2], [0.0, 0, 4.2]], [9, 0], cl)
    assert np.isnan(a[0]) and np.isnan(phi[0]) and a[1] == pytest.approx(4.2) and np.isnan(phi[1])


def _mapping(vessels_data):
    from scipy.spatial import cKDTree
    f = vessels_data / "C3N-00704_ctpa0625_vmtk_mapping.npz"
    surface = vessels_data / "C3N-00704_ctpa0625_vmtk_input.npz"
    if not f.exists() or not surface.exists():
        pytest.skip(f"{f.name} or {surface.name} is not there (research/vessels/vmtk_mapping.py)")
    M = np.load(f)
    cells = np.split(M["cell_ids"], np.cumsum(M["cell_len"])[:-1])
    cl = Centerlines(M["cl_points"], cells,
                     {"MaximumInscribedSphereRadius": M["cl_radius"], "Abscissas": M["cl_abscissa"],
                      "ParallelTransportNormals": M["cl_normal"]},
                     {"GroupIds": M["cell_group"], "Blanking": M["cell_blank"],
                      "CenterlineIds": M["cell_centerline"], "TractIds": M["cell_tract"]})
    vertex = cKDTree(np.load(surface)["verts"]).query(M["surf_points"])[0] < 1e-6
    return M, cl, vertex


@pytest.mark.data
def test_branch_metrics_are_vmtks(vessels_data):
    """At the 21,221 vertices of the surface vmtk mapped (C3N-00704 subtree): with vmtk's end-point
    defect switched on, the abscissa and the angle are vmtk's to rounding; interpolating as vmtk
    meant to moves the abscissa by up to one centerline step (0.3 mm)."""
    M, cl, vertex = _mapping(vessels_data)
    x, group = M["surf_points"][vertex], M["surf_group"][vertex]
    a, phi = branch_metrics(x, group, cl, vmtk_interp=True)
    assert np.abs(a - M["surf_abscissa"][vertex]).max() < 1e-10
    assert np.abs(np.angle(np.exp(1j * (phi - M["surf_angle"][vertex])))).max() < 1e-9
    a, phi = branch_metrics(x, group, cl)
    d = np.abs(a - M["surf_abscissa"][vertex])
    assert 0.05 < np.median(d) < 0.2 and d.max() < 0.31


@pytest.mark.data
@pytest.mark.slow
def test_ray_cast_radius_is_vmtks_distance_to_centerlines(vessels_data):
    """Wall maps in vmtk's coordinates against vmtk's DistanceToCenterlines at the same (group,
    abscissa, angle), away from each group's ends: a median |difference| under 0.04 mm."""
    from scipy.interpolate import RegularGridInterpolator
    from thalweg.store import open_store
    from thalweg.wallmap import group_wall_map
    M, cl, vertex = _mapping(vessels_data)
    m, geo, _ = open_store(vessels_data / "runs" / "C3N-00704_ctpa0625.lung_vessels.duckn.zip").margin(
        "lung_arteries")
    diffs = []
    for g in np.unique(M["cell_group"][M["cell_blank"] == 0]):
        s = vertex & (M["surf_group"] == g)
        if s.sum() < 20:
            continue
        try:
            w = group_wall_map(cl, int(g), m, geo)
        except ValueError:
            continue
        sg, order = np.unique(w.arc_length_mm, return_index=True)
        if len(sg) < 4:
            continue
        rr = w.radius_mm[order]
        f = RegularGridInterpolator((sg, np.r_[w.angle_rad, np.pi]), np.hstack([rr, rr[:, :1]]),
                                    bounds_error=False, fill_value=np.nan)
        a, phi = M["surf_abscissa"][s], M["surf_angle"][s]
        d = f(np.stack([a, phi], 1)) - M["surf_dist"][s]
        diffs.append(d[(a > sg[0] + 1.0) & (a < sg[-1] - 1.0) & np.isfinite(d)])
    d = np.concatenate(diffs)
    assert len(d) > 10000 and np.median(np.abs(d)) < 0.04 and abs(np.median(d)) < 0.02
    assert np.percentile(np.abs(d), 90) < 0.12


def _box(shape=(40, 12, 12)):
    from rankfield.geometry import Geometry
    return Geometry(shape=shape, directions=((1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0)), origin=(0.0, 0, 0))


def test_the_crossing_is_interpolated_between_samples():
    """A margin that falls linearly along x (1 - 0.4 (x - 3)): its zero is at x = 5.5 exactly, from
    any origin, for any step and any length of the direction vector."""
    geo = _box()
    x = np.arange(40, dtype=np.float32)[:, None, None]
    m = np.broadcast_to(1.0 - 0.4 * (x - 3.0), geo.shape).astype(np.float32)
    for step in (0.05, 0.3, 0.7):
        origins = [[3.0, 5, 5], [4.2, 5, 5], [5.49, 5, 5]]
        r = first_crossing(m, geo, origins, [[1.0, 0, 0], [2.0, 0, 0], [0.1, 0, 0]], 20.0, step)
        assert np.allclose(r, [2.5, 1.3, 0.01], atol=1e-5)
    assert np.isnan(first_crossing(m, geo, [[5.51, 5, 5]], [[1.0, 0, 0]], 20.0)[0])       # starts outside
    assert np.isnan(first_crossing(m, geo, [[3.0, 5, 5]], [[1.0, 0, 0]], 2.4)[0])         # beyond its reach
    assert abs(first_crossing(m, geo, [[3.0, 5, 5]], [[1.0, 0, 0]], 2.6)[0] - 2.5) < 1e-5
    # toward -x the margin only rises: the ray runs off the grid, and that is no wall
    assert np.isnan(first_crossing(m, geo, [[3.0, 5, 5]], [[-1.0, 0, 0]], 20.0)[0])
    assert np.isnan(first_crossing(m, geo, [[-3.0, 5, 5]], [[1.0, 0, 0]], 20.0)[0])       # starts off it
    for bad in (dict(reach=0.0), dict(reach=np.nan), dict(reach=5.0, step=0.0)):
        with pytest.raises(ValueError):
            first_crossing(m, geo, [[3.0, 5, 5]], [[1.0, 0, 0]], **bad)
    with pytest.raises(ValueError):
        first_crossing(m, geo, [[3.0, 5, 5]], [[0.0, 0, 0]], 5.0)


def test_ray_chunks_do_not_change_the_distances(monkeypatch):
    import thalweg.kernel.rays as R
    m, geo = tube_field([np.array([[0.0, 0, 0], [0, 0, 30]])], [np.array([2.0, 2.0])])
    rng = np.random.default_rng(0)
    o = np.stack([rng.uniform(-1, 1, 50), rng.uniform(-1, 1, 50), rng.uniform(5, 25, 50)], 1)
    d = rng.normal(size=(50, 3))
    whole = first_crossing(m, geo, o, d, 12.0)
    monkeypatch.setattr(R, "RAY_CHUNK", 700)                            # a few rays at a time
    assert np.array_equal(first_crossing(m, geo, o, d, 12.0), whole, equal_nan=True)
    assert np.isfinite(whole).sum() > 40 and len(np.unique(np.round(whole, 3))) > 40


def test_ostium_marks_missing_and_far_rays():
    from thalweg.wallmap import WallMap, outside_stations
    r = np.array([[2.0, 2.0, 2.0, np.nan], [2.0, 2.0, 2.0, 3.7], [2.0, 2.0, 2.2, 3.5], [np.nan] * 4])
    w = WallMap(np.arange(4.0), angles(4), r, np.zeros((4, 3)), np.zeros((4, 4, 3)))
    o = ostium(w)
    assert o.tolist() == [[False, False, False, True], [False, False, False, True],
                          [False, False, False, False], [True] * 4]          # 3.7 > 1.8 x 2; 3.5 < 1.8 x 2.1
    assert outside_stations(w).tolist() == [False, False, False, True]
    assert ostium(w, factor=1.5)[2, 3]


def test_wall_maps_of_a_tapering_tube_and_their_file(tmp_path):
    """Radius 4 mm narrowing to 1.2: the reach follows the edge's LARGEST radius, so the wide end's
    wall is found; samples without a radius do not break the stations; the file holds the map."""
    m, geo = tube_field([np.array([[0.0, 0, 0], [0, 0, 50]])], [np.array([4.0, 1.2])])
    g = _graph(m, geo)
    e = max(g.edges, key=lambda x: x.length_mm)
    a, b = e.point_range
    rad = list(g.points.radius)
    rad[a + 3] = rad[a + 4] = -1.0
    g = g.model_copy(update={"points": g.points.model_copy(update={"radius": rad})})
    w = edge_wall_map(g, e.id, m, geo, step=1.0)
    mid = (w.arc_length_mm > 6) & (w.arc_length_mm < w.arc_length_mm[-1] - 6)
    assert np.isfinite(w.radius_mm[mid]).all() and np.nanmax(w.radius_mm[mid]) > 3.0
    assert np.abs(np.diff(w.arc_length_mm) - 1.0).max() < 0.05
    short = edge_wall_map(g, e.id, m, geo, step=1.0, reach_factor=0.1, reach_extra=0.0)
    assert np.isnan(short.radius_mm[mid]).all()
    assert wall_maps(g, "t", m, geo, min_length_mm=1000.0) == {}
    maps = wall_maps(g, "t", m, geo, step=1.0)
    write_wall_maps(maps, tmp_path / "w.npz", "t")
    z = np.load(tmp_path / "w.npz")
    assert np.array_equal(z[f"edge_{e.id}_arc_length_mm"], maps[e.id].arc_length_mm)
    assert np.array_equal(z[f"edge_{e.id}_center_mm"], maps[e.id].centers)
    assert np.allclose(z[f"edge_{e.id}_radius_mm"], maps[e.id].radius_mm, atol=1e-5, equal_nan=True)
    assert np.array_equal(z["angle_rad"], angles(72))


def test_group_wall_map_projects_the_normal_and_skips_repeated_points():
    """A straight group along z with a repeated point and a normal tilted out of the normal plane:
    the rays are still perpendicular to the line, start at the normal's projection, and find the tube."""
    from thalweg.wallmap import group_wall_map
    m, geo = tube_field([np.array([[0.0, 0, 0], [0, 0, 30]])], [np.array([2.0, 2.0])])
    z = np.r_[np.linspace(5, 15, 21), 15.0, np.linspace(15.5, 25, 20)]
    n = len(z)
    tilted = np.tile([1.0, 0, 0.6], (n, 1))
    cl = Centerlines(np.stack([0 * z, 0 * z, z], 1), [np.arange(n)],
                     {"MaximumInscribedSphereRadius": np.full(n, 2.0), "Abscissas": z - 5.0,
                      "ParallelTransportNormals": tilted},
                     {"GroupIds": np.array([4]), "Blanking": np.array([0])})
    w = group_wall_map(cl, 4, m, geo, angle_count=8, smooth=0.0)
    assert len(w.arc_length_mm) == n - 1 and np.abs(w.directions[..., 2]).max() < 1e-12
    k = int(np.argmin(np.abs(w.angle_rad)))
    assert np.allclose(w.directions[:, k], [1.0, 0, 0]) and np.abs(w.radius_mm - 2.0).max() < 0.05
    with pytest.raises(ValueError):
        group_wall_map(cl, 9, m, geo)


def test_default_metrics_interpolate_the_normal_and_the_radius():
    """One segment whose normal turns from +x to +y: a point at its middle, toward (1, 1, 0), is at
    angle 0 from the interpolated normal (the segment's first normal alone would give 45 degrees).
    Two cells of one group with different radii: the abscissa is weighted by the interpolated
    radius squared."""
    P = np.array([[0.0, 0, 0], [0, 0, 3]])
    cl = Centerlines(P, [np.array([0, 1])],
                     {"MaximumInscribedSphereRadius": np.array([1.0, 1.0]), "Abscissas": np.array([0.0, 3.0]),
                      "ParallelTransportNormals": np.array([[1.0, 0, 0], [0, 1.0, 0]])},
                     {"GroupIds": np.array([0]), "Blanking": np.array([0]), "CenterlineIds": np.array([0]),
                      "TractIds": np.array([0])})
    _, phi = branch_metrics([[2.0, 2.0, 1.5]], [0], cl)
    assert abs(phi[0]) < 1e-12
    _, phi = branch_metrics([[2.0, 2.0, 1.5]], [0], cl, vmtk_interp=True)
    assert abs(abs(phi[0]) - np.pi / 4) < 1e-12                       # vmtk reads the end normal, +y
    two = Centerlines(np.concatenate([P, P]), [np.array([0, 1]), np.array([2, 3])],
                      {"MaximumInscribedSphereRadius": np.array([1.0, 3.0, 1.0, 1.0]),
                       "Abscissas": np.array([0.0, 3.0, 10.0, 13.0]),
                       "ParallelTransportNormals": np.tile([1.0, 0, 0], (4, 1))},
                      {"GroupIds": np.array([0, 0]), "Blanking": np.array([0, 0]),
                       "CenterlineIds": np.array([0, 1]), "TractIds": np.array([0, 0])})
    a, _ = branch_metrics([[2.0, 0, 1.5]], [0], two)
    assert abs(a[0] - (4 * 1.5 + 1 * 11.5) / 5) < 1e-12               # radii 2 and 1 at the middle
