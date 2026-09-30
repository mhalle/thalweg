"""vmtk's centerline-only stages, run once, saved as the oracles thalweg's ports must reproduce.

    uv run --offline --no-project --python 3.12 --with vmtk --with scipy \\
        python tests/oracle/vmtk_centerline_oracle.py phantom           # -> tests/fixtures/vmtk_oracle/phantom/
    uv run --offline --no-project --python 3.12 --with vmtk --with scipy \\
        python tests/oracle/vmtk_centerline_oracle.py C3N-00704_ctpa0625  # -> $VESSELS_DATA/oracle/<run>/
    uv run --offline --no-project --python 3.12 --with vmtk --with scipy \\
        python tests/oracle/vmtk_centerline_oracle.py small               # every SMALL fixture below
    uv run --offline --no-project --python 3.12 --with vmtk --with scipy \\
        python tests/oracle/vmtk_centerline_oracle.py sections            # the phantom's bifurcation sections

vmtk is an oracle only, never a thalweg dependency: this script runs in the isolated vmtk env
(vmtk 1.5.2 wheels, VTK 9.6.2) and imports nothing from thalweg. Every stage is vmtk's own script
class with the arguments written here; each output is saved whole (points, cells, every point,
cell and field array) in the layout ``thalweg.vmtk.Centerlines.from_npz`` reads:

- ``points`` (N, 3) float64, ``cell_ids`` (flat point ids), ``cell_len`` (points per cell);
- ``pd__<name>`` point arrays, ``cd__<name>`` cell arrays, ``fd__<name>`` field arrays.

Inputs:
- ``phantom``: a synthetic three-level tree (5 targets), built here, small enough to freeze into
  ``tests/fixtures`` so the ports are tested without case data;
- the SMALL synthetic layouts (``SMALL``, each a few tens of KB, frozen into ``tests/fixtures`` next to
  the phantom): each isolates a splitting / grouping / fallback rule the phantom never reaches, so
  those rules are pinned without case data. ``stub`` (a side branch that leaves the trunk halfway
  up and ends 0.5 mm outside its tube), ``reentry`` (a daughter that crosses its sister's tube and
  leaves it again: two splits on one centerline), ``trifurcation`` (three daughters from one
  point), ``short_trunk`` (a trunk shorter than its radius: the touching walk runs off the
  centerlines' start), ``single_line`` (no splitting point, coordinates not float32-exact),
  ``hairpin`` (a daughter that curves back to end beside the root: two candidate tracts for the
  trunk, the last one wrong - vmtk's last-candidate grouping and its MergeTracts last-tract defect),
  ``loopback`` (three candidate tracts for one decision - the first, the deepest and the last are
  three different tracts, so the first-candidate, minimum and last-candidate rules all differ),
  ``short_branches`` (a 1 mm intermediate branch between two bifurcations), ``coincident``
  (repeated consecutive points and an exactly duplicated centerline) and ``degenerate_radii`` (a
  zero-radius tip and a taper as steep as the arc), ``late_start`` (a centerline that starts on
  another's tube wall: the touching walk runs off its start, vmtk's (0, 0) split) and
  ``near_sample`` (split points 0.0005 mm from a sample: vmtk's 1e-6 mm^2 insertion tolerance);
- a run name: our centerlines of the C3N-00704 left-lung subtree in vmtk's convention (one
  polyline per target, source to target, resampled to 0.3 mm, radius looked up from the densified
  graph points), exactly as ``research/vessels/vmtk_branch.py`` and ``vmtk_mapping.py`` built them.

The offset stages leave the points of centerlines that never cross the reference group as vmtk
leaves them - uninitialized memory, different on every run - so tests compare covered points only.

Stages (vmtk defaults unless written): attributes; branch extractor (on the attributes output, so
the split points carry interpolated abscissas and normals - the InterpolateTuple defect shows
there); bifurcation reference systems; offset attributes (reference group: vmtk's default -1, the
first reference system's group, recorded as ``root_group``); merge (Length 0, MergeBlanked 1); resampling (Length 0.5); smoothing (100 iterations,
factor 0.1); centerline geometry (no line smoothing); branch geometry; bifurcation vectors.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vmtk import vmtkscripts

R = "MaximumInscribedSphereRadius"
HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("VESSELS_DATA", Path.home() / "tmp/data/vessels"))


def resample(p, step=0.3):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    t = np.arange(0, s[-1] + 1e-9, step)
    return np.stack([np.interp(t, s, p[:, a]) for a in range(3)], 1)


def lines_polydata(lines, radii):
    pts = np.concatenate(lines); rad = np.concatenate(radii)
    pd = vtk.vtkPolyData()
    p = vtk.vtkPoints(); p.SetData(numpy_to_vtk(pts.astype(np.float64), deep=True)); pd.SetPoints(p)
    ca = vtk.vtkCellArray(); off = 0
    for L in lines:
        ids = vtk.vtkIdList()
        for i in range(len(L)):
            ids.InsertNextId(off + i)
        ca.InsertNextCell(ids); off += len(L)
    pd.SetLines(ca)
    a = numpy_to_vtk(rad.astype(np.float64), deep=True); a.SetName(R); pd.GetPointData().AddArray(a)
    return pd


def dump(pd):
    """A polydata's points, cells and every array, in the npz layout Centerlines.from_npz reads."""
    out = {"points": vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64) if pd.GetPoints() else np.zeros((0, 3))}
    ids, lens = [], []
    for k in range(pd.GetNumberOfCells()):
        c = pd.GetCell(k)
        ids.extend(c.GetPointId(i) for i in range(c.GetNumberOfPoints())); lens.append(c.GetNumberOfPoints())
    out["cell_ids"] = np.array(ids, np.int64); out["cell_len"] = np.array(lens, np.int64)
    for prefix, data in (("pd", pd.GetPointData()), ("cd", pd.GetCellData()), ("fd", pd.GetFieldData())):
        for i in range(data.GetNumberOfArrays()):
            a = data.GetArray(i)
            if a is None:                          # a string array (GetArray returns None for those)
                continue
            out[f"{prefix}__{a.GetName()}"] = vtk_to_numpy(a).copy()
    return out


def phantom():
    """A three-level synthetic tree, 5 targets, shared prefixes exact (the same resampled points).

    Trunk along +z with a slight curve; at the top it splits into a left and a right daughter
    (unequal radii, out of plane); the left daughter splits again into three (a trifurcation-like
    close pair of bifurcations) and the right one into two. Radius shrinks along each branch.
    """
    def arc(p0, d0, d1, length, n=400):
        t = np.linspace(0, 1, n)[:, None]
        d = (1 - t) * d0 + t * d1
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        seg = np.cumsum(d * (length / (n - 1)), axis=0) - d[0] * (length / (n - 1))
        return p0 + seg
    def rad(r0, r1, n=400):
        return np.linspace(r0, r1, n)
    edges = {}
    trunk = arc(np.zeros(3), np.array([0.0, 0.05, 1.0]), np.array([0.1, 0.0, 1.0]), 30.0); edges["t"] = (trunk, rad(3.0, 2.6))
    top = trunk[-1]
    L = arc(top, np.array([-1.0, 0.2, 1.2]), np.array([-1.0, 0.4, 0.4]), 25.0); edges["L"] = (L, rad(2.0, 1.7))
    Rr = arc(top, np.array([1.0, -0.3, 1.0]), np.array([0.8, -0.2, 0.2]), 28.0); edges["R"] = (Rr, rad(2.3, 1.9))
    LL = arc(L[-1], np.array([-1.0, 1.0, 0.5]), np.array([-0.2, 1.0, 0.0]), 18.0); edges["LL"] = (LL, rad(1.3, 1.0))
    LM = arc(L[-1], np.array([-1.0, 0.0, 0.3]), np.array([-1.0, -0.3, -0.4]), 20.0); edges["LM"] = (LM, rad(1.5, 1.1))
    LR = arc(L[-1], np.array([-0.4, -0.6, 1.0]), np.array([0.0, -1.0, 1.0]), 15.0); edges["LR"] = (LR, rad(1.1, 0.9))
    RA = arc(Rr[-1], np.array([1.0, 0.8, 0.1]), np.array([0.5, 1.0, -0.3]), 20.0); edges["RA"] = (RA, rad(1.6, 1.2))
    RB = arc(Rr[-1], np.array([1.0, -0.9, -0.2]), np.array([0.6, -1.0, -0.8]), 22.0); edges["RB"] = (RB, rad(1.7, 1.3))
    paths = [["t", "L", "LL"], ["t", "L", "LM"], ["t", "L", "LR"], ["t", "R", "RA"], ["t", "R", "RB"]]
    lines, radii = [], []
    for pth in paths:
        p = np.concatenate([edges[e][0] if i == 0 else edges[e][0][1:] for i, e in enumerate(pth)])
        r = np.concatenate([edges[e][1] if i == 0 else edges[e][1][1:] for i, e in enumerate(pth)])
        q = resample(p)
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        sq = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
        lines.append(q); radii.append(np.interp(np.arange(len(q)) * 0.3, s, r))
    return lines, radii


# -- the small synthetic layouts (built by the review of 2026-09-29, fidelity/make_inputs.py) --------

def _edge(p0, d, length, r0, r1, n=200):
    d = np.asarray(d, float); d = d / np.linalg.norm(d)
    return p0 + np.linspace(0, length, n)[:, None] * d, np.linspace(r0, r1, n)


def _build(edges, paths, step=0.3):
    """One line per path of edge names, joined (shared points dropped), resampled to ``step``."""
    lines, radii = [], []
    for pth in paths:
        p = np.concatenate([edges[e][0] if i == 0 else edges[e][0][1:] for i, e in enumerate(pth)])
        r = np.concatenate([edges[e][1] if i == 0 else edges[e][1][1:] for i, e in enumerate(pth)])
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        q = resample(p, step)
        lines.append(q); radii.append(np.interp(np.arange(len(q)) * step, s, r))
    return lines, radii


def small_trifurcation():
    E = {"t": _edge(np.zeros(3), [0, 0.05, 1], 20, 2.2, 2.0)}
    top = E["t"][0][-1]
    for k, (az, r) in enumerate([(0, 1.4), (2.1, 1.3), (4.2, 1.2)]):
        E[f"d{k}"] = _edge(top, [np.sin(0.7) * np.cos(az), np.sin(0.7) * np.sin(az), np.cos(0.7)], 15, r, r * 0.8)
    return _build(E, [["t", "d0"], ["t", "d1"], ["t", "d2"]])


def _short_edges():
    E = {"t": _edge(np.zeros(3), [0, 0, 1], 20, 2.0, 2.0)}
    top = E["t"][0][-1]
    E["L"] = _edge(top, [-1, 0, 1], 1.0, 1.6, 1.6, 20)
    Lend = E["L"][0][-1]
    E["LA"] = _edge(Lend, [-1, 0.6, 0.6], 15, 1.2, 1.0)
    E["LB"] = _edge(Lend, [-1, -0.6, 0.4], 15, 1.1, 0.9)
    E["R"] = _edge(top, [1, 0.2, 0.8], 15, 1.5, 1.2)
    E["t1"] = _edge(np.zeros(3), [0, 0, 1], 10, 2.0, 2.0)
    mid = E["t1"][0][-1]
    E["t2"] = _edge(mid, [0, 0, 1], 10, 2.0, 2.0)
    E["S"] = _edge(mid, [1, 0, 0.05], 2.5, 0.8, 0.8, 30)
    return E


def small_short_branches():
    E = _short_edges()
    E["t"] = (np.concatenate([E["t1"][0], E["t2"][0][1:]]), np.concatenate([E["t1"][1], E["t2"][1][1:]]))
    return _build(E, [["t", "L", "LA"], ["t", "L", "LB"], ["t", "R"]])


def small_stub():
    return _build(_short_edges(), [["t1", "t2", "L", "LA"], ["t1", "S"], ["t1", "t2", "R"]])


def small_coincident():
    E = {"t": _edge(np.zeros(3), [0, 0.05, 1], 20, 2.0, 1.9)}
    top = E["t"][0][-1]
    E["a"] = _edge(top, [-0.7, 0, 1], 15, 1.5, 1.2)
    E["b"] = _edge(top, [0.7, 0.2, 1], 15, 1.3, 1.1)
    lines, radii = _build(E, [["t", "a"], ["t", "b"]])
    L2, R2 = [], []
    for L, r in zip(lines, radii):
        rep = np.ones(len(L), int)
        rep[[5, 60, 66, 67, 70, len(L) - 1]] = 2            # repeated samples, near the split and at the tip
        L2.append(np.repeat(L, rep, 0)); R2.append(np.repeat(r, rep))
    return L2 + [L2[0].copy()], R2 + [R2[0].copy()]


def small_reentry():
    from scipy.interpolate import CubicSpline
    E = {"t": _edge(np.zeros(3), [0, 0, 1], 20, 2.0, 2.0)}
    top = E["t"][0][-1]
    E["a"] = _edge(top, [-1, 0, 1.2], 25, 1.5, 1.3)
    ctrl = np.array([[0, 0, 0], [6, 0, 5], [0, 0, 12], [-10, 0, 10.5], [-18, 2, 8]], float)
    bp = top + CubicSpline(np.linspace(0, 1, len(ctrl)), ctrl)(np.linspace(0, 1, 300))
    E["b"] = (bp, np.linspace(1.2, 1.0, len(bp)))
    return _build(E, [["t", "a"], ["t", "b"]])


def small_degenerate_radii():
    E = {"t": _edge(np.zeros(3), [0, 0, 1], 20, 2.0, 2.0)}
    top = E["t"][0][-1]
    E["a"] = _edge(top, [-1, 0, 1], 12, 1.5, 0.0)
    E["b"] = _edge(top, [1, 0.3, 1], 12, 1.4, 1.4)
    lines, radii = _build(E, [["t", "a"], ["t", "b"]])
    L, r = lines[1], radii[1].copy()
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(L, axis=0), axis=1))]
    tail = s > s[-1] - 1.4
    r[tail] = s[-1] - s[tail]                                 # |dr| == |dp| at the tip: den == 0
    radii[1] = r
    return lines, radii


def small_short_trunk():
    E = {"t": _edge(np.zeros(3), [0, 0, 1], 1.5, 2.0, 2.0, 20)}
    top = E["t"][0][-1]
    E["a"] = _edge(top, [-1, 0, 1], 15, 1.5, 1.3)
    E["b"] = _edge(top, [1, 0.3, 1], 15, 1.4, 1.2)
    return _build(E, [["t", "a"], ["t", "b"]])


def small_single_line():
    x = np.arange(0, 10.0001, 0.3)
    L0 = np.stack([x, 0.1 * np.sin(x), 0.37 + 0 * x], 1) + 1e-3 / 3
    return [L0], [np.full(len(L0), 1.1)]


def small_hairpin():
    from scipy.interpolate import CubicSpline
    E = {"t": (np.linspace([0, 0, 0], [0, 0, 20.0], 200), np.linspace(2.0, 1.9, 200)),
         "a": (np.linspace([0, 0, 20.0], [-10, 0, 32], 200), np.linspace(1.5, 1.2, 200))}
    ctrl = np.array([[0, 0, 20], [6, 0, 23], [9, 0, 12], [6, 0, 3], [1.0, 0, 0.8]], float)
    E["b"] = (CubicSpline(np.linspace(0, 1, 5), ctrl)(np.linspace(0, 1, 400)), np.linspace(1.6, 1.5, 400))
    return _build(E, [["t", "a"], ["t", "b"]])


def small_late_start():
    """Centerline B starts ON centerline A's tube wall (at one of A's samples: A's tube value there is
    exactly 0) with a bigger radius and leaves sideways, so its exit from A's tube lies inside its own
    first sphere: the touching walk runs off B's start and vmtk splits at (0, 0.0), leaving B a
    one-point first tract. B's blanked tract then starts exactly on the wall of A's blanked tract
    (tube value 0, which the grouping's ``< -1e-12`` does not count as inside). Binary-exact
    coordinates (A every 0.25 mm)."""
    zA = np.arange(0, 30.0001, 0.25)
    A = np.stack([0 * zA, 0 * zA, zA], 1)
    d = np.array([1.0, 0.0, 0.2]) / np.linalg.norm([1.0, 0.0, 0.2])
    B = np.array([2.0, 0, 10.0]) + np.arange(0, 15.0001, 0.3)[:, None] * d
    return [A, B], [np.full(len(A), 2.0), np.linspace(2.5, 1.5, len(B))]


def small_near_sample():
    """A Y whose touching-sphere split points land 0.0005 mm from a sample (one extra sample on each
    daughter, placed from a first run): past it on daughter a, before it on daughter b. Within
    vmtk's 1e-6 mm^2 insertion tolerance, so the split point is added neither to the tract that ends
    at a's sample nor to the one that starts at b's (the two insertion tests of SplitCenterline)."""
    z = np.arange(0, 20.0001, 0.25)
    trunk = np.stack([0 * z, 0 * z, z], 1)
    da, db = np.array([-0.6, 0.0, 0.8]), np.array([0.8, 0.2, 0.566]) / np.linalg.norm([0.8, 0.2, 0.566])
    lines, radii = [], []
    for d, (r0, r1), extra in ((da, (1.6, 1.2), 0.4104570188440114), (db, (1.5, 1.1), 0.5288727928045939)):
        s = np.sort(np.r_[np.arange(0.3, 15.0001, 0.3), extra])
        lines.append(np.r_[trunk, trunk[-1] + s[:, None] * d])
        radii.append(np.r_[np.full(len(trunk), 2.0), r0 + (r1 - r0) * s / 15.0])
    return lines, radii


def small_loopback():
    """Three candidate tracts for one relabel decision, each a different rule's pick. Centerline 1's
    daughter leaves the trunk at the top, comes down and passes right over the root (inside the
    trunk's tube, so it leaves that tube again: a second split), then loops round and ends beside the
    root. The trunk of centerline 0 then has three candidates on centerline 1 (same blanking, mutual
    in-tube): its trunk (first in cell order), the passing tract (the deepest: the trunk widens from
    2.0 to 2.4, and the pass crosses its axis 1 mm up) and the looping tail (last, and shallowest).
    vmtk takes the last, the corrected rule the deepest, a first-candidate rule the trunk; all three
    group differently."""
    from scipy.interpolate import CubicSpline
    E = {"t": (np.linspace([0, 0, 0], [0, 0, 20.0], 200), np.linspace(2.0, 2.4, 200)),
         "a": (np.linspace([0, 0, 20.0], [-10, 0, 32], 200), np.linspace(1.5, 1.2, 200))}
    ctrl = np.array([[0, 0, 20], [6, 0, 23], [9, -6, 12], [3, -6, 1], [0, -2, 1], [0, 0, 1], [0, 2, 1],
                     [2, 7, 2], [8, 5, 3], [8, -3, 2], [4, -5, 1], [0.6, -1.2, 0.5]], float)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(ctrl, axis=0), axis=1))]
    E["b"] = (CubicSpline(s, ctrl)(np.linspace(0, s[-1], 1200)), np.full(1200, 1.5))
    return _build(E, [["t", "a"], ["t", "b"]])


SMALL = {n[len("small_"):]: f for n, f in list(globals().items()) if n.startswith("small_") and callable(f)}


def case(run):
    """Our centerlines of the subtree, in vmtk's convention, as vmtk_branch.py built them."""
    Z = np.load(DATA / f"{run}_vmtk_input.npz")
    tree = cKDTree(Z["ours_points"])
    paths = np.split(Z["path_points"], np.cumsum(Z["path_len"])[:-1])
    lines = [resample(p) for p in paths if len(p) > 3]
    return lines, [Z["ours_radius"][tree.query(L)[1]] for L in lines]


def sphere_union(lines, radii, h=0.35, pad=3.0):
    """The phantom's tubes as a field on a lattice: max over the centerline samples of r - |x - p|
    (a union of spheres every 0.3 mm, within 0.01 mm of the swept tube), and the lattice's
    (origin, spacing, shape). thalweg's test rebuilds the same field with the same formula."""
    P = np.concatenate(lines)
    r = np.concatenate(radii)
    lo = P.min(0) - r.max() - pad
    hi = P.max(0) + r.max() + pad
    shape = tuple(int(np.ceil((b - a) / h)) + 1 for a, b in zip(lo, hi))
    X = lo + h * np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1).reshape(-1, 3)
    d, i = cKDTree(P).query(X, k=32, workers=-1)
    f = (r[i] - d).max(1)
    return f.reshape(shape), lo, h


def surface_of(f, lo, h):
    """Marching cubes of the field's zero set (VTK), triangulated and cleaned."""
    img = vtk.vtkImageData(); img.SetDimensions(*f.shape); img.SetOrigin(*lo); img.SetSpacing(h, h, h)
    img.GetPointData().SetScalars(numpy_to_vtk(np.ascontiguousarray(f.transpose(2, 1, 0)).ravel(), deep=True))
    mc = vtk.vtkMarchingCubes(); mc.SetInputData(img); mc.SetValue(0, 0.0); mc.ComputeNormalsOff()
    tri = vtk.vtkTriangleFilter(); tri.SetInputConnection(mc.GetOutputPort())
    clean = vtk.vtkCleanPolyData(); clean.SetInputConnection(tri.GetOutputPort()); clean.Update()
    return clean.GetOutput()


def sections():
    """vmtkbifurcationsections on the phantom: its split centerlines (as ``run`` makes them), the
    surface of its tubes (``sphere_union``) cut into groups by vmtkBranchClipper, sections one and
    two distance spheres from each bifurcation. -> tests/fixtures/vmtk_oracle/phantom/
    bifurcation_sections_{1,2}.npz (cell data: vmtk's arrays; points and polygon cells: the sections)."""
    out_dir = HERE.parent / "fixtures" / "vmtk_oracle" / "phantom"
    lines, radii = phantom()
    cl = lines_polydata(lines, radii)
    ca = vmtkscripts.vmtkCenterlineAttributes(); ca.Centerlines = cl; ca.Execute()
    be = vmtkscripts.vmtkBranchExtractor(); be.Centerlines = ca.Centerlines; be.RadiusArrayName = R; be.Execute()
    split = be.Centerlines
    f, lo, h = sphere_union(lines, radii)
    bc = vmtkscripts.vmtkBranchClipper(); bc.Surface = surface_of(f, lo, h); bc.Centerlines = split
    bc.RadiusArrayName = R; bc.GroupIdsArrayName = "GroupIds"; bc.BlankingArrayName = "Blanking"; bc.Execute()
    meta = {}
    for n in (1, 2):
        bs = vmtkscripts.vmtkBifurcationSections(); bs.Surface = bc.Surface; bs.Centerlines = split
        bs.NumberOfDistanceSpheres = n; bs.Execute()
        np.savez_compressed(out_dir / f"bifurcation_sections_{n}.npz", **dump(bs.BifurcationSections))
        meta[n] = bs.BifurcationSections.GetNumberOfCells()
    print(json.dumps({"sections": meta, "field": {"origin": lo.tolist(), "spacing": h, "shape": list(f.shape)}}))


def main():
    names = sys.argv[1:] or ["phantom"]
    if names == ["sections"]:
        return sections()
    if names == ["small"]:
        names = list(SMALL)
    for name in names:
        if name == "phantom":
            lines, radii = phantom(); out_dir = HERE.parent / "fixtures" / "vmtk_oracle" / "phantom"
        elif name in SMALL:
            lines, radii = SMALL[name](); out_dir = HERE.parent / "fixtures" / "vmtk_oracle" / name
        else:
            lines, radii = case(name); out_dir = DATA / "oracle" / name
        run(name, lines, radii, out_dir, full=name not in SMALL, undefined=SMALL_UB.get(name, ()))


# the stages a SMALL fixture skips: the per-line ones on the input (resampling, smoothing, geometry) do
# not see the splitting and grouping those layouts exist for, and ``offset`` repeats ``offset_keep``
# with ReplaceAttributes on; the phantom covers them all. Skipping them keeps the small fixtures at a
# few tens of KB each (the first nine: ~370 KB instead of ~850).
SMALL_SKIP = ("offset", "merge_resampled_unblanked", "resampling", "smoothing", "geometry", "geometry_smoothed")
# stages where vmtk reads past the end of a one-point cell (undefined behavior, not a reference): late_start's
# first tract of B. The port raises ValueError there (bifurcation vectors) or measures the point.
SMALL_UB = {"late_start": ("branch_geometry", "bifurcation_vectors")}


def run(name, lines, radii, out_dir, full=True, undefined=()):
    out_dir.mkdir(parents=True, exist_ok=True)
    T = {}

    def save(stage, pd):
        if (full or stage not in SMALL_SKIP) and stage not in undefined:
            np.savez_compressed(out_dir / f"{stage}.npz", **dump(pd))

    cl = lines_polydata(lines, radii); save("input", cl)
    t = time.time()
    ca = vmtkscripts.vmtkCenterlineAttributes(); ca.Centerlines = cl; ca.Execute(); save("attributes", ca.Centerlines)
    T["attributes"] = time.time() - t; t = time.time()
    be = vmtkscripts.vmtkBranchExtractor(); be.Centerlines = ca.Centerlines; be.RadiusArrayName = R; be.Execute()
    split = be.Centerlines; save("extract", split)
    T["extract"] = time.time() - t; t = time.time()
    rs = vmtkscripts.vmtkBifurcationReferenceSystems(); rs.Centerlines = split; rs.RadiusArrayName = R
    rs.BlankingArrayName = "Blanking"; rs.GroupIdsArrayName = "GroupIds"; rs.Execute(); save("frames", rs.ReferenceSystems)
    T["frames"] = time.time() - t; t = time.time()
    # ReferenceGroupId must name a BIFURCATION group (one with a reference system). -1, vmtk's
    # default, takes the first reference system's group. (Passing the first cell's group - a branch -
    # makes vmtk log "Invalid ReferenceGroupId" and return its input unchanged: the 09-24 research
    # mapping run did that, so its "offset" stage was a no-op.)
    # Without a reference system (no bifurcation) vmtk logs "ReferenceSystems empty" and outputs nothing;
    # the port raises ValueError there, so the offset stages are not written (root_group null).
    root_group = None
    if rs.ReferenceSystems.GetNumberOfPoints():
        oa = vmtkscripts.vmtkCenterlineOffsetAttributes(); oa.Centerlines = split; oa.ReferenceSystems = rs.ReferenceSystems
        oa.ReferenceGroupId = -1; oa.ReplaceAttributes = 1; oa.Execute(); save("offset", oa.Centerlines)
        root_group = int(oa.ReferenceGroupId)
        oa2 = vmtkscripts.vmtkCenterlineOffsetAttributes(); oa2.Centerlines = split; oa2.ReferenceSystems = rs.ReferenceSystems
        oa2.ReferenceGroupId = root_group; oa2.ReplaceAttributes = 0; oa2.Execute(); save("offset_keep", oa2.Centerlines)
    T["offset"] = time.time() - t; t = time.time()
    mg = vmtkscripts.vmtkCenterlineMerge(); mg.Centerlines = split; mg.Length = 0.0; mg.MergeBlanked = 1; mg.Execute()
    save("merge", mg.Centerlines)
    mg2 = vmtkscripts.vmtkCenterlineMerge(); mg2.Centerlines = split; mg2.Length = 0.5; mg2.MergeBlanked = 0; mg2.Execute()
    save("merge_resampled_unblanked", mg2.Centerlines)
    T["merge"] = time.time() - t; t = time.time()
    rsm = vmtkscripts.vmtkCenterlineResampling(); rsm.Centerlines = cl; rsm.Length = 0.5; rsm.Execute(); save("resampling", rsm.Centerlines)
    T["resampling"] = time.time() - t; t = time.time()
    sm = vmtkscripts.vmtkCenterlineSmoothing(); sm.Centerlines = cl; sm.NumberOfSmoothingIterations = 100
    sm.SmoothingFactor = 0.1; sm.Execute(); save("smoothing", sm.Centerlines)
    T["smoothing"] = time.time() - t; t = time.time()
    cg = vmtkscripts.vmtkCenterlineGeometry(); cg.Centerlines = cl; cg.LineSmoothing = 0; cg.Execute(); save("geometry", cg.Centerlines)
    cgs = vmtkscripts.vmtkCenterlineGeometry(); cgs.Centerlines = cl; cgs.LineSmoothing = 1; cgs.OutputSmoothedLines = 0
    cgs.NumberOfSmoothingIterations = 100; cgs.SmoothingFactor = 0.1; cgs.Execute(); save("geometry_smoothed", cgs.Centerlines)
    T["geometry"] = time.time() - t; t = time.time()
    bg = vmtkscripts.vmtkBranchGeometry(); bg.Centerlines = split; bg.LineSmoothing = 0; bg.Execute(); save("branch_geometry", bg.GeometryData)
    T["branch_geometry"] = time.time() - t; t = time.time()
    bv = vmtkscripts.vmtkBifurcationVectors(); bv.Centerlines = split; bv.ReferenceSystems = rs.ReferenceSystems
    bv.NormalizeBifurcationVectors = 0; bv.Execute(); save("bifurcation_vectors", bv.BifurcationVectors)
    T["bifurcation_vectors"] = time.time() - t
    meta = dict(source=name, vtk=vtk.vtkVersion.GetVTKVersion(), radius_array=R, root_group=root_group,
                skipped=([] if full else list(SMALL_SKIP)) + list(undefined),
                lines=len(lines), points=int(sum(len(L) for L in lines)), seconds={k: round(v, 3) for k, v in T.items()},
                stages={"attributes": "defaults", "extract": "on attributes output, RadiusArrayName=" + R,
                        "frames": "defaults", "offset": "ReferenceGroupId=-1 (vmtk default: first frame's group = root_group), ReplaceAttributes=1",
                        "offset_keep": "ReferenceGroupId=root_group, ReplaceAttributes=0",
                        "merge": "Length=0, MergeBlanked=1", "merge_resampled_unblanked": "Length=0.5, MergeBlanked=0",
                        "resampling": "Length=0.5 (on input)", "smoothing": "100 iterations, factor 0.1 (on input)",
                        "geometry": "LineSmoothing=0 (on input)",
                        "geometry_smoothed": "LineSmoothing=1, 100 iterations, factor 0.1, OutputSmoothedLines=0 (on input)",
                        "branch_geometry": "LineSmoothing=0 (on extract)", "bifurcation_vectors": "Normalize=0 (on extract + frames)"})
    (out_dir / "oracle.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
