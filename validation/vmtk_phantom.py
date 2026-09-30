"""vmtk and thalweg against the truth on analytic phantoms: who is right where they disagree?

On the MSB-02664 subtree the two methods agree to ~0.1 mm for radius < 2.5 mm but differ by
~0.6 mm (median) on wider vessels. A phantom has a known axis, so it can say which is right.

    uv run python validation/vmtk_phantom.py prep            # thalweg env: fields, surfaces, seeds
    uv run --offline --no-project --python 3.12 --with vmtk --with scipy \\
        python validation/vmtk_phantom.py vmtk              # vmtk env: vmtkCenterlines per phantom
    uv run python validation/vmtk_phantom.py compare [N ...]  # thalweg env: both vs the true axis,
                                                              # thalweg with N ridge passes (default 1)

Phantoms (0.7 mm grid, slope 10.6 logit/mm): Y bifurcations with a wide trunk (r 3, 5 and 8 mm)
and daughters at 60 degrees, a wide arc (r 5 mm, curvature radius 30 mm), and flattened tubes
(``flat_*``: elliptic sections a x 1.5 mm as true signed distances, a/b 2, 2.5 and 3, the 3:1 one
also rolled 35 degrees about its axis and turned off the lattice onto an anisotropic grid; the
older ``ellipse_*`` files, if present, are the phantom suite's shallower elliptic field). vmtk sees
the field's zero set (marching cubes at 0, unsmoothed - as in the research comparisons), seeded at
the true ends; thalweg traces the field. Scores: distance from each method's points to the true
axis, away from ends and the junction (median, p95), and radius error there; for thalweg also its
path between the seeds' ends alone (``thalweg<N>_main``) and its tip count, since it traces a
tree where vmtk computes one line.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("VESSELS_DATA", Path.home() / "tmp/data/vessels")) / "validation" / "vmtk_phantom"


def phantoms():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import phantom_suite as PS
    out = {}
    for r0 in (3.0, 5.0, 8.0):
        segs, truth = PS.y_phantom(60, r0=r0, r1=0.6 * r0, r2=0.6 * r0, L0=6 * r0, L=6 * r0)
        out[f"y_r{int(r0)}"] = (segs, truth)
    segs, truth = PS.arc(radius_of_curvature=30.0, r=5.0, sweep_deg=120)
    out["arc_r5"] = (segs, truth)
    # flattened tubes along x, semi-axes a (flat width) x b: a true signed distance, so the field is
    # as steep across the width as across the depth (the phantom suite's elliptic() is not: its
    # slope across the width is b/a); "rolled" turns the section 35 degrees about the tube's axis,
    # "oblique" turns the whole tube off the lattice onto phantom_suite's anisotropic grid
    for a, b, roll, oblique in ((3.0, 1.5, 0, False), (3.75, 1.5, 0, False), (4.5, 1.5, 0, False),
                                (4.5, 1.5, 35, False), (4.5, 1.5, 0, True), (4.5, 1.5, 35, True)):
        name = f"flat_{a:g}x{b:g}" + (f"_roll{roll}" if roll else "") + ("_oblique" if oblique else "")
        out[name] = (elliptic_sdf(a, b, 40.0, np.radians(roll)),
                     dict(ends=[(0.0, 0, 0), (40.0, 0, 0)], junctions=[],
                          axes=[((0, 0, 0), (40, 0, 0), b, b)], semi_axes=(a, b), oblique=oblique))
    return out, PS


def elliptic_sdf(a, b, L, roll=0.0, samples=4000):
    """Signed distance (mm, positive inside) to a straight tube along x from 0 to L with flat ends
    and an elliptic section of semi-axes a (along y turned by ``roll``) and b."""
    from scipy.spatial import cKDTree
    t = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    rim = np.stack([a * np.cos(t), b * np.sin(t)], 1)
    tree = cKDTree(rim)
    c, s = np.cos(roll), np.sin(roll)

    def f(X):
        y = c * X[:, 1] + s * X[:, 2]
        z = -s * X[:, 1] + c * X[:, 2]
        d2 = tree.query(np.stack([y, z], 1))[0]
        d2 = np.where((y / a) ** 2 + (z / b) ** 2 <= 1, d2, -d2)
        dx = np.minimum(X[:, 0], L - X[:, 0])
        inside = np.minimum(d2, dx)
        outside = -np.sqrt(np.maximum(-d2, 0) ** 2 + np.maximum(-dx, 0) ** 2)
        return np.where((d2 > 0) & (dx > 0), inside, outside)
    return f


def prep():
    from skimage.measure import marching_cubes
    from thalweg.kernel.field import to_world
    OUT.mkdir(parents=True, exist_ok=True)
    ph, PS = phantoms()
    for name, (segs, truth) in ph.items():
        if callable(segs):                                 # a flattened tube: a field function, its axis
            f, segs = segs, truth["axes"]
            a, b = truth["semi_axes"]
            box = (-4.0, -a - 4, -a - 4), (44.0, a + 4, a + 4)
            if truth.get("oblique"):
                m, geo = PS.field_oblique(f, *box)
                to_w = lambda p: np.asarray(p, float) @ PS.Q.T + PS.SHIFT       # noqa: E731
                segs = [(to_w(p0), to_w(p1), r0, r1) for p0, p1, r0, r1 in segs]
                truth = dict(truth, ends=[to_w(e) for e in truth["ends"]])
            else:
                m, geo = PS.field_of(f, *box, 0.7)
        else:
            pts = np.concatenate([[s[0], s[1]] for s in segs]).astype(float)
            rmax = max(max(s[2], s[3]) for s in segs)
            m, geo = PS.field_of(PS.chain_distance(segs), pts.min(0) - rmax - 4, pts.max(0) + rmax + 4, 0.7)
        v, f, _, _ = marching_cubes(np.pad(m, 1, constant_values=-8.0), level=0.0)
        ends = np.asarray(truth["ends"], float)
        np.savez(OUT / f"{name}.npz", verts=to_world(geo, v - 1.0), faces=f, source=ends[0], targets=ends[1:],
                 margin=m, origin=np.asarray(geo.origin), directions=np.asarray(geo.directions),
                 segs=np.array([np.r_[a, b, ra, rb] for a, b, ra, rb in segs], float))
        print(f"{name}: {len(v)} vertices, source {ends[0].round(1)}, {len(ends) - 1} targets")


def run_vmtk():
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
    from vmtk import vmtkscripts
    for p in sorted(OUT.glob("*.npz")):
        if p.stem.endswith("_vmtk"):
            continue
        Z = np.load(p)
        surf = vtk.vtkPolyData()
        sp = vtk.vtkPoints()
        sp.SetData(numpy_to_vtk(Z["verts"].astype(np.float64), deep=True))
        surf.SetPoints(sp)
        F = Z["faces"].astype(np.int64)
        cells = vtk.vtkCellArray()
        cells.ImportLegacyFormat(numpy_to_vtkIdTypeArray(np.c_[np.full(len(F), 3), F].ravel(), deep=True))
        surf.SetPolys(cells)
        loc = vtk.vtkPointLocator()
        loc.SetDataSet(surf)
        loc.BuildLocator()
        src = [loc.FindClosestPoint(Z["source"])]
        tgt = [loc.FindClosestPoint(t) for t in Z["targets"]]
        cl = vmtkscripts.vmtkCenterlines()
        cl.Surface = surf
        cl.SeedSelectorName = "idlist"
        cl.SourceIds = src
        cl.TargetIds = tgt
        cl.AppendEndPoints = 0
        cl.Resampling = 1
        cl.ResamplingStepLength = 0.3
        cl.Execute()
        c = cl.Centerlines
        np.savez(OUT / f"{p.stem}_vmtk.npz", points=vtk_to_numpy(c.GetPoints().GetData()),
                 radius=vtk_to_numpy(c.GetPointData().GetArray("MaximumInscribedSphereRadius")))
        print(p.stem, c.GetNumberOfPoints(), "points")


def seg_axis(P, segs):
    best = np.full(len(P), np.inf)
    rad = np.zeros(len(P))
    for s in segs:
        a, b, ra, rb = s[:3], s[3:6], s[6], s[7]
        ab = b - a
        t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
        d = np.linalg.norm(P - (a + t[:, None] * ab), axis=1)
        take = d < best
        best[take] = d[take]
        rad[take] = (ra + t * (rb - ra))[take]
    return best, rad


def _main_path(margin, geo, passes, source, target):
    """thalweg's path between the ends nearest the seeds (the one line vmtk computes there), without
    the side branches, and the tree's tip count."""
    from thalweg.centerlines import graph_from_tree
    from thalweg.graph import Points, Source, TubeGraph
    from thalweg.kernel import medial
    T = medial.trace(margin, geo, ridge_passes=passes)
    nodes, edges, pos, rad, s = graph_from_tree(T, "t", margin, geo, Source(), {})
    g = TubeGraph(structures=[s], nodes=nodes, edges=edges, points=Points(position=pos, radius=rad))
    deg = g.degree()
    ends = [nd for nd in g.nodes if deg[nd.id] == 1]
    a = min(ends, key=lambda nd: np.linalg.norm(np.array(nd.position) - source)).id
    b = min(ends, key=lambda nd: np.linalg.norm(np.array(nd.position) - target)).id
    t = g.tree("t")

    def up(n):
        out = []
        while n in t.parent:
            out.append(t.parent[n])
            n = g.edges[t.parent[n]].start_node
        return out
    pa, pb = up(a), up(b)
    path = [e for e in pa + pb if e not in set(pa) & set(pb)]
    P = np.concatenate([g.edge_points(e) for e in path])
    R = np.concatenate([g.edge_radius(e) for e in path])
    return P, R, sum(nd.kind == "tip" for nd in g.nodes)


def compare(passes=(1,)):
    from rankfield.geometry import Geometry
    from thalweg.kernel import medial
    rows = []
    for p in sorted(OUT.glob("*.npz")):
        if p.stem.endswith("_vmtk"):
            continue
        Z = np.load(p)
        V = np.load(OUT / f"{p.stem}_vmtk.npz")
        geo = Geometry(shape=Z["margin"].shape, directions=tuple(map(tuple, Z["directions"])),
                       origin=tuple(Z["origin"]))
        segs = Z["segs"]
        keyp = np.vstack([Z["source"][None], Z["targets"], segs[:, 3:6][:1]])
        rmax = segs[:, 6].max()
        row = dict(phantom=p.stem, trunk_radius_mm=float(segs[0, 6]))
        methods = [("vmtk", V["points"], V["radius"])]
        for n in passes:
            T = medial.trace(Z["margin"], geo, ridge_passes=n)
            methods.append((f"thalweg{n}", np.concatenate([np.array(s["points"]) for s in T.segments]),
                            np.concatenate([np.array(s["radius"]) for s in T.segments])))
            P, R, tips = _main_path(Z["margin"], geo, n, Z["source"], Z["targets"][-1])
            methods.append((f"thalweg{n}_main", P, R))
            row[f"thalweg{n}_tips"] = tips
        for name, P, R in methods:
            away = np.linalg.norm(P[:, None] - keyp[None], axis=2).min(1) > 1.5 * rmax
            if away.sum() < 3:                             # the method gave no usable line
                row[f"{name}_axis_mm_median"] = row[f"{name}_axis_mm_p95"] = None
                row[f"{name}_radius_error_mm_median"] = None
                row[f"{name}_points"] = int(len(P))
                continue
            d, r_true = seg_axis(P, segs)
            row[f"{name}_axis_mm_median"] = round(float(np.median(d[away])), 3)
            row[f"{name}_axis_mm_p95"] = round(float(np.percentile(d[away], 95)), 3)
            row[f"{name}_radius_error_mm_median"] = round(float(np.median((R - r_true)[away])), 3)
        rows.append(row)
        print(json.dumps(row))
    (OUT / "results.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        compare(tuple(int(a) for a in sys.argv[2:]) or (1,))
    else:
        {"prep": prep, "vmtk": run_vmtk}[sys.argv[1]]()
