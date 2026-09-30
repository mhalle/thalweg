"""vmtk and thalweg against the truth on analytic phantoms: who is right where they disagree?

On the MSB-02664 subtree the two methods agree to ~0.1 mm for radius < 2.5 mm but differ by
~0.6 mm (median) on wider vessels. A phantom has a known axis, so it can say which is right.

    uv run python validation/vmtk_phantom.py prep            # thalweg env: fields, surfaces, seeds
    uv run --offline --no-project --python 3.12 --with vmtk --with scipy \\
        python validation/vmtk_phantom.py vmtk              # vmtk env: vmtkCenterlines per phantom
    uv run python validation/vmtk_phantom.py compare [N ...]  # thalweg env: both vs the true axis,
                                                              # thalweg with N ridge passes (default 1)

Phantoms (0.7 mm grid, slope 10.6 logit/mm): Y bifurcations with a wide trunk (r 3, 5 and 8 mm)
and daughters at 60 degrees, a wide arc (r 5 mm, curvature radius 30 mm), and two flattened
tubes (elliptic sections, semi-axes 3 x 1.5 and 4.5 x 1.5 mm): the inscribed radius there is the
minor semi-axis, 1.5 mm, which both methods should read. vmtk sees the
field's zero set (marching cubes at 0, unsmoothed - as in the research comparisons), seeded at
the true ends; thalweg traces the field. Scores: distance from each method's points to the true
axis, away from ends and the junction (median, p95), and radius error there.
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
    for a, b in ((3.0, 1.5), (4.5, 1.5)):                 # flattened tubes: semi-axes a x b, a/b 2 and 3
        f, truth = PS.elliptic(a=a, b=b, L=40.0)
        out[f"ellipse_{a:g}x{b:g}"] = (f, truth)
    return out, PS


def prep():
    from skimage.measure import marching_cubes
    from thalweg.kernel.field import to_world
    OUT.mkdir(parents=True, exist_ok=True)
    ph, PS = phantoms()
    for name, (segs, truth) in ph.items():
        if callable(segs):                                 # an elliptic tube: a field function, its axis
            f, segs = segs, truth["axes"]
            a, b = truth["semi_axes"]
            m, geo = PS.field_of(f, (-4.0, -a - 4, -b - 4), (44.0, a + 4, b + 4), 0.7)
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
        for name, P, R in methods:
            away = np.linalg.norm(P[:, None] - keyp[None], axis=2).min(1) > 1.5 * rmax
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
