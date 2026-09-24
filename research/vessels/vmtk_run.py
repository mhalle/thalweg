"""vmtk's centerlines on the field's surface, against ours (the oracle comparison).

    uv run --no-project --python 3.12 --with vmtk --with scipy python bench/vessels/vmtk_run.py [RUN]

Reads DATA/<RUN>_vmtk_input.npz (vmtk_prep.py), builds a vtkPolyData, runs vmtkCenterlines
(Voronoi diagram + Eikonal + backtrace, radius = MaximumInscribedSphereRadius) from the source
to every target, and compares every vmtk centerline point with our nearest centerline point:
position gap and radius difference. Isolated env: vmtk 1.5.x wheels bring their own VTK.
"""
import os, sys, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
from vmtk import vmtkscripts

DATA = Path(os.environ.get("VESSELS_DATA", Path.home() / "tmp/data/vessels"))
run = sys.argv[1] if len(sys.argv) > 1 else "C3N-00704_ctpa0625"
Z = np.load(DATA / f"{run}_vmtk_input.npz")

pd = vtk.vtkPolyData()
pts = vtk.vtkPoints(); pts.SetData(numpy_to_vtk(Z["verts"].astype(np.float64), deep=True)); pd.SetPoints(pts)
F = Z["faces"].astype(np.int64)
cells = vtk.vtkCellArray()
cells.ImportLegacyFormat(numpy_to_vtkIdTypeArray(np.c_[np.full(len(F), 3), F].ravel(), deep=True))
pd.SetPolys(cells)

t0 = time.time()
c = vmtkscripts.vmtkCenterlines()
c.Surface = pd
c.SeedSelectorName = "pointlist"
c.SourcePoints = Z["source"].tolist()
c.TargetPoints = Z["targets"].ravel().tolist()
c.AppendEndPoints = 1
c.Resampling = 1
c.ResamplingStepLength = 0.3
c.Execute()
cl = c.Centerlines
t_vmtk = time.time() - t0
print("TIMING " + __import__("json").dumps({k: round(float(v), 4) for k, v in ({"vmtk_centerlines_subtree": t_vmtk}).items()}))
P = vtk_to_numpy(cl.GetPoints().GetData()).astype(float)
Rv = vtk_to_numpy(cl.GetPointData().GetArray("MaximumInscribedSphereRadius")).astype(float)
print(f"{run}: vmtkCenterlines {cl.GetNumberOfCells()} centerlines, {len(P)} points in {t_vmtk:.1f} s "
      f"(surface {len(Z['verts'])} vertices)")

ours, Ro = Z["ours_points"], Z["ours_radius"]
d, j = cKDTree(ours).query(P)
dr = Rv - Ro[j]
# ignore the first/last 1.5 mm of each vmtk line (seed snapping at the surface)
ends = np.zeros(len(P), bool)
for k in range(cl.GetNumberOfCells()):
    ids = [cl.GetCell(k).GetPointId(i) for i in range(cl.GetCell(k).GetNumberOfPoints())]
    q = P[ids]; s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
    ends[np.array(ids)[(s < 1.5) | (s > s[-1] - 1.5)]] = True
s = ~ends
print(f"  vmtk point -> our nearest centerline point: median {np.median(d[s]):.3f} mm, p90 {np.percentile(d[s], 90):.3f}, "
      f"p99 {np.percentile(d[s], 99):.3f}, max {d[s].max():.2f}")
print(f"  radius, vmtk MIS - ours: median {np.median(dr[s]):+.3f} mm, p10/p90 {np.percentile(dr[s], 10):+.3f} / "
      f"{np.percentile(dr[s], 90):+.3f}; |diff| median {np.median(np.abs(dr[s])):.3f}")
for lo, hi in ((0, 1.25), (1.25, 1.75), (1.75, 2.5), (2.5, 99)):
    k = s & (Ro[j] >= lo) & (Ro[j] < hi)
    if k.sum() > 20:
        print(f"  our r {lo:4.2f}-{hi:<5}: n {k.sum():5d}  gap median {np.median(d[k]):.3f} mm  "
              f"radius diff median {np.median(dr[k]):+.3f} mm")
# coverage: how much of OUR subtree lies within 1 mm of some vmtk line
back = cKDTree(P).query(ours)[0]
print(f"  our subtree length within 1 mm of a vmtk line: {(back <= 1.0).mean():.1%}")
# routes: for each target, vmtk's line against our own path from the same source
lines = []
for k in range(cl.GetNumberOfCells()):
    ids = [cl.GetCell(k).GetPointId(i) for i in range(cl.GetCell(k).GetNumberOfPoints())]
    lines.append(P[ids])
paths = np.split(Z["path_points"], np.cumsum(Z["path_len"])[:-1])
same, rows = 0, []
for t, o in zip(Z["targets"], paths):
    L = min(lines, key=lambda q: min(np.linalg.norm(q[-1] - t), np.linalg.norm(q[0] - t)))
    if min(np.linalg.norm(L[-1] - t), np.linalg.norm(L[0] - t)) > 2.0 or len(L) < 3:
        rows.append("unreached"); continue
    a_ = (cKDTree(L).query(o)[0] <= 1.0).mean(); b_ = (cKDTree(o).query(L)[0] <= 1.0).mean()
    lo_ = np.sum(np.linalg.norm(np.diff(o, axis=0), axis=1)); lv_ = np.sum(np.linalg.norm(np.diff(L, axis=0), axis=1))
    rows.append((a_, b_, lo_, lv_))
    same += min(a_, b_) >= 0.95
ok = [r for r in rows if r != "unreached"]
print(f"  routes: {len(Z['targets'])} targets, {len(ok)} reached by vmtk, {same} on the same route as ours "
      f"(>= 95 % of each within 1 mm of the other); path length ours/vmtk median "
      f"{np.median([r[2] / r[3] for r in ok]):.3f}")
for t, r in zip(range(len(rows)), rows):
    if r == "unreached" or min(r[0], r[1]) < 0.95:
        print(f"    target {t}: {r if r == 'unreached' else f'ours near vmtk {r[0]:.0%}, vmtk near ours {r[1]:.0%}, lengths {r[2]:.1f} / {r[3]:.1f} mm'}")
line = -np.ones(len(P), np.int64); order = []
for k in range(cl.GetNumberOfCells()):
    ids = [cl.GetCell(k).GetPointId(i) for i in range(cl.GetCell(k).GetNumberOfPoints())]
    order.append(np.array(ids))
np.savez(DATA / f"{run}_vmtk_output.npz", points=P, radius=Rv, gap=d, dr=dr, seconds=t_vmtk,
         line_ids=np.array([len(o) for o in order]), line_points=np.concatenate(order))
