"""vmtk's branch splitting, surface clipping and bifurcation frames, on two centerline sets.

    uv run --no-project --python 3.12 --with vmtk --with scipy python bench/vessels/vmtk_branch.py [RUN]

After vmtk_prep.py + vmtk_run.py. The same subtree and the same field surface, two centerline
sets: "vmtk" (vmtkCenterlines on the surface) and "ours" (the field graph's source-to-tip paths
in vmtk's convention: one polyline per target, source to target, resampled to 0.3 mm, radius in
MaximumInscribedSphereRadius). Each goes through vmtkBranchExtractor (GroupIds, Blanking,
CenterlineIds, TractIds), vmtkBranchClipper (the surface labeled by group) and
vmtkBifurcationReferenceSystems (origin, Normal, UpNormal per bifurcation). Results go to
DATA/<RUN>_vmtk_branch_<set>.npz for branch_partition.py, which does the same without a surface.
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
V = np.load(DATA / f"{run}_vmtk_output.npz")
R = "MaximumInscribedSphereRadius"


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


def resample(p, step=0.3):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    t = np.arange(0, s[-1] + 1e-9, step)
    return np.stack([np.interp(t, s, p[:, a]) for a in range(3)], 1)


surface = vtk.vtkPolyData()
sp = vtk.vtkPoints(); sp.SetData(numpy_to_vtk(Z["verts"].astype(np.float64), deep=True)); surface.SetPoints(sp)
F = Z["faces"].astype(np.int64); cells = vtk.vtkCellArray()
cells.ImportLegacyFormat(numpy_to_vtkIdTypeArray(np.c_[np.full(len(F), 3), F].ravel(), deep=True))
surface.SetPolys(cells)

# vmtk's own centerlines, by cell
P = V["points"]; Rv = V["radius"]
order = np.split(V["line_points"], np.cumsum(V["line_ids"])[:-1])
vm_lines = [P[o] for o in order if len(o) > 2]
vm_radii = [Rv[o] for o in order if len(o) > 2]
# ours: source -> tip paths, radius from the densified graph points
ours_tree = cKDTree(Z["ours_points"])
paths = np.split(Z["path_points"], np.cumsum(Z["path_len"])[:-1])
our_lines = [resample(p) for p in paths if len(p) > 3]
our_radii = [Z["ours_radius"][ours_tree.query(L)[1]] for L in our_lines]

for name, lines, radii in (("vmtk", vm_lines, vm_radii), ("ours", our_lines, our_radii)):
    t0 = time.time()
    cl = lines_polydata(lines, radii)
    be = vmtkscripts.vmtkBranchExtractor(); be.Centerlines = cl; be.RadiusArrayName = R; be.Execute()
    split = be.Centerlines
    t1 = time.time()
    bc = vmtkscripts.vmtkBranchClipper(); bc.Surface = surface; bc.Centerlines = split
    bc.RadiusArrayName = R; bc.GroupIdsArrayName = "GroupIds"; bc.BlankingArrayName = "Blanking"
    bc.Execute()
    clipped = bc.Surface
    t2 = time.time()
    rs = vmtkscripts.vmtkBifurcationReferenceSystems(); rs.Centerlines = split; rs.RadiusArrayName = R
    rs.BlankingArrayName = "Blanking"; rs.GroupIdsArrayName = "GroupIds"; rs.Execute()
    ref = rs.ReferenceSystems
    t3 = time.time()
    # split centerlines: per cell, its points, radius, GroupIds, Blanking
    cd = split.GetCellData()
    gid = vtk_to_numpy(cd.GetArray("GroupIds")).astype(int)
    blank = vtk_to_numpy(cd.GetArray("Blanking")).astype(int)
    cid = vtk_to_numpy(cd.GetArray("CenterlineIds")).astype(int)
    SP = vtk_to_numpy(split.GetPoints().GetData()); SR = vtk_to_numpy(split.GetPointData().GetArray(R))
    cell_pts, cell_len = [], []
    for k in range(split.GetNumberOfCells()):
        c = split.GetCell(k)
        ids = [c.GetPointId(i) for i in range(c.GetNumberOfPoints())]
        cell_pts.append(np.array(ids)); cell_len.append(len(ids))
    out = dict(split_points=SP, split_radius=SR, cell_ids=np.concatenate(cell_pts), cell_len=np.array(cell_len),
               cell_group=gid, cell_blank=blank, cell_centerline=cid,
               clip_points=vtk_to_numpy(clipped.GetPoints().GetData()),
               clip_group=vtk_to_numpy(clipped.GetPointData().GetArray("GroupIds")).astype(int),
               ref_origin=vtk_to_numpy(ref.GetPoints().GetData()) if ref.GetNumberOfPoints() else np.zeros((0, 3)),
               ref_normal=vtk_to_numpy(ref.GetPointData().GetArray("Normal")) if ref.GetNumberOfPoints() else np.zeros((0, 3)),
               ref_upnormal=vtk_to_numpy(ref.GetPointData().GetArray("UpNormal")) if ref.GetNumberOfPoints() else np.zeros((0, 3)),
               ref_group=vtk_to_numpy(ref.GetPointData().GetArray("GroupIds")).astype(int) if ref.GetNumberOfPoints() else np.zeros(0, int),
               seconds=np.array([t1 - t0, t2 - t1, t3 - t2]))
    np.savez(DATA / f"{run}_vmtk_branch_{name}.npz", **out)
    groups = np.unique(gid[blank == 0]); bifs = np.unique(gid[blank == 1])
    print(f"{name}: {len(lines)} centerlines -> {len(groups)} branch groups + {len(bifs)} bifurcation groups; "
          f"{ref.GetNumberOfPoints()} reference systems; clipped surface {clipped.GetNumberOfPoints()} points; "
          f"times: extract {t1 - t0:.1f} s, clip {t2 - t1:.1f} s, frames {t3 - t2:.1f} s")
    print("TIMING " + __import__("json").dumps({k: round(float(v), 4) for k, v in ({f"vmtk_branch_extract_{name}": t1 - t0, f"vmtk_branch_clip_{name}": t2 - t1, f"vmtk_frames_{name}": t3 - t2}).items()}))
