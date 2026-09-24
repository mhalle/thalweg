"""vmtk's branch mapping on the field surface, from our centerlines: the reference for wall_map.py.

    uv run --no-project --python 3.12 --with vmtk --with scipy python bench/vessels/vmtk_mapping.py [RUN]

The standard chain: vmtkCenterlineAttributes (Abscissas, ParallelTransportNormals) ->
vmtkBranchExtractor -> vmtkBranchClipper (surface GroupIds) -> vmtkBifurcationReferenceSystems ->
vmtkCenterlineOffsetAttributes (abscissas and normals offset to the root bifurcation) ->
vmtkBranchMetrics (surface AbscissaMetric, AngularMetric) -> vmtkDistanceToCenterlines ->
vmtkBranchMapping (StretchedMapping) -> vmtkBranchPatching (a longitudinal x circular image of the
surface data). Saves the offset split centerlines, the labeled surface and the patched image.
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
R = "MaximumInscribedSphereRadius"
T = {}


def tick(name, t0):
    T[name] = time.time() - t0
    return time.time()


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
tree = cKDTree(Z["ours_points"])
paths = np.split(Z["path_points"], np.cumsum(Z["path_len"])[:-1])
lines = [resample(p) for p in paths if len(p) > 3]
cl = lines_polydata(lines, [Z["ours_radius"][tree.query(L)[1]] for L in lines])

t0 = time.time()
ca = vmtkscripts.vmtkCenterlineAttributes(); ca.Centerlines = cl; ca.Execute()
t0 = tick("attributes", t0)
be = vmtkscripts.vmtkBranchExtractor(); be.Centerlines = ca.Centerlines; be.RadiusArrayName = R; be.Execute()
split = be.Centerlines
t0 = tick("extract", t0)
bc = vmtkscripts.vmtkBranchClipper(); bc.Surface = surface; bc.Centerlines = split; bc.RadiusArrayName = R
bc.GroupIdsArrayName = "GroupIds"; bc.BlankingArrayName = "Blanking"; bc.Execute()
clipped = bc.Surface
t0 = tick("clip", t0)
rs = vmtkscripts.vmtkBifurcationReferenceSystems(); rs.Centerlines = split; rs.RadiusArrayName = R
rs.BlankingArrayName = "Blanking"; rs.GroupIdsArrayName = "GroupIds"; rs.Execute()
t0 = tick("frames", t0)
gid = vtk_to_numpy(split.GetCellData().GetArray("GroupIds")).astype(int)
root_group = int(gid[0])                                  # the first cell starts at the source
oa = vmtkscripts.vmtkCenterlineOffsetAttributes(); oa.Centerlines = split; oa.ReferenceSystems = rs.ReferenceSystems
oa.ReferenceGroupId = root_group; oa.AbscissasArrayName = "Abscissas"; oa.NormalsArrayName = "ParallelTransportNormals"
oa.GroupIdsArrayName = "GroupIds"; oa.CenterlineIdsArrayName = "CenterlineIds"; oa.ReplaceAttributes = 1; oa.Execute()
offset = oa.Centerlines
t0 = tick("offset", t0)
bm = vmtkscripts.vmtkBranchMetrics(); bm.Surface = clipped; bm.Centerlines = offset
bm.AbscissasArrayName = "Abscissas"; bm.NormalsArrayName = "ParallelTransportNormals"; bm.RadiusArrayName = R
bm.GroupIdsArrayName = "GroupIds"; bm.CenterlineIdsArrayName = "CenterlineIds"; bm.TractIdsArrayName = "TractIds"
bm.BlankingArrayName = "Blanking"; bm.Execute()
abscissa_before_mapping = vtk_to_numpy(bm.Surface.GetPointData().GetArray("AbscissaMetric")).copy()
t0 = tick("metrics", t0)
dc = vmtkscripts.vmtkDistanceToCenterlines(); dc.Surface = bm.Surface; dc.Centerlines = offset
dc.RadiusArrayName = R; dc.Execute()
t0 = tick("distance", t0)
bmap = vmtkscripts.vmtkBranchMapping(); bmap.Surface = dc.Surface; bmap.Centerlines = offset
bmap.ReferenceSystems = rs.ReferenceSystems; bmap.AbscissasArrayName = "Abscissas"
bmap.NormalsArrayName = "ParallelTransportNormals"; bmap.GroupIdsArrayName = "GroupIds"
bmap.CenterlineIdsArrayName = "CenterlineIds"; bmap.TractIdsArrayName = "TractIds"; bmap.RadiusArrayName = R
bmap.BlankingArrayName = "Blanking"; bmap.AngularMetricArrayName = "AngularMetric"
bmap.AbscissaMetricArrayName = "AbscissaMetric"; bmap.Execute()
t0 = tick("mapping", t0)
bp = vmtkscripts.vmtkBranchPatching(); bp.Surface = bmap.Surface; bp.PatchSize = [0.5, 1.0 / 36.0]
bp.LongitudinalMappingArrayName = "StretchedMapping"; bp.CircularMappingArrayName = "AngularMetric"
bp.GroupIdsArrayName = "GroupIds"; bp.Execute()
t0 = tick("patching", t0)

S = bmap.Surface
arr = lambda n: vtk_to_numpy(S.GetPointData().GetArray(n))
cd, pdat = offset.GetCellData(), offset.GetPointData()
cells_ = []
for k in range(offset.GetNumberOfCells()):
    c = offset.GetCell(k); cells_.append(np.array([c.GetPointId(i) for i in range(c.GetNumberOfPoints())]))
img = bp.PatchedData
dims = img.GetDimensions()
patched = {f"patch_{img.GetPointData().GetArrayName(i)}": vtk_to_numpy(img.GetPointData().GetArray(i))
           for i in range(img.GetPointData().GetNumberOfArrays())}
np.savez(DATA / f"{run}_vmtk_mapping.npz",
         cl_points=vtk_to_numpy(offset.GetPoints().GetData()), cl_radius=vtk_to_numpy(pdat.GetArray(R)),
         cl_abscissa=vtk_to_numpy(pdat.GetArray("Abscissas")), cl_normal=vtk_to_numpy(pdat.GetArray("ParallelTransportNormals")),
         cell_ids=np.concatenate(cells_), cell_len=np.array([len(c) for c in cells_]),
         cell_group=vtk_to_numpy(cd.GetArray("GroupIds")).astype(int), cell_blank=vtk_to_numpy(cd.GetArray("Blanking")).astype(int),
         cell_centerline=vtk_to_numpy(cd.GetArray("CenterlineIds")).astype(int),
         cell_tract=vtk_to_numpy(cd.GetArray("TractIds")).astype(int),
         surf_points=vtk_to_numpy(S.GetPoints().GetData()), surf_group=arr("GroupIds").astype(int),
         surf_abscissa=arr("AbscissaMetric"), surf_angle=arr("AngularMetric"), surf_dist=arr("DistanceToCenterlines"),
         surf_abscissa_metrics=abscissa_before_mapping, surf_stretched=arr("StretchedMapping"), patch_dims=np.array(dims),
         seconds=np.array([T[k] for k in ("attributes", "extract", "clip", "frames", "offset", "metrics", "distance", "mapping", "patching")]),
         **patched)
print("vmtk mapping chain (s): " + ", ".join(f"{k} {v:.1f}" for k, v in T.items()))
print(f"surface {S.GetNumberOfPoints()} points with AbscissaMetric/AngularMetric/DistanceToCenterlines/StretchedMapping; "
      f"patched image {dims} with arrays {list(patched)[:6]}")
print("TIMING " + __import__("json").dumps({k: round(float(v), 4) for k, v in ({f"vmtk_mapping_{k}": v for k, v in T.items()}).items()}))
