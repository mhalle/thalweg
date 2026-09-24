"""vmtk's flow extensions on the field mesh, cut at the outlets flow_ext.py chose.

    uv run --no-project --python 3.12 --with vmtk --with scipy python bench/vessels/vmtk_flowext.py [RUN]

After `flow_ext.py RUN prep`. The CFD workflow as vmtk users run it: open the closed surface at each
outlet (vtkClipPolyData with a plane restricted to a ball, the usual interactive clip made
scriptable), keep the connected piece, then vmtkFlowExtensions in boundary-normal mode with an
adaptive length (ExtensionRatio x the ring's mean radius) and the script's default transition
(TransitionRatio 0.25, thin-plate spline). Writes DATA/<RUN>_vmtk_flowext.npz.
"""
import os, sys, time
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
from vmtk import vmtkscripts, vtkvmtk

DATA = Path(os.environ.get("VESSELS_DATA", Path.home() / "tmp/data/vessels"))
run = sys.argv[1] if len(sys.argv) > 1 else "C3N-00704_ctpa0625"
Z = np.load(DATA / f"{run}_vmtk_input.npz")
O = np.load(DATA / f"{run}_flowext_outlets.npz")
RATIO, TRANSITION = float(O["ratio"]), float(O["transition"])


def polydata(verts, faces):
    s = vtk.vtkPolyData()
    p = vtk.vtkPoints(); p.SetData(numpy_to_vtk(verts.astype(np.float64), deep=True)); s.SetPoints(p)
    c = vtk.vtkCellArray()
    c.ImportLegacyFormat(numpy_to_vtkIdTypeArray(np.c_[np.full(len(faces), 3), faces].astype(np.int64).ravel(), deep=True))
    s.SetPolys(c)
    return s


def arrays(s):
    tri = vtk.vtkTriangleFilter(); tri.SetInputData(s); tri.Update(); s = tri.GetOutput()
    F = vtk_to_numpy(s.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    return vtk_to_numpy(s.GetPoints().GetData()).copy(), F.copy()


t0 = time.time()
keep = vtk.vtkImplicitBoolean(); keep.SetOperationTypeToUnion()          # min over outlets of G_k
for c, n, rho in zip(O["center"], O["normal"], O["ball"]):
    plane = vtk.vtkPlane(); plane.SetOrigin(*c); plane.SetNormal(*(-n))     # -(distance past the cut)
    ball = vtk.vtkSphere(); ball.SetCenter(*c); ball.SetRadius(rho)         # |x-c|^2 - rho^2
    g = vtk.vtkImplicitBoolean(); g.SetOperationTypeToIntersection()        # G_k = max: < 0 only past the cut, in the ball
    g.AddFunction(plane); g.AddFunction(ball)
    keep.AddFunction(g)
clipper = vtk.vtkClipPolyData(); clipper.SetInputData(polydata(Z["verts"], Z["faces"])); clipper.SetClipFunction(keep)
clipper.SetValue(0.0); clipper.Update()
conn = vtk.vtkConnectivityFilter(); conn.SetInputConnection(clipper.GetOutputPort()); conn.SetExtractionModeToLargestRegion()
geo = vtk.vtkGeometryFilter(); geo.SetInputConnection(conn.GetOutputPort())
clean = vtk.vtkCleanPolyData(); clean.SetInputConnection(geo.GetOutputPort()); clean.Update()
opened = clean.GetOutput()
t_clip = time.time() - t0

be = vtkvmtk.vtkvmtkPolyDataBoundaryExtractor(); be.SetInputData(opened); be.Update()
rings = be.GetOutput()
bary, mrad = [], []
for i in range(rings.GetNumberOfCells()):
    pts = vtk_to_numpy(rings.GetCell(i).GetPoints().GetData())
    b = pts.mean(0); bary.append(b); mrad.append(np.linalg.norm(pts - b, axis=1).mean())

t0 = time.time()
fe = vmtkscripts.vmtkFlowExtensions(); fe.Surface = opened; fe.Interactive = 0
fe.ExtensionMode = "boundarynormal"; fe.AdaptiveExtensionLength = 1; fe.ExtensionRatio = RATIO
fe.TransitionRatio = TRANSITION; fe.Execute()
t_ext = time.time() - t0

cv, cf = arrays(opened); ev, ef = arrays(fe.Surface)
np.savez(DATA / f"{run}_vmtk_flowext.npz", opened_verts=cv, opened_faces=cf, verts=ev, faces=ef,
         ring_barycenter=np.array(bary), ring_mean_radius=np.array(mrad), seconds=np.array([t_clip, t_ext]))
print(f"vmtk: opened {len(O['center'])} outlets -> {rings.GetNumberOfCells()} boundary rings "
      f"({t_clip:.1f} s); flow extensions ({t_ext:.1f} s): {len(cv)} -> {len(ev)} vertices")
print("TIMING " + __import__("json").dumps({k: round(float(v), 4) for k, v in ({"vmtk_flowext_clip": t_clip, "vmtk_flowext_extend": t_ext}).items()}))
