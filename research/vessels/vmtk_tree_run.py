"""vmtkCenterlines on the whole arterial tree (all tips), timed: the same scope as centerline.py.

    uv run --no-project --python 3.12 --with vmtk --with scipy python bench/vessels/vmtk_tree_run.py [RUN]
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
from vmtk import vmtkscripts

DATA = Path(os.environ.get("VESSELS_DATA", Path.home() / "tmp/data/vessels"))
run = sys.argv[1] if len(sys.argv) > 1 else "C3N-00704_ctpa0625"
Z = np.load(DATA / f"{run}_vmtk_tree_input.npz")
pd = vtk.vtkPolyData()
p = vtk.vtkPoints(); p.SetData(numpy_to_vtk(Z["verts"].astype(np.float64), deep=True)); pd.SetPoints(p)
c = vtk.vtkCellArray()
c.ImportLegacyFormat(numpy_to_vtkIdTypeArray(np.c_[np.full(len(Z["faces"]), 3), Z["faces"]].astype(np.int64).ravel(), deep=True))
pd.SetPolys(c)
t0 = time.time()
cl = vmtkscripts.vmtkCenterlines(); cl.Surface = pd; cl.SeedSelectorName = "pointlist"
cl.SourcePoints = Z["source"].tolist(); cl.TargetPoints = Z["targets"].ravel().tolist()
cl.AppendEndPoints = 1; cl.Resampling = 1; cl.ResamplingStepLength = 0.3; cl.Execute()
t = time.time() - t0
print(f"vmtkCenterlines, whole tree: {cl.Centerlines.GetNumberOfCells()} centerlines for {len(Z['targets'])} targets "
      f"on {len(Z['verts'])} vertices in {t:.1f} s")
print("TIMING " + json.dumps({"vmtk_centerlines_tree": round(t, 4), "vmtk_tree_centerlines_out": cl.Centerlines.GetNumberOfCells()}))
