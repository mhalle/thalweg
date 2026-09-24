"""vmtk's surface curvature (vmtkSurfaceCurvature, mean) on the field mesh and the phantom meshes.

    uv run --no-project --python 3.12 --with vmtk --with scipy python bench/vessels/vmtk_curvature.py [RUN]

After `curvature.py RUN phantoms` (and `curvature.py RUN repeat`, for the other reconstructions' meshes). Default settings (vtkCurvatures mean curvature, no absolute value,
no median filtering). Writes DATA/<RUN>_vmtk_curvature.npz for curvature.py.
"""
import os, sys
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
from vmtk import vmtkscripts

DATA = Path(os.environ.get("VESSELS_DATA", Path.home() / "tmp/data/vessels"))
run = sys.argv[1] if len(sys.argv) > 1 else "C3N-00704_ctpa0625"


def mean_curvature(verts, faces):
    s = vtk.vtkPolyData()
    p = vtk.vtkPoints(); p.SetData(numpy_to_vtk(verts.astype(np.float64), deep=True)); s.SetPoints(p)
    c = vtk.vtkCellArray()
    c.ImportLegacyFormat(numpy_to_vtkIdTypeArray(np.c_[np.full(len(faces), 3), faces].astype(np.int64).ravel(), deep=True))
    s.SetPolys(c)
    k = vmtkscripts.vmtkSurfaceCurvature(); k.Surface = s; k.CurvatureType = "mean"; k.Execute()
    return vtk_to_numpy(k.Surface.GetPointData().GetArray("Curvature")).copy()


Z = np.load(DATA / f"{run}_vmtk_input.npz")
PH = np.load(DATA / f"{run}_curv_phantoms.npz")
import time
_t = time.time()
out = {"real_mean": mean_curvature(Z["verts"], Z["faces"])}
t_real = time.time() - _t
for i in range(int(PH["n"])):
    out[f"p{i}_mean"] = mean_curvature(PH[f"p{i}_verts"], PH[f"p{i}_faces"])
for f in sorted(DATA.glob(f"{run}_curv_repeat_*.npz")):
    R = np.load(f); rep = f.stem.split("_curv_repeat_")[1]
    out[f"repeat_{rep}_mean"] = mean_curvature(R["verts"], R["faces"]); out[f"repeat_{rep}_verts"] = R["verts"]
np.savez(DATA / f"{run}_vmtk_curvature.npz", **out)
print(f"vmtk mean curvature on the field mesh ({len(out['real_mean'])} vertices) and {int(PH['n'])} phantoms")
print("TIMING " + __import__("json").dumps({k: round(float(v), 4) for k, v in ({"vmtk_curvature": t_real, "vmtk_curvature_vertices": len(Z["verts"])}).items()}))
