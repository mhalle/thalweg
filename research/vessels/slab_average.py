"""Thick-slice series synthesized from a thin one: an ideal slab average.

    python bench/vessels/slab_average.py DICOM_DIR OUT.nii.gz [THICKNESS_MM] [INCREMENT_MM]

Each thin slice stands for its own interval (position +- spacing/2). An output slice at z
averages the thin intervals over [z - T/2, z + T/2], weighted by overlap: a boxcar slice
sensitivity profile of width T, sampled every INCREMENT mm (default 2.0 mm at 1.0 mm, the
demo's acquisition). In-plane pixels are untouched. The added through-slice blur is exactly
a boxcar of width T (variance T^2/12), which is what makes this a calibration target for a
PSF estimator. A scanner's real slice profile is not a boxcar; this is the controlled case.
"""
import sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk

src, out = Path(sys.argv[1]), Path(sys.argv[2])
T = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
inc = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

r = sitk.ImageSeriesReader(); r.SetFileNames(r.GetGDCMSeriesFileNames(str(src)))
img = r.Execute()
a = sitk.GetArrayFromImage(img).astype(np.float32)          # (z, y, x) in sitk index order
sp = np.array(img.GetSpacing())                             # (x, y, z)
d = np.array(img.GetDirection()).reshape(3, 3)             # columns = index axes in LPS
dz = sp[2]
nz = a.shape[0]
zc = np.arange(nz) * dz                                     # thin slice centers, along the z index axis
lo_all, hi_all = zc - dz / 2, zc + dz / 2
first, last = zc[0] - dz / 2 + T / 2, zc[-1] + dz / 2 - T / 2    # slabs fully inside the data
zo = np.arange(first, last + 1e-6, inc)
W = np.clip(np.minimum(hi_all[None], (zo + T / 2)[:, None]) - np.maximum(lo_all[None], (zo - T / 2)[:, None]), 0, None)
W /= W.sum(1, keepdims=True)
b = np.einsum("oz,zyx->oyx", W.astype(np.float32), a)
o = sitk.GetImageFromArray(b)
o.SetSpacing((sp[0], sp[1], inc))
o.SetDirection(img.GetDirection())
o.SetOrigin(tuple(np.array(img.GetOrigin()) + d[:, 2] * zo[0]))
sitk.WriteImage(o, str(out), useCompression=True)
print(f"{src.name[:24]}...: {nz} x {dz:.3f} mm -> {len(zo)} x {inc} mm slabs of {T} mm; wrote {out}")
