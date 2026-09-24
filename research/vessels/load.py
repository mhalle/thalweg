"""Decode a lung_vessels ranked store once and cache what the analyses need.

    python bench/vessels/load.py CASE STORE CT
    python bench/vessels/load.py                      # the idc-torso1 demo

CT is a NIfTI file or a DICOM series directory. Writes DATA/<CASE>_cache.npz: labels (argmax),
margins for arteries / veins / their union, the stored distance layer in mm (if the store has
one), the CT resampled onto the store grid, and the grid geometry (LPS). Run with haversack's
venv: ../haversack/.venv/bin/python bench/vessels/load.py ...
"""
import dataclasses, sys, time
from pathlib import Path
import numpy as np
from _data import DATA
import rankfield as rf
import SimpleITK as sitk
from haversack.ranked_store import open_store
from haversack.ranked_restore import parts_of

DEMO = ("idc-torso1",
        Path.home() / "Dropbox/development/haversack/data/duckn_demo/idc-torso1/lung_vessels.duckn",
        Path.home() / "tmp/data/idc-torso1.nii")
CASE, STORE, CT = (sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])) if len(sys.argv) > 3 else DEMO
OUT = DATA / f"{CASE}_cache.npz"

t0 = time.time()
root = open_store(STORE, "r").root
segs = root.attrs.asdict()["duckn"]["extensions"]["seg"]["segments"]
parts = parts_of(root)
k = len(parts) - 1          # the fine stage: last in paint order (a cascade's crop stage comes first)

def _value(s):              # seg 0.7 wrote `label_value`; later stores write `label_values: [v]`
    v = s.get("label_value", s.get("label_values"))
    return v[0] if isinstance(v, list) else v

# leaves of THIS part's layer only: a cascade store names its crop stage's layer too (and stores
# written before haversack 0.13.0 name it wrongly - layer 0 of a lung_vessels store calls
# total's classes 1-4 lung_airways...lung_veins; fixed in haversack 48dddec)
value_of = {s["name"]: _value(s) for s in segs
            if _value(s) is not None and not s.get("members") and int(s.get("layer", 0)) == k}
code = parts[k].field
code = dataclasses.replace(code, ranks=np.asarray(code.ranks[...]),
                           support=np.asarray(code.support[...]),
                           tail=np.asarray(code.tail[...]) if code.tail is not None else None,
                           meta={**code.meta, "shape": list(code.support.shape[1:])})
ch = {n: code.labels.index(value_of[n]) for n in ("lung_arteries", "lung_veins")}
geo = code.geometry                                     # LPS, rows = array axes
origin, dirs = np.asarray(geo.origin, float), np.asarray(geo.directions, float)
spacing = np.linalg.norm(dirs, axis=1)
print(f"{CASE}: part {k} of {len(parts)} ({parts[k].name}), format {code.meta.get('version')}, "
      f"shape {code.support.shape[1:]}, spacing {spacing.round(4)}, channels {ch}")

deficits = np.stack([rf.deficit(code, c) for c in range(code.classes)])
chan = np.argmax(deficits, axis=0)
lab_value = np.asarray(code.labels)[chan].astype(np.uint8)
# analysis convention: 0 background, 1 airways, 2 airway wall, 3 arteries, 4 veins (by NAME)
canon = {"lung_airways": 1, "lung_airways_wall": 2, "lung_arteries": 3, "lung_veins": 4}
labels = np.zeros(lab_value.shape, np.uint8)
for n, v in canon.items():
    if n in value_of:
        labels[lab_value == value_of[n]] = v
del deficits
m_art = rf.margin(code, ch["lung_arteries"]).astype(np.float32)
m_vein = rf.margin(code, ch["lung_veins"]).astype(np.float32)
m_sel = rf.decode_groups(code, [[ch["lung_arteries"], ch["lung_veins"]]], device="cpu").numpy()[0].astype(np.float32)

meta = code.meta
g = root[f"parts/{k}"]
if "distance" in g and "distance_truncation" in meta:
    b = np.asarray(g["distance"])
    T = float(meta["distance_truncation"])
    dist_mm = np.where(b == 0, np.float32(np.inf), (1 - b / meta["distance_max"]) * T).astype(np.float32)
else:
    T, dist_mm = np.nan, np.full(labels.shape, np.inf, np.float32)
print("decode", round(time.time() - t0, 1), "s")

# CT onto the store grid: world point of index i = origin + sum_k i_k * dirs[k]
if CT.is_dir():
    r = sitk.ImageSeriesReader(); r.SetFileNames(r.GetGDCMSeriesFileNames(str(CT)))
    img = r.Execute()
else:
    img = sitk.ReadImage(str(CT))
ref = sitk.Image([int(s) for s in labels.shape[::-1]], sitk.sitkFloat32)   # sitk index = array axes reversed
ref.SetOrigin(tuple(origin))
ref.SetSpacing(tuple(spacing[::-1]))
ref.SetDirection(tuple((dirs[::-1] / spacing[::-1, None]).T.ravel()))
ct = sitk.GetArrayFromImage(sitk.Resample(img, ref, sitk.Transform(), sitk.sitkLinear, -1024.0))
print("CT on grid", ct.shape, "range", ct.min(), ct.max(), round(time.time() - t0, 1), "s")

np.savez(OUT, labels=labels, m_art=m_art, m_vein=m_vein, m_sel=m_sel, dist_mm=dist_mm,
         ct=np.clip(ct, -32768, 32767).astype(np.int16), origin=origin, dirs=dirs, spacing=spacing,
         clip=np.float32(meta["clip"]), T=np.float32(T))
print("wrote", OUT, round(time.time() - t0, 1), "s")
