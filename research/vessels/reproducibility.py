"""Field vs voxels on precision: the same anatomy, several runs, the same world points.

    python bench/vessels/reproducibility.py [PATIENT ...]         (after thick_compare.py)

The ladder's runs up to 2 mm do not change the model's thin-vessel radius on average (§12.2),
and each run's model grid sits at a different sub-voxel offset. So per-point differences
between runs measure an estimator's noise. Two estimators, the same +-0.5 mm search at each
reference centerline point:
- field: the largest distance to the margin's sub-voxel zero crossings, inside the
  interpolated margin (thick_compare.py's numbers);
- voxels: the largest labelmap EDT (anisotropic, mm) among voxels within that search.
Reported per radius bin: mean |run - reference| and RMS, in mm. Not the MAD: voxel EDT values
are quantized, so most voxel differences are exactly 0 and the MAD reads ~0 while the rest jump by
a whole quantum. Lower = more repeatable.
"""
import sys
import numpy as np
from scipy import ndimage as ndi
from _data import DATA
from _field import fine_field
from cases import LADDERS, store

BINS = [0, 1.25, 1.5, 1.75, 2.0, 2.5, 4.0, 99]
g = np.linspace(-0.5, 0.5, 5)
OFFS = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)


def edt_radius(run, cls, P):
    _, labels, grid, _ = fine_field(store(run))
    mask = labels == cls
    edt = ndi.distance_transform_edt(mask, sampling=grid.spacing)
    best = np.full(len(P), -1.0)
    for o in OFFS:
        idx = np.rint(grid.to_index(P + o)).astype(int)
        ok = np.all((idx >= 0) & (idx < mask.shape), 1)
        v = np.full(len(P), -1.0)
        v[ok] = edt[tuple(idx[ok].T)]
        best = np.maximum(best, v)
    return best


for patient in (sys.argv[1:] or ["C3N-00704", "MSB-02664"]):
    L = np.load(DATA / f"{patient}_ladder.npz")
    runs = [r for r, _, t, _ in LADDERS[patient] if t <= 2.0]
    ref = runs[0]
    for cname, cls in (("arteries", 3), ("veins", 4)):
        P = L[f"{cname}/points"]; rr = L[f"{cname}/ref_r"]
        field = {r: L[f"{cname}/r:{r}"] for r in runs}
        vox = {r: edt_radius(r, cls, P) for r in runs}
        print(f"\n{patient} {cname}: |run - {ref}| at the same points, mean / RMS (mm), field vs voxels")
        print(f"{'ref r bin':>11} {'n':>6} " + "".join(f"{r.split('_', 1)[1] + ': field':>22s}{'voxels':>14s}" for r in runs[1:]))
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            s_ = (rr >= lo) & (rr < hi)
            if s_.sum() < 50: continue
            cells = []
            for r in runs[1:]:
                ok = s_ & (field[r] > 0) & (field[ref] > 0) & (vox[r] > 0) & (vox[ref] > 0)
                for est in (field, vox):
                    d = est[r][ok] - est[ref][ok]
                    cells.append(f"{np.mean(np.abs(d)):.3f} / {np.sqrt(np.mean(d * d)):.3f}")
            print(f"{lo:5.2f}-{hi:<5.2f} {s_.sum():6d} " + "".join(f"{c:>22s}" if i % 2 == 0 else f"{c:>14s}" for i, c in enumerate(cells)))
