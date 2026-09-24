"""How much of each centerline graph runs outside its vessel (margin <= 0), field vs voxel graph.

    python bench/vessels/leave_field.py RUN CLASS
Samples every segment polyline at 0.05 mm on the interpolated margin; a polyline that leaves the
vessel has stepped through a connection the field does not make (or cut a corner between points).
"""
import json, sys
import numpy as np
from _data import DATA
from _field import fine_field, sample
from cases import store

run, cls = sys.argv[1], sys.argv[2]
margins, labels, grid, clip = fine_field(store(run)); m = margins[cls]
def polylines(G, key):
    """Segment polylines from the graph (refined points), or the branches' lattice paths."""
    if key == "points":
        return [np.array(s["points"]) for s in G["segments"]]
    return [np.array(b["lattice_points"]) for b in G["branches"] if "lattice_points" in b]


for mode, suffix, key in (("field", "", "points"), ("field", "", "lattice"), ("voxel", "_voxel", "points"),
                          ("voxel", "_voxel", "lattice")):
    G = json.load(open(DATA / f"{run}_{cls}_centerlines{suffix}.json"))
    lines = polylines(G, key)
    if not lines:
        continue
    n_bad, out_len, tot_len, worst = 0, 0.0, 0.0, 0.0
    for p in lines:
        if len(p) < 2: continue
        L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        t = np.arange(0, L[-1] + 1e-9, 0.05)
        q = np.stack([np.interp(t, L, p[:, a]) for a in range(3)], 1)
        v = sample(m, grid, q)
        out = v <= 0
        tot_len += L[-1]; out_len += out.sum() * 0.05
        run_ = np.max(np.diff(np.flatnonzero(np.r_[True, ~out, True])) - 1) * 0.05 if out.any() else 0
        if run_ >= 0.2: n_bad += 1
        worst = max(worst, run_)
    print(f"{run} {cls} [{mode} graph, {'refined' if key == 'points' else 'lattice path'}]: {n_bad} of {len(lines)} polylines leave the vessel for >= 0.2 mm; "
          f"{out_len:.1f} mm of {tot_len/10:.0f} cm outside; longest excursion {worst:.2f} mm")
