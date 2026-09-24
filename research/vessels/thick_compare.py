"""Same patient, thin vs thick: does the model's thin-vessel radius follow the slice thickness?

    python bench/vessels/thick_compare.py [PATIENT ...]

Centerline points come from the thinnest run: its field-mode centerline graph (REF=graph, the
default; run centerline.py first) or, as before, its labelmap skeleton (REF=skeleton), ridge-
refined on that run's own margin either way. Every variant is then measured AT THOSE WORLD POINTS: the model's inscribed
radius from its own margin crossings (searched within +-0.5 mm of the point, inside its own
margin), and its own CT's HU at the point. Same anatomy, same points, only the acquisition
changes. Writes DATA/<patient>_ladder.npz for the image-radius and centerline scripts.
"""
import json, os, sys, time
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from _data import DATA
from _field import fine_field, crossings, sample, inscribed_radius, Image
from cases import LADDERS, store, image

NAMES = {3: "arteries", 4: "veins"}
MARGIN = {3: "lung_arteries", 4: "lung_veins"}
REF = os.environ.get("REF", "graph")     # reference points: "graph" (field centerline graph) or "skeleton" (old)
BINS = [0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 99]


def tangents(pts, radius=2.0):
    tree = cKDTree(pts)
    out = np.zeros_like(pts)
    for i, nb in enumerate(tree.query_ball_point(pts, radius)):
        q = pts[nb] - pts[nb].mean(0)
        out[i] = np.linalg.svd(q, full_matrices=False)[2][0] if len(nb) >= 3 else (1, 0, 0)
    return out


for patient in (sys.argv[1:] or LADDERS):
    t0 = time.time()
    ladder = LADDERS[patient]
    ref_run = ladder[0][0]
    margins, labels, grid, clip = fine_field(store(ref_run))
    res = {}
    for c in (3, 4):
        m = margins[MARGIN[c]]
        if REF == "graph":
            # the field-mode centerline graph (centerline.py), every segment point, junctions once
            G = json.load(open(DATA / f"{ref_run}_{MARGIN[c]}_centerlines.json"))
            pts0 = np.unique(np.round(np.concatenate([np.array(s["points"]) for s in G["segments"]]), 3), axis=0)
        else:
            sk = np.argwhere(skeletonize(labels == c).astype(bool)).astype(float)
            pts0 = grid.to_world(sk)
        tree = cKDTree(crossings(m, grid))
        r, pts = inscribed_radius(pts0, tree, lambda q: sample(m, grid, q) > 0)
        keep = r > 0
        res[c] = dict(points=pts[keep], ref_r=r[keep], tangent=tangents(pts[keep]))
    print(f"== {patient}: reference {ref_run}, {len(res[3]['points'])} artery / {len(res[4]['points'])} vein centerline points "
          f"({time.time()-t0:.0f} s)", flush=True)

    for run, src, thk, note in ladder:
        img = Image(image(src))
        mg, _, gv, _ = (margins, labels, grid, clip) if run == ref_run else fine_field(store(run))
        for c in (3, 4):
            P = res[c]["points"]
            m = mg[MARGIN[c]]
            tree = cKDTree(crossings(m, gv))
            r, _ = inscribed_radius(P, tree, lambda q: sample(m, gv, q) > 0)
            res[c][f"r:{run}"] = r
            res[c][f"hu:{run}"] = img(P)
        print(f"   measured {run} ({thk} mm, {note}) ({time.time()-t0:.0f} s)", flush=True)

    for c in (3, 4):
        d = res[c]; ref = d["ref_r"]
        print(f"\n{patient} {NAMES[c]}: model radius at the reference points, median mm (share found inside)")
        head = f"{'ref r bin':>11} {'n':>6} " + " ".join(f"{t:>14}" for _, _, t, _ in ladder)
        print(head + "    center HU per variant")
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            s = (ref >= lo) & (ref < hi)
            if s.sum() < 30: continue
            cells, hus = [], []
            for run, _, thk, _ in ladder:
                r = d[f"r:{run}"][s]; ok = r > 0
                cells.append(f"{np.median(r[ok]) if ok.any() else np.nan:6.2f} ({ok.mean():4.0%})")
                hus.append(f"{np.median(d[f'hu:{run}'][s]):5.0f}")
            print(f"{lo:5.2f}-{hi:<5.2f} {s.sum():6d} " + " ".join(f"{x:>14}" for x in cells) + "   " + " ".join(hus))
    np.savez(DATA / f"{patient}_ladder.npz",
             **{f"{NAMES[c]}/{k}": v for c in (3, 4) for k, v in res[c].items()},
             runs=np.array([r for r, _, _, _ in ladder]), thickness=np.array([t for _, _, t, _ in ladder]))
    print(f"wrote {DATA / f'{patient}_ladder.npz'} ({time.time()-t0:.0f} s)\n", flush=True)
