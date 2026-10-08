"""Aortic dissection from a dual-lumen field: flap, entry tears, branch feed.

    cd explorations/dissection && ../../../haversack/.venv/bin/python dissection.py

Builds the phantom of ``_dissect.py``, encodes it as a rankfield, and asks the four
questions a dissection study actually needs:

1. **Topology** - do the two lumens form one domain (a tear) or two (an intact flap)?
2. **Flap** - where is the true|false interface in each cross-section?
3. **Entry tears** - where do the lumens communicate through a hole in the flap?
4. **Branch feed** - which lumen supplies each branch (true / false / both)?

Output: ``~/tmp/data/vessels/<run>_dissection.json`` and a console summary. The
ground-truth tear and branch positions are in the JSON so the detector can be scored.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "vessels"))
from _data import DATA  # noqa: E402

from _dissect import (  # noqa: E402
    FLAP_T, GRADIENT, R_OUT, SPACING,
    centerline, encode_field, grid_shape, margins_from_labels, phantom_labels,
)

RUN = sys.argv[1] if len(sys.argv) > 1 else "phantom_arch"


def tick(name, t0, T):
    T[name] = time.time() - t0
    return time.time()


# --- 1. topology ---------------------------------------------------------------

def lumen_topology(labels):
    """Connected components of the lumen union and whether true and false share one.

    Face connectivity (the project's rule for field topology on the native grid) would
    split a tear that is only corner-connected; use 26-connectivity for the union so a
    tear reads as one domain, and report both.
    """
    lum = labels > 0
    out = {}
    for conn, st in ((6, ndi.generate_binary_structure(3, 1)),
                     (26, ndi.generate_binary_structure(3, 3))):
        lab, n = ndi.label(lum, structure=st)
        t_ids = set(np.unique(lab[labels == 1])) - {0}
        f_ids = set(np.unique(lab[labels == 2])) - {0}
        shared = t_ids & f_ids
        out[f"c{conn}"] = {
            "n_components": int(n),
            "true_components": sorted(int(i) for i in t_ids),
            "false_components": sorted(int(i) for i in f_ids),
            "shared_components": sorted(int(i) for i in shared),
            "one_domain": bool(shared),
        }
    return out


# --- 2. flap position in cross-sections ----------------------------------------

def sample_labels(labels, origin_xyz, pts):
    """Nearest-voxel labels at world points."""
    h = SPACING
    origin = np.asarray(origin_xyz, float)
    idx = np.round((pts - origin) / h).astype(int)
    sh = labels.shape
    ok = ((idx[:, 0] >= 0) & (idx[:, 0] < sh[2]) &
          (idx[:, 1] >= 0) & (idx[:, 1] < sh[1]) &
          (idx[:, 2] >= 0) & (idx[:, 2] < sh[0]))
    out = np.zeros(len(pts), np.uint8)
    out[ok] = labels[idx[ok, 2], idx[ok, 1], idx[ok, 0]]   # (z, y, x)
    return out


def measure_flap(labels, origin_xyz, s_cl, P, N, B, step=2.0):
    """A_true(s), A_false(s), flap v-extent: the 1-D dissection signal of design note §5.6."""
    h = SPACING
    area = h * h
    stations = np.arange(8.0, s_cl[-1] - 8.0, step)
    out = []
    for s0 in stations:
        i0 = int(np.argmin(np.abs(s_cl - s0)))
        th = np.linspace(0, 2 * np.pi, 240, endpoint=False)
        rr = np.linspace(0.0, R_OUT, 20)
        R, TH = np.meshgrid(rr, th, indexing="ij")
        u = (R * np.cos(TH)).ravel()
        v = (R * np.sin(TH)).ravel()
        pts = P[i0] + u[:, None] * N[i0] + v[:, None] * B[i0]
        lab = sample_labels(labels, origin_xyz, pts)
        # polar area element ~ r dr dtheta
        w = (R.ravel()) * (2 * np.pi / len(th)) * (R_OUT / 19.0)
        a_t = float(w[lab == 1].sum())
        a_f = float(w[lab == 2].sum())
        # flap: the contact between 1 and 2 along the section; report v of those samples
        contact = (lab == 1) | (lab == 2)
        v_c = v[contact]
        out.append({"s": float(s0), "a_true": a_t, "a_false": a_f,
                    "a_total": a_t + a_f,
                    "v_min": float(v_c.min()) if len(v_c) else 0.0,
                    "v_max": float(v_c.max()) if len(v_c) else 0.0,
                    "flap_v_median": float(np.median(v_c[np.abs(v_c) < FLAP_T])) if len(v_c) else 0.0})
    return out


# --- 3. entry tears -------------------------------------------------------------

def find_tears(labels, origin_xyz, s_cl, P, N, B, gt_tears, gt_branches=None,
               merge_mm=8.0):
    """Communications between true and false: contact patches, clustered and merged.

    An intact flap keeps the two labels apart (wall between them). A tear is a place
    where they touch - a missing wall. Detected as 6-connected T|F face contacts,
    clustered, then clusters closer than ``merge_mm`` are merged (one tear can fragment
    across the raster). Each merged cluster is scored against the ground-truth tears and
    tagged as a tear or a branch-straddle contact.
    """
    t = labels == 1
    f = labels == 2
    contact = t & ndi.binary_dilation(f, structure=ndi.generate_binary_structure(3, 1))
    flip = False
    if not contact.any():
        contact = f & ndi.binary_dilation(t, structure=ndi.generate_binary_structure(3, 1))
        flip = True
    lab, n = ndi.label(contact, structure=ndi.generate_binary_structure(3, 3))
    h = SPACING
    origin = np.asarray(origin_xyz, float)
    blobs = []
    for k in range(1, n + 1):
        m = lab == k
        if m.sum() < 3:
            continue
        idx = np.argwhere(m)
        xyz = np.stack([origin[0] + idx[:, 2] * h,
                        origin[1] + idx[:, 1] * h,
                        origin[2] + idx[:, 0] * h], 1)
        blobs.append({"n": int(m.sum()), "xyz": xyz})
    # merge blobs whose centroids are within merge_mm (single-linkage)
    merged = []
    used = [False] * len(blobs)
    for i, b in enumerate(blobs):
        if used[i]:
            continue
        used[i] = True
        stack = [b]
        xyz = b["xyz"]
        n_vox = b["n"]
        j = i + 1
        while j < len(blobs):
            if not used[j]:
                c0, c1 = xyz.mean(0), blobs[j]["xyz"].mean(0)
                if np.linalg.norm(c0 - c1) < merge_mm:
                    used[j] = True
                    xyz = np.concatenate([xyz, blobs[j]["xyz"]])
                    n_vox += blobs[j]["n"]
            j += 1
        merged.append({"n": n_vox, "xyz": xyz})

    tears = []
    for b in merged:
        c = b["xyz"].mean(0)
        best, best_d = None, 1e9
        for gt in gt_tears:
            i0 = int(np.argmin(np.abs(s_cl - gt["s"])))
            wc = P[i0] + gt["u"] * N[i0]
            d = float(np.linalg.norm(c - wc))
            if d < best_d:
                best, best_d = gt, d
        # is this contact sitting on a branch stub? (straddling "both" branches)
        on_branch = None
        if gt_branches:
            bd = 1e9
            for br in gt_branches:
                base = np.asarray(br["base"], float)
                tip = np.asarray(br["tip"], float)
                ab = tip - base
                tpar = float(np.clip(np.dot(c - base, ab) / max(ab @ ab, 1e-9), 0, 1))
                closest = base + tpar * ab
                dd = float(np.linalg.norm(c - closest))
                if dd < bd:
                    bd, on_branch = dd, br["name"]
            on_branch = on_branch if bd < 6.0 else None
        tears.append({
            "n_voxels": int(b["n"]),
            "centroid_xyz": [round(float(x), 2) for x in c],
            "extent_mm": [round(float(x), 2) for x in (b["xyz"].max(0) - b["xyz"].min(0))],
            "on_branch": on_branch,
            "gt_match": None if best is None else {
                "s": best["s"], "u": best["u"], "r": best["r"],
                "distance_mm": round(best_d, 2),
                "hit": bool(best_d < best["r"] + 3.0),
            },
        })
    tears.sort(key=lambda r: -r["n_voxels"])
    return tears, int(contact.sum()), bool(flip)


# --- 4. branch feed -------------------------------------------------------------

def branch_feed(labels, origin_xyz, margins, gt_branches):
    """At each branch ostium, which lumen's margin wins (true / false / both).

    Samples a ball just inside the aorta at the ostium, and also the first few mm of
    the stub itself - a straddling ("both") branch is split by the flap, so the ostium
    ball alone can look single-sided depending on where it is centered.
    """
    out = []
    m_t, m_f = margins[1], margins[2]
    h = SPACING
    origin = np.asarray(origin_xyz, float)
    rng = np.random.default_rng(0)

    def grab(pts):
        idx = np.round((pts - origin) / h).astype(int)
        sh = labels.shape
        ok = ((idx[:, 0] >= 0) & (idx[:, 0] < sh[2]) &
              (idx[:, 1] >= 0) & (idx[:, 1] < sh[1]) &
              (idx[:, 2] >= 0) & (idx[:, 2] < sh[0]))
        if not ok.any():
            return None
        ii = idx[ok]
        return (labels[ii[:, 2], ii[:, 1], ii[:, 0]],
                m_t[ii[:, 2], ii[:, 1], ii[:, 0]],
                m_f[ii[:, 2], ii[:, 1], ii[:, 0]])

    for b in gt_branches:
        base = np.asarray(b["base"], float)
        tip = np.asarray(b["tip"], float)
        axis = np.asarray(b["axis"], float)
        samples = [
            base - axis * 1.5 + rng.normal(scale=1.0, size=(250, 3)),
            base + axis * 2.0 + rng.normal(scale=b["r"] * 0.45, size=(250, 3)),
            base + axis * 5.0 + rng.normal(scale=b["r"] * 0.45, size=(250, 3)),
        ]
        labs, mts, mfs = [], [], []
        for pts in samples:
            g = grab(pts)
            if g is None:
                continue
            labs.append(g[0]); mts.append(g[1]); mfs.append(g[2])
        if not labs:
            out.append({"name": b["name"], "error": "ostium outside grid"})
            continue
        lab = np.concatenate(labs)
        mt = np.concatenate(mts)
        mf = np.concatenate(mfs)
        frac_t = float(np.mean(lab == 1))
        frac_f = float(np.mean(lab == 2))
        open_t = float(np.mean(mt > 0))
        open_f = float(np.mean(mf > 0))
        if open_t > 0.25 and open_f > 0.25:
            pred = "both"
        elif frac_t > 0.7 and frac_f < 0.15:
            pred = "true"
        elif frac_f > 0.7 and frac_t < 0.15:
            pred = "false"
        else:
            pred = "both"
        out.append({
            "name": b["name"],
            "gt_feed": b["feed"],
            "pred_feed": pred,
            "match": pred == b["feed"],
            "frac_true": round(frac_t, 3),
            "frac_false": round(frac_f, 3),
            "open_true": round(open_t, 3),
            "open_false": round(open_f, 3),
            "margin_true": round(float(mt.mean()), 3),
            "margin_false": round(float(mf.mean()), 3),
        })
    return out


# --- field-quantitative extras ---------------------------------------------------

def tie_sheet_stats(margins, labels):
    """The true|false tie sheet: where ``m_t - m_f`` changes sign inside the lumen.

    The zero set of the margin difference is the field's flap position - sub-voxel once
    you interpolate, and the only place a surface needs to exist. Where an intact wall
    separates the lumens the two fields do not meet; at a tear they do, and the zero set
    becomes a real surface (the missing flap).
    """
    m_t, m_f = margins[1], margins[2]
    lum = labels > 0
    diff = (m_t - m_f).astype(np.float64)
    # sign changes along each axis, restricted to the lumen
    changes = np.zeros(labels.shape, bool)
    for ax in range(3):
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[ax] = slice(0, -1)
        sl1[ax] = slice(1, None)
        a, b = diff[tuple(sl0)], diff[tuple(sl1)]
        flip = ((a > 0) != (b > 0)) & lum[tuple(sl0)] & lum[tuple(sl1)]
        for sl, which in ((sl0, 0), (sl1, 1)):
            changes[tuple(sl)] |= flip
    side_ok = float(np.mean((diff[lum] > 0) == (labels[lum] == 1))) if lum.any() else 0.0
    return {"sheet_voxels": int(changes.sum()),
            "sheet_fraction": round(float(changes.sum() / max(lum.sum(), 1)), 4),
            "side_agreement": round(side_ok, 4),
            "diff_on_true": round(float(diff[labels == 1].mean()), 3),
            "diff_on_false": round(float(diff[labels == 2].mean()), 3)}


def main():
    T = {}
    t0 = time.time()
    s_cl, P, Tvec, N, B = centerline()
    shape, origin = grid_shape(P)
    print(f"phantom grid {shape} @ {SPACING} mm, origin_xyz {origin.round(2)}")
    labels, gt, _ = phantom_labels(shape, origin)
    n_t, n_f = int((labels == 1).sum()), int((labels == 2).sum())
    print(f"labels: true {n_t}  false {n_f}  wall/other {int((labels == 0).sum())}")
    t0 = tick("phantom", t0, T)

    logits = margins_from_labels(labels)
    code, margins = encode_field(logits)
    print(f"field: {code.classes} classes, depth {code.meta.get('depth')}, clip {code.meta.get('clip')}")
    t0 = tick("encode", t0, T)

    topo = lumen_topology(labels)
    print(f"topology 6-conn: one_domain={topo['c6']['one_domain']} "
          f"components={topo['c6']['n_components']} "
          f"shared={topo['c6']['shared_components']}")
    print(f"topology 26-conn: one_domain={topo['c26']['one_domain']} "
          f"components={topo['c26']['n_components']}")
    t0 = tick("topology", t0, T)

    profile = measure_flap(labels, origin, s_cl, P, N, B, step=2.0)
    a_t = [r["a_true"] for r in profile]
    a_f = [r["a_false"] for r in profile]
    print(f"flap profile: {len(profile)} stations, "
          f"A_true median {np.median(a_t):.1f} mm2, A_false median {np.median(a_f):.1f} mm2")
    t0 = tick("flap", t0, T)

    tears, n_contact, flip = find_tears(labels, origin, s_cl, P, N, B, gt["tears"],
                                        gt["branches"])
    gt_hits = {t["s"]: False for t in gt["tears"]}
    for t in tears:
        m = t.get("gt_match")
        if m and m["hit"]:
            gt_hits[m["s"]] = True
    n_hit = sum(gt_hits.values())
    n_branch = sum(1 for t in tears if t.get("on_branch"))
    print(f"tears: {len(tears)} contact clusters ({n_branch} on branch stubs), "
          f"{n_contact} contact voxels, {n_hit}/{len(gt['tears'])} ground-truth tears matched")
    for t in tears:
        m = t.get("gt_match") or {}
        kind = f"branch:{t['on_branch']}" if t.get("on_branch") else "tear?"
        gs = m.get("s", "-")
        gd = m.get("distance_mm", "-")
        gh = m.get("hit", "-")
        print(f"  cluster {t['n_voxels']:5d} vox  {kind:<22} gt s={gs}  dist={gd}  hit={gh}")
    t0 = tick("tears", t0, T)

    feed = branch_feed(labels, origin, margins, gt["branches"])
    n_ok = sum(1 for r in feed if r.get("match"))
    print(f"branch feed: {n_ok}/{len(feed)} match")
    for r in feed:
        print(f"  {r['name']:<16} gt={r.get('gt_feed'):<5} pred={r.get('pred_feed'):<5} "
              f"fracT={r.get('frac_true')} fracF={r.get('frac_false')}")
    t0 = tick("feed", t0, T)

    sheet = tie_sheet_stats(margins, labels)
    print(f"tie sheet: {sheet['sheet_voxels']} voxels, side_agreement {sheet['side_agreement']}")
    t0 = tick("sheet", t0, T)

    out = {
        "run": RUN,
        "kind": "dissection phantom (dual lumen, entry tears, branch stubs)",
        "spacing_mm": SPACING,
        "gradient_logits_per_mm": GRADIENT,
        "grid": {"shape_zyx": list(shape), "origin_xyz": origin.tolist()},
        "counts": {"true": n_t, "false": n_f, "wall_other": int((labels == 0).sum())},
        "ground_truth": {"tears": gt["tears"], "branches": [
            {k: v for k, v in b.items() if k != "axis"} for b in gt["branches"]]},
        "topology": topo,
        "flap_profile": profile,
        "tears_detected": tears,
        "tear_contact_voxels": n_contact,
        "branch_feed": feed,
        "tie_sheet": sheet,
        "timing_s": {k: round(v, 3) for k, v in T.items()},
        "notes": (
            "Flap is a background wall of thickness flap_t where intact; at a tear the wall "
            "is missing and true/false meet at v=0. A two-lumen model like TotalSegmentator "
            "aortic_dissection would show the flap as the true|false tie sheet (contact) and "
            "tears as openings in it; this phantom resolves the wall so the detector can be "
            "scored against known tears."
        ),
    }
    path = DATA / f"{RUN}_dissection.json"
    path.write_text(json.dumps(out, indent=2))
    print("wrote", path)
    return out


if __name__ == "__main__":
    main()
