"""Can a front (level-set style) reconnect what the model leaves apart? An experiment, not a method.

    ../../../haversack/.venv/bin/python explorations/bridging/bridge_gaps.py [PATIENT] [natural|synthetic|both]

The tracer keeps a structure's largest component and drops the rest. A front started on a dropped
fragment and run until it meets the tree joins them wherever its speed image stays high; the level
it has to sink to is the BOTTLENECK of the best path (the minimax path: the highest threshold at
which the two are connected through the image's superlevel set). That level is what this script
measures, with two speed images: the structure's own margin, and the CT.

natural    Every run of the patient's thickness ladder: each dropped fragment of the arteries and
           veins, its gap to the tree, the margin's bottleneck, the CT's bottleneck to its OWN tree
           and, as a leak control, to the OTHER vessel class's tree. The truth is another
           reconstruction of the same acquisition (the thin reference; for the reference, the next
           run): is the fragment part of that run's tree?
synthetic  On the thin reference: a stretch of 2, 4 or 8 mm is cut out of a traced segment
           (treated as unlabeled), and the CT's bottleneck across the cut is compared with the
           bottleneck from the same stump to the nearest vessel of the other class (a join that
           must not be made), in every ladder image sampled on the reference grid.

CT scores are contrast fractions f = (h - bg) / (ref - bg): h the bottleneck HU, bg the local
parenchyma (20th percentile of the unlabeled voxels in the box), ref the fragment's (or the dimmer
stump's) core HU, its 90th percentile.
Paths run through unlabeled voxels (and the structure's own fragments) only, 26-connected.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _thalweg import DATA, LADDERS, image, store  # noqa: E402  (also puts src/ on the path)
from thalweg.kernel import medial  # noqa: E402
from thalweg.kernel.topology import components, field_edges  # noqa: E402

PATIENT = sys.argv[1] if len(sys.argv) > 1 else "C3N-00704"
PART = sys.argv[2] if len(sys.argv) > 2 else "both"
NAMES = {3: "arteries", 4: "veins"}
MARGIN = {3: "lung_arteries", 4: "lung_veins"}
S26 = np.ones((3, 3, 3), bool)
PAD_MM, FAR_MM = 6.0, 20.0
CUTS = (2.0, 4.0, 8.0)
RBINS = [0, 1.25, 2.0, 3.0, 99]
rng = np.random.default_rng(0)
T0 = time.time()


def say(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


# -- loading ----------------------------------------------------------------------------------
def read_ct(run):
    from _field import Image
    src = next(s for lad in LADDERS.values() for r, s, *_ in lad if r == run)
    return Image(image(src))


def on_grid(img, origin, dirs, shape):
    """The CT sampled (trilinear) on a field grid: index i -> world i @ dirs + origin."""
    inv = np.linalg.inv(img.grid.dirs)
    A, b = dirs @ inv, (origin - img.grid.origin) @ inv
    return ndi.affine_transform(img.arr, A.T, offset=b, output_shape=shape, order=1, cval=-1024.0)


def load(run, ct=True):
    from _field import fine_field
    margins, labels, grid, clip = fine_field(store(run))
    R = dict(run=run, labels=labels, origin=grid.origin, dirs=grid.dirs, spacing=grid.spacing, clip=clip,
             m={c: margins[MARGIN[c]] for c in NAMES}, vol={}, main={}, sizes={})
    for c in NAMES:
        idx, r, k, _, _ = field_edges(R["m"][c])
        _, comp = components(len(idx), r, k)
        sizes = np.bincount(comp)
        vol = np.zeros(labels.shape, np.int32)
        vol[tuple(idx.T)] = comp + 1
        R["vol"][c], R["main"][c], R["sizes"][c] = vol, int(np.argmax(sizes)) + 1, sizes
    if ct:
        R["ct"] = on_grid(read_ct(run), grid.origin, grid.dirs, labels.shape)
    say(f"{run}: shape {labels.shape}, " + ", ".join(
        f"{NAMES[c]} {len(R['sizes'][c]) - 1} fragments ({1 - R['sizes'][c].max() / R['sizes'][c].sum():.2%} of voxels)"
        for c in NAMES))
    return R


# -- the bottleneck -----------------------------------------------------------------------------
def _joined(mask, F, M):
    lab, _ = ndi.label(mask, S26)
    return np.intersect1d(np.unique(lab[F]), np.unique(lab[M])).size > 0


def bottleneck(img, F, M, D):
    """The highest level h at which F and M are connected through D & (img >= h) (26-connected).
    +inf: they touch. -inf: not connected even through all of D."""
    base = F | M
    if _joined(base, F, M):
        return np.inf
    if not _joined(base | D, F, M):
        return -np.inf
    vals = np.unique(img[D])
    lo, hi = 0, len(vals)                       # joined at vals[lo]; not joined at vals[hi] (or past the end)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _joined(base | (D & (img >= vals[mid])), F, M):
            lo = mid
        else:
            hi = mid
    return float(vals[lo])


def box_of(idx, pad_mm, R):
    pad = np.ceil(pad_mm / R["spacing"]).astype(int)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, R["labels"].shape)
    return tuple(slice(a, b) for a, b in zip(lo, hi))


def core(hu):
    """A stump's lumen HU: its 90th percentile (the median is diluted by partial-volume wall voxels)."""
    return float(np.percentile(hu, 90))


def fraction(h, bg, ref):
    if not np.isfinite(h):
        return 1.0 if h > 0 else 0.0
    return float(np.clip((h - bg) / max(ref - bg, 1.0), 0.0, 1.5))


# -- natural fragments --------------------------------------------------------------------------
def truth(world, ref, c):
    """Where a fragment's points fall in another run: shares in that run's tree of the class, in
    any component of the class, and in the other vessel class."""
    q = np.rint((world - ref["origin"]) @ np.linalg.inv(ref["dirs"])).astype(int)
    ok = np.all((q >= 0) & (q < ref["labels"].shape), 1)
    if not ok.any():
        return 0.0, 0.0, 0.0
    v = ref["vol"][c][tuple(q[ok].T)]
    lab = ref["labels"][tuple(q[ok].T)]
    n = len(q)
    return float((v == ref["main"][c]).sum() / n), float((v > 0).sum() / n), float((lab == 7 - c).sum() / n)


def natural(R, ref):
    rows = []
    for c in NAMES:
        vol, main = R["vol"][c], R["main"][c]
        main_idx = np.argwhere(vol == main)
        tree = cKDTree(main_idx @ R["dirs"] + R["origin"])
        objs = ndi.find_objects(vol)
        for fid in range(1, len(R["sizes"][c]) + 1):
            if fid == main or objs[fid - 1] is None:
                continue
            sl = objs[fid - 1]
            F_idx = np.argwhere(vol[sl] == fid) + [s.start for s in sl]
            world = F_idx @ R["dirs"] + R["origin"]
            d, near = tree.query(world)
            gap = float(d.min())
            t_main, t_class, t_other = truth(world, ref, c)
            kind = ("break" if t_main >= 0.5 else "fragment" if t_class >= 0.5
                    else "other class" if t_other >= 0.5 else "absent")
            row = dict(cls=NAMES[c], voxels=len(F_idx), gap=gap, truth=kind)
            if gap <= FAR_MM:
                b = box_of(np.vstack([F_idx, main_idx[near[np.argmin(d)]][None]]), PAD_MM, R)
                v, lab = vol[b], R["labels"][b]
                F, M = v == fid, v == main
                bgmask = lab == 0
                D = bgmask | ((v > 0) & ~M & ~F)                     # unlabeled, or the class's other fragments
                ct, m = R["ct"][b], R["m"][c][b]
                bg = float(np.percentile(ct[bgmask], 20))
                ref_hu = core(ct[F])
                row["margin"] = bottleneck(m, F, M, ~M & ~F)
                row["f_own"] = fraction(bottleneck(ct, F, M, D), bg, ref_hu)
                ov = R["vol"][7 - c][b]
                O = ov == R["main"][7 - c]
                if O.any():
                    row["f_other"] = fraction(bottleneck(ct, F, O, D), bg, ref_hu)
            rows.append(row)
    return rows


def rates(rows, clip):
    """True breaks (the other reconstruction has the fragment in its tree) against fragments the
    other reconstruction gives to the other vessel class: how each rule separates them."""
    near = [r for r in rows if "margin" in r]
    P = [r for r in near if r["truth"] == "break"]
    N = [r for r in near if r["truth"] == "other class"]
    if len(P) < 5 or len(N) < 5:
        return
    fin = lambda v: np.nan_to_num(np.array(v, float), posinf=99.0, neginf=-99.0)
    print(f"   true breaks ({len(P)}) against fragments of the other class ({len(N)}):")
    for name, key in (("margin bottleneck", lambda r: r["margin"]), ("CT to own tree", lambda r: r["f_own"]),
                      ("CT own minus other", lambda r: r["f_own"] - r.get("f_other", 0.0))):
        print(f"      {name:20s} AUC {auc(fin([key(r) for r in P]), fin([key(r) for r in N])):.2f}")
    for t in (-2.0, -4.0, -6.0, -clip + 0.5):
        print(f"      join if margin > {t:5.1f}: {np.mean([r['margin'] > t for r in P]):4.0%} of breaks, "
              f"{np.mean([r['margin'] > t for r in N]):4.0%} of the other class")


def report_natural(run, rows, clip):
    print(f"\n== {run}: {len(rows)} dropped fragments, {sum(r['voxels'] for r in rows)} voxels")
    print("   truth in the other reconstruction    n  voxels  gap mm p50 | margin p50 | margin joins | CT joins own"
          "  CT joins other")
    for kind in ("break", "fragment", "other class", "absent"):
        rr = [r for r in rows if r["truth"] == kind]
        if not rr:
            continue
        near = [r for r in rr if "margin" in r]
        mj = sum(r["margin"] > -clip + 0.5 for r in near)
        own = sum(r["f_own"] >= 0.5 for r in near)
        oth = sum(r.get("f_other", 0) >= 0.5 for r in near)
        med = np.median(np.clip([r["margin"] for r in near], -clip, clip)) if near else np.nan
        print(f"   {kind:34s} {len(rr):4d} {sum(r['voxels'] for r in rr):7d}  {np.median([r['gap'] for r in rr]):10.1f} |"
              f"   {med:6.2f}   | {mj:4d} / {len(near):<4d}  | {own:4d} / {len(near):<4d}   {oth:4d} / {len(near):<4d}")
    g = np.array([r["gap"] for r in rows if "margin" in r])
    m = np.clip([r["margin"] for r in rows if "margin" in r], -clip, clip)
    print("   margin bottleneck by gap (dilation alone reaches the plateau, -clip, past ~1.5 mm):")
    for lo, hi in ((0, 1.5), (1.5, 3), (3, 6), (6, 12), (12, 21)):
        sel = (g >= lo) & (g < hi)
        if sel.any():
            print(f"      gap {lo:4.1f}-{hi:<4.1f} mm  n {sel.sum():3d}  p10/p50/p90 "
                  f"{np.percentile(m[sel], 10):6.2f} / {np.percentile(m[sel], 50):6.2f} / {np.percentile(m[sel], 90):6.2f}")
    rates(rows, clip)


# -- synthetic cuts -----------------------------------------------------------------------------
def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return np.nan
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def synthetic(R, cts, per_cut=120):
    """Cuts on the reference run's traced segments; ``cts``: {image name: CT on the reference grid}."""
    geo = SimpleNamespace(origin=R["origin"], directions=R["dirs"])
    pos, neg = [], []
    for c in NAMES:
        tree_ = medial.trace(R["m"][c], geo, ridge_passes=1)
        segs = tree_.segments
        P = np.concatenate([np.array(s["points"]) for s in segs])
        seg_of = np.concatenate([np.full(len(s["points"]), i) for i, s in enumerate(segs)])
        arc = np.concatenate([np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.array(s["points"]), axis=0), axis=1))]
                              for s in segs])
        ptree = cKDTree(P)
        say(f"{NAMES[c]}: {len(segs)} segments traced")
        for L in CUTS:
            cand = [i for i, s in enumerate(segs) if s["length_mm"] >= L + 6.0]
            for i in rng.permutation(cand)[:per_cut]:
                s = segs[i]
                pts, rad = np.array(s["points"]), np.array(s["radius"])
                a = arc[seg_of == i]
                mid = 0.5 * a[-1]
                s0, s1 = mid - L / 2, mid + L / 2
                near = (a >= s0 - 2.5) & (a <= s1 + 2.5)
                r = float(np.median(rad[(a >= s0) & (a <= s1)])) if ((a >= s0) & (a <= s1)).any() else float(np.median(rad))
                idx = np.rint((pts[near] - R["origin"]) @ np.linalg.inv(R["dirs"])).astype(int)
                b = box_of(idx, 10.0, R)
                lab, v = R["labels"][b], R["vol"][c][b]
                own = np.argwhere(v > 0)
                w = (own + [x.start for x in b]) @ R["dirs"] + R["origin"]
                k = ptree.query(w)[1]
                here, sa = seg_of[k] == i, arc[k]

                def part(sel):
                    out = np.zeros(lab.shape, bool)
                    out[tuple(own[here & sel].T)] = True
                    return out

                G = part((sa >= s0) & (sa <= s1))
                A = part((sa >= s0 - 2.5) & (sa < s0))
                B = part((sa > s1) & (sa <= s1 + 2.5))
                if not (A.any() and B.any() and G.any()):
                    continue
                bgmask = lab == 0
                other = np.argwhere(lab == 7 - c)
                N = None
                if len(other):
                    dA = cKDTree(np.argwhere(A) * R["spacing"]).query(other * R["spacing"])[0]
                    if dA.min() >= 1.0:
                        N = np.zeros(lab.shape, bool)
                        N[tuple(other[dA <= dA.min() + 1.5].T)] = True
                        dneg = float(dA.min())
                p = dict(cls=NAMES[c], L=L, r=r, f={}, around={})
                n = dict(cls=NAMES[c], gap=dneg, r=r, f={}) if N is not None else None
                for name, ct in cts.items():
                    x = ct[b]
                    bg = float(np.percentile(x[bgmask], 20))
                    ref_hu = min(core(x[A]), core(x[B]))
                    p["f"][name] = fraction(bottleneck(x, A, B, bgmask | G), bg, ref_hu)
                    p["around"][name] = fraction(bottleneck(x, A, B, bgmask), bg, ref_hu)
                    if n is not None:
                        n["f"][name] = fraction(bottleneck(x, A, N, bgmask), bg,
                                                min(core(x[A]), core(x[N])))
                pos.append(p)
                if n is not None:
                    neg.append(n)
            say(f"{NAMES[c]}: cuts of {L:g} mm done ({len(pos)} cuts, {len(neg)} controls so far)")
    return pos, neg


def report_synthetic(pos, neg, names):
    gapbin = {2.0: (1.0, 3.0), 4.0: (3.0, 6.0), 8.0: (6.0, 12.0)}
    for name in names:
        print(f"\n== synthetic cuts, image {name}")
        print("   cut mm  radius mm      n  f across p10/p50 | f around p50 |  n ctrl  f to other class p50/p90 |  AUC"
              "  | joined at f>=0.5: true / false")
        for L in CUTS:
            for lo, hi in zip(RBINS[:-1], RBINS[1:]):
                pp = [p for p in pos if p["L"] == L and lo <= p["r"] < hi]
                nn = [n for n in neg if gapbin[L][0] <= n["gap"] < gapbin[L][1] and lo <= n["r"] < hi]
                if len(pp) < 5:
                    continue
                fp = np.array([p["f"][name] for p in pp])
                fa = np.array([p["around"][name] for p in pp])
                fn = np.array([n["f"][name] for n in nn]) if nn else np.array([np.nan])
                print(f"   {L:5g}   {lo:4.2f}-{hi:<5.2f} {len(pp):5d}  {np.percentile(fp, 10):6.2f} / {np.median(fp):4.2f}    |"
                      f"   {np.median(fa):5.2f}      | {len(nn):6d}   {np.nanmedian(fn):6.2f} / {np.nanpercentile(fn, 90):4.2f}"
                      f"          | {auc(fp, fn[np.isfinite(fn)]):4.2f}  |   {np.mean(fp >= 0.5):5.0%} / {np.mean(fn >= 0.5):5.0%}")


# -- main ---------------------------------------------------------------------------------------
ladder = LADDERS[PATIENT]
ref_run = ladder[0][0]
out = dict(patient=PATIENT, natural={}, synthetic={})
R0 = load(ref_run)
if PART in ("natural", "both"):
    keep = None
    for k, (run, *_r) in enumerate(ladder[1:], 1):
        R = load(run)
        if k == 1:                                                  # the reference is judged by the next run
            rows = natural(R0, R)
            report_natural(ref_run, rows, R0["clip"])
            out["natural"][ref_run] = rows
        rows = natural(R, R0)
        report_natural(run, rows, R["clip"])
        out["natural"][run] = rows
        del R
    print(f"\n== {PATIENT}: every run pooled")
    rates([r for rows in out["natural"].values() for r in rows], R0["clip"])
if PART in ("synthetic", "both"):
    cts = {ref_run: R0["ct"]}
    for run, *_r in ladder[1:]:
        cts[run] = on_grid(read_ct(run), R0["origin"], R0["dirs"], R0["labels"].shape)
        say(f"CT of {run} on the reference grid")
    pos, neg = synthetic(R0, cts)
    report_synthetic(pos, neg, list(cts))
    out["synthetic"] = dict(cuts=pos, controls=neg)
dest = DATA / f"{PATIENT}_bridging.json"
dest.write_text(json.dumps(out, default=float))
say(f"wrote {dest}")
