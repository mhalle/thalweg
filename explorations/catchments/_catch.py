"""Shared pieces of the catchment prototype: the arterial tree, Strahler groups, nearest groups."""
import json, os, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "vessels"))
from _data import DATA                                                 # noqa: E402

run = sys.argv[1] if len(sys.argv) > 1 else "C3N-00704_ctpa0625"
H, STEP, TAU, DEPTH, TRIM = 1.5, 0.5, 0.25, 4, 3        # grid mm, site spacing mm, mm per logit, ranks, Strahler cut
LOBES = {10: "left upper", 11: "left lower", 12: "right upper", 13: "right middle", 14: "right lower"}
T = {}


def tick(name, t0):
    T[name] = time.time() - t0
    return time.time()


def tree(cls):
    """Segments with Strahler order, parent, and densified points."""
    G = json.load(open(DATA / f"{run}_{cls}_centerlines.json"))
    S = G["segments"]
    kids = defaultdict(list); parent = {}
    for s in S:
        kids[s["a"]].append(s["id"])
    for s in S:
        for c in kids.get(s["b"], []):
            parent[c] = s["id"]
    order = {}
    sys.setrecursionlimit(100000)

    def strahler(i):
        if i not in order:
            ch = [strahler(c) for c in kids.get(S[i]["b"], [])]
            m = max(ch, default=0)
            order[i] = 1 if not ch else (m + 1 if ch.count(m) >= 2 else m)
        return order[i]
    for s in S:
        strahler(s["id"])
    pts, seg = [], []
    for s in S:
        p = np.array(s["points"], float)
        a = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        t = np.arange(0, a[-1] + 1e-9, STEP)
        pts.append(np.stack([np.interp(t, a, p[:, k]) for k in range(3)], 1)); seg.append(np.full(len(t), s["id"]))
    return S, order, parent, np.concatenate(pts), np.concatenate(seg)


def ancestor_at(order, parent, k):
    """segment -> its nearest ancestor-or-self of Strahler order >= k (the root if none)."""
    out = {}
    for i in order:
        j = i
        while order[j] < k and j in parent:
            j = parent[j]
        out[i] = j
    return out


def nearest_groups(X, sites, site_group, depth, k=48):
    """For points X: the `depth` nearest distinct groups, their distances and nearest sites."""
    tr = cKDTree(sites)
    n = len(X)
    G = np.full((n, depth), -1); D = np.full((n, depth), np.inf); P = np.full((n, depth), -1)
    todo = np.arange(n)
    while len(todo):
        d, i = tr.query(X[todo], k=min(k, len(sites)))
        g = site_group[i]
        cnt = np.zeros(len(todo), int)
        gg = np.full((len(todo), depth), -1); dd = np.full((len(todo), depth), np.inf); pp = np.full((len(todo), depth), -1)
        for j in range(d.shape[1]):
            new = (cnt < depth) & ~(gg == g[:, j:j + 1]).any(1)
            r = np.nonzero(new)[0]
            gg[r, cnt[r]] = g[r, j]; dd[r, cnt[r]] = d[r, j]; pp[r, cnt[r]] = i[r, j]
            cnt[r] += 1
        done = (cnt == depth) | (k >= len(sites))
        G[todo[done]], D[todo[done]], P[todo[done]] = gg[done], dd[done], pp[done]
        todo = todo[~done]; k *= 4
    return G, D, P




# SUPPLY: which centerline points feed tissue. "twigs" = Strahler order <= 2 only (the small
# terminal branches; a trunk class keeps just its own side twigs), "all" = every point (the
# first prototype: trunks running past other territories cut them into slivers and specks).
SUPPLY = os.environ.get("SUPPLY", "twigs")
# WALLS: territories never cross these. "lobes" = the crop layer's five lobes (its 3 mm
# total_fast puts the right upper/lower fissure in the wrong place on C3N-00704), "lungs" =
# left vs right only.
WALLS = os.environ.get("WALLS", "lungs")


def supply_mask(order, site_seg):
    if SUPPLY == "all":
        return np.ones(len(site_seg), bool)
    return np.array([order[s] <= 2 for s in site_seg])


def wall_of(lobe_ids):
    """lobe index 1..5 (LOBES order: LUL, LLL, RUL, RML, RLL) -> wall id under WALLS; 0 stays outside."""
    lobe_ids = np.asarray(lobe_ids)
    if WALLS == "lobes":
        return lobe_ids
    return np.where(lobe_ids == 0, 0, np.where(lobe_ids <= 2, 1, 2))
