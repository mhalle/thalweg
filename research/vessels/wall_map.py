"""vmtk's branch mapping without a surface: wall coordinates and ray-cast wall maps from the field.

    python bench/vessels/wall_map.py [RUN]          (after vmtk_mapping.py)

1. Coordinates. vmtk's AbscissaMetric and AngularMetric (vtkvmtkPolyDataCenterline*MetricFilter)
   are point-wise: for a point of group g, each centerline cell of g (with its adjacent bifurcation
   cells, IncludeBifurcations) gives a closest point by the tube function; point, tangent (segment
   direction), normal and abscissa are averaged with weights r^2; the angle is between the point
   and the normal, both projected on the plane normal to the averaged tangent, negative where
   tangent . (position x normal) < 0. Ported here and evaluated at vmtk's surface points.
2. Wall maps. For each branch group, rays from its centerline at every station and angle (in
   vmtk's convention: direction cos(phi) N + sin(phi) (N x T)) run to the first zero crossing of the
   artery margin. r(s, phi) is the wall map - vmtk's patching of a surface, made without one. Any
   quantity at the wall (CT, the margin interval, curvature) maps the same way. Checked against
   vmtk's DistanceToCenterlines at the same (group, abscissa, angle).
"""
import sys, time
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator
from scipy import ndimage as ndi
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _data import DATA
from _field import fine_field, sample, Image
from cases import store, image

run = sys.argv[1] if len(sys.argv) > 1 else "C3N-00704_ctpa0625"
SMOOTH = float(__import__("os").environ.get("STATION_SMOOTH", 0.6))    # mm of arc length; 0 = the raw polyline
M = np.load(DATA / f"{run}_vmtk_mapping.npz")
P, Rad, Absc, Nrm = M["cl_points"], M["cl_radius"], M["cl_abscissa"], M["cl_normal"]
cells = np.split(M["cell_ids"], np.cumsum(M["cell_len"])[:-1])
cgroup, cblank, ccl = M["cell_group"], M["cell_blank"], M["cell_centerline"]
tract = M["cell_tract"]                                 # vmtk's TractIds (order along each centerline)


def closest(x, cell_set, use_radius=True):
    """vmtk PolyBallLine closest point over the cells in cell_set: (value, cell, sub, t) per point.
    use_radius=False is UseRadiusInformationOff: the radii are zero in the search (plain Euclidean)."""
    best = np.full(len(x), np.inf); bc = np.zeros(len(x), int); bs = np.zeros(len(x), int); bt = np.zeros(len(x))
    for ci in cell_set:
        ids = cells[ci]
        if len(ids) < 2: continue
        p0, p1 = P[ids[:-1]], P[ids[1:]]
        r0, r1 = (Rad[ids[:-1]], Rad[ids[1:]]) if use_radius else (np.zeros(len(ids) - 1), np.zeros(len(ids) - 1))
        v0 = p1 - p0; dr = r1 - r0
        den = (v0 * v0).sum(1) - dr * dr
        okd = np.abs(den) > 1e-12
        v1 = x[:, None, :] - p0[None]
        num = (v1 * v0[None]).sum(-1) + dr[None] * r0[None]
        t = np.clip(num / np.where(okd, den, 1.0)[None], 0, 1)
        c = p0[None] + t[..., None] * v0[None]; rc = r0[None] + t * dr[None]
        val = ((x[:, None, :] - c) ** 2).sum(-1) - rc * rc
        val[:, ~okd] = np.inf
        k = val.argmin(1); v = val[np.arange(len(x)), k]
        take = v < best
        best[take] = v[take]; bc[take] = ci; bs[take] = k[take]; bt[take] = t[np.arange(len(x)), k][take]
    return best, bc, bs, bt


def interp(arr, ci, sub, t):
    ids = cells[ci]
    return arr[ids[sub]] * (1 - t)[..., None] + arr[ids[sub + 1]] * t[..., None] if arr.ndim == 2 else \
        arr[ids[sub]] * (1 - t) + arr[ids[sub + 1]] * t


def group_cell_sets(g, include_bifurcations):
    """vmtk: each cell of group g, with (IncludeBifurcations) the blanked cells adjacent along its centerline."""
    sets = []
    for i in np.nonzero(cgroup == g)[0]:
        adj = [j for j in range(len(cells)) if include_bifurcations and j != i and cblank[j] == 1 and
               cgroup[j] != g and ccl[j] == ccl[i] and abs(tract[j] - tract[i]) == 1]
        sets.append([i] + adj)
    return sets


def _average(x, g, include_bifurcations, vmtk_interp=False):
    """Radius^2-weighted closest point, tangent, normal and abscissa over the cells of group g, as the
    vmtk metric filters compute them - with vmtkbranchmetrics.py's settings: UseRadiusInformationOff
    (plain Euclidean closest point; the radius still weights), IncludeBifurcations as given."""
    W = np.zeros(len(x)); A = np.zeros(len(x)); Pt = np.zeros((len(x), 3)); Tg = np.zeros((len(x), 3)); Nm = np.zeros((len(x), 3))
    for cs in group_cell_sets(g, include_bifurcations):
        _, ci, sub, t = closest(x, cs, use_radius=False)
        for u in np.unique(ci):
            s_ = ci == u
            ids = cells[u]
            tt = t[s_]; a0, a1 = ids[sub[s_]], ids[sub[s_] + 1]
            r = Rad[a0] * (1 - tt) + Rad[a1] * tt
            ta = Absc[a0] * (1 - tt) + Absc[a1] * tt
            if vmtk_interp:                   # vtkvmtkCenterlineUtilities::InterpolateTuple: both GetTuple()
                r, ta = Rad[a1], Absc[a1]     # pointers alias one buffer, so it returns the END point's value
            w = r * r
            p0, p1 = P[a0], P[a1]
            Pt[s_] += w[:, None] * (p0 * (1 - tt)[:, None] + p1 * tt[:, None])
            Tg[s_] += w[:, None] * (p1 - p0)
            Nm[s_] += w[:, None] * (Nrm[a1] if vmtk_interp else Nrm[a0] * (1 - tt)[:, None] + Nrm[a1] * tt[:, None])
            A[s_] += w * ta
            W[s_] += w
    return A / W, Pt / W[:, None], Tg, Nm


def metrics(x, g, vmtk_interp=False):
    """vmtk's abscissa (IncludeBifurcationsOn) and angle (IncludeBifurcationsOff) for points x of group g.
    Interpolates along each centerline segment, as vmtk intends. vmtk_interp=True reproduces vmtk's
    output bit for bit instead, defect included (every value taken at the segment's end point)."""
    A, _, _, _ = _average(x, g, True, vmtk_interp)
    _, Pt, Tg, Nm = _average(x, g, False, vmtk_interp)
    Tg /= np.linalg.norm(Tg, axis=1, keepdims=True); Nm /= np.linalg.norm(Nm, axis=1, keepdims=True)
    pos = x - Pt; pos -= (pos * Tg).sum(1, keepdims=True) * Tg; pos /= np.linalg.norm(pos, axis=1, keepdims=True)
    pn = Nm - (Nm * Tg).sum(1, keepdims=True) * Tg; pn /= np.linalg.norm(pn, axis=1, keepdims=True)
    ang = np.arccos(np.clip((pos * pn).sum(1), -1, 1))
    sgn = (Tg * np.cross(pos, pn)).sum(1)
    return A, np.where(sgn < 0, -ang, ang)


# 1. coordinates at vmtk's ORIGINAL surface vertices (the clip-created tie points excluded)
Z = np.load(DATA / f"{run}_vmtk_input.npz")
SP, SG = M["surf_points"], M["surf_group"]
orig = cKDTree(Z["verts"]).query(SP)[0] < 1e-4
t0 = time.time(); t_compat = t_true = 0.0
ourA = np.full(len(SP), np.nan); ourPhi = np.full(len(SP), np.nan); trueA = np.full(len(SP), np.nan); truePhi = np.full(len(SP), np.nan)
for g in np.unique(SG[orig]):
    s = orig & (SG == g)
    if (cgroup == g).any():
        _t = time.time()
        ourA[s], ourPhi[s] = metrics(SP[s], g, vmtk_interp=True)       # the port, checked against vmtk
        t_compat += time.time() - _t; _t = time.time()
        trueA[s], truePhi[s] = metrics(SP[s], g)                          # correct interpolation: the defect's size
        t_true += time.time() - _t
dA = np.abs(ourA - M["surf_abscissa"])[orig]
dPhi = np.abs(np.angle(np.exp(1j * (ourPhi - M["surf_angle"]))))[orig]
print(f"coordinates at {orig.sum()} surface vertices ({time.time() - t0:.1f} s): abscissa |diff| median "
      f"{np.nanmedian(dA):.2e} mm, p99 {np.nanpercentile(dA, 99):.2e}; angle |diff| median {np.degrees(np.nanmedian(dPhi)):.2e} deg, "
      f"p99 {np.degrees(np.nanpercentile(dPhi, 99)):.2e}")
dT = (trueA - M["surf_abscissa"])[orig]
dTphi = np.degrees(np.abs(np.angle(np.exp(1j * (truePhi - M["surf_angle"])))))[orig]
print(f"vmtk's abscissa vs a true interpolation: median {np.nanmedian(dT):+.3f} mm, |diff| p99 {np.nanpercentile(np.abs(dT), 99):.3f} "
      f"(vmtk reads the segment END point's abscissa and radius); angle |diff| p99 "
      f"{np.nanpercentile(dTphi, 99):.2f} deg (its normal too)")

# 2. wall maps by ray casting in the field, per group, in vmtk's coordinates
margins, _, grid, _ = fine_field(store(run)); m = margins["lung_arteries"]
t0 = time.time()
PHI = np.radians(np.arange(-180, 180, 5.0))
maps = {}
for g in np.unique(cgroup[cblank == 0]):
    ci = [i for i in np.nonzero((cgroup == g) & (cblank == 0))[0]]
    ci = max(ci, key=lambda i: len(cells[i]))                         # the longest cell as the station line
    ids = cells[ci]
    step = np.r_[np.inf, np.linalg.norm(np.diff(P[ids], axis=0), axis=1)]
    ids = ids[step > 0.05]                                            # near-duplicate points give no tangent
    if len(ids) < 4: continue
    C = P[ids]
    if SMOOTH > 0:                                                    # the ray frame from a smoothed station line
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
        C = ndi.gaussian_filter1d(C, SMOOTH / np.median(np.diff(arc)), axis=0, mode="nearest")
    Tn = np.gradient(C, axis=0)
    Tn /= np.linalg.norm(Tn, axis=1, keepdims=True)
    Nn = Nrm[ids] - (Nrm[ids] * Tn).sum(1, keepdims=True) * Tn; Nn /= np.linalg.norm(Nn, axis=1, keepdims=True)
    Bn = np.cross(Nn, Tn)
    dirs = np.cos(PHI)[None, :, None] * Nn[:, None, :] + np.sin(PHI)[None, :, None] * Bn[:, None, :]    # (S, A, 3)
    rmax = 3 * Rad[ids].max() + 2.0
    steps = np.arange(0.05, rmax, 0.05)
    q = C[:, None, None, :] + steps[None, None, :, None] * dirs[:, :, None, :]                           # (S, A, K, 3)
    v = sample(m, grid, q.reshape(-1, 3)).reshape(q.shape[:3])
    out = v <= 0
    first = np.where(out.any(-1), out.argmax(-1), -1)
    k = np.clip(first, 1, len(steps) - 1)
    va, vb = np.take_along_axis(v, (k - 1)[..., None], -1)[..., 0], np.take_along_axis(v, k[..., None], -1)[..., 0]
    r = steps[k - 1] + (steps[k] - steps[k - 1]) * va / np.maximum(va - vb, 1e-9)
    r[first < 1] = np.nan
    maps[g] = dict(s=Absc[ids], r=r, C=C, dirs=dirs)
t_maps = time.time() - t0
print(f"wall maps: {len(maps)} groups, {sum(v['r'].size for v in maps.values())} rays, 5 deg x 0.3 mm, in {time.time() - t0:.1f} s")
turn = np.concatenate([np.degrees(np.arccos(np.clip((np.gradient(mp["C"], axis=0)[1:] / np.linalg.norm(np.gradient(mp["C"], axis=0)[1:], axis=1, keepdims=True)
                       * (np.gradient(mp["C"], axis=0)[:-1] / np.linalg.norm(np.gradient(mp["C"], axis=0)[:-1], axis=1, keepdims=True))).sum(1), -1, 1)))
                       for mp in maps.values()])
streak = np.concatenate([np.nanmedian(np.abs(mp["r"][1:-1] - 0.5 * (mp["r"][:-2] + mp["r"][2:])), axis=1) for mp in maps.values()])
print(f"station line smoothed over {SMOOTH} mm: tangent turn per station median {np.median(turn):.2f} deg, p99 {np.percentile(turn, 99):.1f}, "
      f"max {turn.max():.1f}; streak score (median |r - mean of its two neighbor stations|) median {np.nanmedian(streak):.4f} mm, "
      f"p99 {np.nanpercentile(streak, 99):.3f}, max {np.nanmax(streak):.3f}")

# 3. against vmtk's DistanceToCenterlines at the same (group, abscissa, angle)
dd, rows = [], []
for g, mp in maps.items():
    s = orig & (SG == g)
    if s.sum() < 20: continue
    sg_, order = np.unique(mp["s"], return_index=True)
    if len(sg_) < 4: continue
    rr = mp["r"][order]
    ext = np.vstack([rr, rr[:1]])                                     # close the angle axis at +180
    f = RegularGridInterpolator((sg_, np.r_[PHI, np.pi]), np.hstack([rr, rr[:, :1]]), bounds_error=False, fill_value=np.nan)
    a, phi = M["surf_abscissa"][s], M["surf_angle"][s]
    inner = (a > sg_[0] + 1.0) & (a < sg_[-1] - 1.0)                  # away from the group's ends
    ours = f(np.stack([a, np.where(phi >= PHI[0], phi, phi + 2 * np.pi)], 1))
    d = ours - M["surf_dist"][s]
    dd.append(d[inner & np.isfinite(d)])
dd = np.concatenate(dd)
print(f"ray-cast wall radius vs vmtk DistanceToCenterlines at {len(dd)} surface points (group interiors): "
      f"median diff {np.median(dd):+.3f} mm, |diff| median {np.median(np.abs(dd)):.3f}, p90 {np.percentile(np.abs(dd), 90):.3f}")

# a figure: the four longest branches' wall maps, unrolled (s along, phi around). A ray that runs past
# 1.8x its station's median radius (or never exits) has left along a side branch: that is an OSTIUM.
top = sorted(maps, key=lambda g: -np.ptp(maps[g]["s"]))[:4]
fig, ax = plt.subplots(len(top), 1, figsize=(12, 2.4 * len(top)))
for a_, g in zip(np.atleast_1d(ax), top):
    mp = maps[g]; r = mp["r"]
    med = np.nanmedian(r, axis=1, keepdims=True)
    ostium = ~np.isfinite(r) | (r > 1.8 * med)
    shown = np.ma.masked_where(ostium, r)
    cm = plt.get_cmap("viridis").copy(); cm.set_bad("#d0d0d0")
    im = a_.imshow(shown.T, aspect="auto", origin="lower", cmap=cm, extent=(mp["s"][0], mp["s"][-1], -180, 180))
    a_.set_ylabel("angle (deg)")
    a_.set_title(f"group {g}: wall radius r(s, angle), ray-cast from the field; gray = ostium ({ostium.mean():.1%} of rays)", fontsize=9)
    plt.colorbar(im, ax=a_, label="mm")
np.atleast_1d(ax)[-1].set_xlabel("abscissa s (mm, vmtk's offset convention)")
plt.tight_layout(); plt.savefig(DATA / f"{run}_wall_maps.png", dpi=110)
np.savez(DATA / f"{run}_wall_maps.npz", **{f"g{g}_{k}": v for g, mp in maps.items() for k, v in mp.items() if k in ("s", "r")})
print(f"wrote {DATA / f'{run}_wall_maps.png'}")
print("TIMING " + __import__("json").dumps({k: round(float(v), 4) for k, v in ({"coordinates_vmtk_compatible": t_compat, "coordinates": t_true, "wall_maps": t_maps}).items()}))
