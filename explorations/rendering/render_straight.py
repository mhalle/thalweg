"""Proof of concept: a straightened vessel rendered in 3D straight from the field.

    cd explorations/rendering && ../../../haversack/.venv/bin/python render_straight.py [PATIENT] [R_START_MM] [HALF_WIDTH_MM]

No resampled volume: rays run straight through a display box (s along the path, u and v across
it); every sample maps to world as C(s) + u n1(s) + v n2(s) and reads the artery and vein margins
there (trilinear, torch). First hit where either margin turns positive, refined by bisection;
normals by central differences in display space through the same map; Lambert shading, full
opacity. Arteries red, veins blue.

The margins and the artery centerline tree come from thalweg (explorations/_thalweg.py: the store's
fine layer, and thalweg.kernel.medial.trace with ridge_passes=1, the research reference, in memory).
"""
import json, sys, time
import numpy as np
import torch
from scipy.interpolate import splprep, splev
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))                  # explorations/_thalweg.py
from _thalweg import DATA, LADDERS, centerlines, fine_field

patient = sys.argv[1] if len(sys.argv) > 1 else "C3N-00704"
R_START = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
W = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
PX = 0.1                                                         # mm per OUTPUT pixel, both axes
import os as _os
SS = int(_os.environ.get("SS", "2" if _os.environ.get("STYLE") == "npr" else "1"))   # supersampling
RX = PX / SS                                                     # mm per RENDERED pixel
import os
STUB = float(os.environ["STUB"]) if os.environ.get("STUB") else None
STYLE = os.environ.get("STYLE", "shaded")       # "npr": sdfview's illustration style, on white
SMOOTH_SURF, SMOOTH_NORM = 0.3, 0.8              # mm (npr only): silhouettes lightly, shading strongly
# STUB = L mm: show only the path's own artery and the first L mm (along the tree) of every branch
# leaving it; everything else (crossing arteries, veins) is clipped by the graph
ref = LADDERS[patient][0][0]
t0 = time.time()

# the path: trunk -> farthest left-lung tip of the field-mode artery graph (as straighten.py)
G = centerlines(ref, "lung_arteries")
N, S = G["nodes"], G["segments"]
seg_into = {s["b"]: s["id"] for s in S}
def chain_to(node):
    segs = []
    while node in seg_into:
        s = S[seg_into[node]]; segs.append(s["id"]); node = s["a"]
    return segs[::-1]
best, best_len = None, 0
for n in N:
    if n["kind"] == "tip" and n["point"][0] > G["root"][0] + 30:
        ch = chain_to(n["id"]); L = sum(S[j]["length_mm"] for j in ch)
        if L > best_len: best, best_len = ch, L
pts = np.concatenate([np.array(S[j]["points"])[(1 if i else 0):] for i, j in enumerate(best)])
rad = np.concatenate([np.array(S[j]["radius"])[(1 if i else 0):] for i, j in enumerate(best)])
first = int(np.argmax(rad < R_START))                              # start at the lobar artery
pts, rad = pts[first:], rad[first:]
keep = np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-6]
tck, _ = splprep(pts[keep].T, w=1.0 / (0.15 * rad[keep] + 0.05), s=float(keep.sum()), k=3)
fine = np.array(splev(np.linspace(0, 1, 6000), tck)).T
arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(fine, axis=0), axis=1))]
sg = np.arange(0, arc[-1], RX)
C = np.stack([np.interp(sg, arc, fine[:, a]) for a in range(3)], 1)
T = np.gradient(C, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True)
n1 = np.zeros_like(T)
a0 = np.array([0, 0, 1.0]) if abs(T[0, 2]) < 0.9 else np.array([1.0, 0, 0])
n1[0] = np.cross(T[0], a0); n1[0] /= np.linalg.norm(n1[0])
for i in range(1, len(T)):                                          # parallel transport
    v = n1[i - 1] - (n1[i - 1] @ T[i]) * T[i]; n1[i] = v / np.linalg.norm(v)
n2 = np.cross(T, n1)
kappa = np.linalg.norm(np.gradient(T, axis=0), axis=1) / RX
print(f"{patient}: path {arc[-1]:.0f} mm from r = {rad[0]:.1f} mm to {rad[-1]:.1f} mm; half-width {W} mm; "
      f"tightest bend radius {1 / kappa.max():.1f} mm ({(1 / kappa < W).mean():.1%} of stations fold within the box)")
fold = 1 / kappa < W
if fold.any():
    edges = np.flatnonzero(np.diff(np.r_[0, fold.astype(int), 0]))
    print("  folding at s = " + ", ".join(f"{sg[a]:.1f}-{sg[b - 1]:.1f} mm (bend radius {1 / kappa[a:b].max():.1f} mm)"
                                          for a, b in zip(edges[::2], edges[1::2])))

# the field on the device, sampled in world mm
dev = "mps" if torch.backends.mps.is_available() else "cpu"
margins, grid = fine_field(ref)
if STUB is None:
    chans = np.stack([margins["lung_arteries"], margins["lung_veins"]])
else:
    from scipy.spatial import cKDTree
    K = 9.0                                            # logits per mm at the surface (measured), margin -> ~mm
    m_a = margins["lung_arteries"]
    path_set = set(best)
    path_nodes = {S[j]["a"] for j in best} | {S[j]["b"] for j in best}
    out_of = {}
    for sgm in S:
        out_of.setdefault(sgm["a"], []).append(sgm["id"])
    # stub segments: each branch leaving a path node, and its descendants within STUB mm of the junction
    stub_off = {}
    stack = [(c, 0.0) for n in path_nodes for c in out_of.get(n, []) if c not in path_set]
    while stack:
        j, off = stack.pop()
        if off >= STUB or j in stub_off: continue
        stub_off[j] = off
        stack.extend((c, off + S[j]["length_mm"]) for c in out_of.get(S[j]["b"], []))
    # every artery graph point densified to 0.25 mm, with its segment and (for stubs) arc from the junction
    P, SEG, ARC, RAD = [], [], [], []
    for sgm in S:
        q = np.array(sgm["points"]); rq = np.array(sgm["radius"])
        if len(q) < 2: continue
        L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
        t = np.arange(0, L[-1] + 1e-9, 0.25)
        P.append(np.stack([np.interp(t, L, q[:, a]) for a in range(3)], 1)); RAD.append(np.interp(t, L, rq))
        SEG.append(np.full(len(t), sgm["id"])); ARC.append(stub_off.get(sgm["id"], 0.0) + t)
    P, SEG, ARC, RAD = map(np.concatenate, (P, SEG, ARC, RAD))
    # ownership of EVERY lattice point near the path (vessel or not, so the keep fields only make
    # cuts and the artery margin alone makes the walls): the nearest graph point's segment, if the
    # point lies within that graph point's radius + 1.5 mm (else it belongs to no tube of this tree)
    lo_i = np.floor(grid.to_index(C).min(0) - (W * 1.5 + STUB + 5) / grid.spacing).astype(int).clip(0)
    hi_i = np.ceil(grid.to_index(C).max(0) + (W * 1.5 + STUB + 5) / grid.spacing).astype(int).clip(max=np.array(m_a.shape) - 1)
    vox = np.stack(np.meshgrid(*[np.arange(a, b + 1) for a, b in zip(lo_i, hi_i)], indexing="ij"), -1).reshape(-1, 3)
    vox = vox[cKDTree(C).query(grid.to_world(vox))[0] < W * 1.5 + STUB + 5]
    dist, own = cKDTree(P).query(grid.to_world(vox))
    ok = dist <= RAD[own] + 1.5
    vox, own = vox[ok], own[ok]
    seg_v, arc_v = SEG[own], ARC[own]
    keep_path = np.full(m_a.shape, -1.0, np.float32)
    keep_stub = np.full(m_a.shape, -1.0, np.float32)
    is_path = np.isin(seg_v, list(path_set))
    is_stub = np.isin(seg_v, list(stub_off))
    keep_path[tuple(vox[is_path].T)] = 1.0
    keep_stub[tuple(vox[is_stub].T)] = np.clip(STUB - arc_v[is_stub], -1.0, 1.0)   # cut at STUB mm, in mm
    chans = np.stack([np.minimum(m_a / K, keep_path), np.minimum(m_a / K, keep_stub)])
    print(f"  graph clip: {len(path_set)} path segments, {len(stub_off)} branch segments within {STUB:g} mm; "
          f"{is_path.sum()} path / {is_stub.sum()} branch lattice points kept of {len(vox)} owned near the path")
if STYLE == "npr":
    # an illustration may smooth freely: the surface from a lightly smoothed field (voxel-scale bumps
    # off the silhouettes, 1 mm vessels intact), the shading normals from a strongly smoothed one
    from scipy.ndimage import gaussian_filter
    sm = lambda a, mm: np.stack([gaussian_filter(c, mm / grid.spacing) for c in a])
    vol_n = torch.from_numpy(sm(chans, SMOOTH_NORM)[None].astype(np.float32)).to(dev)
    chans = sm(chans, SMOOTH_SURF)
vol = torch.from_numpy(chans[None].astype(np.float32)).to(dev)                                 # (1,2,D,H,Wd)
if STYLE != "npr":
    vol_n = vol
inv = torch.tensor(np.linalg.inv(grid.dirs), dtype=torch.float32, device=dev)
org = torch.tensor(grid.origin, dtype=torch.float32, device=dev)
shape = torch.tensor(vol.shape[2:], dtype=torch.float32, device=dev)
Ct, N1, N2 = (torch.tensor(a, dtype=torch.float32, device=dev) for a in (C, n1, n2))


def field_at(si, u, v, src=None):
    """Both margins at display points (station index si, u, v in mm) -> (..., 2)."""
    src = vol if src is None else src
    p = Ct[si] + u[..., None] * N1[si] + v[..., None] * N2[si]
    idx = (p - org) @ inv                                            # array index (d, h, w)
    g = (2 * idx / (shape - 1) - 1).flip(-1)                         # grid_sample wants (x=w, y=h, z=d)
    out = torch.nn.functional.grid_sample(src, g.reshape(1, -1, 1, 1, 3), align_corners=True,
                                          padding_mode="border")
    return out.reshape(2, -1).T.reshape(*u.shape, 2)


# rays: one per (s, u) pixel, marching v from +W to -W (orthographic, looking along -n2)
us = torch.arange(-W, W + 1e-6, RX, device=dev)
si = torch.arange(len(sg), device=dev)
SI = si[:, None].expand(len(si), len(us)); U = us[None, :].expand(len(si), len(us))
vs = torch.arange(W, -W - 1e-6, -0.05, device=dev)
hit_v = torch.full(U.shape, float("nan"), device=dev)
prev = None
for k, v in enumerate(vs):
    f = field_at(SI, U, torch.full_like(U, v)).amax(-1)
    new = torch.isnan(hit_v) & (f > 0)
    if k and new.any():
        # bisection between the previous and this step
        lo = torch.full_like(U, float(vs[k - 1])); hi = torch.full_like(U, float(v))
        for _ in range(8):
            mid = 0.5 * (lo + hi)
            fm = field_at(SI, U, mid).amax(-1)
            lo = torch.where(fm > 0, lo, mid); hi = torch.where(fm > 0, mid, hi)
        hit_v = torch.where(new, hi, hit_v)
    elif new.any():
        hit_v = torch.where(new, torch.full_like(U, float(v)), hit_v)
hit = ~torch.isnan(hit_v)
V = torch.nan_to_num(hit_v, nan=0.0)
fa = field_at(SI, U, V)
cls = fa.argmax(-1)                                                   # 0 artery, 1 vein
# normals: central differences in display space, of the hit class's margin, through the map
# the stencil: about a voxel in illustration (a trilinear gradient jumps at every cell face, and a
# short stencil shows those jumps as stripes - sdfview: "average the normal over a voxel")
h = 0.6 if STYLE == "npr" else 0.25
def g_of(ds, du, dv):
    s2 = (SI + ds).clamp(0, len(sg) - 1)
    return field_at(s2, U + du, V + dv, vol_n).gather(-1, cls[..., None])[..., 0]
steps = max(1, int(round(h / RX)))
gs = (g_of(steps, 0, 0) - g_of(-steps, 0, 0)) / (2 * steps * RX)
gu = (g_of(0, h, 0) - g_of(0, -h, 0)) / (2 * h)
gv = (g_of(0, 0, h) - g_of(0, 0, -h)) / (2 * h)
nrm = -torch.stack([gs, gu, gv], -1); nrm = nrm / nrm.norm(dim=-1, keepdim=True).clamp_min(1e-6)
# rays that start inside a vessel hit the box face itself: shade that as a flat cap facing the viewer
cap = hit & (V >= W - 1e-4)
nrm = torch.where(cap[..., None], torch.tensor([0.0, 0.0, 1.0], device=dev).expand_as(nrm), nrm)
light = torch.tensor([-0.35, 0.55, 0.76], device=dev); light = light / light.norm()
if STYLE == "npr":
    # sdfview's illustration shading (margin.fragment.glsl, shadeSurface): Gooch warm-cool tone,
    # wrapped diffuse (the far side stays at three quarters), a broad faint warm highlight
    albedo = torch.tensor([[0.78, 0.20, 0.20], [0.30, 0.42, 0.72]] if STUB is None else
                          [[0.78, 0.20, 0.20], [0.92, 0.58, 0.22]], device=dev)[cls]
    ndl = (nrm @ light)[..., None]
    t = 0.5 + 0.5 * ndl
    cool = torch.tensor([0.06, 0.10, 0.30], device=dev) + 0.55 * albedo
    warm = torch.tensor([0.20, 0.16, 0.02], device=dev) + 0.90 * albedo
    diffuse = ((ndl + 0.5) / 1.5).clamp(0, 1)
    rgb = (cool + (warm - cool) * t) * (0.75 + 0.35 * diffuse)
    rd = torch.tensor([0.0, 0.0, -1.0], device=dev)
    half = light - rd; half = half / half.norm()
    spec = (14.0 / 8.0) * (nrm @ half).clamp_min(0)[..., None] ** 12 * ndl.clamp_min(0) * 0.12
    rgb = (rgb + spec * torch.tensor([1.0, 0.97, 0.9], device=dev)) * 1.1
    bg = torch.ones(3, device=dev)
else:
    lam = (nrm @ light).clamp_min(0)
    base = torch.tensor([[0.82, 0.18, 0.16], [0.22, 0.42, 0.90]] if STUB is None else
                        [[0.82, 0.18, 0.16], [0.95, 0.60, 0.20]], device=dev)[cls]
    rgb = base * (0.25 + 0.75 * lam[..., None])
    bg = torch.zeros(3, device=dev)
img = torch.where(hit[..., None], rgb, bg.expand_as(rgb)).clamp(0, 1).cpu().numpy()   # (s, u, 3)
if STYLE == "npr":
    # INK from the depth image (sdfview's INK_GLSL, orthographic, thresholds in mm): on the nearer
    # side of every break, 1 px plus a partial 2nd px withheld from thin features, coloured as a
    # dark shade of what it borders; then the depth cue fades the far side toward the background
    dep = np.where(hit.cpu().numpy(), (W - V).cpu().numpy(), -1.0)
    def shifted(a, o):
        out = np.full_like(a, -1.0)
        ys = slice(max(o[0], 0), a.shape[0] + min(o[0], 0)); yd = slice(max(-o[0], 0), a.shape[0] + min(-o[0], 0))
        xs = slice(max(o[1], 0), a.shape[1] + min(o[1], 0)); xd = slice(max(-o[1], 0), a.shape[1] + min(-o[1], 0))
        out[yd, xd] = a[ys, xs]
        return out
    ss = lambda e0, e1, x: np.clip((x - e0) / (e1 - e0), 0, 1) ** 2 * (3 - 2 * np.clip((x - e0) / (e1 - e0), 0, 1))
    edge = np.zeros_like(dep)
    for r in range(1, 2 * SS + 1):                                  # line width in OUTPUT pixels: 1 + a partial 2nd
        k = 0.4 if r > SS else 1.0
        for o in ((r, 0), (0, r), (r, r), (r, -r)):
            a, b = shifted(dep, o), shifted(dep, (-o[0], -o[1]))
            beyond = lambda d: (d < 0) | (d - dep > 0.5)
            thin = (r > SS) & beyond(a) & beyond(b)
            sil = (a < 0) | (b < 0)
            dip = 0.5 * (a + b) - dep
            slope = 0.5 * np.abs(a - b)
            val = np.where(sil, 1.0, ss(0.08, 0.3, dip) * ss(0.4, 0.8, dip / np.maximum(slope, 1e-6)))
            val = np.where(thin | (dep < 0), 0.0, val)
            edge = np.maximum(edge, k * val)
    ink = img * 0.3 + np.array([0.03, 0.025, 0.025])
    img = img + (ink - img) * (0.9 * edge)[..., None]
    live = dep >= 0
    far = np.clip((dep - dep[live].min()) / max(dep[live].max() - dep[live].min(), 1e-3), 0, 1)
    img = np.where(live[..., None], img + (np.array([0.93, 0.93, 0.94]) - img) * (0.45 * far ** 2)[..., None], img)
if SS > 1:                                                          # box-average back to the output grid
    ns_, nu_ = img.shape[0] // SS * SS, img.shape[1] // SS * SS
    img = img[:ns_, :nu_].reshape(ns_ // SS, SS, nu_ // SS, SS, 3).mean((1, 3))
print(f"rendered {img.shape[0]} x {img.shape[1]} rays, {len(vs)} steps each, on {dev} in {time.time() - t0:.0f} s")

# one continuous strip, s left to right, one pixel per 0.1 mm (x2 for legibility)
img = np.transpose(img, (1, 0, 2))[::-1]                              # rows = u, columns = s
scale = 2
H, Wd = img.shape[0] * scale, img.shape[1] * scale
fig = plt.figure(figsize=(Wd / 100, (H + 80) / 100), dpi=100, facecolor=("white" if STYLE == "npr" else "black"))
ax = fig.add_axes([0, 30 / (H + 80), 1, H / (H + 80)])      # 30 px below for ticks, 50 above for the title
ax.imshow(img, extent=(0, img.shape[1] * PX, -W, W), aspect="equal", interpolation="lanczos" if STYLE == "npr" else "nearest")
ax.set_facecolor(("white" if STYLE == "npr" else "black")); ax.tick_params(colors=("0.15" if STYLE == "npr" else "0.85"), labelsize=8)
ax.set_xticks(np.arange(0, img.shape[1] * PX, 10)); ax.set_yticks([])
for sp_ in ax.spines.values(): sp_.set_visible(False)
what = ("arteries red, veins blue" if STUB is None else
        f"the path's artery red, branches orange cut {STUB:g} mm along the tree from where they leave; all else clipped by the graph")
fig.text(0.003, 1 - 12 / (H + 80), f"{patient}: straightened left-lung artery, r {rad[0]:.1f} -> {rad[-1]:.1f} mm over "
         f"{arc[-1]:.0f} mm, rendered from the field ({what}; +-{W:g} mm of the axis; ticks every 10 mm)",
         color=("0.15" if STYLE == "npr" else "0.85"), fontsize=9, va="top")
out = DATA / f"{patient}_straight3d{'' if STUB is None else f'_stub{STUB:g}'}_w{W:g}{'_npr' if STYLE == 'npr' else ''}.png"
plt.savefig(out, dpi=110, facecolor=("white" if STYLE == "npr" else "black"))
print(f"wrote {out}")
