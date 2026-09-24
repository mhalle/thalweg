"""Flow extensions as implicit geometry: the CFD domain as one field, no mesh surgery.

    python bench/vessels/flow_ext.py [RUN] prep      (choose outlets; then vmtk_flowext.py)
    python bench/vessels/flow_ext.py [RUN]           (build the field domain, compare with vmtk)

vmtk's flow extensions (vtkvmtkPolyDataFlowExtensionsFilter) need an OPEN surface: clip each
outlet, extract the boundary ring, then extrude a cylinder along the ring's normal, radius the
ring's mean distance from its barycenter, length ExtensionRatio x that radius, morphing the ring
into the circle over TransitionRatio of the length. The same domain as a signed field (mm,
positive inside), all closed-form, evaluable anywhere:

    vessel, cut      phi_v(x), and past each outlet plane inside its ball: min(phi_v, -s); then only
                     the main lattice component (the vessel past each cut is a separate remnant)
    extension k      s = n.(x - c), (u, v) in the plane; w = clamp(s / (T L), 0, 1)
                     phi_k = (1 - w) phi_sec(u, v) + w (R - |(u, v) - b|), capped: min(., L - s, s + 0.2)
    domain           max(vessel cut, phi_1, ..., phi_K)

phi_sec is the outlet cross-section's own 2-D signed distance (the field sampled on the cut
plane, 0.05 mm), b and R the ring's barycenter and mean radius (vmtk's definitions, from the
section's zero contour). On the lattice each piece is evaluated only where it can change the
result: the vessel's trilinear interpolant where a field-cell corner is above the clip floor
(elsewhere it is the floor exactly), each cut inside its ball, each extension inside its own box.
Identical to evaluating everything everywhere (float32 rounding), 11x faster. The domain is watertight by construction, with a flat cap per outlet for
the boundary condition. Checked against vmtk's extended surface: signed distance from each vmtk
extension vertex to the field domain's zero set, in the transition and in the cylinder.

Outlets: every terminal branch at least 8 mm long, cut 3 mm before its end (radius >= 0.9 mm there,
the 8 widest), plus the inlet 3 mm into the root branch; the cut ball is 1.6 x the radius.
"""
import sys, time
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes, find_contours
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _data import DATA
from _field import fine_field
from cases import store

run = sys.argv[1] if len(sys.argv) > 1 else "C3N-00704_ctpa0625"
mode = sys.argv[2] if len(sys.argv) > 2 else "compare"
RATIO, TRANSITION, CUT, BALL, H2 = 5.0, 0.25, 3.0, 1.6, 0.05
Z = np.load(DATA / f"{run}_vmtk_input.npz")
M = np.load(DATA / f"{run}_vmtk_mapping.npz")

if mode == "prep":
    cells = np.split(M["cell_ids"], np.cumsum(M["cell_len"])[:-1])
    P, R = M["cl_points"], M["cl_radius"]
    cand = {}
    for cl in np.unique(M["cell_centerline"]):
        on = np.nonzero(M["cell_centerline"] == cl)[0]
        last = on[np.argmax(M["cell_tract"][on])]
        if M["cell_blank"][last] == 0:
            cand[M["cell_group"][last]] = cells[last]
    out = []
    for g, ids in cand.items():
        p = P[ids]; s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        if s[-1] < 8.0: continue
        k = int(np.argmin(np.abs(s - (s[-1] - CUT))))
        t = p[min(k + 3, len(p) - 1)] - p[max(k - 3, 0)]
        if R[ids[k]] >= 0.9:
            out.append((R[ids[k]], p[k], t / np.linalg.norm(t), g))
    out = sorted(out, key=lambda o: -o[0])[:8]
    first = cells[np.nonzero((M["cell_centerline"] == 0) & (M["cell_tract"] == 0))[0][0]]
    p = P[first]; s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    k = int(np.argmin(np.abs(s - CUT))); t = p[min(k + 3, len(p) - 1)] - p[max(k - 3, 0)]
    out.insert(0, (R[first[k]], p[k], -t / np.linalg.norm(t), M["cell_group"][0]))     # the inlet: normal points upstream
    r = np.array([o[0] for o in out])
    np.savez(DATA / f"{run}_flowext_outlets.npz", center=np.array([o[1] for o in out]), normal=np.array([o[2] for o in out]),
             radius=r, ball=BALL * r, group=np.array([o[3] for o in out]), ratio=RATIO, transition=TRANSITION)
    print(f"{len(out)} cuts (inlet + {len(out) - 1} outlets), radius {np.round(r, 2)} mm; now run vmtk_flowext.py")
    sys.exit()

# the vessel as a signed field in mm: the region field vmtk's surface came from, over its wall slope
_, _, grid, _ = fine_field(store(run))
F, lo, clip = Z["region_field"].astype(np.float32), Z["region_lo"], float(Z["clip"])
g = np.gradient(F)
at_idx = lambda a, idx, cval: ndi.map_coordinates(a, idx.T, order=1, mode="constant", cval=cval)
vi = grid.to_index(Z["verts"]) - lo
slope = float(np.median(np.linalg.norm(np.stack([at_idx(a, vi, 0) for a in g], 1) @ np.linalg.inv(grid.dirs).T, axis=1)))
phi_v = lambda x: at_idx(F, grid.to_index(x) - lo, -clip) / slope
O = np.load(DATA / f"{run}_flowext_outlets.npz")
V = np.load(DATA / f"{run}_vmtk_flowext.npz")

t0 = time.time()
ext = []
for c, n, rho in zip(O["center"], O["normal"], O["ball"]):
    u = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0]); u /= np.linalg.norm(u); v = np.cross(n, u)
    ax = np.arange(-rho, rho + 1e-9, H2)
    U, W = np.meshgrid(ax, ax, indexing="ij")
    sec = phi_v(c + U[..., None].reshape(-1, 1) * u + W.reshape(-1, 1) * v).reshape(U.shape)
    inside = (sec > 0) & (U ** 2 + W ** 2 < rho ** 2)
    lab, _ = ndi.label(inside); inside = lab == lab[len(ax) // 2, len(ax) // 2]
    near = ndi.binary_dilation(inside, iterations=3)                 # the true field next to the wall, -1 beyond
    ring = max(find_contours(np.where(near, sec, -1.0), 0.0), key=len) * H2 - rho            # (u, v) in mm
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(ring, axis=0), axis=1))]
    dense_ring = np.stack([np.interp(np.arange(0, d[-1], 0.005), d, ring[:, a]) for a in range(2)], 1)
    sdf = (cKDTree(dense_ring).query(np.stack([U.ravel(), W.ravel()], 1))[0] * np.where(inside.ravel(), 1, -1)).reshape(U.shape)
    ring = np.stack([np.interp(np.linspace(0, d[-1], 200, endpoint=False), d, ring[:, a]) for a in range(2)], 1)
    b = ring.mean(0); R = np.linalg.norm(ring - b, axis=1).mean()
    ext.append(dict(c=c, n=n, u=u, v=v, rho=rho, ax=ax, sdf=sdf, b=b, R=R, L=RATIO * R))


def phi_ext(e, x):
    s = (x - e["c"]) @ e["n"]; uu = (x - e["c"]) @ e["u"]; vv = (x - e["c"]) @ e["v"]
    q = np.stack([(uu - e["ax"][0]) / H2, (vv - e["ax"][0]) / H2], 1)
    ps = ndi.map_coordinates(e["sdf"], q.T, order=1, mode="constant", cval=-e["rho"])
    pc = e["R"] - np.hypot(uu - e["b"][0], vv - e["b"][1])
    w = np.clip(s / (TRANSITION * e["L"]), 0, 1)
    return np.minimum.reduce([(1 - w) * ps + w * pc, e["L"] - s, s + 0.2])


t_build = time.time() - t0
print(f"field domain: {len(ext)} cuts, wall slope {slope:.2f} logit/mm, sections + rings in {t_build:.2f} s")

# rings: vmtk's (mesh boundary) vs ours (the section's zero contour), matched by barycenter
ours_b = np.array([e["c"] + e["b"][0] * e["u"] + e["b"][1] * e["v"] for e in ext]); ours_R = np.array([e["R"] for e in ext])
dB, j = cKDTree(V["ring_barycenter"]).query(ours_b)
print(f"rings: vmtk found {len(V['ring_barycenter'])} boundaries for {len(ext)} cuts; barycenter distance median "
      f"{np.median(dB):.3f} mm (max {dB.max():.3f}); mean radius vmtk - ours median {np.median(V['ring_mean_radius'][j] - ours_R):+.3f} mm "
      f"(|max| {np.abs(V['ring_mean_radius'][j] - ours_R).max():.3f})")

# the domain as a field on a 0.3 mm lattice: the cut vessel, only its main piece (vmtk's connectivity
# step: the vessel past each cut is a separate remnant), then the union with the extensions
h = 0.3
allpts = np.vstack([Z["verts"]] + [e["c"] + e["L"] * e["n"] for e in ext])
b0, b1 = allpts.min(0) - 3, allpts.max(0) + 3
axes = [np.arange(b0[a], b1[a], h) for a in range(3)]
shape = tuple(len(a) for a in axes)


def box(lo_w, hi_w):
    """The lattice index slices covering a world-space box."""
    i0 = np.clip(np.floor((lo_w - b0) / h).astype(int), 0, shape)
    i1 = np.clip(np.ceil((hi_w - b0) / h).astype(int) + 1, 0, shape)
    return tuple(slice(a, b) for a, b in zip(i0, i1))


def points(sl):
    return np.stack(np.meshgrid(*[a[s_] for a, s_ in zip(axes, sl)], indexing="ij"), -1).reshape(-1, 3)


t0 = time.time()
# the vessel: trilinear only where some corner of the enclosing field cell is above the clip floor;
# everywhere else the interpolant IS the floor, exactly
G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
idx = grid.to_index(G) - lo
cell = np.floor(idx).astype(int)
live = ndi.binary_dilation(F > -clip, structure=np.ones((3, 3, 3), bool))   # any of the cell's 8 corners
inb = np.all((cell >= 0) & (cell < np.array(F.shape)), 1)
sel = np.zeros(len(G), bool); sel[inb] = live[tuple(cell[inb].T)]
vc = np.full(len(G), -clip / slope, np.float32)
vc[sel] = at_idx(F, idx[sel], -clip) / slope
vc = vc.reshape(shape); del G, idx, cell
# each cut acts only inside its ball
for e in ext:
    sl = box(e["c"] - e["rho"], e["c"] + e["rho"])
    x = points(sl); s_ = (x - e["c"]) @ e["n"]
    cut = ((s_ > 0) & (np.linalg.norm(x - e["c"], axis=1) < e["rho"])).reshape(vc[sl].shape)
    vc[sl] = np.where(cut, np.minimum(vc[sl], -s_.reshape(cut.shape)), vc[sl])
lab, nlab = ndi.label(vc > 0)
main = np.bincount(lab.ravel())[1:].argmax() + 1
vc = np.where((lab == main) | (lab == 0), vc, -h)
# each extension only inside its own oriented box (outside it the extension is negative and never wins)
vol = vc
for e in ext:
    corners = np.array([e["c"] + a * e["n"] + b * e["u"] + c_ * e["v"] for a in (-0.5, e["L"] + 0.5)
                        for b in (-e["rho"] - 0.5, e["rho"] + 0.5) for c_ in (-e["rho"] - 0.5, e["rho"] + 0.5)])
    sl = box(corners.min(0), corners.max(0))
    vol[sl] = np.maximum(vol[sl], phi_ext(e, points(sl)).reshape(vol[sl].shape))
t_vol = time.time() - t0
lab2, n2 = ndi.label(vol > 0)
_t = time.time()
mv, mf, _, _ = marching_cubes(vol, 0.0, spacing=(h, h, h))
t_mesh = time.time() - _t
edges = np.sort(np.vstack([mf[:, [0, 1]], mf[:, [1, 2]], mf[:, [2, 0]]]), 1)
_, cnt = np.unique(edges, axis=0, return_counts=True)
chi = len(mv) - len(cnt) + len(mf)
ze = np.sort(np.vstack([Z["faces"][:, [0, 1]], Z["faces"][:, [1, 2]], Z["faces"][:, [2, 0]]]), 1)
chi0 = len(Z["verts"]) - len(np.unique(ze, axis=0)) + len(Z["faces"])
print(f"field domain on a {h} mm lattice ({vc.size / 1e6:.0f} M points) in {t_vol:.1f} s: the cut vessel fell into {nlab} pieces, "
      f"{nlab - 1} distal remnants dropped; domain {n2} component(s), meshed to {len(mv)} vertices, every edge shared by two "
      f"triangles: {bool((cnt == 2).all())}; Euler characteristic {chi} (the uncut surface vmtk started from: {chi0})")
np.savez(DATA / f"{run}_flowext_field.npz", verts=mv + b0, faces=mf)
phi_dom = lambda x: ndi.map_coordinates(vol, ((x - b0) / h).T, order=1, mode="constant", cval=-1.0)

# vmtk's extension vertices (past the cut plane): the field domain's value there (mm), by zone
X = V["verts"]
own = np.argmin(np.stack([np.linalg.norm(X - e["c"], axis=1) for e in ext], 1), 1)
_t = time.time(); ph = phi_dom(X); t_eval = time.time() - _t
rows = []
for k, e in enumerate(ext):
    sel = own == k
    s = (X[sel] - e["c"]) @ e["n"]
    zone = np.where(s < 0.05, -1, np.where(s < TRANSITION * e["L"], 0, np.where(s < e["L"] - 0.3, 1, 2)))
    rows.append((ph[sel], zone))
for z, name in ((0, "transition"), (1, "cylinder")):
    d = np.concatenate([p[zz == z] for p, zz in rows])
    print(f"  vmtk extension vertices in the {name}: {len(d)}; field domain value there (mm, + = inside the field domain) "
          f"median {np.median(d):+.3f}, |.| median {np.median(np.abs(d)):.3f}, p90 {np.percentile(np.abs(d), 90):.3f}")
print(f"  vmtk: clip {V['seconds'][0]:.1f} s + extensions {V['seconds'][1]:.1f} s")

# figure: longitudinal slices through three extensions, field domain contour vs vmtk mesh section
pick = [0] + list(np.argsort([-e["R"] for e in ext[1:]])[:2] + 1)
fig, axs = plt.subplots(1, len(pick), figsize=(5 * len(pick), 4.4))
for a_, k in zip(axs, pick):
    e = ext[k]
    ss = np.arange(-3 * e["R"] - 2, e["L"] + 1.5, 0.04); uu = np.arange(-e["rho"] - 1.5, e["rho"] + 1.5, 0.04)
    Sg, Ug = np.meshgrid(ss, uu, indexing="ij")
    Xs = e["c"] + Sg[..., None] * e["n"] + Ug[..., None] * e["u"]
    a_.contour(ss, uu, phi_dom(Xs.reshape(-1, 3)).reshape(Sg.shape).T, [0], colors="C0", linewidths=1.6)
    a_.contour(ss, uu, phi_v(Xs.reshape(-1, 3)).reshape(Sg.shape).T, [0], colors="0.7", linewidths=0.8, linestyles="--")
    # the vmtk surface cut by the same plane (normal v through c)
    Vv, Ff = V["verts"], V["faces"]
    dv = (Vv - e["c"]) @ e["v"]
    tri = dv[Ff]; cross = (tri.min(1) < 0) & (tri.max(1) > 0)
    for f_ in Ff[cross]:
        pts = []
        for i0, i1 in ((0, 1), (1, 2), (2, 0)):
            a0, a1 = dv[f_[i0]], dv[f_[i1]]
            if (a0 < 0) != (a1 < 0):
                p_ = Vv[f_[i0]] + a0 / (a0 - a1) * (Vv[f_[i1]] - Vv[f_[i0]])
                pts.append([(p_ - e["c"]) @ e["n"], (p_ - e["c"]) @ e["u"]])
        if len(pts) == 2:
            pts = np.array(pts)
            if np.all((pts[:, 0] > ss[0]) & (pts[:, 0] < ss[-1]) & (np.abs(pts[:, 1]) < uu[-1])):
                a_.plot(pts[:, 0], pts[:, 1], color="C3", lw=0.9)
    a_.axvline(0, color="0.6", lw=0.6, ls=":"); a_.axvline(TRANSITION * e["L"], color="0.6", lw=0.6, ls=":")
    a_.set_aspect("equal"); a_.set_xlabel("s along the outlet normal (mm)"); a_.set_ylabel("u (mm)")
    a_.set_title(f"{'inlet' if k == 0 else f'outlet {k}'}: R {e['R']:.2f} mm, L {e['L']:.1f} mm", fontsize=9)
axs[0].plot([], [], color="C0", label="field domain (zero set)"); axs[0].plot([], [], color="0.7", ls="--", label="vessel before the cut")
axs[0].plot([], [], color="C3", label="vmtk flow extension (mesh)")
axs[0].legend(fontsize=8, loc="lower left")
plt.tight_layout(); plt.savefig(DATA / f"{run}_flowext.png", dpi=120)
print(f"wrote {DATA / f'{run}_flowext.png'}")
print("TIMING " + __import__("json").dumps({k: round(float(v), 4) for k, v in ({"flowext_sections": t_build, "flowext_domain_lattice": t_vol, "flowext_domain_mesh": t_mesh, "flowext_eval_points": t_eval}).items()}))
