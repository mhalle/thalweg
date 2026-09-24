"""Wall curvature from the field (level-set mean curvature) vs vmtk's mesh curvature.

    python bench/vessels/curvature.py [RUN] phantoms     (writes phantom meshes for vmtk)
    uv run --no-project --python 3.12 --with vmtk --with scipy python bench/vessels/vmtk_curvature.py [RUN]
    python bench/vessels/curvature.py [RUN] repeat       (writes the other reconstructions' meshes)
    python bench/vessels/curvature.py [RUN]              (compares)

vmtk's vmtkSurfaceCurvature is vtkCurvatures on the mesh: a discrete estimate per vertex, known to
be noisy on marching-cubes surfaces. The field gives the same quantity without a mesh: the mean
curvature of the zero level set, H = -1/2 div(grad m / |grad m|) (m > 0 inside, so the outward
normal is -grad m / |grad m|). Convention as vtkCurvatures: H = (k1 + k2) / 2, positive for
convex, so a tube of radius r has H = 1/(2r) and a sphere H = 1/r. Two field estimators:
  fd      central differences of central differences (a +-2 voxel stencil), sampled trilinearly;
  quad    a Gaussian-weighted quadratic fit to the UNCLIPPED lattice samples within 2.8 mm of the
          point, curvature of the fitted quadric's level set there.
The margin is steep (about 10 logit/mm) and clipped at +-8, so it saturates within ~0.75 mm of
the wall: any stencil that reaches the plateau reads the curvature low. The fit leaves the
plateau out, which is why it is the one to use.

1. Phantoms on the real grid's spacing, oblique to the axes: tubes (r 0.75..3 mm) and spheres, the
   margin a clipped ramp with the real field's wall slope. Truth is analytic; both estimators are
   run on the same thing (vmtk on the marching-cubes mesh of the phantom field).
2. The real subtree: field H vs vmtk H at the mesh vertices, and both against the tube proxy
   1/(2 r_wall) (r_wall = vmtk's DistanceToCenterlines) in branch interiors - a proxy, since axial
   curvature and ellipticity are ignored, so the SCATTER about it is the measure, not the bias.
3. Repeatability: other reconstructions of the same acquisition (REPEATS), each run through the
   model on its own grid. At each interior wall point of RUN, the nearest wall vertex of the other
   run's own marching-cubes mesh (within 0.5 mm); each estimator on each run's own data there.
   The spread of the difference is the estimator's noise (anatomy is the same).
"""
import sys
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
from _data import DATA
from _field import fine_field, sample, Grid
from cases import store

run = sys.argv[1] if len(sys.argv) > 1 else "C3N-00704_ctpa0625"
mode = sys.argv[2] if len(sys.argv) > 2 else "compare"
REPEATS = {"C3N-00704_ctpa0625": ["C3N-00704_lung125", "C3N-00704_slab2mm"]}.get(run, [])


def field_curvature(m, grid, pts, sigma_mm=0.0):
    """Mean curvature (k1 + k2)/2 of the level set of m through each world point (vtkCurvatures sign)."""
    idx = grid.to_index(pts)
    lo = np.maximum(np.floor(idx.min(0)).astype(int) - 6, 0)
    hi = np.minimum(np.ceil(idx.max(0)).astype(int) + 7, m.shape)
    f = m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float64)
    if sigma_mm > 0:
        f = ndi.gaussian_filter(f, sigma_mm / grid.spacing)
    g = np.gradient(f)                                               # d/d index
    Hs = [[None] * 3 for _ in range(3)]
    for i in range(3):
        gi = np.gradient(g[i])
        for j in range(i, 3):
            Hs[i][j] = Hs[j][i] = gi[j]
    q = (idx - lo).T
    at = lambda a: ndi.map_coordinates(a, q, order=1, mode="nearest")
    gi = np.stack([at(g[i]) for i in range(3)], 1)                   # (N, 3) index-space gradient
    Hi = np.stack([np.stack([at(Hs[i][j]) for j in range(3)], 1) for i in range(3)], 1)   # (N, 3, 3)
    Dinv = np.linalg.inv(grid.dirs)                                  # world = idx @ D + o
    gw = gi @ Dinv.T                                                 # g_w = D^-1 g_idx
    Hw = np.einsum("ab,nbc,dc->nad", Dinv, Hi, Dinv)                 # H_w = D^-1 H_idx D^-T
    n2 = (gw * gw).sum(1)
    div = (np.trace(Hw, axis1=1, axis2=2) * n2 - np.einsum("na,nab,nb->n", gw, Hw, gw)) / np.maximum(n2, 1e-12) ** 1.5
    return -0.5 * div


QUAD_RAD = float(__import__("os").environ.get("QUAD_RAD", 2.8))


def quad_curvature(m, grid, pts, clip, rad=None):
    """Mean curvature of the level set of a weighted local quadratic fit to the unclipped samples of m."""
    rad = rad or QUAD_RAD
    idx = grid.to_index(pts)
    lo = np.maximum(np.floor(idx.min(0)).astype(int) - 4, 0)
    hi = np.minimum(np.ceil(idx.max(0)).astype(int) + 5, m.shape)
    f = m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    L = np.argwhere(np.abs(f) < clip - 1e-3)
    W = grid.to_world(L + lo); vals = f[tuple(L.T)].astype(np.float64)
    tree = cKDTree(W); out = np.full(len(pts), np.nan)
    for n, nb in enumerate(tree.query_ball_point(pts, rad)):
        if len(nb) < 12:
            continue
        d = W[nb] - pts[n]; w = np.sqrt(np.exp(-(d ** 2).sum(1) / (rad / 2) ** 2))
        A = np.c_[np.ones(len(d)), d, d ** 2, d[:, 0] * d[:, 1], d[:, 0] * d[:, 2], d[:, 1] * d[:, 2]]
        c = np.linalg.lstsq(A * w[:, None], vals[nb] * w, rcond=None)[0]
        g = c[1:4]; H = np.array([[2 * c[4], c[7], c[8]], [c[7], 2 * c[5], c[9]], [c[8], c[9], 2 * c[6]]])
        n2 = g @ g
        out[n] = -0.5 * (np.trace(H) * n2 - g @ H @ g) / n2 ** 1.5
    return out


def wall_slope(m, grid, pts):
    """|grad m| in logits per mm at world points (central differences, trilinear)."""
    g = np.gradient(m.astype(np.float32))
    gi = np.stack([sample(a, grid, pts) for a in g], 1)
    return np.linalg.norm(gi @ np.linalg.inv(grid.dirs).T, axis=1)


margins, _, grid, clip = fine_field(store(run))
m = margins["lung_arteries"]
Z = np.load(DATA / f"{run}_vmtk_input.npz")
verts, faces = Z["verts"], Z["faces"]
slope = float(np.median(wall_slope(m, grid, verts)))

# phantoms: a grid with the real spacing, axis-aligned, shapes oblique to it
sp = grid.spacing
PG = Grid(origin=np.zeros(3), dirs=np.diag(sp))
shape = np.ceil(np.array([30.0, 30.0, 30.0]) / sp).astype(int)
X = PG.to_world(np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1).reshape(-1, 3))
ctr = np.array([15.0, 15.0, 15.0]) + 0.123
axis = np.array([1.0, 0.62, 0.37]); axis /= np.linalg.norm(axis)
PH = []
for kind, r in [("tube", 0.75), ("tube", 1.0), ("tube", 1.5), ("tube", 2.0), ("tube", 3.0), ("sphere", 2.0), ("sphere", 4.0)]:
    v = X - ctr
    d = np.linalg.norm(v - (v @ axis)[:, None] * axis, axis=1) if kind == "tube" else np.linalg.norm(v, axis=1)
    f = np.clip(slope * (r - d), -clip, clip).reshape(shape).astype(np.float32)
    vv, ff, _, _ = marching_cubes(f, level=0.0)
    if np.linalg.det(grid.dirs) < 0:                              # same winding-in-world as the real mesh
        ff = ff[:, ::-1]
    w = PG.to_world(vv)
    keep = np.abs((w - ctr) @ axis) < 8.0 if kind == "tube" else np.ones(len(w), bool)   # tubes: away from the box
    PH.append(dict(kind=kind, r=r, f=f, verts=w, faces=ff, keep=keep, truth=1 / (2 * r) if kind == "tube" else 1 / r))

def repeat_mesh(rep):
    """The other run's artery margin, marching cubes in RUN's subtree box (+3 mm), world coordinates."""
    mB, _, gB, _ = fine_field(store(rep))
    mB = mB["lung_arteries"]
    corners = np.array(np.meshgrid(*[[0, 1]] * 3, indexing="ij")).reshape(3, -1).T
    box = verts.min(0) - 3 + corners * (verts.max(0) - verts.min(0) + 6)
    ib = gB.to_index(box)
    lo = np.maximum(np.floor(ib.min(0)).astype(int), 0); hi = np.minimum(np.ceil(ib.max(0)).astype(int) + 1, mB.shape)
    vv, ff, _, _ = marching_cubes(mB[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], level=0.0)
    return mB, gB, gB.to_world(vv + lo), ff


if mode == "repeat":
    for rep in REPEATS:
        _, _, vB, fB = repeat_mesh(rep)
        np.savez(DATA / f"{run}_curv_repeat_{rep}.npz", verts=vB, faces=fB)
        print(f"{rep}: {len(vB)} vertices in the subtree box")
    sys.exit()

if mode == "phantoms":
    np.savez(DATA / f"{run}_curv_phantoms.npz", n=len(PH),
             **{f"p{i}_{k}": p[k] for i, p in enumerate(PH) for k in ("verts", "faces")})
    print(f"wrote {len(PH)} phantom meshes (wall slope {slope:.2f} logit/mm, clip {clip}); now run vmtk_curvature.py")
    sys.exit()

V = np.load(DATA / f"{run}_vmtk_curvature.npz")
print(f"wall slope {slope:.2f} logit/mm (median over the subtree wall), clip {clip}, grid spacing {np.round(sp, 3)} mm")
print("phantoms: estimate / truth at wall vertices, median [p10, p90]")
for i, p in enumerate(PH):
    t = p["truth"]; k = p["keep"]
    row = f"  {p['kind']:6s} r {p['r']:.2f} mm (H {t:.3f}/mm): "
    for name, h in (("fd", field_curvature(p["f"], PG, p["verts"][k])), ("quad", quad_curvature(p["f"], PG, p["verts"][k], clip))):
        h = h / t
        row += f"{name} {np.nanmedian(h):.3f} [{np.nanpercentile(h, 10):.2f}, {np.nanpercentile(h, 90):.2f}]  "
    hv = V[f"p{i}_mean"][k] / t
    row += f"vmtk {np.median(hv):.3f} [{np.percentile(hv, 10):.2f}, {np.percentile(hv, 90):.2f}]"
    print(row)

# the real subtree, in branch interiors, against the tube proxy
M = np.load(DATA / f"{run}_vmtk_mapping.npz")
SP, SG = M["surf_points"], M["surf_group"]
orig = cKDTree(verts).query(SP)[0] < 1e-4
cells = np.split(M["cell_ids"], np.cumsum(M["cell_len"])[:-1])
ends = {}
for c, g, b in zip(cells, M["cell_group"], M["cell_blank"]):
    if b == 0:
        a = M["cl_abscissa"][c]; lo_, hi_ = ends.get(g, (np.inf, -np.inf)); ends[g] = (min(lo_, a.min()), max(hi_, a.max()))
inner = orig & np.array([g in ends and ends[g][0] + 1.5 < a < ends[g][1] - 1.5 for g, a in zip(SG, M["surf_abscissa"])])
P = SP[inner]
proxy = 1 / (2 * M["surf_dist"][inner])
vi = cKDTree(verts).query(P)[1]
hv = V["real_mean"][vi]
print(f"real subtree: {inner.sum()} wall vertices in branch interiors; estimate / tube proxy 1/(2 r_wall), median [p10, p90], "
      f"and the rank correlation with the proxy")
from scipy.stats import spearmanr
import time
t0 = time.time(); res = {"fd": field_curvature(m, grid, P)}; t_fd = time.time() - t0
t0 = time.time(); res["quad"] = quad_curvature(m, grid, P, clip); t_q = time.time() - t0
res["vmtk"] = hv
for name, h in res.items():
    ok = np.isfinite(h); q = h[ok] / proxy[ok]
    print(f"  {name:5s}: {np.median(q):.3f} [{np.percentile(q, 10):.2f}, {np.percentile(q, 90):.2f}], rho {spearmanr(h[ok], proxy[ok])[0]:.3f}")
ok = np.isfinite(res["quad"])
print("TIMING " + __import__("json").dumps({k: round(float(v), 4) for k, v in ({"curvature_fd": t_fd, "curvature_quad": t_q, "curvature_points": len(P)}).items()}))
print(f"  quad vs vmtk: rho {spearmanr(res['quad'][ok], hv[ok])[0]:.3f}; times: fd {t_fd:.1f} s, quad {t_q:.1f} s for {len(P)} points")
np.savez(DATA / f"{run}_curvature.npz", points=P, proxy=proxy, **res)

print("repeatability: |H_run - H_repeat| at the same wall place, median and p90 (1/mm), and relative to the median proxy H; "
      "Spearman rho between runs")
Hscale = np.median(proxy)
for rep in REPEATS:
    mB, gB, _, _ = repeat_mesh(rep)
    vB = V[f"repeat_{rep}_verts"]
    dB, jB = cKDTree(vB).query(P)
    ok = dB < 0.5
    PB = vB[jB[ok]]
    pairs = {"fd": (res["fd"][ok], field_curvature(mB, gB, PB)),
             "quad": (res["quad"][ok], quad_curvature(mB, gB, PB, clip)),
             "vmtk": (hv[ok], V[f"repeat_{rep}_mean"][jB[ok]])}
    print(f"  {rep}: {ok.sum()} of {len(P)} points matched within 0.5 mm (median {np.median(dB[ok]):.2f} mm)")
    for name, (a, b) in pairs.items():
        f_ = np.isfinite(a) & np.isfinite(b); d = np.abs(a[f_] - b[f_])
        print(f"    {name:5s}: |diff| median {np.median(d):.3f} ({np.median(d) / Hscale:.0%}), p90 {np.percentile(d, 90):.3f}; "
              f"rho {spearmanr(a[f_], b[f_])[0]:.3f}")
