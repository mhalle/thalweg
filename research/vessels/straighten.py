"""Cross-sections with intervals, and the straightened vessel, along one path of the graph.

    python bench/vessels/straighten.py [PATIENT] [THICK_RUN]

The path: root -> the farthest tip in the patient's LEFT lung, through the segment graph of the
thin reference run (centerline.py). Along it, every 0.5 mm:
- a smoothed centerline (cubic spline, smoothing tied to the radius) and parallel-transport
  frames (n1, n2 rotated along the tangent without twist);
- a cross-section: the artery margin sampled on the normal disk at 0.1 mm, the component
  holding the center; area at margin levels -2, -1, 0, +1, +2 logits - the model's own interval;
- the same at the same stations from a THICK variant's store (same patient, same world points);
- the image's radius from psf_image.py at the nearest centerline point (thin series);
- the straightened field: two orthogonal longitudinal cuts of the CT, with the margin's
  0 and +-2 contours.
Writes a figure and DATA/<PATIENT>_straight.npz.
"""
import json, sys
import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _data import DATA
from _field import fine_field, sample, Image
from cases import LADDERS, store, image

patient = sys.argv[1] if len(sys.argv) > 1 else "C3N-00704"
ladder = LADDERS[patient]
ref = ladder[0][0]
thick = sys.argv[2] if len(sys.argv) > 2 else ladder[-1][0]
G = json.load(open(DATA / f"{ref}_lung_arteries_centerlines.json"))
N, S = G["nodes"], G["segments"]
seg_into = {s["b"]: s["id"] for s in S}
# farthest tip in the left lung by path length
def chain_to(node):
    segs = []
    while node in seg_into:
        s = S[seg_into[node]]; segs.append(s["id"]); node = s["a"]
    return segs[::-1]
root_x = G["root"][0]
best, best_len = None, 0
for n in N:
    if n["kind"] != "tip" or n["point"][0] < root_x + 30:
        continue
    ch = chain_to(n["id"]); L = sum(S[j]["length_mm"] for j in ch)
    if L > best_len:
        best, best_len = ch, L
pts = np.concatenate([np.array(S[j]["points"])[(1 if i else 0):] for i, j in enumerate(best)])
rad = np.concatenate([np.array(S[j]["radius"])[(1 if i else 0):] for i, j in enumerate(best)])
print(f"{patient}: path of {len(best)} segments, {best_len:.0f} mm, radius {rad.max():.1f} -> {rad[-1]:.1f} mm")

# smooth centerline, resampled every 0.5 mm: weights 1 / (0.15 r) with s = n hold the deviation
# to ~0.15 r LOCALLY (one global budget let the 19 mm trunk's allowance cut distal corners)
keep = np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-6]
tck, u = splprep(pts[keep].T, w=1.0 / (0.15 * rad[keep] + 0.05), s=float(keep.sum()), k=3)
fine = np.array(splev(np.linspace(0, 1, 4000), tck)).T
arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(fine, axis=0), axis=1))]
sgrid = np.arange(0, arc[-1], 0.5)
C = np.stack([np.interp(sgrid, arc, fine[:, a]) for a in range(3)], 1)
T = np.gradient(C, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True)
r_path = np.interp(sgrid, np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))], rad)
# parallel transport frames
n1 = np.zeros_like(T); n2 = np.zeros_like(T)
a0 = np.array([0, 0, 1.0]) if abs(T[0, 2]) < 0.9 else np.array([1.0, 0, 0])
n1[0] = np.cross(T[0], a0); n1[0] /= np.linalg.norm(n1[0])
for i in range(1, len(T)):
    v = n1[i - 1] - (n1[i - 1] @ T[i]) * T[i]
    n1[i] = v / np.linalg.norm(v)
n2 = np.cross(T, n1)

LEVELS = (-2.0, -1.0, 0.0, 1.0, 2.0)
def sections(margin, grid):
    """Areas (mm^2) of the center component at each margin level, per station."""
    out = np.full((len(C), len(LEVELS)), np.nan)
    for i in range(len(C)):
        R = 1.6 * r_path[i] + 2.0
        g = np.arange(-R, R + 1e-9, 0.1)
        U, V = np.meshgrid(g, g, indexing="ij")
        P = C[i] + U[..., None] * n1[i] + V[..., None] * n2[i]
        v = sample(margin, grid, P.reshape(-1, 3)).reshape(U.shape)
        c = len(g) // 2
        for j, lv in enumerate(LEVELS):
            lab, _ = ndi.label(v > lv)
            if lab[c, c]:
                out[i, j] = (lab == lab[c, c]).sum() * 0.01
            else:                                              # the center is outside at this level
                out[i, j] = 0.0
    return out

mt, _, gt, _ = fine_field(store(ref))
A_thin = sections(mt["lung_arteries"], gt)
mk, _, gk, _ = fine_field(store(thick))
A_thick = sections(mk["lung_arteries"], gk)
# image radius from psf_image.py at the nearest reference centerline point (thin series)
L = np.load(DATA / f"{patient}_ladder.npz"); Pp = np.load(DATA / f"{patient}_psf.npz")
lp = L["arteries/points"]; dist, j = cKDTree(lp).query(C)
# the image radius is only defined where the center contrast has not saturated (r <~ 2.5 mm
# here: the inversion caps at f = 0.98); above that the model's radius is the measurement
valid_img = (dist < 1.0) & (r_path < 2.5)
a_img = np.where(valid_img, Pp[f"arteries/a:{ref}"][j], np.nan)
a_img_thick = np.where(valid_img, Pp[f"arteries/a:{thick}"][j], np.nan)
req = lambda A: np.sqrt(A / np.pi)

# straightened CT: two orthogonal longitudinal cuts, +-W mm
img = Image(image(ladder[0][1]))
W = min(12.0, 1.6 * r_path.max() + 3)
w = np.arange(-W, W + 1e-9, 0.1)
cuts = []
for n in (n1, n2):
    Q = C[:, None, :] + w[None, :, None] * n[:, None, :]
    cuts.append((img(Q.reshape(-1, 3)).reshape(len(C), len(w)),
                 sample(mt["lung_arteries"], gt, Q.reshape(-1, 3)).reshape(len(C), len(w))))

fig, ax = plt.subplots(4, 1, figsize=(15, 13), gridspec_kw=dict(height_ratios=[1, 1, 1.3, 1.3]))
for k, (ct, mm) in enumerate(cuts):
    ax[k].imshow(ct.T, cmap="gray", vmin=-900, vmax=500, aspect="auto", origin="lower",
                 extent=(sgrid[0], sgrid[-1], -W, W))
    ax[k].contour(sgrid, w, mm.T, levels=[0.0], colors="r", linewidths=0.9)
    ax[k].contour(sgrid, w, mm.T, levels=[-2.0, 2.0], colors="orange", linewidths=0.5, linestyles="--")
    ax[k].set_ylabel(f"{'n1' if k == 0 else 'n2'} (mm)")
    ax[k].set_title(f"{patient} straightened CT (thin series), cut along {'n1' if k == 0 else 'n2'}: "
                    f"margin 0 (red), +-2 logits (orange)", fontsize=9)
ax[2].fill_between(sgrid, req(A_thin[:, 4]), req(A_thin[:, 0]), color="C0", alpha=0.25, label="thin model, margin -2..+2 logits")
ax[2].plot(sgrid, req(A_thin[:, 2]), "C0", lw=1, label="thin model (margin 0), equivalent radius")
ax[2].plot(sgrid, req(A_thick[:, 2]), "C3", lw=1, label=f"thick model ({thick.split('_', 1)[1]}), same stations")
ax[2].plot(sgrid, a_img, "k.", ms=2, label="image radius (thin series, PSF from the image; r < 2.5 mm only)")
ax[2].plot(sgrid, a_img_thick, ".", color="gray", ms=2, label="image radius (thick series)")
ax[2].set_ylabel("radius (mm)"); ax[2].legend(fontsize=7, loc="upper right"); ax[2].set_ylim(0, None)
rel = (A_thin[:, 0] - A_thin[:, 4]) / np.maximum(A_thin[:, 2], 1e-6)
ax[3].plot(sgrid, 100 * rel, "C0", lw=1, label="model interval width, (A(-2) - A(+2)) / A(0)")
ax[3].plot(sgrid, 100 * (A_thick[:, 2] / np.maximum(A_thin[:, 2], 1e-6) - 1), "C3", lw=1, label="thick / thin model area - 1")
ax[3].plot(sgrid, 100 * ((a_img / req(A_thin[:, 2])) ** 2 - 1), "k.", ms=2, label="image / thin model area - 1")
ax[3].axhline(0, color="k", lw=0.5); ax[3].set_ylabel("%"); ax[3].set_xlabel("arc length from the trunk (mm)")
ax[3].legend(fontsize=7, loc="lower left"); ax[3].set_ylim(-100, 150)
plt.tight_layout()
figp = DATA / f"{patient}_straightened.png"
plt.savefig(figp, dpi=110)
np.savez(DATA / f"{patient}_straight.npz", s=sgrid, centerline=C, r_path=r_path, A_thin=A_thin, A_thick=A_thick,
         a_img=a_img, a_img_thick=a_img_thick, levels=np.array(LEVELS))

def band(lo, hi):
    q = (r_path >= lo) & (r_path < hi)
    return q
print(f"{'model r':>10} {'n':>4} {'interval +-2 logits':>20} {'thick/thin area':>16} {'image/thin area':>16}")
for lo, hi in ((0, 1.25), (1.25, 1.75), (1.75, 2.5), (2.5, 4), (4, 99)):
    q = band(lo, hi) & (A_thin[:, 2] > 0)
    if q.sum() < 5: continue
    print(f"{lo:4.2f}-{hi:<5} {q.sum():4d} {np.nanmedian(100 * rel[q]):19.1f}% "
          f"{np.nanmedian(100 * (A_thick[q, 2] / A_thin[q, 2] - 1)):+15.1f}% "
          f"{np.nanmedian(100 * ((a_img[q] / req(A_thin[q, 2])) ** 2 - 1)):+15.1f}%")
print(f"wrote {figp}")
