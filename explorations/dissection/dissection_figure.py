"""The dissection figure: labels, cross-sections, flap profile, tear and branch callouts.

    cd explorations/dissection && ../../../haversack/.venv/bin/python dissection_figure.py

Four panels. (A) A coronal-ish slice through the aorta with the two lumens, the flap wall
and the two entry tears. (B) Cross-sections at a tear and between tears. (C) A_true and
A_false along the arch. (D) The contact map: tear contacts vs the straddling branch.
Writes ``~/tmp/data/vessels/<run>_dissection.png``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "vessels"))
from _data import DATA  # noqa: E402

from _dissect import (  # noqa: E402
    R_OUT, SPACING, TEARS,
    centerline, phantom_labels,
)
from dissection import sample_labels  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else "phantom_arch"
out_json = json.loads((DATA / f"{RUN}_dissection.json").read_text())
origin = np.asarray(out_json["grid"]["origin_xyz"], float)
shape = tuple(out_json["grid"]["shape_zyx"])

s_cl, P, T, N, B = centerline()
labels, gt, _ = phantom_labels(shape, origin)

# colors: wall pale, true teal, false clay
CW = "#E7EAEE"
CT = "#2E6B7A"
CF = "#B85C38"
CA = "#1C2430"
CM = "#5B6B7A"


def to_world(idx_zyx):
    return np.stack([
        origin[0] + idx_zyx[:, 2] * SPACING,
        origin[1] + idx_zyx[:, 1] * SPACING,
        origin[2] + idx_zyx[:, 0] * SPACING,
    ], 1)


fig = plt.figure(figsize=(11, 8.5), facecolor="white")
gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22,
                      left=0.07, right=0.97, top=0.92, bottom=0.08)

# --- A: sagittal slice (constant x) through the arch plane ---------------------
ax = fig.add_subplot(gs[0, 0])
ix = int(np.round((0.0 - origin[0]) / SPACING))
ix = np.clip(ix, 0, shape[2] - 1)
sl = labels[:, :, ix]                      # (z, y) — the arch lives in the y-z plane
rgb = np.zeros(sl.shape + (3,), float)
rgb[sl == 0] = (0.91, 0.92, 0.93)
rgb[sl == 1] = (0.18, 0.42, 0.48)
rgb[sl == 2] = (0.72, 0.36, 0.22)
ext = [origin[1], origin[1] + (shape[1] - 1) * SPACING,
       origin[2], origin[2] + (shape[0] - 1) * SPACING]
ax.imshow(rgb, origin="lower", extent=ext, interpolation="nearest")
for t in TEARS:
    i0 = int(np.argmin(np.abs(s_cl - t["s"])))
    wc = P[i0] + t["u"] * N[i0]
    ax.plot(wc[1], wc[2], "o", mfc="none", mec="#C41E3A", ms=12, mew=1.8, zorder=5)
    ax.plot(wc[1], wc[2], "+", color="#C41E3A", ms=10, mew=1.8, zorder=6)
    ax.text(wc[1] + 3, wc[2], f"s={t['s']:.0f}", fontsize=7, color="#C41E3A", zorder=6)
for b in gt["branches"]:
    base = np.asarray(b["base"]); tip = np.asarray(b["tip"])
    ax.plot([base[1], tip[1]], [base[2], tip[2]], "-", color=CA, lw=1.4, zorder=5)
    ax.text(tip[1], tip[2], b["name"][:6], fontsize=7, color=CA, ha="center", va="bottom")
ax.set_xlabel("y (mm)"); ax.set_ylabel("z (mm)")
ax.set_title("A  sagittal slice through the arch: true / false / flap wall", loc="left",
             fontsize=10, color=CA)
ax.set_aspect("equal")

# --- B: flap line profile across v, tear vs intact -----------------------------
ax = fig.add_subplot(gs[0, 1])
vv = np.linspace(-R_OUT, R_OUT, 220)
for col, (s0, title) in enumerate(((38.0, "at entry tear s=38"),
                                   (70.0, "intact flap s=70"))):
    i0 = int(np.argmin(np.abs(s_cl - s0)))
    pts = P[i0] + vv[:, None] * B[i0]          # along v at u = 0
    lab = sample_labels(labels, origin, pts)
    img = np.zeros((len(pts), 3))
    img[lab == 0] = (0.91, 0.92, 0.93)
    img[lab == 1] = (0.18, 0.42, 0.48)
    img[lab == 2] = (0.72, 0.36, 0.22)
    yoff = col * 2.2
    ax.imshow(img.reshape(1, -1, 3), aspect="auto", origin="lower",
              extent=[vv[0], vv[-1], yoff, yoff + 1.6], interpolation="nearest")
    ax.plot([0, 0], [yoff, yoff + 1.6], color="#C41E3A", lw=1.0, ls="--", alpha=0.85)
    ax.text(vv[0], yoff + 1.9, title, fontsize=8, color=CA, va="bottom")
    # wall samples near the flap, as ticks under the strip
    wall = np.where((lab == 0) & (np.abs(vv) < 3.0))[0]
    if len(wall):
        ax.plot(vv[wall], np.full(len(wall), yoff - 0.12), "|", color=CM, ms=6, mew=1.4)
        ax.text(vv[wall].mean(), yoff - 0.35, f"{len(wall)} wall samples in |v|<3 mm",
                fontsize=7, color=CM, ha="center", va="top")
    else:
        ax.text(0, yoff - 0.35, "no wall samples — labels touch at v=0 (tear)",
                fontsize=7, color="#C41E3A", ha="center", va="top")
ax.set_xlim(-R_OUT, R_OUT)
ax.set_ylim(-1.2, 5.4)
ax.set_xlabel("local v (mm) across the flap")
ax.set_yticks([])
ax.set_title("B  label line across the flap: wall band vs tear contact", loc="left",
             fontsize=10, color=CA)

# --- C: area profiles ----------------------------------------------------------
ax = fig.add_subplot(gs[1, 0])
prof = out_json["flap_profile"]
s = [r["s"] for r in prof]
ax.plot(s, [r["a_true"] for r in prof], "-", color=CT, lw=1.8, label="A_true")
ax.plot(s, [r["a_false"] for r in prof], "-", color=CF, lw=1.8, label="A_false")
for t in TEARS:
    ax.axvline(t["s"], color="#C41E3A", lw=1.0, ls="--", alpha=0.8)
for b in gt["branches"]:
    ax.axvline(b["s"], color=CM, lw=0.8, ls=":", alpha=0.8)
    ax.text(b["s"], ax.get_ylim()[1] * 0.96, b["name"][:4], fontsize=6, color=CM,
            rotation=90, va="top", ha="right")
ax.set_xlabel("s along the arch (mm)")
ax.set_ylabel("area (mm$^2$)")
ax.legend(loc="upper right", fontsize=8, frameon=False)
ax.set_title("C  lumen areas along the aorta (dashed red = tears)", loc="left",
             fontsize=10, color=CA)

# --- D: contact clusters (tears vs branch) -------------------------------------
ax = fig.add_subplot(gs[1, 1])
tears = out_json["tears_detected"]
names = []
vals = []
cols = []
for t in tears:
    if t.get("on_branch"):
        names.append(f"branch\n{t['on_branch']}")
        cols.append(CM)
    else:
        m = t.get("gt_match") or {}
        names.append(f"tear s={m.get('s', '?')}")
        cols.append("#C41E3A")
    vals.append(t["n_voxels"])
bars = ax.bar(range(len(vals)), vals, color=cols, width=0.6)
for i, (v, n) in enumerate(zip(vals, names)):
    ax.text(i, v + max(vals) * 0.03, f"{v} vox", ha="center", fontsize=8, color=CA)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel("T|F contact voxels")
ax.set_title("D  where the two labels touch (tears vs straddling branch)", loc="left",
             fontsize=10, color=CA)

fig.suptitle("Aortic dissection from a dual-lumen field — phantom prototype",
             fontsize=12, color=CA, x=0.07, ha="left")
out = DATA / f"{RUN}_dissection.png"
fig.savefig(out, dpi=160, facecolor="white")
print("wrote", out)
