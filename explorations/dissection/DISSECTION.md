# Aortic dissection from a dual-lumen field

*Exploration, 2026-09-25. Synthetic phantom (arch-like aorta, two entry tears, four
branch stubs). Prototype code in this directory; nothing here is in the thalweg package.*

## Why dissection

A dissection is the one vascular geometry a single binary lumen cannot express: a true
lumen and a false lumen, an intimal flap between them, and entry tears where the flap is
missing. The design note already names it (`vmtk-successor.md` §5.6: "dissection flap
position (the true|false lumen tie sheet in `aortic_dissection`)"). TotalSegmentator
ships the task (`aortic_dissection`, weights Dataset 716: `aorta_true_lumen`,
`aorta_false_lumen` at 0.8 mm). What the field adds is the flap as a **tie sheet** rather
than a label, tears as **topology**, and branch feed as a **graph query**.

## The four questions

1. **Topology** — do the two lumens form one continuous domain (a tear exists) or two?
2. **Flap** — where is the true|false interface in each cross-section?
3. **Entry tears** — where do the lumens communicate through a hole in the flap?
4. **Branch feed** — which lumen supplies each branch (true / false / both)?

## How the phantom is built (`_dissect.py`)

An arch-like centerline (cubic Bezier, ~170 mm arc). A diametral flap plane splits the
tube into true (`v < 0`) and false (`v > 0`) in the centerline frame. The flap is a
background wall of thickness 1.2 mm where it is intact, so the two labels do **not**
touch. At an entry tear the wall is missing over a disk, and the labels meet at `v = 0` —
exactly what a two-lumen `aortic_dissection` labelmap shows. Four branch stubs leave the
outer wall: three single-sided (true / false / true), one straddling the flap ("both").

The label map is converted to logits (`-(g/2) * signed distance`, g = 4 logits/mm) and
encoded with rankfield, so every measurement below can in principle read a margin rather
than a voxel count.

## Results (2026-09-25)

| Question | Result |
|---|---|
| Topology | One domain at 6- and 26-connectivity (the tears connect the lumens) |
| Flap profile | 83 stations, A_true ≈ A_false ≈ 177 mm² median (the flap is diametral) |
| Entry tears | **2 / 2** ground-truth tears matched, centroids within 0.6 mm |
| Straddling branch | 87-voxel T\|F contact tagged `branch:renal_left`, not a tear |
| Branch feed | **4 / 4** match (true / false / true / both) |
| Tie sheet | 281 sign-change voxels of `m_true - m_false`, side agreement 1.00 |

Console summary and the scored JSON: `~/tmp/data/vessels/phantom_arch_dissection.json`.
Figure: `phantom_arch_dissection.png` (sagittal arch, flap line profiles, areas, contacts).

## What is actually being demonstrated

- **Contact is not one thing.** T\|F face contact appears at tears (a hole in the flap)
  and at a straddling branch (both lumens open into the same stub). A detector that
  reports every contact as a tear would fire 3 times for 2 tears; tagging the contact by
  its neighborhood separates them.
- **The flap is a wall until it is not.** Where intact, a line across `v` reads
  true | wall | false. At a tear it reads true | false. That 1-D read is the flap
  position of design note §5.6 — a signal, not a mesh.
- **Branch feed is a lumen-side query.** Sampling the ostium and the first few mm of the
  stub with the two margins classifies true / false / both without a flow solve. On a
  real case that is the malperfusion question: does the celiac come off the false lumen?
- **Topology is exact and cheap.** Connected components of the lumen union answer
  "is there a communication at all" before any tear is localized.

## What this phantom is not

- **Not a clinical labelmap.** The phantom resolves the flap as a background wall so
  the detector can be scored. A real `aortic_dissection` two-class output would show
  the flap as the tie sheet itself. The Zenodo FPA case is a CFD mesh with named walls,
  not that labelmap (see Real data).
- **No sub-voxel flap position yet.** The profile reads labels at 0.75 mm. The field
  zero set of `m_true - m_false` is the sub-voxel version; `tie_sheet` counts its sign
  changes but does not yet interpolate the crossing.
- **No image.** Entry tears are a contrast finding as much as a label finding; a real
  study wants the CTA alongside (design note §5.6 image-side sizing).

## Real data (2026-09-25)

**No license-gated task.** TotalSegmentator `aortic_dissection` is Dataset 716 and sits
behind the TotalSegmentator license (`LICENSE_GATED` in haversack). Skipped on purpose.

**Public source that worked:** Moretti et al., *Comparative analysis of patient-specific
aortic dissections through CFD...*, Zenodo 5801938 (CC BY 4.0). Four models (healthy,
fully perfused / partially thrombosed / fully thrombosed dissection) as IGES plus a full
OpenFOAM case per model. Only `Geometries.zip` (103 MB) and `FPA.zip`'s `polyMesh`
(~110 MB via HTTP range) were pulled; the multi-GB result archives were not.

What the FPA (fully perfused) case actually carries:

| Asset | What it is |
|---|---|
| `constant/polyMesh` | 449,424 cells, 1.55M points, ANSA polyhedral mesh in meters |
| Boundary `Dissecazione` | 25,511 wall faces — the dissection (false-lumen) wall |
| Boundary `Main` | 16,829 wall faces — the main (true-lumen) wall |
| `Out_top_*`, `Out_main_lat_*` | named branch outlets |
| `sets/dissebot*`, `sets/main*` | wall-adjacent **cell patches**, not whole lumens |

So the dual-lumen information is real and named, but it is a **CFD fluid domain**, not a
two-class labelmap. Consequences:

- IGES surfaces are 572 unsewn patches in a CAD frame; FPA/PTA/FTA are in **different**
  coordinate systems, so true ≠ FPA − FTA.
- Cell sets are wall-layer patches (14k / 9k cells), not the full lumens.
- Dual-graph flood from the two walls shows the fluid domain is **not one cell-connected
  component** (hanging nodes / disconnected islands): a BFS from one Main cell covers
  39k cells and only 340 of 16,829 Main seeds.
- A nearest-wall split of the dual graph labels 41k cells nearer Dissecazione and 19k
  nearer Main, with **1,935 crossing edges** — the candidate flap/tear set. Voxelized
  contacts form one cluster (507 voxels) near z ≈ −65 mm.

**Figure:** `~/tmp/data/vessels/FPA_real_dissection.png` — Dissecazione (clay) vs Main
(teal) wall samples and the dual-graph contact edges (red).

**Data on disk:** `~/tmp/data/dissection/` (Geometries, FPA_mesh, `FPA_dual.npz`).

## Next, if this thread continues

1. Slice-wise true/false areas on the real mesh from the nearest-wall partition, with
   the crossing-edge set as the flap (this is the §5.6 1-D signal on a patient).
2. Sub-voxel flap position from the margin-difference zero crossing, with an interval.
3. Tear detection on a two-label input (no wall): openings in the true|false contact.
4. Feed the branch table of `docs/deliverables.md` with `feed: true|false|both` per
   branch, plus A_true(s) / A_false(s) / flap angle(s) as the dissection columns.
5. Malperfusion score: for each named branch, distance along the graph to the nearest
   entry tear, per lumen.
6. When a TS license exists: `aortic_dissection` on a real CTA, which is the only path
   to a clinical two-lumen labelmap.

## Running

```bash
cd explorations/dissection
../../../haversack/.venv/bin/python dissection.py          # analysis + JSON
../../../haversack/.venv/bin/python dissection_figure.py   # figure
```

Switches live in `_dissect.py` (tear positions and radii, branch list, flap thickness,
grid spacing).
