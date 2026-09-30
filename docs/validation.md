# Validation beyond one subtree (phase 5)

Phase 1 established reproduction: the tracer reproduces the research trees exactly, and the vmtk
ports reproduce vmtk's own output (see `docs/port-plan.md`). This document records how the port
behaves on inputs it was not tuned on. Numbers are rough (one machine, one session). The scripts
live in `validation/`.

## 1. Analytic phantoms (`validation/phantom_suite.py`, 0.7 mm grid, 3 s)

Margin fields built from capsule chains with a known axis, radius, ends and angles; logit slope
10.6 / mm, clipped at ±8, as in a real store.

| Phantom | Ends (traced / true) | Junctions | Axis error median / p95 (mm) | Radius error median / p95 (mm) | Other |
|---|---|---|---|---|---|
| Y, 30° | 3 / 3 | 1 / 1 | 0.035 / 0.080 | 0.030 / 0.069 | deflection 15.3, 15.9 (true 15, 15) |
| Y, 60° | 3 / 3 | 1 / 1 | 0.022 / 0.100 | 0.026 / 0.091 | deflection 29.7, 30.1 (true 30, 30) |
| Y, 90° | 3 / 3 | 1 / 1 | 0.055 / 0.072 | 0.050 / 0.062 | deflection 45.0, 45.0 (true 45, 45) |
| Trifurcation | 4 / 4 | 1 / 1 | 0.050 / 0.071 | 0.042 / 0.067 | deflection 36.6 to 39.5 (true 36.9) |
| Tapered (3 → 1 mm) | 2 / 2 | 0 / 0 | 0.000 / 0.000 | 0.006 / 0.015 | |
| Arc (radius of curvature 15 mm, r 1.5) | 2 / 2 | 0 / 0 | 0.104 / 0.120 | 0.095 / 0.108 | |
| Thin (r = 0.8 voxel) | 2 / 2 | 0 / 0 | 0.127 / 0.127 | 0.069 / 0.069 | |
| Elliptic (semi-axes 3 × 1.2 mm) | **24 / 2** | **11 / 0** | 0.269 / 2.302 | n/a | aspect ratio 0.393 (true 0.40); Feret 2.36 / 5.76 (true 2.4 / 6.0) |
| Two tubes in contact (0.3 mm overlap) | **13 / 4** | **10 / 0** | 0.141 / 1.005 | 0.128 / 0.537 | one field component |
| Torus | 2 (a tree) | 0 | n/a | n/a | field loops 1 (true 1) |

### What this says

- **Round tubes are traced right.**
  - Topology is exact on every round phantom, including a trifurcation and a tube of radius
    0.8 voxel (diameter 1.6 voxels).
  - The axis is within ~0.1 mm and the radius within ~0.1 mm.
  - Deflection angles are within 1° at 30–90° and within 2.6° on the trifurcation. The chords
    start at arc length r_J, outside the junction's ball. The first version started them at the
    junction point and read wide angles low: 73.6° for a T junction, 59° for 70°.
- **Flattened lumens break the spur rule.**
  - The tracer keeps a terminal branch longer than 2 × (radius at its junction) + 1 mm. In a flat
    lumen that radius is half the *depth*, so side lobes across the *width* survive as branches:
    24 ends on a tube with 2.
  - The same happens on real tubes. The idc-torso1 `total` esophagus traces as 13 tips and the
    colon as 16. The aorta traces clean: 3 tips, which are its real ends.
  - The sections are right (aspect ratio and Feret widths), so the failure is in the tracer's
    pruning, not in the field or the measures.
  - This is the first tube requirement that the vessel tuning does not meet. The fix is a
    pruning rule relative to the wall (below).
- **Contact is connectivity.**
  - Two tubes of one class whose walls overlap form one field component, and the tracer builds
    a ladder between them. That is correct for the field (they *are* connected), but a reader of
    the graph should know.
  - Where contact is between classes (artery–vein), the classes stay separate. Within one class,
    the contact shows as a loop in the field (`qc.json` `field_loops`); flagging the short
    junction-to-junction edges it produces is not done yet.
- **Loops are seen but not kept.**
  - The field's topology finds the torus's loop (genus 1).
  - The tracer builds trees, so the graph drops the cycle. The format allows cycles; the tracer
    does not produce them yet.
  - `qc.json` carries the field's loop count per tree (`field_loops`: 0–7 on the ladder below).

## 2. Second patient and reconstruction ladders

`thalweg run` (the tier-1 batch, default options) was run on both patients' ladders: the same
scan reconstructed at 0.625 to 5 mm, each run through TotalSegmentator `lung_vessels`. Each case
took 21–39 s for the three trees. One bug surfaced on the way and was fixed: newer stores record
one labeling scheme per cascade stage, and the graph's `Source` expected a single one. Numbers
regenerated 2026-09-30 with the current code (the round-2 review's runs); the deflection column
changed when angle chords moved outside the junction's ball (§1).

| Run | Tree | Edges | Length (cm) | Radius p50 (mm) | Strahler ≥ 3 (cm) | Strahler ≥ 4 (cm) | Area median (mm²) | Deflection median (°) | Outside field (mm) | Field loops |
|---|---|---|---|---|---|---|---|---|---|---|
| C3N 0.625 | arteries | 1065 | 1052.0 | 1.55 | 228.8 | 95.3 | 6.63 | 44.6 | 2.2 | 4 |
| C3N 1.25 (lung kernel) | arteries | 1056 | 1041.0 | 1.55 | 233.3 | 91.3 | 6.42 | 44.6 | 0.5 | 2 |
| C3N 2 mm slab | arteries | 1002 | 1019.6 | 1.57 | 231.5 | 94.9 | 6.88 | 44.7 | 0.9 | 6 |
| C3N 3.75 | arteries | 552 | 694.2 | 1.73 | 153.4 | 43.2 | 10.04 | 44.4 | 0.5 | 3 |
| C3N 0.625 | veins | 950 | 937.9 | 1.58 | 207.7 | 86.6 | 6.49 | 46.6 | 0.6 | 2 |
| C3N 1.25 | veins | 933 | 911.5 | 1.57 | 203.9 | 71.7 | 6.38 | 48.9 | 1.2 | 3 |
| C3N 2 mm slab | veins | 900 | 905.0 | 1.60 | 201.9 | 72.0 | 6.58 | 48.7 | 1.1 | 2 |
| C3N 3.75 | veins | 496 | 621.9 | 1.74 | 134.5 | 55.9 | 9.96 | 43.9 | 3.3 | 2 |
| C3N 0.625 | airways | 265 | 343.7 | 1.17 | 62.8 | 28.5 | 4.52 | 27.9 | 2.0 | 2 |
| C3N 1.25 | airways | 251 | 313.9 | 1.20 | 59.0 | 31.6 | 4.91 | 28.4 | 1.8 | 3 |
| C3N 2 mm slab | airways | 229 | 316.5 | 1.23 | 56.9 | 28.4 | 4.97 | 28.9 | 9.0 | 1 |
| C3N 3.75 | airways | 212 | 215.2 | 1.47 | 43.7 | 25.0 | 10.04 | 33.2 | 1.2 | 0 |
| MSB 0.625 | arteries | 590 | 706.9 | 1.66 | 142.9 | 51.9 | 7.58 | 46.9 | 0.5 | 3 |
| MSB 1.25 | arteries | 592 | 709.8 | 1.66 | 146.7 | 53.3 | 7.95 | 46.8 | 1.7 | 5 |
| MSB 2 mm slab | arteries | 518 | 662.9 | 1.67 | 137.8 | 45.3 | 7.69 | 45.4 | 0.0 | 5 |
| MSB 5 mm (DLIR) | arteries | 240 | 324.7 | 1.99 | 66.8 | 25.7 | 13.32 | 48.6 | 1.7 | 7 |
| MSB 0.625 | veins | 467 | 536.4 | 1.63 | 125.1 | 44.6 | 7.26 | 47.9 | 3.5 | 5 |
| MSB 1.25 | veins | 474 | 557.0 | 1.64 | 127.7 | 43.6 | 7.41 | 49.4 | 0.5 | 1 |
| MSB 2 mm slab | veins | 427 | 508.6 | 1.65 | 115.5 | 37.3 | 7.49 | 46.8 | 3.1 | 2 |
| MSB 5 mm (DLIR) | veins | 196 | 251.8 | 2.02 | 45.3 | 22.9 | 16.35 | 42.9 | 0.4 | 1 |
| MSB 0.625 | airways | 162 | 186.4 | 1.34 | 35.7 | 21.7 | 6.25 | 34.0 | 0.2 | 7 |
| MSB 1.25 | airways | 151 | 178.0 | 1.38 | 38.6 | 19.7 | 5.89 | 35.7 | 0.0 | 4 |
| MSB 2 mm slab | airways | 126 | 160.2 | 1.47 | 32.4 | 15.9 | 7.28 | 31.9 | 0.7 | 2 |
| MSB 5 mm (DLIR) | airways | 79 | 84.2 | 1.96 | 23.4 | 9.2 | 16.66 | 32.5 | 0.0 | 0 |

- **Up to 2 mm, arteries and veins hold.**
  - Total length moves −6.2 to +3.8 %; the radius, area and deflection medians stay within 5 %.
  - Strahler ≥ 3 length moves up to 3.6 % for arteries and 7.7 % for veins (MSB, 2 mm).
- **Airways do not hold, even at 1.25–2 mm.** The airways are the thinnest tree:
  - length falls 5–14 %;
  - the radius median rises 3–10 %;
  - the area median moves −6 to +16 %.

  Airway statistics are not comparable across reconstructions of one scan.
- **At 3.75–5 mm, thin vessels are dropped**, as the research found (§12.2 of the design note):
  - length falls 34–54 %;
  - every median rises, because only thick vessels remain.

  `qc.json` flags these runs (`acquisition.coarse_slices`, from the source spacing: True for C3N
  3.75 and MSB 5 mm).
- **Strahler order is fragile.** Strahler ≥ 4 length of the C3N veins moves 87 → 72 cm between
  0.625 and 1.25 mm, while total length moves 3 %. An order counts how many thin tips survive, so
  one lost tip can demote a whole subtree. Report order-based statistics with this caveat, or use
  a statistic based on diameter.
- **The second patient works unchanged.** MSB-02664 has a smaller tree at 0.625 mm (707 cm of
  arteries against 1052), and nothing failed.
- **Centerline outside the field** is 0–3.5 mm per tree, out of meters, with one exception: the
  C3N 2 mm airways at 9.0 mm.

## 3. Tubes that are not vessels

On the idc-torso1 `total` store (1.5 mm), the tracer returned:

| Structure | Edges | Tips | Junctions | Length |
|---|---|---|---|---|
| Trachea | 16 | 9 | 7 | 28.1 cm |
| Esophagus | 23 | 13 | 10 | 43.0 cm |
| Aorta | 4 | 3 | 1 | 54.1 cm |
| Colon | 28 | 16 | 12 | 101.7 cm |

The aorta is nearly right. Its root is the deepest point, in the arch; the descending aorta
runs 436 mm to the bifurcation. But the aortic root traces as two 33 mm "tips" from one
junction. The bulb of the sinuses is wider than it is long, so the same round-lumen spur rule
lets a lobe through. The others show the flat-lumen spur problem of §1.

With the two new tracer options (`--prune wall --ridge-passes 4`):

| Structure | Edges | Tips | Junctions | Length | Capped ends, made / attempted |
|---|---|---|---|---|---|
| Aorta | 4 | 3 | 1 | 53.9 cm | 1 / 2 |
| Esophagus | 3 | 3 | 0 (root of degree 3) | 27.0 cm | 1 / 2 |
| Trachea | 6 | 4 | 2 | 23.3 cm | 3 / 4 |
| Colon | 8 | 6 | 2 | 73.5 cm | 1 / 4 |

- **Wall pruning takes most of the lobes out.** The esophagus goes from 43 cm with 13 tips to
  27 cm with 3 (an adult esophagus is ~25 cm); the colon goes from 16 tips to 6.
- **It does not remove the aortic-root lobe.** Measured across the ascending aorta, the lobe
  reaches past the parent's wall.
- **Capping ends of large, irregular tubes mostly fails** (reported, never left open). The cut
  two radii back from a tip lands where the section still touches a neighboring branch or
  another lobe.
- **What single tubes need:** a tracer built for them (the main path between the two farthest
  ends, then branches that must protrude), and cuts placed by section shape rather than a fixed
  distance. That is the next piece of tube work.

## 4. vmtk on a second tree, and against the truth

**MSB-02664, a 57-tip subtree** (`research/vessels/vmtk_prep.py` + `vmtk_run.py`, unchanged). The
thalweg trace of this patient reproduces the research JSON exactly (590 segments).
- **Targets:** vmtk reached 55 of 57. It logged "degenerate descent" twice, and targets 35 and
  36 got lines of 0.8 and 1.0 mm. `vmtk_run.py`'s "reached" test (a line end within 2 mm of the
  target) accepts those, and counted 57.
- **Vessels under 2.5 mm radius:** the methods agree as on C3N-00704. vmtk points lie a median
  0.10–0.11 mm from ours, and vmtk's radius is +0.02–0.03 mm larger.
- **Wide vessels (r ≥ 2.5 mm):** the raw numbers differ by 0.63 mm median, and vmtk's radius
  reads 0.22 mm smaller. This gap is the comparison's own artifact (found by the round-2
  review):
  - `vmtk_prep.py` keeps the surface only within (our radius + 1.5 mm) of our centerline, so
    where the lumen reaches farther, vmtk sees a cut wall. 9.3 % of MSB's surface vertices lie on
    that cut.
  - 70 % of vmtk's inscribed spheres at r ≥ 2.5 mm touch the cut.
  - The wide-vessel points that do not touch it agree to 0.107 mm and +0.031 mm, as the narrow
    ones do.

**Who is right: analytic phantoms** (`validation/vmtk_phantom.py prep | vmtk | compare 1 2 3 4`).
vmtk runs on the field's zero set, seeded at the true ends; thalweg traces the field. Distances
are to the true axis, away from ends and junctions. Each cell is the axis error median /
radius error median, in mm:

| Phantom | vmtk | thalweg, 1 pass | 2 passes | 3 passes | 4 passes |
|---|---|---|---|---|---|
| Y, trunk r 3 mm | 0.003 / +0.002 | 0.022 / −0.027 | 0.010 / −0.011 | 0.005 / −0.002 | 0.002 / −0.001 |
| Y, trunk r 5 mm | 0.002 / −0.006 | 0.105 / −0.108 | 0.028 / −0.027 | 0.007 / −0.012 | 0.003 / −0.009 |
| Y, trunk r 8 mm | 0.001 / −0.004 | 0.102 / −0.102 | 0.026 / −0.025 | 0.007 / −0.010 | 0.002 / −0.007 |
| Arc, r 5 mm | 0.003 / −0.006 | 0.101 / −0.100 | 0.025 / −0.029 | 0.006 / −0.010 | 0.003 / −0.007 |

**With one pass, vmtk is more accurate on clean round tubes.** The cause is thalweg's ridge
refinement:
- the refinement searches a fixed 5 × 5 × 5 grid of offsets within ±0.5 mm (0.25 mm steps), so
  the refined point is quantized to that grid;
- a point off the axis by δ has an inscribed radius smaller by δ.

The size of the error depends on where the axis falls relative to the grid.
- These phantoms have axes parallel to a grid axis at a fixed offset, and read ~0.1 mm off.
- On oblique tubes with axes off the lattice, one pass reads 0.03–0.07 mm small at radii from
  0.75 to 5 mm, on both a 0.7 mm isotropic grid and the store's 1.0 × 0.7 × 0.7 grid. Four
  passes leave 0.007–0.016 mm (the round-2 review's sweep).

A coarse-to-fine search fixes this: each further pass searches a grid four times finer around
the best point so far.

**On the real subtrees, four passes remove the old radius offset against vmtk.** Excluding the
prep-cut points (round-2 review):

| Subtree | Gap, 1 pass | vmtk − ours radius, 1 pass | Gap, 4 passes | vmtk − ours radius, 4 passes |
|---|---|---|---|---|
| C3N-00704 | 0.085 | +0.030 | 0.060 | −0.006 |
| MSB-02664 | 0.106 | +0.027 | 0.074 | −0.013 |

(mm.) The research finding "vmtk reads +0.029 mm larger" was this quantization. The MSB result
holds in every radius band.

## 5. The two tracer options, and the defaults chosen

**Decided 2026-09-30:** four ridge passes are the default; length pruning stays the default,
with `--prune wall` as an option. The research reference is `--ridge-passes 1`. Every number in
§1–§3 and in §4's first part was taken with one pass (the reference) unless it says otherwise.

Effect on the trees:

| Case | Tree | Reference: tips / length (cm) / radius p50 (mm) | 4 passes | Wall pruning |
|---|---|---|---|---|
| C3N-00704 | arteries | 537 / 1052.0 / 1.551 | 537 / 1036.0 / 1.599 | 488 / 1024.8 / 1.554 |
| C3N-00704 | veins | 483 / 937.9 / 1.578 | 483 / 924.0 / 1.627 | 434 / 910.5 / 1.585 |
| C3N-00704 | airways | 133 / 343.7 / 1.167 | 133 / 339.2 / 1.224 | 115 / 333.3 / 1.167 |
| MSB-02664 | arteries | 298 / 706.9 / 1.655 | 298 / 697.9 / 1.704 | 240 / 666.8 / 1.657 |
| MSB-02664 | veins | 237 / 536.4 / 1.632 | 237 / 529.6 / 1.679 | 205 / 514.7 / 1.634 |
| MSB-02664 | airways | 82 / 186.4 / 1.345 | 82 / 183.9 / 1.403 | 61 / 171.1 / 1.355 |

**Runtime** (C3N arteries; the four combinations alternated in one process, three repeats; load
3.5–6.4):

| | Trace only, median | Whole `thalweg run` (C3N / MSB) |
|---|---|---|
| Reference | 4.84 s | 37.7 / 33.1 s |
| 4 passes | 8.96 s (1.85×) | 52.3 / 46.7 s (+39 / +41 %) |
| Wall pruning | 4.71 s (no cost) | (fewer branches make the table faster) |

**Four passes (`--ridge-passes 4`):**
- **For:**
  - It removes the radius deficit, ~0.03–0.07 mm on oblique vessels and ~0.1 mm on
    grid-aligned phantoms.
  - It matches vmtk on the phantoms, and turns vmtk's +0.030 mm into −0.006 mm on the real
    subtrees (§4).
  - Every tree's median radius rises ~0.05 mm; paths get ~1.5 % shorter as they straighten onto
    the axis; tips do not change.
- **Against:**
  - The trace takes 1.85× as long (+~40 % on a whole run).
  - Sub-voxel tubes over-read: an oblique r = 0.5 mm tube goes from +0.029 to +0.098 mm on the
    store's 1.0 × 0.7 × 0.7 grid, and the suite's thin phantom's radius error goes from 0.069 to
    0.090 mm.
  - C3N arteries' `outside_mm` goes from 2.2 to 4.0 mm, and one junction pair merges
    (1065 → 1064 edges).

**Wall pruning (`--prune wall`):**
- **For:**
  - It is what makes flattened tubes usable: esophagus 13 → 3 tips, colon 16 → 6, elliptic
    phantom 24 → 3 ends (true 2) (§3).
  - It costs no time.
  - On shallow-angle round phantoms (10–45°) it removed no real side branch.
- **Against:**
  - It removes 9–26 % of tips on vessel trees and cuts order statistics: Strahler ≥ 4 length
    −12 % (C3N arteries), −14 % (C3N veins), −20 % (MSB arteries).
  - Whether the removed tips are real short stubs is unknown: nothing here says either way.
  - On the two-tube contact phantom it deletes one whole tube's axis (3 of 4 true ends,
    39.5 mm of 60 mm). Measured from one tube's axis, the other tube's tip lies inside the merged
    wall. That matters for the self-contact requirement (colon loops that touch).

## 5b. Airway walls

The model labels the airway lumen and its wall as separate classes; `thalweg run` measures the
wall at every airway section station (branch table: `wall_area_mm2`, `wall_area_percent`,
`wall_thickness_mm`; `summary.json`: Pi10). A phantom lumen of 2 mm in a 1 mm wall reads a wall
area within 5 %, WA% within 2 points and thickness within 0.05 mm (`tests/test_measure.py`).

| Case | Pi10 (mm) | WA%, Strahler 1 / 2 / 3 | Thickness (mm), Strahler 1 / 2 / 3 |
|---|---|---|---|
| C3N-00704 0.625 | 4.17 (1151 stations) | 85.6 / 73.7 / 61.1 | 1.37 / 1.30 / 1.12 |
| MSB-02664 0.625 | 4.34 (431 stations) | 85.4 / 69.8 / 67.8 | 1.45 / 1.36 / 1.30 |

**The wall is at the model's resolution.** The measured thickness is nearly constant, 1.3–1.45 mm
at every order: about two voxels of the 0.7 mm grid, the thinnest shell the model draws. Real
small-airway walls are thinner, so small-airway WA% and Pi10 read high, the airway analog of the
vessels' ~1 mm radius floor. Pi10 here (4.2–4.3 mm) is above the ~3.6–3.8 mm usually reported for
healthy lungs on CT for the same reason. Compare these across cases on one model and grid, not
against the literature.

## 5c. Bronchoarterial pairing

Each airway centerline sample is paired with the nearest artery sample within 8 mm that runs
parallel (|cos| ≥ 0.7). Each airway branch then gets its companion artery and the median
bronchus-to-artery diameter ratio (`thalweg.pairing`; a synthetic parallel pair gives exactly
the radius ratio, and a crossing artery nearer than the partner does not pair).

| Case | Airway samples paired | Paired branches | Ratio median | Paired branches above 1 | Ratio by Strahler 1 / 2 / 3 |
|---|---|---|---|---|---|
| C3N-00704 0.625 | 84 % | 235 | 0.58 | 4 % | 0.53 / 0.62 / 0.62 |
| MSB-02664 0.625 | 66 % | 118 | 0.51 | 7 % | 0.48 / 0.50 / 0.64 |

The medians sit below the ~0.65–0.7 usually reported for healthy lungs on CT. Two causes are
plausible, and neither has been separated from the other:
- the airway lumen class at the model's resolution;
- pairing to the nearest parallel artery, which can be a wider parent rather than the bronchus's
  own companion.

As with the walls, compare across cases on one model and grid.

## 6. Not yet validated

- **Thin-caliber ground truth.** The ~1 mm radius floor belongs to the model; checking the
  method below it needs a phantom scan or a gated high-resolution acquisition.
- **vmtk on non-round phantoms.** Not motivated by the MSB wide-vessel gap (a prep artifact,
  §4), but still open.
- **Phantoms off the lattice and on anisotropic grids,** scoring sections against the truth
  (the suite's round tubes are axis-aligned; its equivalent diameter is scored against the traced
  radius, and reads 1–2 % small against the true one).
