# Validation beyond one subtree (phase 5)

Phase 1 established reproduction: the tracer reproduces the research trees exactly, and the vmtk
ports reproduce vmtk's own output (see `docs/port-plan.md`). This document records how the port
behaves on inputs it was not tuned on. Numbers are rough (one machine, one session). The scripts
live in `validation/`.

## 1. Analytic phantoms (`validation/phantom_suite.py [--oblique]`, a few seconds each)

Margin fields built from capsule chains with a known axis, radius, ends and angles; logit slope
10.6 / mm, clipped at ±8, as in a real store. Each phantom is traced with the defaults (four ridge
passes) on two grids:

- **on the lattice**: 0.7 mm isotropic, the phantom's axes on lattice lines (the best case);
- **oblique**: the same phantom turned by a random rotation and shifted by a fraction of a voxel,
  on an anisotropic 0.62 × 0.7 × 0.8 mm grid, so no axis follows the lattice; scored after
  turning the trace back.

"Sections" is the branch table's equivalent diameter over the TRUE diameter at the edge's middle
(edges whose shape is reliable). Each pair of cells is on the lattice / oblique.

| Phantom | Ends, junctions (traced / true), lattice | Oblique | Axis error median / p95 (mm), lattice | Oblique | Radius error median / p95 (mm), lattice | Oblique | Other, lattice | Oblique |
|---|---|---|---|---|---|---|---|---|
| Y, 30° | 3 / 3, 1 / 1 | 3 / 3, 1 / 1 | 0.005 / 0.012 | 0.007 / 0.018 | 0.003 / 0.015 | 0.013 / 0.020 | sections 0.989; deflection 15.6, 16.0 | sections 0.988; deflection 7.4, 7.7 |
| Y, 60° | 3 / 3, 1 / 1 | 3 / 3, 1 / 1 | 0.002 / 0.016 | 0.007 / 0.016 | 0.004 / 0.009 | 0.014 / 0.019 | sections 0.989; deflection 29.9, 30.1 | sections 0.991; deflection 30.1, 30.2 |
| Y, 90° | 3 / 3, 1 / 1 | 3 / 3, 1 / 1 | 0.003 / 0.006 | 0.007 / 0.015 | 0.006 / 0.008 | 0.013 / 0.019 | sections 0.989; deflection 45.0, 45.0 | sections 0.989; deflection 45.0, 45.1 |
| Trifurcation | 4 / 4, 1 / 1 | 4 / 4, 2 / 1 | 0.004 / 0.014 | 0.008 / 0.018 | 0.008 / 0.014 | 0.011 / 0.020 | sections 0.987; deflection 37.0, 37.2, 38.7 | sections 0.986; no reliable angle |
| Tapered (3 → 1 mm) | 2 / 2, 0 / 0 | 2 / 2, 0 / 0 | 0.000 / 0.000 | 0.009 / 0.022 | 0.011 / 0.030 | 0.016 / 0.026 | sections 0.960 | sections 0.954 |
| Arc (radius of curvature 15 mm, r 1.5) | 2 / 2, 0 / 0 | 2 / 2, 0 / 0 | 0.011 / 0.026 | 0.011 / 0.023 | 0.010 / 0.017 | 0.010 / 0.021 | sections 0.982 | sections 0.982 |
| Two tubes in contact (0.3 mm overlap) | 13 / 4, 10 / 0 | 9 / 4, 7 / 0 | 0.003 / 0.836 | 0.022 / 0.454 | 0.024 / 0.351 | 0.009 / 0.369 |  | sections 1.391 |
| Thin (r 0.56 mm = 0.8 of the 0.7 mm voxel) | 2 / 2, 0 / 0 | 2 / 2, 0 / 0 | 0.000 / 0.000 | 0.038 / 0.083 | 0.090 / 0.090 | 0.011 / 0.022 | sections 0.845 | sections 0.755 |
| Elliptic (semi-axes 3 × 1.2 mm) | 24 / 2, 11 / 0 | 14 / 2, 12 / 0 | 0.289 / 2.136 | 0.463 / 1.913 | n/a | n/a | aspect 0.376, Feret 2.35 / 6.08, area 0.936 | aspect 0.351, Feret 2.39 / 6.89, area 1.121 |
| Torus | 2 / –, 0 / – | 2 / –, 0 / – | n/a | n/a | n/a | n/a | field loops 1 | field loops 1 |

### What this says

- **Round tubes are traced right, off the lattice too.**
  - Topology is exact on every round phantom on both grids, including a tube of radius 0.8 voxel.
    Oblique, the trifurcation traces as two close bifurcations (2 junctions for 1), as vmtk
    splits one too.
  - The axis is within 0.01 mm (median) on the lattice and 0.04 mm oblique; the radius within
    0.02 mm on both (the thin tube reads 0.09 mm small on the lattice, 0.01 oblique).
  - **Sections read 1–2 % small against the true diameter** on both grids (0.982–0.991 on round
    tubes; 0.95–0.96 on the tapered one, whose median station lies where it is thinnest over a
    3 mm window; 0.76–0.85 on the thin tube, a section 1.6 voxels across).
  - Deflection angles are within 1° at 60° and 90° on both grids and at 30° on the lattice. The
    chords start at arc length r_J, outside the junction's ball (the first version started them
    at the junction point and read wide angles low: 73.6° for a T junction, 59° for 70°).
  - **At 30° the angle depends on where the junction lands.** Oblique, the tracer places the
    30° Y's junction 5.7 mm upstream of the true branch point (the daughters' tubes overlap for
    that long), and the deflection measured from there reads 7.4° and 7.7° for 15°. On the
    lattice the junction lands 0.8 mm off and the angles read right. For shallow bifurcations
    of overlapping tubes the junction's position, and so the angle, is not well determined. The
  table's rotation is the worst of four tried: three others read 15.3–15.9° or no reliable
  angle (the round-8 review).
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
    the contact shows as a loop in the field (`qc.json` `field_loop_count`); flagging the short
    junction-to-junction edges it produces is not done yet.
- **Loops are seen but not kept.**
  - The field's topology finds the torus's loop (genus 1).
  - The tracer builds trees, so the graph drops the cycle. The format allows cycles; the tracer
    does not produce them yet.
  - `qc.json` carries the field's loop count per tree (`field_loop_count`: 0–7 on the ladder below).

## 2. Second patient and reconstruction ladders

`thalweg run --ridge-passes 1 --root deepest` (the research reference settings, not the defaults:
§5 gives what the defaults change) was run on both patients' ladders: the same
scan reconstructed at 0.625 to 5 mm, each run through TotalSegmentator `lung_vessels`. Each case
took 24–42 s for the three trees (without the station table; the lung additions of §5b–§5e are
included). One bug surfaced on the way and was fixed: newer stores record one labeling scheme per
cascade stage, and the graph's `Source` expected a single one. Numbers regenerated 2026-09-30
with the current code. The deflection column is the median over branches with a defined angle; it
fell about 2° from the earlier table when angle chords moved outside the junction's ball (§1).
Every other column reproduced to the digit. Inlet rooting (the default) changes edge directions,
so Strahler lengths and deflections under the defaults differ from these; lengths, radii, areas
and the QC columns do not depend on the root.

| Run | Tree | Edges | Length (cm) | Radius p50 (mm) | Strahler ≥ 3 (cm) | Strahler ≥ 4 (cm) | Area median (mm²) | Deflection median (°) | Outside field (mm) | Field loops |
|---|---|---|---|---|---|---|---|---|---|---|
| C3N 0.625 | arteries | 1065 | 1052.0 | 1.55 | 228.8 | 95.3 | 6.63 | 42.8 | 2.2 | 4 |
| C3N 1.25 (lung kernel) | arteries | 1056 | 1041.0 | 1.55 | 233.3 | 91.3 | 6.42 | 42.7 | 0.5 | 2 |
| C3N 2 mm slab | arteries | 1002 | 1019.6 | 1.57 | 231.5 | 94.9 | 6.88 | 42.1 | 0.9 | 6 |
| C3N 3.75 | arteries | 552 | 694.2 | 1.73 | 153.4 | 43.2 | 10.04 | 41.9 | 0.5 | 3 |
| C3N 0.625 | veins | 950 | 937.9 | 1.58 | 207.7 | 86.6 | 6.49 | 44.4 | 0.6 | 2 |
| C3N 1.25 | veins | 933 | 911.5 | 1.57 | 203.9 | 71.7 | 6.38 | 45.5 | 1.2 | 3 |
| C3N 2 mm slab | veins | 900 | 905.0 | 1.60 | 201.9 | 72.0 | 6.58 | 45.0 | 1.1 | 2 |
| C3N 3.75 | veins | 496 | 621.9 | 1.74 | 134.5 | 55.9 | 9.96 | 39.9 | 3.3 | 2 |
| C3N 0.625 | airways | 265 | 343.7 | 1.17 | 62.8 | 28.5 | 4.52 | 27.5 | 2.0 | 2 |
| C3N 1.25 | airways | 251 | 313.9 | 1.20 | 59.0 | 31.6 | 4.91 | 27.9 | 1.8 | 3 |
| C3N 2 mm slab | airways | 229 | 316.5 | 1.23 | 56.9 | 28.4 | 4.97 | 27.2 | 9.0 | 1 |
| C3N 3.75 | airways | 212 | 215.2 | 1.47 | 43.7 | 25.0 | 10.04 | 30.9 | 1.2 | 0 |
| MSB 0.625 | arteries | 590 | 706.9 | 1.66 | 142.9 | 51.9 | 7.58 | 43.8 | 0.5 | 3 |
| MSB 1.25 | arteries | 592 | 709.8 | 1.66 | 146.7 | 53.3 | 7.95 | 43.1 | 1.7 | 5 |
| MSB 2 mm slab | arteries | 518 | 662.9 | 1.67 | 137.8 | 45.3 | 7.69 | 42.8 | 0.0 | 5 |
| MSB 5 mm (DLIR) | arteries | 240 | 324.7 | 1.99 | 66.8 | 25.7 | 13.32 | 45.8 | 1.7 | 7 |
| MSB 0.625 | veins | 467 | 536.4 | 1.63 | 125.1 | 44.6 | 7.26 | 44.5 | 3.5 | 5 |
| MSB 1.25 | veins | 474 | 557.0 | 1.64 | 127.7 | 43.6 | 7.41 | 44.6 | 0.5 | 1 |
| MSB 2 mm slab | veins | 427 | 508.6 | 1.65 | 115.5 | 37.3 | 7.49 | 44.4 | 3.1 | 2 |
| MSB 5 mm (DLIR) | veins | 196 | 251.8 | 2.02 | 45.3 | 22.9 | 16.35 | 40.8 | 0.4 | 1 |
| MSB 0.625 | airways | 162 | 186.4 | 1.34 | 35.7 | 21.7 | 6.25 | 33.4 | 0.2 | 7 |
| MSB 1.25 | airways | 151 | 178.0 | 1.38 | 38.6 | 19.7 | 5.89 | 35.3 | 0.0 | 4 |
| MSB 2 mm slab | airways | 126 | 160.2 | 1.47 | 32.4 | 15.9 | 7.28 | 31.7 | 0.7 | 2 |
| MSB 5 mm (DLIR) | airways | 79 | 84.2 | 1.96 | 23.4 | 9.2 | 16.66 | 32.1 | 0.0 | 0 |

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
- **The MSB airway root is the trachea, but not a clean inlet.** In all four MSB reconstructions
  the model's trachea tapers out inside the field instead of running off it (0 truncated ends),
  and the inlet is one of three prongs at the taper (widths 5.2, 4.1, 3.9 mm), so the tree has a
  spurious junction at depth 0 and its `bifurcation_depth` is offset by one against C3N, whose
  root is the truncated tracheal end. The same stores drop a separate component of 16.6 % of the
  airway lattice, 5 cm lateral of the trachea and touching the grid's top (`qc.json`
  `dropped_components`): probably mislabeled. Read `dropped_components.lattice_point_share` and the
  airway root's `attributes.end_kind` before comparing airway depths across cases.

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
| Flattened, semi-axes 3 × 1.5 mm | 0.260 / +0.062 | 0.269 / −0.056 | | | 0.258 / +0.060 |
| Flattened, semi-axes 4.5 × 1.5 mm | 0.221 / +0.073 | **1.005** / −0.087 | | | **1.020** / +0.030 |

**On flattened tubes the axis is ill-defined, and thalweg's wanders.** The inscribed radius of a
flat section is its minor semi-axis (1.5 mm here); both methods read it within 0.09 mm. At 2:1
both put the centerline 0.26 mm from the axis - the same place, so it is the sampled field's
(a 3 mm-deep section on a 0.7 mm grid), not either method. At 3:1 vmtk stays within 0.22 mm but
thalweg's centerline wanders across the flat width, a median 1.0 mm and up to 3.8 mm from the
axis: a flat lumen's medial set is a sheet, and the tracer's minimal path through it is not
held to its middle as vmtk's Voronoi-based path is (the same cause as the comb of side branches
in §1). Centerlines of flat tubes - esophagus, colon, compressed veins - need a rule that keeps
the path at the middle of the width; this is the tube track's open item.

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
  - C3N arteries' `length_outside_field_mm` goes from 2.2 to 4.0 mm, and one junction pair merges
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

| Case | Pi10 (mm) | Pi10 of a constant wall of the median thickness | WA%, Strahler 1 / 2 / 3 | Thickness (mm), Strahler 1 / 2 / 3 | Branches behind those medians, Strahler 1 / 2 / 3 |
|---|---|---|---|---|---|
| C3N-00704 0.625 | 4.17 (1151 of 2205 stations) | 4.40 | 84.7 / 72.8 / 60.6 | 1.41 / 1.25 / 1.12 | 75 / 19 / 7 |
| MSB-02664 0.625 | 4.34 (431 of 1103 stations) | 4.46 | 81.7 / 65.9 / 68.9 | 1.37 / 1.11 / 1.25 | 32 / 5 / 1 |

**The wall is at the model's resolution.** The measured thickness is nearly constant, 1.1–1.45 mm
at every order: about two voxels of the 0.7 mm grid, the thinnest shell the model draws. Its
correlation with lumen diameter is 0.07 and 0.02 on the two cases. Real small-airway walls are
thinner, so small-airway WA% reads high, the airway analog of the vessels' ~1 mm radius floor.

**Pi10 restates that constant thickness; it does not measure these airways.** A wall of the
median thickness t at every caliber has Pi10 = √(π t (10/π + t)): 4.40 and 4.46 mm, against the
fitted 4.17 and 4.34 (`summary.json` reports both, as `pi10_mm` and `constant_wall_pi10_mm`). The
fit also runs over every station, from lumens under a voxel (internal perimeter 0.75 mm) to the
trachea (52 mm); restricting it to perimeters of 6–20 mm moves C3N by 0.16 mm, as much as the
difference between the two cases (0.17). So a difference in Pi10 between cases reflects the
model's shell and the mix of stations, not the patients. It is above the ~3.6–3.8 mm usually
reported for healthy lungs on CT for the same reason.

**Under half of the stations have a wall measure, and the middle orders have the fewest.** A
station is measured only where the outer contour closes inside the section window. Where a
neighboring airway's wall touches (at and near every bifurcation), the contour runs on into the
neighbor and the station is dropped: 52 % of stations are measured on C3N and 39 % on MSB. By
Strahler order the measured share runs 56 / 40 / 25 / 8 / 59 / 99 % on C3N and 40 / 16 / 23 / 29 /
72 / 90 % on MSB: the trachea and main bronchi are nearly always measured, the orders between
them and the tips least. Dropping is the safe choice (two phantom airways with
touching walls give no wall stations, not wrong ones), but it biases the sample toward isolated
airways. A branch needs at least 3 wall stations to get wall measures: 132 of 265 (C3N) and 59 of
162 (MSB) branches have any wall station, and 105 and 42 have enough. The MSB medians for orders
2 and 3 rest on 5 branches and 1. `wall_station_count` beside `station_count` says how much each
branch's medians rest on.

## 5c. Bronchoarterial pairing

Each airway centerline sample is paired with the nearest artery sample within 8 mm that runs
parallel (|cos| ≥ 0.7). Each airway branch then gets the artery edge holding most of its paired
samples and the median bronchus-to-artery diameter ratio (`thalweg.pairing`; a synthetic
parallel pair gives exactly the radius ratio, and a crossing artery nearer than the partner does
not pair).

| Case | Airway samples paired | Paired branches | Ratio median | Paired branches above 1 | Ratio by Strahler 1 / 2 / 3 | Ratios resting on under 5 samples |
|---|---|---|---|---|---|---|
| C3N-00704 0.625 | 84 % | 235 | 0.58 | 4 % | 0.53 / 0.62 / 0.62 | 23 |
| MSB-02664 0.625 | 66 % | 118 | 0.51 | 7 % | 0.48 / 0.50 / 0.64 | 25 |

**`paired_artery_edge` is the nearest parallel artery, not a verified companion.** Measured by
the round-3 review on the two cases (C3N / MSB):

- both children of an airway bifurcation paired with one artery edge: 29 % / 33 % of bifurcations;
- a child airway's artery not downstream of its parent airway's artery: 25 % / 39 %;
- a branch's paired samples spread over a median of 3 / 2 artery edges, with 60 % / 74 % of them
  on the one reported.

A wider parent artery 3 mm away beats the true companion 4 mm away (a phantom gives 0.33 where
0.83 is right). Two columns say how far to trust a branch's pairing: `paired_sample_count`, and
`paired_artery_consistent`, true when the artery edge is that of the nearest paired ancestor
airway branch or lies downstream of it, and no airway branch outside its own line (a sibling, an
uncle, a cousin) has the same one. 126 of 235 paired branches pass on C3N and 45 of 118 on MSB.
The flag checks a branch against what lies above it, so a wrong pairing high in the tree passes
and fails its correctly paired descendants instead.

**The ratio is low because of the calibers the model draws, not because of mis-pairing.** The
medians sit below the ~0.65–0.7 usually reported for healthy lungs on CT. Over the consistent
branches alone the ratio is the same: 0.59 against 0.58 over all on C3N, 0.49 against 0.51 on MSB
(`summary.json` reports both medians). There the airway radius is about 1.1 mm against 1.9–2.0 mm
for the artery: the airway lumen class sits at the model's floor beside a wider artery class. As
with the walls, compare across cases on one model and grid.

**Making the pairing follow the trees makes it worse (tried 2026-09-30, not kept).** Each airway
branch was allowed to pair only with its parent's artery edge or edges downstream of it, walking
from the trachea. With the parent's majority edge as the anchor, the paired share of samples fell
from 84 % to 42 % (C3N) and 66 % to 30 % (MSB); siblings sharing one artery edge went from 33 of
114 bifurcations (29 %) to 26 of 63 (41 %), and from 18 of 54 (33 %) to 28 of 39 (72 %); the ratio stayed at 0.58 and moved
0.51 → 0.55. With the looser common ancestor of the parent's partners as the anchor: 59 % and
34 % paired, sharing 29 of 84 and 26 of 40. One early wrong partner locks a whole airway subtree
out of its arteries. A pairing that is right at the first generations needs anatomy (lobar and
segmental names), not nearest-parallel geometry.

## 5d. Artery/vein plausibility

`qc.json` `artery_vein` (`thalweg.plausibility`): the share of each tree's centerline inside the
crop stage's `pulmonary_vein` class.

| Case | Arteries inside | Veins inside | Plausible |
|---|---|---|---|
| C3N-00704 0.625 (misnamed crop classes, found by value) | 0.00 % | 5.4 % | yes |
| C3N-00704 1.25 | 0.06 % | 5.3 % | yes |
| C3N-00704 2 mm slab | 0.00 % | 5.6 % | yes |
| C3N-00704 3.75 | 0.00 % | 7.0 % | yes |
| MSB-02664 0.625 | 0.00 % | 7.4 % | yes |
| MSB-02664 1.25 | 0.00 % | 7.9 % | yes |
| MSB-02664 2 mm slab | 0.00 % | 7.8 % | yes |
| MSB-02664 5 mm | 0.00 % | 13.3 % | yes |

(The §2 runs. With the names swapped every case fails, by the rule; the test suite checks the
swap on the two 0.625 mm cases.)

A first version tested the veins' root against the class and failed every correct case: the
model's veins run on past the vein trunks into the left atrium, so the root lies 27–36 mm beyond
the class, inside `heart`. The length share separates the trees on all eight, thick slices
included. Eight reconstructions of two patients is thin evidence for the thresholds (veins ≥ 1 %,
arteries ≤ 2 %).

## 5e. Tree statistics and the radius interval

`summary.json` `<tree>.tree_statistics` (`thalweg.statistics`), default options (four ridge passes, inlet
root), 0.625 mm reconstructions:

| Case | Tree | Streams by Strahler order | Bifurcation ratio | Length ratio | Diameter ratio | Small-vessel volume fraction | Orientation entropy |
|---|---|---|---|---|---|---|---|
| C3N-00704 | arteries | 536, 160, 57, 22, 8, 2, 1 | 2.88 | 1.29 | 1.46 | 3.5 % | 0.992 |
| C3N-00704 | veins | 482, 142, 49, 16, 7, 3, 1 | 2.74 | 1.22 | 1.57 | 2.1 % | 0.993 |
| C3N-00704 | airways | 133, 48, 17, 5, 2, 1 | 2.73 | 1.47 | 1.43 | 13.9 % | 0.984 |
| MSB-02664 | arteries | 297, 91, 32, 10, 4, 1 | 3.05 | 1.44 | 1.59 | 1.4 % | 0.985 |
| MSB-02664 | veins | 236, 68, 24, 6, 2, 1 | 3.07 | 1.31 | 1.66 | 1.4 % | 0.985 |
| MSB-02664 | airways | 81, 26, 10, 4, 2, 1 | 2.40 | 1.44 | 1.36 | 7.9 % | 0.966 |

- **Horton's ratios are in the range reported for pulmonary trees:** bifurcation ratios near 3
  and diameter ratios near 1.5. On a synthetic perfect binary tree the code returns exactly 2.
- **The small-vessel fraction is tiny, and the model decides it.** A 5 mm² cross-section is a
  1.26 mm radius, just above the model's ~1 mm floor, so almost nothing falls under it. It cannot
  be compared with "BV5" from segmentations that reach smaller vessels.
- **Orientation entropy is saturated** (0.97–0.99): a lung tree points everywhere, so this
  statistic separates little here. It also rises with the number of segments (evenly spread
  directions read 0.67 at 20 segments, 0.97 at 300), so a small tree reads low for its size alone.
- **The per-point radius interval** (`radius_lower_mm`, `radius_upper_mm`: margin +2 and −2
  logits) has a half-width of 0.14–0.21 mm on the C3N arteries, widening slowly with radius: 1.01 mm
  reads 0.80–1.13, 1.64 reads 1.47–1.74, 3.56 reads 3.30–3.76. On the Y phantom it is 2 / slope, as it
  should be.

The full tier-1 run (three trees, four ridge passes, lobes, walls, pairing, statistics, QC) takes
about 60 s on C3N-00704 under load and writes 2.4 MB.

## 5f. The branch partition

`thalweg.partition` labels points with the branch whose tube function is lowest, vmtk's
branch-clipper rule without the surface (`thalweg.vmtk.partition`).

- **Against vmtkBranchClipper (vmtk 1.5.2, C3N-00704 subtree).** The clipper's output holds the
  input surface's vertices (21,221 of its 21,236 reach the output) and the points it inserts along
  each cut. At every one of those vertices the label is vmtk's, with vmtk's centerlines and with
  ours. The inserted points (about 7,000) lie where two groups' values tie, which is what a cut
  is; there 3,225 (vmtk's centerlines) and 3,251 (ours) carry the other neighbor's label, by a
  value gap of 4.7e-8 and 1.5e-4 mm² in the median and 0.34 and 0.58 mm² at most (the clipper
  places them by linear interpolation along mesh edges).
- **The pruned search is exact.** It gives the labels and values of evaluating every (point,
  segment) pair, on random tubes of 0.3–9 mm radius and on the clipper's points.
- **As a volume, by graph edge** (`thalweg run --branch-volumes`, C3N-00704 0.625 mm):

  | Tree | Lattice points inside | Labeled (traced piece) | Edges with volume | Partition volume | Σ π r² ds along the centerlines | Time |
  |---|---|---|---|---|---|---|
  | arteries | 305,946 | 303,418 | 1059 of 1064 | 150.4 ml | 176.2 ml | 18–23 s |
  | airways | 83,974 | 83,652 | 265 of 265 | 41.5 ml | 42.8 ml | 1.4–1.8 s |

  The labeled points are exactly the tracer's piece (`traced_lattice_point_count`): pieces are
  told apart by the field's own connectivity, as the tracer does it.

  The centerline integral counts junction volume once per edge that meets there, and takes each
  section as its inscribed circle (smaller than a section that is not round); on the arteries the
  first effect wins and it reads 17 % above the partition. The partition's volume is the lattice count times
  the voxel volume, so a branch thinner than a voxel can get none (5 short artery edges here).
  The arteries take longest because their trunk's radius (14 mm) widens the search.

## 5g. Branch metrics and wall maps

- **vmtk's branch metrics** (`thalweg.vmtk.metrics`), at the 21,221 vertices of the surface vmtk
  mapped on the C3N-00704 subtree: with vmtk's end-point defect switched on, AbscissaMetric
  agrees to 3e-14 mm and AngularMetric to 5e-13 rad. Interpolating along each segment, as vmtk
  meant to, moves the abscissa by a median 0.11 mm and at most 0.30 mm, one centerline step. The
  angle moves too, because the normal is interpolated instead of read at the segment's end: not
  at all in the median, 1.9° at the 99th percentile, more than 5.7° at 44 vertices and up to 159°
  at a few close to the centerline's end points, where the angle is ill-conditioned.
- **Rays** (`thalweg.kernel.rays`): in a phantom tube of radius 2 mm on a 0.7 mm grid, the wall
  is found at 1.97–2.00 mm from the axis, and at the right distances from a point off the axis.
  On a margin that is linear along a ray the crossing is exact for any step. A ray that starts
  outside, does not leave within its reach, or runs off the grid while still inside has no
  crossing.
- **Wall maps in vmtk's coordinates** against vmtk's DistanceToCenterlines at the same (group,
  abscissa, angle), 13,443 surface vertices away from the groups' ends: median difference
  −0.009 mm, median |difference| 0.025 mm, 90th percentile 0.084 mm. 92 groups, 177,336 rays,
  2.4 s.
- **Wall maps per graph edge** (`thalweg export --wall-maps`): the radius is measured from the
  traced centerline, which lies within about 0.1 mm of a phantom tube's axis, so a round tube's
  map varies by that much around the angle while its mean and the wall points themselves are the
  cylinder's (within 0.03 and 0.05 mm). On the C3N-00704 arteries: 949 edges, 1.46 million rays
  at 0.5 mm × 5°, 27 s, 6.2 MB; 3.8 % of the rays pass through an ostium (no wall, or one beyond
  1.8 × the station's median radius). 10 of the 20,291 stations lie outside the structure (the
  smoothed path cuts a corner): all their rays are empty, and they are counted apart.

## 5h. Boundary reference systems and flow extensions

The C3N-00704 subtree, opened at the nine cuts vmtk was given (an inlet and eight outlets,
`research/vessels/flow_ext.py`), with `export.surface` doing the cutting.

- **Rings.** vmtk's boundary barycenter and radius are averages over the ring's vertices. Computed
  that way (`vmtk_vertex_mean=True`) ours equal vmtk's on eight of the nine rings (to 1e-4 mm) and
  differ by 0.016 mm and 0.012 mm on the ninth: the exact plane clip makes the same ring as VTK's
  clipper. The default averages along the ring's length instead (the polygon's perimeter centroid,
  and the mean distance from it integrated along every edge), which does not depend on where the
  mesh happens to put vertices: it moves the barycenter by 0.05–0.26 mm and the radius by
  0.01–0.04 mm on these rings. On a ring with 40 of its 42 vertices on one side the vertex average
  is 0.8 mm off; the default is exact.
- **Extensions** (ratio 5, transition 0.25, as vmtk was run). In the straight part, 13,639 of
  vmtk's extension vertices: their distance from our cylinder's axis differs from our radius by a
  median 0.02 mm (90th percentile 0.09 mm), which is the ring difference above. In the transition
  (a smoothstep blend here, a thin-plate spline in vmtk), vmtk's 4,053 vertices lie within a
  median 0.03 mm of our surface (90th percentile 0.04 mm, at most 0.2 mm; the surface sampled
  densely, so these are upper bounds).
- **A ring of any shape.** Each ring vertex goes to the circle point at its own fraction of the
  ring's length. Sending vertices out radially from the barycenter, as the first version did,
  folded the tube on rings that are not star-shaped (an L- or C-shaped section: a third to a half
  of the tube's faces inward, and the manifold check passed it); one of the 436 artery caps was
  such a ring.
- **No thin triangles along the tubes.** Over the transition the vertices also slide round the
  circle to even spacing (both spacings increase round the ring, so the circle's points stay in
  order). Before this a ring with a very short edge carried a strip of thin triangles the
  whole length of its tube: faces under 1e-5 mm², 43 on the C3N-00704 airways and 234 on the
  arteries, now none. Poorly shaped faces (quality under 0.1) remain only in the first 15 % of
  a tube, next to the ring whose own edges are that short: 73 of 90,358 new faces on the airways,
  766 of 544,228 on the arteries; the straight parts are clean (median quality 0.87).
- **Rings the blend would pass through.** Keeping the circle's points in order does not keep
  every intermediate layer a simple polygon: for a ring whose barycenter lies outside it (a thin C,
  a hook) some layers cross themselves - 27 and 38 of 100 blend steps on the round-7 review's
  test rings - with every triangle still facing out, so the manifold check does not see it.
  Testing the layers alone was not enough: the round-8 review found C rings (outer radius 3 mm,
  gaps 30–120°) whose layers were each simple while the triangle strips between them crossed
  (12–58 crossing face pairs), and a short transition can morph the whole ring inside the first
  strip. Now a cap is extruded unchanged (a straight prism of its own section,
  `extension_end_shape` `ring`) when its ring does not contain its barycenter, or when a layer or
  a section through the strips between layers (7 per strip) is not simple. A plus-shaped ring (not
  convex, barycenter inside) still morphs, and its strips sampled 60 times each stay simple. None
  of the 545 caps of the C3N-00704 airways and arteries needed the fallback.
- **The mesh stays closed and manifold**, and its volume grows by the tubes' (π R² L per cap,
  within 3 % on the Y phantom).
- **Collisions.** An extension is straight and knows nothing of its surroundings.
  `export.extension_collisions` counts, per extension, the tube vertices that lie inside the
  structure more than one radius past the point where the extension's axis leaves it; the CLI
  prints how many extensions have any, and the sidecar carries the count
  (`extension_vertices_inside_structure`). It samples the field at vertices; it does not test
  triangles against each other, and it does not see two extensions crossing in the open.
- **Whole trees** (`thalweg export --mesh --flow-extensions 5`, all end kinds), C3N-00704:

  | Tree | Caps | Median extension length | Extensions running back into the structure | Faces | Time |
  |---|---|---|---|---|---|
  | airways | 109 | 3.8 mm | 13 | 0.24 million | 2.7 s |
  | arteries | 436 | 6.1 mm | 7 | 1.03 million | 9.7 s |

## 5i. Bifurcation sections

`branching.bifurcation_sections` places vmtk's bifurcation sections (`thalweg.vmtk.sections`) and
cuts them from the field. The oracle (`tests/oracle/vmtk_centerline_oracle.py sections`) runs
vmtkBranchClipper and vmtkBifurcationSections on the phantom tree, on a surface marching-cubed
from a field that thalweg's test rebuilds exactly (a union of spheres along the centerlines).

- **Placement.** With vmtk's defects on (`vmtk_interp`, `vmtk_steps`), the ten section points and
  normals at one and at two distance spheres are vmtk's to 1e-10 mm; the default moves the points
  by at most 0.2 µm here.
- **vmtk's measures** of its own section polygons (area; MinSize and MaxSize, the calipers through
  the center along the nearest and farthest boundary points; Shape, their ratio) are recomputed
  exactly.
- **Cut from the field instead of the surface**, on the same field: areas within 0.3 % and
  calipers within 0.03 mm on every closed section. At one sphere, one section's plane also
  crosses a neighboring daughter at the trifurcation: vmtk cuts only its own group's surface and
  returns an open polygon, the field's contour would take in the neighbor (82 % more area), and
  both flag it as not closed. A section that is not closed (or whose point lies outside the
  structure) gets no sizes: they would measure the section window.
- **Whole trees** (`thalweg export --bifurcation-sections`), C3N-00704: airways 366 sections, 309
  closed, 18 s; arteries 1,302 sections, 1,172 closed, 70 s (mostly vmtk's branch extraction on
  the whole tree). Median shape (MinSize / MaxSize) of the closed ones: 0.82 and 0.88.

## 5j. Curvature, distance to centerlines, tube function, straightened view

- **Mean curvature** (`kernel.curvature`, a weighted quadric fit to the unclipped margin samples
  within 2.8 mm): on tubes of radius 0.75, 1.5 and 3 mm and spheres of 2 and 4 mm, oblique to an
  anisotropic 0.62–0.7 mm grid, the median over the zero set's vertices is 1.02–1.04 × the truth,
  with the 10th–90th percentiles within 0.99–1.05. A cavity reads negative. On the C3N-00704
  meshes (`thalweg export --mesh --curvature`, without flow extensions) every vertex gets a
  value; the median is 0.26 (airways) and 0.24 (arteries) 1/mm, a tube of about 2 mm radius. The
  fit averages over its 2.8 mm window, so where the curvature changes within it (a saddle, a
  bifurcation's crotch) it reads low: the inner equator of a torus of radii 3 and 2 mm (H =
  −0.25) reads −0.09.
- **Distance to centerlines** (`vmtkdistancetocenterlines`, its defaults: the Euclidean distance to
  the nearest centerline point): at 5,690 of the 28,449 surface points vmtk mapped on the
  C3N-00704 subtree, from the centerlines vmtk used, ours equals vmtk's to 1e-9 mm (1e-14 in
  practice). On the whole-tree meshes the wall lies a median 0.13 mm (airways) and 0.09 mm
  (arteries) beyond the traced radius. With the partition's pruning this runs on the arteries'
  232,000 vertices in the 25 s the whole export takes.
- **Tube function**: on a straight tube it is d² − r² at distance d from the axis, exactly.
- **Straightened view**: a tube of radius 2 mm bent through a quarter circle straightens to
  centered disks of area π r² within 3 %, with arc length 10π mm and no twist between sections.

## 5k. Labelmaps and distance images as input (degraded mode)

`thalweg.volume.VolumeStore` reads a labelmap or a signed distance image (SimpleITK: the `volumes`
extra) and gives each structure a field like the model's margin: the mask's signed distance in
mm (the wall half a voxel out), times 10 logits/mm, clipped at ±8. Every verb runs on it.

- **Geometry**: on an oblique, anisotropic grid every voxel's world point equals SimpleITK's (the
  round-8 review: NIfTI, NRRD and MHA, both handednesses, spacing 0.7 × 1.3 × 2.9 mm, one-slice
  images and signed label types, all to 1e-14 mm).
- **Distance to the staircase**, exact across any grid axis: each voxel center's distance to the
  box of the nearest voxel of the other kind. The first version subtracted half the finest
  spacing everywhere, which put centers across a 2 mm axis of a 2 × 0.5 × 0.5 mm grid 0.75 mm too
  far from the wall (the zero set was right; the wall slope and the ±2 logit levels were not).
- **The Y phantom as a labelmap**: three branches, radii below the true 1.8, 2.2 and 3.0 mm by
  0.11–0.13 mm at 0.35 mm voxels and 0.13–0.27 mm at 0.7 mm - up to half a voxel, the inscribed
  ball of a staircase touching its inner corners, not an offset. The field of the same phantom
  reads 1.79, 2.20 and 3.01.
- **C3N-00704, the store's own labelmap** (the argmax of its logits) against its ranked field:
  airways 270 edges and 3,417 mm against 265 and 3,392 mm; arteries 1,067 and 10,531 mm against
  1,064 and 10,360 mm (on its 0.7 × 0.7 × 1.0 mm grid). The centerlines lie a median 0.2 mm apart
  (90th percentile 0.6 mm), and the labelmap's radius is a median 0.05–0.06 mm smaller. So a labelmap gives nearly the same tree, a
  little rougher.
- **DICOM SEG** (read with highdicom, the `dicom` extra): a two-segment SEG (overlapping
  segments; binary and fractional) of synthetic oblique, anisotropic CT slices. Every segmented
  voxel lands inside the true tube in world space and no other voxel is labeled; the overlap is
  kept (each segment is its own mask); the trunk's centerline lies a median 0.2 mm from the true
  axis.
- **A real IDC SEG**: NLST patient 217076's TotalSegmentator (v1.5.6) segmentation of the CT
  series thalweg's demo stores come from (IDC, `nlst`, `totalsegmentator_ct_segmentations`, CC BY
  4.0; 80 binary segments in 5,695 frames, 190 MB). It opens in 3.4 s; each segment is named by its
  label ("Aorta", "Pulmonary artery", "Left Upper lobe of lung", ...). Against the source CT,
  read separately with SimpleITK:
  - the structures sit on the right tissue: trachea voxels a median −982 HU, aorta 37, liver 66,
    upper lung lobe −879, T8 vertebra 254;
  - their boundaries sit on the CT's edges: the share of trachea, aorta and T8 boundary voxels
    whose HU fits the tissue is highest unshifted and drops when the segmentation is moved 2 mm
    along any axis (trachea 0.70 against 0.55–0.61, T8 0.87 against 0.64–0.72);
  - left is left: the spleen and the left lung lobes lie at +x (LPS), the liver and the right
    lobes at −x.

  The round-8 review found the SEG lattice lands on whole CT voxels (no fractional offset; the
  SEG's rows run opposite to the CT's, which the geometry carries), and found the synthetic edge
  cases exact: an empty segment or middle slice, missing source slices, sparse frames,
  fractional values up to 100, a label-map SEG, reversed source order.

  `thalweg centerlines` traces the aorta (one tube, 39.5 cm), the pulmonary artery (8 tips) and
  the trachea (12 tips) from the SEG in 14 s. The distance transform runs on each mask's
  bounding box only (the field is flat beyond 0.8 mm of the wall), which took a segment of this
  512 × 512 × 249 grid from 12 s to under 1 s with an identical result.
- **What degraded mode cannot give**: the model's interval. The ±2 logit levels become ±0.2 mm
  offsets of the wall - a convention, not an uncertainty. The graph's
  `source.labeling_scheme` says `degraded:labelmap`, `degraded:sdf` or `degraded:dicom-seg`.

## 5l. A 0-D flow model (svZeroDSolver input)

`thalweg export --zero-d M.json` writes the structure as an svZeroDSolver network (`thalweg.solver`):
one `BloodVessel` per edge with R = 8μ/π ∫ds/r⁴ and L = ρ/π ∫ds/r² along its traced radii (CGS;
blood μ 0.04 P, ρ 1.06 g/cm³; a rigid wall unless E·h is given), a junction per branching node,
a steady inflow at the root and one resistance per outlet as placeholders.

- A straight tube gives exactly the Poiseuille resistance and inductance; a tapering one the
  closed-form integral exactly (the radius is taken as linear between samples; a trapezoid rule,
  as first written, read a 3 → 0.5 mm taper 6 % high).
- Solving the network for steady flow (`solver.steady_pressures`, resistances only): flows add up
  at every junction and at the outlets (50 mL/s in, 50.000 out on the C3N-00704 arteries: 1,064
  vessels, 528 junctions, 536 outlets), and every vessel's pressure drop is its flow times its
  resistance. The round-8 review matched it against an independent nodal solve to 1.5e-13 on
  20 random trees.
- **Not run through svZeroDSolver itself**: it is not installed here. The file follows its
  documented schema (`simulation_parameters`, `boundary_conditions`, `junctions`, `vessels`) and
  carries a `thalweg` block the solver ignores.

## 6. Not yet validated

- **Thin-caliber ground truth.** The ~1 mm radius floor belongs to the model; checking the
  method below it needs a phantom scan or a gated high-resolution acquisition.
- **Flat tubes.** vmtk against thalweg on flattened phantoms is now measured (§4): the radius
  agrees, the centerline of a 3:1 tube does not stay at its middle. Fixing that is the tube
  track's work, not validation.
- **Off the lattice** (§1, `--oblique`): done for every phantom; sections against the true
  diameter read 1–2 % small on round tubes on both grids.
- **Degraded input** (§5k) is checked on one phantom and one case's own labelmap, not on
  labelmaps from other tools.
