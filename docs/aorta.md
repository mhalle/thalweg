# Aorta measurements: design note (parked draft, 2026-10-01)

> **Status: parked (2026-10-01).** The user set this aside as a research topic rather than a
> straightforward build. Nothing here is built. Kept as the starting point if the thread resumes;
> the open decisions at the end are still open.

The first structure-specific analysis after vmtk parity (`docs/port-plan.md`, "Structure-specific
analyses"). It turns the aorta's centerline and sections into the measurements clinicians and
device planners use: diameters at standard landmarks, the largest diameter per segment, and the
lengths between landmarks. Each value carries the model's own interval and says how its landmark
was placed.

## What already exists

- **The aorta's centerline**, traced from the field with vessel settings (length pruning, no
  recentering), and re-rooted at its inlet. On idc-torso1 it is 4 edges, 53.9 cm: the descending
  aorta runs 436 mm to the iliac bifurcation, and the aortic root traces as two 33 mm lobes (the
  sinus bulb is wider than it is long; validation §3).
- **Sections perpendicular to the path** at any station: area, equivalent diameter, minimum and
  maximum caliper (Feret) width, aspect ratio, centroid offset, and the areas at ±2 logits
  (`kernel.sections`, `measure`).
- **Named neighbors in the same store.** The `total` store holds `heart`, `brachiocephalic_trunk`,
  `common_carotid_artery_left`, `subclavian_artery_left`, `iliac_artery_left/right`, the kidneys
  and the vertebrae T1-L5. The `trunk_cavities` store holds the thoracic and abdominal cavities,
  whose shared boundary is the diaphragm. There are no renal, celiac or mesenteric arteries in
  `total`.

## What to build

### 1. The main path

The aorta's measurements run along one path: from the aortic valve plane to the iliac
bifurcation. The traced tree gives more than that (root lobes, arch branches the model labels as
aorta where they leave it), so the path is chosen, not taken whole:

- **Start, the valve plane**: where the aorta's field meets the heart's. The two classes' shared
  boundary (where both margins are near zero) is a surface patch; its centroid and normal give the
  plane, and the path starts at the station nearest it.
- **End, the bifurcation**: where the aorta meets both iliac arteries. The path ends where the
  aorta's section first touches both iliacs' fields.
- **Between them**: the graph's path between the edges nearest those two points. The root lobes
  and any stubs are off it.

### 2. Landmarks

Each landmark is an arc position on the main path, with how it was placed:

| Landmark | Placed by | How |
|---|---|---|
| Annulus / valve plane | neighbor | aorta-heart boundary (above) |
| Sinuses of Valsalva | shape | the largest section within the first 40 mm |
| Sinotubular junction | shape | the narrowest section between the sinuses and the ascending aorta's widest point |
| Mid-ascending | shape | midway along the path from the sinotubular junction to the brachiocephalic origin |
| Brachiocephalic origin (proximal arch) | neighbor | where the brachiocephalic trunk's field meets the aorta's: the arc position of their shared boundary's centroid |
| Left common carotid origin | neighbor | the same, for the left common carotid artery |
| Left subclavian origin | neighbor | the same, for the left subclavian artery |
| Mid-arch | neighbor | midway between the left common carotid and left subclavian origins |
| Proximal descending | neighbor | 20 mm along the path beyond the left subclavian origin |
| Diaphragm (hiatus) | neighbor | where the path crosses from the thoracic cavity into the abdominal cavity (`trunk_cavities` store); without that store, reported missing (no vertebral proxy: see below) |
| Renal level | proxy | each kidney's level along the path, reported separately: the station nearest the plane through that kidney's centroid, normal to the path (the renal arteries are not labeled); infrarenal = 10 mm below the lower of the two |
| Bifurcation | neighbor | aorta-iliac contact (above) |

A landmark whose neighbor is missing from the store, or does not touch the aorta, is reported as
missing with the reason, never guessed. Proxy landmarks say so.

**No landmark is placed by vertebral level.** TotalSegmentator's vertebra names are a fixed
7 cervical / 12 thoracic / 5 lumbar template, and it forces numbering variants into it - a name
duplicated, a name skipped, one vertebra split across two names, two merged under one (the
2026-09-24 spine survey: about 5 % of countable scans have a true numbering variant, plus
labeling errors; the NLST demo case skips T3, so its lower thoracic names are off by one). And
the anatomy varies on its own: the aortic hiatus is classically at T12 but lies a level above or
below it in many people. A "T12" landmark would compound the two. Vertebral levels may still be
reported beside a landmark, as information, with that caveat.

### 3. Measurements

At each landmark, on the plane perpendicular to the path:

- **diameters**: the equivalent diameter (from the area), and the minimum and maximum caliper
  widths - the clinical "maximum diameter perpendicular to the centerline" is the maximum
  caliper width;
- **the interval**: the same diameters at ±2 logits;
- **area, aspect ratio, and whether the contour closed** (a section that also cuts a branch
  origin is flagged rather than measured as if round).

Along the path:

- **the diameter profile**: the existing station table, with arc length measured from the valve
  plane and the segment each station lies in;
- **the largest diameter per segment** (root, ascending, arch, descending thoracic, abdominal)
  and where it lies;
- **lengths between landmarks** along the path (the stent-graft planning lengths, e.g. left
  subclavian to diaphragm, renal level to bifurcation).

### 4. Output and interface

- `thalweg aorta STORE -o aorta.json [--graph G] [--cavities TRUNK_CAVITIES_STORE]`: traces the
  aorta (or reads it from G) and writes the landmarks, per-segment maxima and lengths as JSON,
  plus `--stations` for the profile table.
- `thalweg run` adds the same block to `summary.json` when the aorta is among its structures.
- Library: `thalweg.aorta.measure(graph, store, ...)` returns the same dict.

Keys follow the naming rule (whole words, both ends of a range): e.g.
`maximum_caliper_diameter_mm`, `maximum_caliper_diameter_lower_mm` /
`maximum_caliper_diameter_upper_mm`, `arc_length_from_valve_mm`, `placed_by` (`neighbor`,
`shape`, `proxy`).

## Validation

- **A phantom aorta with known answers**: a root bulb (sinus 36 mm) narrowing to a sinotubular
  junction (28 mm), an ascending segment (32 mm), an arch with three branch tubes at known arc
  positions, a descending aorta tapering to 20 mm, and two iliac tubes; heart, cavity and kidney
  stand-ins as separate fields. Pass: landmark arc positions within 5 mm, diameters within
  0.5 mm or 2 %, on the lattice and oblique/anisotropic grids.
- **The two demo cases** (idc-torso1, nlst-217076; both `total` and `trunk_cavities`): every
  landmark found or reported missing with a reason; diameters within published adult ranges
  (a sanity check, not ground truth); the interval width at each landmark.
- **Optional, needs you**: a few landmark diameters measured by hand in Slicer on one case, to
  compare against.

## Decisions needed

1. **Which boundary.** TotalSegmentator's `aorta` class is the model's boundary, which on
   contrast CT is close to the lumen and on non-contrast CT is closer to the outer wall.
   Guidelines measure outer wall to outer wall. The proposal: report the model's boundary, say so
   in the output, and not adjust it.
2. **Proxies.** No vertebral proxies (above): the diaphragm comes from the cavities store or is
   missing. The renal level has no true neighbor in `total` (no renal arteries); the kidney
   proxy is name-free but anatomically loose (renal arteries arise from about L1 to L2, often
   unequally, with accessory arteries common). Use it, marked as a proxy, or report the renal
   level as missing until a model labels the renal arteries? The proposal: use it, marked, and
   give the two kidneys' levels separately rather than one plane.
3. **Interface.** A separate `thalweg aorta` verb plus a block in `run`'s summary, as above?
4. **Scope of the first version.** Landmarks, per-segment maxima and lengths. Deferred: aneurysm
   sac volume and a fitted healthy-reference profile (for "dilated by x %"), which need a
   reference model to be credible.

## Not in scope

The heart's own measurements (valve, chambers), the pulmonary artery, branch-vessel analysis
beyond their origins, and anything needing contrast-specific classes (thrombus, false lumen).
