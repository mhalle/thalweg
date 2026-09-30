# thalweg

Centerlines, branches and wall geometry of tubular structures read from a segmentation
model's continuous field (rankfield margins), never from a triangulated surface except at
export. A successor to vmtk. Private, Apache-2.0. Incubated in medseg (2026-09-23/24).

**Status (2026-09-30):** the first version of the library is on `main` (checked out in the
worktree `.worktrees/port`, excluded locally; the main folder is another session's branch). `docs/port-plan.md` has the per-module status and the phase
order, `docs/validation.md` the behavior beyond the build case, `docs/format/thalweg-json.md`
the graph format. The research evidence is in `docs/vmtk-successor.md` §12–13 (read §6 for the
package design). `research/vessels/` stays the frozen reference: kernel ports reproduce it
exactly (tests marked `data`).

## Layout

- `research/vessels/` — the incubation scripts, **copied unchanged** from medseg (branch
  `vessels-incubation`, `a1a493a`, which stays a branch on purpose). They are the reference the
  port is checked against: a ported function must reproduce its script's output. Do not refactor
  them in place. Their docstrings still say `bench/vessels/...`; the depth is the same, so the
  relative paths still work from `research/vessels/`.
- `explorations/` — kept here but off the main line (not ported, no API promise): vascular
  catchments and named lung segments (`explorations/catchments/`), straightened 3D rendering
  (`explorations/rendering/render_straight.py`). They get stores, fields and centerlines from the
  library through `explorations/_thalweg.py` (centerlines with `ridge_passes=1`, the research
  reference; outputs verified identical to the research-script versions, 2026-09-30) and still take
  `DATA`, `LADDERS` and the CT reader from `research/vessels/`. Run them with haversack's venv.
- `docs/` — the design note, the vmtk comparison, the SlicerHeart write-up, and
  `deliverables.md` (proposed batch product: a 2–5 MB core package per case, opt-in extras, on-demand queries).
- `src/thalweg/` — `kernel/` (numpy/scipy, no files or names), `vmtk/` (vmtk ports, numpy
  only), and the pipeline modules (store, centerlines, graph, adapters, branching, measure, lobes,
  pairing, plausibility, statistics, case, export, cli). `tests/test_layering.py` enforces the split.
- `tests/` — `fixtures/vmtk_oracle/phantom/` (frozen vmtk output on a synthetic tree, in git);
  `oracle/vmtk_centerline_oracle.py` regenerates it and the case oracle
  (`$VESSELS_DATA/oracle/C3N-00704_ctpa0625/`) in the isolated vmtk env.
- `validation/` — phase 5 scripts: `phantom_suite.py` (analytic phantoms) and `vmtk_phantom.py`
  (vmtk vs thalweg vs the truth).

## Environments

- thalweg itself: `uv sync --extra test --extra tables` in the repo (its own `.venv`, Python
  3.12); `uv run pytest` (data tests skip without `~/tmp/data/vessels`), `uv run thalweg ...`.
- Research scripts: **haversack's venv**, from `research/vessels/`:
  `../../../haversack/.venv/bin/python script.py [RUN]`. Never a bare `python3` (Homebrew's).
  It has rankfield, duckn, torch (MPS), scipy, skimage, SimpleITK.
- vmtk (oracle only, never a dependency): an isolated env, run from the repo root:
  `uv run --no-project --python 3.12 --with vmtk --with scipy python research/vessels/vmtk_*.py`
  (vmtk 1.5.2 wheels bring VTK 9.6.2).
- Stores come from haversack:
  `haversack segment <dicom dir | nii> --task ts.v2:lung_vessels -o <run>.lung_vessels.duckn.zip`
  (the `ts.v2:` catalog prefix is required). Format 0.4 on new stores.

## Data (outside git and Dropbox, always)

`research/vessels/_data.py`: caches, stores and figures in `$VESSELS_DATA` =
`~/tmp/data/vessels` (stores in `runs/`, per-run npz, `*_voxel.*` outputs of the old voxel
mode); DICOM in `$VESSELS_DICOM` = `~/tmp/data/idc_vessels` (`manifest.csv`, `CITATIONS.txt`,
`derived/*_slab2mm.nii.gz`). Cases (`cases.py` LADDERS): **C3N-00704** (cptac_luad; CTPA 0.625,
LUNG 1.25, 2 mm slab average, 3.75) and **MSB-02664** (cmb_brca; CTA-PE 0.625, 1.25, 2 mm slab,
5.0 DLIR), CC BY 4.0, from IDC; plus the idc-torso1 demo store
(`haversack/data/duckn_demo/idc-torso1/lung_vessels.duckn`). The vmtk comparisons use the
C3N-00704 0.625 left-lung subtree (51 tips), the MSB-02664 0.625 subtree (57 tips) and analytic
phantoms (`validation/vmtk_phantom.py`). `vmtk_prep.py` keeps the surface only within our radius +
1.5 mm of our centerline, so vmtk sees a cut wall on wide vessels: exclude points whose sphere
touches the cut (docs/validation.md §4). `.gitignore` refuses data extensions. Downloads
need the user's permission.

## Established facts — don't relitigate

- **The ~1 mm radius floor belongs to the input model** (TotalSegmentator `lung_vessels`, 0.7 mm
  grid), not the method: vmtk fed the same surface matches our radii to 0.03 mm. Same-patient
  thin vs thick: radius at the same world points moves ≤ 0.05 mm up to 2 mm slices; at 3.75–5 mm
  thin vessels are DROPPED, not fattened. Never list the floor as a method difference.
- **Field connectivity** on the native grid (`_topo.py`): face neighbors both > 0; face diagonals
  by the asymptotic decider fa·fb > fc·fd; body diagonals by cell closure + a 9³ trilinear sample;
  loops = genus of the Lewiner marching-cubes surface. Verified 1:1 against surface components.
  One mode (the field) everywhere; the voxel mode is kept only for comparison (`GRAPH=voxel`).
- Field radius is ~2× as repeatable as voxel EDT across reconstructions (r ≥ 1.25 mm). Use mean
  |Δ| / RMS, never MAD (EDT is quantized).
- vs vmtk (same subtree): centerlines 0.086 mm, radius +0.029 mm, 50/51 routes (with 1 ridge pass;
  4 passes give 0.060 mm and -0.006 mm - the +0.029 was the refinement's quantization, docs/validation.md §4); branch clipper
  reproduced at 100 % of original vertices (2 s vs 185–228 s); wall coordinates bit-exact;
  wall maps 0.02–0.03 mm; curvature 6× more repeatable; flow extensions 0.02–0.03 mm. Whole
  subtree analysis ~5.5 s vs ~240 s (95 % of vmtk's is the clipper).
- **vmtk defect:** `vtkvmtkCenterlineUtilities::InterpolateTuple` calls `GetTuple()` twice into
  one buffer, so every "interpolated" radius/abscissa/normal is the segment's END value
  (wall-map abscissa +0.114 mm median, angle up to 1.9°; in the offset filter, abscissas up to
  0.29 mm and normals up to 12.4°). Correct interpolation is the default;
  `vmtk_interp=True` reproduces vmtk only for comparison. Not reported upstream (user's call).
- Curvature: the margin is ~10.6 logit/mm, clipped at ±8, so finite-difference stencils hit the
  plateau and read ~12 % low. Use the Gaussian-weighted quadric fit to unclipped samples (2.8 mm).
- vmtk's centerlines are NOT used by our pipeline (only in comparisons). vmtk's centerline-only
  processing is now ported to numpy (`thalweg.vmtk`, 2026-09-30), so vmtk runs only in the oracle
  script. Our field -> graph -> vmtk convention -> ported extractor reproduces vmtk's own groups
  exactly on the C3N-00704 subtree (`tests/test_branching.py`).
- **Trees are rooted at their inlet by default (2026-09-30):** the widest end running off the
  field, else the end of the widest terminal edge (`centerlines.inlet`, `reroot`); `--root deepest`
  keeps the tracer's deepest point (the research reference, which the vmtk-comparison tests pin).
- **Tracer defaults (decided 2026-09-30):** `ridge_passes=4` (coarse-to-fine radius refinement:
  removes a 0.03-0.07 mm radius deficit, matches vmtk, 1.85x trace time), `prune="length"` (the
  reference; `wall` is an option for flat lumens). `ridge_passes=1` is the research reference and
  what the reproduction tests pin. docs/validation.md §5.
- **The vmtk port's flag convention:** every public `thalweg.vmtk` function defaults to the
  correct behavior; `vmtk_<name>=True` reproduces a vmtk or VTK defect (`vmtk_float32`,
  `vmtk_steps`, `vmtk_merge`, `vmtk_last_tract`, `vmtk_interp`, `vmtk_fallback`, `vmtk_cell_data`,
  `vmtk_discard_smoothing`, `vmtk_two_point_cells`), each documented with its measured effect in
  its module docstring. Small vmtk fixtures (`tests/fixtures/vmtk_oracle/<name>/`, ~0.4 MB,
  `vmtk_centerline_oracle.py small`) protect the grouping rules without case data. `thalweg.vmtk.VMTK_FLAGS`
  lists them; `branching.vmtk_branching(vmtk_compatible=True)` turns them all on. Oracle tests
  pass every flag of their stage. The residual 1e-14 differences are FMA contraction in vmtk's
  arm64 build (replayed bit for bit with `fma`).
- The research mapping run of 09-24 (`vmtk_mapping.py`) passed the first cell's group (a
  branch) as vmtk's offset reference group; vmtk rejects that and returns its input, so that
  run's "offset" stage was a no-op. The oracle script uses vmtk's default (-1) instead.

## Pitfalls already paid for

- vmtk prints progress without newlines: parse `TIMING` lines anywhere in a line.
- `ndi.binary_dilation` defaults to FACE connectivity; "any of a cell's 8 corners" needs a
  3×3×3 structure.
- zsh does not word-split `$VAR`, `set -- $r`, or `for a in "x y"`: pass arguments explicitly.
- `dict.setdefault` evaluates its default; the demo NIfTI stores axes permuted (slice axis = the
  index axis most aligned with world S); spline smoothing needs per-point weights.
- The raw centerline polyline kinks (5°/station median, 161° at near-duplicate points): build ray
  frames from a smoothed station line (0.6 mm), and drop near-duplicate points first.
- Other agents share this machine: load averages 2–4 are normal; compare timings within one
  session, alternate conditions, don't trust RSS.

## Working with the user

- American English, in files and in chat.
- The user decides names, defaults, scope and direction: give a recommendation and wait at
  decision points. Commit only when asked; branch off `main` first. Never push without asking.
- Keep an old algorithm for comparison when replacing it. Warn before long benchmarks; rough
  numbers are fine (don't rerun vmtk rounds for precision).
- Defect reports upstream state the component's wrong behavior, project-neutral.

## Next (agreed 2026-09-24; progress in docs/port-plan.md)

Done on `main` (2026-09-30): (1) as a library (`thalweg.vmtk`, `branching`; no CLI verb runs the
grouping yet), (2) (`kernel.geometry`, `kernel.sections`, `measure`), (3) and (4) as first versions
(`case`, `export`); (5) started (`docs/validation.md`). Tier 1 of the lung batch is built: trees
rooted at their inlet, lobes per branch (`lobes`), airway walls and Pi10 (`measure`),
bronchoarterial pairing (`pairing`), artery/vein plausibility (`plausibility`), the per-point
radius interval, and Horton ratios / small-vessel fraction / orientation entropy (`statistics`).
Of the "nice" list below, those tree statistics are therefore done.

Must: (1) native centerline processing (arc length + parallel-transport frames, branch grouping,
bifurcation frames), checked bit for bit against saved vmtk outputs; (2) centerline geometry
(curvature, torsion, tortuosity), bifurcation angles, sections along branches with intervals;
(3) one decode-once pipeline with a spatial index so every stage scales to the whole tree;
(4) export: graph JSON + mesh with named caps; (5) validation beyond one subtree (phantoms, a
second tree/patient, thin-caliber ground truth). Nice: tree statistics (OSMnx-style panel +
Strahler/Horton, Murray, small-vessel fraction, orientation entropy, territories, persistence
barcode; test stability on the reconstruction ladder), harmonic mapping, CFD meshing, manual
correction, Slicer integration.

Product target (proposal, `docs/deliverables.md`): the `lung_vessels` store already holds the
fine layer (arteries, veins, AIRWAYS + airway wall, 0.7 mm) and the crop stage's `total_fast`
(117 classes at 3 mm; misnamed in stores emitted before haversack 0.13.0). Batch = graph + branch
table + case summary + QC for three trees; the rest on demand.

## Related

Siblings: `../rankfield` (decoding), `../haversack` (produces stores), `../duckn`, `../sdfview`.
Incubation history and the long-form findings: medseg's memory notes
(`~/.claude/projects/-Users-halazar-Dropbox-development-medseg/memory/project_vessel_caliber_psf.md`,
`project_thalweg.md`). Spun-off threads that may become users: GI tract / virtual endoscopy,
diaphragm sheet, body coordinate system (from linear structures).
