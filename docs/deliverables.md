# What one batch run should produce

*2026-09-24, a proposal written at the close of the incubation thread; the formats and the cut
between tiers are the user's to decide.*

> **Status (2026-10-08).** `thalweg run` builds most of tier 1; tiers 2 and 3 exist as `thalweg
> export` options and library calls rather than as stored layers.
>
> - **Tier 1, built:**
>   - the graph, with radius and its interval per point, lobe per point and provenance per edge;
>   - the branch table: length, radius, area with its interval, tortuosity, curvature and torsion,
>     bifurcation angles, lobe, Strahler order and depth, airway walls, bronchoarterial pairing;
>   - the case summary: counts and lengths per lobe with lobe volumes and length densities,
>     Horton ratios, small-vessel fraction, orientation entropy, Pi10;
>   - QC: dropped pieces (with a warning when they are large), field loops, the coarse-slice
>     flag, artery/vein plausibility, the thalweg version and store.
> - **Tier 1, missing:**
>   - polylines resampled to 0.3 mm (the graph keeps the tracer's samples, about one per voxel);
>   - the area ratio at a branch's origin, and artery-vein ratios;
>   - the model version in QC, and a preview image;
>   - contrast along branches (needs the CT);
>   - bridged gaps (the tracer does not bridge);
>   - the haversack serve artifact.
> - **Tier 2:**
>   - branch volumes: built (`run --branch-volumes`); the partition is not yet written back into
>     the store as a label layer;
>   - surfaces and wall maps: available on demand (`export --mesh`, `--wall-maps`), not stored.
> - **Tier 3:** built as on-demand calls: sections, straightened views, wall curvature, a capped
>   CFD domain with flow extensions and a 0-D model. Meshing a chosen subtree is not built.

## The principle

The expensive part is the models (GPU minutes) and, after them, decoding and building the
whole-tree graph (~0.5 s decode + 2–6 s trace per tree (measured 2026-09-30) on an M2). Everything downstream measured so far takes
seconds per subtree once those exist (`vmtk-successor.md` §12.7). So:

- **Batch what is global, small and reusable:** results that need the whole tree, fit in a few
  MB, and serve most questions without reloading anything.
- **Leave to on-demand queries what is local, large, or depends on a user's choice:** a
  section at a point, a straightened view, a CFD domain with chosen outlets. These take seconds
  against the store plus the batch graph and need no model rerun.

The ranked store is already the cache of the expensive part. The batch product is a compact
**index and measurement set** beside it.

## Inputs: less loading than it looks

- **`lung_vessels` alone carries most of it.** Its store has two layers: the fine layer
  (arteries, veins, airways, airway wall at 0.7 mm) and the cascade's crop stage (`total_fast`,
  117 classes at 3 mm: lobes, heart, great vessels). Stores emitted before haversack 0.13.0
  misname the crop layer; re-emit them. C3N-00704's store is 78 MB.
- **`total` at 1.5 mm, only the parts needed** (organs for the lobes, cardiac for the heart and
  great vessels), where 3 mm lobes and roots are too coarse: lobe volumes as density
  denominators, root anchoring. Not all five parts.
- **The CT, only for image-derived columns:** contrast along centerlines (QC, filling-defect
  screening), image-side sizing below the model floor (later). Reading a series costs about as
  much as decoding, so make it opt-in.

## Tier 1, always: the core package (target 2–5 MB per case)

For **three trees**: arteries, veins and airways (the airways come free in the same layer).

1. **Graph:** nodes (root, bifurcations, tips) and branches with 0.3 mm polylines, radius
   and its model interval per point, class, lobe, generation and Strahler order, provenance
   (field-connected, bridged). Size: ~25 m of tree ≈ 80k points ≈ 2 MB as float32 columns.
2. **Branch table**, one row per branch (a few thousand rows):
   - length, radius (mean, min) with interval, area, tortuosity (circuity plus one angle-based
     metric), curvature and torsion summaries;
   - bifurcation angles and area ratio at the branch's origin;
   - lobe, order, class;
   - airway wall thickness (the wall is labeled);
   - contrast along the branch when the CT is included.
3. **Case summary**, per class × lobe:
   - counts, total length, densities per lobe volume;
   - Horton ratios, small-vessel volume fraction (BV5-like), artery–vein ratios;
   - orientation entropy (the OSMnx-style panel);
   - the **bronchoarterial ratio** (bronchus diameter over artery diameter) at paired
     branches, a clinical measure that needs airway
     and artery in one geometry, which this input has and most pipelines do not.
4. **QC:**
   - completeness (tips per lobe, disconnected pieces, loops, bridged gaps);
   - an acquisition-regime flag: slices > ~3 mm drop thin vessels, so thin-order statistics are
     invalid there (§12.2);
   - artery/vein plausibility, from the `total` layer. (Proposed as a test on where each tree
     roots; built as the share of each tree's length inside the `pulmonary_vein` class, because
     the root test failed on correct cases: docs/validation.md §5d.);
   - model and store versions, and one preview image.

Parquet for tables (cohorts concatenate), JSON for the summary and QC. Ideally a haversack serve
artifact beside `labels`, `preview` and `statistics`, so a cohort run fills a cache that later
queries read.

**Who it covers:** cohort and morphometry studies, COPD, pulmonary hypertension and
bronchiectasis measures, QC of segmentation at scale, and anyone who wants "the numbers" per case.
That is plausibly the 50–75 % majority of research users.

## Tier 2, opt-in flags (tens of MB)

- **Branch partition and territories** as label layers in the store (volume per branch,
  approximate perfusion territories per terminal branch).
- **Surfaces** per class with branch labels, for display and printing.
- **Wall maps** r(s, θ) per branch, for wall-shape research.

## Tier 3, on demand (seconds, from store + graph)

Sections anywhere, straightened views of a clicked path, curvature fields, a CFD domain with
the user's outlets and flow extensions, meshes of a chosen subtree.

## Who the batch does not serve, and why that is fine

- **CFD modelers** choose outlets and need body-fitted meshes: tier 3 plus meshing.
- **Planning and visualization** is interactive by nature.
- **Small-vessel caliber research** sits below the model's ~1 mm floor: it needs image-side
  sizing or a higher-resolution model, and the batch should say so rather than report numbers.

## Cost to reach it

Tier 1 needs the port's "must" items: native centerline processing, the whole-tree pipeline with
a spatial index, geometry and angles, export, plus airway graphs (the same code on another
class). Measured batch cost after the models (2026-09-30, an M2 shared with other jobs, so rough):
about 60 s per case for three trees with the defaults (four ridge passes, lobes, walls, pairing,
statistics, QC), and 21–39 s with one ridge pass and without the lung additions. Tracing and
sectioning dominate; nothing runs on a GPU yet.
