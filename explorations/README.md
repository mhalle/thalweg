# explorations

Work that lives with thalweg but is not on the main line: nothing here is ported into
`src/thalweg/` or promised as API. The scripts run with haversack's environment and read everything
through `_thalweg.py`, a path shim to this checkout's `src/` plus a few helpers: the run's store
(`thalweg.store.FieldStore`), its crop layer (`FieldStore.field(0)`), fine-layer margins, and
centerline trees traced in memory by `thalweg.kernel.medial.trace(..., ridge_passes=1)` (the
research reference, which `tests/test_medial.py` pins to `research/vessels/centerline.py`'s JSON).
No research centerline JSON is read. What the library does not provide (`DATA`, `DICOM`, the
`LADDERS` table, the CT reader `Image`) still comes from `research/vessels/`.

- `rendering/render_straight.py` - straightened 3D rendering of a vessel path directly from the
  field (per-sample display-to-world warp, no resampled volume), with an illustration style
  (`STYLE=npr`). Design note §12.5. Whether rendering belongs here or in sdfview is open.
- `catchments/` - vascular catchments (territories) as a rankfield. `catchments.py` ranks every
  lung voxel by distance to the arterial branch groups (Strahler >= 3), builds a real
  `rankfield.RankField` from the top-4 rankings, runs haversack's distance and junction layers on
  it unmodified, adds the one plane rankfield lacks (d1, distance to the supplying vessel),
  decodes coarser levels with `rf.decode_groups`, and tests whether pulmonary veins run between
  arterial territories. `catchments_figure.py [RUN] [Y_MM]` evaluates one coronal slice directly
  at 0.25 mm. `catchment_store.py` writes it as a real haversack store (duckn zarr zip, convention 1.2, 6.6 MB) with
  its own README inside (`STORE_README.md` + the generic format reference). `airway_catchments.py` and `airway_figure.py`
  do the same for the airway tree (thalweg's centerlines of `lung_airways`) and compare it with the arteries (they agree best at segment scale:
  ARI 0.70 vs 0.37 for random partitions). `_catch.py` holds the shared pieces and two switches:
  - `SUPPLY=twigs` (default): only Strahler order <= 2 segments feed tissue. With `all`, trunks
    running past other territories cut them into slivers (152 specks < 5 mm2 on one slice vs 14).
  - `WALLS=lungs` (default): left vs right only. `lobes` uses the crop layer's lobes, whose 3 mm
    total_fast puts C3N-00704's right upper/lower fissure in the wrong place.
  Results on C3N-00704 (2026-09-24, twigs + lungs): 158 classes; coarser levels exact (winner
  99.996 %, margins ~0.006 mm near boundaries, 0.07-0.27 % of near-boundary voxels need more than
  4 ranks); distance layer vs the geometric bisector p90 0.42 mm; all planes ~4.5 MB deflated;
  veins sit 2.1 mm from a territory boundary vs 3.7 mm for random points matched on distance from
  an artery (27 % vs 16 % within 1 mm), fading at Strahler >= 5. The junction layer at a 20 mm
  truncation covers EVERY lung voxel: a partition this fine has triple lines everywhere.
  `segments.py` names the 18 bronchopulmonary segments from the airway tree (lobes by position,
  segments by position within the lobe), `artery_segments.py` carries the names onto the arteries
  through the bronchoarterial pairing, and `segments_on_ct.py` draws them over the CT. On
  C3N-00704 the artery tree confirms the lobe naming, and the small right upper lobe is the scan's
  own (non-aerated anterior right upper chest), not a labeling error. Details: `CATCHMENTS.md`.
