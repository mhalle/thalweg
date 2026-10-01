# thalweg

Centerlines, branches and wall geometry of tubular structures (vessels, airways, bowel), read
directly from a segmentation model's continuous field rather than from a triangulated surface.
A successor to [vmtk](https://github.com/vmtk/vmtk) for the era of automatic, named, multi-class
segmentation.

The *thalweg* is the line joining the deepest points of a river channel: a centerline defined by
a field. Here the field is a model's per-class margin, decoded from a
[rankfield](https://github.com/mhalle/rankfield) store (e.g. one written by haversack).

**Status: first version on `main`** (plan and per-module status in `docs/port-plan.md`).
`src/thalweg/` holds:

- `thalweg.kernel` - numpy/scipy only, no files or names: field sampling and sub-voxel zero
  crossings, field connectivity and topology, the seed-free centerline tracer (and recentering
  on section centroids), cross-sections
  with the model's interval and non-round shape measures, spline curve geometry;
- `thalweg.vmtk` - numpy ports of vmtk's centerline-only filters (attributes, resampling, branch
  extractor, bifurcation frames and vectors, offsets, merge, smoothing, centerline and branch
  geometry), checked against vmtk 1.5.2's own output. By default they fix the vmtk/VTK defects
  found on the way; `vmtk_*=True` flags reproduce vmtk exactly;
- the pipeline - `store` (read a ranked store), `centerlines` (a structure's centerlines as a
  graph), `graph` (the `.thalweg.json` format, `docs/format/thalweg-json.md`), `adapters` and
  `branching` (vmtk's grouping on the graph), `measure` (the branch table, airway walls), `lobes`
  (a lung lobe per branch), `pairing` (bronchoarterial pairing), `partition` (which branch every
  point belongs to; branch volumes), `wallmap` (the wall unrolled, r(s, angle)), `straighten` (a straightened view along a path), `volume` (a labelmap or distance image as
  input, degraded mode), `solver` (an svZeroDSolver flow model), `plausibility` (artery/vein check), `statistics` (Horton ratios and other whole-tree numbers), `case` (one case, decoded
  once: the batch product), `export` (a capped surface, VTP, SWC, Slicer markups), `cli`.

Install and test (from the repo):

    uv sync --extra test --extra tables --extra volumes --extra dicom
    uv run pytest                      # ~4-9 min with the case data in ~/tmp/data/vessels
    uv run pytest -m "not slow"        # the quick loop, ~1-1.5 min (timed 2026-10-01 at load 2-4)

Use:

    uv run thalweg structures STORE
    uv run thalweg run STORE -o OUT/                    # graph, branch table, stations, summary, QC
    uv run thalweg centerlines STORE -s lung_arteries -o arteries.thalweg.json.gz
    uv run thalweg centerlines STORE -s esophagus -o esophagus.thalweg.json.gz
                                                        # a flat tube: wall pruning + recentering by default
    uv run thalweg table arteries.thalweg.json.gz STORE -o branches.parquet --stations stations.parquet
    uv run thalweg export arteries.thalweg.json.gz STORE -s lung_arteries --mesh arteries.vtp

Also here:

- `docs/vmtk-successor.md` - the design note and every measurement so far (§12: field
  connectivity and topology, seed-free centerlines, caliber and the model's resolution floor,
  straightened rendering, and vmtk's surface stages reproduced from the field, with a
  stage-by-stage speed comparison).
- `docs/vmtk-guide.md` - thalweg for vmtk users: every vmtk tool and array mapped to thalweg,
  the differences explained, and the vmtk port layer for vmtk developers.
- `docs/vmtk-vs-field-method.md` - the method against vmtk, point by point.
- `docs/slicerheart-opportunities.md` - where it fits with SlicerHeart.
- `research/vessels/` - the incubation scripts, copied unchanged from `medseg/bench/vessels/`
  (medseg commit `a1a493a`, branch `vessels-incubation`). They are the reference the port is
  checked against. Run them from `research/vessels/` with haversack's environment
  (`../../../haversack/.venv/bin/python script.py`); the vmtk comparisons run in an isolated
  env (`uv run --no-project --python 3.12 --with vmtk --with scipy python ...`). Data stays
  in `~/tmp/data/` (see `_data.py`).
- `explorations/` - kept with thalweg but off the main line and not ported: straightened 3D
  rendering (`explorations/rendering/render_straight.py`, design note §12.5).

## Plan

`docs/port-plan.md`: the vmtk features, module by module, with their status; the tube (not only
vessel) requirements on the data model; the phase order. `docs/validation.md`: how the port
behaves beyond the case it was built on. vmtk stays the test oracle, never a dependency.

## License

Apache License 2.0 (`LICENSE`). The repository is private for now.
