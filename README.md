# thalweg

Centerlines, branches and wall geometry of tubular structures (vessels, airways, bowel), read
directly from a segmentation model's continuous field rather than from a triangulated surface.
A successor to [vmtk](https://github.com/vmtk/vmtk) for the era of automatic, named, multi-class
segmentation.

The *thalweg* is the line joining the deepest points of a river channel: a centerline defined by
a field. Here the field is a model's per-class margin, decoded from a
[rankfield](https://github.com/mhalle/rankfield) store (e.g. one written by haversack).

It reads a ranked store, or in a degraded mode a labelmap, a signed distance image or a DICOM
SEG. It traces whole trees without seeds and measures them, and it exports surfaces, centerlines
and models for vmtk, Slicer, CFD meshers and svZeroDSolver.

**Status (2026-10-08):** a working library and CLI on `main`, covering vmtk's centerline,
branching, geometry, mapping and CFD-preparation stages (`docs/port-plan.md`), validated on
phantoms and real cases (`docs/validation.md`). Coming from vmtk? Read `docs/vmtk-guide.md`.

`src/thalweg/` has three layers, kept apart by `tests/test_layering.py`:

- `thalweg.kernel` - numpy/scipy/skimage only, no files or names: field sampling and sub-voxel
  zero crossings, field connectivity and topology, the seed-free centerline tracer and its
  pruning, recentering on section centroids, cross-sections with the model's interval and
  non-round shape measures, rays to the wall, wall curvature, spline curve geometry;
- `thalweg.vmtk` - numpy ports of vmtk's centerline-only filters (attributes, resampling, branch
  extractor, bifurcation frames, vectors and sections, offsets, merge, smoothing, centerline and
  branch geometry, branch metrics, the clipper's partition rule), checked against vmtk 1.5.2's
  own output. By default they fix the vmtk/VTK defects found on the way; `vmtk_*=True` flags
  reproduce vmtk exactly;
- the pipeline:
  - inputs: `store` (a ranked store), `volume` (a labelmap, distance image or DICOM SEG, in
    degraded mode);
  - the graph: `centerlines` (tracing, inlet rooting, combining), `graph` (the `.thalweg.json`
    format, `docs/format/thalweg-json.md`), `adapters` (to vmtk's convention), `branching`
    (vmtk's grouping, frames, vectors and bifurcation sections on the graph);
  - measures: `measure` (the branch table and station profile, airway walls), `partition`
    (which branch every point belongs to; branch volumes), `wallmap` (the wall unrolled,
    r(s, angle)), `straighten` (a straightened view along a path), `statistics` (Horton ratios
    and other whole-tree numbers), `lobes`, `pairing` (bronchoarterial), `plausibility`
    (artery/vein);
  - outputs: `case` (one case, decoded once: the batch product), `export` (capped surface with
    flow extensions, vmtk-convention centerlines, SWC, Slicer markups), `solver` (an
    svZeroDSolver model), `cli`.

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

Inputs other than a ranked store: `thalweg centerlines lumen.nii.gz ...` (a labelmap; name its
labels with `--label 3=lung_arteries`), a signed distance image (`--sdf-inside positive` if it is
positive inside), or a DICOM SEG file or the folder IDC delivers it in.

## Documentation

`docs/README.md` indexes every document and says which are kept current and which are dated
records. The main ones:

- `docs/vmtk-guide.md` - thalweg for vmtk users: every vmtk tool and array mapped to thalweg,
  the differences explained, and the vmtk port layer for vmtk developers. The fullest account of
  the system.
- `docs/format/thalweg-json.md` - the graph format and every output beside it.
- `docs/validation.md` - measured behavior: phantoms, second patients, flat tubes, the vmtk
  comparisons, the defaults chosen.
- `docs/port-plan.md` - vmtk module by module, and the status of each.
- `docs/vmtk-successor.md`, `docs/vmtk-vs-field-method.md`, `docs/slicerheart-opportunities.md` -
  records from incubation (2026-09-23/24), each with a status note.

Also here:

- `research/vessels/` - the incubation scripts, copied unchanged from `medseg/bench/vessels/`
  (medseg commit `a1a493a`, branch `vessels-incubation`). They are the reference the port is
  checked against. Run them from `research/vessels/` with haversack's environment
  (`../../../haversack/.venv/bin/python script.py`); the vmtk comparisons run in an isolated
  env (`uv run --no-project --python 3.12 --with vmtk --with scipy python ...`). Data stays
  in `~/tmp/data/` (see `_data.py`).
- `explorations/` - kept with thalweg but off the main line and not ported: straightened 3D
  rendering, vascular catchments and named lung segments, and aortic dissection
  (`explorations/README.md`).
- `validation/` - the scripts behind `docs/validation.md`.

vmtk stays the test oracle, never a dependency.

## License

Apache License 2.0 (`LICENSE`). The repository is private for now.
