# thalweg

Centerlines, branches and wall geometry of tubular structures (vessels, airways, bowel), read
directly from a segmentation model's continuous field rather than from a triangulated surface.
A successor to [vmtk](https://github.com/vmtk/vmtk) for the era of automatic, named, multi-class
segmentation.

The *thalweg* is the line joining the deepest points of a river channel: a centerline defined by
a field. Here the field is a model's per-class margin, decoded from a
[rankfield](https://github.com/mhalle/rankfield) store (e.g. one written by haversack).

**Status: incubated, not yet ported.** `src/thalweg/` is empty. What exists:

- `docs/vmtk-successor.md` - the design note and every measurement so far (§12: field
  connectivity and topology, seed-free centerlines, caliber and the model's resolution floor,
  straightened rendering, and vmtk's surface stages reproduced from the field, with a
  stage-by-stage speed comparison).
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

Port in layers, each checked against the research scripts' outputs: the field kernel (sampling,
sub-voxel crossings, connectivity, topology), the centerline graph and radius, native centerline
processing (frames, grouping, bifurcation frames, replacing vmtk's), partition, wall
coordinates and maps, curvature, sections, flow extensions and export; then one pipeline that
decodes once and scales to whole trees. vmtk stays as the test oracle, never a dependency.

## License

Apache License 2.0 (`LICENSE`). The repository is private for now.
