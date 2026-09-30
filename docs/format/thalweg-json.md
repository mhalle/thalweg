# `.thalweg.json`: the tube graph format (version 0.1)

thalweg's own file for the centerlines of tubular structures: vessels, airways, bowel, ducts.
It is a single JSON document, gzipped when the name ends in `.gz` (`.thalweg.json.gz`). The
machine-readable definition is [`thalweg-0.1.schema.json`](thalweg-0.1.schema.json), generated
from the pydantic models in `src/thalweg/graph.py` by `thalweg schema -o …`; a test keeps the
two identical. The models also enforce the invariants below, which a JSON Schema cannot express.

It is standalone on purpose: it does not depend on duckn or on the store it was read from. It
carries enough source information to find that store again. Version 0.x may change without
migration; 1.0 will be the first stable version.

## Shape

```json
{
  "format": "thalweg", "version": "0.1", "units": "mm", "space": "LPS",
  "created_by": "thalweg 0.0.1",
  "structures": [ { "name": "lung_arteries", "source": {…}, "roots": [0],
                    "method": "thalweg.trace", "parameters": {…}, "statistics": {…} } ],
  "nodes":  [ { "id": 0, "kind": "root", "position": [x, y, z], "structure": "lung_arteries",
                "attributes": {} }, … ],
  "edges":  [ { "id": 0, "structure": "lung_arteries", "start_node": 0, "end_node": 1,
                "point_range": [0, 57], "length_mm": 17.2,
                "provenance": { "method": "field" },
                "branch": 0, "generation": 0, "attributes": {} }, … ],
  "points": { "position": [[x, y, z], …], "radius": [r, …],
              "columns": { "branch_group": [3, 3, …], … } }
}
```

- **Coordinates** are LPS millimeters (DICOM's patient space), in the world of the source store.
- **Structures.** A file holds one or more structures, each with its own nodes and edges. Every
  node and edge names its structure.
  - `source` records where the structure came from: `store` (path), `labeling_scheme` (the scheme
    that names this class, e.g. `ts.v2:lung_vessels`), `part` (the store's `parts/<part>`),
    `label_value`, and `grid` (`shape`, `directions`, `origin` of the field).
  - `parameters` records the tracer's settings: `connectivity` (`field` or `voxel`),
    `cover_scale`, `cover_constant_mm`, `cost_epsilon_mm`, `ridge_passes` (default 4; 1 is the
    research reference), `prune` (default `length`) and `root` (`inlet`, the default, or
    `deepest`).
  - `statistics` records counts from the tracer and the graph (below).
- **Nodes** have one of these kinds:
  - `root`: where the tree starts. By default (`parameters.root: "inlet"`) this is the
    structure's inlet: the widest end running off the field (a trachea, a trunk leaving the crop),
    or failing that the end of the widest terminal edge (the pulmonary trunk; the veins' atrial
    end). `attributes.end` says what the node was (`tip` or `truncated`; a truncated one also has
    `on_grid_boundary`). With `root: "deepest"` it is the tracer's own start, the deepest point,
    which need not be an end: the arteries' deepest point lies inside the pulmonary trunk (the
    research reference). `statistics.deepest_point` records the tracer's start either way, and
    `statistics.inlet_end_width_mm` beside `widest_end_width_mm` how the inlet was chosen: an end
    running off the field is preferred only when it is at least half as wide as the widest end;
  - `junction`: degree 3 or more;
  - `tip`: a free end, degree 1;
  - `truncated`: an end where the structure runs off the edge of the source field, degree 1.
    It is not a real tip, and Strahler order ignores it;
  - `joint`: degree 2, where an edge was split for another reason.

  A root that sits on the grid edge carries `attributes.on_grid_boundary: true`.
- **Edges** are polylines between two nodes. `point_range: [start, stop)` is the edge's slice of
  the shared point table.
  - `provenance.method` says how the edge was made:
    - `field`: connected in the model's field;
    - `voxel`: the 26-connected labelmap graph, kept for comparison;
    - `bridged`: a gap was closed, and the edge then carries `gap_mm` and `reason`.
  - `branch` is the tracer's branch id.
  - `generation` is the tracer's attachment depth: how many traced branches the edge's branch
    hangs from. It is **not** the number of bifurcations to the root; the branch table's
    `bifurcation_depth` is.
- **Points** are stored one column per quantity.
  - `position` and `radius` are always present. `radius` is the inscribed-ball radius in mm, and
    -1 where no inside point was found.
  - Further columns live under `columns`, each exactly as long as the table, with `null` where a
    value is undefined. Values are scalars.
- **Cycles are allowed.** Nothing in the format requires a tree: `roots` may be empty, or list
  one node per component.
  - Consumers that need a tree (Strahler order, the branch table's angles, the vmtk adapter,
    SWC) check with `TubeGraph.tree()` and refuse anything else with a clear error.
  - The tracer produces trees, so a loop in the field (an anastomosis, two branches of one class
    in contact) is dropped from the graph. The QC's `field_loops` counts such loops.

## Invariants (enforced when a document is read or built)

- Ids are positions: `nodes[i].id == i`, `edges[i].id == i`. Structure names are unique.
- An edge's first and last samples sit on its start and end nodes, within 1e-6 mm.
- Both end nodes belong to the edge's structure, and so do the structure's roots.
- Point ranges are at least two samples long, do not overlap, and together cover the table.
- `length_mm` is the edge polyline's length, within 1e-6 mm.
- Kinds agree with degree:
  - `tip` and `truncated` have degree 1;
  - `joint` has degree 2;
  - `junction` has degree 3 or more;
  - `root` may have any degree.
- `bridged` carries `gap_mm` and `reason`; `field` and `voxel` carry neither.
- Numbers are finite (no NaN or infinity), `length_mm` ≥ 0, and a radius is ≥ 0 or exactly -1.
- `version` has the form `major.minor`; a reader accepts its own major version.
- **Orientation (checked by `tree()`):** in a structure that is a tree with one root, every edge
  points away from the root.

## Keys written by thalweg 0.0.1

**`statistics`, from the tracer:**
- `traced_lattice_points`, `lattice_points` (of the whole structure);
- `components`, `component_sizes` (only the largest piece is traced; the rest are listed here);
- `zero_crossings`, `cell_interior_joins`;
- `branches_before_pruning`, `branches`, `pruned_by_wall` (with `prune: "wall"`);
- `max_distance_mm`, `deepest_point` (where the tracer started);
- `inlet_end_width_mm`, `widest_end_width_mm` (with `root: "inlet"`).

**`statistics`, from the graph:**
- `edges`, `tips`, `truncated_ends`, `junctions`, `joints`, `length_mm`;
- `edges_joined_to_their_start_node`: a child branch starts at its own refined point near the
  junction, and the junction's position is prepended to close the gap.

**`statistics.vmtk_bifurcations`** (from `thalweg.branching.annotate`): how junctions and vmtk's
bifurcations correspond.

**Attributes.** From the tracer, on a root node: `end` and `on_grid_boundary` (see Nodes). From
`thalweg.branching.annotate`, which no CLI verb runs yet:
- on a node, `bifurcation_frames`: a list of `{group, origin, normal, up_normal}`, vmtk's
  bifurcation reference systems at that junction;
- on an edge:
  - `branch_groups`, vmtk's group ids along it;
  - `bifurcation_vector`, `{bifurcation_group, in_plane_angle_rad, out_of_plane_angle_rad}`.
    The angles are vmtk's; their definitions are in `thalweg.vmtk.vectors`.

**Point columns:**
- `branch_group` (vmtk group id) and `bifurcation_region` (1 inside vmtk's bifurcation region,
  else 0), both from `thalweg.branching.annotate`;
- `lobe` (written by `thalweg run` for lung stores): 1 left upper, 2 left lower, 3 right upper,
  4 right middle, 5 right lower, 0 outside every lobe (the hilum). Lobes come from the store's own
  crop stage (`thalweg.lobes`). The branch table gets `lobe` (the lobe holding most of the edge's
  length, by name; null when most of it lies outside every lobe) and `lobe_length_fraction`.
- `radius_lower_mm`, `radius_upper_mm` (written by `thalweg run`): the model's own interval on
  each sample's radius, the distance to the surfaces where the margin is +2 and -2 logits (a
  half-width of 0.14–0.21 mm on the lung arteries measured). `radius_lower_mm` is 0 where the stricter surface
  does not contain the sample.

## The tube requirements (docs/port-plan.md), as of 0.1

- **Sections that are not round.** Per-station section descriptors live in the station table
  (`--stations`): area, the ±2-logit areas, perimeter, Feret widths, aspect ratio and centroid
  offset. Contours and the station frames are not stored yet. Per-point section columns are
  possible (scalar columns) but none are written.
- **Graphs that are not trees.** Allowed by the format, as are single tubes and `truncated`
  ends. The tracer produces trees only.
- **Nested layers** (lumen plus wall, true plus false lumen). Each layer can be its own
  structure. There is no formal link between layers yet; it will need a field on `Structure` (a
  format change).
- **Bridged gaps**: edges with `provenance.method: "bridged"`, a length and a reason. The tracer
  does not bridge yet.
- **Self-contact.** Connectivity is always decided by the field, so two branches that touch are
  connected only where the field connects them. Where it does (walls overlapping within one
  class), the graph cannot tell that from a real junction; QC reports `field_loops`.
- **Missing, and likely to force a change:**
  - vector-valued columns (frames, tangents);
  - units and descriptions per column.

## Beside it

- **Branch table** (`thalweg table … -o branches.parquet`, or `thalweg run`): one row per edge,
  defined in `src/thalweg/measure.py`. `thalweg run` adds, where the store allows:
  - `lobe`, `lobe_length_fraction` (a lung store; above);
  - for an airway whose wall the model labels: `wall_area_mm2`, `wall_area_percent`,
    `wall_thickness_mm`, `internal_perimeter_mm` (medians over the edge's stations) and
    `wall_station_count`. `thalweg table` writes these too;
  - for an airway traced with the arteries: `paired_artery_edge` (the nearest parallel artery
    edge, not a verified companion), `paired_fraction`, `paired_sample_count`,
    `bronchus_to_artery_ratio` (`thalweg.pairing`). `thalweg run` only.

  Columns a structure does not have are null in its rows.
- **Station profile** (`--stations stations.parquet`, or `thalweg run`): one row per
  cross-section, every `--step` mm (default 1) along each edge's interior:
  - `structure`, `edge`, `arc_length_mm` (from the edge's start);
  - `position_x_mm`, `position_y_mm`, `position_z_mm` (LPS, the section's center on the path);
  - `traced_radius_mm` (the graph's radius there);
  - `area_mm2`, with `area_low_mm2` and `area_high_mm2` (the areas inside the margin's +2 and
    -2 logit contours);
  - `equivalent_diameter_mm`, `perimeter_mm`, `min_feret_mm`, `max_feret_mm`, `aspect_ratio`,
    `centroid_offset_mm`, `contour_closed`;
  - the wall columns above, per station, for an airway with a labeled wall.
- **`summary.json`** (`thalweg run`): one object per structure, keyed by its name.
  - `edges`; `nodes` (a count per node kind); `length_mm`;
  - `radius_percentiles_mm` (`p10`, `p50`, `p90` of the traced radius over the samples);
  - `strahler_order`: per order, `edges` and `length_mm`; `unordered`: the same for edges that
    lead only to truncated ends and so have no order;
  - `sectioned_branches`: edges with at least one measured section;
  - `coarse_slices` (see `qc.json`), and a `warning` string when it is true;
  - `tree` (`thalweg.statistics`):
    - `horton`: `bifurcation_ratio`, `length_ratio`, `diameter_ratio`, and `orders`, per
      Strahler order `streams`, `mean_length_mm`, `mean_diameter_mm`;
    - `volume_mm3`, `small_vessel_volume_fraction`, `small_area_mm2` (the threshold, 5);
    - `orientation_entropy`;
  - `wall` (airways with a labeled wall): `pi10_mm`, `slope`, `stations` (those with a wall
    measure) out of `lumen_stations`, `internal_perimeter_range_mm`, and
    `constant_wall_pi10_mm` (what a wall of the median thickness at every caliber would give: on
    TotalSegmentator's wall class Pi10 restates it, docs/validation.md §5b);
  - `bronchoarterial` (airways traced with the arteries): `paired_sample_share`,
    `paired_branches`, `bronchus_to_artery_ratio_median`, `share_of_paired_branches_above_1`,
    `reach_mm`, `parallel_cosine`;
  - `lobes` (a lung store): per lobe name, and `outside_lobes`: `edges`, `tips`, `length_mm`,
    and for a lobe `lobe_volume_ml`, `length_mm_per_ml`.
- **`qc.json`** (`thalweg run`; meanings in `src/thalweg/case.py`):
  - `thalweg` (the version), `store` (its path);
  - `labeling_scheme`: a string, or a list with one scheme per cascade stage in newer stores;
  - `acquisition`: keyed by part number (as a string), `source_spacing_mm` and `coarse_slices`
    (the largest spacing exceeds 3 mm);
  - `lobes`: `named_by` and `part`, or `named_by: null` and a `reason`;
  - `artery_vein` (`thalweg.plausibility`): `plausible` (true, false, or null with a `reason`
    when it cannot be checked), `reasons`, the two
    `…_length_share_inside_pulmonary_vein` shares, and the roots' distances
    `…_root_to_pulmonary_vein_mm`, `…_root_to_heart_mm`;
  - `structures`: per structure `dropped_components` (`count`, `lattice_share`, `largest`),
    `truncated_ends`, `outside_mm`, `length_mm`, `unrefined_points`, `cell_interior_joins`,
    `field_loops`;
  - `timings_s`.
- **vmtk-compatible export:** `thalweg export … --vmtk-centerlines C.vtp [--vmtk-exact]` writes
  the source-to-tip paths after vmtk's centerline attributes and branch extractor (not the later
  bifurcation stages). The integer arrays (`GroupIds`, `Blanking`, `CenterlineIds`, `TractIds`)
  are Int32, as vmtk's filters require.
- **Surface:** `thalweg export … --mesh M.vtp [--cap-kinds tip,truncated,root] [--refine N]`
  writes the structure's zero set: marching cubes at level 1e-5 on the margin's trilinear
  interpolant, N times finer.
  - The surface is closed, consistently wound with normals pointing out of the structure, and
    manifold, or it is not written.
  - A flat cap is cut at every end of the requested kinds (default `tip` and `truncated`). A
    degree-1 root on the grid's edge counts as truncated; any other degree-1 root is `root`, the
    inlet.
  - Cell data `BoundaryId` (Int32): 0 the wall, 1..K the caps.
  - The sidecar `M.vtp.boundaries.json` holds `space` (`"LPS"`), `units` (`"mm"`) and:
    - `boundaries`, one object per BoundaryId in order:
      - the wall: `{"id": 0, "name": "wall", "area_mm2"}`;
      - each cap: `{"id", "name"` (e.g. `"lung_arteries tip 812"`), `"end"` (tip, truncated or
        root), `"node"` (graph node id), `"edge"` (the graph edge cut), `"center"` (where the
        centerline crosses the cap's plane), `"normal"` (unit, pointing out of the structure),
        `"inscribed_radius_mm"` (the centerline's radius at the cut), `"area_mm2"`,
        `"centroid"` (the cap's area centroid, on the plane)`}`;
    - `skipped`: `[{"name", "end", "node", "reason"}]`, every requested end that got no cap, and
      why. The reasons are:
      - the edge is too short to clear the junction;
      - its radius is below the minimum;
      - the centerline leaves the structure at the cut;
      - the section reaches another cap or branch;
      - the plane leaves an open boundary;
      - the far side stays attached (a double wall);
      - the section lies on a detached piece.

    Every end of the requested kinds appears exactly once, either in `boundaries` or in `skipped`.
- **SWC and Slicer markups:** `--swc`, `--markups`.
