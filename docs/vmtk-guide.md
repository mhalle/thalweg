# thalweg for vmtk users

*A conversion guide: what each vmtk tool becomes in thalweg, and how the two differ.*

This guide assumes you have used vmtk: you know `vmtkcenterlines`, `vmtkbranchextractor`,
`vmtkbranchclipper`, `vmtkflowextensions` and the arrays they write (`MaximumInscribedSphereRadius`,
`GroupIds`, `Blanking`, ...), and you have chained them with `--pipe`. It assumes nothing about
thalweg. The main text is written for a vmtk user. Boxes marked **For vmtk developers** go into
the implementation: which vtkVmtk class a function follows, where thalweg departs from it, and
the defects found on the way. A maintainer can read only those boxes and the developer section
(§13) and get the whole picture.

Numbers quoted here come from `docs/validation.md`, which holds the measurements, scripts and
data behind each one, except where a number is marked as the research prototype's (one ridge
pass; `docs/vmtk-vs-field-method.md` and `docs/vmtk-successor.md`). vmtk comparisons were made against vmtk 1.5.2; vmtk option names are
taken from the current vmtk source.

---

## Contents

1. [The one difference that explains the rest](#1-the-one-difference-that-explains-the-rest)
2. [A first session, side by side](#2-a-first-session-side-by-side)
3. [Input: a field instead of a surface](#3-input-a-field-instead-of-a-surface)
4. [Coordinates and units](#4-coordinates-and-units)
5. [Centerlines](#5-centerlines)
6. [The data model: a graph instead of polylines with arrays](#6-the-data-model-a-graph-instead-of-polylines-with-arrays)
7. [Branches and bifurcations](#7-branches-and-bifurcations)
8. [Geometry and cross-sections](#8-geometry-and-cross-sections)
9. [Partition, mapping and distances](#9-partition-mapping-and-distances)
10. [Surfaces and CFD preparation](#10-surfaces-and-cfd-preparation)
11. [What thalweg adds, and what it does not do](#11-what-thalweg-adds-and-what-it-does-not-do)
12. [How closely thalweg reproduces vmtk](#12-how-closely-thalweg-reproduces-vmtk)
13. [For vmtk developers: the port layer](#13-for-vmtk-developers-the-port-layer)
14. [Command reference](#14-command-reference)
15. [Glossary](#15-glossary)

---

## 1. The one difference that explains the rest

**vmtk works on a surface. thalweg works on a field.**

In vmtk, a lumen is first segmented (level sets, seeds, vesselness), then turned into a
triangulated surface (`vmtkmarchingcubes`), smoothed and remeshed. Everything after that reads
the surface: centerlines come from its Voronoi diagram, branches from overlapping inscribed
spheres, coordinates from harmonic maps, sections from cutting it with planes.

thalweg starts from a **field**: a function defined everywhere in the volume, positive inside the
structure and negative outside, whose zero level is the wall. Its main source is a segmentation
model's own output: for each class, the margin between that class's logit and the best
competitor's, stored in a *ranked store*. The field locates the wall between voxels. A labelmap,
a signed distance image or a DICOM SEG can stand in for it (§3).

Nothing in thalweg's analysis is a surface. Connectivity, centerlines, radii, sections, the
branch partition, wall maps and curvature are all read from the field. A surface is made only
when you export one, for a mesher or a viewer.

Five consequences run through the rest of the guide:

| | vmtk | thalweg |
|---|---|---|
| What you start from | An image you segment, then a surface | A model's field (or a labelmap / distance image / SEG) |
| What you click | Seeds: sources and targets | Nothing: whole trees, every tip, no seeds |
| What a structure is | One unnamed lumen | A named class (`lung_arteries`, `aorta`, `esophagus`); artery and vein stay apart |
| What a measurement is | One number | A number and the model's own interval (±2 logits) |
| How results are stored | Polylines with point and cell arrays | A graph file (nodes, edges, points), plus Parquet tables |

> **For vmtk developers.** The field is the per-class margin `m_c = l_c − max_{k≠c} l_k` of the
> model's logits, decoded from a rankfield store (`thalweg.store.FieldStore.margin`). Its
> trilinear interpolant is the object every algorithm reads. On TotalSegmentator models the
> margin's slope at the wall is about 10 logits/mm, and stores clip it at ±8 logits, so the field
> is a signed-distance-like function within about 0.8 mm of the wall and flat beyond. Algorithms
> that need distance farther in (the inscribed radius, the tracer's cost) use the distance to the
> field's sub-voxel zero crossings instead (`kernel.field.crossings`), which has unlimited range.

---

## 2. A first session, side by side

A typical vmtk session on a segmented vessel tree:

```
vmtkmarchingcubes -ifile seg.mha -l 0.5 --pipe vmtksurfacesmoothing -passband 0.1 -iterations 30 \
  -ofile surface.vtp
vmtkcenterlines -ifile surface.vtp -seedselector openprofiles -resampling 1 -resamplingstep 0.5 \
  -ofile cl.vtp
vmtkbranchextractor -ifile cl.vtp -radiusarray MaximumInscribedSphereRadius -ofile cl_split.vtp
vmtkbranchclipper -ifile surface.vtp -centerlinesfile cl_split.vtp -ofile clipped.vtp
vmtkcenterlinegeometry -ifile cl_split.vtp -smoothing 1 -ofile cl_geometry.vtp
```

The thalweg equivalent, on a ranked store:

```
thalweg structures case.duckn.zip                        # what the store names
thalweg run case.duckn.zip -o out/                       # graph, branch table, profile, summary, QC
```

`thalweg run` traces every requested structure (by default the lung arteries, veins and airways),
measures every branch and writes:

- `graph.thalweg.json.gz`: the centerline graph (§6);
- `branches.parquet`: one row per branch: length, radius, section area with its interval,
  calipers, curvature, tortuosity, Strahler order, bifurcation angles, lobe;
- `stations.parquet`: one row per cross-section along every branch (the diameter profile);
- `summary.json`, `qc.json`: whole-tree numbers and checks.

The same, step by step:

```
thalweg centerlines case.duckn.zip -s lung_arteries -o arteries.thalweg.json.gz
thalweg table arteries.thalweg.json.gz case.duckn.zip -o branches.parquet --stations stations.parquet
thalweg export arteries.thalweg.json.gz case.duckn.zip -s lung_arteries \
  --mesh arteries.vtp --flow-extensions 10 --vmtk-centerlines arteries_vmtk.vtp
```

Three things a vmtk user will miss, on purpose:

- **No seeds.** The tracer finds every branch. There is no `-seedselector`.
- **No pipe.** Each verb reads files and writes files. The graph file is the hand-off between
  verbs, as the `.vtp` is between vmtk scripts. Composition happens in Python (the example below;
   §14.2 maps each vmtk script to its Python call).
- **No surface preparation.** There is no marching cubes, smoothing, remeshing or capping before
  analysis; those exist only for export (§10).

Python users can do the same with the library:

```python
from thalweg.centerlines import centerline_graph
from thalweg.store import open_store
from thalweg.measure import branch_table

store = open_store("case.duckn.zip")
graph = centerline_graph(store, "lung_arteries")
m, geometry, ref = store.margin("lung_arteries")
rows = branch_table(graph, "lung_arteries", m, geometry)        # list of dicts, one per branch
graph.write("arteries.thalweg.json.gz")
```

---

## 3. Input: a field instead of a surface

### 3.1 What thalweg reads

| Input | How it is read | What the field is |
|---|---|---|
| **Ranked store** (`*.duckn.zip`, `*.duckn/`) written by haversack | `thalweg.store.FieldStore` (rankfield's reader) | The model's own margin for each class: sub-voxel wall, and an interval |
| **Labelmap** (NIfTI, NRRD, MetaImage, Slicer `.seg.nrrd`) | `thalweg.volume.VolumeStore` (SimpleITK; the `volumes` extra) | The mask's signed distance in mm × 10 logits/mm, clipped at ±8: a staircase wall, rounded |
| **Signed distance image** (a float image) | `VolumeStore` | −distance × 10 (negative inside, ITK's convention), or `--sdf-inside positive` |
| **DICOM SEG** (a file, or a folder holding one, as IDC delivers them) | `VolumeStore` (highdicom; the `dicom` extra) | Each segment's mask, as for a labelmap; segments may overlap |

`thalweg structures FILE` lists what a file names. A store names its classes after the model
(`lung_arteries`, `aorta`); a labelmap names them from a Slicer `.seg.nrrd` header or as
`label_<value>`, and `--label 3=lung_arteries` names them yourself. A DICOM SEG uses its segment
labels; TotalSegmentator's display names ("Esophagus", "Small Intestine") are matched to the
model's names ("esophagus", "small_bowel") wherever a name is looked up.

### 3.2 Coming from a vmtk surface

thalweg does **not** read a surface. If all you have is a vmtk `.vtp`/`.stl` lumen, turn it into
a signed distance image first - vmtk does this itself:

```
vmtksurfacemodeller -ifile lumen.vtp -samplespacing 0.3 -ofile lumen_sdf.mha
thalweg centerlines lumen_sdf.mha -s lumen_sdf -o lumen.thalweg.json.gz
```

The sign of `vmtksurfacemodeller`'s output depends on the surface: it comes from
`vtkSurfaceReconstructionFilter`, whose sign follows its normal propagation (`-negativeinside` only
flips it; on vmtk 1.5.2 a sphere read positive inside and a tube negative). thalweg reads negative
inside by default, refuses a distance image read with the wrong sign (the "inside" would cover
most of the image's border) and tells you to pass `--sdf-inside positive`.

The structure is named after the file (`lumen_sdf`), so it gets the vessel settings. For a
flattened lumen pass `--prune wall --recenter` (§5.4).

> [!NOTE]
> A surface-derived field carries the surface's staircase and smoothing shrinkage, and no
> interval. thalweg then measures that surface - as vmtk does. The gains described in this guide
> (sub-voxel wall, intervals, artery/vein separation) need a model's field.

### 3.3 Degraded mode

Labelmaps, distance images and SEGs are read in **degraded mode**. Everything works, with two
differences, both recorded in the graph (`source.labeling_scheme` is `degraded:labelmap`,
`degraded:sdf` or `degraded:dicom-seg`):

- the wall is the labelmap's staircase (rounded by the distance transform), not a sub-voxel wall;
- the "±2 logit interval" becomes a fixed ±0.2 mm offset of the wall - a convention, not an
  uncertainty.

> **For vmtk developers.** The labelmap's signed distance is exact on anisotropic grids
> (`volume.signed_distance`): the distance from each voxel center to the nearest point of the
> other class's cell boundary, not to its center, so the zero level sits half a voxel past the
> last inside voxel along every axis. Multiplying by the model's typical wall slope (10 logits/mm)
> and clipping at ±8 makes the degraded field look, to every algorithm, like a model's.

---

## 4. Coordinates and units

- **Millimeters, LPS** (DICOM patient space) everywhere: the graph, tables, meshes, sidecars,
  SWC, markups.
- **Grids carry direction cosines.** A store's or image's grid is `origin + index @ directions`
  (one direction row per array axis, its length the spacing). Oblique and anisotropic
  acquisitions are traced in place; nothing is resampled to an axis-aligned isotropic grid.
  (vmtk's `vtkImageData` carries origin and spacing; before VTK 9 it has no orientation, so
  oblique data is usually resampled or reoriented first.)

> [!WARNING]
> The `.vtp` files thalweg writes hold LPS millimeters but no coordinate-system tag. A surface you
> exported from Slicer in RAS for vmtk will appear mirrored in x and y beside them. Negate x and y
> of one or the other, and check a landmark before trusting an overlay. Slicer markups written by
> thalweg declare `"coordinateSystem": "LPS"` and load in place.

---

## 5. Centerlines

### 5.1 vmtkcenterlines → `thalweg centerlines`

| | `vmtkcenterlines` | `thalweg centerlines` |
|---|---|---|
| Input | A surface; seeds (`-seedselector pickpoint/openprofiles/carotidprofiles/...`) | A field and a structure name |
| What is traced | Minimal paths from the source(s) to each target | The whole connected structure: every tip, found automatically |
| Domain | The Voronoi diagram of the surface (Delaunay tessellation) | The voxel lattice, connected the way the field's interpolant connects it |
| Cost | `-costfunction` (default `1/R`), integrated along the Voronoi diagram | length × 1/(d + ε)², d the distance to the wall, ε = 0.1 mm |
| Radius | `MaximumInscribedSphereRadius`: the Voronoi ball radius | The largest inscribed ball near each point, refined onto the ridge (4 coarse-to-fine passes) |
| Output | One polyline per target, shared stretches repeated | A graph: each stretch once, as an edge between nodes (§6) |
| Failure mode | A target not reached ("degenerate descent"); a seed in the wrong place | A pruned real branch or a kept spur (rules below) |

**How the tracer works** (TEASAR-style; `thalweg.kernel.medial.trace`):

1. **Distance.** Each interior lattice point's distance to the field's zero crossings.
2. **Paths.** Dijkstra from the **deepest point** (largest distance) over the lattice, with the
   cost above. Two lattice neighbors are joined only if the field's trilinear interpolant joins
   them - not by a 6/18/26-connectivity rule.
3. **Branches without seeds.** Take the uncovered point farthest from the root along the tree,
   walk its minimal path back to the tree, cover everything within 1.5 × d + 1 mm of the new
   path, repeat until nothing is uncovered.
4. **Spurs.** Drop terminal branches shorter than 2 × (radius at their junction) + 1 mm
   ("length" pruning). Flat tubes use a different rule (§5.4).
5. **Radius.** Move each path point to the largest inscribed ball within ±0.5 mm and take its
   radius; refine coarse to fine (`--ridge-passes`, default 4).
6. **Split** the tree into edges between nodes (root, junctions, tips), as vmtk's branch splitting
   would.

Then the tree is **re-rooted at its inlet** (`--root inlet`, the default): the widest end running
off the field (a trachea, a trunk leaving the crop) if it is at least half as wide as the widest
end of all; otherwise the widest end itself, measured over its terminal edge (the pulmonary
trunk). `--root deepest` keeps the tracer's own start.

**What maps to what:**

| vmtk option | thalweg |
|---|---|
| `-seedselector`, `-sourceids`, `-targetids`, `-sourcepoints`, `-targetpoints` | None: seed-free. One path from the root to a node: `straighten.path_to(graph, structure, node)`; every path below a node or a point: `thalweg.adapters.to_vmtk(graph, structure, source=...)` |
| `-endpoints` (append the open-profile barycenters) | None: tips end where the field's inside ends; a structure running off the field ends at a `truncated` node |
| `-resampling`, `-resamplingstep` | The graph keeps the tracer's samples (about one per voxel). Resample on export (`to_vmtk(..., step=)`) or in vmtk's own convention (`thalweg.vmtk.resample_centerlines`) |
| `-costfunction` | Not exposed; the tracer's integrand is 1/(d + ε)² |
| `-delaunaytolerance`, `-simplifyvoronoi`, `-usetetgen` | None: no tessellation |
| `-radiusarray` | Always `points.radius` in the graph; `MaximumInscribedSphereRadius` in vmtk-compatible export |
| `vmtknetworkextraction`, `vmtkcenterlinesnetwork` | The same `thalweg centerlines` run: it is a network extractor by construction |

### 5.2 Connectivity: what is one structure, and what is a loop

vmtk decides connectivity implicitly, by what the surface connects. thalweg decides it explicitly
from the field (`--connectivity field`, the default): two lattice neighbors are connected only if
the field's trilinear interpolant is positive along the way (across faces, face saddles and cell
interiors, with the marching-cubes asymptotic decider). `--connectivity voxel` uses the
26-connected labelmap instead, for comparison only.

Two practical effects:

- **Artery and vein stay apart.** They are different classes, so where they touch they are not
  merged. On one surface of "the vessels", every artery-vein contact becomes a loop: 73-95 per
  tree pair on the cases measured, counted with the field's own connectivity, and 105-132
  counting corner-only contacts (research prototype).
- **Only the largest connected piece is traced.** The others are listed in the structure's
  statistics (`component_lattice_point_counts`), and `thalweg run` warns when more than 10 % of a
  structure lies outside the traced piece.

The tracer builds trees. A true loop in the field (an anastomosis, two branches of one class in
contact) is cut; `qc.json` counts loops (`field_loop_count`, the genus of the field's zero
surface).

### 5.3 Radius

Both radii are the radius of the largest inscribed ball, so they mean the same thing:

- vmtk: the Voronoi ball at the Voronoi vertex the path passes through;
- thalweg: the distance from the path point to the field's zero crossings, at the point moved
  onto the ridge.

On the same field, vmtk's radius reads 0.006 mm smaller than thalweg's in the median (four ridge
passes; validation §4). Every
radius in thalweg also has an interval: `radius_lower_mm` and `radius_upper_mm` are the same
measurement at the model's +2 and −2 logit levels (written by `thalweg run`).

> **For vmtk developers.** With a single ridge pass (the research prototype's setting) thalweg's
> point is quantized to a 0.25 mm grid near the ridge, and a point off the axis by d reads a
> radius smaller by about d: vmtk then reads +0.03 mm larger. Four passes remove that. The
> remaining difference is the surface itself: vmtk's Voronoi balls touch a triangulated surface,
> thalweg's touch the field's sub-voxel crossings.

### 5.4 Flat tubes: esophagus, trachea, colon, bowel

vmtk and thalweg's default spur rule both assume round lumens. In a flattened tube the inscribed
radius is half the *depth*, so side lobes across the *width* look like branches, and the path
wanders from side to side, because every ball across the width is about as large.

thalweg has two rules for this, applied by default to structures named `esophagus`, `trachea`,
`colon`, `small_bowel` or `duodenum` (or a DICOM SEG's "Esophagus", "Small Intestine", ...), and
overridable with `--prune` and `--recenter`. The choice is by name: a labelmap's `label_<n>` or a
distance image named after its file gets the vessel settings unless you pass
`--prune wall --recenter`, or name it (`--label 1=esophagus`).

- `--prune wall`: a terminal branch is kept only if its tip reaches beyond the parent's wall by
  more than its own (median) radius, and at least 1 mm - the wall found by rays cast through the
  field from the parent's axis, clear of the junction;
- `--recenter`: every point is moved to the area centroid of its cross-section, three rounds, with
  the nodes held.

On elliptic phantoms of 2:1 to 3:1 this gives two ends and a path within 0.002 mm (median) of the
axis, against 0.1-0.3 mm for vmtk and 0.26-1.14 mm for the plain tracer (validation §4). vmtk
fails outright on the 2:1 tube (a 2-point line).

### 5.5 Smoothing and resampling

| vmtk | thalweg |
|---|---|
| `vmtkcenterlineresampling -length L` | `thalweg.vmtk.resample_centerlines` (the same vtkCleanPolyData + vtkSplineFilter, ported) |
| `vmtkcenterlinesmoothing -iterations -factor` | `thalweg.vmtk.smooth_centerlines` (the same Laplacian relaxation, ported) |
| (implicit in `vmtkcenterlinegeometry -smoothing`) | thalweg's own measures fit a **smoothing spline whose budget is tied to the radius**: each point may move about 15 % of its local radius (`kernel.geometry.SmoothPath`), so a 19 mm trunk and a 1 mm twig are smoothed in proportion. Recentered structures use 2 % |

The graph itself is never smoothed: it holds what the tracer found.

---

## 6. The data model: a graph instead of polylines with arrays

### 6.1 What vmtk gives you, and what thalweg gives you

vmtk's centerlines are a `vtkPolyData`: one polyline cell per source-to-target path, shared
stretches duplicated, and named point/cell arrays added by each script. After
`vmtkbranchextractor`, the cells are split into pieces labeled by group, centerline and tract.

thalweg's centerlines are a **graph** (`.thalweg.json`, optionally gzipped; schema in
`docs/format/`):

```
structures   one per traced class: name, source (store, part, label, grid), roots, the tracer's
             parameters, statistics
nodes        id, kind (root, junction, tip, truncated, joint), position, structure, attributes
edges        id, structure, start_node, end_node, point_range, length_mm, provenance, attributes
points       position, radius, and named columns (one value per point)
```

Each stretch of vessel is stored once, as an edge between two nodes. Edges point away from the
root. A file can hold several structures (arteries, veins and airways of one case).

### 6.2 Array by array

| vmtk array | Where it is in thalweg |
|---|---|
| `MaximumInscribedSphereRadius` (point) | `points.radius` (mm; −1 where no inside point was found) |
| `Abscissas` (point) | Arc length along an edge: the station table's `arc_length_mm`; along a vmtk path: `thalweg.vmtk.centerline_attributes` |
| `ParallelTransportNormals` (point) | `kernel.geometry.transport_frames` (thalweg's stations); `thalweg.vmtk.centerline_attributes` (vmtk's) |
| `CenterlineIds` (cell) | Not needed: a path from the root to a tip is the chain of edges between them (`TubeGraph.tree()`) |
| `TractIds` (cell) | An edge *is* a tract between two nodes |
| `GroupIds`, `Blanking` (cell) | Point columns `branch_group` and `bifurcation_region`, from running vmtk's grouping on the graph (`thalweg.branching.annotate`; §7) |
| `Length`, `Curvature`, `Torsion`, `Tortuosity`, `Frenet*` | Per branch in the branch table (`length_mm`, `curvature_mean_per_mm`, `torsion_mean_absolute_per_mm`, `distance_metric`, ...); note `distance_metric` is length / chord, and vmtk's `Tortuosity` is that minus 1. vmtk's own per point via `thalweg.vmtk.centerline_geometry` |
| Bifurcation reference systems (`Normal`, `UpNormal`) | On junction nodes: `attributes["bifurcation_frames"]` (from `branching.annotate`) |
| Bifurcation vectors and angles | On edges: `attributes["bifurcation_vector"]`; thalweg's own `deflection_deg`, `sibling_angle_deg` in the branch table |

### 6.3 Node kinds, and what vmtk has no word for

- `root`: where the tree starts (its inlet by default).
- `junction`: a branching point. vmtk has no junction *point*: it has a bifurcation *group*, the
  blanked region where the branches' tubes overlap (§7).
- `tip`: a free end.
- `truncated`: where the structure runs off the edge of the field (a crop, the scan's edge). Not a
  real tip: Strahler order ignores it, and a cap is still cut there for CFD.
- `joint`: degree 2, where an edge was split for another reason (re-rooting merges the tracer's
  own, so they are rare).

Per edge, `provenance.method` says how it was made: `field` (connected by the field), `voxel`
(the labelmap comparison mode), or `bridged` (a gap closed, with its length and reason; reserved,
the tracer does not bridge yet).

### 6.4 Getting vmtk's convention back

`thalweg export GRAPH STORE -s NAME --vmtk-centerlines C.vtp` writes the graph in vmtk's own
convention: the source-to-tip paths, resampled and split into vmtk's tracts and groups (the
branch extractor's output), with `MaximumInscribedSphereRadius`,
`Abscissas`, `ParallelTransportNormals`, and vmtk's `GroupIds`, `CenterlineIds`, `TractIds` and
`Blanking` from the ported branch extractor. vmtk and Slicer's VMTK extension read it like any
`vmtkcenterlines` + `vmtkbranchextractor` output, so the rest of a vmtk pipeline can run on
thalweg's centerlines.

> **For vmtk developers.** `adapters.to_vmtk` builds the paths: every edge densified to 0.1 mm,
> the edges of each path concatenated, the whole path resampled to `step` from the source so
> shared stretches share their samples exactly, and each sample given the radius of the nearest
> densified point (`step` defaults to 0.3 mm). That is the input `vtkvmtkCenterlineBranchExtractor` expects (coincident
> samples along shared stretches). The source can be a node or a point a given distance into an
> edge.

---

## 7. Branches and bifurcations

### 7.1 Two ways to cut a tree into branches

vmtk (`vmtkbranchextractor`) splits the source-to-target polylines where one centerline's tube
leaves the others': a **branch group** per stretch owned by one set of centerlines, and a blanked
**bifurcation group** for the overlap region around each junction. Groups are defined by the
spheres, so their ends lie roughly one radius from the geometric branching point.

thalweg's graph splits at **junction points**: an edge runs from node to node, and the branching
volume is shared out among the edges that meet there.

Both are available. thalweg's own measurements use edges. vmtk's groups are computed on the graph
when you ask for them:

```python
from thalweg.branching import vmtk_branching, annotate
b = vmtk_branching(graph, "lung_arteries")          # vmtk's chain, ported, on the graph's paths
graph = annotate(graph, "lung_arteries", b)          # writes groups, frames and vectors back
```

`annotate` adds the point columns `branch_group` and `bifurcation_region`, the bifurcation frames
on junction nodes, and the bifurcation vectors on edges. Where the tracer's single junction
corresponds to two of vmtk's close bifurcations (or none, where vmtk merged it into a neighbor's
region), `statistics.vmtk_bifurcations` counts the cases.

| vmtk | thalweg |
|---|---|
| `vmtkbranchextractor` | `thalweg.vmtk.extract_branches` (ported); on the graph, `branching.vmtk_branching` |
| `vmtkcenterlinemerge` | `thalweg.vmtk.merge_centerlines` (ported) |
| `vmtkcenterlineoffsetattributes` | `thalweg.vmtk.offset_attributes` (ported) |
| `vmtkbifurcationreferencesystems` | `thalweg.vmtk.bifurcation_reference_systems` (ported); on the graph's junctions via `annotate` |
| `vmtkbifurcationvectors` | `thalweg.vmtk.bifurcation_vectors` (ported); on edges via `annotate` |
| `vmtkbifurcationsections -distancespheres N` | `thalweg export ... --bifurcation-sections S.parquet --distance-spheres N` (§7.3) |

### 7.2 Bifurcation angles

vmtk's bifurcation vectors give in-plane and out-of-plane angles relative to the bifurcation's
reference system. thalweg reports those (on edges, after `annotate`) and its own two, in the
branch table:

- `deflection_deg`: the angle between the branch's initial direction and the parent's continuation
  through the junction (0 = straight on);
- `sibling_angle_deg`: the angle to the nearest sibling.

Directions are chords starting one junction radius from the junction - outside the junction's
ball, where every branch still points into it - over max(2 × radius, 3 mm). `angle_reliable` is
False when a chord could not reach that length. At shallow angles (about 30° and below) between
overlapping tubes the junction's position, and so the angle, is not well determined by either
method.

### 7.3 Bifurcation sections

`vmtkbifurcationsections` places a plane on each branch next to each bifurcation, N touching
spheres away, then cuts the **surface** with it. thalweg places the same planes - vmtk's points
and normals to 1e-10 mm on the phantom oracle - and cuts the **field**:

```
thalweg export graph.thalweg.json.gz store.zip -s lung_arteries --bifurcation-sections s.parquet
```

Each row carries vmtk's measures - `area_mm2`, `min_size_mm`, `max_size_mm`, `shape` (vmtk's
`BifurcationSectionArea`, `...MinSize`, `...MaxSize`, `...Shape`: the two calipers through the
center and their ratio) - plus thalweg's (Feret widths, aspect ratio, the areas at ±2 logits),
`contour_closed` (vmtk's `BifurcationSectionClosed`), the plane (`position_*_mm`, `normal_*`),
the group, bifurcation group and orientation. Areas agree with vmtk's surface sections of the same field
within 0.3 %. A single tube has no bifurcation; the export skips the table with a notice.

---

## 8. Geometry and cross-sections

### 8.1 Centerline and branch geometry

| vmtk | thalweg |
|---|---|
| `vmtkcenterlinegeometry` (per point: curvature, torsion, Frenet frame, tortuosity) | vmtk's own: `thalweg.vmtk.centerline_geometry` (ported). thalweg's: per branch in the table |
| `vmtkbranchgeometry` (per group) | vmtk's: `thalweg.vmtk.branch_geometry` (ported). thalweg's: the branch table |

thalweg's branch table reports, per edge: `length_mm`, `chord_mm`, `distance_metric` (length /
chord), `sum_of_angles_rad_per_mm`, `inflection_count_metric`, `curvature_mean_per_mm`,
`curvature_max_per_mm` and `torsion_mean_absolute_per_mm`. They come from the radius-tied spline
(§5.5), measured over the edge's **interior**: one radius clear of each end, where every traced
path hooks into the junction's ball or the tube's rounded end. On a branch shorter than
max(4 × its radius, 3 mm) they are not reported (`shape_reliable` False): a spline over a few
samples measures its own wiggle.

> **For vmtk developers.** `vtkvmtkCenterlineGeometry` differentiates the (optionally
> Laplacian-smoothed) polyline by finite differences; curvature on a lattice-quantized path is
> dominated by the quantization. thalweg's spline (`scipy.interpolate.splprep`, per-point weights
> 1 / (0.15 r + 0.05)) gives analytic derivatives with a deviation budget proportional to the
> local radius. Torsion is reported only where curvature exceeds a floor, where the binormal is
> defined.

### 8.2 Cross-sections

`vmtkcenterlinesections` and `vmtkbranchsections` cut the surface with planes normal to the
centerline. thalweg samples the **field** on those planes (`kernel.sections`), every `--step` mm
(default 1) along each branch, and finds the wall as a sub-pixel contour:

| | vmtk sections | thalweg stations |
|---|---|---|
| Area | Polygon of the surface cut | Area inside the field's contour at level 0 |
| Interval | — | `area_lower_mm2` / `area_upper_mm2`: the contours at +2 and −2 logits |
| Sizes | `CenterlineSectionMinSize`, `...MaxSize` (calipers through the center), `...Shape`; `vmtkbranchsections` writes the same as `BranchSection*` | Equivalent diameter, minimum and maximum Feret widths, aspect ratio (second moments of area; 1 = round) |
| Non-round lumens | Shape index | Aspect ratio, Feret widths, `centroid_offset_mm` (how far the section's centroid lies from the path) |
| Closed? | `CenterlineSectionClosed` | `contour_closed`; an open section's window is widened twice, and a section still open is listed but left out of the medians |
| Near junctions | Cut anyway | Stations within the junction radius + 1 mm are left out: the plane cuts the neighbors too |

The branch table gives medians over each branch's stations; `stations.parquet` gives every one.

### 8.3 Wall thickness

`vmtksurfacethickness` measures between two surfaces. thalweg measures between two classes when
the model labels a lumen and its wall (TotalSegmentator's `lung_airways` and
`lung_airways_wall`): wall area, wall-area percent, thickness, and Pi10 per case (`thalweg run`).

### 8.4 Surface curvature

`vmtksurfacecurvature -type mean` computes mesh curvature (vtkCurvatures). thalweg computes the
mean curvature of the field's zero level set by a weighted quadric fit to the unclipped field
samples within 2.8 mm (or three voxels on coarser grids) of each point (`kernel.curvature`), and
writes it on an exported mesh (`--curvature`, point data `MeanCurvature`, vtkCurvatures' sign).
It reads 1.02-1.04 × the truth on oblique tube and sphere phantoms and is about six times as
repeatable as mesh curvature across reconstructions of one scan. Like any fit, it averages over
its window at saddles and crotches.

### 8.5 Straightened views

`vmtkimagecurvedmpr` resamples an image on planes along a centerline. `thalweg.straighten`
does the same for any volume on the field's grid (the CT, the field itself), with parallel
transport frames along any path (`straighten.path_to(graph, structure, node)` gives the path from
the root to a node).

---

## 9. Partition, mapping and distances

### 9.1 vmtkbranchclipper → the partition rule, anywhere

`vmtkbranchclipper` cuts the **surface** into branches: each point goes to the group whose tube
function (|x − c|² − r² over the group's centerline balls) is lowest. thalweg evaluates the same
rule at **any points** (`thalweg.partition.label_points`) - mesh vertices, or every lattice point
inside the structure:

- `partition.label_field`: the partition as a label volume on the field's grid;
- `thalweg run --branch-volumes`: a `volume_mm3` per branch in the table (the volumes sum to the
  traced piece's volume: the structure without the pieces the tracer dropped).

Two sets of tubes can drive it: the graph's edges (thalweg's partition: every edge takes part, so
the junction volume is shared among the edges meeting there), or vmtk's groups with the blanked
bifurcation regions left out (`partition.group_tubes`: what `vmtkbranchclipper` does). With vmtk's
groups it reproduces vmtk's label at every one of the 21,221 vertices of the clipper's input
surface. The search is exact and fast: a spatial index finds, per point, the only segments that
can have the lowest value.

Not ported: cutting the surface itself along the partition (`-groupids`, `-insideout`). The
export caps at graph ends instead (§10).

### 9.2 Branch metrics and mapping

| vmtk | thalweg |
|---|---|
| `vmtkbranchmetrics` (`AbscissaMetric`, `AngularMetric`) | `thalweg.vmtk.branch_metrics`, at any labeled points: vmtk's values to 3e-14 mm and 5e-13 rad on the case's surface vertices (with vmtk's interpolation defect reproduced) |
| `vmtkbranchmapping` (harmonic and stretched mapping on the surface) | **Wall maps** r(s, θ) by ray casting in the field (`thalweg.wallmap`, `thalweg export --wall-maps W.npz`): from every station of a branch, rays at every angle to the first zero crossing. Per graph edge in thalweg's frame, or per vmtk group in vmtk's coordinates, where they lie a median 0.025 mm from vmtk's `DistanceToCenterlines`. vmtk's `StretchedMapping` is not ported |
| `vmtkbranchpatching` | The wall map is already the raster: `radius_mm[station, angle]`. Any wall quantity maps by sampling it at `WallMap.wall_points()` |

Ostia show in a wall map as rays that find no wall, or one far beyond the branch's own
(`wallmap.ostium`).

### 9.3 Distance to centerlines and the tube function

| vmtk | thalweg |
|---|---|
| `vmtkdistancetocenterlines` | `thalweg export --mesh M.vtp --distance-to-centerlines` (point data `DistanceToCenterlines`, and the traced radius at the nearest centerline point as
`CenterlineRadius` - thalweg's name; vmtk writes it under the radius array's name, only with
`-centerlineradius 1`); `partition.distance_to_centerlines` for any points, with or without radius information (`-useradius`) |
| `vmtkcenterlinemodeller`, `vmtkpolyballmodeller` | `partition.tube_function(tubes, geometry, shape)`: the tube function sampled on a lattice, from graph edges or vmtk groups |
| `vmtksurfacemodeller` | Not needed: the field is already the signed function |

> **For vmtk developers.** `vmtk.partition.distance_to_centerlines` reproduces
> `vtkvmtkPolyDataDistanceToCenterlines` within 1e-9 mm (1e-14 in practice) at 5,690 of the
> 28,449 mapped surface points. The
> pipeline version adds a radius-class spatial index: a segment's value at x is at least
> dist(x, segment)² − R², and the value at the nearest segment midpoint bounds the minimum from
> above, so only segments within √(U + R²) can win.

---

## 10. Surfaces and CFD preparation

thalweg makes a surface only for export: `thalweg export GRAPH STORE -s NAME --mesh M.vtp`.

| vmtk | thalweg |
|---|---|
| `vmtkmarchingcubes` | The field's zero set, marching cubes on its trilinear interpolant (`--refine N` meshes it N times finer) |
| `vmtksurfacesmoothing`, `vmtksurfacekiteremoval` | Not needed: the zero set has no staircase to remove, and is not shrunk |
| `vmtksurfaceremeshing`, `vmtksurfacedecimation` | Not built (planned: remeshing by projection onto the zero set, sized by the radius). Remesh the export with vmtk if a mesher needs it |
| `vmtksurfaceclipper`, `vmtksurfaceendclipper` + `vmtksurfacecapper` | A flat, named cap cut at every end of the requested kinds (`--cap-kinds`, default `tip,truncated`), only the cut branch's triangles clipped |
| `vmtkboundaryreferencesystems`, `vmtkboundarylabeler` | Every cap's name, center, normal, area, ring barycenter and ring mean radius in the sidecar `M.vtp.boundaries.json` |
| `vmtkflowextensions` | `--flow-extensions RATIO [--extension-transition T]` |
| `vmtkmeshgenerator`, `vmtktetgen`, `vmtkboundarylayer` | Not in thalweg: feed the exported surface and its named caps to vmtk, TetGen or gmsh |
| `vmtkmeshwallshearrate` and other solver post-processing | Not in thalweg |
| (no vmtk equivalent) | `--zero-d M.json`: an svZeroDSolver input from the graph (§10.3) |

### 10.1 Caps and boundary ids

vmtk's capper writes `CellEntityIds` with the wall at 1 and caps from 2 (`-entityidoffset 1`).
thalweg writes cell data **`BoundaryId`: 0 for the wall, 1..K for the caps**, and the sidecar
describes each id: its name (e.g. `lung_arteries tip 812`), the graph node and edge it caps, the
cut's center and outward normal, the centerline's inscribed radius there, the cap's area and
centroid, and vmtk's boundary reference system (`ring_barycenter`, `ring_mean_radius_mm`). Add 1 to
`BoundaryId` if a downstream tool expects vmtk's numbering.

Ends that cannot be capped cleanly are listed under `skipped` with the reason (the edge is too
short for the cut, the cut's center falls outside the structure, the far side stays attached, or
the radius is below the minimum), never left open: the surface is closed, consistently wound and
manifold, or it is not written.

### 10.2 Flow extensions

| `vmtkflowextensions` option | thalweg |
|---|---|
| `-extensionmode boundarynormal` | Always: each cap is extended along its own normal (vmtk's default is `centerlinedirection`) |
| `-adaptivelength 1 -extensionratio R` | `--flow-extensions R`: R × the ring's mean radius long (default R = 10 in the Python API) |
| `-transitionratio T` | `--extension-transition T` (default 0.25): the ring morphs into a circle over the first T of the length, with a smoothstep blend so the wall has no kink |
| `-adaptiveradius 1` (vmtk's default) | Always: the ring morphs to a circle of its own mean radius |
| `-extensionlength`, `-extensionradius`, `-preserveshape`, `-interpolationmode` | Not exposed |

Each extension is a straight tube that ends in a flat cap keeping the original cap's id and name.
Its vertices lie on vmtk's cylinders to a median 0.02 mm. A ring that is not star-shaped about
its barycenter, or would fold, falls back to a straight prism of its own shape. The export prints
how many extensions run back into the structure (`extension_vertices_inside_structure` in the
sidecar): a straight extension from a curved vessel can.

> **For vmtk developers.** thalweg maps ring vertices to the target circle by their fraction of
> the ring's arc length (with the circle turned to lie closest to the ring), not radially from the
> barycenter: radial mapping folds rings that are not star-shaped. Over the transition the
> vertices also slide to even spacing, so a ring with a very short edge does not drag a strip of
> thin triangles down the tube. The boundary reference system averages the ring **along its
> length**; vmtk averages its vertices, which biases the barycenter toward dense stretches of the
> mesh (0.05-0.26 mm on the case's rings). `boundary_reference_system(..., vmtk_vertex_mean=True)`
> reproduces vmtk's.

### 10.3 A 0-D flow model

`--zero-d M.json` writes an svZeroDSolver input from the graph: one `BloodVessel` per edge with
Poiseuille resistance R = 8μ/π ∫ds/r⁴ and inductance L = ρ/π ∫ds/r² (integrated exactly for a
radius linear between samples), junctions at branching nodes, a steady inflow at the root and a
resistance at every other end (placeholders to replace with the study's own; `--inflow`,
`--outlet-resistance`). Units are CGS, as SimVascular's.

---

## 11. What thalweg adds, and what it does not do

### 11.1 Added

- **Names.** Every structure is a model class. Lung branches carry their lobe; airway branches
  their wall measures and their paired artery (bronchus-to-artery ratio).
- **Intervals.** Radius and area come with the model's own ±2-logit interval.
- **Topology from the field.** Connectivity decided by the interpolant; loops counted by the
  genus of the zero surface; artery and vein never merged.
- **Whole trees, no seeds, every tip.** With `truncated` ends where the field's crop cuts a
  structure, so a cropped trunk is not mistaken for a tip.
- **Per-edge provenance**, and QC: pieces not traced, length outside the field, field loops.
- **Whole-tree statistics.** Strahler order, Horton's ratios, small-vessel volume fraction,
  orientation entropy.
- **Flat tubes** (§5.4) and degraded inputs (§3.3).

### 11.2 Not done (use vmtk or another tool)

- **Segmentation.** thalweg reads a model's field; a model run (haversack) or a labelmap makes it.
- **Surface input.** Convert to a distance image first (§3.2).
- **Interactive seeds and picking.** There is nothing to pick; to take one path, take it from the
  graph.
- **Surface remeshing and volumetric meshing; CFD solving and its post-processing.**
- **Cutting the surface into branches**, and vmtk's `StretchedMapping`.
- **Cycles in the traced graph.** The format allows them; the tracer builds trees.

---

## 12. How closely thalweg reproduces vmtk

Two kinds of comparison, both in `docs/validation.md`:

**Where thalweg computes the same quantity its own way** (centerline, radius, sections, curvature),
on the same field:

| Quantity | Agreement |
|---|---|
| Centerline position (C3N-00704 subtree, 51 targets) | vmtk lies a median 0.085 mm from thalweg's path with one ridge pass, 0.060 mm with four; 50 of 51 routes identical (research prototype) |
| Radius | vmtk − thalweg: +0.030 mm with one ridge pass, −0.006 mm with four |
| Bifurcation section areas | within 0.3 % |
| Wall-map radii vs vmtk's `DistanceToCenterlines` | median 0.025 mm |
| Flow-extension vertices vs vmtk's | median 0.02 mm |
| Flat tubes (2:1 to 3:1) | thalweg 0.002 mm from the axis (median), vmtk 0.1-0.3 mm; vmtk fails on 2:1 |

**Where thalweg ports vmtk's filter** (`thalweg.vmtk`; §13), on vmtk's own input, with vmtk's
defects reproduced: integers exact, floats within 1e-9 of vmtk's output. Examples: the branch
extractor gives vmtk's 95 groups and 44 bifurcations; `branch_metrics` vmtk's values to 3e-14 mm;
`distance_to_centerlines` to 1e-14 mm; the partition vmtk's label at all 21,221 clipper input
vertices.

Robustness on a whole tree (research prototype): vmtk failed 1-2 of 51 targets ("degenerate
descent"); thalweg traces the whole tree, and 12 of about 3,000 segments briefly leave the vessel
(6.7 mm in 32 m).

---

## 13. For vmtk developers: the port layer

### 13.1 Architecture

thalweg has three layers, kept apart by a test (`tests/test_layering.py`):

| Layer | What it holds | Depends on |
|---|---|---|
| `thalweg.kernel` | Algorithms on arrays and grids: field sampling and zero crossings, field connectivity and topology, the tracer, recentering, sections, rays, curvature, spline geometry | numpy, scipy, skimage |
| `thalweg.vmtk` | Ports of vmtk's centerline-only filters, on vmtk's own data convention | numpy only (no VTK, no vmtk) |
| pipeline (`store`, `volume`, `graph`, `centerlines`, `adapters`, `branching`, `measure`, `partition`, `wallmap`, `export`, `solver`, `case`, `cli`, ...) | Files, names, the graph, the verbs | the two layers above |

vmtk itself is never imported. vmtk is the **test oracle**: its outputs on phantoms and on a real
subtree are frozen in `tests/fixtures/vmtk_oracle/`, regenerated by
`tests/oracle/vmtk_centerline_oracle.py` in an isolated environment
(`uv run --no-project --python 3.12 --with vmtk ...`).

### 13.2 The vmtk port

`thalweg.vmtk.Centerlines` is vmtk's `vtkPolyData` as arrays: `points` (N, 3) float64, `cells` (a
list of point-id arrays), and `point_data` / `cell_data` / `field_data` dicts keyed by vmtk's own
array names. Each module follows one vtkVmtk class line for line:

| Module | vtkVmtk class |
|---|---|
| `attributes` | `vtkvmtkCenterlineAttributesFilter` |
| `resampling` | the `vmtkcenterlineresampling` script (VTK's `vtkCleanPolyData` + `vtkSplineFilter`) |
| `polyball` | `vtkvmtkPolyBallLine` (vectorized over points and segments) |
| `sphere_distance` | `vtkvmtkCenterlineSphereDistance` |
| `branches` | `vtkvmtkCenterlineBranchExtractor` |
| `frames` | `vtkvmtkCenterlineBifurcationReferenceSystems` |
| `offset` | `vtkvmtkCenterlineReferenceSystemAttributesOffset` |
| `merge` | `vtkvmtkMergeCenterlines` |
| `smoothing` | `vtkvmtkCenterlineSmoothing` |
| `geometry` | `vtkvmtkCenterlineGeometry` |
| `branch_geometry` | `vtkvmtkCenterlineBranchGeometry` |
| `vectors` | `vtkvmtkCenterlineBifurcationVectors` |
| `metrics` | `vtkvmtkPolyDataCenterlineAbscissaMetricFilter`, `...AngularMetricFilter` |
| `sections` | `vtkvmtkPolyDataBifurcationSections` (the plane placement; the cut is the field's) |
| `partition` | `vtkvmtkPolyDataCenterlineGroupsClipper`'s labeling rule; `vtkvmtkPolyDataDistanceToCenterlines` |
| `utilities` | `vtkvmtkCenterlineUtilities` |
| `_vtk` (private) | the VTK and `vtkvmtkMath` routines the filters share |

### 13.3 The flag convention, and the defects found

Every public function in `thalweg.vmtk` defaults to the **correct** behavior. Where vmtk, or the
VTK it runs on, has a defect or loses precision, a keyword `vmtk_<name>=True` reproduces it.
`VMTK_FLAGS` maps each of the eleven main ported functions to its flags, so
`**{f: True for f in VMTK_FLAGS[name]}` asks for vmtk's numbers exactly; the sections' and
sphere-distance functions and `export.boundary_reference_system` take their flags directly. The
oracle tests pass every flag of their stage. The switches that turn several on at once cover
their own chain only:

- `branching.vmtk_branching(..., vmtk_compatible=True)`: attributes, branch extractor, frames,
  offset, vectors and branch geometry;
- `thalweg export --vmtk-exact`: with `--vmtk-centerlines`, the attributes and the branch
  extractor; with `--bifurcation-sections`, the branching chain and the section placement;
- nothing turns on `vmtk_vertex_mean`; pass it to `boundary_reference_system`.

| Flag | Where | vmtk's behavior | Effect |
|---|---|---|---|
| `vmtk_float32` | branches, merge, smoothing, geometry, frames, vectors, branch geometry | Points stored in a default `vtkPoints` (VTK_FLOAT), so coordinates are float32-rounded | Branch extractor: points move by up to 2e-6 mm (phantom) / 8e-6 mm (case), grouping unchanged. Smoothing: up to 1e-4 / 2e-4 mm. Centerline geometry: torsion differs by up to 0.36 /mm on the phantom, where vmtk's torsion is dominated by the rounding |
| `vmtk_interp` | offset, metrics, vectors, sections | `vtkvmtkCenterlineUtilities::InterpolateTuple` calls `GetTuple` twice; both return the same internal buffer, so every "interpolated" radius, abscissa or normal is the segment's **end** value | Radius-weighted averages and abscissas use end values instead of interpolated ones |
| `vmtk_steps` | branches, sphere distance, branch geometry, vectors, sections | The tube-exit and touching-sphere searches count steps from the **squared** segment length (`Distance2BetweenPoints`) while sizing them as a fraction of the radius | Search resolution off by a factor of the segment length; split points move up to 0.09 mm; curvature/torsion of subsampled groups shift slightly |
| `vmtk_merge` | branches | `MergeTracts` emits the run reaching a centerline's last tract as a second cell of the same group | One extra cell (5 vs 4 on the `hairpin` fixture); never on the oracles |
| `vmtk_last_tract` | branches | In `PointInTubeGroupTracts` the running minimum is declared inside the loop, so the **last** passing tract wins, not the one with the smallest tube value | 25 relabel decisions on the case oracle have several candidates, and none changes a group (the last candidate is also the deepest); it decides the `hairpin` fixture |
| `vmtk_two_point_cells` | attributes | A two-point centerline is a `vtkLine`, which `vtkPolyLine::SafeDownCast` rejects, so it gets no abscissas or normals | Two-point centerlines keep zeros |
| `vmtk_cell_data` | resampling, merge | `vtkSplineFilter` copies the cell data of the input line by its index among the lines, but vertex cells come first in the cell numbering | Shifted cell data whenever cleaning turned a degenerate line into a vertex |
| `vmtk_fallback` | vectors | With no touching point on an adjacent cell, vmtk takes the cell's last point, which for an upstream branch is the end point itself: a zero vector with full weight | Meaningless angles (π/2, 0) for short parents |
| `vmtk_discard_smoothing` | branch geometry | With line smoothing and sphere subsampling, the smoothed points are overwritten by a subsampling of the raw cell, so `LineSmoothing` has no effect | Corrected group curvature is 0.93 (phantom) / 0.56 (case) of vmtk's, median |
| `vmtk_vertex_mean` | export (boundary reference system) | The ring barycenter is the mean of its vertices, biased toward dense stretches of the mesh | 0.05-0.26 mm on the case's rings |

Each flag is documented in its module's docstring: what vmtk does, why it is wrong, and how big
the effect is on the phantom and case oracles. Several are reportable upstream as defects.

### 13.4 Algorithms that are not ports

These replace a vmtk stage instead of porting it, because the field makes a different method
possible:

- **Centerlines** (`kernel.medial`): minimal paths on the field-connected lattice, not the Voronoi
  diagram of a surface (§5).
- **Connectivity and loops** (`kernel.topology`): face saddles by the asymptotic decider; cell
  interiors by sampling the trilinear interpolant; genus by Lewiner marching cubes.
- **Sections** (`kernel.sections`): field sampled on the plane, contour at level 0 by marching
  squares, interval contours at ±2.
- **Recentering** (`kernel.recenter`): section area centroids, with rejection rules for sections
  that are open, belong to a junction's merged lumen, reach past a path's end, fold into a
  neighbor's plane, or sit at a waist between touching tubes.
- **Wall pruning** (`kernel.medial.prune_by_wall`): protrusion beyond the parent's wall, measured
  by rays through the field.
- **Wall maps** (`kernel.rays`, `wallmap`): ray casting to the first zero crossing.
- **Curvature** (`kernel.curvature`): weighted quadric fit to the unclipped field samples.

---

## 14. Command reference

### 14.1 Verbs

```
thalweg structures STORE                                   what a store or image names
thalweg centerlines STORE -s NAME [-s NAME ...] -o OUT.thalweg.json[.gz]
    [--part N] [--connectivity field|voxel] [--ridge-passes N] [--prune auto|length|wall]
    [--[no-]recenter] [--root inlet|deepest]
thalweg table GRAPH STORE -o BRANCHES.parquet [--stations STATIONS.parquet] [-s NAME] [--step MM]
thalweg run STORE -o DIR [-s NAME ...] [--step MM] [--no-stations] [--branch-volumes] [...]
thalweg export GRAPH [STORE] -s NAME
    [--mesh M.vtp [--cap-kinds ...] [--refine N] [--curvature] [--distance-to-centerlines]
                  [--flow-extensions R [--extension-transition T]]]
    [--vmtk-centerlines C.vtp] [--vmtk-exact] [--swc T.swc] [--markups M.mrk.json]
    [--wall-maps W.npz [--wall-map-step MM]] [--bifurcation-sections S.parquet [--distance-spheres N]]
    [--zero-d M.json [--inflow Q] [--outlet-resistance R]]
thalweg summary GRAPH                                      counts and lengths of a graph file
thalweg schema [-o FILE]                                   the graph format's JSON Schema
```

Every verb that reads a STORE also takes `--sdf-inside negative|positive` and
`--label VALUE=NAME` (§3). `export` needs STORE only for outputs that read the field (mesh,
wall maps, sections).

### 14.2 vmtk script → thalweg, at a glance

| vmtk | thalweg CLI | thalweg Python |
|---|---|---|
| `vmtkcenterlines`, `vmtknetworkextraction`, `vmtkdelaunayvoronoi` | `centerlines` (no Voronoi diagram is built) | `centerlines.centerline_graph` |
| `vmtkcenterlineresampling` | — | `vmtk.resample_centerlines`; `adapters.to_vmtk(step=)` |
| `vmtkcenterlinesmoothing` | — | `vmtk.smooth_centerlines` |
| `vmtkcenterlineattributes` | — | `vmtk.centerline_attributes` |
| `vmtkbranchextractor` | `export --vmtk-centerlines` | `vmtk.extract_branches`, `branching.vmtk_branching` |
| `vmtkcenterlinemerge` | — | `vmtk.merge_centerlines` |
| `vmtkcenterlineoffsetattributes` | — | `vmtk.offset_attributes` |
| `vmtkbifurcationreferencesystems` | — | `vmtk.bifurcation_reference_systems`, `branching.annotate` |
| `vmtkbifurcationvectors` | — | `vmtk.bifurcation_vectors`, `branching.annotate` |
| `vmtkbifurcationsections` | `export --bifurcation-sections` | `branching.bifurcation_sections` |
| `vmtkcenterlinegeometry`, `vmtkbranchgeometry` | `table`, `run` (thalweg's measures) | `measure.branch_table`; `vmtk.centerline_geometry`, `vmtk.branch_geometry` |
| `vmtkcenterlinesections`, `vmtkbranchsections` | `table --stations`, `run` | `measure.branch_table(stations_out=...)`, `kernel.sections` |
| `vmtksurfacethickness` | `run` (labeled walls) | `measure.branch_table(outer=...)` |
| `vmtkbranchclipper` | `run --branch-volumes` | `partition.label_points`, `label_field`, `group_tubes` |
| `vmtkbranchmetrics` | — | `vmtk.branch_metrics` |
| `vmtkbranchmapping`, `vmtkbranchpatching` | `export --wall-maps` | `wallmap.wall_maps`, `edge_wall_map`, `group_wall_map` |
| `vmtkdistancetocenterlines` | `export --mesh --distance-to-centerlines` | `partition.distance_to_centerlines` |
| `vmtkcenterlinemodeller`, `vmtkpolyballmodeller` | — | `partition.tube_function` |
| `vmtksurfacecurvature` | `export --mesh --curvature` | `kernel.curvature.mean_curvature` |
| `vmtkmarchingcubes`, `vmtksurfacecapper`, `vmtksurfaceclipper` | `export --mesh` | `export.capped_surface` |
| `vmtkboundaryreferencesystems` | `export --mesh` (sidecar) | `export.boundary_reference_system` |
| `vmtkflowextensions` | `export --mesh --flow-extensions` | `export.flow_extensions` |
| `vmtkimagecurvedmpr` | — | `straighten.straighten`, `straighten.path_to` |
| `vmtksurfacemodeller` | — (not needed) | — |
| `vmtksurfacereader`, `vmtkimagereader`, `vmtksurfacewriter`, ... | Inputs: stores, labelmaps, distance images, DICOM SEG (§3). Outputs: `.thalweg.json`, Parquet, VTP, SWC, Slicer markups, svZeroDSolver JSON | `store.open_store`, `graph.TubeGraph.read` / `write`, `export.*` |
| `vmtkmeshgenerator`, `vmtktetgen` | — (use vmtk, TetGen or gmsh on the export) | — |

---

## 15. Glossary

- **Field / margin.** The function thalweg reads: for a class c, `l_c − max(other logits)`,
  positive where c wins. Its zero level is the wall.
- **Ranked store.** The file haversack writes: a compressed encoding of every class's margin on
  the model's grid (`*.duckn.zip`), read with rankfield.
- **Degraded mode.** A labelmap, distance image or DICOM SEG turned into a field: staircase wall,
  no interval.
- **Lattice.** The grid's points. The tracer's paths run between them; the field joins them.
- **Zero crossing.** Where the field changes sign along a grid edge, found to sub-voxel precision.
  The inscribed radius is the distance to the nearest one.
- **Ridge.** The medial surface of the inside, where the inscribed radius is locally largest; the
  tracer's points are refined onto it.
- **Station.** A cross-section along a branch, every `--step` mm.
- **Interval.** The same measurement at the field's +2 and −2 logit levels: the model's own
  uncertainty of the wall's position.
- **Truncated end.** Where a structure leaves the field's grid; not a real tip.
- **Inlet.** The end a tree is rooted at: the widest end leaving the field if it is at least half
  as wide as the widest end of all, otherwise the widest end.
- **Flat tube.** A lumen much wider than deep (esophagus, trachea, bowel), traced with wall
  pruning and recentering by default.
- **Group, bifurcation group, blanking** (vmtk's). A stretch owned by one set of centerlines; the
  overlap region around a junction; the flag marking it. Available on thalweg's graph through
  `branching.annotate`.
