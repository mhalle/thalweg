# The port: which vmtk features thalweg carries, and in what order

Consolidated 2026-09-30 from `vmtk-successor.md` (§1.1 inventory, §5 mapping, §12.6–12.7
results), `deliverables.md` and CLAUDE.md "Next". This file is the working list for
`src/thalweg/`; the design notes stay the reasoning behind it.

## Status words

Updated 2026-09-30 (on `main`); the module that holds each item is named.

- **ported**: a numpy port in `thalweg.vmtk` reproduces vmtk's filter on the phantom and case
  oracles (integers exact, floats within 1e-9; defects fixed by default, `vmtk_*` flags
  reproduce them).
- **done**: implemented in thalweg from the field (kernel or pipeline) and tested.
- **native**: the research prototype computes it from the field and was checked against vmtk;
  not yet in `src/thalweg`.
- **via vmtk**: the prototype runs vmtk's own code on *our* centerlines. The port must replace it.
- **prototype**: computed from the field, with no vmtk comparison yet.
- **todo**: not started.
- **replaced**: the field makes the stage unnecessary.
- **out**: stays with vmtk, TetGen, gmsh, a solver or sdfview; thalweg feeds it.

Phase numbers are the port order below. "T2"/"T3" are the deliverables' opt-in and on-demand
tiers, ported after phase 5.

## vmtk, module by module

### Image preprocessing and segmentation

| vmtk | thalweg | Status | Phase |
|---|---|---|---|
| `vmtkimagevesselenhancement`, `vmtkimagefeatures`, `vmtkimageobjectenhancement`, `vmtkimagesmoothing` | The model's field; vesselness on the margin's Hessian is a hypothesis (§5.1) | replaced | — |
| `vmtklevelsetsegmentation`, `vmtkimageinitialization`, `vmtkactivetubes`, `vmtkimageseeder` | An engine run (haversack). Local refinement and gap bridging are later work | replaced | — |

### Surface

| vmtk | thalweg | Status | Phase |
|---|---|---|---|
| `vmtkmarchingcubes` | Zero set of the margin (Lewiner), export only (`export.surface`) | done | 4 |
| `vmtksurfacesmoothing`, `vmtksurfacekiteremoval` | Nothing to repair: no staircase, no shrinkage | replaced | — |
| `vmtksurfaceremeshing`, `vmtksurfacedecimation` | Remesh by projection onto the zero set, sized by the radius (§5.2); `export.surface(refine=)` meshes a finer interpolant meanwhile | todo | 4 (optional) |
| `vmtksurfacecapper` | Flat caps at graph ends, carrying the end's name (`export.surface`, `end_cuts`) | done | 4 |
| `vmtksurfaceclipper`, `vmtksurfaceendclipper` | Exact plane clip of the zero set, limited to the cut branch (`export.surface`); the implicit cut min(φ, −s) is in `flow_ext.py` | done | 4 |

### Centerlines

| vmtk | thalweg | Status | Phase |
|---|---|---|---|
| `vmtkcenterlines`, `vmtkdelaunayvoronoi` | Seed-free minimal paths on the field-connected lattice, ridge-refined radius (`kernel.medial`, `kernel.topology`, `kernel.field`; reproduces `centerline.py` exactly). vmtk is a median 0.086 mm away, 50 of 51 routes identical, radius +0.029 mm with 1 ridge pass (0.060 mm and −0.006 mm with 4 passes, docs/validation.md §4) | done | 1 |
| `vmtknetworkextraction`, `vmtkcenterlinesnetwork` | The same graph: the whole tree, every tip, no seeds (`centerlines`) | done | 1 |
| `vmtkcenterlineresampling` | `vmtk.resampling`; the graph -> vmtk-convention adapter is `adapters.to_vmtk` (reproduces the oracle input exactly) | ported | 1 |
| `vmtkcenterlinesmoothing` | `vmtk.smoothing`; thalweg's own radius-tied spline is in `kernel.sections.stations` | ported | 2 |

### Branches and bifurcations

| vmtk | thalweg | Status | Phase |
|---|---|---|---|
| `vmtkcenterlineattributes` | Abscissa and parallel-transport normals (`vmtk.attributes`) | ported | 1 |
| `vmtkbranchextractor` | Tube-function overlap grouping: GroupIds, Blanking, CenterlineIds, TractIds (`vmtk.branches`, `vmtk.polyball`); on the graph via `branching`. Field -> graph -> port gives vmtk's own 95 groups and 44 bifurcations exactly | ported | 1 |
| `vmtkcenterlinemerge` | One polyline per branch group (`vmtk.merge`) | ported | 1 |
| `vmtkcenterlineoffsetattributes` | Abscissa and normals offset to the root bifurcation (`vmtk.offset`) | ported | 1 |
| `vmtkbifurcationreferencesystems` | Bifurcation frames (origin, normal, up-normal) (`vmtk.frames`); on the graph's junction nodes via `branching.annotate` | ported | 1 |
| `vmtkbranchclipper` | The labeling rule, no surface: a point belongs to the group whose tube function is lowest (`vmtk.partition`), evaluated anywhere with an exact spatial pruning (`partition`). vmtk's label at every vertex of the clipper's input surface, with vmtk's centerlines and ours; the points the clipper inserts on its cuts are ties. As a volume: `partition.label_field`, a volume per branch (`thalweg run --branch-volumes`). Cutting the surface itself along the partition is not ported (the export caps at graph ends) | done | 1 |
| `vmtkbifurcationvectors` | Bifurcation angles (in-plane and out-of-plane) (`vmtk.vectors`); thalweg's own `deflection_deg` and `sibling_angle_deg` in `measure` | ported | 2 |
| `vmtkbifurcationsections` | Sections one radius from the bifurcation, read from the field | todo | 2 |

### Geometry and sections

| vmtk | thalweg | Status | Phase |
|---|---|---|---|
| `vmtkcenterlinegeometry` | Curvature, torsion, Frenet frame, tortuosity (`vmtk.geometry`); thalweg's own spline geometry in `kernel.geometry` | ported | 2 |
| `vmtkbranchgeometry` | Per-branch length, curvature, torsion, tortuosity (`vmtk.branch_geometry`); thalweg's per-branch table in `measure` | ported | 2 |
| `vmtkcenterlinesections`, `vmtkbranchsections` | Normal-plane sections sampled from the field, sub-voxel contours, area ± the model's interval, Feret widths, aspect ratio (`kernel.sections`, `measure`; pixel areas equal `straighten.py`'s). No vmtk oracle (vmtk's needs the clipped surface) | done | 2 |
| `vmtksurfacethickness` | Wall area, WA%, thickness and Pi10 between the labeled lumen and wall (`measure`, `case`; airways) | done | 2 |
| `vmtksurfacecurvature` | Level-set mean curvature by a quadric fit (`curvature.py`): 1.02–1.04 × truth on phantoms, 6× vmtk's repeatability | native | T3 |

### Mapping and modeling

| vmtk | thalweg | Status | Phase |
|---|---|---|---|
| `vmtkbranchmetrics` | AbscissaMetric and AngularMetric as point-wise formulas (`wall_map.py`). Bit for bit with vmtk's end-point defect switched on; correct interpolation is the default | native | T2 |
| `vmtkbranchmapping` | Wall maps r(s, θ) by ray casting: 1.8 s, 0.02–0.03 mm from vmtk's | native | T2 |
| `vmtkbranchpatching` | 2-D patch images of wall data; the ray-cast map is already a raster. vmtk's own patching run saved nothing (`patch_*` arrays empty) | todo | T2 |
| `vmtkdistancetocenterlines` | Distance to the tube function | prototype | 2 |
| `vmtkpolyballmodeller`, `vmtkcenterlinemodeller` | The tube function as one more signed field | native | T3 |
| `vmtksurfacemodeller` | The field is already the signed function | replaced | — |

### CFD

| vmtk | thalweg | Status | Phase |
|---|---|---|---|
| `vmtkflowextensions` | Implicit extensions (`flow_ext.py`): one watertight domain, 0.022 mm from vmtk's surface | native | T3 |
| `vmtkboundaryreferencesystems`, `vmtkboundarylabeler` | Named caps with ring barycenter, normal and mean radius (the ring code is in `flow_ext.py`) | prototype | 4 |
| `vmtkmeshgenerator`, `vmtktetgen`, `vmtkboundarylayer` | vmtk, TetGen or gmsh, fed the zero set, the radius sizing and the named boundaries | out | — |
| `vmtkmeshwallshearrate`, `vmtkmeshvorticityhelicity`, `vmtkmeshlambda2`, `vmtkparticletracer`, `vmtkpathlineanimator` | Solver post-processing; results can be mapped onto the (s, θ) chart | out | — |

### Viewing and IO

| vmtk | thalweg | Status | Phase |
|---|---|---|---|
| `vmtkimageviewer`, `vmtksurfaceviewer` | sdfview renders the field | out | — |
| `vmtkimagecurvedmpr` | Straightened field along any path (`straighten.py`, `explorations/rendering/`) | native | T3 |
| `vmtk*reader`, `vmtk*writer` | In: a ranked store via rankfield (`store`; a plain labelmap or SDF in degraded mode is todo). Out: `.thalweg.json`, Parquet tables, VTP mesh and centerlines, SWC, Slicer markups (`graph`, `measure`, `export`); a 0-D/1-D solver input is todo | done (in part) | 1 in, 4 out |

## What thalweg adds that vmtk lacks

These are carried in the data model from phase 1, not bolted on afterwards:

- **Names.** Classes come from the model; points carry their lobe, and the branch table gives
  each edge's lobe and bifurcation depth.
- **Field topology.** Connectivity is decided by the interpolant, loops by the genus of the zero
  set, and artery–vein contacts stay separate.
- **Intervals.** Radius and area carry the model's own ± interval.
- **Provenance per edge:** field-connected or bridged.
- **Whole trees with no seeds.**

Built since: lobes per branch (`lobes`, from the store's crop stage), Horton ratios, small-vessel
volume fraction and orientation entropy (`statistics`), the per-point radius interval. Later:
Murray's exponent, fractal dimension, the OSMnx panel, territories (`explorations/catchments`).

## Tubes, not only vessels: requirements on the data model

thalweg covers tubular structures in general. TotalSegmentator alone labels, among others:
- the trachea, esophagus, colon, small bowel and duodenum;
- the aorta, IVC, portal and splenic veins, and iliac arteries;
- the airways with their wall.

vmtk's vessel assumptions are a round lumen, a tree with a source, one surface, and a tip at
every end. None of them holds for all of these, so the phase-1 data model must drop them. Fixing
them after the vessel code exists would be a rewrite.

1. **Cross-sections that aren't round.** A section records:
   - area and perimeter;
   - equivalent diameter;
   - minimum and maximum width (Feret);
   - aspect ratio (minor / major axis of the section's second moments; 1 = round);
   - the contour itself.

   The inscribed radius is one field among these, not *the* radius. vmtk's maximal inscribed
   sphere under-reports any flattened or irregular lumen: the trachea's saber sheath, the colon,
   the esophagus, a dissection.
2. **General graphs.**
   - **Cycles are allowed.** Some are anatomy (circle of Willis), not defects.
   - **A single tube with no branching is a first-class case** (esophagus, colon, ureter).
   - **Truncated ends:** an end cut off by the field of view is flagged `truncated`, not
     counted as a tip. This matters for tip counts and Strahler order.
3. **Layers around one centerline.**
   - Nested classes share one axis: airway lumen + airway wall; aortic lumen + thrombus +
     wall; true + false lumen.
   - Thickness between layers is a standard quantity along the tube, not an airway special
     case.
4. **Collapsed and interrupted lumens.** Colon and small bowel collapse; thin tubes fragment.
   A bridged gap records its length and the reason it was bridged, so statistics can exclude
   it.
5. **Self-contact.** Adjacent colon loops touch like arteries and veins do. Connectivity is
   always decided by the field. Never fall back to 26-connectivity, whose corner links join
   what the field keeps apart.

## Structure-specific analyses (a tier after phase 2)

These are built on phase 2's sections and geometry. Suggested order: airways (same store as the
vessels; walls and pairing are built: `measure`, `pairing`), then the aorta (the most-used clinical measurement), then the GI
tract (the hardest lumen).

**Any tube**
- A diameter profile along the arc length, with a fitted healthy reference: stenosis %, local
  dilation, taper rate.
- Tortuosity beyond vmtk's: distance metric, sum of angles, inflection count.
- Contact and clearance: which named structure each stretch of wall touches, and the distance
  between tubes. It needs names, which vmtk does not have.

**Aorta and large vessels**
- Diameters perpendicular to the centerline at standard landmarks, placed from neighboring named
  structures: sinus, sinotubular junction, ascending, arch, descending, infrarenal.
- Maximum aneurysm diameter perpendicular to the centerline, and sac volume.
- Lengths between landmarks and landing-zone diameters, for stent-graft planning.

**Airways**
- Wall thickness and wall-area % by generation, and Pi10: done (`measure`, `case`).
- Bronchus-to-artery ratio at paired branches: done (`pairing`, `bronchus_to_artery_ratio`).
- Total airway count, and tapering along a path (bronchiectasis).
- Anatomical branch naming (segment naming exists in `explorations/catchments`).

**Colon, small bowel, esophagus**
- A fly-through camera path: the centerline of maximal clearance.
- Total length and flexure positions.
- Wall unfolding (virtual dissection; exploring on `dissection-proto`), which generalizes the
  wall maps to large, irregular lumens.
- Shape index from wall curvature, for polyp candidates (the curvature code exists).
- Prone/supine correspondence by arc length along the named tube.

**Trees**
- Strahler/Horton order and small-vessel volume fraction: done (`measure`, `statistics`).
  Murray's exponent and fractal dimension: not yet.
- Territories per branch, as rankfields (`explorations/catchments`). The liver's Couinaud
  segments follow from the portal venous tree the same way lung segments follow from the
  bronchi.

**Across scans.** The same tube in two acquisitions, matched by named nodes and arc length:
- prone vs supine;
- inspiration vs expiration (airway collapse);
- follow-up (aneurysm growth).

**Consumers short of 3-D CFD**
- The graph with per-edge length, radius and Poiseuille resistance, for 0-D/1-D solvers
  (§5.8).
- Named surfaces for printing and device sizing.

## Port order

**Phase 0: scaffolding.**
- Layout: a kernel layer (numpy, torch optional, no tasks or files) and a pipeline layer
  (graph, naming, IO, provenance). A layering test enforces the split, as in haversack.
- Geometry: rankfield's `Geometry` (origin + one direction row per array axis, LPS mm; the
  vocabulary of the store and of `research/vessels/_field.py`'s `Grid`). labelfield's `Grid` is axis-aligned
  (no direction cosines), so it does not fit oriented CT grids; decided 2026-09-30.
- The CLI uses click, like haversack.
- vmtk is only a test oracle, run in its isolated `uv run --with vmtk` env.

**Phase 1: field to branch graph, reproducing vmtk where vmtk defines the quantity.**
1. Port `centerline.py` (distance to zero crossings, field lattice, Dijkstra, ridge radius) as
   the graph builder.
2. Add a vmtk-convention adapter: one polyline per target, 0.3 mm spacing.
3. Port attributes, branch extractor, merge, offset attributes and bifurcation frames.
4. Port the tube-function partition.

Oracles: `tests/oracle/vmtk_centerline_oracle.py` runs every centerline-only vmtk stage on a
synthetic tree (frozen in `tests/fixtures/vmtk_oracle/`) and on C3N-00704's 51-tip subtree
(`$VESSELS_DATA/oracle/C3N-00704_ctpa0625/`).

The target is bit-exact, with vmtk's defects reproducible behind a flag. The graph is built to
the tube requirements above from the start: cycles, truncated ends, layers, bridged-gap
provenance, and section records that are not only a radius. The vessel oracles exercise only
part of that.

**Phase 2: geometry.** Smoothing, curvature, torsion, tortuosity, bifurcation vectors and
angles, sections with intervals, and the per-branch table. vmtk oracles exist for every
centerline-only stage; `vmtkbranchsections` has none (it needs vmtk's clipped surface).

**Phase 3: decode once, whole tree.**
- Decode once and keep a spatial index, so every stage scales to 500+ branches.
- Reference: about 23 s from decode to a finished whole tree today.

**Phase 4: export.**
- The graph as `.thalweg.json`, and the branch table as Parquet.
- A zero-set mesh with named caps and boundary frames.

**Phase 5: validation beyond one subtree.** Phantoms, a second tree or patient, and
thin-caliber ground truth. It also includes one tube that is not a vessel (the trachea or the
esophagus, both in `total`), which is the real test of the general data model.

**Then:**
- the T2/T3 stages, from their native prototypes: wall metrics and maps, wall curvature, flow
  extensions, straightened views, partition label layers;
- the structure-specific tier above.

## CLI (built)

```
thalweg structures STORE
thalweg centerlines STORE -s lung_arteries [-s ...] -o arteries.thalweg.json.gz [--part N] [--graph field|voxel]
                    [--ridge-passes N (default 4)] [--prune length|wall] [--root inlet|deepest]
thalweg table arteries.thalweg.json.gz STORE -o branches.parquet [--stations stations.parquet] [-s NAME] [--step MM]
thalweg run STORE -o OUT/ [-s NAME ...] [--step MM] [--no-stations] [--branch-volumes]
            [--ridge-passes N] [--prune ...] [--root ...]
                                               # graph, branches, stations, summary.json, qc.json
thalweg export arteries.thalweg.json.gz STORE -s lung_arteries [--mesh M.vtp [--cap-kinds ...] [--refine N]] [--vmtk-centerlines C.vtp [--vmtk-exact]] [--swc T.swc] [--markups M.mrk.json]
thalweg summary arteries.thalweg.json.gz
thalweg schema [-o FILE]
```

vmtk's grouping on the graph (`branching.annotate`) is a library call; no verb writes it into
the graph yet.

## Open decisions

1. **What is the data model: our graph or vmtk's polylines?** DECIDED 2026-09-30: our graph, with
   vmtk's convention as an adapter (`adapters.to_vmtk`) for the oracle tests and vmtk-compatible
   export.
2. **Where do the test fixtures live?** DECIDED 2026-09-30: data tests skip when
   `~/tmp/data/vessels` is absent (marked `data`); small synthetic vmtk oracles are frozen in git
   (`tests/fixtures/vmtk_oracle/`), regenerated by `tests/oracle/vmtk_centerline_oracle.py`.
3. **Graph format.** DECIDED 2026-09-30: our own format, `.thalweg.json` (gzip optional,
   `.thalweg.json.gz`), standalone and never a duckn extension. It holds:
   - a header: format version, source store, model, class names, parameters;
   - nodes (tip, truncated end, bifurcation) and edges, with cycles allowed;
   - provenance per edge: field-connected, or bridged with its length and reason;
   - per-point data stored column-wise: position, arc length, radius with its interval,
     section descriptors, layer ids.

   The schema is published as JSON Schema generated from pydantic models, as in duckn.

   No existing format covers cycles, truncated ends, intervals, non-round sections, layers,
   provenance and names together. Candidates surveyed: VTP, Slicer markups, SWC, TubeTK
   `.tre`, Neuroglancer skeletons, NetworkX/GraphML, Voreen/MeVisLab vessel graphs,
   svZeroDSolver. They become lossy exporters instead:
   - VTP in vmtk's convention (vmtk, Slicer, ParaView, and the oracle tests);
   - Slicer markups;
   - SWC (trees only);
   - svZeroDSolver / 1-D solver input.

   The per-branch table stays Parquet, derived from the graph.
4. **Defaults of the tracer's `ridge_passes` and `prune`.** DECIDED 2026-09-30: four ridge passes
   by default (the one-pass research reference stays reproducible with `ridge_passes=1`, which the
   reproduction tests pin); length pruning stays the default (`--prune wall` is an option). The
   facts behind the choice are in `docs/validation.md` §5.
