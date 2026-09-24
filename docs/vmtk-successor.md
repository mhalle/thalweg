# A successor to vmtk: vascular modeling on fields instead of surfaces

Status: **design exploration, 2026-09-23. Nothing is built or decided.** Measurements on the
lung-vessel demo, two thin-slice CTAs, same-patient thickness ladders, an image-only PSF, a
centerline kernel checked against vmtk, the field-only pipeline, straightened 3D renders and vmtk's branch partition
without a surface are in §12–12.6; §13 compares the method with vmtk; §14 maps it onto SlicerHeart. They
revised §5.3, §9 and §10. Claims marked
*measured* cite work already done in this workspace. Claims marked *hypothesis* are
untested and come with the experiment that would test them.

## 0. The argument in one paragraph

vmtk is organized around one object: **a triangulated lumen surface**. Everything upstream
exists to produce that surface: seeds, level sets, marching cubes. Everything downstream
recovers structure from it: smoothing, remeshing and capping to repair it, a Voronoi diagram
to find its medial axis, overlapping balls to split it into branches, and harmonic maps to
give it coordinates. That structure fit 2004, when segmentation was the hard, interactive
step and its only output was a binary lumen. None of that holds now. Segmentation is cheap
and names the vessels (TotalSegmentator ships about 40 vascular classes). The model's output
is a **field**: rankfield stores its per-class signed margins, and that field locates surfaces
to about a micron on a 1.5 mm grid. Alongside it we store a nearest-surface distance field and
a junction layer that marks where two named structures meet. A successor should make the
**field the archive** and every geometric object a **derived view with its parameters and
provenance**, a rule the ranked stores already follow. The objects are the surface,
centerline, graph, cross-section, coordinate chart and CFD mesh. Four things follow. The
medial axis becomes a ridge of a signed distance field instead of a simplified Voronoi
diagram. Branch topology is partly *read* from label adjacency instead of *inferred* from
geometry. Every measurement can carry an interval instead of a single number. The workflow
becomes seed-free and cohort-scale. vmtk remains the oracle and the CFD back end, not
something to fight.

---

## 1. What vmtk is

### 1.1 Inventory

About 150 `vmtkScripts/*.py` pypes scripts over the `vtkVmtk` C++ layer (listed from
`github.com/vmtk/vmtk` master, 2026-09-23). Grouped by pipeline stage:

| Stage | Modules | Core algorithm |
|---|---|---|
| Image preprocessing | `vmtkimagevesselenhancement`, `vmtkimagefeatures`, `vmtkimageobjectenhancement`, `vmtkimagesmoothing` | Frangi / Sato / vessel-enhancing diffusion (VED); gradient, upwind, FWHM feature images |
| Segmentation | `vmtkimageinitialization`, `vmtklevelsetsegmentation`, `vmtkactivetubes`, `vmtkimageseeder` | Colliding fronts, fast marching, threshold init → geodesic active contours; active tubes (centerline + radius evolved on the image) |
| Surface | `vmtkmarchingcubes`, `vmtksurfacesmoothing`, `vmtksurfaceremeshing`, `vmtksurfacedecimation`, `vmtksurfacecapper`, `vmtksurfaceclipper`, `vmtksurfaceendclipper`, `vmtksurfacekiteremoval` | Taubin/Laplacian smoothing; radius-adaptive isotropic remeshing; capping of open ends |
| Centerlines | `vmtkcenterlines`, `vmtkdelaunayvoronoi`, `vmtkcenterlinesnetwork`, `vmtknetworkextraction`, `vmtkcenterlinesmoothing`, `vmtkcenterlineresampling` | Delaunay → embedded Voronoi diagram (circumsphere centers = maximal inscribed spheres) → Eikonal on the Voronoi diagram with speed ∝ radius → steepest-descent backtrace |
| Branches | `vmtkbranchextractor`, `vmtkbranchclipper`, `vmtkbifurcationreferencesystems`, `vmtkbifurcationvectors`, `vmtkbifurcationsections`, `vmtkcenterlinemerge` | Antiga–Steinman decomposition: polyball (tube function) overlap between centerlines defines bifurcation regions; group/tract ids; bifurcation frames |
| Geometry | `vmtkcenterlinegeometry`, `vmtkcenterlineattributes`, `vmtkcenterlinesections`, `vmtkbranchgeometry`, `vmtkbranchsections`, `vmtkbranchmetrics`, `vmtksurfacecurvature`, `vmtksurfacethickness` | Frenet curvature/torsion/tortuosity; abscissa + parallel-transport normals; planar sections normal to the centerline |
| Mapping | `vmtkbranchmapping`, `vmtkbranchpatching`, `vmtkcenterlineoffsetattributes`, `vmtkdistancetocenterlines` | Abscissa + angular metrics; harmonic (Laplace) maps; 2-D patch maps of surface data (e.g. WSS) |
| Modeling | `vmtkpolyballmodeller`, `vmtkcenterlinemodeller`, `vmtksurfacemodeller` | The union of spheres along a centerline as an implicit function |
| CFD pre | `vmtkflowextensions`, `vmtkmeshgenerator`, `vmtkboundarylayer`, `vmtktetgen`, `vmtkboundaryreferencesystems`, `vmtkboundarylabeler` | Cylindrical inlet/outlet extensions; TetGen with radius-based sizing; prismatic boundary layers |
| CFD post | `vmtkmeshwallshearrate`, `vmtkmeshvorticityhelicity`, `vmtkmeshlambda2`, `vmtkparticletracer`, `vmtkpathlineanimator` | WSS, vorticity, λ₂, particle tracing on solver output |
| Viewing / IO | `vmtkimageviewer`, `vmtkimagecurvedmpr`, `vmtksurfaceviewer`, `vmtk*reader/writer` | VTK render windows; curved MPR along a centerline |

### 1.2 What vmtk got right, which a successor must keep

1. **Centerline + radius is the canonical tube description** (the "tube function", a union of
   maximal inscribed spheres). It serves as a skeleton, a shape model, a CFD sizing function
   and a coordinate axis at once.
2. **Objective, reproducible decomposition.** Branch splitting is defined geometrically, not
   drawn by a person (Antiga & Steinman 2004), so it can be compared across a population.
3. **Vessel-native coordinates.** Abscissa, parallel-transport angle and bifurcation
   reference frames make vessel data comparable across subjects (Piccinelli et al. 2009).
4. **A composable command-line pipeline.** `vmtkimagereader … --pipe vmtklevelsetsegmentation
   --pipe vmtkmarchingcubes …` was scriptable 20 years ago, and it is exactly the shape an
   agent can drive today.
5. **Labeled boundaries for CFD.** The 1.5.x work (per-boundary labels on caps and flow
   extensions) shows the field still values this.

### 1.3 It is alive, which changes what a successor is for

vmtk is **being revived, not abandoned**: v1.5.0 (2026-07), 1.5.1 and 1.5.2 (2026-09-17/18)
brought PyPI wheels with ITK bundled, a WASM build, memory-leak fixes throughout the
centerline filters, boundary labels on every capper, and *"centerline branch splitting for
reduced-order models, ported from SimVascular"*. SlicerVMTK has grown to about 18 modules
(Extract Centerline, Cross-Section Analysis, Stenosis Measurement 1D/2D/3D, Guided
Artery/Vein Segmentation, CFD Mesh Generator, Arterial Calcification Preprocessor, …).

So the successor is not about replacing dead software. It is about **a different primary
representation**. Once vmtk is pip-installable, it becomes the *oracle* and the *CFD
back end*, in the same role the MLX package took after the torch decision.

---

## 2. Why vmtk is shaped the way it is

Each structural choice traces to a constraint of its time:

| 2004 constraint | vmtk's structural answer | Cost that persists |
|---|---|---|
| Segmentation is the hard, interactive step | Seeds, level sets, feature images; a human in the loop per case | Not cohort-scale; results depend on the operator |
| Output is one binary lumen | One unnamed surface; topology inferred from geometry | "Kissing" vessels merge into false loops; artery vs vein is invisible; nothing is named |
| Image discarded after segmentation | The surface is the only geometric truth | Every downstream error traces to surface noise |
| Marching cubes staircases | Smoothing, remeshing, kite removal, Voronoi simplification | Smoothing shrinks thin vessels and moves walls; parameters are tuned per case |
| Voronoi medial axis is noisy | Surface must be capped and clean; seeds pick the inlet and outlets | Spurious Voronoi spikes; seed placement is manual (`vmtkcenterlinesnetwork` later removed that step) |
| Deterministic pipelines | One number per measurement | No uncertainty on area, stenosis, radius or topology |
| CPU, VTK data model | `vtkPolyData` in, `vtkPolyData` out | Hard to batch; hard to run on a GPU; VTK is a heavy dependency |

---

## 3. What changed

### 3.1 Segmentation is cheap, named and multi-class

TotalSegmentator vascular classes (from `upstream/TotalSegmentator` `map_to_binary.py` and
`map_tasks_config.py` at `23bf617`, 2026-09-09; model spacings from the checkpoints' own
`plans.json` where TS does not override them):

| Task | Vascular classes | Model grid (mm) | Crop / trainer |
|---|---|---|---|
| `total` (cardiac part, **one 18-class softmax**) | aorta, pulmonary_vein, brachiocephalic_trunk, subclavian_artery_{L,R}, common_carotid_artery_{L,R}, brachiocephalic_vein_{L,R}, atrial_appendage_left, superior/inferior_vena_cava, portal_vein_and_splenic_vein, iliac_artery_{L,R}, iliac_vena_{L,R} (+ heart) | 1.5 iso (`total_fast`: 3) | whole body |
| `lung_vessels` | lung_arteries, lung_veins (+ lung_airways, lung_airways_wall), **one softmax** | **0.703 × 0.703 × 1.0** | lungs (5 lobes); **SkeletonRecall** |
| `coronary_arteries` | coronary_arteries (one class, the whole tree) | **0.7 iso** | heart; **SkeletonRecall** |
| `liver_vessels` | liver_vessels (one class; MSD Task08 model) | 0.80 × 0.80 × 1.5 | liver |
| `headneck_bones_vessels` | internal_carotid_artery_{L,R}, internal_jugular_vein_{L,R} | 0.75 × 0.75 × 1.0 | neck |
| `aortic_dissection` | aorta_true_lumen, aorta_false_lumen | 0.8 iso | none |
| `aortic_sinuses` | right/left/non coronary cusp | 0.7 iso | heart |
| `renal_arteries` | celiac_trunk, superior_mesenteric_artery, renal_arteries | 1.5 iso | none |
| `heartchambers_highres` | aorta, pulmonary_artery (+ chambers, myocardium) | model-native (plans not checked) | heart |
| `brain_structures`, `brain_aneurysm` | venous_sinuses; brain_aneurysm (no parent arteries) | 0.5 × 0.5 × 1.0; **0.39 × 0.39 × 0.5** | brain; none |

**The fine vessel tasks run at acquisition resolution, not at 1.5 mm.** Only the large
named vessels in `total` (and `renal_arteries`) sit on the coarse grid, and those are the
vessels that tolerate it. The tree-shaped tasks (lung, coronary) run at about 0.7 mm, which is
thin-slice CT's own in-plane spacing, and **were trained with a connectivity-preserving loss**
(`nnUNetTrainerSkeletonRecall`, Kirchhoff et al., ECCV 2024). Both facts change §5.3, §5.4 and
§9.

Two further facts matter structurally:

- **Every large named vessel in `total` is in the same softmax** (the cardiac part). The
  boundary between aorta and brachiocephalic trunk, or between aorta and iliac artery, is
  therefore an *exact* within-softmax interface. `junction` stores it, and the ranked union
  algebra holds across it. Across tasks (for example `renal_arteries` against the cardiac
  aorta), composition is only painter's algorithm (`haversack/docs/ranked-composition.md` §3).
- **Artery and vein are separate classes of one softmax** (`lung_vessels`; the `total` cardiac
  part). Where they touch, the model has already decided which is which, and the contact is a
  tie sheet between them, not a merge. A binary lumen cannot express this.

haversack also runs MOOSE, MRSegmentator, stock nnU-Net folders (for example a TopCoW or a
coronary model), MONAI bundles and VoxTell free-text prompts ("right vertebral artery").
The successor should be **engine-agnostic**: field in, geometry out.

### 3.2 The model's output is a field, and the field is sharp

From the ranked-store work (all *measured*):

- The renderable object is the per-structure signed margin `m_c = l_c − max(others)`: positive
  inside, zero on the boundary, exact under interpolation within a softmax, with unions exact
  (`m_S = max_in − max_out`). See `memory/project_margin_field_rendering.md`.
- The margin's trilinear zero set is **0.001 mm / 1°** from the exact surface on analytic
  phantoms. Quantization is 6–18 µm. SurfaceNets crossing placement is 0.0006 voxel median.
  Resampling this field is **dequantization, not smoothing**.
- Format 0.3 (rankfield, `keep="shell"` + log byte) restores 1.5 mm from 3 mm with 0.0037 % of
  voxels moved and thin structures within +0.4 % by volume, at about 2 MB per torso.

For vessels specifically: **vmtk's entire surface-repair stage exists to undo staircasing
that the field never has.**

### 3.3 Geometric layers already exist

- **`distance`**: nearest-surface distance in mm, uint8, truncated at 2 voxels, with the sign
  taken from `ranks[0]`. It is a cache, droppable and rebuildable. A per-*selection* exact
  signed field is **reseeded** from `ranks[0]` + `distance` with a band Eikonal on the crop:
  about **1 s for the lung-lobe selection at 1.5 mm** (from `total`), validated to a median
  of 3 µm on 1–3 voxel vessels. A `lung_vessels` crop at 0.7 × 0.7 × 1.0 mm has about 7× the
  voxels of the same lungs at 1.5 mm. Its reseed cost is unmeasured.
- **`junction`** + **`junction_pair`**: signed mm distance to the tie sheet between the two
  leading real classes, stored near triple lines, where two structures' interface meets a
  third label.
- **Owner pairs / flip edges**: a boundary-element view of the store. Airway wall thickness
  was measured from it at a median of 1.815 mm, 79 % of it sub-voxel.
- **Set operations are field Booleans**: union = max, intersection = min, difference =
  `min(s_A, −s_B)`. They are sign-exact everywhere.

### 3.4 Uncertainty is in the data

Margins, the stored tail and temperature give per-voxel confidence and alternatives
(`haversack/docs/ranked-probabilities.md`). vmtk's pipeline had nothing comparable to
propagate.

### 3.5 GPU, cohort scale and addressed artifacts

A warm whole-body `total` takes 36 s on an L40S, at about $2 per 100 whole-body cases. Serve
already addresses artifacts by `/v1/<source>/<ident>/<task>/…`, and stages of a cascade
replay *exactly* from a store. So a derived vessel graph can be an addressed, cached,
per-case artifact across a thousand cases, not a session on one person's workstation.

### 3.6 The LLM era

The scarce resource is no longer code. It is **validated semantics**: knowing that a
centerline is right, and being able to prove it. Three consequences:

- **Seed picking becomes naming.** "Left common carotid from its origin to the bifurcation"
  resolves against labels plus a graph. No clicks.
- **Composable, typed verbs with machine-readable provenance** are what an agent (or a
  person) can drive and audit. pypes was already the right shape. Keep that shape and type it
  (pydantic at the wire, as haversack's parameters are).
- **Effort goes into oracles and discriminating tests.** The distance-field work found three
  plausible-looking wrong implementations, and only a sphere test separated them. Vascular
  geometry will have the same failure mode.

---

## 4. The restructure: invert the dependency

```
vmtk (surface-centric)
  image ─seeds─▶ level set ─▶ binary lumen ─▶ marching cubes ─▶ SURFACE
                                                        │ smooth / remesh / cap
                                                        ▼
                          Voronoi ─▶ centerlines ─▶ branches ─▶ sections / maps / mesh

successor (field-centric)
  image ─engines─▶ RANKED FIELD (archive: margins of named classes, per softmax)
                        │
         ┌──────────────┼───────────────────────────────┬─────────────────────┐
         ▼              ▼                               ▼                     ▼
   selection ──▶ signed field (reseeded, unbounded)   label adjacency     confidence
   (named set)         │                               (junctions)          (margins)
                       ▼                                    │                     │
                 medial ridge + radius ◀────────────────────┘                     │
                       │                                                           │
                       ▼                                                           │
                 VESSEL GRAPH (named nodes/edges, polylines, radius ± interval) ◀──┘
                       │
      ┌───────────┬────┴──────┬──────────────┬─────────────┬──────────────┐
      ▼           ▼           ▼              ▼             ▼              ▼
   surface     sections    vessel chart   straightened   CFD mesh /     1-D/0-D ROM
   (view)      (± band)    (s, θ, r)      field          SDF solver     (graph+A(s))
```

Rules, each of them precedent in this workspace:

1. **The field is the archive.** Every derived object is a cache keyed by its parameters,
   rebuildable from the archive alone, with provenance appended. This is how `distance`,
   `junction` and `occupancy` already work.
2. **A composite is a view, not a thing.** The selection ("aorta + iliacs"), the grouping and
   the branch decomposition are chosen at query time. Every failure recorded in
   `ranked-store-composition` came from fixing a grouping too early.
3. **Name first, infer second.** Use label identity and adjacency wherever the model provides
   them. Fall back to geometry only within a label.
4. **Every measurement has an interval.** Evaluate at margin levels ±δ (or at sampled
   temperatures) and report the spread.
5. **Surfaces are for display and meshing, never for analysis.**

---

## 5. Module-by-module mapping

| vmtk | Successor | What changes |
|---|---|---|
| Vesselness, feature images | Kept for *refinement*; also computed on the margin field | Hessian of `m_c` is a clean tube detector with no intensity heterogeneity |
| Level sets, colliding fronts | Local refinement on the native image; gap bridging | Not the main segmentation path |
| Marching cubes + smoothing + remeshing | Zero-set extraction of the selection field; remesh by projection | No shrinkage and no staircase. Sizing from the radius field |
| Capping, end clipping | Caps = normal planes at named label ends / graph leaves, clipped against the field | Caps carry the vessel's name |
| Delaunay/Voronoi centerlines | Ridge of the reseeded signed field; the same Eikonal functional on the voxel grid | No seeds; no Voronoi spikes; radius = SDF |
| Branch extractor / clipper | Two levels: semantic (junction sheets) + geometric (tube-function overlap) within a label | Report where learned and geometric cuts disagree |
| Bifurcation reference systems | Unchanged mathematics, fed by the graph | Frames get names ("aortic arch → left subclavian") |
| Centerline sections, stenosis | Sampled from the field on normal planes; zero-crossing contours, sub-voxel | Area ± interval; no surface cutting |
| Curved MPR | Restore the logits onto a tube-native `(s, θ, r)` grid | Straightened *field*, not straightened pixels |
| Branch mapping / patching | Vessel chart `(s, θ)`, with any field mapped onto it | Includes wall *contact* (what each wall touches) |
| Polyball modeller | Tube function as one more SDF (max over balls) | Composes with field Booleans |
| Flow extensions, TetGen, boundary layers | vmtk/TetGen/gmsh remain the back end, fed an implicit surface + sizing + named boundaries | Also mesh-free and ROM consumers |
| CFD post (WSS, λ₂) | Unchanged; results mapped onto the vessel chart | — |
| Viewers | sdfview's renderer contract | Field rendering, not polydata |

### 5.1 Segmentation becomes an engine plus refinement

The primary path is an engine run (TS task, stock nnU-Net model, MONAI bundle, VoxTell prompt).
vmtk's segmentation tools survive in two narrow roles:

- **Local refinement on the native image.** *Measured* on organs (surface-relaxation
  experiment, 2026-09-02): the data term only corrects **bias** within a one-voxel
  envelope. It is neutral on good 1.5 mm predictions of soft-tissue organs and helps most on
  high-contrast boundaries (liver +0.28 mm). *Hypothesis:* contrast-enhanced lumen is the
  best case the data term will ever see (high contrast, thin), so refinement pays off for
  vessels much more than for organs. *Test:* the relaxation harness on an AVT or ImageCAS
  lumen against its reference, 3 mm hint vs 1.5 mm model.
- **Gap bridging** (§5.4): colliding fronts or a minimal path in the *margin* field between
  two fragments of one named vessel.

**Vesselness on the margin field** (*hypothesis*): Frangi on intensity suffers from contrast
heterogeneity, calcium and neighboring bone. The margin `m_c` of a vessel class has none of
those, so the eigen-analysis of its Hessian should give a clean tube direction even where the
vessel is thinner than a voxel. This matters for §5.3's unresolved regime.

### 5.2 The surface is a view

- Extract the zero set of the selection's signed field (SurfaceNets or dual contouring on the
  interpolant). Then **remesh by projection**: move vertices to the zero set with a Newton
  step on `m / |∇m|`, using the signed pair field, never `gap1` (settled 2026-08-30).
- Target edge length comes from the local radius (§5.3), which is exactly
  `vmtksurfaceremeshing`'s radius-adaptive sizing. Here the radius field exists everywhere
  without first computing centerlines on a surface.
- **No smoothing step.** vmtk's Taubin/Laplacian smoothing shrinks thin vessels and moves walls
  by an amount that depends on the operator's settings. The field needs neither.
- **Prerequisite (known):** today's stored `distance` is built from the *labelmap* and carries
  terraces: 0.50 mm / 20° normals on phantoms. Rebuilt from the *margin's zero set* it is
  0.05 mm / 4°. Any medial-axis or meshing work must start from the margin-built field (§10
  step 1).

### 5.3 Centerlines: the ridge of a signed field

vmtk's centerline is the path minimizing `∫ ds / R` over the Voronoi diagram, where `R` is the
maximal inscribed sphere radius. Inside a vessel, `R` at a medial point **is** the unsigned
distance to the wall. So the same functional can be solved on the voxel grid over the
selection's signed field `s`:

1. **Selection field.** Reseed an *unbounded* signed field for the named selection (the stored
   2-voxel truncation hides the medial axis of anything wider than about 4 voxels). Use the
   crop, band Eikonal, seeded from margin zero crossings: ~1 s scale *measured* for lungs.
2. **Endpoints without seeds.** The inlet is the named proximal end: the junction sheet with
   the parent label (for example aorta → left subclavian), or the named cut where a class ends.
   Outlets are local maxima of geodesic distance from the inlet within the selection (the
   standard "farthest point" rule), filtered by persistence (§5.4) so bumps do not become
   branches.
3. **Minimal paths.** Fast marching from the inlet with cost `1/(ε + |s|)^p`, then backtrace by
   gradient descent. This is the same Eikonal-plus-backtrace that `vmtkcenterlines` runs on
   the Voronoi diagram, only on a dense grid.
4. **Ridge refinement.** Snap each path sample to the local maximum of `|s|` in the plane
   normal to the tangent (1-D parabolic fit along two in-plane axes). This removes voxel
   wobble. The sub-voxel radius is `|s|` at the ridge point.
5. **Radius is the SDF.** The maximal inscribed sphere radius, vmtk's key array, is read
   directly and is sub-voxel accurate wherever the surface is.

**Two regimes, split near r ≈ 2 mm, and the model's floor is its own (§12.2).** An earlier
draft counted the regimes in model voxels and concluded that at 0.7 mm most of a lung tree is
"resolved". On the 0.7 mm grid only 2–4 % of the centerline length has a model radius under
one voxel, but the model's radius has a **floor near r ≈ 1 mm** (about 1.4 voxels). The
same-patient test settles where the floor comes from. The same anatomy was blurred from
0.625 mm slices to 1.25 and 2 mm. At the same world points the model's radius moved by
≤ 0.05 mm while the vessel centers lost 100–300 HU. Blurred further, to 3.75–5 mm slices, the
model **drops** thin vessels (16–26 % of the r < 1.25 mm points kept) rather than fattening
them. A first reading of the 2 mm demo blamed the scanner for inflated thin-vessel radii;
that reading is withdrawn. Whether the model's thin radii are *correct* is a separate
question. It needs a PSF the images do not supply precisely enough (§12.2): on two CT
angiograms, image and model radii agree within ~0.1 mm; on the 2 mm demo, the image reads
~0.4 mm smaller.

This floor belongs to the **input model**, not to the method. Any method measuring the same
segmentation inherits it: vmtk, fed this field's surface, matched our radii to 0.03 mm
(§12.2). It sits at clinical-CT resolution, where the scanner's own blur (~2 mm full width)
is already wider than a 1 mm vessel. It moves with the input. A segmenter trained on
higher-resolution data (photon-counting or ultra-high-resolution CT, micro-CT for specimens)
would lower it roughly in proportion, to ~0.35 mm at the same 1.4 voxels on a 0.25 mm grid,
and nothing in this section depends on the grid. Below whatever floor the segmenter has,
caliber survives as contrast, which is the image-side sizing above, given a well-measured PSF.

- **Sized by the field** (r ≳ 2 mm: hilar, lobar and larger segmental vessels, and on
  `total`'s grid the aorta, iliacs, carotids, pulmonary trunk and main portal vein). The ridge
  of `s` is the medial axis and `|s|` is the radius. Model and image radii agree within about
  0.1–0.3 mm here (§12). *Hypothesis:* this beats vmtk's Voronoi on a smoothed marching-cubes
  surface. The spurious spikes of a noisy Voronoi diagram have no counterpart in a field
  without terraces.
- **Placed by the field, sized by the image** (r ≲ 2 mm: most of a peripheral lung tree, and
  distal coronary branches). A ridge-refined centerline sits in the right place, and the
  margin localizes the model's boundary to sub-voxel precision. But below the floor, that
  boundary is not the lumen. **The caliber must come from the image**, from a
  blurred-cylinder fit with the acquisition's point-spread function (PSF), centered on the
  field's ridge. That is the job vmtk's FWHM feature image and active tubes were built for:
  *the field places the tube, the image sizes it.* The PSF must come from the image itself
  (an edge fit that does not use the model's boundary) or from the DICOM. §12 shows that
  fitting it against the model's boundary is unreliable. Where even the ridge of `s` is one
  sample deep, use the **ridge of the margin** `m_c`, with direction from its Hessian (§5.1).
- **Anisotropy.** `lung_vessels` is 0.7 × 0.7 × 1.0 mm and `liver_vessels` 0.8 × 0.8 × 1.5 mm,
  so a vessel running along z is resolved differently from one in-plane. Do every step (reseed,
  cost, ridge fit) in mm, never in voxel units. The field has no preferred axis, so the
  anisotropy is a sampling-density issue, not a bias, as long as the kernels respect the
  spacing.

`vmtkcenterlinesnetwork` (seed-free, 2018) and `vmtknetworkextraction` should be run as oracles
on both regimes.

### 5.4 Topology: a two-level graph

**Semantic level, read from labels.** Nodes are named vessels. An edge exists where two named
classes share a within-softmax interface, which the stored `junction` layer and owner-pair
tables already enumerate. The **ostium** (branch origin) of, say, the left subclavian artery is
where the aorta|subclavian tie sheet meets background. That is a triple line, and the
`junction` layer exists to represent exactly that. Its zero set on the surface is the ostium
rim, and a plane fit gives the ostium plane and normal. vmtk reconstructs this geometrically
from polyball overlap. Here the model has already drawn it.

**Geometric level, inferred within a label.** For trees under one class (`lung_arteries`,
`coronary_arteries`, `liver_vessels`), build a skeleton graph from §5.3, as vmtk does.

Three topological questions the field answers and a binary lumen cannot:

1. **Breaks.** Vessel masks fragment, and in older models this is the most common failure of
   vessel segmentation. The lung and coronary models were trained with SkeletonRecall to
   resist it, and on the demo it worked (§12). The main arterial component holds 99.8 % of
   the voxels. Only 11 arterial and 8 venous fragments lie within 2.1 mm of their own tree,
   totaling 136 and 182 voxels. Lowering the margin level from 0 to −4 bridges almost
   nothing. Keep the mechanism, but expect it to matter for models without a topology loss.
   It works like this. Take the superlevel sets of the selection margin `m_S` as the level drops
   below zero. Two fragments merge at level `−δ`, at a saddle, along the maximin ("widest")
   path. **δ is how confident the model was that the gap is real.** A 0-dimensional cubical
   persistence diagram (gudhi / cripser class of tools) ranks every candidate bridge by δ and
   by bridge length. Add geometric continuation (the two ends' tangents agree) and gaps get
   bridged with a stated confidence. **Limit:** margins floor at `−clip` (8 logits), and the
   kept band is about 1 voxel wide for vessels. So only *short, shallow* gaps are visible, and
   anything longer reads as a uniform `−clip`. The geometric continuation must carry the rest,
   and those bridges get flagged as geometric only.
2. **Kissing vessels.** An artery touching a vein in the lung makes a false loop in a binary
   lumen. In the `lung_vessels` softmax, the contact is an artery|vein tie sheet, so the graph
   never merges them. *Measured* on the demo (§12): arteries alone have first Betti number 2,
   veins 1, and their binary union **135**, so **132 false loops** from 6,153 touching voxel
   faces, which a single-lumen pipeline would have to find and cut. Those counts treat voxels
   touching at a corner as connected. Under the field's own topology (§12.3) the union has
   95 loops on the demo and 88 / 73 on the two CT angiograms: still dozens. Within one class (two branches of `lung_arteries` touching), 1-D
   persistence of the selection flags short-lived loops as likely contacts.
3. **Outside-body junk.** Cascade models label air outside the patient: *measured*, 73 % of
   `lung_airways` on CT_Abdo, 1,034 of 1,060 stray components under 50 voxels. Intersect every
   selection with the real `body_mask()` (not its bounding box) before skeletonizing, or the
   graph grows hundreds of spurious leaves. (It did not occur on the idc-torso1 demo: 0.00 %
   of any class lies in air connected to the volume border. It depends on the case. Keep the
   mask.)

### 5.5 Branch splitting and bifurcations

The Antiga–Steinman decomposition is a field operation already. Each centerline's tube
function is an SDF (max over its balls). The bifurcation region is where two tube functions
both claim a point, and the clipping points sit one radius beyond divergence. Implement it
with field Booleans, not polyball VTK filters.

There are now **two decompositions**: the learned one (where the model moved from aorta to
subclavian) and the geometric one (Antiga–Steinman). Both are views. **Report their
disagreement per junction.** It is informative. A consistent offset means the training
annotators' convention differs from the geometric definition, like the ~0.8 mm inward
convention measured on AMOS hand segmentations. A scattered offset means one of them is
unstable. Bifurcation reference systems (`vmtkbifurcationreferencesystems`) are unchanged
mathematics on better inputs.

### 5.6 Sections and measurements, with intervals

- **Cross-sections** come from sampling the field on a disk normal to the centerline: a
  zero-crossing contour at sub-voxel accuracy, with area from the contour. No surface
  cutting, and no dependence on mesh density. Use the Crofton-style estimator rather than
  counting (`haversack/docs/ranked-measurement.md`).
- **Intervals.** Take the contour at `m = ±δ` for a δ calibrated per class (§8 tier 3), or
  sample from the stored probabilities. Area, minimum diameter, **percent stenosis**, aneurysm
  sac volume and neck width all become `value [lo, hi]`. For a clinical measure like
  stenosis, the interval matters more than the point estimate. *Limit:* inside the
  saturation band only, which is `clip/k`. It was measured at about 1.16 mm for thin
  structures in the 1.5 mm `total` stores (k ≈ 6.9 logits/mm). The steepness `k` of a 0.7 mm
  model is unmeasured and likely higher per mm, so the band would be narrower in mm but a
  similar number of voxels. Either way it is plenty for a boundary.
- **Tortuosity, curvature, torsion.** Use the Frenet quantities on the ridge-refined
  centerline, with smoothing parameters recorded in provenance. Better, estimate them from a
  fitted spline whose knot spacing is tied to the radius, so the noise scale is stated rather
  than tuned.
- **Straightened field.** rankfield restores logits onto an arbitrary grid. A tube-native grid
  `(s, θ, r)` along the centerline makes the lumen boundary a function `r(s, θ)`, the zero
  crossing along each ray. Area profile, eccentricity, stenosis, dissection flap position
  (the true|false lumen tie sheet in `aortic_dissection`) and wall contact all become 1-D and
  2-D signal processing. This replaces `vmtkimagecurvedmpr`, which straightened *pixels*.
  (*Needs:* a non-affine `Mapping`, meaning point-sampled restore. The fused kernel is
  trilinear at arbitrary points, so this is plumbing, not research.)

### 5.7 Vessel coordinates

Abscissa `s` and parallel-transport angle `θ` give each vessel a chart. The ranked store is
Eulerian, a world grid with no material addresses. The chart supplies material-like addresses
for tubes, which is the role the Auckland scaffolds play for organs
(`memory/project_liver_atlas_registration.md`). Map any per-point quantity onto `(s, θ)`: WSS
from CFD, wall thickness (owner-pair boundary differencing, as for the airway wall), and a
**contact map** of what the outside of each wall touches (liver, bone, the paired vein), read
from the owner pairs. A population atlas of a vessel is then a set of functions on a common
chart, aligned at named landmarks: ostia and bifurcations.

### 5.8 CFD and reduced-order models: three consumers

1. **Body-fitted meshes.** Keep vmtk (1.5.x wheels), TetGen or gmsh as the back end. Give it
   the projected surface (§5.2), a sizing function from the radius, and **named boundaries**:
   `inlet: aorta_ascending`, `outlet: left_subclavian`. vmtk 1.5.1 just added
   label-identified boundaries and per-boundary flow-extension lengths, so the interface
   already exists.
2. **Mesh-free.** Immersed-boundary, lattice-Boltzmann and cut-cell solvers consume an SDF
   directly. The reseeded selection field is exactly their input, so there is no meshing
   stage.
3. **1-D/0-D reduced-order models.** These need a graph with `A(s)` per edge and junction
   topology. This is the graph of §5.4 plus the area profiles of §5.6, with intervals. vmtk
   just ported SimVascular's ROM branch splitting, which shows demand, and the graph here
   carries names and uncertainty that SimVascular's lacks.

### 5.9 Visualization

sdfview already renders fields by its client contract (reseeded selection fields, analytic
shell integration, junction reconstruction in world space). Vessel-specific additions: tubes
colored by `(s, θ)` functions, a centerline overlay, and uncertainty shown as shell width.

---

## 6. Data model and packaging

**Objects** (Python API first, CLI verbs over it, then serve endpoints):

| Object | Is | Keyed by |
|---|---|---|
| `FieldArchive` | rankfield parts (+ `distance`/`junction` caches) | case, engine, task, model version |
| `Selection` | a named set of classes within one softmax (or painter's across, flagged) | archive + class names |
| `SignedField` | reseeded, unbounded, cropped signed distance | selection + reseed parameters |
| `VesselGraph` | named nodes, edges with polylines, radius ± interval, provenance per edge (semantic / geometric / bridged-with-δ) | selection + extraction parameters |
| `Chart` | `(s, θ)` per edge, parallel-transport frames, bifurcation frames | graph |
| `Section`, `Profile` | per-`s` contours, area/diameter ± interval | graph + δ |
| `SurfaceView`, `MeshView` | projected zero set; exported mesh with named boundaries | selection + sizing |

**Layering** (haversack's pattern, enforced by a test): a **kernel layer** in torch + numpy
(reseed, fast marching, ridge refinement, persistence, sampling on arbitrary points) that
knows nothing of tasks or files. A **pipeline layer** holds naming, graphs, IO and provenance.
Use GPU where measurement says it wins. The distance work showed a dense form wins on CUDA
(0.5 s) and *loses* on Apple silicon to a numpy band (15.4 vs 2.7 s), so every kernel needs a
reference implementation and a per-backend decision. **No VTK in the core.** VTK and vmtk sit
at the edges, as mesh export and as the oracle.

**Recommendation:** a **new sibling library** that consumes rankfield, the way rankfield was
split out of haversack, not a haversack module. haversack produces fields, and this library
turns fields into vascular geometry. It should also accept a plain labelmap or an SDF, in
degraded mode, so its value does not depend on the ranked format. The name is yours to choose.

**Serialization** is deliberately open (rankfield's rule). The graph wants a small JSON
(nodes/edges/polylines/intervals/provenance) that can travel beside a duckn store as an
extension or on its own. Decide that once the object stabilizes.

---

## 7. Where the LLM era enters the design

- **Queries instead of clicks.** `graph.path("aorta", to="left_common_carotid_artery")` or
  "stenosis profile of the right renal artery" resolve against names. The typed verb surface
  is small enough for an agent to use correctly, and every result carries provenance an agent
  can cite: which engine, which δ, which edges were bridged and at what confidence.
- **Engines on demand.** A structure not in the catalog becomes a VoxTell prompt or a stock
  nnU-Net model (TopCoW for the circle of Willis). The geometry layer does not care where the
  field came from.
- **Oracles over opinions.** Cheap code makes it easy to build something that renders
  plausibly and is wrong. That happened three times with the distance field. The investment
  should go into §8, not into features.

---

## 8. Validation plan

**Tier 0: analytic phantoms** (truth is exact; haversack already has `phantoms`). Straight,
curved and helical tubes; tori; Y bifurcations with known angles and radius ratios (Murray's
law and not); stenoses of known area; sub-voxel tubes swept across radius 0.3 to 3 voxels to
**locate the regime boundary of §5.3**; kissing tubes; tubes with a gap of known length.
Score centerline distance, radius error, section area, branch-point position and topology.
*Choose tests that discriminate*: a straight axis-aligned tube passes almost any
implementation, as a planar surface did for the chamfer distance.

**Tier 1: vmtk parity.** Run `vmtkcenterlines`, `vmtkbranchextractor` and
`vmtkcenterlinesections` on the *same* case (the aorta and its branches on idc-torso1,
from its surface) and compare. On resolved vessels the two should agree within the surface's
own error. Disagreement then decomposes into surface error vs algorithm, because tier 0
supplies the truth for both.

**Tier 2: public references.** AVT (aortic vessel trees), SimVascular's Vascular Model
Repository (segmentations + centerlines + CFD), ASOCA / ImageCAS (coronary CTA), PARSE
(pulmonary artery), TopCoW (circle of Willis, CTA + MRA). Each supplies lumen references;
some supply centerlines.

**Tier 3: uncertainty calibration.** Do the `[lo, hi]` intervals of §5.6 cover the reference
at their stated rate? This is where δ per class is set. Without it, intervals are decoration.

---

## 9. Honest limits

- **Resolution is the input model's, and every method shares it.** Today's segmenter
  (TotalSegmentator `lung_vessels`, 0.7 mm grid) draws a radius floor near 1 mm, about 1.4
  of its voxels, on every scan. The floor does not follow slice thickness on the same anatomy
  (§12.2), so it comes from the model's training labels and grid, not from this method. vmtk
  given the same surface measures the same radii, to 0.03 mm. Two consequences:
  - Whether radii below ~1.5 mm are right cannot be settled from these images. The image-side
    estimate agrees within ~0.1 mm on two CT angiograms, reads ~0.4 mm smaller on the 2 mm
    demo, and moves with its PSF assumption.
  - Every caliber-derived quantity inherits the doubt: area, stenosis, ROM resistance (∝ r⁻⁴)
    and CFD wall position. Ground truth needs a phantom or a gated high-resolution scan (§8).

  The limit moves with the input. A segmenter trained at higher resolution lowers it roughly in
  proportion, with no change to the method, which works in mm on any grid; storage grows as
  1/h³ (about 20× at 0.25 mm, cropped to the lungs). Below the segmenter's floor, image-side
  sizing with a scanner-measured PSF can go further on today's scans.
- **The PSF is the image method's weakest input.** Fitted against the model's boundary
  (§12.1), σ absorbs the model's scatter (0.76 vs 1.09 mm, arteries vs veins, one scan).
  Fitted from the image alone with a free edge per profile (§12.2), it is consistent across
  reconstructions. It sees an added 2 mm boxcar, but reads it as +0.475 mm² against the
  true +0.333, in both patients. It sits near 0.9–1.0 mm even for 0.625 mm slices, which is
  more than the slice profile explains: the big hilar walls it is measured on pulsate (none
  of these scans is ECG-gated) and are not ideal steps. So σ is probably an upper bound, and
  image radii scale with it.
- **Too-thick slices lose vessels silently.** At 3.75–5 mm the model finds only 16–26 % of
  the thinnest centerline points and 38–70 % at r 1.25–1.5 mm (§12.2). A tree extracted from
  a thick scan is pruned, not merely coarser. This too is the input model's behavior, and any
  method downstream inherits it. The acquisition therefore belongs in a vessel graph's
  provenance, beside the model.
- **Only part of the field measurements was re-taken at 0.7 mm.** The saturation band and
  the zero-crossing radii were re-taken on the demo (§12). The 0.001 mm zero-set accuracy and
  the format-0.3 restore error still come from 1.5 mm and 3 mm `total` stores and phantoms.
  The demo store is format 0.2. Its bytes predate 0.3's true gaps and cannot be upgraded, so
  a 0.3 measurement needs a re-emit.
- **Coverage.** TS has no intracranial arterial tree, and `brain_aneurysm` has no parent
  vessels, so vmtk's signature application (aneurysm neck, parent artery geometry) needs
  another engine. Coronaries, liver and lung trees are single-class, so their internal
  topology is entirely geometric, just as in vmtk.
- **The margin is a band, not a global field.** It is informative only to `clip/k`: about
  1.16 mm (0.77 voxel) for thin structures at 1.5 mm, and **0.88 mm (1.25 voxels)** for the
  0.7 mm lung vessels (§12). Then it floors at `−clip`. Medial work on large vessels must use the reseeded
  distance. The margin serves boundaries, sub-voxel ridges and short gaps only.
- **The stored `distance` has to be rebuilt first**, from the margin's zero set instead of the
  labelmap (0.50 → 0.05 mm on phantoms). Medial axes amplify surface noise. Voronoi spikes
  are vmtk's version of this problem, and terraced EDT ridges would be ours.
- **Cross-task junctions are painter's.** The renal arteries' ostia on the cardiac-part aorta
  are only geometric. `map_to_binary.py` lists a `renal_arteries_auxiliary` aorta class, but
  haversack's task audit (2026-09-22) found that no public v2 checkpoint carries it. So the
  renal model's own softmax has no aorta to form a junction with.
- **Learned cuts are conventions.** Where TS stops "aorta" and starts "iliac" reflects its
  training annotations, not an anatomical definition. That is why §5.5 keeps the geometric
  decomposition beside it.
- **Trust.** Much of the CFD community's confidence rests on vmtk-validated pipelines, and
  anything used for clinical decisions (FFR-CT-like) is a regulatory matter well outside this
  scope. Keeping vmtk as the meshing back end and the tier-1 oracle is also the trust
  strategy.

---

## 10. A sequence, cheapest first

1. **Margin-built `distance`** (the known fix from the surface-relaxation addendum). It is a
   prerequisite for everything medial, and it improves rendering on its own.
2. **Phantom suite + medial kernel** (reseed → fast march → ridge refinement), scored against
   analytic truth and against vmtk on the same phantoms. Output: the resolved/unresolved
   boundary in voxels.
3. **Aorta and branches on idc-torso1** (one softmax): semantic graph from junctions, ostia
   from triple lines, centerlines per label, sections with intervals. Compare with vmtk on the
   extracted surface.
4. **Lung arteries and veins.** Done on idc-torso1 and two thin-slice CTAs, including the
   same-patient thickness ladders and an image-only PSF (§12–12.2). Remaining: a body mask on
   the full CT (the outside-body check fails on cropped grids), a phantom-tree test of
   zero-set accuracy, and a ground truth for thin-vessel caliber (a phantom or a gated scan).
5. **Centerline kernel, branch graph, vmtk comparison, sections with intervals, straightened
   vessel.** Done as a prototype on the lung trees (§12.2). Remaining: resolve voxel-graph
   corner connections against the field; step 3's semantic graph on the `total` aorta.
6. **Decisions** (below), then packaging.

## 11. Decisions that are yours

- **Scope:** geometric analysis only, or through to CFD (and whose solvers)?
- **Home:** new sibling library (recommended) vs a haversack layer; the name.
- **vmtk's role:** oracle + mesh back end (recommended, as MLX became), or a runtime dependency
  of the core.
- **Graph serialization:** a duckn extension or standalone JSON.
- **First target anatomy:** aorta + branches (one softmax, named branches, 1.5 mm), lung
  vessels (0.7 mm, artery/vein in one softmax, hardest topology, emits already on hand), or
  coronaries (0.7 mm iso, SkeletonRecall, the clinically loudest use case, single class).

## 12. First measurements: the idc-torso1 `lung_vessels` demo (2026-09-23)

The store is the one sdfview's demo menu opens ("lung vessels + airways — 0.7 mm"):
`haversack/data/duckn_demo/idc-torso1/lung_vessels.duckn`. It is ranked format 0.2, one
5-class softmax (background, airways, airway wall, arteries, veins; depth 5 = exhaustive,
clip 8), a 321 × 370 × 486 array at 1.0 × 0.703 × 0.703 mm, with `distance` and `junction`
layers. The source is IDC `cptac_ccrcc` C3N-01524, Philips Brilliance 64, "NEPHROGENIC"
contrast phase, **2.0 mm slices at a 1.0 mm increment, kernel B**, 0.651 mm pixels. Scripts:
`research/vessels/{load,measure,radius,psf}.py`, run with haversack's venv. Decoding takes 11 s
and every analysis under 3 min on the M2.

**Checks first.** The CT was resampled onto the store grid from the duckn geometry. Shifting
it ±1–3 voxels on each axis against the artery centerlines, zero shift gives the highest
centerline HU, and every 1-voxel shift is at least 50 HU darker. `ranks[0]` agrees with the argmax of
the decoded deficits on 99.9986 % of voxels; the rest are quantized ties.

| Quantity | Arteries | Veins | Notes |
|---|---|---|---|
| Volume | 221 mL | 253 mL | airways 61 mL, airway wall 85 mL |
| Components (26-conn.) | 43; main holds 99.8 % | 47; main 97.7 % | SkeletonRecall shows |
| Fragments within 2.1 mm of own tree (break candidates) | 11 (136 vox) | 8 (182 vox) | the rest are isolated specks; 2 venous fragments (9 vox) touch arteries |
| Components ≥ 20 vox at margin level 0 / −1 / −2 / −4 | 13 / 12 / 13 / 15 | 22 / 22 / 22 / 22 | lowering the level bridges almost nothing |
| In outside air | 0.00 % | 0.00 % | CT_Abdo's 73 % airway leak does not recur here |
| First Betti number b₁ alone → in the binary union | 2 | 1 | **union: 135, so 132 false loops**; 6,153 contact faces |
| Margin steepness `k` at the surface, p10/50/90 | 6.3 / 9.1 / 12.2 logits/mm | 6.1 / 9.1 / 12.1 | band `clip/k` = **0.88 mm = 1.25 voxels** (1.5 mm `total`: 1.16 mm = 0.77 voxel) |
| Interior saturated (m ≥ clip) | 50 % | 47 % | |
| Centerline radius p10/50/90, labelmap EDT | 0.99 / 1.41 / 2.22 mm | 1.00 / 1.57 / 2.22 | |
| Centerline radius, zero crossings at the voxel | 0.66 / 1.27 / 1.99 | 0.76 / 1.35 / 2.01 | skeleton voxels sit off the ridge |
| Centerline radius, zero crossings, ridge-refined | 1.01 / 1.51 / 2.23 | 1.10 / 1.57 / 2.26 | median EDT − ridge: −0.03 / −0.04 mm |
| Centerline length with radius < 1 model voxel | 2.9 % | 1.8 % | model grid ≈ all "resolved" |

**First reading (superseded by §12.1): model radii for thin vessels track the scanner.**
Centerline HU rises steadily with the model's radius: −683 HU at r < 1 mm, −343 at
1.5–2 mm, and +115 to +141 (blood with contrast) only above 3 mm. An edge-spread fit on the
walls of vessels with r > 4 mm estimates the blur. §12.1 shows the fit also absorbs the
model's boundary scatter:

| | Arteries | Veins |
|---|---|---|
| PSF σ (orientation-averaged Gaussian) | 1.02 mm (FWHM 2.40) | 1.11 mm (FWHM 2.62) |
| Lumen HU (large vessels) | 128 | 115 |
| Half-max level vs model boundary | 0.49 mm inside | 0.56 mm inside |

(The background of that fit is mixed hilar tissue, −231 / −159 HU, so the half-max offset is
only indicative.)

For a disc of radius `a` blurred by σ, the center reads `f = 1 − exp(−a²/2σ²)` of the
lumen–parenchyma contrast. Inverting at each centerline point, with parenchyma at −860 HU:

| Model radius (ridge) | Share of arterial centerline | Center HU (art.) | Image radius `a`, arteries | Image radius `a`, veins |
|---|---|---|---|---|
| < 1.25 mm | 23 % | −665 | 0.67 mm | 0.76 mm |
| 1.25–1.50 | 26 % | −576 | 0.83 | 0.92 |
| 1.50–1.75 | 29 % | −389 | 1.16 | 1.24 |
| 1.75–2.00 | 9 % | −155 | 1.61 | 1.71 |
| 2.00–2.50 | 6 % | +10 | 2.10 | 2.22 |
| 2.50–3.00 | 2 % | +91 | 2.60 | 2.71 |

Above r ≈ 2 mm (≈ 2σ) model and image agree within 0.1–0.2 mm. Below it the model's lumen
is 0.4–0.5 mm too large in radius: a factor of 1.6–1.8, or about 3× in cross-sectional area
for the thinnest vessels. That is most of the tree by length. Consequences:

1. ~~§5.3's regime boundary is σ of the acquisition, not the model grid.~~ Not supported:
   see §12.1. The boundary near r ≈ 2 mm held, but its cause is open.
2. The ridge refinement works: it moves radii from 1.27 to 1.51 mm (median) and matches the
   labelmap EDT to 0.03 mm. **Centerline placement** from the field is sound. **Caliber**
   below 2σ is not.
3. "Field precision" and "lumen accuracy" separate here. The zero set locates the model's
   boundary to microns, and below r ≈ 2 mm the model's boundary is not the lumen.
4. This is exactly the measurement vmtk's image-side tools exist for (FWHM features, active
   tubes). They become the caliber stage in the successor, run on the field's centerlines.

**Caveats** (see §9): the PSF is anisotropic and was fitted as one σ; circular sections and a
uniform lumen HU are assumed; there is no ground truth.

### 12.1 Two thin-slice CT angiograms (2026-09-23)

Found with `research/vessels/idc_pairs.py`: contrast chest studies in IDC v24 that have a thin
and a thick reconstruction of the same study. Both are CC BY 4.0. DICOM in
`~/tmp/data/idc_vessels/` (manifest + citations), stores in `~/tmp/data/vessels/runs/`.

- **cptac_luad C3N-00704**, 2001-04-10: CTPA, Isovue 370, GE LightSpeed VCT. Series 4 is
  0.625 mm STANDARD at 0.564 mm pixels. The same study has 1.25 mm LUNG and 3.75 mm STANDARD
  recons. A cavitating right-upper-lobe mass; the left lung is clean.
- **cmb_brca MSB-02664**: CTA for PE, GE Revolution Apex. Series 304 is 0.625 mm STANDARD at
  0.633 mm pixels. The same study has 1.25 mm and 5.0 mm ("DLIR M") recons. Clear lungs;
  shallow inspiration.

Both ran as `ts.v2:lung_vessels` on the M2 (haversack main before 0.13.0), writing ranked
format 0.4 and seg 0.9. Runs took 2.5 and 3.6 minutes. Analysis: the same four scripts, with
the case name as the argument.

C3N-00704 first ran the MPS pool out of memory twice, in the ranked encoder. rankfield sized
the encoder's slab by the 1-byte shell mask, so at K = 5 the slab was the whole volume. That
store was written through a wrapper that fixed the slab (identical bytes, 5.9 GB peak). **Fixed
upstream:** rankfield 0.3.5 (`1b343a9`) sizes the slab from a counted memory budget, and
haversack 0.13.0 uses it, so the wrapper was removed. rankfield 0.3.5 (`fe07414`) also lets
`margin()` / `deficit()` read a stored field directly. The fix was confirmed on this same case.
Plain `haversack segment` on C3N-00704 (work in `~/tmp/data/vessels/bench_encode/`) completed at
a 5.87 GB peak, including with a forced 1-byte budget (one-plane slabs). rankfield 0.3.6's Metal
selection kernel took the encode from 14.1 s to 3.5 s. Every array of the wrapper-written store
(ranks, support, tail, distance, junction, both parts) is byte-identical to the integrator's
0.3.5 and 0.3.6 stores; only layer 0's names differ. Re-decoding the MSB-02664 store under
rankfield 0.3.6 also reproduced the cache byte for byte (labels, all three margins, CT).

| | Demo idc-torso1 | C3N-00704 | MSB-02664 |
|---|---|---|---|
| Acquisition | 2.0 mm, kernel B, nephrogenic phase | 0.625 mm, STANDARD, CTPA | 0.625 mm, STANDARD, CTA-PE |
| Artery / vein volume | 221 / 253 mL | 152 / 184 mL | 196 / 161 mL |
| Main component, art / vein | 99.8 / 97.7 % | 99.2 / 99.1 % | 98.1 / 96.1 % |
| Break candidates (fragments ≤ 2.1 mm from own tree), art / vein | 11 / 8 | 0 / 3 | 1 / 5 |
| False loops if arteries and veins share one lumen | 132 | 105 | 129 |
| Margin steepness `k` median; band `clip/k` | 9.1; 0.88 mm | 9.3 / 9.0; 0.86–0.89 mm | 8.3; 0.97 mm |
| Ridge radius p10 / median, arteries | 1.01 / 1.51 mm | 1.10 / 1.54 | 1.02 / 1.61 |
| Centerline under one model voxel, art / vein | 2.9 / 1.8 % | 1.8 / 2.3 % | 3.6 / 3.9 % |
| Median EDT − ridge radius | −0.03 / −0.04 mm | −0.03 / −0.03 | −0.04 / −0.03 |
| Center contrast `f` at model r < 1.25 mm (art) | **0.20** | **0.34** | **0.43** |
| `f` at model r 1.50–1.75 mm (art) | 0.48 | 0.72 | 0.75 |
| Fitted σ, art / vein | 1.02 / 1.11 mm | **0.76 / 1.09** | 0.80 / 0.98 |
| Image radius at model r < 1.25 / 1.50–1.75 mm (art) | 0.67 / 1.16 | 0.69 / 1.21 | 0.85 / 1.33 |

(`f` = (center HU − parenchyma) / (large-vessel lumen − parenchyma). Parenchyma −857 / −724 /
−683 HU; lumen 128 / 538 / 346 HU.)

**What holds on all three scans.**
- The margin band is about 0.9 mm (1.2–1.4 voxels).
- Merging arteries and veins into one lumen creates 105–132 false loops counting corner-only contacts, 73–95 under the field's own topology (§12.3).
- SkeletonRecall trees are nearly unbroken, and lowering the margin level bridges almost
  nothing.
- Ridge refinement matches the labelmap EDT to 0.03–0.04 mm.
- Model and image radii agree above r ≈ 2 mm and diverge below it, the model's being the
  larger.

**What the thin scans change.** The model's radius distribution is essentially the same on a
2 mm and two 0.625 mm scans: p10 ≈ 1.0 mm, median ≈ 1.5–1.6 mm. If the model drew the
blurred footprint, the sharper scans would give thinner radii. Instead, at the same model
radius the sharper scans show much higher center contrast (0.34–0.43 vs 0.20). That is what
a floor independent of the scanner produces. So the floor near r ≈ 1 mm (about 1.4 voxels
of the 0.7 mm grid) looks like **the model's own**: its training labels, its grid or its
loss. It does not look like the acquisition's. The patients differ, though, so this is
suggestive, not controlled.

**What did not hold up: σ fitted against the model's boundary.** The edge fit measures
distance to the *model's* boundary, so σ_fit² ≈ σ_psf² + σ_model², and the model's boundary
scatter inflates it. Within one scan it gave 0.76 mm (arteries) and 1.09 mm (veins). That is
one scanner, so the gap is the fit, not the PSF. With σ = 1.09, C3N's vein "image radius"
even exceeds the model's above r ≈ 1.5 mm. Absolute image radii are uncertain by the σ
error. A PSF has to be measured from the image alone, or taken from the DICOM.

**Other findings.**
- The outside-body check in `measure.py` is invalid on these stores. Their grid is the lung
  crop plus 20 mm, so the trachea carries lung air to the crop border and everything aerated
  reads as "outside" (95 % of C3N's airways). It needs a body mask built on the full CT. The
  demo's 0.00 % was right only because its crop kept the lungs off the border.
- Layer 0 of both stores (the cascade's 118-class crop stage) is misnamed: its values 1–4
  carry `lung_*` names and the rest `label_N`. **Fixed upstream** in haversack `48dddec`
  (0.13.0): each layer is named from its own model. These two stores predate the fix and
  keep the wrong names until re-emitted. The analysis reads the fine part's layer only.
- MSB-02664's airway tree splits into pieces (largest 83.7 %), likely from the shallow
  inspiration. Not investigated.

The decisive test, the same patient thin versus thick, is §12.2.

### 12.2 The open items, executed (2026-09-24)

Scripts in `research/vessels/` (run with haversack's venv; data in `~/tmp/data/vessels/`):

| Script | What it does |
|---|---|
| `slab_average.py` | Builds 2 mm boxcar slab averages of the thin series. They reproduce a boxcar of the thin series' per-slice means to 0.1 HU, at zero shift. |
| `cases.py` | The two thickness ladders. |
| `thick_compare.py` | Model radius and center HU per variant at the thin run's centerline points. |
| `psf_image.py` | Image-only PSF, plus image radius with the PSF projected on each vessel's cross-section. |
| `centerline.py` | Seed-free centerline graph. |
| `vmtk_prep.py`, `vmtk_run.py` | The vmtk comparison, in an isolated `uv --with vmtk` env. |
| `straighten.py` | Sections with intervals and the straightened vessel. |

Seven more `ts.v2:lung_vessels` runs used plain `haversack segment` at 0.13.0 with rankfield
0.3.6: C3N at 1.25 mm LUNG, a 2 mm slab and 3.75 mm STANDARD; MSB at 0.625 mm again, 1.25,
2 mm slab and 5.0 mm DLIR. Each took 2–2.5 min at a 5.3–5.7 GB peak, with no memory
workaround.

**1. Same patient, thin vs thick.** The model's radius at the thin run's centerline points
(median mm; the share of points still inside the variant's vessel in parentheses), with the
center HU of each variant's own image:

| Patient, model r bin | 0.625 mm | 1.25 mm | 2 mm slab | thick (3.75 / 5.0) | center HU, thin → thick |
|---|---|---|---|---|---|
| C3N, < 1.25 | 1.15 | 1.08 (91 %) | 1.13 (96 %) | 1.02 (**26 %**) | −208 → −325 (2 mm) → −528 |
| C3N, 1.25–1.50 | 1.43 | 1.38 | 1.43 | 1.31 (70 %) | −1 → −132 → −407 |
| C3N, 1.50–1.75 | 1.63 | 1.61 | 1.64 | 1.59 (92 %) | 251 → 111 → −242 |
| MSB, < 1.25 | 1.09 | 1.13 (94 %) | 1.09 (86 %) | 1.25 (**17 %**) | −175 → −275 → −524 |
| MSB, 1.25–1.50 | 1.43 | 1.45 | 1.42 | 1.19 (38 %) | −28 → −136 → −457 |
| MSB, 1.50–1.75 | 1.64 | 1.64 | 1.63 | 1.43 (66 %) | 132 → 45 → −335 |

Above 2 mm every variant agrees within ~0.1 mm, and veins behave the same way. **The
model's thin-vessel radius does not follow the slice blur.** Up to 2 mm it is unchanged; at
3.75–5 mm the thin vessels disappear instead of fattening. The ~1 mm floor is the model's.

**2. The PSF from the image alone.** Each of 4,000 large-vessel wall profiles per patient
(reference radius > 4 mm, lung outside) is sampled with a cubic spline on the native grid.
Each gets its own free edge position, and `(σ_xy, σ_z)` are solved from
`σ_n² = σ_xy²(1 − n_z²) + σ_z² n_z²`. The image radius inverts the center contrast of a disc
under the PSF projected on the vessel's cross-section; that table is checked against Monte
Carlo and the isotropic closed form.

| Series | σ_xy / σ_z (mm) |
|---|---|
| C3N 0.625 / 1.25 LUNG / 2 mm slab / 3.75 | 0.95 / 1.04, 0.51 / 0.88 (edge-enhancing kernel, not Gaussian), 0.95 / 1.25, 0.81 / 2.19 |
| MSB 0.625 / 1.25 / 2 mm slab / 5.0 DLIR | 0.98 / 0.91, 0.97 / 0.96, 0.98 / 1.14, 0.97 / 2.67 |
| Demo (2 mm, kernel B) | 0.78 / 1.04 |

- **Calibration.** The slab series' σ_z² rises by **+0.475 / +0.478 mm²** against the known
  **+0.333**, identically in both patients. The estimator sees added blur and over-reads it
  by ~0.14 mm².
- **σ_z of the 0.625 mm series is ~0.9–1.0 mm,** far more than a 0.625 mm slice profile
  explains. Big hilar walls pulsate (no scan is gated) and are not steps, so these σ are
  best read as upper bounds.
- **Fixed on the way.** The demo NIfTI stores its axes permuted, so "the third index is the
  slice axis" was wrong for it. The slice axis is now the index axis most aligned with
  world S.

**3. The image radius against the model**, arteries, median mm (image radius is undefined
above r ≈ 2.5 mm, where the center contrast saturates):

| Model r bin | C3N model → image at 0.625 / 1.25 / 2 mm / 3.75 | MSB model → image at 0.625 / 1.25 / 2 mm / 5.0 | Demo model → image |
|---|---|---|---|
| < 1.25 | 1.09 → 1.00 / 0.97 / 0.92 / 0.74 | 1.02 → 1.11 / 1.04 / 1.05 / 0.88 | 1.05 → 0.62 |
| 1.25–1.50 | 1.39 → 1.26 / 1.48 / 1.16 / 0.95 | 1.40 → 1.34 / 1.26 / 1.27 / 1.08 | 1.40 → 0.75 |
| 1.50–1.75 | 1.60 → 1.65 / 1.80 / 1.53 / 1.25 | 1.61 → 1.66 / 1.57 / 1.59 / 1.36 | 1.60 → 1.01 |

- **Consistency.** With each image's own PSF, the image radius is stable across 0.625,
  1.25 and 2 mm on the same anatomy (MSB 1.11 / 1.04 / 1.05). It reads 15–25 % low at
  3.75–5 mm, and the LUNG kernel does not fit a Gaussian.
- **§12.1's "model 0.3–0.5 mm too large" is withdrawn for the angiograms.** With an image-only
  PSF, image and model agree within ~0.1 mm there. The earlier gap came from a σ too small
  (0.76–0.80), fitted against the model's boundary.
- **On the demo the gap remains** (0.62 vs 1.05). That scan differs in phase (nephrogenic),
  kernel and patient. With a model floor that ignores the acquisition, the likeliest reading
  is that the demo's thinnest vessels really are thinner than the floor the model draws them
  at. The σ uncertainty still covers much of it.
- **Absolute thin-vessel caliber stays unverified:** it scales with σ, and σ is an upper
  bound here.

**4. Centerline kernel and graph** (`centerline.py`). The pipeline:
- the unbounded distance from every interior voxel to the margin's zero crossings;
- Dijkstra on the 26-neighbor graph, cost = length × 1/(d + 0.1)²;
- seed-free branches (farthest uncovered voxel, cover radius 1.5 d + 1 mm, spurs shorter than
  2 R + 1 mm pruned);
- ridge-refined radius;
- branches split at junctions into a segment tree, checked to be a connected tree.

| | Segments | Tips | Junctions (degree > 3) | Centerline | Radius p10/50/90 |
|---|---|---|---|---|---|
| C3N arteries | 1,067 | 538 | 529 (7) | 10.5 m | 0.93 / 1.54 / 2.40 mm |
| C3N veins (main component) | 956 | 486 | 470 (11) | 9.4 m | 0.93 / 1.56 / 2.66 |
| MSB arteries | 596 | 301 | 295 (3) | 7.1 m | 0.94 / 1.64 / 4.23 |
| MSB veins (main component) | 470 | 239 | 231 (5) | 5.4 m | 0.91 / 1.62 / 3.46 |

Each tree takes 12–16 s on the M2, 7 s of it the decode. The pulmonary veins drain
separately into the left atrium, which the task does not segment, so "main component" is
not the whole venous tree.

**5. Against vmtk 1.5.2** (`vmtkCenterlines`, the real thing). The input is the C3N
left-lung subtree below one segment: 100 segments, 51 tips. Its surface is the field's own
zero set by marching cubes (42 k triangles, unsmoothed). My tips are the targets.

| Measure | Result |
|---|---|
| Routes | vmtk reached 50/51 targets. 49 followed the same route as ours (≥ 95 % of each line within 1 mm of the other); the two failures are vmtk's own "degenerate descent" |
| Coverage | 99.0 % of our subtree centerline lies within 1 mm of a vmtk line |
| Position | vmtk line → our polyline: median **0.086 mm**, p99 0.61, max 0.98 |
| Radius | vmtk maximal inscribed sphere − our ridge radius: median **+0.029 mm** (p10/p90 −0.013 / +0.071), the same in every radius class |
| Path length | ours / vmtk = 1.022 (the voxel path's residual wiggle) |
| Time | vmtk 4.3 s for 51 targets on 42 k triangles; ours ≈ 9 s for all 538 branches after the decode |

Two things found while building it:
- **A test error of mine.** A first run fed vmtk only the childless tips and reported "32 %
  uncovered". Every TEASAR branch runs to its own tip, parents included.
- **A defect of the voxel graph.** 26-connectivity links voxels that touch only at a corner.
  11 of 538 artery branches cross outside the field (m ≤ 0) for ≥ 0.2 mm, 10.9 mm in
  total. One of those cut the vmtk surface into two pieces. A path must be validated against
  the field (or the graph restricted to steps the interpolated margin keeps positive).

**6. Sections with intervals and the straightened vessel** (`straighten.py`; figures
`~/tmp/data/vessels/{C3N-00704,MSB-02664}_straightened.png`). The longest left-lung arterial
path runs 205–219 mm, from the trunk (r ≈ 14–19 mm) to a 0.7–0.8 mm branch. Along it:
- a spline centerline with local smoothing (deviation ≲ 0.15 r) and parallel-transport
  frames, with every station's center inside the vessel;
- sections every 0.5 mm, the margin sampled at 0.1 mm on the normal disk, taking the
  component that holds the center.

The ±2-logit interval of the area is 8 % for large vessels, 24–29 % at r 1.75–4 mm and
39–41 % at r 1.25–1.75 mm. In mm that is roughly ±0.2 mm of radius: the model's own
uncertainty band is narrower than the image method's σ doubt. The 2 mm model's area at the
same stations is within ±3 % of the thin model's. The straightened CT shows the margin's
zero contour on the bright lumen for the whole path.

**What this changes in the design.**
- The field gives **placement** (vmtk-grade, seed-free, whole tree) and **topology**.
- Thin-vessel **caliber** is a property of the input model: a ~1 mm floor, unverified below
  ~1.5 mm. It is shared with vmtk or any method fed the same segmentation, and it lowers with
  a higher-resolution segmenter, with no change to the method.
- Thick acquisitions **prune** the tree (again the model's behavior). Record the acquisition
  beside the model.
- The image-side sizing that vmtk's FWHM/active-tube tools would do needs a PSF measured on
  something other than pulsating hilar walls before it can arbitrate.

### 12.3 Is the field better than the voxels? (2026-09-24)

Two tests, chosen because each has an answer that does not depend on the method.

**Topology** (`field_topology.py`). Vessel trees are counted three ways:
- the labelmap with corner-connected voxels (26-connectivity);
- the labelmap with face-connected voxels only (6-connectivity);
- the field itself: the margin's trilinear interpolant sampled 2× finer, thresholded at 0,
  face-connected.

The answer key is anatomy: an arterial tree should be one piece with no loops.

| | corner-connected | face-connected | field |
|---|---|---|---|
| Arteries, loops (C3N / MSB / demo) | 6 / 4 / 2 | 7 / 3 / 2 | 7 / 4 / 2 |
| Arteries, pieces ≥ 20 voxels | 13 / 17 / 13 | 14 / 20 / 16 | 13 / 18 / 9 |
| Veins, loops | 2 / 1 / 1 | 3 / 8 / 1 | 4 / 4 / 1 |
| Veins, pieces ≥ 20 voxels | 19 / 44 / 22 | 19 / 47 / 25 | 16 / 40 / 23 |
| Artery + vein union, loops | 113 / 134 / 135 | 86 / 71 / 96 | 88 / 73 / 95 |

- **The field sits between the two conventions.** It is face-connectivity that does not break
  diagonal thin vessels: never more large pieces than face-only, and no corner merges.
- **For the class trees the differences are a few pieces or loops out of ~500 branches.**
  The arteries' 2–7 loops appear under every convention, so they are in the model's output,
  not the connectivity rule.
- **Loop counts are not converged.** A 3× supersample of one crop changed the arteries' loops
  from 5 to 2.
- **The one large effect is the artery+vein union,** where corner connectivity inflates
  contact loops by 25–85 %.
- **Verdict:** the field is no significant topological improvement over a sensible voxel
  convention.

**Measurement repeatability** (`reproducibility.py`). The runs at 0.625, 1.25 and 2 mm do not
change the model's radius on average (§12.2), and their model grids sit at different
sub-voxel offsets. Radius at the same world points, compared across runs, therefore measures
an estimator's own noise. Both estimators use the same ±0.5 mm search: the field's inscribed
radius (distance to its sub-voxel zero crossings) and the labelmap distance transform.
Mean |run − reference| / RMS in mm:

| | r < 1.25 | 1.25–1.5 | 1.5–1.75 | 2–2.5 | 2.5–4 |
|---|---|---|---|---|---|
| C3N arteries, 2 mm slab: field | 0.088 / 0.130 | 0.041 / 0.066 | 0.036 / 0.053 | 0.040 / 0.053 | 0.034 / 0.046 |
| C3N arteries, 2 mm slab: voxels | 0.118 / 0.181 | 0.088 / 0.142 | 0.079 / 0.129 | 0.078 / 0.122 | 0.093 / 0.139 |
| MSB arteries, 1.25 mm: field | 0.090 / 0.126 | 0.042 / 0.082 | 0.033 / 0.067 | 0.040 / 0.057 | 0.047 / 0.065 |
| MSB arteries, 1.25 mm: voxels | 0.102 / 0.166 | 0.080 / 0.146 | 0.063 / 0.124 | 0.078 / 0.124 | 0.087 / 0.131 |

Veins and the other variants read the same way.

- **From r ≈ 1.25 mm up, the field's radius is about 2× as repeatable** (1.7–2.7× in mean
  |Δ|, 1.7–3× in RMS) in every case, both patients, arteries and veins. The field's
  residual, 0.03–0.05 mm, includes the model's real run-to-run change, so its own noise is
  smaller still.
- **Below 1.25 mm the two estimators tie** (0.09–0.18 mm): there the model's own variation,
  not quantization, dominates.

**The hint.** The field's advantage is **measurement precision**, not topology. Voxel
quantization costs about 0.04 mm of radius noise at every vessel size above the thin floor,
and the sub-voxel zero set removes it. Structure (components, loops, which vessel connects to
which) comes out the same from a good voxel convention.

### 12.4 One mode: the field everywhere (2026-09-24)

Since the structure agrees and the field is used for everything else, the voxel steps were
replaced (`GRAPH=voxel` / `REF=skeleton` keep the old ones for comparison; outputs kept as
`*_voxel.*`).

**Field topology on the native grid** (`_topo.py`). The nodes are the labelmap's lattice
points: a point is inside exactly when its class wins, m > 0. Which neighbors are joined is
decided by the interpolant:
- face neighbors: joined when both are positive;
- diagonals across a face: joined by the asymptotic decider, positive face saddle
  (f_a f_b > f_c f_d), when neither of the face's other corners is positive;
- diagonals through a cell: closed under those rules, with the rare interior "tunnel" cases
  decided by sampling the trilinear cell (0–4 per case).

Loops are the genus of the zero surface from marching cubes with the Lewiner topological case
table. Cost: the same order as the voxel graph (7.2 vs 6.3 s to the graph on C3N arteries),
no supersampling.

**Verified two ways.**
- On all three scans (arteries, veins, union), every marching-cubes surface anchors in exactly
  one graph component and every component has a surface. The surplus surfaces are small
  enclosed cavities in the union (1 / 6 / 3).
- On a crop, native field topology equals the 3× supersample for the arteries (24 pieces,
  2 loops) and is within 2 pieces and 1 loop elsewhere, the supersample still converging.

Two bugs in the check itself were found and fixed on the way:
- mapping surfaces to components by nearest vertex, which crossed into neighboring blobs;
- a floor-cell anchor that failed for vertices sitting on an exact tie (m = 0).

Native field numbers (C3N / MSB / demo):

| | Loops (genus) | Pieces | Pieces ≥ 20 voxels |
|---|---|---|---|
| Arteries | 4 / 3 / 1 | 41 / 52 / 71 | 13 / 19 / 15 |
| Veins | 2 / 5 / 1 | 55 / 98 / 66 | 19 / 45 / 25 |
| Artery + vein union | 86 / 78 / 89 | | |

**The centerline graph on field connectivity** (`centerline.py`, `GRAPH=field`).
- The tree is nearly the same as before: C3N arteries 537 branches, 1,065 segments, 10.52 m
  (voxel mode: 538 / 1,067 / 10.53 m).
- A join the field makes can curve around a face saddle, so a straight chord between joined
  points can still clip the outside. Chords that leave the vessel get their midpoint moved to
  the local ridge, repeated.
- Result (`leave_field.py`), over the four trees: **43 segments / 38 mm outside in voxel mode
  → 12 / 6.7 mm in field mode**, out of 32 m, longest excursion 1.40 → 0.70 mm. The residue
  sits at pinch points narrower than the 0.35 mm repair search.

**Everything downstream rerun on field-graph reference points** (`REF=graph`):
- thin vs thick, image PSF, image radius, repeatability and straightening reproduce §12.2 and
  §12.3 to within bin noise (e.g. C3N r < 1.25 mm at 0.625 / 2 / 3.75 mm: 1.10 / 1.08 / 1.00,
  26 % found; image-only σ 0.95 / 1.05 mm; field vs voxel repeatability 0.036 vs 0.080 mm at
  r 1.5–1.75);
- vmtk improves where structure matters: **51/51 targets reached, 50 on the same route**
  (the one exception is a vmtk line that ends after 0.8 mm), **99.8 %** of our subtree within
  1 mm of a vmtk line (voxel mode 50/51, 49, 99.0 %); position 0.086 mm and radius +0.029 mm
  unchanged.

**Result:** one mode, the field, with the grid only as its sampling. Structure is now exact
for the interpolant and independently verified; measurements were already the field's. It
costs the same as the voxel pipeline.

### 12.5 Straightened 3D rendering from the field (2026-09-24)

`explorations/rendering/render_straight.py` renders a straightened vessel without resampling any volume:
- Rays run straight through a display box (s along the path, u and v across it).
- Each sample maps to world as C(s) + u·n₁(s) + v·n₂(s) and reads the artery and vein margins
  there.
- The surface is the first hit, refined by bisection. Normals are central differences in
  display space, through the same map. Box-face hits are shaded as flat caps.

This is the per-sample display-to-world warp a click-to-straighten viewer needs, run offline
(17–45 s on the M2 in Python; a shader would be interactive).

Example: C3N-00704's longest left-lung artery, 145 mm from a 6 mm lobar artery to 0.8 mm,
at ±8 and ±15 mm of the axis.
- **Folding:** the script reports where the box is wider than the local bend radius. At
  ±15 mm that happens at 1.2 % of stations, with nothing visibly duplicated.
- **Illustration style** (`STYLE=npr`) follows sdfview's: Gooch warm-cool tone, wrapped
  diffuse, a broad faint highlight, colored depth-break ink with the thin-feature rule, and a
  depth cue, on white.
- **Smoothing is allowed in illustration.** The surface comes from the field smoothed
  σ = 0.3 mm, the normals from σ = 0.8 mm over a ~voxel stencil. A shorter stencil striped the
  surfaces, where the trilinear gradient jumps at cell faces. Rendering is 2× supersampled so
  the ink is antialiased.
- **Graph clipping** (`STUB=L`: the path plus the first L mm of each branch) is prototyped but
  has a known defect: the keep field is set on vessel voxels only, so its staircase replaces
  the wall. It was set aside in favor of showing both vessel classes.

Figures: `~/tmp/data/vessels/C3N-00704_straight3d_w{8,15}*.png`.

### 12.6 vmtk's branch partition and bifurcation frames without a surface (2026-09-24)

`vmtk_branch.py` (vmtk 1.5.2, isolated env) and `branch_partition.py`, on the C3N-00704
left-lung subtree (51 tips, surface from the field's zero set).

**Two kinds of vmtk stage.**
- *Centerline-only* stages (branch extraction, bifurcation reference systems, centerline
  geometry and attributes) work on polylines and never touch a surface.
- *Surface* stages (branch clipper, mapping, sections, meshing) are the ones the field replaces.

**vmtk's branch extractor runs on our centerlines unchanged**, in vmtk's convention (one polyline
per target, source to target, `MaximumInscribedSphereRadius`): 95 branch groups and 44
bifurcations, against 94 and 43 from vmtk's own centerlines. That is the "new front end" at the
centerline level.

**The branch clipper is a point-wise formula.** It labels a surface point with the non-bifurcation
group whose tube function is lowest (`vtkvmtkPolyDataCenterlineGroupsClipper`; tube function from
`vtkvmtkPolyBallLine`: closest point in 4-D (x, r) with Minkowski dot, t clamped to [0, 1], value
|x − c(t)|² − r(t)²). Ported to torch and evaluated at the clipper's own output points:
- at the **original surface vertices**, identical to vmtk's labels: **99.995 %** (1 of 21,225)
  with vmtk's centerlines and **100.000 %** with ours;
- the remaining points are the ~7,000 the clipper creates on its cuts. They are exact ties
  (~6,000 written into both neighboring groups), so they have no single correct label;
- speed: **2–4 s against 183–217 s** for vmtkBranchClipper;
- the same function labels the **volume**: 16,601 lattice points inside the subtree in 1.6 s,
  95 groups, a label field instead of surface patches.

**Our centerlines against vmtk's**, both through vmtk's splitting:
- 95 of our groups map onto 94 of vmtk's;
- the wall points get the same group **95.6 %** of the time. Differences sit near bifurcations
  (median 2.3 mm from a bifurcation origin, against 4.8 mm for agreeing points), where each set
  places the branch boundary slightly differently;
- **bifurcation frames**: 43 of our 44 match vmtk's 43 within 5 mm. Origins are a median
  **0.58 mm** apart (p90 0.95), normals differ by 5.2°, up-normals by 6.3°.

**Result:** vmtk's branch partition and bifurcation frames need no surface. The partition is
exact as a field function, 50–100× faster, and volumetric; the frames come from centerlines,
which ours supply. Next in this line: (s, θ) wall maps (branch mapping), level-set curvature and
implicit flow extensions (§12.7).

### 12.7 Wall maps, wall curvature and flow extensions without a surface (2026-09-24)

The rest of vmtk's surface stages that a CFD or morphology study uses, done from the field and
checked against vmtk on the same subtree and centerlines as §12.6. Scripts: `vmtk_mapping.py`,
`wall_map.py`, `curvature.py` + `vmtk_curvature.py`, `flow_ext.py` + `vmtk_flowext.py`.

**Branch mapping: vmtk's wall coordinates are point-wise too.** vmtk's surface chain (clipper →
bifurcation frames → offset attributes → `vmtkBranchMetrics` → `vmtkDistanceToCenterlines` →
`vmtkBranchMapping` → `vmtkBranchPatching`) ran on our centerlines in 189 s, 183 s of it the
clipper. Its per-vertex abscissa and angle
(`vtkvmtkPolyDataCenterline{Abscissa,Angular}MetricFilter`) are formulas: per centerline cell of
the point's group, a closest point on the polyline (with the settings `vmtkbranchmetrics.py`
forces: Euclidean, `UseRadiusInformationOff`; bifurcation cells included for the abscissa and
excluded for the angle), then r²-weighted averages of point, tangent, normal and abscissa.
Ported to numpy and evaluated at vmtk's surface vertices: **abscissa identical (p99 1e-14 mm),
angle identical (p99 1e-8°)**, 1 s for 21,221 vertices.

**A vmtk defect found on the way.** `vtkvmtkCenterlineUtilities::InterpolateTuple` fetches both
end values of a segment with `vtkDataArray::GetTuple(id)`, which returns a pointer into one
internal buffer, so both pointers see the second value, and every "interpolated" radius, abscissa
and normal is the segment's **end-point** value. Bit-exact agreement above requires reproducing
that. Against a true interpolation, vmtk's surface abscissa reads a median **0.114 mm**
downstream (up to one 0.3 mm centerline step), and its angle is off by up to 1.9° (p99), from the
normals. Small at vmtk's usual sampling, but it is a real error, and it scales with the
centerline step.

**Wall maps by ray casting.** For each branch group, rays leave the centerline at every station
and every 5°, in vmtk's angular convention (cos φ N + sin φ (N × T), N the parallel-transport
normal), and stop at the margin's first sub-voxel zero crossing:
- 92 groups, **178,056 rays in 1.8 s**; r(s, φ) is vmtk's patched map, made without a surface
  (vmtk's own `vmtkBranchPatching` returned an empty image on this subtree);
- **the ray frame comes from the station line smoothed over 0.6 mm of arc length.** The raw
  polyline turns a median 5° per 0.3 mm station, and up to 161° where near-duplicate points
  leave the tangent undefined. Each kink tilts one station's ray plane, and its rays hit the
  wall obliquely and read long: single-station streaks, with streak strength tracking the turn
  (ρ 0.54). Smoothing brings the worst turn to 18° and the p99 streak strength from 0.102 to
  0.037 mm; above 1 mm it starts cutting corners on real bends. The stations keep vmtk's
  abscissa;
- against vmtk's `DistanceToCenterlines` at the same (group, abscissa, angle): median
  **−0.009 mm**, |diff| median 0.025 mm, p90 0.084 mm (0.020 / 0.081 from the raw polyline,
  which is what vmtk measures from);
- rays that run past 1.8× the station's median radius, or never exit, are **ostia**: 0–2.3 % of
  a branch's map, the side branches' footprints on the wall. Any wall quantity (CT, the margin
  interval, curvature) maps the same way.

**Wall curvature from the level set.** Mean curvature H = (k₁ + k₂)/2 of the zero set,
H = −½ div(∇m/|∇m|), vtkCurvatures' sign (a tube of radius r has 1/(2r)). The margin is steep
(10.6 logit/mm) and clipped at ±8, so it saturates within ~0.75 mm of the wall. Central
differences reach that plateau and read the curvature low. The estimator that works is a
**Gaussian-weighted quadratic fit to the unclipped samples within 2.8 mm**, taking the curvature
of the fitted quadric's level set.

| | Phantoms: estimate / truth (tubes r 0.75–3 mm, spheres) | Repeatability: median \|ΔH\| between reconstructions of one scan (lung125, slab2mm) | ρ between reconstructions |
|---|---|---|---|
| Field, quadric fit (2.8 mm) | **1.02–1.04** (p10–p90 within ±5 %) | **10 %, 7 %** of H | **0.88, 0.93** |
| Field, central differences | 0.80–0.91 | 13 %, 11 % | 0.84, 0.87 |
| vmtk (`vmtkSurfaceCurvature`, mean, on the same marching-cubes mesh) | 1.04–1.41 (p10–p90 0.6–2.6) | 59 %, 57 % | 0.37, 0.42 |

On the real subtree the quadric fit also tracks the tube proxy 1/(2 r_wall) best (ρ 0.52, against
0.17 for vmtk). It takes 0.7 s for 11,454 wall points. The phantoms are on the real grid's
spacing (1.0 × 0.7 × 0.7 mm), oblique to it, with the real wall slope and clip. vmtk's figures
are on the marching-cubes mesh with default settings; vmtk users often smooth first, which
trades noise for shrinkage.

**Flow extensions as implicit geometry.** vmtk needs an open surface: clip each outlet, extract
the boundary ring, then extrude a cylinder along the ring normal. The cylinder's radius is the
ring's mean distance from its barycenter, its length ExtensionRatio × that radius, and the ring
morphs into the circle over TransitionRatio of the length. The field version is a signed
distance (mm), closed-form and evaluable anywhere:
- **cut:** past each outlet plane, inside a ball of 1.6 r, the field becomes min(φ, −s). The cut
  vessel falls into pieces; keep the main component. vmtk takes the same connectivity step.
- **extension k:** with s along the normal and (u, v) in the plane, w = clamp(s/(TL), 0, 1) and
  φₖ = (1 − w) φ_sec(u, v) + w (R − |(u, v) − b|), capped at s = L. φ_sec is the outlet
  section's own 2-D signed distance, from the field's sub-pixel contour on the cut plane.
- **domain:** max(cut vessel, φ₁, …, φ_K).

On 9 cuts (the inlet plus the 8 widest terminal branches, ExtensionRatio 5, TransitionRatio 0.25):
- **rings:** barycenters a median 0.067 mm from vmtk's, mean radius 0.020 mm smaller. vmtk
  averages over mesh ring vertices, we average over arc length;
- **vmtk's extension surface against the field domain:** **0.022 mm** in the transition and
  **0.032 mm** in the cylinder (|diff| median; p90 0.048 and 0.054 mm). The transition agrees
  although vmtk morphs with a thin-plate spline and the field blends signed distances;
- **one closed domain:** on a 0.3 mm lattice (25 M points, 2.2 s), 12 distal remnants dropped, one
  component, every edge shared by two triangles, and the same Euler characteristic as the uncut
  surface (0: the subtree's one loop survives). The ends are flat caps for boundary conditions,
  where vmtk leaves the tubes open;
- the field domain is what an immersed-boundary, cut-cell or lattice-Boltzmann solver takes
  directly; a body-fitted mesher takes its zero set.

**Result.** Of vmtk's surface stages, branch partition (§12.6), branch mapping, wall curvature and
flow extensions now run from the field. vmtk's own coordinates are reproduced bit for bit, and
curvature is 6× more repeatable than vmtk's. What still needs a mesh is body-fitted volume
meshing (TetGen, boundary layers) for a CFD solver that requires one; export is the one place a
surface is made.

**Speed, stage by stage** (`bench_stages.py`; one session on the M2, load average 2.3–4.0 from
other work, compute only, excluding imports and file loading; our side measured twice with
agreement within 5 %, vmtk once; both start from the same decoded field, 4.9 s):

| Stage | vmtk | Field | |
|---|---|---|---|
| Surface for vmtk (marching cubes) | 0.02 s | not needed | |
| Centerlines | 3.7 s (subtree, 51 seeded targets) | 9.0 s (whole tree, 537 branches, no seeds) | different scopes: ours always traces the whole tree; vmtk on the whole tree (243k vertices, 537 targets) was started and stopped as too long to wait for |
| Branch extraction, frames, offsets | 3–4 s; < 0.01 s | vmtk's own, run on our centerlines | |
| Branch clipper (surface labels) | **185–228 s** | **2.2 s** | ~90× |
| Partition of the volume | — | 1.6 s | |
| Wall coordinates (abscissa, angle) | 0.87 s | 0.55 s | |
| Distance to centerlines / wall maps | 3.2 s (per vertex) | 1.8 s (the full map, 178k rays) | |
| Harmonic mapping, patching | 1.6 s; 0.09 s (empty image) | not ported (arc-length map) | |
| Curvature | **0.05 s** | 0.67 s | vmtk faster, 6× noisier |
| Flow extensions | 0.1 s (clip + extend, open surface) | 0.3 s (sections); 0.003 s to evaluate at points | |
| The domain on a 0.3 mm lattice + mesh | — | 2.2 s + 0.2 s | was 26–28 s before restricting each piece to where it acts |

The subtree analysis is **~240 s in vmtk, 95 % of it the clipper**, against **~5.5 s** from the
field. Without the clipper vmtk is ~13 s, so the gap comes from one stage. The lattice step
first evaluated every cut and extension over the whole 25 M-point box (17 s for the extensions,
8.5 s for the cuts). It now evaluates each piece only where it can change the result: the
vessel's interpolant where a field-cell corner is above the clip floor (2.5 % of the box;
elsewhere the value is the floor exactly), each cut inside its ball, each extension inside its
own box. The result is identical (float32 rounding, the same mesh) at 2.2 s.

## 13. vmtk and the field-based method, compared (2026-09-24)

**vmtk** starts from an image, has a person segment a lumen with seeds and level sets, and
turns it into a **triangulated surface** (marching cubes, then smoothing and remeshing).
Everything afterwards is computed from that surface:
- centerlines are minimal paths over the surface's Voronoi diagram, with radius as the maximal
  inscribed sphere;
- branches are split by overlapping those spheres;
- coordinates come from harmonic maps;
- CFD meshes are built with TetGen.

It is mature, validated and widely used, especially for CFD.

**The field-based method** starts from a segmentation model's **continuous field**: the
per-class margins in a ranked store, which locate surfaces between voxels. Nothing is
converted to a surface for analysis:
- **connectivity** is decided by the field's interpolated values on the original grid, and
  loops by the genus of its zero surface;
- **centerlines** are seed-free minimal paths over that field-connected graph, weighted by an
  unlimited-range distance to the field's zero crossings, then refined onto the ridge;
- **radius, cross-sections and position** are read from the field;
- **surfaces** are only a view, extracted or rendered when needed.

The method works on any grid and any model. Its geometric accuracy is bounded by the model
it is given, as vmtk's is bounded by the segmentation it is given.

| | vmtk | Field-based method |
|---|---|---|
| Input | An image; segmentation is part of the workflow | A model's output field (here TotalSegmentator `lung_vessels` via haversack) |
| Segmentation | Interactive: seeds, level sets, vesselness | Automatic, one model run, no seeds |
| Naming | One unnamed lumen | Named classes; artery and vein separate |
| Core representation | Triangle surface | Continuous field; the grid is only its sampling |
| Surface quality | Staircased by marching cubes; smoothing shrinks thin vessels | Located between voxels by the field; no smoothing for analysis |
| Centerlines | Voronoi diagram of the surface, seeded source and targets | Minimal paths on the field graph; branches without seeds |
| Centerline agreement (§12.4) | — | vmtk a median **0.086 mm** from ours; 50 of 51 routes identical |
| Radius | Maximal inscribed sphere on the Voronoi diagram | Ridge-refined distance to the zero crossings; vmtk reads **+0.029 mm** larger (median) |
| Radius repeatability (§12.3) | Not tested (depends on the surface given) | About **2×** a voxel-based radius (0.03–0.05 vs 0.06–0.10 mm) across reconstructions of one scan |
| Topology (§12.3–12.4) | Inferred from one surface; artery–vein contacts become false loops | Exact for the field, verified two ways; artery–vein contacts stay separate (73–95 false loops avoided per tree pair) |
| Robustness (§12.4) | 1–2 of 51 targets fail ("degenerate descent") | Whole tree traced; 12 of ~3,000 segments briefly leave the vessel, 6.7 mm in 32 m |
| Speed (M2) | 4.3 s for a 51-target subtree | ~9 s for a whole 538-branch tree, after a 7 s decode |
| Uncertainty | None; one number per measurement | The model's own interval (±2 logits ≈ ±0.2 mm of radius on small vessels) |
| Branch splitting, frames, wall maps (§12.6–12.7) | Mature: ball-overlap splitting, bifurcation frames, patch mapping; all but the frames run on the surface | vmtk's splitting and frames run on our centerlines; its partition and wall coordinates reproduced bit for bit as field formulas, 50–100× faster, volumetric; wall maps by ray casting (1.8 s, 0.02–0.03 mm from vmtk's) |
| Wall curvature (§12.7) | Mesh curvature (vtkCurvatures), noisy on marching-cubes surfaces | Level-set curvature by a local quadric fit: 1.02–1.04 × truth on phantoms, **6×** vmtk's repeatability across reconstructions |
| CFD | Mature: flow extensions, boundary layers, TetGen, wall shear stress | Flow extensions as implicit geometry: one watertight domain, 0.02–0.03 mm from vmtk's (§12.7); body-fitted meshing still vmtk's (or any mesher, from the zero set) |
| Visualization | VTK surfaces; curved planar reformation straightens pixels on a 2-D cut | The field rendered directly, including straightened 3D views of any path (§12.5) |
| Maturity | Two decades, papers, Slicer integration, a community | Prototype scripts, two patients plus a demo, vmtk comparisons on one subtree |

**Shared limit: the segmentation model (not a difference between the methods).** Both
methods measure what the segmentation gives them. Fed the same field surface, vmtk's radii
matched ours to 0.03 mm. Two properties of today's input (TotalSegmentator `lung_vessels`,
0.7 mm grid) therefore apply to either method:
- **A radius floor near 1 mm.** It sits at about 1.4 of the model's voxels and doesn't move
  with the scan: blurring the same anatomy from 0.625 mm to 2 mm slices left it unchanged. It
  comes from the model's training labels and grid, which sit at roughly clinical-CT
  resolution, where the scanner's own blur (~2 mm full width) is already wider than a 1 mm
  vessel.
- **Thin vessels lost at thick slices.** At 3.75–5 mm slices the model drops them rather than
  widening them.

Both move with the input. A segmenter trained on higher-resolution data (photon-counting or
ultra-high-resolution CT, micro-CT for specimens) lowers the floor roughly in proportion, to
~0.35 mm on a 0.25 mm grid. The field method needs no change for that; it works in mm on any
grid. Below the segmenter's floor, caliber survives in how bright a vessel appears, and with
the scanner's blur known from its specifications or a phantom, image-side sizing (the field
places the tube, the image sizes it) can go further on today's scans.

**In short.** The field-based method replaces vmtk's **front half** (segmentation, surface
repair, seeded centerlines) with something automatic, named and more precise. Where both
compute the same centerline and radius, they agree to ~0.1 mm and 0.03 mm. Most of vmtk's
**back half** also runs from the field without a surface (§12.6–12.7): branch partition,
bifurcation frames, wall coordinates and maps, wall curvature and flow extensions, reproducing
vmtk where vmtk defines the quantity (bit for bit for the coordinates). Body-fitted CFD meshing
(TetGen, boundary layers) stays vmtk's, fed the field's zero set; export is the one place a
surface is made. Resolution is set by the segmentation model, the same for both, and it improves
with the model without changes to the method.

## 14. SlicerHeart: where the method fits (2026-09-24)

Full write-up: [`slicerheart-opportunities.md`](slicerheart-opportunities.md).

SlicerHeart's vessel handling is traditional. `CardiacDeviceSimulator` casts device handles
along rays from a user-drawn centerline to a smoothed closed surface of the lumen segment
(`deformHandlesToVesselWalls`). `PDAQuantification` requires SlicerVMTK's centerline
extraction. That chain is where the field plugs in: seed-free centerlines, sub-voxel walls,
compression with an interval, and clearance to named neighbors (coronaries, aortic root).

Pilots, in order:
1. field lumen and coronary clearance in the device simulator;
2. graph-driven virtual injection and best C-arm angle in the Virtual Cath Lab;
3. a straightened planning view with the device in place.

Caveats: congenital anatomy needs specialized models; SlicerHeart's valve work is 3D echo, with
no echo model in this pipeline yet; coronary ostia cross model boundaries; and the input models'
~1 mm caliber floor limits r⁻⁴ uses, not device sizing.

## References

- Antiga L, Steinman DA. Robust and objective decomposition and mapping of bifurcating
  vessels. *IEEE TMI* 2004.
- Antiga L, Piccinelli M, Botti L, Ene-Iordache B, Remuzzi A, Steinman DA. An image-based
  modeling framework for patient-specific computational hemodynamics. *Med Biol Eng Comput*
  2008.
- Piccinelli M, Veneziani A, Steinman DA, Remuzzi A, Antiga L. A framework for geometric
  analysis of vascular structures: application to cerebral aneurysms. *IEEE TMI* 2009.
- Izzo R, Steinman D, Manini S, Antiga L. The Vascular Modeling Toolkit: a Python library for
  the analysis of tubular structures in medical images. *JOSS* 2018.
- Kirchhoff Y, et al. Skeleton Recall Loss for connectivity conserving and resource efficient
  segmentation of thin tubular structures. *ECCV* 2024. (The trainer behind TS
  `lung_vessels` and `coronary_arteries`.)
- vmtk releases v1.5.0–v1.5.2 (2026-07 … 2026-09-18), github.com/vmtk/vmtk; SlicerVMTK,
  github.com/vmtk/SlicerExtension-VMTK.
- This workspace: `haversack/docs/ranked-{composition,measurement,probabilities,reconstruction}.md`,
  `haversack/docs/ranked-distance-gpu.md`, `rankfield/docs/{overview,format}.md`,
  `medseg/docs/translucency-by-crossings.md`; memory notes on the distance/junction layers, margin
  rendering, surface relaxation and the liver atlas.
