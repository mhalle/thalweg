# The field-based vessel method and SlicerHeart: where they fit together

*2026-09-24. Source reviewed: [SlicerHeart](https://github.com/SlicerHeart/SlicerHeart) `master`
(32 scripted modules, README and module code as of 2026-09-24). The field-based method is described
in [`vmtk-successor.md`](vmtk-successor.md) and summarized against vmtk in
[`vmtk-vs-field-method.md`](vmtk-vs-field-method.md). Nothing here is built.*

## SlicerHeart in brief

SlicerHeart (Lasso et al., *Front. Cardiovasc. Med.* 2022) is a 3D Slicer extension for cardiac
image import, quantification, surgical planning and implant placement. It comes out of pediatric
and congenital cardiology (CHOP, PerkLab). Its strengths:

- **Device and procedure planning:** Cardiac Device Simulator (Harmony pulmonary valve, cylinder
  devices), ASD/VSD, TCAV valve and ValveClip simulators, the Virtual Cath Lab (simulated C-arm and
  fluoroscopy from CT or 4D CT, with devices and virtual contrast), and the Baffle Planner.
- **Valve modeling from 3D and 4D echo:** annulus, leaflet and papillary analysis, valve
  quantification, batch export and annulus shape statistics.
- **Simulation export:** leaflet and chordae FEM models for FEBio, and meshes for svFSI and
  SimVascular with unit and coordinate conversion.
- **Import:** vendor 3D/4D ultrasound, cine MRI reconstruction, Mimics projects and
  electroanatomic maps (Carto, NavX, Rhythmia).

**Its vessel handling is traditional.**

- `PDAQuantification` requires SlicerVMTK's centerline extraction on a segmented vasculature.
- `CardiacDeviceSimulator` takes a centerline curve the user draws and a vessel-lumen segment. The
  segment becomes Slicer's closed-surface representation, which is smoothed. Each device handle is
  then pushed out along a ray from the centerline until it meets that surface
  (`deformHandlesToVesselWalls`, via a `vtkModifiedBSPTree` line intersection), and compression
  metrics follow from the deformed handles.
- `TCAVValveSimulator` builds its centerline from an aortic annulus.

That chain (segment, smoothed surface, ray cast) is where the field plugs in directly.

## What the field-based method brings

Measured on two thin-slice CT angiograms and a demo scan (evidence in `vmtk-successor.md` §12–13):

- **Seed-free, named centerlines.** A branch graph for a whole lung arterial tree (538 branches)
  in about 9 s after decoding. Artery and vein come from one model and stay separate at contacts.
- **Sub-voxel walls.** The wall is where the field crosses zero between voxels, with no smoothing.
  Radius is about **2× as repeatable** as a voxel-based radius across reconstructions of one scan,
  and agrees with vmtk to **0.03 mm** on the same surface.
- **Intervals.** The model's own uncertainty (±2 logits ≈ ±0.2 mm of radius on small vessels) turns
  every measurement into a range.
- **Neighbors.** Every structure the models name is available, with distances in mm.
- **Straightened views** rendered directly from the field, with no resampled volume, including an
  illustration style.

## Module by module

| SlicerHeart module | What it does now | What the field adds | Models available today |
|---|---|---|---|
| **CardiacDeviceSimulator** (Harmony pulmonary valve, cylinder devices), **TCAVValveSimulator** | User-drawn centerline; device handles cast to a smoothed lumen surface; compression map | Seed-free centerline and radius; handles land where the field crosses zero along the ray (sub-voxel, no smoothing shrinkage, ~2× more repeatable); **compression with an interval** | `heartchambers_highres` (aorta, pulmonary artery); `total` cardiac part (great vessels in one model) |
| Same, **clearance** | None | **What lies outside the wall, in mm**: distance from the device envelope to the coronaries or the aortic root. Coronary compression is the classic contraindication for pulmonary valve implants | `coronary_arteries` (0.7 mm), `aortic_sinuses`, `aorta` |
| **PDAQuantification** | vmtk centerline on a segmented vasculature; duct length and diameters | Field centerline and radius, and the **narrowest cross-section with an interval**, the number that sizes a device | Needs a model that labels the duct (TotalSegmentator has no such class); a labelmap-only input degrades gracefully |
| **Virtual Cath Lab** | Simulated fluoroscopy from CT; virtual contrast from a segmentation | **Anti-aliased contrast from the field**: each ray's projected thickness computed analytically, with no staircase. **Virtual injection**: click a point and fill only the downstream subtree, using the graph. **Best C-arm angle** per segment: least foreshortening and least overlap with *named* vessels | `lung_vessels`, `coronary_arteries`, `total` |
| **FluoroFlowCalculator** (split of flow between the lungs from fluoroscopy) | Measures the split from fluoroscopy sequences | A CT-side estimate to compare against: per-lung arterial tree morphometry and a 1-D/0-D resistance model from the graph. Resistance scales as r⁻⁴, so this is a research use until small-vessel caliber is verified | `lung_vessels` (arteries and veins separate) |
| **ImportExportSimulationModel**, **ValveFemExport** (svFSI, SimVascular, FEBio) | Converts meshes, units and coordinate systems | Inputs those solvers want: surfaces from the field's zero set, a radius-aware sizing function, **inlets and outlets named from the graph**, and a 1-D model for SimVascular's reduced-order solver | Any |
| **4D sequences** (4D CT in the Virtual Cath Lab, cine MRI) | Frame-wise visualization | **Distensibility** (area change over the cardiac cycle along a vessel chart). Changes are 5–15 %, so the field's 2× repeatability and its intervals matter | Needs the model run on each frame |
| **AnnulusShapeAnalyzer**, **ValveBatchExport** (population statistics) | Cohorts of annulus shapes | The same idea for vessels: seed-free cohort runs, with measurements on a per-vessel coordinate chart (arc length, angle) aligned at named branch points | IDC cohorts, `total` |
| **Visualization** | Surface models, echo volume rendering | **Straightened vessel views with the device rendered in place.** The device simulator already works in a centerline-aligned frame, and the straightened renderer is that same frame | — |
| **LeafletAnalysis**, **ValveFemExport** (thin structures) | Leaflet surfaces from 3D echo; FEM needs a mid-surface and a thickness | **Thin-sheet geometry**: sub-voxel surfaces, and thickness from the boundary on each side (the airway-wall demo measured wall thickness, 79 % of it below one voxel) | No valve-leaflet model feeds this yet; a future engine |

## Pilots, in order

1. **Field-based lumen for the device simulator, with coronary clearance.** It is the smallest
   change with the clearest clinical value. Replace the closed-surface ray cast with the field's
   zero crossing, add a compression interval, and add a clearance measurement to the coronaries and
   the aortic root. The centerline comes from the graph, the lumen and neighbors from
   `heartchambers_highres` + `coronary_arteries`. A cardiac CT from IDC is enough to start.
2. **Virtual Cath Lab: injection and best angle.** Graph-driven downstream filling, field-based
   projected thickness, and a best-angle search that avoids named overlapping vessels. These
   capabilities depend on named topology, which a single-lumen tool cannot provide.
3. **Straightened planning view.** The device along the straightened vessel with its neighbors:
   today's renderer plus a device model. It reads more directly than a 3-D surface scene for
   judging clearance.

## Caveats

- **Anatomy.** SlicerHeart's core users work on congenital and pediatric hearts (PDA, septal
  defects, the double-outlet right ventricle cases the Baffle Planner targets). TotalSegmentator is
  trained on adult anatomy, so a PDA or a Fontan circulation may be mislabeled or missed. The field
  method takes any model's output, so this is a question of models; haversack runs any nnU-Net,
  which leaves room for specialized congenital models.
- **Echo.** SlicerHeart's valve work is 3D and 4D echo; everything here has been CT. The approach
  applies to any model that produces class probabilities, but there is no echo model in this
  pipeline yet.
- **Cross-model boundaries.** Coronary ostia on the aorta come from different models
  (`coronary_arteries` vs `heartchambers_highres`), so they can only be located geometrically, not
  read off a shared model boundary.
- **Caliber.** The input models' radius floor near 1 mm limits anything that depends on radius to
  the fourth power (resistance models, small-vessel flow). It does not limit device sizing, which
  works at 5–15 mm radii. The floor belongs to the segmentation model, not the method, and lowers
  with a higher-resolution model.

## Integration path

Slicer consumes markups curves, segmentations and volumes, and the method's outputs map onto them
directly:

- **Centerline graph → `vtkMRMLMarkupsCurveNode`s**, per-point radius and interval as curve
  measurement arrays. These are the curve nodes `CardiacDeviceSimulator` and `PDAQuantification`
  already take.
- **Field → scalar volumes** (margins, or a signed distance in mm) for volume rendering and
  thresholding.
- **Surfaces → models**, extracted from the zero set.
- **Transport:** the planned haversack Slicer client (`medseg/docs/slicer-modal-design.md`). The server
  runs the model and returns the ranked store; the client decodes it.

## References

- Lasso A, Herz C, Nam H, et al. SlicerHeart: An open-source computing platform for cardiac image
  analysis and modeling. *Front. Cardiovasc. Med.* 2022;9:886549.
  doi:10.3389/fcvm.2022.886549
- Barak-Corren Y, Daemer M, Gupta M, et al. Virtual Cath Lab: versatile open-source simulator for
  education and procedural planning in congenital heart interventions. *JSCAI* 2025;4(11):103937.
  doi:10.1016/j.jscai.2025.103937
- Nam HH, et al. Simulation of transcatheter atrial and ventricular septal defect device closure
  within three-dimensional echocardiography-derived heart models on screen and in virtual reality.
  *J Am Soc Echocardiogr* 2020. doi:10.1016/j.echo.2020.01.011
- Vigil C, et al. Modeling tool for rapid virtual planning of the intracardiac baffle in
  double-outlet right ventricle. *Ann Thorac Surg* 2021. doi:10.1016/j.athoracsur.2021.02.058
- SlicerHeart source: `CardiacDeviceSimulator/CardiacDeviceSimulator.py`
  (`deformHandlesToVesselWalls`, `processVesselSegment`), `PDAQuantification/PDAQuantification.py`
  (SlicerVMTK `extractcenterline` dependency), `TCAVValveSimulator/TCAVValveSimulator.py`.
