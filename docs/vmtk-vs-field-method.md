# vmtk and the field-based method, compared

*2026-09-24. A summary of the evidence in [`vmtk-successor.md`](vmtk-successor.md) §12–13. The
test data are two thin-slice CT angiograms from the NCI Imaging Data Commons (cptac_luad
C3N-00704, cmb_brca MSB-02664; CC BY 4.0) and the idc-torso1 demo, segmented with
TotalSegmentator `lung_vessels` through haversack. Scripts are in `research/vessels/`.*

## The two approaches

**vmtk** starts from an image, has a person segment a lumen with seeds and level sets, and
turns it into a **triangulated surface** (marching cubes, then smoothing and remeshing).
Everything afterwards is computed from that surface:

- centerlines are minimal paths over the surface's Voronoi diagram, with radius taken as the
  maximal inscribed sphere;
- branches are split by overlapping those spheres;
- coordinates come from harmonic maps;
- CFD meshes are built with TetGen.

It is mature, validated and widely used, especially for CFD.

**The field-based method** starts from a segmentation model's **continuous field**: the
per-class margins stored in a ranked store, which locate surfaces between voxels. Nothing is
converted to a surface for analysis:

- **connectivity** is decided by the field's own interpolated values on the original grid, and
  loops come from the genus of its zero surface;
- **centerlines** are seed-free minimal paths over that field-connected graph, weighted by an
  unlimited-range distance to the field's zero crossings, then refined onto the ridge;
- **radius, cross-sections and position** are all read from the field;
- **surfaces** are only a view, extracted or rendered when needed. The straightened 3D renders
  sample the field directly.

The method works on any grid and any model. Its geometric accuracy is bounded by the model it
is given, exactly as vmtk's is bounded by the segmentation it is given.

## Point by point

| | vmtk | Field-based method |
|---|---|---|
| **Input** | An image; segmentation is part of the workflow | A model's output field (here TotalSegmentator `lung_vessels` via haversack) |
| **Segmentation** | Interactive: seeds, level sets, vesselness | Automatic, one model run, no seeds |
| **Naming** | One unnamed lumen | Named classes; artery and vein are separate |
| **Core representation** | Triangle surface | Continuous field; the grid is only its sampling |
| **Surface quality** | Staircased by marching cubes; must be smoothed, which shrinks thin vessels | Located between voxels by the field; no smoothing needed for analysis |
| **Centerlines** | Voronoi diagram of the surface, with seeds for source and targets | Minimal paths on the field graph; branches found without seeds |
| **Centerline agreement** (measured) | — | vmtk lies a median **0.086 mm** from ours; 50 of 51 routes identical |
| **Radius** | Maximal inscribed sphere on the Voronoi diagram | Ridge-refined distance to the field's zero crossings; vmtk reads **+0.029 mm** larger (median) |
| **Radius repeatability** | Not tested (it depends on the surface given) | About **2×** as repeatable as a voxel-based radius (0.03–0.05 vs 0.06–0.10 mm) across reconstructions of the same scan |
| **Topology** | Inferred from one surface; artery–vein contacts become false loops | Exact for the field, verified by two independent methods; artery–vein contacts stay separate (73–95 false loops avoided per tree pair) |
| **Robustness** (measured) | 1–2 of 51 targets fail ("degenerate descent") | Whole tree traced; 12 of about 3,000 segments briefly leave the vessel, 6.7 mm in 32 m |
| **Speed** (Apple M2) | 4.3 s for a 51-target subtree | About 9 s for a whole 538-branch tree, after a 7 s decode |
| **Uncertainty** | None; one number per measurement | The model's own interval (±2 logits ≈ ±0.2 mm of radius on small vessels) |
| **Branch splitting, frames, wall maps** | Mature: ball-overlap splitting, bifurcation reference frames, patch mapping; all but the frames run on the surface | vmtk's splitting and frames run on our centerlines; its partition and wall coordinates reproduced bit for bit as field formulas, 50–100× faster, and volumetric; wall maps by ray casting (1.8 s, 0.02–0.03 mm from vmtk's) |
| **Wall curvature** | Mesh curvature (vtkCurvatures), noisy on marching-cubes surfaces | Level-set curvature from a local quadric fit: 1.02–1.04 × truth on phantoms, **6×** vmtk's repeatability across reconstructions |
| **CFD** | Mature: flow extensions, boundary layers, TetGen meshing, wall shear stress | Flow extensions as implicit geometry: one watertight domain, 0.02–0.03 mm from vmtk's; body-fitted meshing still vmtk's (or any mesher, from the zero set) |
| **Visualization** | VTK surfaces; curved planar reformation straightens pixels on a 2-D cut | The field rendered directly, including straightened 3D views of any path, with no resampled volume |
| **Maturity** | Two decades, papers, Slicer integration, a user community | Prototype scripts, two patients plus a demo, vmtk comparisons on one subtree |

## Shared limit: the segmentation model

This is not a difference between the methods. Both measure what the segmentation gives them:
fed the same field surface, vmtk's radii matched ours to 0.03 mm. Two properties of today's
input (TotalSegmentator `lung_vessels` on a 0.7 mm grid) therefore apply to either method:

- **A radius floor near 1 mm.** It sits at about 1.4 of the model's voxels and doesn't move
  with the scan: blurring the same anatomy from 0.625 mm to 2 mm slices left it unchanged. It
  comes from the model's training labels and grid, which sit at roughly clinical-CT
  resolution, where the scanner's own blur (~2 mm full width at half maximum) is already
  wider than a 1 mm vessel.
- **Thin vessels lost at thick slices.** At 3.75–5 mm slices the model drops them rather than
  widening them.

Both move with the input. A segmenter trained on higher-resolution data (photon-counting or
ultra-high-resolution CT, micro-CT for specimens) would lower the floor roughly in
proportion, to about 0.35 mm on a 0.25 mm grid. The field method needs no change for that; it
works in millimeters on whatever grid it is given.

Below the segmenter's floor, caliber survives in how bright a vessel appears. With the
scanner's blur known from its specifications or a phantom, image-side sizing (the field places
the tube, the image sizes it) can go further on today's scans.

## In short

The field-based method replaces vmtk's **front half** (segmentation, surface repair, seeded
centerlines) with something automatic, named and more precise. Where both compute the same
centerline and radius, they agree to within about 0.1 mm and 0.03 mm. Most of vmtk's **back
half** also runs from the field without a surface: branch partition, bifurcation frames, wall
coordinates and maps, wall curvature and flow extensions, reproducing vmtk where vmtk defines the
quantity (bit for bit for the coordinates). Body-fitted CFD meshing (TetGen, boundary layers)
stays vmtk's, fed the field's zero set; export is the one place a surface is made.
Resolution is set by the segmentation model, the same for both, and it improves with the model
without changes to the method.
