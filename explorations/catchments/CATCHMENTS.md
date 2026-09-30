# Vascular catchments as a rankfield

*Exploration, 2026-09-24. One patient (C3N-00704, CTPA 0.625 mm, TotalSegmentator `lung_vessels`).
Prototype code in this directory; nothing here is in the thalweg package.*

## The idea

A **catchment** (territory) of a vessel branch is the tissue it supplies. The standard estimate
assigns each point of tissue to its nearest branch. Keeping not only the nearest branch but a
short ranked list (nearest, next nearest, and so on) with how far behind each one is, gives
exactly the structure of a **rankfield**, the encoding haversack uses for segmentation output:

| rankfield (segmentation) | catchments |
|---|---|
| class | a group of branches (a subtree) |
| logit | minus the distance to the group, in units of 0.25 mm |
| winner (rank 1) | the branch group that supplies the voxel |
| rank 2, 3, ... | the territories next door, nearest first |
| gap to the winner | d2 − d1: how much farther the next territory is; zero on the boundary |

So territories get everything rankfield and haversack already provide: sub-voxel boundaries,
compact storage, and the derived distance and junction layers.

**The store file:** `~/tmp/data/vessels/C3N-00704_ctpa0625.lung_artery_catchments.duckn.zip`
(5.3 MB), a standard haversack ranked store (duckn zarr zip) with its own `README.md` inside. It
is written by `catchment_store.py` through haversack's own builder, and read back with the
standard readers: ranks, support and the distance layer are byte-identical to the prototype's.

## Anatomical grounding

Bronchopulmonary segments are each "supplied by a tertiary bronchus and its own segmental
artery", and "the veins and lymphatics run along the edges of the segments"; the right lung has
10 segments and the left 8. The same arrangement repeats in the secondary pulmonary lobule, with
"the centrilobular arteriole and bronchiole" at its center and veins and lymphatics in the
interlobular septa at its edge ([Normal anatomy of the lungs, Radiology Key](https://radiologykey.com/normal-anatomy-of-the-lungs-2/)).
So arterial territories should have veins on their boundaries at every scale. The vein test
below checks this at the scale the model resolves: the finest territories here average ~14 mL
(about 2.4 cm across), but the branches that define them are at least ~1 mm in radius, so they
are subsegmental arteries, not centrilobular arterioles. The lobular version of the pattern needs
a model that sees far smaller vessels.

Two anatomical facts bear on the walls: accessory fissures are common (the inferior accessory
fissure in 40-50 % of specimens), and segmental branching varies (separate anterior and medial
basal origins in 4-10 % of left lower lobes). Walls from a lobe model are a default, not a truth.

## How the prototype builds it

1. **The tree.** The pulmonary artery centerline graph (1,065 branch segments, 537 tips) from
   `research/vessels/centerline.py`.
2. **Groups: trim the bottom of the tree.** Each segment gets a Strahler order (1 = a terminal
   twig; two joining branches of order k make order k+1). Every segment of order ≥ 3 is a group;
   smaller branches belong to their nearest ancestor of order ≥ 3. That gives about 200 groups
   per tree, which fits 8-bit ranks. Coarser levels (order ≥ 4, ≥ 5) are unions of these groups.
3. **Who supplies tissue.** Only terminal branches (order ≤ 2) count as sources (`SUPPLY=twigs`).
   Large trunks carry blood past tissue without supplying it; counting them cut neighboring
   territories into slivers. A group's territory is the region nearest to any of its twigs.
   Groups with no twigs of their own (55 trunk groups) get no territory, which leaves 158.
4. **Walls.** Territories never cross between the left and right lung (`WALLS=lungs`). Lobe
   fissures should also be walls, but this store's lobe map (the 3 mm `total_fast` stage of the
   `lung_vessels` cascade) mislabels most of the right upper lobe as right lower, so lobe walls
   (`WALLS=lobes`) are off by default until a better lobe model is used.
5. **Ranking.** At every lung voxel on a 1.5 mm grid (2,178 mL of lung), the four nearest
   groups and their distances, straight-line to the centerline.
6. **Encoding.** Those rankings become a `rankfield.RankField` directly (ranks plus
   one support byte per rank), without ever forming a dense 158-channel volume.
7. **The store.** `catchment_store.py` writes it as a duckn zarr zip through haversack's builder,
   adds `d1` and the wall map, declares the Strahler hierarchy as nested duckn groups (`members`,
   seg 0.9; every level complete on its own: 67 territories at Strahler ≥ 4, 21 at ≥ 5), records
   the source series, and puts a README inside that says what differs from a segmentation store.

## What the store holds

| Plane | Meaning | Source |
|---|---|---|
| `ranks` | which groups are nearest, in order | the ranking |
| `support` | how far each trails the winner (d_j − d1) | the ranking, through rankfield's log byte curve |
| `distance` | mm to the nearest territory boundary (including the lung surface) | haversack's `distance_field`, unmodified |
| `junction` | signed distance to the two-territory sheet near triple lines | haversack's `junction_field`, unmodified |
| `d1` | mm from the voxel to the vessel that supplies it | **new**: a softmax only needs differences, so rankfield stores none; here the absolute value matters (poorly supplied tissue) |

A rankfield's `tail` (the leftover probability) has no meaning here and is omitted. All five
planes together: **about 4.5 MB** compressed for both lungs.

## Results

**The format carries it.** haversack's distance and junction functions ran on the catchment
store without changes. The distance layer agrees with the geometric distance to the boundary
(the bisector between the two nearest sources) within a median of 0.13 mm, p90 0.42 mm.

**One store serves every level of the hierarchy.** rankfield's `decode_groups` gives each group
of classes its own margin, so the coarser territories decode from the order-3 store:

| Level | Regions | Winner agrees with a direct recomputation | Boundary margin error near boundaries |
|---|---|---|---|
| Strahler ≥ 4 | 67 | 99.996 % of lung voxels | median 0.006 mm (0.07 % of voxels > 0.05 mm) |
| Strahler ≥ 5 | 21 | 99.997 % | median 0.006 mm (0.27 % > 0.05 mm) |

The small misses are voxels where the nearest *other* coarse region was not among the four
stored ranks. More depth, or a distance plane for that level, removes them.

**Pulmonary veins run between arterial territories**, as anatomy says (the veins are
intersegmental). A naive comparison overstates this: veins never pass through arteries, and any
point far from every artery lies near a territory boundary by construction. So the veins are
compared with random lung points **matched on distance from the nearest artery**:

| Level | Veins: median distance d2 − d1 | Matched random points | Within 1 mm of a boundary (veins / matched) |
|---|---|---|---|
| Strahler ≥ 3 (158 regions) | 2.1 mm | 3.7 mm | 27 % / 16 % |
| Strahler ≥ 4 (67) | 3.5 mm | 5.5 mm | 18 % / 11 % |
| Strahler ≥ 5 (21) | 10.0 mm | 10.5 mm | 9 % / 6 % |

The effect is clear at the fine levels and mostly gone at the coarsest, one patient.


*Figure (`~/tmp/data/vessels/C3N-00704_ctpa0625_catchments_hires_twigs_lungs.png`): one coronal slice evaluated directly at 0.25 mm. Left: 21
territories at Strahler ≥ 5. Middle: 158 at Strahler ≥ 3. Right: distance to the nearest boundary
between territories (the lung surface not counted), with vein centerlines in blue running along
the dark boundaries.*

## Airways: the second tree agrees

The fine layer of the same `lung_vessels` store has the airway lumen, so the airway tree comes free:
`airway_centerlines.py` runs the frozen `research/vessels/centerline.py` on the `lung_airways`
class, unchanged except for one extra entry in its class table, added in memory. The tree has 265
segments, 134 tips and 3.4 m of centerline, rooted at the top of the trachea (median lumen
radius 1.2 mm; 3.4 s after decoding). The model follows airways less far out than arteries
(1,065 segments), so the airway levels are coarser.

`airway_catchments.py` builds airway territories under the same rules (Strahler groups, terminal
branches as sources, left/right walls) and compares them with the arterial ones at matched
region counts, against partitions from the same number of random seeds with the same walls:

| Arteries | Airways | ARI (null) | NMI (null) |
|---|---|---|---|
| Strahler ≥ 3 (157) | Strahler ≥ 2 (79) | 0.39 (0.29) | 0.75 (0.70) |
| Strahler ≥ 4 (66) | Strahler ≥ 2 (79) | 0.48 (0.30) | 0.75 (0.68) |
| **Strahler ≥ 5 (20)** | **Strahler ≥ 4 (12)** | **0.70 (0.37)** | **0.79 (0.62)** |
| Strahler ≥ 6 (11) | Strahler ≥ 4 (12) | 0.57 (0.41) | 0.75 (0.62) |

The two trees divide the lungs alike, most clearly at a dozen to twenty regions, the scale of the
bronchopulmonary segments. The arteries subdivide further where the airway model stops.

- **Veins lie on the airway boundaries too** at the finer levels: 3.9 vs 5.3 mm (79 regions) and
  6.3 vs 8.6 mm (34) from a boundary, against random points matched on distance from an airway;
  no effect at 12 and 7 regions.
- **Arteries run with the bronchi; veins do not.** Among vessel points within 15 mm of an airway,
  the median distance to the nearest airway centerline is 5.7 mm for arteries and 9.3 mm for
  veins (within 3 mm: 9 % against 2 %).
- `airway_figure.py` draws one coronal slice at 0.25 mm (airway territories filled, both sets of
  boundaries). On that slice 42 % of the airway interior boundaries lie within 3 mm of an arterial
  boundary (median 3.6 mm apart), against 15 % (10.5 mm) with the arterial boundaries shifted 25
  mm. That baseline is crude (the shift brings boundary-free ground from outside the lung into
  it), so treat the 3-D ARI comparison as the evidence and the slice as illustration.

Segment names are the next step: the ~18 segmental bronchi, named, would turn the airway level
into anatomical segments and label the matching arterial territories.

## Named segments (prototype, `segments.py`)

Lobes from the airway tree by position (the largest subtrees below each main bronchus; the
highest is the upper lobe, the more anterior of the other two the middle lobe), then within each
lobe the first real branches below the lobar bronchus, named by their position in the lobe
against templates, one name each. Territories: nearest named terminal airway, left/right walls.

On C3N-00704 (2026-09-30):
- **Lobes:** left upper and lower agree with the store's lobe map at Dice 0.96 and 0.95, right
  lower 0.84, right middle 0.75. Identifying lobes by branching order had put the middle lobe's
  takeoff above the upper lobe's and swapped them (Dice 0.00); position fixed it.
- **Left lung and right lower lobe look anatomically plausible** in the slices (superior segments
  posterior-superior, posterior basal posterior-inferior, apicoposterior at the apex).
- **Right upper and middle lobes are not reliable.** The model follows only 10 airway tips into
  the right upper lobe, so the middle lobe's airways claim the anterior upper lobe (RB5 spans the
  whole anterior right lung; the middle lobe comes out at 251 mL, larger than the upper lobe's
  198 mL, which anatomy does not allow). RB2 is missing, and naming confidence there is 0.14-0.37.
  The right upper lobe agrees with the store's map at Dice 0.16, but the store's own right upper
  lobe (103 mL) is also wrong, so neither is a reference.
- **LB3/LB4 may be swapped** (LB4 confidence 0.09; the region named LB3 reaches down where the
  lingula should be).
- Arterial territories fall mostly inside one named segment (86 % at Strahler >= 4, 78 % at >= 5).
- **Veins show no preference for the named segment boundaries** (12.2 vs 12.6 mm matched), unlike
  the finer airway and arterial territories. At this scale the prototype's segment boundaries are
  too uncertain to test the anatomy.

**Names carried onto the arteries** (`artery_segments.py`). A segmental bronchus has its own
segmental artery beside it, and the model follows the arteries much farther (537 tips against 134).
Arterial points within 8 mm of a named airway and running parallel to it are paired (30 % of
them); an arterial subtree whose paired points agree (>= 80 %) takes that name whole; the rest
(1 % of twig length) inherit the name of the nearest named branch along the tree, not through
space. Territories then come from the named arterial twigs:
- the left lobes agree with the store's lobe map at Dice 0.99 and 0.99 (0.96 and 0.95 from the
  airways alone): the arteries reach the edges of the lung, so the boundaries fill in;
- arteries and airways give the same segment to 86 % of lung voxels;
- the right upper and middle lobes do not change in substance: pairing carries the airways' names,
  errors and gaps included (RB2 and RB4 have no territory);
- veins: a weak preference for the segment boundaries (17 % within 2 mm, against 11 % matched).

**The artery tree confirms the lobe naming.** Middle-lobe arteries meet lower-lobe arteries first
(100 % of twigs; median 81 mm up the tree, against 146 mm to meet the upper lobe), lower-lobe
arteries meet the middle lobe first, and the upper-lobe arteries join both at one trunk: the
interlobar artery carrying the middle and lower lobes, and a separate upper-lobe trunk, as in the
textbook. What is labeled middle lobe is the middle lobe.

**And the CT explains the small right upper lobe** (`segments_on_ct.py`). The right upper lobe comes
out at 197 mL (about half of a typical one) and the whole right lung at 981 mL against 1,197 mL on
the left. On the CT the anterior right upper chest is not aerated at the upper levels (soft-tissue
density where the anterior segment's lung would be; the aerated right lung there is only its
posterior half). The lung mask excludes that tissue and the model finds almost no airways in it,
hence the small lobe, the 10 airway tips and the missing branches. Tumor, collapse or consolidation
is a question for a radiologist; this is a lung-adenocarcinoma collection. The segment labeling
itself is coherent with the anatomy the scan shows. One doubt remains on the left: LB4 (superior
lingular) appears anterolaterally at z = -60 mm, higher than the lingula usually reaches, which fits
a possible LB3/LB4 swap.

Next, if this continues: a second, healthier case to test the naming; a reference labeling; and
fissure walls from a trustworthy lobe model.

## Things learned the hard way

- **Trunks are not sources.** With every centerline point counted, one slice had 152 specks under
  5 mm² and 32 % of territories in several pieces; with twigs only, 14 and 8 %.
- **Check the walls.** A wrong fissure fenced off a small, bright pseudo-lobe in the right apex.
- **Match your controls.** The vein effect is about half as large after matching on distance
  from an artery as it looks without.
- **The format's clip promise does not hold.** A segmentation store guarantees that an unnamed
  class is at least `clip` behind; this one keeps the 4 nearest territories whatever their
  distance, so decoders that floor absent classes at the clip can overstate coarse margins.
- **haversack's verifier** fails the store on two checks by design (no softmax; margins in units
  of 0.25 mm rather than logits, which no haversack reader converts yet).
- **The junction layer is not sparse here.** At a 20 mm truncation it covers every lung voxel: a
  partition this fine has triple lines everywhere. A smaller reach would restore sparsity.

## Open questions

- **Which distance.** Straight-line distance to the centerline is the simplest choice. The
  alternatives are the vessel tube function (the same function that partitions the lumen, so one
  partition would cover lumen and tissue), flow weighting (bigger branches reach farther, r³ by
  Murray's law), and distance measured through the lung rather than straight.
- **Anatomical names.** Strahler order is not anatomy: the 18 bronchopulmonary segments (10 right,
  8 left) fall between arterial orders 4 (67 territories) and 5 (21). The airway territories now
  exist and agree with the arterial ones (above); naming the segmental bronchi is what remains.
- **A trustworthy lobe map** (`total` at 1.5 mm, organs part) to restore fissure walls.
- **Depth**, and whether coarse levels get their own distance planes.
- **More anatomy:** venous (drainage) catchments from the same store.
- **Applications:** lung volume downstream of an embolus (a volumetric obstruction index);
  distance from a tumor to its intersegmental plane (the surgical margin in segmentectomy);
  Couinaud liver segments as portal-vein catchments (compare with TotalSegmentator's
  `liver_segments`).
- **More than one patient.**

## Running it

From this directory, with haversack's environment:

```bash
../../../haversack/.venv/bin/python catchments.py            # the store, the checks, the vein test (~3 min)
../../../haversack/.venv/bin/python catchments_figure.py     # one coronal slice at 0.25 mm; optional args: RUN Y_MM
../../../haversack/.venv/bin/python catchment_store.py       # the duckn store, from catchments.py's output
../../../haversack/.venv/bin/python airway_centerlines.py    # the airway tree (frozen centerline.py, one class added)
../../../haversack/.venv/bin/python airway_catchments.py     # airway vs arterial territories, veins, pairing
../../../haversack/.venv/bin/python airway_figure.py         # one coronal slice at 0.25 mm; optional args: RUN Y_MM ART_K AIR_K
../../../haversack/.venv/bin/python segments.py              # named bronchopulmonary segments and their catchments
../../../haversack/.venv/bin/python artery_segments.py       # the names carried onto the arteries; arterial-source segments
../../../haversack/.venv/bin/python segments_on_ct.py        # the segments over the CT (lung window); optional args: RUN Z_AXIAL
```

Switches: `SUPPLY=twigs|all`, `WALLS=lungs|lobes`. Outputs go to `~/tmp/data/vessels/`
(`<run>_catchments_<supply>_<walls>.npz` and `.png`, `<run>_catchments_hires_<supply>_<walls>.png`).
