# Vascular catchments stored as a ranked field

This store uses haversack's **ranked segmentation store** format, but **it is not a segmentation**.
No model produced these classes, and nothing in it is a probability. It holds **arterial
catchments**: for every voxel of the lungs, the branches of the pulmonary artery tree that are
nearest to it, in order, and how much farther each one is than the nearest. The nearest branch is
the one taken to supply that tissue.

A prototype from thalweg (`explorations/catchments/`), 2026-09-24, one case. The generic format
reference follows as an appendix. Read this part first: several of the format's rules mean
something different here, and one does not hold at all.

## What the classes are

- **Class 0 = outside the lungs** (the `background` segment).
- **Classes 1..K = arterial territories.** Each is one branch segment of the artery centerline
  tree with Strahler order ≥ 3, together with every smaller branch hanging from it. Segment
  names say which tree segment and its Strahler order; `extensions.thalweg.classes` lists them.
- **Only small terminal branches (Strahler order ≤ 2) count as sources.** A territory is the
  tissue nearer to one of its own twigs than to any other territory's twigs. Large trunks carry
  blood past tissue without supplying it, so they are not sources; a trunk class holds only the
  territory of its own side branches, and trunks with none have no class.
- **Walls:** a territory never crosses between the left and right lung (`parts/0/walls`).
  Lobe fissures are not walls in this version, because the lobe map available for this case was
  wrong in the right lung.
- **Distance** is straight-line distance to the branch centerline, sampled every 0.5 mm.

## How to read the arrays

| Array | Here it means |
|---|---|
| `ranks` (4, Z, Y, X) | class + 1 of the 4 nearest territories, nearest first; 0 = none |
| `support` (3, Z, Y, X) | how much farther rank j is than rank 1, as a byte through the level table. The "logit" is **−distance / 0.25 mm** (`gap_unit`), so a decoded gap in logits × 0.25 = **mm of extra distance** (d_j − d_1) |
| `distance` (Z, Y, X) | mm to the nearest place the winning territory changes, **including the lung surface**: how deep a voxel sits inside its territory. Decoded as in the appendix; truncated at 20 mm |
| `junction`, `junction_pair` | the appendix's triple-line layer. **Not sparse here:** at a 20 mm truncation a partition this fine has triple lines everywhere, and the layer covers every lung voxel |
| `d1` (Z, Y, X) uint16 | **not in the generic format.** mm from the voxel to the centerline of the branch that supplies it: (value − 1) × 0.01; 0 = outside the lungs. A softmax needs only differences, so the format never stores an absolute score; here the absolute distance is the point (poorly supplied tissue) |
| `walls` (Z, Y, X) uint8 | 0 outside, 1 left lung, 2 right lung |
| `occupancy` | the generic skip index, unchanged |
| `tail` | absent: there is no probability mass to account for |

The winning territory at a voxel is `ranks[0] - 1`, as in any ranked store.

## The rule that does not hold: the clip

The generic reference says a class not named at a voxel is at least `clip` behind the winner.
**Not here.** This store keeps the **4 nearest** territories whatever their distance
(`keep = "nearest"`), so an unnamed territory is only known to be farther than the 4th. A decoder
that floors absent classes at −`clip` (32 logits = 8 mm) can overstate a margin where a fifth
territory is nearer than 8 mm. At the finest level this does not affect the winner, and near
boundaries the second territory is always stored; it matters for coarse group margins deep
inside large territories (see below).

## The hierarchy: groups

The coarser territories are declared as nested duckn groups (`members`, seg 0.9):
`strahler5_<segment>` contains `strahler4_<segment>` groups, which contain the stored classes.
**Each level is complete on its own:** the `strahler4_*` segments partition the lungs, and so do
the `strahler5_*` segments, even where a group has a single member. Take a level by its prefix.
The unions are authoritative for this store: the tree grouping is how the classes were made.

To get a group's territory and its own boundary margin, decode the union of its classes, for
example with `rankfield.decode_groups(code, [[classes of group A], [classes of group B], ...])`,
which returns each group's margin (its best member against the best non-member), not the maximum
of its members' margins. Checked against a direct recomputation: the winner agrees at 99.996 %
of lung voxels, and near boundaries the margin agrees to a median of 0.006 mm; 0.07 % (Strahler
≥ 4) and 0.27 % (Strahler ≥ 5) of near-boundary voxels need a territory beyond the stored 4.

## haversack's verifier

`tools/ranked_verify.py` fails this store on two checks, both by design: there is no `softmax`
block (no softmax produced these classes), and `gap_unit` is not logits. The second is a real
limit: no haversack reader converts a non-logit unit yet, so a tool that reports margins "in
logits" is reporting units of 0.25 mm here. Rank order, winners and boundaries are unaffected.
Its warnings about a missing `tail` and `frame` are expected for the same reasons.

## Geometry

1.5 mm isotropic grid, axis-aligned in LPS, `centering: cell`; the origin is the first voxel's
position. Array axes are Z, Y, X, as in every haversack store.

## Quick start

```python
import numpy as np, rankfield as rf
from haversack.ranked_store import open_store, read_segmentation
from haversack.ranked_restore import parts_of

st = open_store("C3N-00704_ctpa0625.lung_artery_catchments.duckn.zip")
code = parts_of(st.root)[0].field                  # a rankfield.RankField
winner = code.ranks[0].astype(int) - 1             # territory class per voxel; 0 = outside
d1 = st.root["parts/0/d1"][:]
d1_mm = np.where(d1 > 0, (d1 - 1) * 0.01, np.nan)  # distance to the supplying branch

segs = {s.id: s for s in read_segmentation(st.root).segments}
def classes(sid):                                  # a group's classes, resolving members
    s = segs[sid]
    return list(s.label_values or []) + [c for m in (s.members or []) for c in classes(m)]
groups = [sid for sid in segs if sid.startswith("strahler5_")]
margins = rf.decode_groups(code, [classes(g) for g in groups])   # logits; x 0.25 = mm
```

## Caveats

One patient (C3N-00704, CTPA 0.625 mm, TotalSegmentator `lung_vessels`). Straight-line distance
is the simplest territory model; the vessel tube function, flow weighting and distance measured
through the lung are untested alternatives. Strahler order is not anatomy: the 18
bronchopulmonary segments (10 right, 8 left) fall between orders 4 and 5 and need named branches. The territory
model's inputs inherit the segmentation model's limits, including its ~1 mm vessel radius floor.
Details and results: thalweg `explorations/catchments/CATCHMENTS.md`.
