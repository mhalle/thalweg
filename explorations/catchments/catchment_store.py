"""Write the catchment rankfield as a real store: a duckn zarr zip that haversack, rankfield and
the viewers read.

    cd explorations/catchments && ../../../haversack/.venv/bin/python catchment_store.py [RUN]

After catchments.py (reads DATA/<run>_catchments_<SUPPLY>_<WALLS>.npz). The ranked planes go
through haversack's own builder (`ranked_build._build_into`), handed an emit-shaped directory with
a `source_grid` part (the path FastSurfer uses: the grid is stated, not derived from a model run),
so occupancy, the distance and junction layers, the segments, validation and the generic format
README are the standard ones. Then, inside the same write:

- `parts/0/d1`: mm to the supplying vessel, uint16 in 0.01 mm with 0 = outside the lungs
  (the store's zero-means-nothing rule), and `parts/0/walls` (0 outside, 1 left, 2 right lung);
- the Strahler hierarchy as nested duckn groups (`members`, seg 0.9): order >= 5 territories
  containing order >= 4 territories containing the stored classes. These unions are
  authoritative here - the tree grouping is how the classes were made;
- provenance that says what actually happened (no model ran on these classes);
- README.md: what this store is and is not, then haversack's generic format reference.
"""
import json, shutil, sys, tempfile, time
from pathlib import Path
import numpy as np

import haversack
from haversack import ranked_build
from haversack.ranked_store import open_store, segmentation, read_segmentation
from _catch import run, TAU, TRIM, SUPPLY, WALLS, tree, ancestor_at, DATA

src_npz = DATA / f"{run}_catchments_{SUPPLY}_{WALLS}.npz"
out = DATA / f"{run}.lung_artery_catchments.duckn.zip"
TRUNC = 20.0                                                          # mm, distance and junction truncation
Z = np.load(src_npz)
H = float(Z["spacing"]); origin = [float(v) for v in Z["origin"]]
group_seg = [int(v) for v in Z["group_segment"]]
K = len(group_seg)
to_zyx = lambda a: np.ascontiguousarray(np.transpose(a, (0, 3, 2, 1)) if a.ndim == 4 else np.transpose(a, (2, 1, 0)))
ranks, support = to_zyx(Z["ranks"]), to_zyx(Z["support"])
d1, walls = to_zyx(Z["d1"]), to_zyx(Z["lobe"]).astype(np.uint8)
inside = walls > 0

S, order, parent, _, _ = tree("lung_arteries")
names = {0: "outside the lungs"}
for c, g in enumerate(group_seg):
    names[c + 1] = f"arterial territory, segment {g} (Strahler {order[g]})"

meta_codec = {"version": "0.4", "mode": "ranked", "classes": K + 1, "depth": int(ranks.shape[0]),
              "clip": 32.0, "gap_unit": f"-distance / {TAU} mm", "gap_curve": "log", "gap_range": 64.0,
              "gap_origin": 0.5, "keep": "nearest", "support_max": 255, "rank_sentinel": 0, "exhaustive": False}
part = {**meta_codec, "labels": list(range(K + 1)), "task": "thalweg:lung_artery_catchments",
        "engine": "thalweg", "haversack": haversack.__version__,
        "source_grid": {"spacing_zyx": [H, H, H], "origin_xyz": origin,
                        "direction_xyz": [1, 0, 0, 0, 1, 0, 0, 0, 1], "shape_zyx": list(ranks.shape[1:])}}
meta = {"task": "thalweg:lung_artery_catchments", "image": f"{run}.lung_vessels.duckn.zip",
        "depth": part["depth"], "clip": part["clip"], "parts": {"catchments": part}}

t0 = time.time()
emit = Path(tempfile.mkdtemp(prefix="catchment_emit_", dir=DATA))
try:
    (emit / "meta.json").write_text(json.dumps(meta))
    np.save(emit / "catchments_ranks.npy", ranks)
    np.save(emit / "catchments_support.npy", support)

    def say(*a, **k):
        k.pop("flush", None); k.pop("file", None)
        print(*a, **k)

    with open_store(out, "w") as st:
        source = {"type": "dataset", "format": "DICOM",
                  "identifier": "1.3.6.1.4.1.14519.5.2.1.4801.5885.265238222707560839809959746263",
                  "doi": "10.7937/K9/TCIA.2018.PAT12TBS",
                  "description": "Imaging Data Commons, collection cptac_luad (CPTAC-LUAD), patient C3N-00704, "
                                 "series 4 'CTPA (40% ASIR)', 0.625 mm; crdc_series_uuid "
                                 "86782735-ac4f-4083-b7b1-f32b1c63bafe; CC BY 4.0. The arterial tree came from "
                                 "TotalSegmentator lung_vessels on this series (haversack store "
                                 f"{run}.lung_vessels.duckn.zip)"}
        ranked_build._build_into(st, emit, out, run, "all", True, TRUNC / H, names, meta, say, source=source)
        root = st.root
        g = root["parts/0"]
        eff, direction = [H, H, H], [1, 0, 0, 0, 1, 0, 0, 0, 1]
        a3 = ranked_build.attrs(direction, eff, origin, list_axis=False, centering="cell")
        chunks, shards = ranked_build.layout(d1.shape)
        q = np.where(inside, np.clip(np.round(d1 * 100) + 1, 1, 65535), 0).astype(np.uint16)
        for name, arr in (("d1", q), ("walls", walls)):
            z = g.create_array(name, shape=arr.shape, dtype=arr.dtype, chunks=chunks, shards=shards,
                               compressors=ranked_build.zarr.codecs.ZstdCodec(level=9), attributes=a3)
            z[:] = arr
        pa = g.attrs.asdict()
        pa["duckn"].setdefault("extensions", {})["thalweg"] = {
            "kind": "vascular catchments",
            "d1": {"meaning": "distance from the voxel to the centerline of the branch that supplies it",
                   "unit": "mm", "decode": "(value - 1) * 0.01; 0 = outside the lungs"},
            "walls": {"meaning": "territories never cross a wall", "values": {"0": "outside the lungs",
                                                                              "1": "left lung", "2": "right lung"}}}
        g.attrs.update(pa)

        # the hierarchy: nested groups, container first (seg 0.9 section 5)
        leaves = {s.id: s for s in read_segmentation(root).segments}
        leaf_id = {c + 1: f"c{c + 1}" for c in range(K)}
        anc4, anc5 = ancestor_at(order, parent, 4), ancestor_at(order, parent, 5)
        g4 = {}
        for c, s_ in enumerate(group_seg):
            g4.setdefault(anc4[s_], []).append(leaf_id[c + 1])
        # EVERY level is complete: the strahler4_* groups partition the lungs, and so do the
        # strahler5_* groups, even where a group has a single member - a reader takes a level by
        # its prefix and never has to know which territories were left implicit
        g5 = {}
        for t in g4:
            g5.setdefault(anc5[t], []).append(f"strahler4_{t}")
        from duckn import Segment
        seg5 = [Segment(id=f"strahler5_{t}", name=f"Strahler >= 5 territory of segment {t} (Strahler {order[t]})",
                        members=sorted(m)) for t, m in sorted(g5.items())]
        seg4 = [Segment(id=f"strahler4_{t}", name=f"Strahler >= 4 territory of segment {t} (Strahler {order[t]})",
                        members=sorted(m)) for t, m in sorted(g4.items())]
        bg = [s for s in leaves.values() if s.role == "background"]
        rest = [s for s in leaves.values() if s.role != "background"]
        seg_ext = segmentation(bg + seg5 + seg4 + rest)
        ra = root.attrs.asdict()
        ext = ra["duckn"]["extensions"]
        ext["seg"] = seg_ext.model_dump(exclude_none=True)
        ext["provenance"]["processing"] = [
            {"name": "Segmentation",
             "description": "TotalSegmentator lung_vessels via haversack: the arteries and veins "
                            "whose centerlines define the branches. The store it came from is named in "
                            "haversack.source_file.",
             "software": {"name": "haversack", "version": haversack.__version__}},
            {"name": "Centerline graph",
             "description": "seed-free centerlines on the field of lung_arteries, split into branch "
                            "segments with Strahler orders",
             "software": {"name": "thalweg research/vessels/centerline.py", "version": "incubation"}},
            {"name": "Catchments",
             "description": "every lung voxel ranked by straight-line distance to the arterial branch "
                            f"groups: classes = segments of Strahler order >= {TRIM} (smaller branches "
                            f"belong to their nearest such ancestor); sources = {SUPPLY} (twigs = Strahler "
                            f"<= 2 only); walls = {WALLS}; logit = -distance / {TAU} mm; the {part['depth']} "
                            "nearest groups kept",
             "software": {"name": "thalweg explorations/catchments/catchments.py", "version": "prototype"},
             "parameters": {"strahler_trim": TRIM, "supply": SUPPLY, "walls": WALLS, "tau_mm": TAU,
                            "grid_mm": H, "depth": part["depth"]}},
            {"name": "Store layout",
             "description": "haversack's ranked builder (occupancy, distance and junction layers at a "
                            f"{TRUNC} mm truncation), plus d1, walls and the Strahler hierarchy as groups",
             "software": {"name": "thalweg explorations/catchments/catchment_store.py", "version": "prototype"}}]
        ext["thalweg"] = {"kind": "vascular catchments", "tree": "lung_arteries", "case": run,
                          "levels": {"stored classes": f"Strahler >= {TRIM}", "groups": ["Strahler >= 4", "Strahler >= 5"]},
                          "classes": [{"value": c + 1, "segment": s_, "strahler": order[s_]} for c, s_ in enumerate(group_seg)]}
        root.attrs.update(ra)

        generic = ranked_build.README.read_text(encoding="utf-8")
        st.write_text("README.md", (Path(__file__).parent / "STORE_README.md").read_text(encoding="utf-8")
                      + "\n\n---\n\n# Appendix: the generic ranked-store format reference\n\n"
                      + "*(haversack's README for segmentation stores, unchanged. Read it with the "
                      + "differences above in mind.)*\n\n" + generic)
finally:
    shutil.rmtree(emit, ignore_errors=True)
print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB) in {time.time() - t0:.1f} s: {K} classes, "
      f"{len(seg4)} Strahler>=4 groups, {len(seg5)} Strahler>=5 groups")
