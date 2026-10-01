"""Read a ranked store: which structures it holds, and one structure's margin on its grid.

A haversack ranked store (``*.duckn.zip`` or a directory) is a zarr group whose ``parts/<i>``
hold rankfield parts; the duckn ``seg`` extension on the root names the classes. thalweg reads it
through rankfield's own reader (no haversack needed) and the seg extension's leaves:

- seg 0.9 and earlier: a leaf carries ``label_value(s)`` and ``layer`` (default 0), the index of
  its part in paint order;
- seg 0.10 (duckn convention 1.2): the same leaves, with ``layers[i].path`` naming the part group
  of layer ``i`` (``"parts/1"``).

A structure can appear in several parts (a cascade's coarse crop stage and its fine stage);
:meth:`FieldStore.margin` takes the finest (smallest voxel volume) unless a part is named.
The margin of class c is ``l_c - max(others)``: positive where c wins, its zero set the boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import ThalwegError


@dataclass(frozen=True)
class StructureRef:
    """One class of one part: its name, the part's index (``parts/<index>``), its label value and
    the labeling scheme that names it (from the segment's designation, else the store's single
    scheme; a cascade store has one scheme per stage, so the store-level one can be a list)."""

    name: str
    part: int
    label_value: int
    scheme: str | None = None


# TotalSegmentator `total` / `total_fast` label values of the crop-stage classes thalweg reads
# other spellings of TotalSegmentator's names, after canonical_name (a DICOM SEG written by
# TotalSegmentator labels its segments with their SNOMED display names)
ALIASES = {"small_intestine": "small_bowel",
           "left_upper_lobe_of_lung": "lung_upper_lobe_left",
           "left_lower_lobe_of_lung": "lung_lower_lobe_left",
           "right_upper_lobe_of_lung": "lung_upper_lobe_right",
           "middle_lobe_of_right_lung": "lung_middle_lobe_right",
           "right_lower_lobe_of_lung": "lung_lower_lobe_right"}


def canonical_name(name: str) -> str:
    """A structure name as TotalSegmentator spells it: case folded, every run of other characters an
    underscore, then :data:`ALIASES` (``"Small Intestine"`` -> ``small_bowel``, ``"Left Upper lobe
    of lung"`` -> ``lung_upper_lobe_left``)."""
    import re
    key = re.sub(r"[^0-9a-z]+", "_", str(name).casefold()).strip("_")
    return ALIASES.get(key, key)


def matching(structures, name: str, part: int | None = None) -> list:
    """The structures called ``name`` (in ``part``): by exact name, else by :func:`canonical_name`."""
    inpart = [s for s in structures if part is None or s.part == part]
    exact = [s for s in inpart if s.name == name]
    if exact:
        return exact
    key = canonical_name(name)
    return [s for s in inpart if canonical_name(s.name) == key]


TOTAL_VALUES = {"lung_upper_lobe_left": 10, "lung_lower_lobe_left": 11, "lung_upper_lobe_right": 12,
                "lung_middle_lobe_right": 13, "lung_lower_lobe_right": 14, "heart": 51, "pulmonary_vein": 53}


class FieldStore:
    """An open ranked store. ``structures`` lists every (name, part, label value) it names."""

    def __init__(self, path):
        import rankfield.store as rs
        self.path = Path(path).expanduser()
        try:
            self.root = rs.open_group(self.path)
        except FileNotFoundError as e:
            raise ThalwegError(f"no store at {self.path}") from e
        self._rs = rs
        attrs = self.root.attrs.asdict().get("duckn", {})
        self.seg = attrs.get("extensions", {}).get("seg") or {}
        self.labeling_scheme = self.seg.get("labeling_scheme")
        self.part_indices = rs.part_indices(self.root)
        self._fields = {}
        self.structures = self._structures()

    # -- naming ---------------------------------------------------------------------------
    def _layer_part(self, layer: int) -> int | None:
        """The part index of a seg layer, or None when the layer is not a ranked part (a derived
        labelmap, say): its segments are then not structures this reader can decode."""
        layers = self.seg.get("layers")
        if layers:
            if layer >= len(layers):
                return None
            path = str(layers[layer].get("path", ""))
            if not path.startswith("parts/"):
                return None
            return int(path.split("/")[1])
        return self.part_indices[layer] if layer < len(self.part_indices) else None

    def _structures(self) -> list[StructureRef]:
        out = []
        for s in self.seg.get("segments", []):
            if s.get("members") or s.get("role") == "background" or s.get("background"):
                continue
            v = s.get("label_value", s.get("label_values"))
            if isinstance(v, list):
                if len(v) != 1:
                    continue
                v = v[0]
            if v is None:
                continue
            scheme = next((d.get("scheme") for d in s.get("designations", []) if d.get("scheme")), None)
            if scheme is None and isinstance(self.labeling_scheme, str):
                scheme = self.labeling_scheme
            part = self._layer_part(int(s.get("layer", 0)))
            if part is None:
                continue
            out.append(StructureRef(str(s["name"]), part, int(v), scheme))
        return out

    @property
    def names(self) -> list[str]:
        return sorted({s.name for s in self.structures})

    # -- fields ---------------------------------------------------------------------------
    def field(self, part: int):
        """The rankfield ``RankField`` of ``parts/<part>``, its planes read into memory once.

        rankfield's readers hand out lazy zarr planes, and every ``margin`` call reads and
        decompresses all of them again; reading them once here makes each further structure of
        the same part a decode only (rankfield's own advice)."""
        if part not in self._fields:
            if part not in self.part_indices:
                raise ThalwegError(f"{self.path.name} has no part {part}; parts: {self.part_indices}")
            import dataclasses
            parts = self._rs.read_parts(self.root)
            code = dict(zip(self.part_indices, parts))[part].field
            self._fields[part] = dataclasses.replace(
                code, ranks=None if code.ranks is None else np.asarray(code.ranks[...]),
                support=np.asarray(code.support[...]),
                tail=None if code.tail is None else np.asarray(code.tail[...]))
        return self._fields[part]

    def ref(self, name: str, part: int | None = None) -> StructureRef:
        cands = matching(self.structures, name, part)
        if not cands:
            raise ThalwegError(f"{self.path.name} has no structure {name!r}"
                               + (f" in part {part}" if part is not None else "")
                               + f"; it names: {', '.join(self.names)}")
        if len(cands) == 1:
            return cands[0]
        vol = [abs(np.linalg.det(np.asarray(self.geometry(s.part).directions, float))) for s in cands]
        return cands[int(np.argmin(vol))]

    def ref_by_name_or_value(self, name: str, value: int | None = None) -> StructureRef:
        """A crop-stage class by name, or - in stores emitted before haversack 0.13, whose crop stage
        is named ``label_<value>`` - by its TotalSegmentator value (default: :data:`TOTAL_VALUES`).
        Raises ThalwegError if neither."""
        value = TOTAL_VALUES[name] if value is None else value
        if matching(self.structures, name):
            return self.ref(name)
        cand = [s for s in self.structures if s.name == f"label_{value}" and s.label_value == value]
        if not cand:
            raise ThalwegError(f"{self.path.name} has no class {name!r} (nor label_{value})")
        return cand[0]

    def geometry(self, part: int):
        """A part's grid placement, read from its attributes without decoding its planes."""
        return self._rs.array_geometry(self.root[f"parts/{part}"]["ranks"])

    def labelmap_mask(self, ref: StructureRef) -> np.ndarray:
        """Where the structure's class is the argmax of the part's decoded logits (the labelmap).

        The voxel comparison graph of the research reference connects these points; it differs from
        ``margin > 0`` only at exact ties (margin == 0) that the argmax gives to this class."""
        import rankfield as rf
        code = self.field(ref.part)
        win = np.argmax(np.stack([rf.deficit(code, c) for c in range(code.classes)]), axis=0)
        return win == code.labels.index(ref.label_value)

    def margin(self, name: str, part: int | None = None):
        """(margin float32 array, rankfield Geometry, StructureRef) of structure ``name``."""
        import rankfield as rf
        ref = self.ref(name, part)
        code = self.field(ref.part)
        try:
            ch = code.labels.index(ref.label_value)
        except ValueError as e:
            raise ThalwegError(
                f"part {ref.part} does not encode label value {ref.label_value} ({name})") from e
        return rf.margin(code, ch).astype(np.float32), code.geometry, ref

    def clip(self, part: int) -> float:
        return float(self.field(part).meta["clip"])


def open_store(path, sdf_inside: str = "negative"):
    """A ranked store (:class:`FieldStore`), or - for a NIfTI, NRRD or MetaImage file - a labelmap
    or signed distance image read in degraded mode (:class:`thalweg.volume.VolumeStore`; a distance
    image is negative inside by default, ``sdf_inside="positive"`` for the other convention)."""
    from .volume import VolumeStore, dicom_directory, is_volume, seg_files
    if hasattr(path, "margin") and hasattr(path, "structures"):     # already open
        return path
    path = Path(path).expanduser()
    if is_volume(path):
        return VolumeStore(path, sdf_inside=sdf_inside)
    if dicom_directory(path):
        segs = seg_files(path)
        if len(segs) == 1:                                       # a folder holding one SEG (IDC's layout)
            return VolumeStore(segs[0], sdf_inside=sdf_inside)
        raise ThalwegError(f"{path} is a directory of DICOM files: thalweg reads a DICOM SEG file, a "
                           "labelmap or distance image (NIfTI, NRRD, MetaImage), or a ranked store")
    if path.is_file() and not path.name.lower().endswith(".zip"):
        raise ThalwegError(f"{path.name} is not a ranked store, a NIfTI, NRRD or MetaImage image, "
                           "or a DICOM file")
    return FieldStore(path)
