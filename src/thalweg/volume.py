"""Read a plain image instead of a ranked store: a labelmap or a signed distance field.

thalweg's own input is a ranked store, whose margins are the model's logits. Many segmentations
exist only as a labelmap (a NIfTI, NRRD or MetaImage of integer labels, a Slicer ``.seg.nrrd``) or
as a signed distance field. :class:`VolumeStore` reads those (through SimpleITK: the ``volumes``
extra) and offers the same interface as :class:`thalweg.store.FieldStore`, so every verb runs on
them - in a degraded mode, because a labelmap has no field between its voxels:

- **Labelmap** (an integer image). Each label value is a structure, named from a Slicer
  ``.seg.nrrd`` header (``SegmentN_Name`` / ``SegmentN_LabelValue``) when there is one, else
  ``label_<value>``. Its margin is the mask's signed distance in mm (positive inside; the boundary
  half a voxel beyond the last voxel inside), times ``SLOPE`` logits/mm, clipped at +-``CLIP``: a
  field the tracer and the sections read like the model's, with a wall slope like the model's.
  Its zero set is the labelmap's staircase, rounded; sections, radii and curvature measure that
  staircase, not a smooth wall.
- **Signed distance field** (a floating-point image, one structure named after the file). Its
  margin is -``sdf`` x ``SLOPE`` (``sdf_inside="negative"``, ITK's convention) or +``sdf`` x
  ``SLOPE``, clipped the same way.

What a degraded field does not have: the model's own interval. The +-2 logit levels the branch
table and the radius interval read become +-0.2 mm offsets of the wall - a convention, not an
uncertainty. Every structure's ``scheme`` says which mode it came from (``degraded:labelmap`` or
``degraded:sdf``), and the graph's ``source.labeling_scheme`` carries it on.

Geometry: SimpleITK's physical space is LPS, as thalweg's. Its array is indexed (z, y, x); the
rankfield ``Geometry`` has one direction row per array axis, so row 0 is the image's third axis
(its direction cosine times its spacing), row 2 the first.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from .errors import ThalwegError
from .store import TOTAL_VALUES, StructureRef

SLOPE = 10.0              # logits per mm: about the model's wall slope on the lung vessels
CLIP = 8.0
SUFFIXES = (".nii", ".nii.gz", ".nrrd", ".nhdr", ".mha", ".mhd")


def is_volume(path) -> bool:
    name = Path(path).name.lower()
    return any(name.endswith(s) for s in SUFFIXES)


@dataclass
class _Field:
    """What ``FieldStore.field`` returns that the pipeline reads: ``frame`` and ``meta``."""
    frame: dict
    meta: dict


def _geometry(img):
    from rankfield.geometry import Geometry
    D = np.asarray(img.GetDirection(), float).reshape(3, 3)          # columns: the image axes
    sp = np.asarray(img.GetSpacing(), float)
    rows = [D[:, a] * sp[a] for a in (2, 1, 0)]                        # array axes are (z, y, x)
    return Geometry(shape=tuple(img.GetSize()[::-1]),
                    directions=tuple(tuple(map(float, r)) for r in rows),
                    origin=tuple(map(float, img.GetOrigin())))


def signed_distance(mask: np.ndarray, geometry) -> np.ndarray:
    """Signed distance (mm, positive inside) of a boolean mask on ``geometry``'s grid, the boundary
    half a voxel beyond the mask's last voxel: each voxel center's distance to the nearest voxel
    center of the other kind, less half the grid's smallest spacing."""
    sp = np.linalg.norm(np.asarray(geometry.directions, float), axis=1)
    if not mask.any():
        return np.full(mask.shape, -np.inf, np.float32)
    inside = ndimage.distance_transform_edt(mask, sampling=sp)
    outside = ndimage.distance_transform_edt(~mask, sampling=sp)
    half = 0.5 * float(sp.min())
    return np.where(mask, inside - half, -(outside - half)).astype(np.float32)


class VolumeStore:
    """A labelmap or signed distance image with ``FieldStore``'s interface (see the module
    docstring). ``names``: label value -> structure name, overriding the file's own names;
    ``sdf_inside``: ``"negative"`` (ITK's convention) or ``"positive"``."""

    def __init__(self, path, names: dict[int, str] | None = None, sdf_inside: str = "negative"):
        try:
            import SimpleITK as sitk
        except ImportError as e:
            raise ThalwegError("reading a labelmap or distance image needs SimpleITK: "
                               "install thalweg's `volumes` extra") from e
        if sdf_inside not in ("negative", "positive"):
            raise ThalwegError(f"sdf_inside must be 'negative' or 'positive'; got {sdf_inside!r}")
        self.path = Path(path).expanduser()
        if not self.path.exists():
            raise ThalwegError(f"no image at {self.path}")
        try:
            img = sitk.ReadImage(str(self.path))
        except RuntimeError as e:
            raise ThalwegError(f"{self.path.name}: {e}") from e
        if img.GetDimension() != 3 or img.GetNumberOfComponentsPerPixel() != 1:
            raise ThalwegError(f"{self.path.name}: a single-component 3-D image is needed")
        self.array = sitk.GetArrayFromImage(img)
        self._geometry = _geometry(img)
        self.sdf_inside = sdf_inside
        if np.issubdtype(self.array.dtype, np.integer):
            self.kind = "labelmap"
            file_names = self._segment_names(img)
            values = [int(v) for v in np.unique(self.array) if v != 0]
            label = {v: (names or {}).get(v, file_names.get(v, f"label_{v}")) for v in values}
        else:
            self.kind = "sdf"
            stem = self.path.name
            for suffix in SUFFIXES:
                stem = stem[:-len(suffix)] if stem.lower().endswith(suffix) else stem
            label = {1: (names or {}).get(1, stem)}
        self.labeling_scheme = f"degraded:{self.kind}"
        self.structures = [StructureRef(n, 0, v, self.labeling_scheme) for v, n in sorted(label.items())]
        seen = [s.name for s in self.structures]
        if len(set(seen)) != len(seen):
            raise ThalwegError(f"{self.path.name}: two labels share a name ({seen})")
        self.part_indices = [0]
        self._margins: dict[int, np.ndarray] = {}

    @staticmethod
    def _segment_names(img) -> dict[int, str]:
        """Label value -> name from a Slicer .seg.nrrd header (SegmentN_Name, SegmentN_LabelValue)."""
        keys = set(img.GetMetaDataKeys())
        out = {}
        k = 0
        while f"Segment{k}_Name" in keys:
            value = f"Segment{k}_LabelValue"
            layer = f"Segment{k}_Layer"
            if value in keys and (layer not in keys or img.GetMetaData(layer).strip() == "0"):
                out[int(img.GetMetaData(value))] = img.GetMetaData(f"Segment{k}_Name").strip()
            k += 1
        return out

    @property
    def names(self) -> list[str]:
        return sorted(s.name for s in self.structures)

    def ref(self, name: str, part: int | None = None) -> StructureRef:
        for s in self.structures:
            if s.name == name and part in (None, 0):
                return s
        raise ThalwegError(f"{self.path.name} has no structure {name!r}; it names: {', '.join(self.names)}")

    def ref_by_name_or_value(self, name: str, value: int | None = None) -> StructureRef:
        value = TOTAL_VALUES.get(name) if value is None else value
        for s in self.structures:
            by_value = self.kind == "labelmap" and s.name == f"label_{value}" and s.label_value == value
            if s.name == name or by_value:
                return s
        raise ThalwegError(f"{self.path.name} has no class {name!r} (nor label_{value})")

    def geometry(self, part: int = 0):
        return self._geometry

    def field(self, part: int = 0) -> _Field:
        return _Field(frame={}, meta={"clip": CLIP, "kind": self.kind})

    def clip(self, part: int = 0) -> float:
        return CLIP

    def labelmap_mask(self, ref: StructureRef) -> np.ndarray:
        if self.kind == "labelmap":
            return self.array == ref.label_value
        return self.margin(ref.name)[0] > 0

    def margin(self, name: str, part: int | None = None):
        """(margin float32, Geometry, StructureRef): see the module docstring."""
        ref = self.ref(name, part)
        if ref.label_value not in self._margins:
            if self.kind == "labelmap":
                d = signed_distance(self.array == ref.label_value, self._geometry)
            else:
                sdf = self.array.astype(np.float32)
                d = -sdf if self.sdf_inside == "negative" else sdf
            self._margins[ref.label_value] = np.clip(SLOPE * d, -CLIP, CLIP).astype(np.float32)
        return self._margins[ref.label_value], self._geometry, ref
