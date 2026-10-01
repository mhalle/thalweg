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
- **DICOM SEG** (a Segmentation object, read with highdicom: the ``dicom`` extra). Each segment is
  a structure named by its ``SegmentLabel`` (``segment_<number>`` if it has none), its label value
  the ``SegmentNumber``. Segments may overlap - each is its own mask, decoded when first used. A
  fractional segment counts where its value is at least 0.5. The grid is the one the
  segmentation's frames define (highdicom's volume geometry), padded by ``SEG_PAD`` voxels on
  every side: a SEG usually omits empty frames, so its grid ends at the segmentation and the
  structures would otherwise touch its edge.

What a degraded field does not have: the model's own interval. The +-2 logit levels the branch
table and the radius interval read become +-0.2 mm offsets of the wall - a convention, not an
uncertainty. Every structure's ``scheme`` says which mode it came from (``degraded:labelmap``,
``degraded:sdf`` or ``degraded:dicom-seg``), and the graph's ``source.labeling_scheme`` carries it
on.

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
SEG_SOP_CLASSES = ("1.2.840.10008.5.1.4.1.1.66.4", "1.2.840.10008.5.1.4.1.1.66.7")   # SEG, label map SEG
SEG_PAD = 3


def is_dicom(path) -> bool:
    """A file with the DICOM preamble ("DICM" at byte 128)."""
    p = Path(path).expanduser()
    try:
        with open(p, "rb") as f:
            head = f.read(132)
    except (OSError, IsADirectoryError):
        return False
    return len(head) == 132 and head[128:] == b"DICM"


def is_volume(path) -> bool:
    name = Path(path).expanduser().name.lower()
    return any(name.endswith(s) for s in SUFFIXES) or is_dicom(path)


def seg_files(path) -> list[Path]:
    """The DICOM SEG files directly in a directory (by SOP class; read without pixel data)."""
    p = Path(path).expanduser()
    if not p.is_dir():
        return []
    out = []
    for f in sorted(p.iterdir()):
        if f.is_file() and is_dicom(f):
            try:
                import pydicom
            except ImportError:                    # without the `dicom` extra no SEG can be read anyway
                return []
            try:
                head = pydicom.dcmread(str(f), stop_before_pixels=True)
            except Exception:                      # noqa: BLE001 - an unreadable file is not a SEG
                continue
            if str(head.get("SOPClassUID", "")) in SEG_SOP_CLASSES:
                out.append(f)
    return out


def dicom_directory(path) -> bool:
    """A directory holding DICOM files (an image series, which thalweg does not read)."""
    p = Path(path).expanduser()
    if not p.is_dir():
        return False
    return any(is_dicom(f) for f in list(p.iterdir())[:20] if f.is_file())


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


def _to_other_cell(mask: np.ndarray, sp: np.ndarray) -> np.ndarray:
    """For each voxel where ``mask`` is true: the distance (mm) from its center to the CELL (the
    voxel's box) of the nearest voxel where it is false - along the line to that voxel's center,
    where the line enters its box: |v| (1 - min_k s_k / (2 |v_k|)) for the offset v between the
    centers. On a straight wall that is exactly the distance to the wall, on any spacing."""
    d, idx = ndimage.distance_transform_edt(mask, sampling=sp, return_indices=True)
    here = np.indices(mask.shape)
    v = (idx - here).astype(float) * sp[:, None, None, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.min(np.where(v != 0, sp[:, None, None, None] / (2 * np.abs(v)), np.inf), axis=0)
    return np.where(mask, d * (1.0 - np.minimum(frac, 1.0)), 0.0)


def signed_distance(mask: np.ndarray, geometry) -> np.ndarray:
    """Signed distance (mm, positive inside) of a boolean mask on ``geometry``'s grid to its
    staircase surface - the boundary of the union of the inside voxels' boxes: from each voxel
    center, the distance to the box of the nearest voxel of the other kind (exact on a straight
    wall along any grid axis, whatever the spacing)."""
    sp = np.linalg.norm(np.asarray(geometry.directions, float), axis=1)
    if not mask.any():
        return np.full(mask.shape, -np.inf, np.float32)
    if mask.all():
        return np.full(mask.shape, np.inf, np.float32)
    return (_to_other_cell(mask, sp) - _to_other_cell(~mask, sp)).astype(np.float32)


def _mask_margin(mask: np.ndarray, geometry) -> np.ndarray:
    """The margin of a mask: its signed distance x SLOPE, clipped at +-CLIP. The margin is flat
    beyond CLIP / SLOPE mm of the wall, so the distance transform runs only on the mask's
    bounding box grown by that much (and two voxels): the same array, a fraction of the work."""
    out = np.full(mask.shape, -CLIP, np.float32)
    if not mask.any():
        return out
    sp = np.linalg.norm(np.asarray(geometry.directions, float), axis=1)
    grow = np.ceil(CLIP / SLOPE / sp).astype(int) + 2
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - grow, 0)
    hi = np.minimum(idx.max(0) + grow + 1, mask.shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    from rankfield.geometry import Geometry
    sub = Geometry(shape=tuple(int(b - a) for a, b in zip(lo, hi)), directions=geometry.directions,
                   origin=geometry.origin)                  # only the spacing is read
    out[box] = np.clip(SLOPE * signed_distance(mask[box], sub), -CLIP, CLIP)
    return out


def _check_inside(inside: np.ndarray, name: str, sdf_inside: str) -> None:
    """Refuse a distance image read with the wrong sign: an inside that is empty, or that covers most
    of the image's border (a structure seldom does; the background always does)."""
    other = "positive" if sdf_inside == "negative" else "negative"
    hint = f"is the distance {other} inside? (sdf_inside / --sdf-inside {other})"
    if not inside.any():
        raise ThalwegError(f"{name}: no voxel is inside, read as {sdf_inside} inside; {hint}")
    faces = ([inside[i] for i in (0, -1)] + [inside[:, i] for i in (0, -1)]
             + [inside[:, :, i] for i in (0, -1)])
    border = float(np.mean(np.concatenate([f.ravel() for f in faces])))
    if border > 0.5:
        raise ThalwegError(f"{name}: read as {sdf_inside} inside, the inside covers {border:.0%} of the "
                           f"image's border; {hint}")


class VolumeStore:
    """A labelmap, signed distance image or DICOM SEG with ``FieldStore``'s interface (see the
    module docstring). ``names``: label value (segment number) -> structure name, overriding the
    file's own names; ``sdf_inside``: ``"negative"`` (ITK's convention) or ``"positive"``."""

    def __init__(self, path, names: dict[int, str] | None = None, sdf_inside: str = "negative"):
        if sdf_inside not in ("negative", "positive"):
            raise ThalwegError(f"sdf_inside must be 'negative' or 'positive'; got {sdf_inside!r}")
        self.path = Path(path).expanduser()
        if not self.path.exists():
            raise ThalwegError(f"no image at {self.path}")
        self.sdf_inside = sdf_inside
        self.part_indices = [0]
        self._margins: dict[int, np.ndarray] = {}
        if is_dicom(self.path):
            self._open_seg(names)
            return
        try:
            import SimpleITK as sitk
        except ImportError as e:
            raise ThalwegError("reading a labelmap or distance image needs SimpleITK: "
                               "install thalweg's `volumes` extra") from e
        try:
            img = sitk.ReadImage(str(self.path))
        except RuntimeError as e:
            raise ThalwegError(f"{self.path.name}: {e}") from e
        if img.GetDimension() != 3 or img.GetNumberOfComponentsPerPixel() != 1:
            raise ThalwegError(f"{self.path.name}: a single-component 3-D image is needed")
        self.array = sitk.GetArrayFromImage(img)
        self._geometry = _geometry(img)
        a = self.array
        if (not np.issubdtype(a.dtype, np.integer) and np.isfinite(a).all() and a.min() >= 0
                and np.array_equal(a, np.round(a))):
            self.array = a.astype(np.int64)            # a labelmap stored as floats: never negative,
        if np.issubdtype(self.array.dtype, np.integer):    # whole numbers (a distance image has both signs)
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
        self._name(label)

    def _name(self, label: dict[int, str]) -> None:
        self.labeling_scheme = f"degraded:{self.kind}"
        self.structures = [StructureRef(n, 0, v, self.labeling_scheme) for v, n in sorted(label.items())]
        seen = [s.name for s in self.structures]
        if len(set(seen)) != len(seen):
            raise ThalwegError(f"{self.path.name}: two labels share a name ({seen})")

    def _open_seg(self, names) -> None:
        """A DICOM SEG: its segments and the grid its frames define (see the module docstring)."""
        try:
            import highdicom as hd
            import pydicom
        except ImportError as e:
            raise ThalwegError("reading a DICOM SEG needs highdicom: install thalweg's `dicom` extra") from e
        head = pydicom.dcmread(str(self.path), stop_before_pixels=True)
        if str(head.get("SOPClassUID", "")) not in SEG_SOP_CLASSES:
            raise ThalwegError(f"{self.path.name} is a DICOM file but not a Segmentation (SEG); "
                               f"its SOP class is {head.get('SOPClassUID', 'unknown')}")
        self.kind = "dicom-seg"
        self._seg = hd.seg.segread(str(self.path))
        g = self._seg.get_volume_geometry()
        if g is None:
            raise ThalwegError(f"{self.path.name}: its frames do not form a regular 3-D volume")
        self._seg_affine = np.asarray(g.affine, float)
        self._seg_shape = tuple(int(n) for n in g.spatial_shape)
        from rankfield.geometry import Geometry
        A = self._seg_affine[:3, :3]
        origin = self._seg_affine[:3, 3] - SEG_PAD * A.sum(1)
        self._geometry = Geometry(shape=tuple(n + 2 * SEG_PAD for n in self._seg_shape),
                                  directions=tuple(tuple(map(float, A[:, a])) for a in range(3)),
                                  origin=tuple(map(float, origin)))
        label = {}
        for item in self._seg.SegmentSequence:
            n = int(item.SegmentNumber)
            label[n] = (names or {}).get(n, str(getattr(item, "SegmentLabel", "") or f"segment_{n}"))
        self._name(label)

    def _seg_mask(self, number: int) -> np.ndarray:
        vol = self._seg.get_volume(segment_numbers=[number], combine_segments=False)
        if not (np.allclose(vol.affine, self._seg_affine) and tuple(vol.spatial_shape) == self._seg_shape):
            raise ThalwegError(f"{self.path.name}: segment {number} is on another grid than the others")
        a = np.asarray(vol.array)
        a = a[..., 0] if a.ndim == 4 else a
        mask = a >= 0.5 if np.issubdtype(a.dtype, np.floating) else a > 0
        return np.pad(mask, SEG_PAD)

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
        from .store import matching
        found = matching(self.structures, name) if part in (None, 0) else []
        if found:
            return found[0]
        where = f" in part {part} (an image has only part 0)" if part not in (None, 0) else ""
        names = ", ".join(self.names) or "nothing (no label in the image)"
        raise ThalwegError(f"{self.path.name} has no structure {name!r}{where}; it names: {names}")

    def ref_by_name_or_value(self, name: str, value: int | None = None) -> StructureRef:
        from .store import matching
        value = TOTAL_VALUES.get(name) if value is None else value
        found = matching(self.structures, name)
        if found:
            return found[0]
        for s in self.structures:
            if self.kind == "labelmap" and s.name == f"label_{value}" and s.label_value == value:
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
        if self.kind == "dicom-seg":
            return self._seg_mask(ref.label_value)
        return self.margin(ref.name)[0] > 0

    def margin(self, name: str, part: int | None = None):
        """(margin float32, Geometry, StructureRef): see the module docstring."""
        ref = self.ref(name, part)
        if ref.label_value not in self._margins:
            if self.kind in ("labelmap", "dicom-seg"):
                mask = (self.array == ref.label_value if self.kind == "labelmap"
                        else self._seg_mask(ref.label_value))
                self._margins[ref.label_value] = _mask_margin(mask, self._geometry)
            else:
                sdf = self.array.astype(np.float32)
                d = -sdf if self.sdf_inside == "negative" else sdf
                d = np.where(np.isfinite(d), d, -np.inf)           # NaN or inf: no distance, outside
                _check_inside(d > 0, self.path.name, self.sdf_inside)
                self._margins[ref.label_value] = np.clip(SLOPE * d, -CLIP, CLIP).astype(np.float32)
        return self._margins[ref.label_value], self._geometry, ref
