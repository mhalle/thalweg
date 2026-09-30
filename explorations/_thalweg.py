"""What the explorations read, through the thalweg library.

Importing this module puts this checkout's ``src/`` (thalweg) and ``research/vessels/`` (the frozen
reference's data helpers) on ``sys.path``; run the explorations with haversack's venv, which has
thalweg's dependencies (rankfield, pydantic, click, scikit-image) as well as torch and SimpleITK.

From the library (``thalweg.store``, ``thalweg.kernel.medial``):

- :func:`field_store` - the run's ranked store, opened once per process (``FieldStore``);
- :func:`crop_field` - the crop stage's ``rankfield.RankField`` (``parts/0``: the 3 mm
  ``total_fast`` with the lobes), planes read into memory once;
- :func:`fine_field` - margins of named fine-layer classes and their grid;
- :func:`centerlines` - a structure's centerline tree, traced by
  ``thalweg.kernel.medial.trace(..., ridge_passes=1)`` on ``FieldStore.margin(cls)``, as a dict of the
  research reference's shape (run, cls, graph, root, scale, const, branches, nodes, segments;
  ``tests/test_medial.py`` pins it to ``research/vessels/centerline.py``'s JSON). Computed in memory
  and kept per process; nothing is written, so the research JSON in ``$VESSELS_DATA`` is never read
  or overwritten.

Not in the library, imported from ``research/vessels/`` unchanged: ``DATA`` and ``DICOM``
(``_data.py``: ``$VESSELS_DATA`` and ``$VESSELS_DICOM``), the thickness ladders ``LADDERS``, a
run's store path and image source (``cases.py``), and, lazily because ``_field.py`` imports
haversack, ``Grid`` (world <-> index) and ``Image`` (a CT sampled in world mm).
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO / "research" / "vessels", _REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

from _data import DATA, DICOM  # noqa: E402,F401
from cases import LADDERS, image, store  # noqa: E402,F401
from thalweg.kernel import medial  # noqa: E402
from thalweg.store import FieldStore, open_store  # noqa: E402


@functools.cache
def field_store(run: str) -> FieldStore:
    """The run's ranked store (``DATA/runs/<run>.lung_vessels.duckn.zip``), opened once."""
    return open_store(store(run))


def crop_field(run: str):
    """The crop stage's RankField (``parts/0``), what ``_field.parts_of(...)[0].field`` gave."""
    return field_store(run).field(0)


def fine_field(run: str, classes=("lung_arteries", "lung_veins")):
    """(margins {name: float32 array}, research ``Grid``) of fine-layer classes (the finest part
    that names each; all of them share one grid in a lung_vessels store)."""
    from _field import Grid
    margins, geo = {}, None
    for name in classes:
        margins[name], g, _ = field_store(run).margin(name)
        if geo is not None and g != geo:
            raise ValueError(f"{name} is not on the grid of {classes[0]}")
        geo = g
    return margins, Grid(np.asarray(geo.origin, float), np.asarray(geo.directions, float))


def ct(run: str):
    """The run's CT (DICOM series or NIfTI) as the research ``Image``, sampled in world mm."""
    from _field import Image
    patient = next(p for p, ladder in LADDERS.items() if any(r == run for r, *_ in ladder))
    src = next(s for r, s, *_ in LADDERS[patient] if r == run)
    return Image(image(src))


@functools.cache
def _trace(run: str, cls: str) -> medial.MedialTree:
    m, geo, _ = field_store(run).margin(cls)
    return medial.trace(m, geo, ridge_passes=1)


def centerlines(run: str, cls: str) -> dict:
    """The centerline tree of ``cls`` in the research reference's JSON shape. The trace runs once per
    run and class; every call shares its lists, so treat them as read-only."""
    T = _trace(run, cls)
    return dict(run=run, cls=cls, graph="field", root=T.root.tolist(), scale=medial.SCALE,
                const=medial.CONST, branches=T.branches, nodes=T.nodes, segments=T.segments)
