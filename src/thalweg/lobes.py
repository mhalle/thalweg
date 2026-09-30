"""Which lung lobe each branch lies in, from the lobe classes a lung_vessels store already holds.

A TotalSegmentator ``lung_vessels`` store is a cascade: its crop stage is a ``total``-family model
that labels the five lobes (values 10-14), at 1.5-3 mm, beside the fine vessel layer. So every
case carries its own lobe map, registered to the vessels by construction.

- :func:`lobe_fields` decodes the five lobe margins. Lobes are found by name
  (``lung_upper_lobe_left`` ...). Stores emitted before haversack 0.13 name the crop stage's classes
  ``label_<value>``; there the TotalSegmentator values 10-14 are used instead, and ``named_by`` says
  so (``"value (misnamed store)"``).
- :func:`point_lobes`: each centerline sample's lobe, the lobe whose margin wins there (the argmax
  of the five margins, where the best is > 0); 0 outside every lobe - the hilum and the mediastinum,
  where the big vessels and bronchi run before they enter a lobe.
- :func:`edge_lobes`: each edge's lobe, the one holding most of its length, and that share.
- :func:`lobe_volumes`: each lobe's volume, mm^3, from the crop grid (voxels the lobe wins).

Lobe numbers: 1 left upper, 2 left lower, 3 right upper, 4 right middle, 5 right lower
(TotalSegmentator's order, value - 9); :data:`LOBES` maps them to names.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import ThalwegError
from .graph import TubeGraph
from .kernel.field import sample

LOBES = {1: "lung_upper_lobe_left", 2: "lung_lower_lobe_left", 3: "lung_upper_lobe_right",
         4: "lung_middle_lobe_right", 5: "lung_lower_lobe_right"}
TOTAL_VALUES = {1: 10, 2: 11, 3: 12, 4: 13, 5: 14}          # TotalSegmentator total / total_fast


@dataclass
class LobeFields:
    margins: np.ndarray          # (5, Z, Y, X) float32, lobe k at index k - 1
    geometry: object             # the crop part's geometry
    part: int
    named_by: str                # "name" or "value (misnamed store)"


def lobe_fields(store) -> LobeFields:
    """The five lobe margins of a lung_vessels store (see the module docstring)."""
    by_name = [s for s in store.structures if s.name in LOBES.values()]
    if len({s.name for s in by_name}) == 5:
        refs = [store.ref(LOBES[k]) for k in LOBES]
        named_by = "name"
    else:
        refs = []
        for k in LOBES:
            cand = [s for s in store.structures if s.name == f"label_{TOTAL_VALUES[k]}"
                    and s.label_value == TOTAL_VALUES[k]]
            if not cand:
                raise ThalwegError(f"{store.path.name} names no lung lobes (by name or as label_10-14)")
            refs.append(cand[0])
        named_by = "value (misnamed store)"
    parts = {r.part for r in refs}
    if len(parts) != 1:
        raise ThalwegError(f"the lobes lie in several parts: {sorted(parts)}")
    part = parts.pop()
    ms = []
    geometry = None
    for r in refs:
        m, geometry, _ = store.margin(r.name, part)
        ms.append(m)
    return LobeFields(np.stack(ms), geometry, part, named_by)


def point_lobes(graph: TubeGraph, fields: LobeFields) -> np.ndarray:
    """Lobe number (1-5) of every sample in the point table; 0 outside every lobe."""
    P = graph.positions()
    if not len(P):
        return np.zeros(0, np.int64)
    v = np.stack([sample(m, fields.geometry, P, cval=-8.0) for m in fields.margins])
    best = v.argmax(0)
    return np.where(v.max(0) > 0, best + 1, 0).astype(np.int64)


def edge_lobes(graph: TubeGraph, structure: str, lobe_of_point: np.ndarray) -> dict[int, tuple[int, float]]:
    """Per edge: (the lobe holding most of its length, that share of its length); (0, share) when
    most of it lies outside every lobe."""
    out = {}
    for e in graph.edges:
        if e.structure != structure:
            continue
        a, b = e.point_range
        p = graph.positions()[a:b]
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        lab = lobe_of_point[a:b]
        # a segment belongs half to each end sample's lobe
        w = np.zeros(6)
        np.add.at(w, lab[:-1], seg / 2)
        np.add.at(w, lab[1:], seg / 2)
        total = w.sum()
        k = int(np.argmax(w))
        out[e.id] = (k, float(w[k] / total) if total > 0 else 0.0)
    return out


def lobe_volumes(fields: LobeFields) -> dict[int, float]:
    """Each lobe's volume, mm^3: the crop voxels it wins (its margin > 0) times the voxel volume."""
    voxel = abs(float(np.linalg.det(np.asarray(fields.geometry.directions, float))))
    return {k: float((fields.margins[k - 1] > 0).sum()) * voxel for k in LOBES}


def annotate(graph: TubeGraph, fields: LobeFields
             ) -> tuple[TubeGraph, dict[str, dict[int, tuple[int, float]]]]:
    """The graph with a ``lobe`` point column (0-5, :data:`LOBES`), and each structure's edge lobes."""
    pl = point_lobes(graph, fields)
    cols = dict(graph.points.columns)
    cols["lobe"] = [int(v) for v in pl]
    g = graph.model_copy(update={"points": graph.points.model_copy(update={"columns": cols})})
    return g, {s.name: edge_lobes(g, s.name, pl) for s in g.structures}


def lobe_rows(rows: list[dict], edge_lobe: dict[int, tuple[int, float]]) -> None:
    """Add ``lobe`` (name, or None outside every lobe) and ``lobe_length_fraction`` to branch-table
    rows of one structure, in place."""
    for r in rows:
        k, share = edge_lobe.get(r["edge"], (0, 0.0))
        r["lobe"] = LOBES.get(k)
        r["lobe_length_fraction"] = share


def lobe_summary(graph: TubeGraph, structure: str, edge_lobe: dict[int, tuple[int, float]],
                 volumes: dict[int, float]) -> dict:
    """Per lobe (and ``outside_lobes``): edges, tips, length, the lobe's volume and length density."""
    kind = {nd.id: nd.kind for nd in graph.nodes}
    out = {}
    for k in [*LOBES, 0]:
        es = [e for e in graph.edges if e.structure == structure and edge_lobe.get(e.id, (0, 0))[0] == k]
        length = sum(e.length_mm for e in es)
        row = dict(edges=len(es), tips=sum(kind[e.end_node] == "tip" for e in es), length_mm=round(length, 3))
        if k:
            ml = volumes[k] / 1000.0
            row.update(lobe_volume_ml=round(ml, 2),
                       length_mm_per_ml=round(length / ml, 3) if ml > 0 else None)
        out[LOBES.get(k, "outside_lobes")] = row
    return out
