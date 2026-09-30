"""Bronchoarterial pairing: each airway branch's companion artery, and their diameter ratio.

Pulmonary arteries run beside the bronchi they supply; veins do not (in the catchments
exploration, artery centerlines lie a median 5.7 mm from the nearest airway centerline, veins
9.3 mm). :func:`pair` pairs every airway centerline sample with the nearest artery sample that is
within ``reach`` mm (default 8) and runs parallel to it (|cos| of the two tangents >= ``parallel``,
default 0.7), the rule of ``explorations/catchments/artery_segments.py``.

Per airway edge (:func:`airway_rows`):

- ``paired_artery_edge``: the artery edge holding most of its paired samples. It is the nearest
  parallel artery, not a verified companion: on the two cases measured about 30 % of airway
  bifurcations pair both children with one artery edge, and a branch's samples spread over a
  median of 2-3 artery edges (docs/validation.md §5c);
- ``paired_fraction``: the share of its samples that found a partner, and
  ``paired_sample_count``, how many did (a ratio resting on a handful is weak);
- ``bronchus_to_artery_ratio``: the median over paired samples of the airway's lumen diameter over
  the partner artery's diameter (both the traced, ridge-refined inscribed diameters). This is the
  clinical bronchoarterial ratio: above 1 means the bronchus is wider than its artery, the CT sign
  of bronchiectasis (and of normal variation at altitude or with age).

Diameters are inscribed-ball diameters, so a flattened lumen reads its smaller width; both trees'
radii carry the model's ~1 mm floor, so ratios of the finest branches are the least reliable.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .graph import TubeGraph

REACH_MM, PARALLEL = 8.0, 0.7


def _samples(graph: TubeGraph, structure: str):
    """(positions, unit tangents, radii, edge id) of every sample of one structure's edges."""
    P, T, R, E = [], [], [], []
    for e in graph.edges:
        if e.structure != structure:
            continue
        p, r = graph.edge_points(e), graph.edge_radius(e)
        t = np.gradient(p, axis=0)
        n = np.linalg.norm(t, axis=1, keepdims=True)
        t = np.divide(t, n, out=np.zeros_like(t), where=n > 0)
        P.append(p), T.append(t), R.append(r), E.append(np.full(len(p), e.id))
    if not P:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0, int)
    return np.concatenate(P), np.concatenate(T), np.concatenate(R), np.concatenate(E)


def pair(graph: TubeGraph, airway: str, artery: str, reach: float = REACH_MM, parallel: float = PARALLEL):
    """Per airway sample: (partner artery edge or -1, airway radius, partner radius or NaN, airway
    edge). Among the artery samples within ``reach``, the nearest parallel one is the partner."""
    for name in (airway, artery):
        graph.structure(name)                                  # a ThalwegError if it is not there
    Pa, Ta, Ra, Ea = _samples(graph, airway)
    Pv, Tv, Rv, Ev = _samples(graph, artery)
    partner = np.full(len(Pa), -1)
    r_partner = np.full(len(Pa), np.nan)
    if len(Pa) and len(Pv):
        tree = cKDTree(Pv)
        for i, cands in enumerate(tree.query_ball_point(Pa, reach)):
            if not cands:
                continue
            c = np.asarray(cands)
            ok = np.abs(Tv[c] @ Ta[i]) >= parallel
            if not ok.any():
                continue
            c = c[ok]
            j = c[np.argmin(np.linalg.norm(Pv[c] - Pa[i], axis=1))]
            partner[i], r_partner[i] = Ev[j], Rv[j]
    return partner, Ra, r_partner, Ea


def airway_rows(graph: TubeGraph, airway: str, artery: str, rows: list[dict], **kw) -> dict:
    """Add the pairing columns (see the module docstring) to the airway's branch-table rows, in
    place; returns a summary: the paired share of airway samples, the median ratio, and the share
    of paired airway branches whose ratio exceeds 1."""
    partner, ra, rv, edge = pair(graph, airway, artery, **kw)
    ok = (partner >= 0) & (ra > 0) & (rv > 0)
    by_edge = {}
    for r in rows:
        on = edge == r["edge"]
        n = int(on.sum())
        both = on & ok
        if both.any():
            ids, counts = np.unique(partner[both], return_counts=True)
            ratio = float(np.median(ra[both] / rv[both]))
            by_edge[r["edge"]] = ratio
            r.update(paired_artery_edge=int(ids[np.argmax(counts)]), paired_fraction=float(both.sum() / n),
                     paired_sample_count=int(both.sum()), bronchus_to_artery_ratio=ratio)
        else:
            r.update(paired_artery_edge=None, paired_fraction=0.0 if n else None,
                     paired_sample_count=0, bronchus_to_artery_ratio=None)
    ratios = np.array(list(by_edge.values()))
    return dict(paired_sample_share=float(ok.mean()) if len(ok) else 0.0,
                paired_branches=len(ratios),
                bronchus_to_artery_ratio_median=float(np.median(ratios)) if len(ratios) else None,
                share_of_paired_branches_above_1=float((ratios > 1).mean()) if len(ratios) else None,
                reach_mm=kw.get("reach", REACH_MM), parallel_cosine=kw.get("parallel", PARALLEL))
