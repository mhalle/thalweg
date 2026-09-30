"""Whole-tree statistics: Horton's ratios, the small-vessel volume fraction, orientation entropy.

All are read from the graph alone (positions, traced radii, the tree), per structure:

- **Streams and Horton's laws.** A stream is a maximal chain of edges of one Strahler order (an
  edge continues its parent's stream when both have the same order). With N_k streams of order k,
  mean stream length L_k and mean stream diameter D_k, Horton's ratios are the factors by which
  these change per order, from least-squares lines of log N_k, log L_k, log D_k on k:
  ``bifurcation_ratio`` R_b (N_k / N_k+1), ``length_ratio`` R_l (L_k+1 / L_k), ``diameter_ratio``
  R_d (D_k+1 / D_k). Orders with no stream, and edges without an order (they lead only to
  truncated ends), are left out; fewer than three populated orders gives None.
- **Small-vessel volume fraction.** Volume is the sum over centerline segments of pi r^2 ds (r the
  mean traced radius of the segment's ends). ``small_vessel_volume_fraction`` is the share of it
  in segments whose cross-section pi r^2 is under ``small_area_mm2`` (default 5 mm^2: the "BV5"
  of pulmonary vascular pruning studies). The model's radius floor (~1 mm, area ~3 mm^2) sits just
  below that threshold, so this fraction depends on the model as much as on the anatomy.
- **Orientation entropy.** The entropy of the centerline's direction distribution (undirected,
  length-weighted, in ``ORIENTATION_BINS`` equal-area bins of the hemisphere), over its maximum:
  0 for a tree running one way, 1 for directions spread evenly (OSMnx's street-orientation entropy,
  in three dimensions). It rises with the number of segments as well as with their spread (evenly
  spread directions read 0.67 at 20 segments, 0.87 at 72, 0.99 at 1000), so a small tree reads low
  for its size alone: compare trees of similar size.
"""
from __future__ import annotations

import numpy as np

from .graph import TubeGraph
from .measure import strahler

POLAR_BINS, AZIMUTH_BINS = 6, 12
ORIENTATION_BINS = POLAR_BINS * AZIMUTH_BINS


def streams(graph: TubeGraph, structure: str) -> list[dict]:
    """Strahler streams: dicts with ``order``, ``edges`` (in flow order), ``length_mm`` and
    ``diameter_mm`` (twice the length-weighted mean traced radius)."""
    tree = graph.tree(structure)
    order = strahler(graph, structure, tree)
    head = {}                                   # edge -> the first edge of its stream
    for eid in tree.order:
        e = graph.edges[eid]
        up = tree.parent.get(e.start_node)
        head[eid] = head[up] if up is not None and order[up] == order[eid] and order[eid] is not None else eid
    chains: dict[int, list[int]] = {}
    for eid in tree.order:
        chains.setdefault(head[eid], []).append(eid)
    out = []
    for h, eids in chains.items():
        if order[h] is None:
            continue
        length = wsum = 0.0
        for eid in eids:
            p, r = graph.edge_points(eid), graph.edge_radius(eid)
            seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
            rm = 0.5 * (r[1:] + r[:-1])
            ok = rm > 0
            length += float(seg.sum())
            wsum += float((seg[ok] * rm[ok]).sum())
        out.append(dict(order=order[h], edges=eids, length_mm=length,
                        diameter_mm=2.0 * wsum / length if length > 0 else 0.0))
    return out


def _ratio(orders, values, increasing: bool) -> float | None:
    k = np.asarray(orders, float)
    v = np.asarray(values, float)
    ok = v > 0
    if ok.sum() < 3:
        return None
    slope = np.polyfit(k[ok], np.log(v[ok]), 1)[0]
    return float(np.exp(slope if increasing else -slope))


def horton(graph: TubeGraph, structure: str) -> dict:
    """Horton's ratios and the per-order stream table (see the module docstring)."""
    st = streams(graph, structure)
    orders = sorted({s["order"] for s in st})
    table, length, diameter = {}, [], []
    for k in orders:
        mine = [s for s in st if s["order"] == k]
        length.append(float(np.mean([s["length_mm"] for s in mine])))
        diameter.append(float(np.mean([s["diameter_mm"] for s in mine])))
        table[str(k)] = dict(streams=len(mine), mean_length_mm=round(length[-1], 3),
                             mean_diameter_mm=round(diameter[-1], 3))
    n = [table[str(k)]["streams"] for k in orders]
    return dict(bifurcation_ratio=_ratio(orders, n, increasing=False),
                length_ratio=_ratio(orders, length, increasing=True),
                diameter_ratio=_ratio(orders, diameter, increasing=True), orders=table)


def small_vessel_volume_fraction(graph: TubeGraph, structure: str, small_area_mm2: float = 5.0) -> dict:
    total = small = 0.0
    for e in graph.edges:
        if e.structure != structure:
            continue
        p, r = graph.edge_points(e), graph.edge_radius(e)
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        rm = 0.5 * (r[1:] + r[:-1])
        ok = rm > 0
        area = np.pi * rm[ok] ** 2
        vol = area * seg[ok]
        total += float(vol.sum())
        small += float(vol[area < small_area_mm2].sum())
    return dict(volume_mm3=round(total, 2),
                small_vessel_volume_fraction=round(small / total, 4) if total else None,
                small_area_mm2=small_area_mm2)


def orientation_entropy(graph: TubeGraph, structure: str) -> float | None:
    """Normalized entropy of the centerline's undirected, length-weighted directions."""
    w = np.zeros(ORIENTATION_BINS)
    for e in graph.edges:
        if e.structure != structure:
            continue
        d = np.diff(graph.edge_points(e), axis=0)
        seg = np.linalg.norm(d, axis=1)
        ok = seg > 0
        u = d[ok] / seg[ok, None]
        u = u + 0.0                                               # no negative zeros
        z, y, x = u[:, 2], u[:, 1], u[:, 0]                       # undirected: one of each +-u pair,
        flip = (z < 0) | ((z == 0) & ((y < 0) | ((y == 0) & (x < 0))))   # the equator folded too
        u = np.where(flip[:, None], -u, u) + 0.0
        # equal-area bins: uniform in cos(theta), uniform in azimuth
        polar = np.minimum((u[:, 2] * POLAR_BINS).astype(int), POLAR_BINS - 1)
        az = np.minimum(((np.arctan2(u[:, 1], u[:, 0]) + np.pi) / (2 * np.pi) * AZIMUTH_BINS).astype(int),
                        AZIMUTH_BINS - 1)
        np.add.at(w, polar * AZIMUTH_BINS + az, seg[ok])
    if w.sum() == 0:
        return None
    p = w[w > 0] / w.sum()
    return max(0.0, float(-(p * np.log(p)).sum() / np.log(ORIENTATION_BINS)))


def tree_statistics(graph: TubeGraph, structure: str) -> dict:
    """The statistics above for one structure, as a JSON-ready dict."""
    out = dict(horton=horton(graph, structure))
    out.update(small_vessel_volume_fraction(graph, structure))
    out["orientation_entropy"] = orientation_entropy(graph, structure)
    return out
