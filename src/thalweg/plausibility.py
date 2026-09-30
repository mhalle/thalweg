"""Are the arteries arteries and the veins veins? A check against the heart's classes.

A lung_vessels store's crop stage labels the heart and the pulmonary veins' trunks at the left
atrium (``pulmonary_vein``) - no pulmonary artery class. The model's vein tree runs on through
those trunks into the atrium, so a real vein tree has a measurable share of its centerline inside
the ``pulmonary_vein`` class (5-13 % on the cases measured), and an artery tree almost none
(0-0.06 %). :func:`check` reports both shares, and the roots' distances to the class and to the
heart as information, and calls the pair ``plausible`` when

- the veins' share is at least ``VEIN_INSIDE_SHARE`` (1 %),
- the arteries' share is at most ``ARTERY_INSIDE_SHARE`` (2 %), and below the veins'.

A model that swapped arteries and veins, or a tree traced from the wrong piece, fails these; the
reasons say which. (A first version tested the veins' root against the class: the root lies
27-36 mm past it, inside the atrium, so that test failed on every correct case.)
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .errors import ThalwegError
from .graph import TubeGraph
from .kernel.field import crossings, sample

VEIN_INSIDE_SHARE = 0.01
ARTERY_INSIDE_SHARE = 0.02
TOTAL_VALUES = {"heart": 51, "pulmonary_vein": 53}          # TotalSegmentator total / total_fast


def _distance(store, name: str):
    """A function: world points -> distance (mm) to class ``name`` (0 inside), from its margin's
    zero crossings on its own grid."""
    ref = store.ref_by_name_or_value(name, TOTAL_VALUES[name])
    m, geo, _ = store.margin(ref.name, ref.part)
    X = crossings(m, geo)
    if not len(X):
        raise ThalwegError(f"class {name} is empty")
    tree = cKDTree(X)

    def dist(P):
        P = np.atleast_2d(np.asarray(P, float))
        d = tree.query(P)[0]
        return np.where(sample(m, geo, P, cval=-8.0) > 0, 0.0, d)
    return dist


def _inside_share(graph: TubeGraph, structure: str, dist) -> float:
    total = inside = 0.0
    for e in graph.edges:
        if e.structure != structure:
            continue
        p = graph.edge_points(e)
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        mid = 0.5 * (p[1:] + p[:-1])
        inside += float(seg[dist(mid) == 0.0].sum())
        total += float(seg.sum())
    return inside / total if total else 0.0


def check(graph: TubeGraph, store, arteries: str = "lung_arteries", veins: str = "lung_veins") -> dict:
    """The artery/vein plausibility record (see the module docstring)."""
    names = {s.name for s in graph.structures}
    if not {arteries, veins} <= names:
        return dict(plausible=None, reason=f"needs both {arteries} and {veins} in the graph")
    try:
        to_pv = _distance(store, "pulmonary_vein")
        to_heart = _distance(store, "heart")
    except ThalwegError as e:
        return dict(plausible=None, reason=f"no heart classes in the store ({e})")
    ra = np.asarray(graph.nodes[graph.structure(arteries).roots[0]].position)
    rv = np.asarray(graph.nodes[graph.structure(veins).roots[0]].position)
    rec = dict(
        artery_root_to_pulmonary_vein_mm=round(float(to_pv(ra)[0]), 2),
        vein_root_to_pulmonary_vein_mm=round(float(to_pv(rv)[0]), 2),
        artery_root_to_heart_mm=round(float(to_heart(ra)[0]), 2),
        vein_root_to_heart_mm=round(float(to_heart(rv)[0]), 2),
        artery_length_share_inside_pulmonary_vein=round(_inside_share(graph, arteries, to_pv), 4),
        vein_length_share_inside_pulmonary_vein=round(_inside_share(graph, veins, to_pv), 4))
    reasons = []
    a, v = rec["artery_length_share_inside_pulmonary_vein"], rec["vein_length_share_inside_pulmonary_vein"]
    if v < VEIN_INSIDE_SHARE:
        reasons.append(f"only {v:.1%} of the veins' length runs through the pulmonary vein class "
                       f"(expected at least {VEIN_INSIDE_SHARE:.0%})")
    if a > ARTERY_INSIDE_SHARE:
        reasons.append(f"{a:.1%} of the arteries' length runs inside the pulmonary vein class")
    if a >= v:
        reasons.append("the arteries run through the pulmonary vein class at least as much as the veins do")
    rec.update(plausible=not reasons, reasons=reasons)
    return rec
