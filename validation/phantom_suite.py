"""Phase 5 validation: analytic tube phantoms with known truth, through thalweg's own pipeline.

    uv run python validation/phantom_suite.py [--spacing 0.7] [--out results.json]

Every phantom is a margin field built from capsule chains (tests/phantoms.py): known centerline,
radius, ends, junctions and branching angles. For each, the field is traced
(kernel.medial.trace), graphed (centerlines.graph_from_tree) and measured (measure.branch_table),
and scored:

- topology: traced ends vs true ends (matched within radius + 1 voxel), junction count;
- centerline: distance from traced points to the true axis (median, p95);
- radius: traced radius vs true radius away from ends and junctions (median |error|, p95);
- sections: the section's equivalent diameter over the TRACED diameter (median ratio; a
  consistency check, not accuracy against the truth);
- angles: the table's deflection at each junction vs the true angle between parent and daughter.

Phantoms: Y bifurcations at 30/60/90 degrees, a trifurcation, a tapered branch, a curved (arc)
tube, a flattened (elliptic) tube, a closed loop (torus: the tracer builds trees, so the loop's
presence is checked in the field topology instead), two tubes in contact, and a thin tube (radius
0.8 voxel, diameter 1.6 voxels). The round tubes are axis-aligned (their axes on lattice lines):
the ridge refinement's quantization reads best-case here. Numbers go to stdout and optionally JSON;
nothing here asserts - tests/ does that for the cases that must hold.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rankfield.geometry import Geometry

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from phantoms import CLIP, SLOPE, capsule_distance  # noqa: E402

from thalweg.centerlines import graph_from_tree  # noqa: E402
from thalweg.graph import Points, Source, TubeGraph  # noqa: E402
from thalweg.kernel import medial  # noqa: E402
from thalweg.kernel.topology import components, field_edges, surface_loops  # noqa: E402
from thalweg.measure import branch_table  # noqa: E402


def field_of(dist_fn, lo, hi, spacing):
    shape = tuple(int(np.ceil((b - a) / spacing)) + 1 for a, b in zip(lo, hi))
    geo = Geometry(shape=shape, directions=((spacing, 0, 0), (0, spacing, 0), (0, 0, spacing)),
                   origin=tuple(lo))
    idx = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1).reshape(-1, 3)
    X = np.asarray(lo) + idx * spacing
    d = dist_fn(X)
    return np.clip(d * SLOPE, -CLIP, CLIP).astype(np.float32).reshape(shape), geo


def chain_distance(segments):
    """segments: list of (a, b, ra, rb) -> the union's signed distance function."""
    def f(X):
        d = np.full(len(X), -np.inf)
        for a, b, ra, rb in segments:
            d = np.maximum(d, capsule_distance(X, np.asarray(a, float), np.asarray(b, float), ra, rb))
        return d
    return f


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def y_phantom(angle_deg, r0=3.0, r1=2.0, r2=2.0, L0=30.0, L=25.0):
    """Trunk along +z; daughters at +-angle/2 from the trunk's continuation in the x-z plane."""
    top = np.array([0.0, 0, L0])
    h = np.radians(angle_deg / 2)
    d1, d2 = np.array([np.sin(h), 0, np.cos(h)]), np.array([-np.sin(h), 0, np.cos(h)])
    segs = [((0, 0, 0), top, r0, r0), (top, top + L * d1, r1, r1), (top, top + L * d2, r2, r2)]
    truth = dict(ends=[(0, 0, 0), top + L * d1, top + L * d2], junctions=[top], axes=segs,
                 deflections=[angle_deg / 2, angle_deg / 2])
    return segs, truth


def trifurcation():
    top = np.array([0.0, 0, 30])
    dirs = [unit([0.6, 0, 0.8]), unit([-0.3, 0.52, 0.8]), unit([-0.3, -0.52, 0.8])]
    segs = [((0, 0, 0), top, 3.0, 3.0)] + [(top, top + 22 * d, 1.8, 1.8) for d in dirs]
    return segs, dict(ends=[(0, 0, 0)] + [top + 22 * d for d in dirs], junctions=[top], axes=segs,
                      deflections=[np.degrees(np.arccos(d[2])) for d in dirs])


def tapered():
    return [((0, 0, 0), (0, 0, 40), 3.0, 1.0)], dict(ends=[(0, 0, 0), (0, 0, 40)], junctions=[],
                                                      axes=[((0, 0, 0), (0, 0, 40), 3.0, 1.0)],
                                                      deflections=[])


def arc(radius_of_curvature=15.0, r=1.5, sweep_deg=150, n=60):
    t = np.radians(np.linspace(0, sweep_deg, n))
    P = np.stack([radius_of_curvature * np.cos(t), radius_of_curvature * np.sin(t), np.zeros(n)], 1)
    segs = [(P[k], P[k + 1], r, r) for k in range(n - 1)]
    return segs, dict(ends=[P[0], P[-1]], junctions=[], axes=segs, deflections=[])


def elliptic(a=3.0, b=1.2, L=40.0):
    def f(X):
        s = np.clip(X[:, 0], 0, L)
        inside_len = np.where((X[:, 0] >= 0) & (X[:, 0] <= L), 0.0, np.abs(X[:, 0] - s))
        e = np.sqrt((X[:, 1] / a) ** 2 + (X[:, 2] / b) ** 2)
        return (1 - e) * b - inside_len
    return f, dict(ends=[(0, 0, 0), (L, 0, 0)], junctions=[], axes=[((0, 0, 0), (L, 0, 0), b, b)],
                   deflections=[], semi_axes=(a, b))


def torus(R=12.0, r=2.0):
    def f(X):
        q = np.hypot(X[:, 0], X[:, 1]) - R
        return r - np.hypot(q, X[:, 2])
    return f, dict(loops=1)


def contact(gap=-0.3, r=1.5):
    """Two parallel tubes whose walls overlap by -gap mm (contact) - one structure, two tubes."""
    c = 2 * r + gap
    segs = [((0, 0, 0), (0, 0, 30), r, r), ((c, 0, 0), (c, 0, 30), r, r)]
    return segs, dict(ends=[(0, 0, 0), (0, 0, 30), (c, 0, 0), (c, 0, 30)], junctions=[], axes=segs,
                      deflections=[])


def score(name, m, geo, truth, spacing):
    out = dict(phantom=name)
    ncomp_idx = field_edges(m)
    ncomp, _ = components(len(ncomp_idx[0]), ncomp_idx[1], ncomp_idx[2])
    out["field_components"] = int(ncomp)
    if "loops" in truth:
        genus, *_ = surface_loops(m)
        out["field_loops"] = int(genus)
        out["true_loops"] = truth["loops"]
    T = medial.trace(m, geo)
    nodes, edges, pos, rad, s = graph_from_tree(T, name, m, geo, Source(), {"graph": "field"})
    g = TubeGraph(structures=[s], nodes=nodes, edges=edges, points=Points(position=pos, radius=rad))
    ends = np.array([n.position for n in g.nodes if n.kind in ("tip", "truncated")]
                    + [n.position for n in g.nodes if n.kind == "root" and sum(
                        1 for e in g.edges if n.id in (e.start_node, e.end_node)) == 1])
    out["traced_ends"] = len(ends)
    out["traced_junctions"] = sum(n.kind == "junction" for n in g.nodes)
    if "ends" in truth:
        te = np.asarray(truth["ends"], float)
        out["true_ends"] = len(te)
        out["true_junctions"] = len(truth["junctions"])
        d = np.linalg.norm(te[:, None] - ends[None], axis=2).min(1) if len(ends) else np.full(len(te), np.inf)
        out["end_error_mm_max"] = round(float(d.max()), 3)
    if "axes" in truth:
        P, R = g.positions(), g.radii()
        segs = truth["axes"]
        dist = np.min([_seg_dist(P, a, b) for a, b, *_ in segs], axis=0)
        keyp = np.array([*truth.get("ends", []), *truth.get("junctions", [])], float).reshape(-1, 3)
        away = (np.linalg.norm(P[:, None] - keyp[None], axis=2).min(1) > 5.0 if len(keyp)
                else np.ones(len(P), bool))
        out["axis_error_mm_median"] = round(float(np.median(dist[away])), 3)
        out["axis_error_mm_p95"] = round(float(np.percentile(dist[away], 95)), 3)
        if "semi_axes" not in truth:
            true_r = _true_radius(P, segs)
            err = np.abs(R - true_r)[away & (R > 0)]
            out["radius_error_mm_median"] = round(float(np.median(err)), 3)
            out["radius_error_mm_p95"] = round(float(np.percentile(err, 95)), 3)
    rows = branch_table(g, name, m, geo, step=1.0)
    eq = [r["equivalent_diameter_mm"] / (2 * r["radius_mean_mm"]) for r in rows
          if r.get("equivalent_diameter_mm")]
    if eq:
        out["equivalent_diameter_over_traced_diameter"] = round(float(np.median(eq)), 3)
    if "semi_axes" in truth:
        a, b = truth["semi_axes"]
        ar = [r["aspect_ratio"] for r in rows if r.get("aspect_ratio")]
        mf = [(r["min_feret_mm"], r["max_feret_mm"]) for r in rows if r.get("max_feret_mm")]
        out["aspect_ratio"] = round(float(np.median(ar)), 3) if ar else None
        out["true_aspect_ratio"] = round(b / a, 3)
        out["feret_mm"] = ([round(float(np.median([x[0] for x in mf])), 3),
                            round(float(np.median([x[1] for x in mf])), 3)] if mf else None)
        out["true_feret_mm"] = [2 * b, 2 * a]
    if truth.get("deflections"):
        got = sorted(r["deflection_deg"] for r in rows
                     if r.get("deflection_deg") is not None and r["start_kind"] == "junction")
        out["deflection_deg"] = [round(x, 1) for x in got]
        out["true_deflection_deg"] = sorted(round(x, 1) for x in truth["deflections"])
    return out


def _seg_dist(P, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def _true_radius(P, segs):
    """Radius of the nearest axis segment at the closest point."""
    best_d = np.full(len(P), np.inf)
    r = np.zeros(len(P))
    for a, b, ra, rb in segs:
        a, b = np.asarray(a, float), np.asarray(b, float)
        ab = b - a
        t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
        d = np.linalg.norm(P - (a + t[:, None] * ab), axis=1)
        take = d < best_d
        best_d[take] = d[take]
        r[take] = (ra + t * (rb - ra))[take]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacing", type=float, default=0.7)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sp = a.spacing
    cases = []
    for ang in (30, 60, 90):
        cases.append((f"y{ang}", *y_phantom(ang)))
    cases += [("trifurcation", *trifurcation()), ("tapered", *tapered()), ("arc", *arc()),
              ("contact", *contact())]
    cases.append(("thin", [((0, 0, 0), (0, 0, 30), 0.8 * sp, 0.8 * sp)],
                  dict(ends=[(0, 0, 0), (0, 0, 30)], junctions=[],
                       axes=[((0, 0, 0), (0, 0, 30), 0.8 * sp, 0.8 * sp)], deflections=[])))
    results = []
    for name, segs, truth in cases:
        pts = np.concatenate([[s[0], s[1]] for s in segs]).astype(float)
        rmax = max(max(s[2], s[3]) for s in segs)
        m, geo = field_of(chain_distance(segs), pts.min(0) - rmax - 4, pts.max(0) + rmax + 4, sp)
        results.append(score(name, m, geo, truth, sp))
        print(json.dumps(results[-1]))
    f, truth = elliptic()
    m, geo = field_of(f, (-4, -7, -5), (44, 7, 5), sp)
    results.append(score("elliptic", m, geo, truth, sp))
    print(json.dumps(results[-1]))
    f, truth = torus()
    m, geo = field_of(f, (-18, -18, -6), (18, 18, 6), sp)
    results.append(score("torus", m, geo, truth, sp))
    print(json.dumps(results[-1]))
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
