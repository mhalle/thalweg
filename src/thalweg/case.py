"""One case, decoded once: its structures traced, measured, summarized and checked.

:class:`Case` holds an open store (each part's planes read into memory once) and the margins it
has decoded, so tracing, measuring and checking several structures reads the store once.
:meth:`Case.run` is the batch product (docs/deliverables.md, tier 1): the graph, the branch table,
the station profile, a per-structure summary and a QC record.

QC reports what a reader of the numbers must know before trusting them:

- ``dropped_components``: the structure's connected pieces other than the traced one (count and
  their share of the structure's lattice points) - small fragments, or a second tree;
- ``truncated_ends``: ends where the structure runs off the field's grid;
- ``outside_mm``: centerline length that runs where the margin is <= 0 (sampled every 0.05 mm) -
  a path cutting a corner the field does not make;
- ``unrefined_points``: centerline points whose inscribed-ball search found no inside point;
- ``cell_interior_joins``: lattice joins decided inside a cell (the rare "tunnel" configurations);
- ``field_loops``: the loops of the structure's field (the genus of its zero surface, all pieces).
  The traced graph is a tree, so a loop in the field - an anastomosis, or two branches of one
  class in contact - is not in the graph; this says how many there are;
- per case, ``lobes``: how the lobe classes were found (``named_by``: by name, or by value in a
  store that predates haversack's naming fix), or why there are none;
- per case, ``acquisition``: the source image's spacing (from the part's frame record) and
  ``coarse_slices`` when its largest spacing exceeds 3 mm - at 3.75-5 mm slices the model drops
  thin vessels rather than widening them, so thin-order statistics are invalid there
  (docs/validation.md §2). The store records spacing, not slice thickness: a thin series
  resampled to wide spacing, or thick slices at close increments, can fool it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import __version__
from .centerlines import centerline_graph, combine
from . import lobes as lobes_mod
from .errors import ThalwegError
from .graph import TubeGraph
from .kernel.field import sample
from .kernel.topology import surface_loops
from .measure import branch_table, strahler
from .store import FieldStore, open_store

COARSE_MM = 3.0


def outside_length(graph: TubeGraph, structure: str, margin: np.ndarray, geometry,
                   step: float = 0.05) -> float:
    """Length (mm) of the structure's centerlines lying where the margin is <= 0."""
    total = 0.0
    for e in graph.edges:
        if e.structure != structure:
            continue
        p = graph.edge_points(e)
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        n = np.maximum(np.ceil(seg / step).astype(int), 1)
        t = np.concatenate([np.arange(k) / k for k in n])
        a = np.repeat(p[:-1], n, axis=0)
        b = np.repeat(p[1:], n, axis=0)
        q = a + t[:, None] * (b - a)
        w = np.repeat(seg / n, n)
        total += float(w[sample(margin, geometry, q) <= 0].sum())
    return total


def summarize(graph: TubeGraph, structure: str, rows: list[dict] | None = None) -> dict:
    """Counts, lengths and Strahler orders of one structure. ``strahler_order`` maps each order
    to its edge count and length; ``unordered`` holds the edges that have none (they lead only to
    truncated ends). Order-based statistics are fragile across reconstructions of one scan (one
    lost thin tip can demote a subtree; docs/validation.md §2)."""
    edges = [e for e in graph.edges if e.structure == structure]
    kinds = {}
    for nd in graph.nodes:
        if nd.structure == structure:
            kinds[nd.kind] = kinds.get(nd.kind, 0) + 1
    order = strahler(graph, structure)
    by_order = {}
    for e in edges:
        o = order[e.id]
        n, L = by_order.get(o, (0, 0.0))
        by_order[o] = (n + 1, L + e.length_mm)
    r = graph.radii()
    pts = np.concatenate([np.arange(*e.point_range) for e in edges]) if edges else np.zeros(0, int)
    rr = r[pts]
    rr = rr[rr > 0]
    unordered = by_order.pop(None, (0, 0.0))
    out = dict(edges=len(edges), nodes=kinds, length_mm=round(sum(e.length_mm for e in edges), 3),
               radius_percentiles_mm=({f"p{q}": round(float(np.percentile(rr, q)), 3) for q in (10, 50, 90)}
                                      if len(rr) else {}),
               strahler_order={str(o): dict(edges=n, length_mm=round(L, 3))
                               for o, (n, L) in sorted(by_order.items())},
               unordered=dict(edges=unordered[0], length_mm=round(unordered[1], 3)))
    if rows:
        out["sectioned_branches"] = sum(1 for x in rows if x.get("area_mm2"))
    return out


@dataclass
class Case:
    """An open store and what has been decoded from it (see the module docstring)."""

    store: FieldStore
    margins: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    decodes: int = 0
    lobe_error: str | None = None

    @classmethod
    def open(cls, path) -> "Case":
        return cls(open_store(path))

    def margin(self, name: str, part: int | None = None):
        """``(margin, geometry, ref)`` of a structure, decoded once per (name, resolved part)."""
        ref = self.store.ref(name, part)
        key = (name, ref.part)
        if key not in self.margins:
            t = time.time()
            self.margins[key] = self.store.margin(name, ref.part)
            self.timings[f"decode {name} (part {ref.part})"] = round(time.time() - t, 3)
            self.decodes += 1
        return self.margins[key]

    def trace(self, names, log=None, **kw) -> TubeGraph:
        if len(set(names)) != len(names):
            raise ThalwegError(f"a structure is named twice: {list(names)}")
        graphs = []
        for n in names:
            m, geo, ref = self.margin(n)
            t = time.time()
            graphs.append(centerline_graph(self.store, n, margin=(m, geo, ref), log=log, **kw))
            self.timings[f"trace {n}"] = round(time.time() - t, 3)
        return graphs[0] if len(graphs) == 1 else combine(graphs)

    def measure(self, graph: TubeGraph, step: float = 1.0, stations: bool = True):
        rows, prof = [], ([] if stations else None)
        for s in graph.structures:
            t = time.time()
            m, geo, _ = self.margin(s.name, s.source.part)
            rows.extend(branch_table(graph, s.name, m, geo, step=step, stations_out=prof))
            self.timings[f"measure {s.name}"] = round(time.time() - t, 3)
        return rows, prof

    def qc(self, graph: TubeGraph) -> dict:
        out = {}
        for s in graph.structures:
            t = time.time()
            m, geo, _ = self.margin(s.name, s.source.part)
            st = s.statistics
            sizes = st.get("component_sizes", [])
            dropped = sizes[1:]
            total = max(sum(sizes), 1)
            edges = [e for e in graph.edges if e.structure == s.name]
            pts = np.concatenate([np.arange(*e.point_range) for e in edges]) if edges else np.zeros(0, int)
            out[s.name] = dict(
                dropped_components=dict(count=len(dropped), lattice_share=round(sum(dropped) / total, 5),
                                        largest=dropped[0] if dropped else 0),
                truncated_ends=sum(nd.kind == "truncated" for nd in graph.nodes if nd.structure == s.name),
                outside_mm=round(outside_length(graph, s.name, m, geo), 3),
                length_mm=round(sum(e.length_mm for e in edges), 3),
                unrefined_points=int((graph.radii()[pts] < 0).sum()),
                cell_interior_joins=st.get("cell_interior_joins"),
                field_loops=int(surface_loops(m)[0]))
            self.timings[f"qc {s.name}"] = round(time.time() - t, 3)
        return out

    def lobes(self, log=None):
        """The store's lobe fields (thalweg.lobes), or None - with ``lobe_error`` saying why - when
        the store names no lobes (a store that is not a lung cascade)."""
        t = time.time()
        try:
            f = lobes_mod.lobe_fields(self.store)
        except ThalwegError as e:
            self.lobe_error = str(e)
            if log:
                log(f"no lobes: {e}")
            return None
        self.timings["decode lobes"] = round(time.time() - t, 3)
        return f

    def acquisition(self, part: int) -> dict:
        frame = self.store.field(part).frame or {}
        spacing = (frame.get("source") or {}).get("spacing")
        if not spacing:
            return dict(source_spacing_mm=None, coarse_slices=None)
        return dict(source_spacing_mm=[float(v) for v in spacing],
                    coarse_slices=bool(max(spacing) > COARSE_MM))

    def run(self, names, step: float = 1.0, stations: bool = True, log=None, **trace_options) -> dict:
        """The batch product for ``names``: graph, rows, stations, summary and qc (see above).
        ``trace_options`` go to :func:`thalweg.centerlines.centerline_graph` (ridge_passes, prune)."""
        g = self.trace(names, log=log, **trace_options)
        lobes = self.lobes(log=log)
        edge_lobe = {}
        if lobes is not None:
            g, edge_lobe = lobes_mod.annotate(g, lobes)
        rows, prof = self.measure(g, step=step, stations=stations)
        for s in g.structures:
            if s.name in edge_lobe:
                lobes_mod.lobe_rows([r for r in rows if r["structure"] == s.name], edge_lobe[s.name])
        parts = sorted({st.source.part for st in g.structures if st.source.part is not None})
        acquisition = {str(p): self.acquisition(p) for p in parts}
        summary = {}
        for s in g.structures:
            sm = summarize(g, s.name, [r for r in rows if r["structure"] == s.name])
            coarse = acquisition.get(str(s.source.part), {}).get("coarse_slices")
            sm["coarse_slices"] = coarse
            if s.name in edge_lobe:
                sm["lobes"] = lobes_mod.lobe_summary(g, s.name, edge_lobe[s.name],
                                                     lobes_mod.lobe_volumes(lobes))
            if coarse:
                sm["warning"] = ("coarse slices: the model drops thin vessels here, so tip counts, lengths "
                                 "and order statistics are not comparable with thin-slice cases")
                if log:
                    log(f"WARNING {s.name}: {sm['warning']}")
            summary[s.name] = sm
        qc = dict(thalweg=__version__, store=str(self.store.path), labeling_scheme=self.store.labeling_scheme,
                  acquisition=acquisition,
                  lobes=(dict(named_by=lobes.named_by, part=lobes.part) if lobes is not None
                         else dict(named_by=None, reason=self.lobe_error)),
                  structures=self.qc(g), timings_s=dict(self.timings))
        return dict(graph=g, rows=rows, stations=prof, summary=summary, qc=qc)
