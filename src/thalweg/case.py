"""One case, decoded once: its structures traced, measured, summarized and checked.

:class:`Case` holds an open store (each part's planes read into memory once) and the margins it
has decoded, so tracing, measuring and checking several structures reads the store once.
:meth:`Case.run` is the batch product (docs/deliverables.md, tier 1): the graph, the branch table,
the station profile, a per-structure summary and a QC record.

QC reports what a reader of the numbers must know before trusting them:

- ``dropped_components``: the structure's connected pieces other than the traced one (``component_count``, and
  their share of the structure's lattice points) - small fragments, or a second tree;
- ``truncated_end_count``: ends where the structure runs off the field's grid;
- ``length_outside_field_mm``: centerline length that runs where the margin is <= 0 (sampled every 0.05 mm) -
  a path cutting a corner the field does not make;
- ``unrefined_point_count``: centerline points whose inscribed-ball search found no inside point;
- ``cell_interior_join_count``: lattice joins decided inside a cell (the rare "tunnel" configurations);
- ``field_loop_count``: the loops of the structure's field (the genus of its zero surface, all pieces).
  The traced graph is a tree, so a loop in the field - an anastomosis, or two branches of one
  class in contact - is not in the graph; this says how many there are;
- per case, ``lobes``: how the lobe classes were found (``named_by``: by name, or by value in a
  store that predates haversack's naming fix), or why there are none;
- per case, ``artery_vein``: whether the arteries and veins are plausibly what they are called
  (:mod:`thalweg.plausibility`: the veins, not the arteries, run through the pulmonary vein class);
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
from .centerlines import centerline_graph, combine, radius_interval
from . import lobes as lobes_mod
from . import plausibility
from .errors import ThalwegError
from .graph import TubeGraph
from .kernel.field import sample
from .kernel.topology import surface_loops
from .measure import branch_table, pi10, strahler
from .pairing import airway_rows
from .statistics import tree_statistics
from .store import FieldStore, open_store

COARSE_MM = 3.0
WALL_OF = {"lung_airways": "lung_airways_wall"}       # a lumen and the class the model labels its wall
PAIRS = {"lung_airways": "lung_arteries"}             # an airway tree and the artery tree beside it


def outside_length(graph: TubeGraph, structure: str, margin: np.ndarray, geometry,
                   step: float = 0.05) -> float:
    """Length (mm) of the structure's centerlines lying where the margin is <= 0."""
    total = 0.0
    for e in graph.structure_edges(structure):
        p = graph.edge_points(e)
        seg = graph.edge_segments(e)[0]
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
    to its ``edge_count`` and length; ``unordered`` holds the edges that have none (they lead only to
    truncated ends). Order-based statistics are fragile across reconstructions of one scan (one
    lost thin tip can demote a subtree; docs/validation.md §2)."""
    edges = graph.structure_edges(structure)
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
    rr = r[graph.point_rows(structure)]
    rr = rr[rr > 0]
    unordered = by_order.pop(None, (0, 0.0))
    out = dict(edge_count=len(edges), node_count_by_kind=kinds,
               length_mm=round(sum(e.length_mm for e in edges), 3),
               radius_percentiles_mm=({f"p{q}": round(float(np.percentile(rr, q)), 3) for q in (10, 50, 90)}
                                      if len(rr) else {}),
               strahler_order={str(o): dict(edge_count=n, length_mm=round(L, 3))
                               for o, (n, L) in sorted(by_order.items())},
               unordered=dict(edge_count=unordered[0], length_mm=round(unordered[1], 3)))
    if rows:
        out["sectioned_branch_count"] = sum(1 for x in rows if x.get("area_mm2"))
    return out


@dataclass
class Case:
    """An open store and what has been decoded from it (see the module docstring)."""

    store: FieldStore
    margins: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    decodes: int = 0
    pi10: dict = field(default_factory=dict)
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
            g = centerline_graph(self.store, n, margin=(m, geo, ref), log=log, **kw)
            graphs.append(radius_interval(g, n, m, geo))
            self.timings[f"trace {n}"] = round(time.time() - t, 3)
        return graphs[0] if len(graphs) == 1 else combine(graphs)

    def outer(self, name: str, part: int | None, m: np.ndarray):
        """For a lumen whose wall the store labels (:data:`WALL_OF`), the lumen-or-wall margin on
        the lumen's grid; otherwise None."""
        wall = WALL_OF.get(name)
        if wall is None or not any(s.name == wall for s in self.store.structures):
            return None
        mw, _, ref = self.margin(wall, part)
        return np.maximum(m, mw) if mw.shape == m.shape else None

    def measure(self, graph: TubeGraph, step: float = 1.0, stations: bool = True):
        rows, prof = [], []
        self.pi10 = {}
        for s in graph.structures:
            t = time.time()
            m, geo, _ = self.margin(s.name, s.source.part)
            outer = self.outer(s.name, s.source.part, m)
            mine = []
            rows.extend(branch_table(graph, s.name, m, geo, step=step, stations_out=mine, outer=outer))
            if outer is not None:
                self.pi10[s.name] = dict(pi10(mine), station_count=len(mine))
            prof.extend(mine)
            self.timings[f"measure {s.name}"] = round(time.time() - t, 3)
        return rows, (prof if stations else None)

    def qc(self, graph: TubeGraph) -> dict:
        out = {}
        for s in graph.structures:
            t = time.time()
            m, geo, _ = self.margin(s.name, s.source.part)
            st = s.statistics
            sizes = st.get("component_lattice_point_counts", [])
            dropped = sizes[1:]
            total = max(sum(sizes), 1)
            edges = graph.structure_edges(s.name)
            pts = graph.point_rows(s.name)
            out[s.name] = dict(
                dropped_components=dict(component_count=len(dropped),
                                        lattice_point_share=round(sum(dropped) / total, 5),
                                        largest_lattice_point_count=dropped[0] if dropped else 0),
                truncated_end_count=sum(nd.kind == "truncated" for nd in graph.nodes
                                        if nd.structure == s.name),
                length_outside_field_mm=round(outside_length(graph, s.name, m, geo), 3),
                length_mm=round(sum(e.length_mm for e in edges), 3),
                unrefined_point_count=int((graph.radii()[pts] < 0).sum()),
                cell_interior_join_count=st.get("cell_interior_join_count"),
                field_loop_count=int(surface_loops(m)[0]))
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

    def volumes(self, graph: TubeGraph, rows: list[dict]) -> dict[str, float]:
        """Add ``volume_mm3`` to the rows: each edge's share of its structure's volume, from the
        branch partition of the field (:mod:`thalweg.partition`). Returns each structure's total."""
        from . import partition
        total = {}
        for s in graph.structures:
            t = time.time()
            m, geo, _ = self.margin(s.name, s.source.part)
            labels = partition.label_field(m, geo, partition.edge_tubes(graph, s.name),
                                           points=graph.positions()[graph.point_rows(s.name)])
            vol = partition.label_volumes(labels, geo)
            partition.volume_rows([r for r in rows if r["structure"] == s.name], vol)
            total[s.name] = round(sum(vol.values()), 2)
            self.timings[f"partition {s.name}"] = round(time.time() - t, 3)
        return total

    def run(self, names, step: float = 1.0, stations: bool = True, branch_volumes: bool = False,
            log=None, **trace_options) -> dict:
        """The batch product for ``names``: graph, rows, stations, summary and qc (see above).
        ``branch_volumes`` adds each branch's volume (:meth:`volumes`; about a third more time).
        ``trace_options`` go to :func:`thalweg.centerlines.centerline_graph` (ridge_passes, prune, root)."""
        g = self.trace(names, log=log, **trace_options)
        lobes = self.lobes(log=log)
        edge_lobe = {}
        if lobes is not None:
            g, edge_lobe = lobes_mod.annotate(g, lobes)
        rows, prof = self.measure(g, step=step, stations=stations)
        partition_volume = self.volumes(g, rows) if branch_volumes else {}
        for s in g.structures:
            if s.name in edge_lobe:
                lobes_mod.lobe_rows([r for r in rows if r["structure"] == s.name], edge_lobe[s.name])
        names_here = {s.name for s in g.structures}
        pairing = {}
        for airway, artery in PAIRS.items():
            if airway in names_here and artery in names_here:
                mine = [r for r in rows if r["structure"] == airway]
                pairing[airway] = airway_rows(g, airway, artery, mine)
        parts = sorted({st.source.part for st in g.structures if st.source.part is not None})
        acquisition = {str(p): self.acquisition(p) for p in parts}
        summary = {}
        for s in g.structures:
            sm = summarize(g, s.name, [r for r in rows if r["structure"] == s.name])
            coarse = acquisition.get(str(s.source.part), {}).get("coarse_slices")
            sm["coarse_slices"] = coarse
            sm["tree_statistics"] = tree_statistics(g, s.name)
            if s.name in partition_volume:
                sm["partition_volume_mm3"] = partition_volume[s.name]
            if s.name in self.pi10:
                sm["pi10"] = self.pi10[s.name]
            if s.name in pairing:
                sm["bronchoarterial"] = pairing[s.name]
            if s.name in edge_lobe:
                sm["lobes"] = lobes_mod.lobe_summary(g, s.name, edge_lobe[s.name],
                                                     lobes_mod.lobe_volumes(lobes))
            if coarse:
                sm["warning"] = ("coarse slices: the model drops thin vessels here, so tip counts, lengths "
                                 "and order statistics are not comparable with thin-slice cases")
                if log:
                    log(f"WARNING {s.name}: {sm['warning']}")
            summary[s.name] = sm
        qc = dict(thalweg_version=__version__, store=str(self.store.path),
                  labeling_scheme=self.store.labeling_scheme, acquisition=acquisition,
                  lobes=(dict(named_by=lobes.named_by, part=lobes.part) if lobes is not None
                         else dict(named_by=None, reason=self.lobe_error)),
                  artery_vein=plausibility.check(g, self.store),
                  structures=self.qc(g), timings_s=dict(self.timings))
        return dict(graph=g, rows=rows, stations=prof, summary=summary, qc=qc)
