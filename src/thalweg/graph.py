"""The tube graph and its file, ``.thalweg.json`` (format version 0.1).

One file holds one or more **structures** (arteries, veins, airways, a colon...), each a graph of
**nodes** joined by **edges**; every edge is a polyline whose samples live in one shared, column-
wise **point table** (``points.position[i]``, ``points.radius[i]``, optional named columns), and
the edge names its slice of it (``point_range: [start, stop)``).

Invariants the model enforces (a document that breaks one does not validate):

- ids are positions: ``nodes[i].id == i``, ``edges[i].id == i``; structure names are unique;
- an edge's first and last samples sit on its start and end nodes (within 1e-6 mm), and both
  nodes belong to the edge's structure, as do its structure's roots;
- point ranges are polylines of at least two samples, do not overlap, and cover the table;
- ``length_mm`` is the polyline's length (within 1e-6 mm);
- kinds agree with degree: ``tip`` and ``truncated`` have degree 1, ``joint`` 2, ``junction`` 3 or
  more (``root`` any);
- provenance: ``bridged`` carries ``gap_mm`` and ``reason``; the other methods carry neither;
- numbers are finite; ``length_mm`` >= 0; a radius is >= 0 or exactly -1 (the tracer's "no inside
  point found");
- ``version`` is ``major.minor``.

What the model allows, because tubes are not all vessel trees (docs/port-plan.md):

- **node kinds** ``root``, ``junction``, ``tip`` (a free end), ``truncated`` (an end cut off by the
  edge of the field, not a tip), ``joint`` (degree 2: where an edge is split for another reason);
- **cycles**: nothing requires a tree; ``Structure.roots`` may be empty or hold one node per
  component. Consumers that need a tree (Strahler order, the branch table's angles, the vmtk
  adapter, SWC) call :meth:`TubeGraph.tree`, which raises a clear error on a cycle;
- **orientation**: in a structure that is a tree with one root, every edge points away from the
  root (start node nearer the root). :meth:`TubeGraph.tree` checks it;
- **provenance per edge**: how the edge was made (``field``: connected in the field; ``voxel``:
  the 26-connected comparison graph; ``bridged``: a gap closed, with its length and reason);
- **extra point columns** (section area, radius interval, layer thickness...) under
  ``points.columns``, each as long as the table (``None`` where undefined).

Serialized field names are whole words. Coordinates are LPS millimeters. Version 0.x may change
without migration; 1.0 will be the first promise.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .errors import ThalwegError

FORMAT = "thalweg"
FORMAT_VERSION = "0.1"
SUFFIX = ".thalweg.json"
NODE_TOLERANCE_MM = 1e-6

Vec3 = tuple[float, float, float]
NodeKind = Literal["root", "junction", "tip", "truncated", "joint"]
DEGREE = {"tip": (1, 1), "truncated": (1, 1), "joint": (2, 2), "junction": (3, None)}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Grid(_Model):
    """Where the source field's samples sit: ``world = origin + index @ directions`` (array order)."""

    shape: tuple[int, int, int]
    directions: tuple[Vec3, Vec3, Vec3]
    origin: Vec3


class Node(_Model):
    id: int
    kind: NodeKind
    position: Vec3
    structure: str
    attributes: dict[str, Any] = Field(default_factory=dict)


class Provenance(_Model):
    method: Literal["field", "voxel", "bridged"]
    gap_mm: float | None = Field(None, description="bridged only: the length of the closed gap")
    reason: str | None = Field(None, description="bridged only: why the gap was closed")

    @model_validator(mode="after")
    def _bridged(self):
        if self.method == "bridged" and (self.gap_mm is None or self.gap_mm < 0 or not self.reason):
            raise ValueError("a bridged edge needs gap_mm >= 0 and a reason")
        if self.method != "bridged" and (self.gap_mm is not None or self.reason is not None):
            raise ValueError(f"a {self.method} edge carries no gap_mm or reason")
        return self


class Edge(_Model):
    id: int
    structure: str
    start_node: int
    end_node: int
    point_range: tuple[int, int] = Field(description="[start, stop) rows of the point table")
    length_mm: float = Field(ge=0)
    provenance: Provenance
    tracer_branch: int | None = Field(
        None, description="the tracer's branch this edge was cut from; the tracer starts at its deepest "
                          "point, so this numbering does not follow a tree re-rooted at its inlet")
    tracer_generation: int | None = Field(
        None, description="how many tracer branches that branch hangs from, counted from the tracer's "
                          "deepest point (not the number of bifurcations between the edge and the root)")
    attributes: dict[str, Any] = Field(default_factory=dict)


class Points(_Model):
    position: list[Vec3]
    radius: list[float] = Field(description="inscribed-ball radius, mm (-1 where no inside point was found)")
    columns: dict[str, list[float | int | None]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _lengths(self):
        n = len(self.position)
        if len(self.radius) != n:
            raise ValueError(f"points.radius has {len(self.radius)} rows, position {n}")
        r = np.asarray(self.radius, dtype=float)
        if len(r) and ((r < 0) & (r != -1.0)).any():
            raise ValueError("a radius is negative and not the sentinel -1")
        for k, v in self.columns.items():
            if len(v) != n:
                raise ValueError(f"points.columns[{k!r}] has {len(v)} rows, position {n}")
        return self


class Source(_Model):
    store: str | None = None
    labeling_scheme: str | None = None
    part: int | None = None
    label_value: int | None = None
    grid: Grid | None = None


class Structure(_Model):
    name: str
    source: Source = Field(default_factory=Source)
    roots: list[int] = Field(default_factory=list)
    method: str = Field(description="how the centerlines were made, e.g. 'thalweg.trace'")
    parameters: dict[str, Any] = Field(default_factory=dict)
    statistics: dict[str, Any] = Field(default_factory=dict)


@dataclass
class Tree:
    """One structure as a rooted tree: ``children[node]`` edges leaving it, ``parent[node]`` the edge
    arriving (absent at the root), edges in breadth-first order from the root."""

    root: int
    children: dict[int, list[int]]
    parent: dict[int, int]
    order: list[int]


class TubeGraph(_Model):
    """A ``.thalweg.json`` document."""

    format: Literal["thalweg"] = FORMAT
    version: str = Field(FORMAT_VERSION, pattern=r"^\d+\.\d+$")
    units: Literal["mm"] = "mm"
    space: Literal["LPS"] = "LPS"
    created_by: str | None = None
    structures: list[Structure] = Field(default_factory=list)
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    points: Points = Field(default_factory=lambda: Points(position=[], radius=[]))

    @model_validator(mode="after")
    def _consistent(self):
        n = len(self.points.position)
        names = [s.name for s in self.structures]
        if len(set(names)) != len(names):
            raise ValueError("structure names are not unique")
        for i, nd in enumerate(self.nodes):
            if nd.id != i:
                raise ValueError(f"nodes[{i}] has id {nd.id}: ids must be positions")
            if nd.structure not in names:
                raise ValueError(f"node {nd.id} names unknown structure {nd.structure!r}")
        pos = np.asarray(self.points.position, dtype=float).reshape(-1, 3)
        used = np.zeros(n, bool)
        deg = [0] * len(self.nodes)
        for i, e in enumerate(self.edges):
            if e.id != i:
                raise ValueError(f"edges[{i}] has id {e.id}: ids must be positions")
            a, b = e.point_range
            if not (0 <= a < b <= n) or b - a < 2:
                raise ValueError(f"edge {e.id}: point_range [{a}, {b}) is not a polyline in a table of {n}")
            if used[a:b].any():
                raise ValueError(f"edge {e.id}: point_range [{a}, {b}) overlaps another edge's")
            used[a:b] = True
            if e.structure not in names:
                raise ValueError(f"edge {e.id} names unknown structure {e.structure!r}")
            for end, row in ((e.start_node, a), (e.end_node, b - 1)):
                if not 0 <= end < len(self.nodes):
                    raise ValueError(f"edge {e.id} joins node {end}, which does not exist")
                nd = self.nodes[end]
                if nd.structure != e.structure:
                    raise ValueError(f"edge {e.id} ({e.structure}) joins node {end} of {nd.structure}")
                if np.abs(pos[row] - np.asarray(nd.position)).max() > NODE_TOLERANCE_MM:
                    raise ValueError(f"edge {e.id}'s end sample {row} is not at node {end}")
                deg[end] += 1
            poly = float(np.linalg.norm(np.diff(pos[a:b], axis=0), axis=1).sum())
            if abs(e.length_mm - poly) > 1e-6 + 1e-9 * poly:
                raise ValueError(f"edge {e.id}: length_mm {e.length_mm} is not its polyline's length {poly}")
        if n and not used.all():
            raise ValueError(f"{int((~used).sum())} point rows belong to no edge")
        for nd in self.nodes:
            lo, hi = DEGREE.get(nd.kind, (0, None))
            if deg[nd.id] < lo or (hi is not None and deg[nd.id] > hi):
                raise ValueError(f"node {nd.id} is a {nd.kind} of degree {deg[nd.id]}")
        for s in self.structures:
            for r in s.roots:
                if not 0 <= r < len(self.nodes) or self.nodes[r].structure != s.name:
                    raise ValueError(f"structure {s.name!r}: root {r} is not one of its nodes")
        return self

    # -- numpy views -----------------------------------------------------------------------
    def positions(self) -> np.ndarray:
        return np.asarray(self.points.position, dtype=np.float64).reshape(-1, 3)

    def radii(self) -> np.ndarray:
        return np.asarray(self.points.radius, dtype=np.float64)

    def edge_points(self, edge: Edge | int) -> np.ndarray:
        """An edge's samples (``edge`` an Edge or an edge id; ids are positions)."""
        e = self.edges[edge] if isinstance(edge, (int, np.integer)) else edge
        a, b = e.point_range
        return np.asarray(self.points.position[a:b], dtype=np.float64)

    def edge_radius(self, edge: Edge | int) -> np.ndarray:
        e = self.edges[edge] if isinstance(edge, (int, np.integer)) else edge
        a, b = e.point_range
        return np.asarray(self.points.radius[a:b], dtype=np.float64)

    def structure_edges(self, name: str) -> list[Edge]:
        """The edges of one structure, in id order."""
        return [e for e in self.edges if e.structure == name]

    def point_rows(self, name: str) -> np.ndarray:
        """Row indices into the point table of one structure's samples (its edges' ranges)."""
        ranges = [np.arange(*e.point_range) for e in self.structure_edges(name)]
        return np.concatenate(ranges) if ranges else np.zeros(0, int)

    def edge_segments(self, edge: Edge | int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """An edge's polyline segments: ``(length, midpoint, radius)`` per segment, the radius the
        mean of the traced radii at its two ends (<= 0 where either end has none)."""
        p, r = self.edge_points(edge), self.edge_radius(edge)
        rm = np.where((r[1:] > 0) & (r[:-1] > 0), 0.5 * (r[1:] + r[:-1]), -1.0)
        return np.linalg.norm(np.diff(p, axis=0), axis=1), 0.5 * (p[1:] + p[:-1]), rm

    def structure(self, name: str) -> Structure:
        for s in self.structures:
            if s.name == name:
                return s
        raise ThalwegError(
            f"no structure {name!r}; the graph has {', '.join(s.name for s in self.structures)}")

    def degree(self) -> dict[int, int]:
        deg = {nd.id: 0 for nd in self.nodes}
        for e in self.edges:
            deg[e.start_node] += 1
            deg[e.end_node] += 1
        return deg

    def tree(self, structure: str) -> Tree:
        """The structure as a tree rooted at its single root, edges pointing away from it.

        Raises ThalwegError if it has no single root, is not connected, has a cycle, or has an edge
        pointing toward the root - consumers that need a tree call this instead of assuming one."""
        s = self.structure(structure)
        if len(s.roots) != 1:
            raise ThalwegError(f"{structure!r} has {len(s.roots)} roots; a tree needs one")
        edges = [e for e in self.edges if e.structure == structure]
        nodes = {nd.id for nd in self.nodes if nd.structure == structure}
        if len(edges) != len(nodes) - 1:
            raise ThalwegError(f"{structure!r} is not a tree: {len(edges)} edges for {len(nodes)} nodes "
                               "(a cycle, or pieces not connected)")
        children: dict[int, list[int]] = {}
        parent: dict[int, int] = {}
        for e in edges:
            children.setdefault(e.start_node, []).append(e.id)
            if e.end_node in parent:
                raise ThalwegError(f"{structure!r}: node {e.end_node} has two arriving edges "
                                   "(a cycle, or edges not oriented away from the root)")
            parent[e.end_node] = e.id
        root = s.roots[0]
        if root in parent:
            raise ThalwegError(f"{structure!r}: an edge arrives at the root; edges must point away from it")
        order, frontier, seen = [], [root], {root}
        while frontier:
            nxt = []
            for n in frontier:
                for eid in children.get(n, []):
                    order.append(eid)
                    end = self.edges[eid].end_node
                    if end in seen:
                        raise ThalwegError(f"{structure!r} has a cycle through node {end}")
                    seen.add(end)
                    nxt.append(end)
            frontier = nxt
        if len(order) != len(edges):
            raise ThalwegError(f"{structure!r}: {len(edges) - len(order)} edges are not reachable from the "
                               "root along edge directions")
        return Tree(root, children, parent, order)

    # -- files -----------------------------------------------------------------------------
    def dumps(self) -> str:
        return self.model_dump_json(exclude_none=True)

    def write(self, path) -> Path:
        """Write ``path`` (``.thalweg.json``, or ``.thalweg.json.gz`` gzipped). The document is
        validated again first, so a model changed in place after construction cannot write a file
        that :meth:`read` would refuse."""
        path = Path(path)
        try:
            type(self).model_validate(self.model_dump())
        except ValidationError as e:
            raise ThalwegError(f"not writing an invalid graph:\n{e}") from e
        data = self.dumps().encode()
        if path.suffix == ".gz":
            with gzip.open(path, "wb", compresslevel=6) as f:
                f.write(data)
        else:
            path.write_bytes(data)
        return path

    @classmethod
    def read(cls, path) -> "TubeGraph":
        path = Path(path)
        try:
            raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
            doc = json.loads(raw)
        except (OSError, ValueError) as e:
            raise ThalwegError(f"cannot read {path}: {e}") from e
        if not isinstance(doc, dict) or doc.get("format") != FORMAT:
            raise ThalwegError(f"{path} is not a thalweg graph "
                               f"(format {doc.get('format') if isinstance(doc, dict) else None!r})")
        major = str(doc.get("version", "")).split(".")[0]
        if major != FORMAT_VERSION.split(".")[0]:
            raise ThalwegError(
                f"{path}: format version {doc.get('version')!r}; this reader knows {FORMAT_VERSION}")
        try:
            return cls.model_validate(doc)
        except ValidationError as e:
            raise ThalwegError(f"{path} is not a valid thalweg graph:\n{e}") from e


def json_schema() -> dict:
    """The JSON Schema of ``.thalweg.json`` (generated from the models above)."""
    s = TubeGraph.model_json_schema()
    s["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    s["title"] = f"thalweg tube graph, format {FORMAT_VERSION}"
    return s
