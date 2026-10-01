"""Export: vmtk-compatible centerlines as VTP, and the structure's surface with named caps.

The surface is the one place thalweg makes a mesh (docs/vmtk-successor.md §5.2): the zero set of
the structure's margin, cut flat at chosen ends so a CFD tool gets open boundaries it can name.

**The wall.** Marching cubes (Lewiner) on the margin at level 1e-5, not 0: a ranked store's
margins are quantized, so exact zeros occur, and a zero grid value makes degenerate triangles
whose removal leaves holes (the same reason as :func:`thalweg.kernel.topology.surface_loops`).
The faces are wound so their normals point OUT of the structure on either grid handedness
(skimage's winding follows the index frame; a right-handed index->world map, det > 0, keeps it
pointing in, so it is reversed there).

**Ends.** :func:`end_cuts` places a cut at each end of the requested kinds: ``tip`` (a free end),
``truncated`` (an end the grid's edge cut off, including a root on the grid's edge) and ``root``
(a degree-1 root: the inlet). An end it cannot cut is returned as a :class:`Skipped` with its
reason - every end of those kinds is either a cut or a skip.

**Cuts.** A cut is a plane ``cut_mm`` back from its end along its edge, normal to the edge's
direction there, pointing out of the structure. The zero set is split exactly along each plane -
only the triangles crossing it around this branch's section (found from a ball of ``ball`` x the
local radius, grown along the section's loop), so another vessel crossing the plane elsewhere is
untouched - and the inner hole is capped by ear-clipping its loop in the plane. Split points are
shared between inner neighbors, so the result stays watertight; the part beyond each cut comes
loose as its own component and only the main component is kept. A cut whose far side stays
attached (a loop or a contact reaching around the plane) would leave its far-side cap inside the
surface, a zero-thickness double wall: such a cut is undone and reported, never kept. (Cutting
the FIELD instead, min(m, -slope * s), gives flat caps too, but the plane-wall rim becomes a
one-cell chamfer: on a 0.7 mm grid about half a small cap's area is not on the plane.)

**Checks.** The written surface is closed, consistently oriented and manifold (every edge in two
faces, no directed edge twice, one fan per vertex) with a positive volume; :func:`surface` raises
ThalwegError rather than return anything else (:func:`mesh_defects` says what is wrong).

**Flow extensions.** :func:`flow_extensions` replaces caps by straight tubes that morph each
cap's ring into a circle and end in a flat cap (vmtk's ``vmtkflowextensions``), and
:func:`boundary_reference_system` gives a cap's ring barycenter, mean radius and normal (vmtk's
``vmtkboundaryreferencesystems``); both work on the capped mesh, so the domain stays closed.

**Names.** Every face gets a ``BoundaryId``: 0 the wall, k >= 1 the k-th cap (``Mesh.caps[k -
1]``, contiguous: a skipped end has no id). :func:`write_vtp_mesh` writes the .vtp and a JSON
sidecar ``<mesh>.boundaries.json``: each boundary's id, name, end kind, node, plane center,
outward normal, inscribed radius, area, area centroid, ring barycenter and ring mean radius (and
its flow extension's length and start, if it has one), and the skipped ends with their reasons.

The VTP writers produce VTK XML PolyData (ASCII) with no VTK dependency. Integer arrays are
written as Int32 (vtkIntArray): vmtk's filters read GroupIds, Blanking, CenterlineIds and TractIds
as vtkIntArray and refuse anything else.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .errors import ThalwegError
from .graph import TubeGraph
from .kernel.field import crossings, sample, to_world
from .vmtk.centerlines import Centerlines

LEVEL = 1e-5                     # the iso level: just inside the zero set (exact zeros leave holes)
END_KINDS = ("tip", "truncated", "root")


# -- VTK XML PolyData --------------------------------------------------------------------------

def _data_array(name, a, kind="Float64"):
    a = np.asarray(a)
    comps = 1 if a.ndim == 1 else a.shape[1]
    if kind.startswith("Int"):
        body = " ".join(str(int(v)) for v in a.ravel())
    else:
        body = " ".join(repr(float(v)) for v in a.ravel())
    nm = f' Name="{name}"' if name else ""
    return f'<DataArray type="{kind}"{nm} NumberOfComponents="{comps}" format="ascii">{body}</DataArray>'


def _kind(name, a):
    """Int32 for integer data (ids: vmtk wants vtkIntArray), Float64 otherwise."""
    a = np.asarray(a)
    if np.issubdtype(a.dtype, np.integer) or a.dtype == bool:
        if a.size and (a.min() < np.iinfo(np.int32).min or a.max() > np.iinfo(np.int32).max):
            raise ThalwegError(f"array {name!r} does not fit Int32 (range {a.min()}..{a.max()})")
        return "Int32"
    return "Float64"


def _polydata_xml(points, lines=None, polys=None, point_data=None, cell_data=None) -> str:
    points = np.asarray(points, float).reshape(-1, 3)
    parts = ['<?xml version="1.0"?>',
             '<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" header_type="UInt64">',
             "<PolyData>"]
    nl = len(lines) if lines is not None else 0
    npoly = len(polys) if polys is not None else 0
    parts.append(f'<Piece NumberOfPoints="{len(points)}" NumberOfVerts="0" NumberOfLines="{nl}" '
                 f'NumberOfStrips="0" NumberOfPolys="{npoly}">')
    parts.append("<Points>" + _data_array(None, points) + "</Points>")
    for tag, cells in (("Lines", lines), ("Polys", polys)):
        if cells is None or len(cells) == 0:
            continue
        if isinstance(cells, np.ndarray):
            conn = cells.ravel()
            offs = np.arange(1, len(cells) + 1) * cells.shape[1]
        else:
            conn = np.concatenate([np.asarray(c, np.int64) for c in cells])
            offs = np.cumsum([len(c) for c in cells])
        parts.append(f"<{tag}>" + _data_array("connectivity", conn, "Int64")
                     + _data_array("offsets", offs, "Int64") + f"</{tag}>")
    for tag, data in (("PointData", point_data), ("CellData", cell_data)):
        if data:
            parts.append(f"<{tag}>" + "".join(_data_array(k, v, _kind(k, v)) for k, v in data.items())
                         + f"</{tag}>")
    parts += ["</Piece>", "</PolyData>", "</VTKFile>"]
    return "\n".join(parts) + "\n"


def write_vtp_centerlines(cl: Centerlines, path) -> Path:
    """vmtk-convention centerlines (``thalweg.vmtk.Centerlines``) as a .vtp vmtk and Slicer read."""
    path = Path(path)
    path.write_text(_polydata_xml(cl.points, lines=cl.cells, point_data=cl.point_data,
                                  cell_data=cl.cell_data))
    return path


# -- ends and cuts ----------------------------------------------------------------------------------

@dataclass
class Cut:
    center: np.ndarray
    normal: np.ndarray            # unit, pointing out of the kept structure
    radius: float                 # the centerline's inscribed radius at the cut, mm
    name: str
    edge: int | None = None       # the graph edge it cuts: only that branch's triangles are clipped
    node: int | None = None       # the end node it caps
    kind: str | None = None       # tip, truncated or root


@dataclass
class Skipped:
    """An end that got no cap, and why."""
    name: str
    reason: str
    node: int | None = None
    kind: str | None = None

    def __str__(self):
        return f"{self.name}: {self.reason}"


@dataclass
class Mesh:
    vertices: np.ndarray          # (V, 3) world mm
    faces: np.ndarray             # (F, 3), wound outward
    boundary: np.ndarray          # (F,) 0 wall, k the cap caps[k - 1]
    names: list[str] = field(default_factory=list)       # names[k]; names[0] = "wall"
    skipped: list[Skipped] = field(default_factory=list)  # ends with no cap, and why
    caps: list[Cut] = field(default_factory=list)         # caps[k - 1]: the cut behind BoundaryId k
    extensions: dict[int, dict] = field(default_factory=dict)  # BoundaryId -> its flow extension
    point_data: dict[str, np.ndarray] = field(default_factory=dict)  # named per-vertex arrays for the .vtp


def wall_slope(margin: np.ndarray, geometry) -> float:
    """The margin's median gradient magnitude at its zero crossings, logit/mm."""
    X = crossings(margin, geometry)
    if len(X) > 20000:                                                  # a fixed subsample is plenty
        X = X[np.random.default_rng(0).choice(len(X), 20000, replace=False)]
    h = 0.1
    g = np.stack([(sample(margin, geometry, X + h * e) - sample(margin, geometry, X - h * e)) / (2 * h)
                  for e in np.eye(3)], 1)                               # central differences, world mm
    return float(np.median(np.linalg.norm(g, axis=1)))


def end_kind(node, degree: int) -> str | None:
    """The end kind of a node (tip, truncated, root) or None if it is not an end (degree != 1).
    A degree-1 root on the grid's edge (``attributes.on_grid_boundary``) is a truncated end."""
    if degree != 1:
        return None
    if node.kind in ("tip", "truncated"):
        return node.kind
    if node.kind == "root":
        return "truncated" if node.attributes.get("on_grid_boundary") else "root"
    return None


def ends(graph: TubeGraph, structure: str, kinds=("tip", "truncated")) -> list[tuple[int, str]]:
    """(node id, end kind) of every end of ``structure`` whose kind is in ``kinds``."""
    bad = set(kinds) - set(END_KINDS)
    if bad:
        raise ThalwegError(f"unknown end kind(s) {sorted(bad)}; ends are {', '.join(END_KINDS)}")
    deg = graph.degree()
    out = []
    for nd in graph.nodes:
        if nd.structure == structure:
            k = end_kind(nd, deg[nd.id])
            if k in kinds:
                out.append((nd.id, k))
    return out


def end_cuts(graph: TubeGraph, structure: str, cut_mm: float | None = None, kinds=("tip", "truncated"),
             min_radius: float = 0.0, margin: np.ndarray | None = None,
             geometry=None) -> tuple[list[Cut], list[Skipped]]:
    """A cut at each end of ``kinds`` (see :func:`ends`), ``cut_mm`` back along its edge, the
    normal the edge's direction there, out of the structure. Returns (cuts, skipped): every end of
    those kinds is in exactly one of them.

    Default ``cut_mm``: twice the branch's radius near its end (the median over the edge's last
    half), at least 1 mm. A traced end reaches into the rounded end of the tube, whose sections are
    smaller than the tube's; a cut within one radius of the end would cap a partial section.
    Skipped, with the reason: an end whose cut would lie closer than ``cut_mm`` to the edge's other
    end (a junction: its section would run into the neighboring branches there); one whose radius
    at the cut is below ``min_radius``; and, given the ``margin`` (and its ``geometry``), one whose
    cut center is not inside the structure (margin <= 0: the centerline left the field there)."""
    nodes = {nd.id: nd for nd in graph.nodes}
    want = dict(ends(graph, structure, kinds))
    cuts, skipped = [], []
    for e in graph.edges:
        if e.structure != structure:
            continue
        for end, flip in ((e.end_node, False), (e.start_node, True)):
            if end not in want:
                continue
            nd, kind = nodes[end], want[end]
            name = f"{structure} {kind} {nd.id}"
            p, r = graph.edge_points(e), graph.edge_radius(e)
            if flip:
                p, r = p[::-1], r[::-1]
            s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
            pos = r[len(r) // 2:]
            pos = pos[pos > 0]
            back = cut_mm if cut_mm is not None else max(2.0 * float(np.median(pos)) if len(pos) else 1.0,
                                                         1.0)
            at = s[-1] - back
            # the cut must clear the edge's other end (a junction, usually) by ``back`` too - by
            # default two radii - or its section runs into the neighboring branches there
            if at < back:
                skipped.append(Skipped(name, f"its edge is too short: {s[-1]:.1f} mm, and the cut sits "
                                             f"{back:.1f} mm from the end and must clear the edge's other "
                                             "end by as much", nd.id, kind))
                continue
            ok = r > 0
            rad = float(np.interp(at, s[ok], r[ok])) if ok.any() else 0.5
            if rad < min_radius:
                skipped.append(Skipped(name, f"the radius at the cut, {rad:.2f} mm, is below {min_radius} mm",
                                       nd.id, kind))
                continue
            c = np.array([np.interp(at, s, p[:, a]) for a in range(3)])
            if margin is not None:
                v = float(sample(margin, geometry, c[None])[0])
                if v <= 0:
                    why = (f"the cut's center is outside the structure (margin {v:.2f}): "
                           "the centerline leaves it there")
                    skipped.append(Skipped(name, why, nd.id, kind))
                    continue
            q = np.array([np.interp(min(at + 1.0, s[-1]), s, p[:, a]) for a in range(3)])
            o = np.array([np.interp(max(at - 1.0, 0.0), s, p[:, a]) for a in range(3)])
            n = q - o
            n /= np.linalg.norm(n)
            cuts.append(Cut(c, n, rad, name, edge=e.id, node=nd.id, kind=kind))
    return cuts, skipped


# -- the surface -----------------------------------------------------------------------------------

class _Rows:
    """An array grown in blocks (amortized appends; ``.a`` is the live view)."""

    def __init__(self, a: np.ndarray):
        self._buf = np.array(a)
        self.n = len(a)

    @property
    def a(self) -> np.ndarray:
        return self._buf[:self.n]

    def extend(self, rows) -> None:
        rows = np.asarray(rows, self._buf.dtype).reshape((-1,) + self._buf.shape[1:])
        need = self.n + len(rows)
        if need > len(self._buf):
            buf = np.empty((max(need, 2 * len(self._buf) + 64),) + self._buf.shape[1:], self._buf.dtype)
            buf[:self.n] = self._buf[:self.n]
            self._buf = buf
        self._buf[self.n:need] = rows
        self.n = need


def _clip(V: np.ndarray, F: np.ndarray, B: np.ndarray, cand: np.ndarray, c: Cut, rho: float, eligible=None):
    """Cut the surface along one plane, within the candidate faces ``cand`` (alive face indices).

    Only triangles that CROSS the plane are touched: the seeds are candidate crossing faces with a
    vertex within ``rho`` of the cut center that ``eligible`` accepts, and the set grows across
    shared crossing edges until the section's boundary loop is complete. Each is split in two -
    the inner piece on the inner side's vertices, the outer piece on duplicates of the edge
    points - so the part beyond the plane comes loose as its own component. The cut fails (None)
    if a loop edge would need a face outside ``cand`` or a face of another cap.

    New points get ids from ``len(V)`` on. Returns (processed face indices, new points, new faces,
    their boundary ids, directed boundary edges of the inner side on the plane, and of the outer
    side), or None."""
    Fc = F[cand]
    vids, Fl = np.unique(Fc, return_inverse=True)
    Fl = Fl.reshape(-1, 3)
    P = V[vids]
    sd = (P - c.center) @ c.normal
    sd = np.where(np.abs(sd) < 1e-9, -1e-9, sd)                          # on the plane counts as inside
    pos = sd > 0
    crossing_f = pos[Fl].any(1) & ~pos[Fl].all(1)
    near_v = np.linalg.norm(P - c.center, axis=1) < rho
    ok = np.ones(len(vids), bool) if eligible is None else eligible(P, c)
    near_v &= ok
    seed = crossing_f & near_v[Fl].any(1)
    if not seed.any():
        return None
    # the crossing faces, joined through their shared crossing edges (a crossing triangle has
    # exactly two): the plane's section loops; keep the ones holding a seed
    cf = np.nonzero(crossing_f)[0]
    n = len(vids)
    Fx = Fl[cf]
    fe = np.sort(np.stack([Fx[:, [0, 1]], Fx[:, [1, 2]], Fx[:, [2, 0]]], 1), axis=2)
    ek = (fe[..., 0] * n + fe[..., 1])[pos[fe[..., 0]] != pos[fe[..., 1]]].reshape(-1, 2)
    flat = ek.ravel()
    order = np.argsort(flat, kind="stable")
    fo = np.repeat(np.arange(len(cf)), 2)[order]
    same = flat[order][1:] == flat[order][:-1]
    _, lab = connected_components(coo_matrix((np.ones(int(same.sum())), (fo[:-1][same], fo[1:][same])),
                                             shape=(len(cf), len(cf))), directed=False)
    loop = np.isin(lab, np.unique(lab[seed[cf]]))
    _, cnt = np.unique(ek[loop].ravel(), return_counts=True)
    if (cnt != 2).any():
        return None                                                      # the loop leaves the candidates
    sel = np.zeros(len(Fl), bool)
    sel[cf[loop]] = True
    proc = cand[sel]
    if (B[proc] != 0).any():
        return None                                                      # it would cut another cap
    if eligible is not None and ok[np.unique(Fl[sel])].mean() < 0.8:
        return None                                                 # the section runs into other branches
    new_pts, extra, extra_b, cut_edges, outer_edges, split = [], [], [], [], [], {}
    base = len(V)

    def at(a, b, outer):                                                 # a, b local ids
        key_ = (min(a, b), max(a, b), outer)
        if key_ not in split:
            t = sd[a] / (sd[a] - sd[b])
            split[key_] = base + len(new_pts)
            new_pts.append(P[a] + t * (P[b] - P[a]))
        return split[key_]

    for fl, f, bid in zip(Fl[sel], F[proc], B[proc]):
        inside = sd[fl] <= 0
        n_in = int(inside.sum())
        odd = int(np.nonzero(inside if n_in == 1 else ~inside)[0][0])
        i1, i2 = (odd + 1) % 3, (odd + 2) % 3
        a, b, cc = f[odd], f[i1], f[i2]
        pi, qi = at(fl[odd], fl[i1], False), at(fl[odd], fl[i2], False)
        po, qo = at(fl[odd], fl[i1], True), at(fl[odd], fl[i2], True)
        if n_in == 1:                                                   # a inside, b and cc outside
            extra += [(a, pi, qi), (po, b, cc), (po, cc, qo)]
            cut_edges.append((pi, qi))
            outer_edges.append((qo, po))
        else:                                                           # a outside, b and cc inside
            extra += [(a, po, qo), (pi, b, cc), (pi, cc, qi)]
            cut_edges.append((qi, pi))
            outer_edges.append((po, qo))
        extra_b += [bid, bid, bid]
    return proc, np.array(new_pts).reshape(-1, 3), extra, extra_b, cut_edges, outer_edges


def _loops(edges):
    """Directed boundary edges -> closed loops of vertex ids (open chains are dropped)."""
    nxt = dict(edges)
    loops, seen = [], set()
    for a in list(nxt):
        if a in seen:
            continue
        loop, v = [], a
        while v not in seen and v in nxt:
            seen.add(v)
            loop.append(v)
            v = nxt[v]
        if len(loop) >= 3 and v == a:
            loops.append(loop)
    return loops


def _ear_clip(xy: np.ndarray) -> list[tuple[int, int, int]]:
    """Triangulate a simple polygon (k, 2); returns index triples, counterclockwise. A clockwise
    polygon is triangulated reversed and its triples mapped back."""
    xy = np.asarray(xy, float)
    area2 = float((xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]).sum())
    if area2 < 0:
        k = len(xy)
        return [(k - 1 - a, k - 1 - b, k - 1 - c) for a, b, c in _ear_clip(xy[::-1])]
    idx = list(range(len(xy)))
    tris = []

    def cross(o, a, b):
        return ((a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1])
                - (a[..., 1] - o[..., 1]) * (b[..., 0] - o[..., 0]))

    while len(idx) > 3:
        P = xy[idx]
        prev, nxt = np.roll(P, 1, 0), np.roll(P, -1, 0)
        for k in np.nonzero(cross(prev, P, nxt) > 0)[0]:                 # convex corners, in order
            a, b, c = prev[k], P[k], nxt[k]
            inside = (cross(a, b, P) >= 0) & (cross(b, c, P) >= 0) & (cross(c, a, P) >= 0)
            inside[[k - 1, k, (k + 1) % len(idx)]] = False
            if not inside.any():
                tris.append((idx[k - 1], idx[k], idx[(k + 1) % len(idx)]))
                idx.pop(k)
                break
        else:
            break                                                     # no ear (degenerate): fan the rest
    tris += [(idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1)]
    return tris


def _components(nv: int, F: np.ndarray) -> np.ndarray:
    """Each face's connected component (through shared vertices)."""
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]]])
    _, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(nv, nv)),
                                  directed=False)
    return lab[F[:, 0]]


def _refined_into(margin: np.ndarray, lo, hi, refine: int, clip: float) -> np.ndarray:
    """The margin's trilinear interpolant on ``[lo, hi]`` (index box, inclusive), ``refine`` times
    finer, padded by one sample of ``-clip`` on every side: float32, built in place (no coordinate
    grid). Sample o of the finer box is the interpolant at coarse index lo + o / refine, which is
    what map_coordinates(order=1) gives at those points."""
    sub = np.asarray(margin[tuple(slice(a, b + 1) for a, b in zip(lo, hi))], np.float32)
    shape = tuple((b - a) * refine + 1 for a, b in zip(lo, hi))
    out = np.full(tuple(s + 2 for s in shape), -clip, np.float32)
    inner = out[1:-1, 1:-1, 1:-1]
    if refine == 1:
        inner[...] = sub
        return out
    f = sub
    for ax in range(3):                          # trilinear = linear along each axis in turn
        g = inner if ax == 2 else np.empty(f.shape[:ax] + (shape[ax],) + f.shape[ax + 1:], np.float32)

        def along(sl, ax=ax):
            ix = [slice(None)] * 3
            ix[ax] = sl
            return tuple(ix)
        a0, a1 = f[along(slice(0, -1))], f[along(slice(1, None))]
        for j in range(refine):
            t = j / refine
            g[along(slice(j, shape[ax] - 1, refine))] = a0 if j == 0 else (1 - t) * a0 + t * a1
        g[along(slice(-1, None))] = f[along(slice(-1, None))]
        f = g
    return out


def _box(margin: np.ndarray):
    """The index box (inclusive) of the margin's positive part, grown by 2 voxels within the grid."""
    inside = margin > 0
    pos = [np.nonzero(inside.any(tuple(b for b in range(3) if b != a)))[0] for a in range(3)]
    if not len(pos[0]):
        raise ThalwegError("the margin has no positive voxel: there is no surface to make")
    lo = np.maximum(np.array([p[0] for p in pos]) - 2, 0)
    hi = np.minimum(np.array([p[-1] for p in pos]) + 2, np.array(margin.shape) - 1)
    return lo, hi


def zero_set(margin: np.ndarray, geometry, refine: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """The margin's zero set as (vertices (V, 3) world mm, faces (F, 3)), wound outward (see the
    module docstring), from its trilinear interpolant sampled ``refine`` times finer over the
    structure's box."""
    from skimage.measure import marching_cubes
    refine = int(refine)
    clip = float(max(8.0, -float(margin.min())))
    lo, hi = _box(margin)
    fp = _refined_into(margin, lo, hi, refine, clip)
    verts, faces, _, _ = marching_cubes(fp, level=LEVEL, method="lewiner", allow_degenerate=False)
    del fp
    dirs = np.asarray(geometry.directions, float)
    V = (verts.astype(np.float64) - 1.0) @ (dirs / refine) + to_world(geometry, lo.astype(float))
    F = faces.astype(np.int64)
    if np.linalg.det(dirs) > 0:                  # skimage winds inward on a right-handed frame
        F = F[:, ::-1].copy()
    return V, F


def mesh_defects(vertices: np.ndarray, faces: np.ndarray) -> dict:
    """What keeps a triangle mesh from being a closed, consistently oriented, outward, manifold
    surface: counts of boundary edges (in one face), non-manifold edges (in more than two), directed
    edges used twice (inconsistent winding), non-manifold vertices (more than one fan), and whether
    its signed volume is not positive. All zero / False for a good surface."""
    F = np.asarray(faces, np.int64)
    nv = len(vertices)
    und = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    _, cnt = np.unique(und[:, 0] * nv + und[:, 1], return_counts=True)
    d = F * nv + np.roll(F, -1, axis=1)                        # corner i: edge F[i] -> F[i + 1]
    ds = np.sort(d.ravel())
    out = {"boundary_edges": int((cnt == 1).sum()), "nonmanifold_edges": int((cnt > 2).sum()),
           "repeated_directed_edges": int((np.diff(ds) == 0).sum())}
    out["nonmanifold_vertices"] = 0
    if not any(out.values()) and len(F):
        # fans: link corner (f, i) at v = F[f, i] to the corner at v of the face across edge v -> next
        order = np.argsort(d.ravel())
        twin = F.ravel() + np.roll(F, -1, axis=1).ravel() * nv   # the reverse edge: next -> v
        g = order[np.searchsorted(ds, twin)]                     # flat corner index of the twin edge's start
        across = (g // 3) * 3 + (g % 3 + 1) % 3                  # the twin face's corner at v
        nc = 3 * len(F)
        k, _ = connected_components(coo_matrix((np.ones(nc), (np.arange(nc), across)), shape=(nc, nc)),
                                    directed=False)
        out["nonmanifold_vertices"] = int(k - len(np.unique(F)))
    V = np.asarray(vertices, float)
    out["signed_volume_mm3"] = float(np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum()
                                     / 6)
    return out


def _is_good(defects: dict) -> bool:
    counts = [v for k, v in defects.items() if k != "signed_volume_mm3"]
    return not any(counts) and defects["signed_volume_mm3"] > 0


def _cut_all(V0, F0, ctree, todo, ball, eligible):
    """Every cut of ``todo`` ((index, Cut) pairs) in turn on the wall (V0, F0). Returns V, F, B
    (the cut's index + 1 on its split faces' caps, 0 elsewhere), S (+1 inner cap, -1 outer cap, 0
    wall), the indices made, and (index, reason) of those not made."""
    V = _Rows(V0)
    F = _Rows(F0)
    B = _Rows(np.zeros(len(F0), np.int64))
    S = _Rows(np.zeros(len(F0), np.int8))
    alive = _Rows(np.ones(len(F0), bool))
    newc = _Rows(np.zeros((0, 3)))                          # centroids of the faces added so far
    n0 = len(F0)
    made, failed = [], []
    for i, c in todo:
        rho = ball * c.radius + 1.0
        reach = 2.0 * rho + 2.0
        old = np.array(ctree.query_ball_point(c.center, reach), dtype=np.int64)
        new = n0 + np.nonzero(np.linalg.norm(newc.a - c.center, axis=1) < reach)[0]
        cand = np.concatenate([old, new])
        cand = cand[alive.a[cand]]
        res = _clip(V.a, F.a, B.a, cand, c, rho, eligible) if len(cand) else None
        loops, open_ = [], False
        if res is not None:
            loops = _loops(res[4])
            closed = {v for lp in loops for v in lp}
            open_ = any(a not in closed for a, _ in res[4])
        if res is None or open_ or not loops:
            failed.append((i, "its section reaches another cap, another branch or beyond the local patch"
                           if res is None else "the plane leaves an open boundary" if open_
                           else "the plane misses the surface"))
            continue
        proc, pts, extra, extra_b, _, outer = res
        alive.a[proc] = False
        V.extend(pts)
        Va = V.a
        u = np.cross(c.normal, [1.0, 0, 0] if abs(c.normal[0]) < 0.9 else [0, 1.0, 0])
        u /= np.linalg.norm(u)
        w = np.cross(c.normal, u)
        caps, sides = [], []
        # the inner side's cap faces +normal (out of the kept structure); the part beyond the plane
        # gets its own cap facing -normal, so it is closed too: normally it comes loose and is
        # dropped; where it stays attached the cut is undone (see surface)
        for side, lps in ((1, loops), (-1, _loops(outer))):
            for loop in lps:
                xy = np.stack([(Va[loop] - c.center) @ u, side * ((Va[loop] - c.center) @ w)], 1)
                twice_area = np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1])
                if twice_area < 0:                             # counterclockwise seen from the cap's side
                    loop, xy = loop[::-1], xy[::-1]
                tri = [(loop[a], loop[b], loop[cc]) for a, b, cc in _ear_clip(xy)]
                caps += tri
                sides += [side] * len(tri)
        add = np.array(extra + caps, dtype=np.int64).reshape(-1, 3)
        F.extend(add)
        B.extend(np.r_[np.array(extra_b, np.int64), np.full(len(caps), i + 1, np.int64)])
        S.extend(np.r_[np.zeros(len(extra), np.int8), np.array(sides, np.int8)])
        alive.extend(np.ones(len(add), bool))
        newc.extend(Va[add].mean(1))
        made.append(i)
    keep = alive.a
    return V.a, F.a[keep], B.a[keep], S.a[keep], made, failed


def surface(margin: np.ndarray, geometry, cuts: list[Cut], ball: float = 1.6, keep_main: bool = True,
            refine: int = 1, graph=None, check: bool = True) -> Mesh:
    """The zero set of ``margin``, clipped flat at ``cuts`` and capped (see the module docstring).

    ``graph``: the TubeGraph the cuts came from. With it, a cut clips only triangles whose nearest
    centerline point lies on the cut's own edge, so another vessel passing within the ball is left
    alone (the spatial index: a KD tree of the graph's points and their edges). A cut that cannot
    be closed - its plane leaves an open boundary within reach, or it would cut another cut's cap -
    or whose far side stays attached (a double wall), or whose cap is not on the kept surface, is
    not made and is listed in ``Mesh.skipped``, never left as a hole.

    ``refine``: extract the zero set from the margin's trilinear interpolant sampled that many times
    finer (over the structure's box). Marching cubes connects surface points by chords, so a coarse
    grid reads small sections a little small; the cap areas on the phantom are within 3 % either way.

    ``check``: raise ThalwegError unless the result is closed, consistently oriented, outward and
    manifold (:func:`mesh_defects`)."""
    from scipy.spatial import cKDTree
    V0, F0 = zero_set(margin, geometry, refine)
    ctree = cKDTree(V0[F0].mean(1))
    eligible = None
    if graph is not None:
        pts = graph.positions()
        owner = np.empty(len(pts), np.int64)
        for e in graph.edges:
            owner[e.point_range[0]:e.point_range[1]] = e.id
        gtree = cKDTree(pts)

        def eligible(Vs, c):
            return np.ones(len(Vs), bool) if c.edge is None else owner[gtree.query(Vs)[1]] == c.edge
    undone = {}
    while True:                                  # each pass undoes at least one cut, or is the last
        todo = [(i, c) for i, c in enumerate(cuts) if i not in undone]
        V, F, B, S, made, failed = _cut_all(V0, F0, ctree, todo, ball, eligible)
        comp = _components(len(V), F)
        main = np.argmax(np.bincount(comp)) if len(F) else 0
        cap = S != 0
        bc, sc, cc = B[cap], S[cap], comp[cap]
        if keep_main:                            # a far-side cap left on the kept surface
            undo = np.unique(bc[(sc < 0) & (cc == main)]) - 1
        else:                                    # a far-side cap on its own cut's inner side
            ncomp = int(comp.max()) + 1
            pairs = [bc[sel] * ncomp + cc[sel] for sel in (sc > 0, sc < 0)]
            undo = np.intersect1d(*pairs) // ncomp - 1
        undo = undo.tolist()
        if not undo:
            break
        for i in undo:
            undone[i] = ("the part beyond the plane stays attached (a loop or a contact reaches around it): "
                         "a cap there would leave a double wall inside the surface")
    reasons = {**undone, **dict(failed)}
    if keep_main and len(F):
        keepf = comp == main
        F, B, S = F[keepf], B[keepf], S[keepf]
    used = np.unique(F)
    remap = -np.ones(len(V), np.int64)
    remap[used] = np.arange(len(used))
    V, F = V[used], remap[F]
    # contiguous ids for the caps that exist, in cut order
    have = set(np.unique(B[S > 0]).tolist())
    capped = [i for i in made if i + 1 in have]
    for i in made:
        if i + 1 not in have:
            reasons[i] = "its cap is not on the kept surface (the section lies on a detached piece)"
    ids = np.zeros(len(cuts) + 1, np.int64)
    ids[[i + 1 for i in capped]] = np.arange(1, len(capped) + 1)
    B = ids[B]
    skipped = [Skipped(cuts[i].name, reasons[i], cuts[i].node, cuts[i].kind) for i in sorted(reasons)]
    caps = [cuts[i] for i in capped]
    mesh = Mesh(V, F, B, ["wall"] + [c.name for c in caps], skipped, caps)
    if check:
        d = mesh_defects(V, F)
        if not _is_good(d):
            raise ThalwegError("the capped surface is not closed and manifold ("
                               + ", ".join(f"{k} {v}" for k, v in d.items()) + "); not writing it")
    return mesh


def capped_surface(graph: TubeGraph, structure: str, margin: np.ndarray, geometry,
                   kinds=("tip", "truncated"), cut_mm: float | None = None, refine: int = 1,
                   ball: float = 1.6, check: bool = True) -> Mesh:
    """The structure's surface with a cap at every end of ``kinds`` that can take one: the ends
    :func:`end_cuts` skips and the cuts :func:`surface` cannot make are both in ``Mesh.skipped``,
    so ``len(mesh.caps) + len(mesh.skipped)`` is the number of ends of those kinds."""
    cuts, pre = end_cuts(graph, structure, cut_mm=cut_mm, kinds=kinds, margin=margin, geometry=geometry)
    mesh = surface(margin, geometry, cuts, ball=ball, refine=refine, graph=graph, check=check)
    mesh.skipped = sorted(pre + mesh.skipped, key=lambda s: (s.node is None, s.node))
    n = len(ends(graph, structure, kinds))
    if len(mesh.caps) + len(mesh.skipped) != n:                    # a bookkeeping bug, not a user error
        raise RuntimeError(f"{len(mesh.caps)} caps + {len(mesh.skipped)} skipped != {n} ends")
    return mesh


def cap_ring(mesh: Mesh, k: int) -> np.ndarray:
    """The vertex ids around cap ``k``, in order, counterclockwise seen from outside the structure
    (along the cap's normal). ThalwegError if there is no such cap or it is not a disk (its rim is
    not one simple loop: a hole, a pinch, or two pieces)."""
    if not 1 <= int(k) <= len(mesh.caps):
        raise ThalwegError(f"there is no cap {k}; the mesh has {len(mesh.caps)}")
    f = mesh.faces[mesh.boundary == k]
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    have = set(map(tuple, e.tolist()))
    rim = [(a, b) for a, b in have if (b, a) not in have]
    loops = _loops(rim)
    if len(loops) != 1 or len(loops[0]) != len(rim):
        raise ThalwegError(f"cap {k} ({mesh.names[k]}) is not a disk: its rim has {len(rim)} edges in "
                           f"{len(loops)} closed loop(s)")
    return np.asarray(loops[0], np.int64)


def boundary_reference_system(mesh: Mesh, k: int, vmtk_vertex_mean: bool = False) -> dict:
    """vmtk's boundary reference system of cap ``k`` (``vmtkboundaryreferencesystems``): the
    ring's barycenter, its mean distance from the barycenter (vmtk's boundary radius) and the
    outward normal.

    The barycenter and the radius are averages along the ring's length (the polygon's perimeter
    centroid, and the mean distance from it integrated along each edge), so they do not depend on
    how the ring's vertices are spaced.
    vmtk averages the vertices, which pulls both toward
    wherever the mesh happens to be dense; ``vmtk_vertex_mean=True`` reproduces that (on the
    C3N-00704 subtree's nine rings it moves the barycenter by 0.05-0.26 mm)."""
    P = mesh.vertices[cap_ring(mesh, k)]
    Q = np.roll(P, -1, axis=0)
    w = np.linalg.norm(Q - P, axis=1)
    if vmtk_vertex_mean:
        b = P.mean(0)
        radius = float(np.linalg.norm(P - b, axis=1).mean())
    else:
        mid = 0.5 * (P + Q)
        b = (mid * w[:, None]).sum(0) / w.sum()                # the polygon's perimeter centroid, exactly
        t = np.linspace(0.0, 1.0, 33)                          # composite Simpson along each edge
        simpson = np.r_[1.0, np.tile([4.0, 2.0], 15), 4.0, 1.0] / 96.0
        along = np.linalg.norm(P[:, None, :] + t[None, :, None] * (Q - P)[:, None, :] - b, axis=2)
        radius = float(((along * simpson).sum(1) * w).sum() / w.sum())
    return dict(barycenter=b, mean_radius_mm=radius, normal=np.asarray(mesh.caps[k - 1].normal, float),
                perimeter_mm=float(w.sum()))


def flow_extensions(mesh: Mesh, ratio: float = 10.0, transition: float = 0.25, caps=None,
                    check: bool = True) -> Mesh:
    """The mesh with a flow extension at its caps (vmtk's ``vmtkflowextensions``, boundary-normal
    mode with adaptive length): each cap is replaced by a tube along its normal, ``ratio`` x the
    ring's mean radius long, that morphs the cap's ring into a circle of that radius about the
    ring's barycenter over the first ``transition`` of its length (a smoothstep blend, so the wall
    has no kink where it starts or ends), then runs straight, and ends in a flat cap.

    Each ring vertex starts toward the circle point at its own fraction of the ring's length (the
    circle turned to lie as close to the ring as it can); a ring that is not star-shaped about its
    barycenter folds if its vertices are sent out radially. Over the transition the vertices also
    slide round to even spacing, so by the straight part a ring with a very short edge no longer
    drags a strip of thin triangles along the tube; they stay only where the wall's own ring has
    them.

    For a ring whose barycenter lies outside it (a C or a hook), no blend toward a circle is
    guaranteed to stay a simple polygon, and some layers of it would pass through themselves -
    triangles that keep their orientation, so no manifold check sees it. Every layer is tested;
    if one is not simple, the cap is extruded unchanged instead (a straight prism of its own
    section, ``Mesh.extensions[k]["end_shape"]`` ``"ring"`` rather than ``"circle"``).

    The tube's faces are wall (``BoundaryId`` 0); the new end cap keeps the cap's id and name.
    ``caps``: the BoundaryIds to extend (default: all). ``Mesh.extensions[k]`` records each one:
    ``length_mm``, ``radius_mm``, ``transition``, the ``ring_barycenter`` it grew from and its
    ``vertices`` (the range of new vertex ids); the cap's ``center`` moves to the extension's end.
    A cap that already has an extension is refused. An extension is straight and knows nothing of
    what is around it: see :func:`extension_collisions`."""
    from dataclasses import replace
    if not ratio > 0 or not 0 <= transition <= 1:
        raise ThalwegError(f"ratio must be positive and transition in [0, 1]; got {ratio}, {transition}")
    if caps is None:
        ids = list(range(1, len(mesh.caps) + 1))
    else:
        if any(int(k) != k for k in caps):
            raise ThalwegError(f"caps must be BoundaryIds (integers); got {list(caps)}")
        ids = sorted({int(k) for k in caps})
    again = [k for k in ids if k in mesh.extensions]
    if again:
        raise ThalwegError(f"cap {again[0]} already has a flow extension")
    V = [mesh.vertices]
    nv = len(mesh.vertices)
    keep = ~np.isin(mesh.boundary, ids)
    F, B = [mesh.faces[keep]], [mesh.boundary[keep]]
    new_caps = list(mesh.caps)
    ext = dict(mesh.extensions)
    for k in ids:
        ring = cap_ring(mesh, k)
        ref = boundary_reference_system(mesh, k)
        n, b, R = ref["normal"], ref["barycenter"], ref["mean_radius_mm"]
        P = mesh.vertices[ring]
        P = P - ((P - b) @ n)[:, None] * n                      # in the cap's plane (they already are)
        e1 = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)                                    # (e1, e2, n) right-handed: ccw about n
        edge = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
        phi = 2.0 * np.pi * np.r_[0.0, np.cumsum(edge)[:-1]] / edge.sum()
        z = (P - b) @ e1 + 1j * ((P - b) @ e2)
        wv = 0.5 * (edge + np.roll(edge, 1))
        turn = np.angle((wv * z * np.exp(-1j * phi)).sum())     # the rotation bringing the circle nearest
        m = len(ring)
        even = 2.0 * np.pi * np.arange(m) / m                   # the same vertices, evenly spaced
        shift = np.angle(np.exp(1j * (phi - even)).mean())
        L = ratio * R
        h = ref["perimeter_mm"] / m                             # layers about as far apart as ring vertices
        n_layers = max(2, int(np.ceil(L / h)))
        s = L * np.arange(1, n_layers + 1) / n_layers
        t = np.clip(s / (transition * L), 0.0, 1.0) if transition > 0 else np.ones(n_layers)
        w = t * t * (3.0 - 2.0 * t)                             # smoothstep
        # each layer's circle point: the vertex's own arc-length fraction at the ring, evenly spaced by
        # the end of the transition (both increase round the ring, so the circle points stay in order)
        angle = (1.0 - w)[:, None] * phi[None] + w[:, None] * (even + shift)[None] + turn
        circle = b + R * (np.cos(angle)[..., None] * e1 + np.sin(angle)[..., None] * e2)
        plane = (1.0 - w)[:, None, None] * P[None] + w[:, None, None] * circle
        rel = plane - b
        xy = np.stack([rel @ e1, rel @ e2], axis=2)
        shape = "circle"
        ring_xy = np.stack([(P - b) @ e1, (P - b) @ e2], axis=1)
        if not _inside(ring_xy, np.zeros(2)) or not _strips_simple(np.concatenate([ring_xy[None], xy]), w):
            # the blend would pass through itself (a C- or hook-shaped ring, its barycenter outside
            # it): extrude the ring unchanged instead - a straight prism of the cap's own section
            plane, shape = np.broadcast_to(P, (n_layers,) + P.shape), "ring"
        layers = plane + s[:, None, None] * n
        first = nv
        ids_layer = [ring] + [nv + j * m + np.arange(m) for j in range(n_layers)]
        V.append(layers.reshape(-1, 3))
        nv += n_layers * m
        for lo, hi in zip(ids_layer[:-1], ids_layer[1:]):
            a, a2 = lo, np.roll(lo, -1)
            c, c2 = hi, np.roll(hi, -1)
            F.append(np.concatenate([np.stack([a, a2, c2], 1), np.stack([a, c2, c], 1)]))
            B.append(np.zeros(2 * m, mesh.boundary.dtype))
        end = b + L * n
        last = ids_layer[-1]
        if shape == "circle":
            V.append(end[None])
            F.append(np.stack([np.full(m, nv), last, np.roll(last, -1)], 1))
            B.append(np.full(m, k, mesh.boundary.dtype))
            nv += 1
        else:                                                   # the ring's own polygon, ear-clipped
            rel = P - b
            tri = np.asarray(_ear_clip(np.stack([rel @ e1, rel @ e2], 1)), np.int64)
            F.append(last[tri])
            B.append(np.full(len(tri), k, mesh.boundary.dtype))
        new_caps[k - 1] = replace(mesh.caps[k - 1], center=end)
        ext[k] = dict(length_mm=float(L), radius_mm=float(R), transition=float(transition), end_shape=shape,
                      ring_barycenter=b, cut_center=np.asarray(mesh.caps[k - 1].center, float),
                      vertices=(first, nv))
    out = Mesh(np.concatenate(V), np.concatenate(F), np.concatenate(B), list(mesh.names), list(mesh.skipped),
               new_caps, ext)
    if check:
        d = mesh_defects(out.vertices, out.faces)
        if not _is_good(d):
            raise ThalwegError("the extended surface is not closed and manifold ("
                               + ", ".join(f"{a} {v}" for a, v in d.items()) + ")")
    return out


def _simple(xy: np.ndarray) -> bool:
    """Whether a closed polygon (k, 2) is simple: no two edges that do not share a vertex cross or
    touch (collinear edges touch only where they overlap). Zero-length edges are skipped."""
    xy = np.asarray(xy, float)
    keep = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1) > 1e-12
    xy = xy[keep]
    k = len(xy)
    if k < 3:
        return False
    a, b = xy, np.roll(xy, -1, axis=0)
    i, j = np.triu_indices(k, 2)
    keep = ~((i == 0) & (j == k - 1))                             # the closing edge meets the first
    i, j = i[keep], j[keep]

    def orient(p, q, r):
        return np.sign((q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0]))
    p1, p2, q1, q2 = a[i], b[i], a[j], b[j]
    o1, o2, o3, o4 = orient(p1, p2, q1), orient(p1, p2, q2), orient(q1, q2, p1), orient(q1, q2, p2)
    hit = (o1 * o2 <= 0) & (o3 * o4 <= 0)
    colinear = (o1 == 0) & (o2 == 0)
    if colinear.any():                                            # collinear pairs: do their spans overlap?
        d = p2[colinear] - p1[colinear]
        u = lambda x: ((x - p1[colinear]) * d).sum(1) / (d * d).sum(1)   # noqa: E731
        lo = np.minimum(u(q1[colinear]), u(q2[colinear]))
        hi = np.maximum(u(q1[colinear]), u(q2[colinear]))
        hit[colinear] = (hi >= 0) & (lo <= 1)
    return not hit.any()


def _inside(xy: np.ndarray, p) -> bool:
    """Whether point p lies inside the polygon (k, 2) (even-odd rule)."""
    x, y = xy[:, 0], xy[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cross = (y > p[1]) != (y2 > p[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        xs = x + (p[1] - y) * (x2 - x) / (y2 - y)
    return bool(np.count_nonzero(cross & (p[0] < xs)) % 2)


STRIP_SAMPLES = 7


def _strips_simple(layers_xy: np.ndarray, w: np.ndarray) -> bool:
    """Whether the flow extension's strips between consecutive layers (ring first) stay simple
    where the shape changes: every layer, and sections through each strip's triangles at
    ``STRIP_SAMPLES`` heights. A quad a_i a_i+1 c_i+1 c_i is split along a_i c_i+1, so the section at
    fraction t runs through (1 - t) a_i + t c_i, then (1 - t) a_i + t c_i+1, for each i."""
    changing = np.r_[True, w[:-1] < 1.0]                          # strip j joins layer j and j + 1
    t = np.linspace(0.0, 1.0, STRIP_SAMPLES + 2)[1:-1]
    for j in np.nonzero(changing)[0]:
        a, c = layers_xy[j], layers_xy[j + 1]
        if not _simple(c):
            return False
        c2 = np.roll(c, -1, axis=0)
        for tau in t:
            v = (1 - tau) * a + tau * c
            d = (1 - tau) * a + tau * c2
            if not _simple(np.stack([v, d], axis=1).reshape(-1, 2)):
                return False
    return True


def extension_collisions(mesh: Mesh, margin: np.ndarray, geometry) -> dict[int, int]:
    """Per flow extension: how many of its vertices lie inside the structure beyond the stub it
    grew from - the extension has run into another branch (or back into its own, around a bend).

    The vessel goes on a little past the cut (the stub the cut removed), so an extension starts
    inside the field; where its axis leaves the field the stub has ended. Vertices more than one
    radius past that point with margin > 0 are counted. An extension whose axis never leaves the
    field within its length has every vertex counted that is inside. The count is also written to
    ``Mesh.extensions[k]["vertices_inside_structure"]``. It is a sign, not a proof: it samples the
    field at the tube's vertices and does not test triangles against each other."""
    from .kernel.rays import first_crossing
    out = {}
    for k, e in mesh.extensions.items():
        n = np.asarray(mesh.caps[k - 1].normal, float)
        a, z = e["vertices"]
        x = mesh.vertices[a:z - 1]                               # the tube's layers (then the end center)
        s = (x - e["ring_barycenter"]) @ n
        leave = first_crossing(margin, geometry, e["ring_barycenter"][None], n[None], e["length_mm"])[0]
        past = s > (leave + e["radius_mm"] if np.isfinite(leave) else 0.0)
        inside = sample(margin, geometry, x[past], cval=-1.0) > 0 if past.any() else np.zeros(0, bool)
        out[k] = int(inside.sum())
        e["vertices_inside_structure"] = out[k]
    return out


def _cap_geometry(mesh: Mesh, k: int):
    f = mesh.faces[mesh.boundary == k]
    V = mesh.vertices
    cr = np.cross(V[f[:, 1]] - V[f[:, 0]], V[f[:, 2]] - V[f[:, 0]])
    a = 0.5 * np.linalg.norm(cr, axis=1)
    area = float(a.sum())
    centroid = (V[f].mean(1) * a[:, None]).sum(0) / area if area > 0 else V[f].mean((0, 1))
    return area, centroid


def boundaries(mesh: Mesh) -> dict:
    """The sidecar document: the wall and each cap that exists (id = BoundaryId), and the skipped
    ends (see the module docstring)."""
    def vec(v):
        return [round(float(x), 6) for x in v]

    wall_area, _ = _cap_geometry(mesh, 0)
    rows = [{"id": 0, "name": "wall", "area_mm2": round(wall_area, 6)}]
    for k, c in enumerate(mesh.caps, 1):
        area, centroid = _cap_geometry(mesh, k)
        ref = boundary_reference_system(mesh, k)
        row = {"id": k, "name": c.name, "cap_kind": c.kind, "node": c.node, "edge": c.edge,
               "center": vec(c.center), "normal": vec(c.normal),
               "inscribed_radius_mm": round(float(c.radius), 6), "area_mm2": round(area, 6),
               "centroid": vec(centroid), "ring_barycenter": vec(ref["barycenter"]),
               "ring_mean_radius_mm": round(ref["mean_radius_mm"], 6)}
        if k in mesh.extensions:
            e = mesh.extensions[k]
            if "vertices_inside_structure" in e:
                row["extension_vertices_inside_structure"] = e["vertices_inside_structure"]
            row.update(extension_length_mm=round(e["length_mm"], 6),
                       extension_radius_mm=round(e["radius_mm"], 6), extension_end_shape=e["end_shape"],
                       extension_transition=e["transition"], cut_center=vec(e["cut_center"]),
                       extension_start_barycenter=vec(e["ring_barycenter"]))
        rows.append(row)
    return {"space": "LPS", "units": "mm", "boundaries": rows,
            "skipped": [{"name": s.name, "cap_kind": s.kind, "node": s.node, "reason": s.reason}
                        for s in mesh.skipped]}


def write_vtp_mesh(mesh: Mesh, path) -> Path:
    """The mesh as .vtp (cell data ``BoundaryId``, Int32; point data ``Mesh.point_data``) plus
    ``<path>.boundaries.json`` (:func:`boundaries`)."""
    path = Path(path)
    for name, a in mesh.point_data.items():
        if len(a) != len(mesh.vertices):
            raise ThalwegError(f"point array {name!r} has {len(a)} values for {len(mesh.vertices)} vertices")
    path.write_text(_polydata_xml(mesh.vertices, polys=np.asarray(mesh.faces, np.int64),
                                  point_data=dict(mesh.point_data) or None,
                                  cell_data={"BoundaryId": np.asarray(mesh.boundary, np.int64)}))
    Path(str(path) + ".boundaries.json").write_text(json.dumps(boundaries(mesh), indent=1) + "\n")
    return path


# -- SWC and Slicer markups -------------------------------------------------------------------------

def write_swc(graph: TubeGraph, structure: str, path) -> Path:
    """One structure as SWC (``n type x y z radius parent``, ids from 1): the tree neuroscience and
    kimimaro-style tools read. Lossy: SWC is a tree of points, so it holds no provenance, no extra
    columns, and no cycles (anything but a rooted tree raises ThalwegError). Type is 0
    (undefined); a radius the tracer could not refine (-1) is written as 0."""
    tree = graph.tree(structure)
    out_of = {n: [graph.edges[i] for i in ids] for n, ids in tree.children.items()}
    lines = ["# SWC written by thalweg: n type x y z radius parent (LPS mm)"]
    nid = 0
    node_swc = {}
    root = graph.nodes[tree.root]
    nid += 1
    node_swc[root.id] = nid
    r0 = next((graph.edge_radius(e)[0] for e in out_of.get(root.id, [])), 0.0)
    x, y, z = root.position
    lines.append(f"{nid} 0 {x:.4f} {y:.4f} {z:.4f} {max(r0, 0.0):.4f} -1")
    stack = [root.id]
    while stack:
        n = stack.pop()
        for e in out_of.get(n, []):
            p, r = graph.edge_points(e), graph.edge_radius(e)
            parent = node_swc[n]
            for k in range(1, len(p)):
                nid += 1
                x, y, z = p[k]
                lines.append(f"{nid} 0 {x:.4f} {y:.4f} {z:.4f} {max(r[k], 0.0):.4f} {parent}")
                parent = nid
            node_swc[e.end_node] = parent
            stack.append(e.end_node)
    path = Path(path)
    path.write_text("\n".join(lines) + "\n")
    return path


MARKUPS_SCHEMA = ("https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/"
                  "Schema/markups-schema-v1.0.3.json#")


def write_slicer_markups(graph: TubeGraph, structure: str, path, every: int = 1) -> Path:
    """One structure as 3D Slicer markups (``.mrk.json``): one open curve per edge, named
    "<structure> edge <id>", control points every ``every``-th sample, LPS. Lossy: radius and the
    graph's connectivity are not carried (Slicer's curves are independent)."""
    markups = []
    for e in graph.edges:
        if e.structure != structure:
            continue
        p = graph.edge_points(e)
        keep = list(range(0, len(p), max(1, every)))
        if keep[-1] != len(p) - 1:
            keep.append(len(p) - 1)
        markups.append({"type": "Curve", "name": f"{structure} edge {e.id}", "coordinateSystem": "LPS",
                        "coordinateUnits": "mm",
                        "controlPoints": [{"id": str(i + 1), "position": [float(v) for v in p[k]]}
                                          for i, k in enumerate(keep)]})
    path = Path(path)
    path.write_text(json.dumps({"@schema": MARKUPS_SCHEMA, "markups": markups}))
    return path
