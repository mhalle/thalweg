"""Connectivity decided by the field on the native grid, and the voxel rules kept for comparison.

The field's inside is {m > 0} of the margin's trilinear interpolant. Its lattice points are the
voxels whose class wins (the labelmap), so the nodes are the voxel labelmap's; only the question
"are these two neighbors connected?" differs, and the interpolant answers it exactly:

- face neighbors (6): both positive -> the edge between them is positive (linear along an edge);
- face diagonals (12): connected within their shared face; if either other face corner is
  positive the cube edges already connect them, otherwise the bilinear face is connected iff its
  saddle is positive: f_a f_b > f_c f_d (the marching-cubes asymptotic decider);
- body diagonals (8): connected within their cell; the 8 corners are closed under the edge and
  face rules first, and a pair both positive but still apart is decided inside the cell by the
  trilinear interpolant sampled 9 x 9 x 9 (the interior "tunnel" configurations; rare).

Loops: the genus of the interpolant's zero surface, extracted by marching cubes with the Lewiner
topological case table (ambiguous faces and cells resolved as the trilinear interpolant does), so
b1 = sum of the genus of its closed components (cavities aside, b2 counted separately).
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage.measure import marching_cubes

OFFSETS = [np.array(o) - 1 for o in np.ndindex(3, 3, 3) if o > (1, 1, 1)]      # 13 forward offsets


def _gather(m, q, fill=-1.0):
    ok = np.all((q >= 0) & (q < m.shape), 1)
    out = np.full(len(q), fill, np.float32)
    out[ok] = m[tuple(q[ok].T)]
    return out


def _interior_connected(vals):
    """vals (8,) corner values, bit order (u, v, w); is corner 0 joined to corner 7 inside the cell?"""
    t = np.linspace(0, 1, 9)
    U, V, W = np.meshgrid(t, t, t, indexing="ij")
    f = np.zeros_like(U)
    for c in range(8):
        bu, bv, bw = c & 1, (c >> 1) & 1, (c >> 2) & 1
        f += vals[c] * (U if bu else 1 - U) * (V if bv else 1 - V) * (W if bw else 1 - W)
    lab, _ = ndi.label(f > 0)
    return lab[0, 0, 0] > 0 and lab[0, 0, 0] == lab[-1, -1, -1]


def _cell_closure(F):
    """F (n, 8) corner values of n cells -> (n, 8, 8) bool: corners joined within the cell by the
    edge and face rules (not yet the interior rule)."""
    pos = F > 0
    n = len(F)
    A = np.zeros((n, 8, 8), bool)
    idx = np.arange(8)
    A[:, idx, idx] = pos
    for i in range(8):
        for j in range(i + 1, 8):
            diff = i ^ j
            nb = bin(diff).count("1")
            if nb == 1:                                             # cube edge
                c = pos[:, i] & pos[:, j]
            elif nb == 2:                                           # face diagonal
                bits = [b for b in (1, 2, 4) if diff & b]
                k, l = i ^ bits[0], i ^ bits[1]                     # the face's other two corners
                c = pos[:, i] & pos[:, j] & ((pos[:, k] | pos[:, l]) | (F[:, i] * F[:, j] > F[:, k] * F[:, l]))
            else:
                continue
            A[:, i, j] = A[:, j, i] = c
    for _ in range(3):                                              # transitive closure, paths <= 8
        A = A | np.einsum("nij,njk->nik", A.astype(np.uint8), A.astype(np.uint8)).astype(bool)
    return A


def field_edges(m, mask=None):
    """Nodes: voxels with m > 0 (within mask if given). Edges: neighbor pairs the interpolant joins.
    Returns (idx (N,3), rows, cols, offsets used per edge as index into OFFSETS)."""
    inside = m > 0 if mask is None else (m > 0) & mask
    idx = np.argwhere(inside)
    node = -np.ones(m.shape, np.int64); node[tuple(idx.T)] = np.arange(len(idx))
    rows, cols, kinds = [], [], []
    tunnels = 0
    for k, o in enumerate(OFFSETS):
        q = idx + o
        okq = np.all((q >= 0) & (q < m.shape), 1)
        a = np.nonzero(okq)[0]
        b = node[tuple(q[a].T)]
        keep = b >= 0
        a, b = a[keep], b[keep]
        nz = np.nonzero(o)[0]
        if len(nz) == 2:                                            # face diagonal: the face's other corners
            u = np.zeros(3, int); v = np.zeros(3, int); u[nz[0]] = o[nz[0]]; v[nz[1]] = o[nz[1]]
            fa, fb = m[tuple(idx[a].T)], m[tuple(idx[b].T)]
            fc, fd = _gather(m, idx[a] + u), _gather(m, idx[a] + v)
            join = (fc > 0) | (fd > 0) | (fa * fb > fc * fd)
            a, b = a[join], b[join]
        elif len(nz) == 3:                                          # body diagonal: close the cell
            unit = [np.eye(3, dtype=int)[i] * o[i] for i in range(3)]
            corners = [sum(unit[t] for t in range(3) if c >> t & 1) if c else np.zeros(3, int) for c in range(8)]
            F = np.stack([_gather(m, idx[a] + cv) for cv in corners], 1)
            full = np.all(F > 0, 1)
            join = full.copy()
            part = np.nonzero(~full)[0]
            if len(part):
                A = _cell_closure(F[part])
                j = A[:, 0, 7]
                amb = np.nonzero(~j)[0]                             # both positive, apart by edges and faces
                for t in amb:
                    j[t] = _interior_connected(F[part][t])
                tunnels += int(j[amb].sum())
                join[part] = j
            a, b = a[join], b[join]
        rows.append(a); cols.append(b); kinds.append(np.full(len(a), k))
    return idx, np.concatenate(rows), np.concatenate(cols), np.concatenate(kinds), tunnels


def voxel_edges(mask, connectivity=26):
    idx = np.argwhere(mask)
    node = -np.ones(mask.shape, np.int64); node[tuple(idx.T)] = np.arange(len(idx))
    rows, cols, kinds = [], [], []
    for k, o in enumerate(OFFSETS):
        if connectivity == 6 and np.count_nonzero(o) > 1:
            continue
        if connectivity == 18 and np.count_nonzero(o) > 2:
            continue
        q = idx + o
        okq = np.all((q >= 0) & (q < mask.shape), 1)
        a = np.nonzero(okq)[0]
        b = node[tuple(q[a].T)]
        keep = b >= 0
        rows.append(a[keep]); cols.append(b[keep]); kinds.append(np.full(keep.sum(), k))
    return idx, np.concatenate(rows), np.concatenate(cols), np.concatenate(kinds)


def components(n, rows, cols):
    G = coo_matrix((np.ones(len(rows), bool), (rows, cols)), shape=(n, n))
    return connected_components(G, directed=False)


def surface_loops(m, box=None, clip=8.0):
    """Loops b1 of {m > 0} from the genus of its zero surface (b1 = total genus over all boundary
    components, cavities included), plus what check_against_graph needs: the number of surface
    components, each vertex's component, and each vertex's inside lattice anchor."""
    if box is not None:
        m = m[tuple(slice(lo, hi) for lo, hi in zip(*box))]
    f = np.pad(m, 1, constant_values=-clip)
    # level just above 0: an exact tie (m == 0, possible with quantized gaps) is outside, as in the graph
    verts, faces, normals, _ = marching_cubes(f, level=1e-5, method="lewiner", allow_degenerate=False)
    # connected surface components via shared vertices
    V = len(verts)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    ncomp, lab = components(V, e[:, 0], e[:, 1])
    E = np.unique(np.sort(e, 1), axis=0)
    chiV = np.bincount(lab, minlength=ncomp)
    chiE = np.bincount(lab[E[:, 0]], minlength=ncomp)
    chiF = np.bincount(lab[faces[:, 0]], minlength=ncomp)
    chi = chiV - chiE + chiF
    genus = (2 - chi) // 2
    # each vertex sits on a lattice edge between an inside and an outside point: its inside
    # endpoint (in unpadded coordinates) is the anchor that ties the surface to the graph
    # (the nearest inside lattice point around the vertex: never ambiguous, even for a vertex on
    # a lattice point - an exact tie, m = 0, puts the surface there and its inside neighbor in the
    # next cell over, so the search spans the 3 x 3 x 3 lattice neighborhood)
    rc = np.clip(np.rint(verts).astype(int), 1, np.array(f.shape) - 2)
    best_d = np.full(V, np.inf); anchor = np.zeros((V, 3), int)
    for c in np.ndindex(3, 3, 3):
        q = rc + np.array(c) - 1
        ins = f[tuple(q.T)] > 1e-5
        d = np.linalg.norm(verts - q, axis=1)
        take = ins & (d < best_d)
        best_d[take] = d[take]; anchor[take] = q[take]
    assert np.isfinite(best_d).all(), "a surface vertex with no inside lattice point next to it"
    anchor = anchor - 1
    return int(genus.sum()), ncomp, lab, anchor, genus


def check_against_graph(shape, idx, glab, surf_lab, anchor):
    """Every surface component must anchor in exactly one graph component (else the graph splits
    what the surface joins); a graph component with k surfaces holds k - 1 cavities (else a graph
    component with no surface would be an error). Returns (agree, cavities, problems)."""
    node = -np.ones(shape, np.int64); node[tuple(idx.T)] = np.arange(len(idx))
    nv = node[tuple(anchor.T)]
    assert (nv >= 0).all(), "a surface anchor that is not a graph node"
    g_of_v = glab[nv]
    problems = []
    s2g = {}
    for s in np.unique(surf_lab):
        gs = np.unique(g_of_v[surf_lab == s])
        if len(gs) != 1:
            problems.append(("surface spans graph components", int(s), gs.tolist()))
        s2g[int(s)] = int(gs[0])
    per_g = np.bincount(list(s2g.values()), minlength=glab.max() + 1)
    missing = np.nonzero(per_g == 0)[0]
    if len(missing):
        problems.append(("graph components without a surface", len(missing)))
    cavities = int(np.clip(per_g - 1, 0, None).sum())
    return not problems, cavities, problems
