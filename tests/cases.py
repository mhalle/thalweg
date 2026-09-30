"""Helpers shared by the data tests: the research comparisons' choices, restated on thalweg's graph."""


def reference_source(g):
    """research/vessels/vmtk_prep.py's source: the left-lung (x > root + 15 mm) segment with the
    most tips in [15, 60], started one radius (at least 1 mm) + 1 mm past its junction."""
    out_of = {}
    for e in g.edges:
        out_of.setdefault(e.start_node, []).append(e.id)

    def downstream(j):
        res, stack = [], [j]
        while stack:
            k = stack.pop()
            res.append(k)
            stack.extend(out_of.get(g.edges[k].end_node, []))
        return res

    root_x = g.nodes[g.structures[0].roots[0]].position[0]
    best = None
    for e in g.edges:
        ds = downstream(e.id)
        tips = [k for k in ds if not out_of.get(g.edges[k].end_node)]
        if 15 <= len(tips) <= 60 and g.edge_points(e.id)[:, 0].mean() > root_x + 15:
            if best is None or len(tips) > best[1]:
                best = (e.id, len(tips))
    s0 = best[0]
    return (s0, max(g.edge_radius(s0)[0], 1.0) + 1.0)
