"""vmtk's branch grouping, frames and bifurcation vectors, run on thalweg's graph.

vmtk splits a tree into **groups** by where the tube of one centerline leaves the others' (Antiga
& Steinman 2004): a branch group per stretch owned by one set of centerlines, and a **bifurcation
group** (blanked) for the overlap region around each junction. Bifurcation frames and vectors are
defined on those groups. :func:`vmtk_branching` runs the ported chain (:mod:`thalweg.vmtk`) on a
structure's source-to-tip paths (:func:`thalweg.adapters.to_vmtk`):

    attributes -> branch extractor -> bifurcation reference systems -> offset attributes
              -> bifurcation vectors -> branch geometry

``vmtk_compatible=True`` turns every ``vmtk_*`` flag on (vmtk's numbers, defects included, for
comparison); the default is the corrected behavior. :func:`annotate` writes the result back onto
the graph:

- point columns ``branch_group`` (vmtk group id) and ``bifurcation_region`` (1 inside a blanked
  bifurcation group, else 0) for the structure's points, None elsewhere;
- on each junction node, ``attributes["bifurcation_frames"]``: vmtk's bifurcation(s) at that
  junction - the blanked groups on its incident edges within the junction's radius + 2 mm - each
  with its reference system (group, origin, normal, up-normal). Usually one; two where the tracer
  makes one junction of what vmtk splits into two close bifurcations (or none where vmtk merged
  the junction into a neighbor's bifurcation region). ``Structure.statistics["vmtk_bifurcations"]``
  counts junctions with none, one and several, and vmtk bifurcations shared by several junctions;
- on each edge, ``attributes["branch_groups"]`` (the vmtk groups along it, in order) and, for an
  edge leaving a junction, ``attributes["bifurcation_vector"]``: vmtk's in-plane and out-of-plane
  angles (radians; definitions in :mod:`thalweg.vmtk.vectors`) of the edge's own group - the first
  unblanked group after its start junction's bifurcation - at that bifurcation. An edge whose start
  junction has no vmtk bifurcation, or that ends inside it, gets none.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .adapters import VmtkPaths, to_vmtk
from .graph import TubeGraph
from .vmtk import (VMTK_FLAGS, Centerlines, ReferenceSystems, bifurcation_reference_systems,
                   bifurcation_vectors, branch_geometry, centerline_attributes, extract_branches,
                   offset_attributes)


@dataclass
class Branching:
    paths: VmtkPaths
    split: Centerlines
    frames: ReferenceSystems
    offset: Centerlines
    vectors: Centerlines
    geometry: Centerlines


def _flags(name: str, on: bool) -> dict:
    return {f: on for f in VMTK_FLAGS.get(name, ())}


def vmtk_branching(graph: TubeGraph, structure: str, source=None, vmtk_compatible: bool = False,
                   step: float = 0.3) -> Branching:
    """vmtk's grouping chain on one structure (see the module docstring)."""
    paths = to_vmtk(graph, structure, source=source, step=step)
    cl = centerline_attributes(paths.centerlines, **_flags("centerline_attributes", vmtk_compatible))
    split = extract_branches(cl, **_flags("extract_branches", vmtk_compatible))
    frames = bifurcation_reference_systems(split, **_flags("bifurcation_reference_systems", vmtk_compatible))
    if len(frames.points):
        offset = offset_attributes(split, frames, -1, **_flags("offset_attributes", vmtk_compatible))
        vectors = bifurcation_vectors(split, frames, **_flags("bifurcation_vectors", vmtk_compatible))
    else:                                           # a tube without a bifurcation: nothing to offset or angle
        offset = split
        vectors = Centerlines(np.zeros((0, 3)), [], {k: np.zeros(0) for k in (
            "GroupIds", "BifurcationGroupIds", "BifurcationVectorsOrientation",
            "InPlaneBifurcationVectorAngles", "OutOfPlaneBifurcationVectorAngles")})
    geometry = branch_geometry(split, **_flags("branch_geometry", vmtk_compatible))
    return Branching(paths, split, frames, offset, vectors, geometry)


def point_groups(graph: TubeGraph, structure: str, b: Branching) -> tuple[np.ndarray, np.ndarray]:
    """(group, blanked) for every point of the table (-1 where the structure's paths do not reach).

    A graph point is matched to the nearest split point of ONE centerline that runs through its
    edge (all centerlines through an edge share its samples), never to a neighboring branch."""
    n = len(graph.points.position)
    group = np.full(n, -1, np.int64)
    blank = np.full(n, -1, np.int64)
    sp = b.split
    cid = sp.cell_data["CenterlineIds"]
    gid = sp.cell_data["GroupIds"]
    bl = sp.cell_data["Blanking"]
    by_line: dict[int, tuple] = {}
    for k in range(sp.n_cells):
        by_line.setdefault(int(cid[k]), []).append(k)
    owner = {}
    for i, chain in enumerate(b.paths.edges):
        for e in chain:
            owner.setdefault(e, i)
    trees = {}
    for e in graph.edges:
        if e.structure != structure or e.id not in owner:
            continue
        line = owner[e.id]
        if line not in trees:
            cells = by_line.get(line, [])
            pts = np.concatenate([sp.cell_points(k) for k in cells]) if cells else np.zeros((0, 3))
            lab = np.concatenate([np.full(len(sp.cells[k]), k) for k in cells]) if cells else np.zeros(0, int)
            trees[line] = (cKDTree(pts) if len(pts) else None, lab)
        tree, lab = trees[line]
        if tree is None:
            continue
        a, z = e.point_range
        _, j = tree.query(graph.positions()[a:z])
        k = lab[j]
        group[a:z] = gid[k]
        blank[a:z] = bl[k]
    return group, blank


def annotate(graph: TubeGraph, structure: str, b: Branching) -> TubeGraph:
    """A copy of ``graph`` with vmtk's groups, frames and bifurcation angles written on (see above)."""
    group, blank = point_groups(graph, structure, b)
    n = len(group)
    cols = dict(graph.points.columns)
    old_g = cols.get("branch_group", [None] * n)
    old_b = cols.get("bifurcation_region", [None] * n)
    cols["branch_group"] = [int(g) if g >= 0 else old for g, old in zip(group, old_g)]
    cols["bifurcation_region"] = [int(x) if x >= 0 else old for x, old in zip(blank, old_b)]
    points = graph.points.model_copy(update={"columns": cols})
    # frames onto junction nodes, by group: the blanked groups on the junction's incident edges near it
    nodes = []
    fr = b.frames
    frame_of = {int(g): j for j, g in enumerate(fr.point_data.get("GroupIds", []))}
    incident: dict[int, list] = {}
    for e in graph.edges:
        if e.structure == structure:
            incident.setdefault(e.start_node, []).append(e)
            incident.setdefault(e.end_node, []).append(e)
    positions = graph.positions()
    counts = dict(junctions_without=0, junctions_with_one=0, junctions_with_several=0)
    used: dict[int, int] = {}
    on_paths = {n for chain in b.paths.edges for eid in chain
                for n in (graph.edges[eid].start_node, graph.edges[eid].end_node)}
    for nd in graph.nodes:
        if nd.structure == structure and nd.kind == "junction" and nd.id in on_paths:
            edges_here = incident.get(nd.id, [])
            reach = max((float(graph.edge_radius(e)[0 if e.start_node == nd.id else -1]) for e in edges_here),
                        default=0.0) + 2.0
            groups = set()
            for e in edges_here:
                a, z = e.point_range
                near = np.linalg.norm(positions[a:z] - np.asarray(nd.position), axis=1) <= reach
                groups |= {int(g) for g, bb in zip(group[a:z][near], blank[a:z][near]) if bb == 1}
            found = sorted(g for g in groups if g in frame_of)
            counts["junctions_without" if not found else ("junctions_with_one" if len(found) == 1
                                                          else "junctions_with_several")] += 1
            for g in found:
                used[g] = used.get(g, 0) + 1
            if found:
                attrs = dict(nd.attributes)
                attrs["bifurcation_frames"] = [dict(
                    group=g, origin=[float(v) for v in fr.points[frame_of[g]]],
                    normal=[float(v) for v in fr.point_data["Normal"][frame_of[g]]],
                    up_normal=[float(v) for v in fr.point_data["UpNormal"][frame_of[g]]]) for g in found]
                nd = nd.model_copy(update={"attributes": attrs})
        nodes.append(nd)
    counts["bifurcations_shared_by_several_junctions"] = sum(1 for v in used.values() if v > 1)
    counts["bifurcations_on_no_junction"] = len(set(frame_of) - set(used))
    # groups and bifurcation vectors onto edges
    v = b.vectors
    vec_by_group = {}
    for k in range(len(v.points)):
        vec_by_group.setdefault(int(v.point_data["GroupIds"][k]), []).append(k)
    start_frames = {nd.id: {f["group"] for f in nd.attributes.get("bifurcation_frames", [])} for nd in nodes}
    edges = []
    for e in graph.edges:
        if e.structure == structure:
            a, z = e.point_range
            seq = [int(g) for g in group[a:z] if g >= 0]
            runs = [g for i, g in enumerate(seq) if i == 0 or g != seq[i - 1]]
            attrs = dict(e.attributes)
            attrs["branch_groups"] = runs
            # the edge's own group: the first unblanked group after its start junction's bifurcation
            # run (an edge can start on its parent's group and run through a later bifurcation)
            start_bifs = start_frames.get(e.start_node, set())
            blanked = {int(g) for g, bb in zip(group[a:z], blank[a:z]) if g >= 0 and bb == 1}
            own = None
            for i, g in enumerate(runs):
                if g in start_bifs and g in blanked:
                    own = next((h for h in runs[i + 1:] if h not in blanked), None)
                    b0 = g
                    break
            if own is not None:
                orient = v.point_data["BifurcationVectorsOrientation"]
                bif = v.point_data["BifurcationGroupIds"]
                down = [k for k in vec_by_group.get(int(own), [])
                        if int(orient[k]) == 1 and int(bif[k]) == b0]
                # a parent inside its sphere all the way to the centerline's start has no vector (NaN rows)
                down = [k for k in down if np.isfinite(v.point_data["InPlaneBifurcationVectorAngles"][k])]
                if down:
                    k = down[0]
                    attrs["bifurcation_vector"] = dict(
                        bifurcation_group=int(v.point_data["BifurcationGroupIds"][k]),
                        in_plane_angle_rad=float(v.point_data["InPlaneBifurcationVectorAngles"][k]),
                        out_of_plane_angle_rad=float(v.point_data["OutOfPlaneBifurcationVectorAngles"][k]))
            e = e.model_copy(update={"attributes": attrs})
        edges.append(e)
    structures = [st.model_copy(update={"statistics": dict(st.statistics, vmtk_bifurcations=counts)})
                  if st.name == structure else st for st in graph.structures]
    return graph.model_copy(
        update={"points": points, "nodes": nodes, "edges": edges, "structures": structures})
