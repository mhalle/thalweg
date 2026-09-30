"""A reduced-order flow model of a tree: svZeroDSolver input.

svZeroDSolver (SimVascular's 0-D solver) takes a JSON network of vessels, junctions and boundary
conditions. :func:`zero_d_model` writes one from a structure's graph:

- **vessels**: one per graph edge, a ``BloodVessel`` with Poiseuille resistance
  ``R = 8 mu / pi * integral ds / r^4`` and inductance ``L = rho / pi * integral ds / r^2``, both
  integrated along the edge, its traced radius taken as linear between samples (the integrals
  are then exact; a trapezoid rule over 1/r^4 reads a steep taper 6 % high), capacitance 0 (a rigid
  wall; give ``wall_stiffness`` = E h, the wall's Young's modulus times its thickness, for a
  thin-wall compliance ``C = 3 pi / (2 E h) * integral r^3 ds``), no stenosis term;
- **junctions**: one per branching node (and per node joining two edges), inlet the edge arriving,
  outlets the edges leaving - the tree oriented from its root;
- **boundary conditions**: a steady inflow ``inflow`` at the root's edge and a resistance
  ``outlet_resistance`` (to a distal pressure 0) at every other end. These are placeholders to
  replace with the study's own - the geometry does not know them.

Units are CGS, as SimVascular's: cm, g, s; resistance in dyn s / cm^5, flow in mL/s (cm^3/s);
viscosity ``mu`` 0.04 P and density ``rho`` 1.06 g/cm^3 by default (blood). A sample without a
radius takes its edge's smallest traced radius.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .errors import ThalwegError
from .graph import TubeGraph

MU = 0.04            # P = dyn s / cm^2
RHO = 1.06           # g / cm^3


def _integrals(graph: TubeGraph, edge) -> tuple[float, float, float, float]:
    """(length, integral ds/r^4, integral ds/r^2, integral r^3 ds) of an edge, in cm, exact for the
    radius varying linearly along each segment between samples."""
    p = graph.edge_points(edge) / 10.0
    r = graph.edge_radius(edge) / 10.0
    if not (r > 0).any():
        raise ThalwegError(f"edge {edge.id} has no traced radius")
    r = np.where(r > 0, r, r[r > 0].min())
    ds = np.linalg.norm(np.diff(p, axis=0), axis=1)
    a, b = r[:-1], r[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        same = np.isclose(a, b, rtol=1e-9)
        inv4 = np.where(same, ds / a ** 4, ds * (a ** -3 - b ** -3) / (3.0 * (b - a)))
        inv2 = ds / (a * b)
    cube = ds * (a ** 3 + a ** 2 * b + a * b ** 2 + b ** 3) / 4.0
    return float(ds.sum()), float(inv4.sum()), float(inv2.sum()), float(cube.sum())


def zero_d_model(graph: TubeGraph, structure: str, inflow: float = 1.0,
                 outlet_resistance: float = 1000.0, mu: float = MU, rho: float = RHO,
                 wall_stiffness: float | None = None, cycles: int = 1, points_per_cycle: int = 101) -> dict:
    """The svZeroDSolver input for one structure (see the module docstring), as a dict."""
    if inflow <= 0 or outlet_resistance < 0 or mu <= 0 or rho <= 0:
        raise ThalwegError("inflow, viscosity and density must be positive, the outlet resistance "
                           "not negative")
    if wall_stiffness is not None and wall_stiffness <= 0:
        raise ThalwegError("wall_stiffness (E h) must be positive")
    tree = graph.tree(structure)
    edges = [graph.edges[e] for e in tree.order]
    index = {e.id: k for k, e in enumerate(edges)}                  # vessel ids in flow order
    vessels, bcs = [], [dict(bc_name="INFLOW", bc_type="FLOW",
                             bc_values=dict(Q=[float(inflow), float(inflow)], t=[0.0, 1.0]))]
    outlets = 0
    for e in edges:
        length, inv4, inv2, cube = _integrals(graph, e)
        values = dict(R_poiseuille=8.0 * mu / np.pi * inv4, L=rho / np.pi * inv2,
                      C=(3.0 * np.pi / (2.0 * wall_stiffness) * cube) if wall_stiffness else 0.0,
                      stenosis_coefficient=0.0)
        v = dict(vessel_id=index[e.id], vessel_name=f"edge_{e.id}", vessel_length=length,
                 zero_d_element_type="BloodVessel", zero_d_element_values=values)
        ends = {}
        if e.start_node == tree.root:
            ends["inlet"] = "INFLOW"
        if not tree.children.get(e.end_node):
            name = f"OUT{outlets}"
            outlets += 1
            ends["outlet"] = name
            bcs.append(dict(bc_name=name, bc_type="RESISTANCE",
                            bc_values=dict(R=float(outlet_resistance), Pd=0.0)))
        if ends:
            v["boundary_conditions"] = ends
        vessels.append(v)
    if sum(1 for e in edges if e.start_node == tree.root) != 1:
        raise ThalwegError(f"{structure!r}: the root must start exactly one edge for a single inflow")
    junctions = []
    for node, kids in sorted(tree.children.items()):
        if node == tree.root or node not in tree.parent:
            continue
        junctions.append(dict(junction_name=f"J{node}", junction_type="NORMAL_JUNCTION",
                              inlet_vessels=[index[tree.parent[node]]],
                              outlet_vessels=[index[k] for k in kids]))
    return dict(simulation_parameters=dict(number_of_cardiac_cycles=int(cycles),
                                           number_of_time_pts_per_cardiac_cycle=int(points_per_cycle),
                                           steady_initial=True),
                boundary_conditions=bcs, junctions=junctions, vessels=vessels,
                thalweg=dict(structure=structure, units="cgs", viscosity_poise=mu, density_g_per_cm3=rho,
                             boundary_conditions="placeholders: a steady inflow and equal outlet "
                                                 "resistances"))


def write_zero_d_model(model: dict, path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(model, indent=1) + "\n")
    return path


def steady_pressures(model: dict) -> tuple[dict[int, float], dict[int, float]]:
    """Steady flow through a model written by :func:`zero_d_model` (resistances only, rigid wall):
    ``(flow per vessel id, pressure at each vessel's inlet)``, solved on the tree. A check of the
    network's consistency, not a replacement for the solver."""
    vessels = {v["vessel_id"]: v for v in model["vessels"]}
    kids: dict[int, list[int]] = {}
    for j in model["junctions"]:
        kids.setdefault(j["inlet_vessels"][0], []).extend(j["outlet_vessels"])
    r_out = {b["bc_name"]: b["bc_values"]["R"] for b in model["boundary_conditions"]
             if b["bc_type"] == "RESISTANCE"}
    total: dict[int, float] = {}

    def resistance(v: int) -> float:                                   # the vessel and all below it
        own = vessels[v]["zero_d_element_values"]["R_poiseuille"]
        below = kids.get(v, [])
        if not below:
            out = r_out[vessels[v]["boundary_conditions"]["outlet"]]
        else:
            out = 1.0 / sum(1.0 / resistance(k) for k in below)
        total[v] = own + out
        return total[v]

    root = next(v for v, d in vessels.items() if d.get("boundary_conditions", {}).get("inlet") == "INFLOW")
    q_in = next(b for b in model["boundary_conditions"] if b["bc_name"] == "INFLOW")["bc_values"]["Q"][0]
    resistance(root)
    flow, pressure = {}, {}
    stack = [(root, q_in, q_in * total[root])]
    while stack:
        v, q, p = stack.pop()
        flow[v], pressure[v] = q, p
        p_end = p - q * vessels[v]["zero_d_element_values"]["R_poiseuille"]
        for k in kids.get(v, []):
            stack.append((k, p_end / total[k], p_end))
    return flow, pressure
