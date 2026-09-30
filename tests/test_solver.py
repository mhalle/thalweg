"""svZeroDSolver input from a graph: Poiseuille vessels on a known tube, the network's shape, and
steady flow through it."""
import json

import numpy as np
import pytest

from thalweg.errors import ThalwegError
from thalweg.graph import Edge, Node, Points, Provenance, Structure, TubeGraph
from thalweg.solver import MU, RHO, steady_pressures, write_zero_d_model, zero_d_model


def _tree(spec, radius):
    """spec: [(start, end, end kind, end position)], node 0 the root at the origin; straight edges
    sampled every 0.5 mm, ``radius(edge index, t in [0, 1])`` the traced radius."""
    nodes = {0: Node(id=0, kind="root", position=(0.0, 0.0, 0.0), structure="t")}
    edges, pos, rad = [], [], []
    for k, (a, b, kind, q) in enumerate(spec):
        nodes[b] = Node(id=b, kind=kind, position=tuple(map(float, q)), structure="t")
        p0, p1 = np.array(nodes[a].position), np.array(q, float)
        n = max(3, int(np.linalg.norm(p1 - p0) / 0.5) + 1)
        t = np.linspace(0, 1, n)
        at = len(pos)
        pos.extend(map(tuple, p0 + t[:, None] * (p1 - p0)))
        rad.extend(radius(k, t))
        edges.append(Edge(id=k, structure="t", start_node=a, end_node=b, point_range=(at, at + n),
                          length_mm=float(np.linalg.norm(p1 - p0)), provenance=Provenance(method="field")))
    return TubeGraph(structures=[Structure(name="t", roots=[0], method="test")],
                     nodes=[nodes[i] for i in sorted(nodes)], edges=edges,
                     points=Points(position=pos, radius=rad))


def test_a_straight_tube_is_poiseuille():
    """A 40 mm tube of radius 2 mm: R = 8 mu L / (pi r^4), L = rho L / (pi r^2), in CGS."""
    g = _tree([(0, 1, "tip", (0, 0, 40))], lambda k, t: np.full(len(t), 2.0))
    m = zero_d_model(g, "t")
    (v,) = m["vessels"]
    L, r = 4.0, 0.2
    assert v["vessel_length"] == pytest.approx(L)
    assert v["zero_d_element_values"]["R_poiseuille"] == pytest.approx(8 * MU * L / (np.pi * r ** 4))
    assert v["zero_d_element_values"]["L"] == pytest.approx(RHO * L / (np.pi * r ** 2))
    assert v["zero_d_element_values"]["C"] == 0.0
    assert v["boundary_conditions"] == {"inlet": "INFLOW", "outlet": "OUT0"}
    stiff = zero_d_model(g, "t", wall_stiffness=1e5)["vessels"][0]["zero_d_element_values"]["C"]
    assert stiff == pytest.approx(3 * np.pi / (2 * 1e5) * r ** 3 * L)


def test_a_tapering_tube_integrates_along_its_length():
    g = _tree([(0, 1, "tip", (0, 0, 40))], lambda k, t: 1.0 + 2.0 * t)          # 1 mm to 3 mm
    R = zero_d_model(g, "t")["vessels"][0]["zero_d_element_values"]["R_poiseuille"]
    exact = 8 * MU / np.pi * (4.0 / (3 * 0.2)) * (0.1 ** -3 - 0.3 ** -3)       # r = 0.1 + 0.05 z (cm)
    assert R == pytest.approx(exact, rel=0.03)


def test_a_bifurcation_network_and_its_steady_flow(tmp_path):
    """A trunk and two daughters (one twice the other's radius): one junction, two outlets; with
    equal outlet resistances the wider daughter takes more of the flow, flows add up, and the
    pressure drop along each vessel is flow x resistance."""
    spec = [(0, 1, "junction", (0, 0, 30)), (1, 2, "tip", (-10, 0, 50)), (1, 3, "tip", (10, 0, 50))]
    g = _tree(spec, lambda k, t: np.full(len(t), [3.0, 1.0, 2.0][k]))
    m = zero_d_model(g, "t", inflow=5.0, outlet_resistance=100.0)
    assert [j["junction_name"] for j in m["junctions"]] == ["J1"]
    assert m["junctions"][0]["inlet_vessels"] == [0] and sorted(m["junctions"][0]["outlet_vessels"]) == [1, 2]
    names = {b["bc_name"] for b in m["boundary_conditions"]}
    assert names == {"INFLOW", "OUT0", "OUT1"}
    flow, pressure = steady_pressures(m)
    assert flow[0] == pytest.approx(5.0) and flow[1] + flow[2] == pytest.approx(5.0)
    thin, wide = (1, 2) if m["vessels"][1]["vessel_name"] == "edge_1" else (2, 1)
    assert flow[wide] > flow[thin]
    R = {v["vessel_id"]: v["zero_d_element_values"]["R_poiseuille"] for v in m["vessels"]}
    assert pressure[1] == pytest.approx(pressure[0] - flow[0] * R[0])
    assert pressure[wide] - flow[wide] * R[wide] == pytest.approx(flow[wide] * 100.0)   # the outlet
    path = write_zero_d_model(m, tmp_path / "m.json")
    assert json.loads(path.read_text())["vessels"][0]["zero_d_element_type"] == "BloodVessel"


def test_refusals():
    g = _tree([(0, 1, "tip", (0, 0, 40))], lambda k, t: np.full(len(t), 2.0))
    for bad in (dict(inflow=0), dict(outlet_resistance=-1), dict(wall_stiffness=0)):
        with pytest.raises(ThalwegError):
            zero_d_model(g, "t", **bad)
    none = _tree([(0, 1, "tip", (0, 0, 40))], lambda k, t: np.full(len(t), -1.0))
    with pytest.raises(ThalwegError, match="no traced radius"):
        zero_d_model(none, "t")
    two = _tree([(0, 1, "tip", (0, 0, 40)), (0, 2, "tip", (0, 0, -40))], lambda k, t: np.full(len(t), 2.0))
    with pytest.raises(ThalwegError, match="exactly one edge"):
        zero_d_model(two, "t")
