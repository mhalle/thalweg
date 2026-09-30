"""vmtk's centerline convention as plain arrays: the input and output of every ported filter.

vmtk hands centerlines around as a ``vtkPolyData``: shared points, polyline cells (one per
centerline, or per group piece after splitting), point arrays (radius, abscissas, normals) and cell
arrays (group, centerline and tract ids, blanking). :class:`Centerlines` is that, without VTK:
``points`` (N, 3) float64, ``cells`` a list of point-id arrays, and two dicts of named arrays.
Array names are vmtk's own, so a port reads like the C++ it follows.

This is the adapter layer's container - the vmtk convention used by the oracle tests and by
vmtk-compatible export. thalweg's own data model is the graph (``thalweg.graph``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

RADIUS = "MaximumInscribedSphereRadius"
ABSCISSAS = "Abscissas"
NORMALS = "ParallelTransportNormals"
GROUP_IDS = "GroupIds"
CENTERLINE_IDS = "CenterlineIds"
TRACT_IDS = "TractIds"
BLANKING = "Blanking"


@dataclass
class Centerlines:
    """Points, polyline cells and named point/cell/field arrays (vmtk's vtkPolyData, as numpy).

    ``point_data[name]`` has one row per point, ``cell_data[name]`` one row per cell. Cells index
    into ``points``; two cells may share points (vmtk's split centerlines do not, but the input
    convention allows it).
    """

    points: np.ndarray
    cells: list[np.ndarray]
    point_data: dict[str, np.ndarray] = field(default_factory=dict)
    cell_data: dict[str, np.ndarray] = field(default_factory=dict)
    field_data: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=np.float64).reshape(-1, 3)
        self.cells = [np.asarray(c, dtype=np.int64) for c in self.cells]

    # -- construction -----------------------------------------------------------------
    @classmethod
    def from_lines(cls, lines, radii=None, radius_name: str = RADIUS) -> "Centerlines":
        """One cell per polyline, points not shared (vmtk's ``vtkPolyData`` built line by line)."""
        lines = [np.asarray(L, dtype=np.float64).reshape(-1, 3) for L in lines]
        pts = np.concatenate(lines) if lines else np.zeros((0, 3))
        cells, off = [], 0
        for L in lines:
            cells.append(np.arange(off, off + len(L)))
            off += len(L)
        pd = {}
        if radii is not None:
            pd[radius_name] = np.concatenate([np.asarray(r, dtype=np.float64) for r in radii])
        return cls(pts, cells, pd)

    @classmethod
    def from_npz(cls, path_or_data) -> "Centerlines":
        """The oracle layout: ``points``, ``cell_ids`` + ``cell_len``, ``pd__/cd__/fd__<name>``."""
        z = np.load(path_or_data) if not hasattr(path_or_data, "files") else path_or_data
        lens = z["cell_len"]
        cells = np.split(z["cell_ids"], np.cumsum(lens)[:-1]) if len(lens) else []
        pd = {k[4:]: z[k] for k in z.files if k.startswith("pd__")}
        cd = {k[4:]: z[k] for k in z.files if k.startswith("cd__")}
        fd = {k[4:]: z[k] for k in z.files if k.startswith("fd__")}
        return cls(z["points"], cells, pd, cd, fd)

    def to_npz_dict(self) -> dict[str, np.ndarray]:
        out = {"points": self.points,
               "cell_ids": np.concatenate(self.cells) if self.cells else np.zeros(0, np.int64),
               "cell_len": np.array([len(c) for c in self.cells], np.int64)}
        out.update({f"pd__{k}": v for k, v in self.point_data.items()})
        out.update({f"cd__{k}": v for k, v in self.cell_data.items()})
        out.update({f"fd__{k}": v for k, v in self.field_data.items()})
        return out

    # -- access -----------------------------------------------------------------------
    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def n_points(self) -> int:
        return len(self.points)

    def cell_points(self, k: int) -> np.ndarray:
        return self.points[self.cells[k]]

    def cell_array(self, name: str, k: int) -> np.ndarray:
        """A point array's rows along cell ``k``."""
        return self.point_data[name][self.cells[k]]

    def copy(self) -> "Centerlines":
        return Centerlines(self.points.copy(), [c.copy() for c in self.cells],
                           {k: v.copy() for k, v in self.point_data.items()},
                           {k: v.copy() for k, v in self.cell_data.items()},
                           {k: v.copy() for k, v in self.field_data.items()})


@dataclass
class ReferenceSystems:
    """vmtk's bifurcation reference systems: one point (origin) per bifurcation group, with
    ``Normal``, ``UpNormal`` and ``GroupIds`` point arrays (vtkvmtkCenterlineBifurcationReferenceSystems)."""

    points: np.ndarray
    point_data: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_npz(cls, path_or_data) -> "ReferenceSystems":
        z = np.load(path_or_data) if not hasattr(path_or_data, "files") else path_or_data
        return cls(np.asarray(z["points"], np.float64).reshape(-1, 3),
                   {k[4:]: z[k] for k in z.files if k.startswith("pd__")})
