"""Faithful ports of vmtk's centerline-only filters, on vmtk's own convention (numpy, no VTK).

The adapter layer: :class:`Centerlines` is vmtk's polydata as arrays, and each module follows one
vtkVmtk filter line for line, checked against vmtk 1.5.2's own output (tests/oracle). thalweg's
data model is the graph; these ports give it vmtk's groups, frames and attributes group for group,
and make vmtk-compatible export possible. vmtk itself is never imported.

Modules:

- :mod:`.centerlines` - the container (:class:`Centerlines`, :class:`ReferenceSystems`) and vmtk's
  array names;
- :mod:`.utilities` - vtkvmtkCenterlineUtilities: group / centerline / tract queries, InterpolateTuple;
- :mod:`.attributes` - vtkvmtkCenterlineAttributesFilter: abscissas, parallel-transport normals;
- :mod:`.resampling` - vmtkCenterlineResampling: vtkCleanPolyData + vtkSplineFilter;
- :mod:`.polyball` - vtkvmtkPolyBallLine, the tube function, vectorized over points and segments;
- :mod:`.sphere_distance` - vtkvmtkCenterlineSphereDistance: the touching-sphere walk;
- :mod:`.branches` - vtkvmtkCenterlineBranchExtractor: split, group and merge tracts;
- :mod:`.frames` - vtkvmtkCenterlineBifurcationReferenceSystems: one frame per bifurcation;
- :mod:`.offset` - vtkvmtkCenterlineReferenceSystemAttributesOffset: attributes relative to a frame;
- :mod:`.merge` - vtkvmtkMergeCenterlines: one polyline per branch group;
- :mod:`.smoothing` - vtkvmtkCenterlineSmoothing: Gauss-Seidel Laplacian relaxation;
- :mod:`.geometry` - vtkvmtkCenterlineGeometry: curvature, torsion, Frenet frames, tortuosity;
- :mod:`.branch_geometry` - vtkvmtkCenterlineBranchGeometry: geometry per branch group;
- :mod:`.vectors` - vtkvmtkCenterlineBifurcationVectors: bifurcation vectors and angles;
- :mod:`._vtk` (private) - the VTK and vtkvmtkMath routines the filters share, one port of each.

**The flag convention.** A public function's default is the correct behavior. Where vmtk (or the
VTK it runs) has a defect or loses precision, a keyword ``vmtk_<name>: bool = False`` reproduces it;
each flag is documented in its module's docstring (what vmtk does, why it is wrong, how big the
effect is). ``vmtk_float32`` always means vmtk's float32 point storage (a default ``vtkPoints`` is
VTK_FLOAT); the default keeps float64. The oracle tests pass True for every flag of their stage.
:data:`VMTK_FLAGS` maps each public function to its flags, so ``**{f: True for f in
VMTK_FLAGS[name]}`` asks for everything as vmtk does it.
"""
import inspect

from .attributes import centerline_attributes
from .branch_geometry import branch_geometry
from .branches import extract_branches
from .centerlines import (ABSCISSAS, BLANKING, CENTERLINE_IDS, GROUP_IDS, NORMALS, RADIUS, TRACT_IDS,
                          Centerlines, ReferenceSystems)
from .frames import bifurcation_reference_systems
from .geometry import centerline_geometry
from .merge import merge_centerlines
from .offset import offset_attributes
from .resampling import resample_centerlines
from .smoothing import smooth_centerlines
from .vectors import bifurcation_vectors

VMTK_FLAGS: dict[str, tuple[str, ...]] = {
    f.__name__: tuple(p for p in inspect.signature(f).parameters if p.startswith("vmtk_"))
    for f in (centerline_attributes, resample_centerlines, extract_branches, bifurcation_reference_systems,
              offset_attributes, merge_centerlines, smooth_centerlines, centerline_geometry, branch_geometry,
              bifurcation_vectors)}
"""Each public function's ``vmtk_*`` flags, read off its signature."""

__all__ = ["Centerlines", "ReferenceSystems", "RADIUS", "ABSCISSAS", "NORMALS", "GROUP_IDS",
           "CENTERLINE_IDS", "TRACT_IDS", "BLANKING", "VMTK_FLAGS",
           "centerline_attributes", "resample_centerlines", "extract_branches",
           "bifurcation_reference_systems", "offset_attributes", "merge_centerlines", "smooth_centerlines",
           "centerline_geometry", "branch_geometry", "bifurcation_vectors"]
