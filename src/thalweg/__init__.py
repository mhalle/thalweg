"""thalweg: centerlines, branches and wall geometry of tubular structures, read from a field.

A successor to vmtk that reads a segmentation model's continuous field (rankfield margins in a
ranked store; or, in a degraded mode, a labelmap, signed distance image or DICOM SEG) instead of
a triangulated surface. Three layers, kept apart by ``tests/test_layering.py``:

- ``thalweg.kernel``: numpy/scipy/skimage algorithms on arrays and grids (field sampling and
  zero crossings, field connectivity and topology, the centerline tracer and its pruning,
  recentering, sections, rays, curvature, spline geometry); knows nothing of files, names or
  formats;
- ``thalweg.vmtk``: faithful ports of vmtk's centerline-only filters on vmtk's own convention,
  numpy only, checked against vmtk's output; vmtk itself is never imported;
- the pipeline: ``store`` and ``volume`` (inputs), ``centerlines`` (a structure's centerlines as
  a graph, rooted at its inlet), ``graph`` (the ``.thalweg.json`` model), ``adapters`` (graph <->
  vmtk's convention), ``branching`` (vmtk's grouping, frames, angles and sections on the graph),
  ``measure`` (the branch table, airway walls), ``partition`` (the branch each point belongs
  to), ``wallmap`` (the wall unrolled), ``straighten``, ``lobes``, ``pairing``, ``plausibility``,
  ``statistics`` (the lung batch's anatomy and whole-tree numbers), ``case`` (one case decoded
  once: the batch product), ``export`` (capped surface, flow extensions, VTP, SWC, Slicer
  markups), ``solver`` (an svZeroDSolver model), ``cli``.

``docs/README.md`` indexes the documentation; ``docs/vmtk-guide.md`` describes the system for a
vmtk user.
"""
__version__ = "0.0.1"

from .errors import ThalwegError  # noqa: E402
from .graph import TubeGraph  # noqa: E402

__all__ = ["ThalwegError", "TubeGraph", "__version__"]
