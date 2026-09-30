"""thalweg: centerlines, branches and wall geometry of tubular structures, read from a field.

A successor to vmtk that reads a segmentation model's continuous field (rankfield margins in a
ranked store) instead of a triangulated surface. Two layers (docs/vmtk-successor.md §6):

- ``thalweg.kernel``: numpy/scipy algorithms on arrays and grids (field sampling, field
  connectivity, the centerline tracer); knows nothing of files, names or formats;
- the pipeline: ``store`` (read a ranked store), ``centerlines`` (a structure's centerlines as a
  graph), ``graph`` (the ``.thalweg.json`` model), ``adapters`` (graph <-> vmtk's convention),
  ``branching`` (vmtk's grouping, frames and angles on the graph), ``measure`` (the branch table),
  ``case`` (one case decoded once: the batch product), ``export`` (capped surface, VTP, SWC,
  Slicer markups), ``cli``.

``thalweg.vmtk`` holds faithful ports of vmtk's centerline-only filters on vmtk's own convention,
checked against vmtk's output; vmtk itself is never imported.
"""
__version__ = "0.0.1"

from .errors import ThalwegError  # noqa: E402
from .graph import TubeGraph  # noqa: E402

__all__ = ["ThalwegError", "TubeGraph", "__version__"]
