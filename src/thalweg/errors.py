"""thalweg's exception type: what went wrong, said in the caller's terms.

The pipeline raises :class:`ThalwegError` (a ``ValueError``) for requests it cannot satisfy - a
missing structure, an empty field, an unreadable or invalid file, a graph that is not the tree a
consumer needs - and the command line turns it into a one-line error. The kernel and the vmtk
ports stay free of pipeline types and raise ``ValueError``; the pipeline converts at its boundary.
"""


class ThalwegError(ValueError):
    """A request thalweg cannot satisfy (a missing structure, an empty field, a bad file)."""
