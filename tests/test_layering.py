"""The two layers stay apart (docs/vmtk-successor.md §6), checked on the source, not at runtime.

- ``thalweg.kernel``: numpy, scipy, skimage (the zero set), optionally torch, rankfield's geometry
  value type, and the standard library minus its file formats. Nothing that knows files, names,
  formats or the command line.
- ``thalweg.vmtk``: numpy and the standard library (same exclusions), plus its own modules.
  Never vmtk or VTK.
- Neither imports the pipeline (store, graph, centerlines, cli).
"""
import ast
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).parents[1] / "src" / "thalweg"
# no file formats, no command line
STDLIB = set(sys.stdlib_module_names) - {"sqlite3", "tkinter", "json", "gzip", "zipfile", "csv", "pickle",
                                         "tarfile", "shelve", "argparse", "configparser", "xml"}
KERNEL_OK = {"numpy", "scipy", "skimage.measure", "torch", "rankfield.geometry"} | STDLIB
VMTK_OK = {"numpy"} | STDLIB
PIPELINE = {"store", "graph", "centerlines", "cli", "errors", "adapters", "branching", "measure", "case",
            "export"}


def imports(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name, 0
        elif isinstance(node, ast.ImportFrom):
            yield (node.module or ""), node.level


def allowed(name: str, ok: set) -> bool:
    return any(name == o or name.startswith(o + ".") for o in ok)


@pytest.mark.parametrize("layer,ok", [("kernel", KERNEL_OK), ("vmtk", VMTK_OK)])
def test_layer_imports(layer, ok):
    bad = []
    for f in sorted((SRC / layer).glob("*.py")):
        for name, level in imports(f):
            if level == 1:                                  # a sibling in the same layer
                continue
            if level > 1 or name.startswith("thalweg"):
                top = name.split(".")[1] if name.startswith("thalweg.") else name.split(".")[0]
                if level > 1 or top in PIPELINE or (top != layer):
                    bad.append(f"{f.name}: {'.' * level}{name}")
                continue
            if not allowed(name, ok):
                bad.append(f"{f.name}: {name}")
    assert not bad, f"{layer} imports outside its layer: {bad}"
