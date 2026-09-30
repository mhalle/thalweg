"""The ``vmtk_*`` flag convention across the ports (see thalweg.vmtk's docstring).

Every public entry point defaults to the correct behavior, lists its flags in ``VMTK_FLAGS``,
documents each in its module docstring, and each flag changes the output where its defect bites.
"""
import inspect
import math
import sys
from pathlib import Path

import numpy as np
import pytest

import thalweg.vmtk as V
from thalweg.vmtk import Centerlines, ReferenceSystems

EXPECTED = {
    "centerline_attributes": ("vmtk_two_point_cells",),
    "resample_centerlines": ("vmtk_cell_data",),
    "extract_branches": ("vmtk_steps", "vmtk_merge", "vmtk_float32", "vmtk_last_tract"),
    "bifurcation_reference_systems": ("vmtk_float32",),
    "offset_attributes": ("vmtk_interp",),
    "merge_centerlines": ("vmtk_float32", "vmtk_cell_data"),
    "smooth_centerlines": ("vmtk_float32",),
    "centerline_geometry": ("vmtk_float32",),
    "branch_geometry": ("vmtk_float32", "vmtk_steps", "vmtk_discard_smoothing"),
    "bifurcation_vectors": ("vmtk_interp", "vmtk_fallback", "vmtk_steps", "vmtk_float32"),
}
# defects that need a layout no public-level fixture here has; each is pinned by the named test
# (vmtk_merge was, until the hairpin fixture: it bites there together with vmtk_last_tract)
COVERED_ELSEWHERE: dict[tuple[str, str], str] = {}
HAIRPIN = Path(__file__).parent / "fixtures" / "vmtk_oracle" / "hairpin"

ALL = [(name, flag) for name, flags in EXPECTED.items() for flag in flags]


def test_flag_table():
    assert V.VMTK_FLAGS == EXPECTED
    for name in EXPECTED:
        assert name in V.__all__ and callable(getattr(V, name))


@pytest.mark.parametrize("name,flag", ALL)
def test_flag_defaults_false_and_documented(name, flag):
    f = getattr(V, name)
    assert inspect.signature(f).parameters[flag].default is False
    assert flag in sys.modules[f.__module__].__doc__, f"{flag} not documented in {f.__module__}"


# -- each flag bites ------------------------------------------------------------------------------

def _arrays(obj):
    out = {"points": obj.points}
    if isinstance(obj, Centerlines):
        out["cells"] = np.concatenate(obj.cells) if obj.cells else np.zeros(0)
        out.update({"cd:" + k: v for k, v in obj.cell_data.items()})
    out.update({"pd:" + k: v for k, v in obj.point_data.items()})
    return out


def _differ(a, b) -> bool:
    x, y = _arrays(a), _arrays(b)
    return x.keys() != y.keys() or any(
        x[k].shape != y[k].shape or not np.array_equal(x[k], y[k], equal_nan=True) for k in x)


def _degenerate_lines():
    rng = np.random.default_rng(0)
    L0 = np.cumsum(rng.normal(size=(12, 3)) * 0.4 + [0.5, 0, 0], axis=0)
    lines = [L0, np.repeat(L0[3:4], 3, axis=0), L0[:9] + [0, 0, 1], L0[:7] + [0, 1, 0]]
    cl = Centerlines.from_lines(lines, [np.ones(len(L)) for L in lines])
    cl.cell_data["CenterlineIds"] = np.arange(len(lines), dtype=np.int32)
    return cl


def _split_with_a_collapsed_cell(ext: Centerlines) -> Centerlines:
    """Split centerlines with one more cell, first, whose points all coincide (vtkCleanPolyData makes it
    a vertex, numbered before the lines)."""
    p = np.concatenate([np.repeat(ext.points[:1], 3, axis=0), ext.points])
    cells = [np.arange(3)] + [c + 3 for c in ext.cells]
    pd = {k: np.concatenate([np.repeat(v[:1], 3, axis=0), v]) for k, v in ext.point_data.items()}
    g = int(ext.cell_data["GroupIds"].max()) + 1
    first = {"GroupIds": g, "CenterlineIds": 0, "TractIds": 0, "Blanking": 0}
    cd = {k: np.concatenate([np.array([first.get(k, v[0])], v.dtype), v]) for k, v in ext.cell_data.items()}
    return Centerlines(p, cells, pd, cd)


def _short_parent_y():
    """A Y whose parent is shorter than its radius: no touching sphere upstream (vmtk's fallback)."""
    s = np.arange(0, 10.0001, 0.1)[:, None]
    parent = np.array([-0.5, 0, 0]) + np.arange(0, 0.5001, 0.1)[:, None] * np.array([1.0, 0, 0])
    da, db = np.array([1.0, 1, 0]) / math.sqrt(2), np.array([1.0, -1, 0.3]) / math.sqrt(2.09)
    t = np.arange(0, 1.0001, 0.1)[:, None]
    lines = [parent, t * da * 2, 2 * da + s * da, parent, t * db * 2, 2 * db + s * db]
    cl = Centerlines.from_lines(lines, [np.full(len(L), 0.95) for L in lines])
    cl.cell_data = {"GroupIds": np.array([0, 1, 2, 0, 1, 3]), "CenterlineIds": np.array([0, 0, 0, 1, 1, 1]),
                    "TractIds": np.array([0, 1, 2, 0, 1, 2]), "Blanking": np.array([0, 1, 0, 0, 1, 0])}
    rs = ReferenceSystems(np.zeros((1, 3)), {"Normal": np.array([[0.0, 0, 1]]),
                                             "UpNormal": np.array([[1.0, 0, 0]]), "GroupIds": np.array([1])})
    return cl, rs


def _scenario(name, flag, o):
    """(positional args, extra keywords) of a call where ``flag`` bites (phantom oracle unless crafted)."""
    L = lambda s: Centerlines.from_npz(o[s])  # noqa: E731
    if name == "resample_centerlines":
        return (_degenerate_lines(), 0.3), {}
    if name == "extract_branches":
        if flag in ("vmtk_merge", "vmtk_last_tract"):          # a daughter back beside the root
            return (Centerlines.from_npz(HAIRPIN / "attributes.npz"),), {}
        return (L("attributes"),), {}
    if name == "centerline_attributes":                        # a two-point centerline
        return (Centerlines.from_lines([np.array([[0.0, 0, 0], [1, 2, 3]]), np.eye(3)]),), {}
    if name == "bifurcation_reference_systems":
        return (L("extract"),), {}
    if name == "offset_attributes":
        E = L("extract")
        return (E, V.bifurcation_reference_systems(E), -1), {}
    if name == "merge_centerlines":
        if flag == "vmtk_cell_data":
            return (_split_with_a_collapsed_cell(L("extract")), 0.5), {"merge_blanked": False}
        return (L("extract"),), {}
    if name in ("smooth_centerlines", "centerline_geometry"):
        return (L("input"),), {}
    if name == "branch_geometry":
        return (L("extract"),), ({"line_smoothing": True} if flag == "vmtk_discard_smoothing" else {})
    if name == "bifurcation_vectors":
        if flag == "vmtk_fallback":
            return _short_parent_y(), {}
        return (L("extract"), ReferenceSystems.from_npz(o["frames"])), {}
    raise KeyError(name)


@pytest.mark.parametrize("name,flag", [x for x in ALL if x not in COVERED_ELSEWHERE])
def test_flag_changes_the_result_where_the_defect_bites(phantom_oracle, name, flag):
    args, kw = _scenario(name, flag, phantom_oracle)
    f = getattr(V, name)
    vmtk = {g: True for g in EXPECTED[name]}
    a = f(*args, **kw, **vmtk)
    b = f(*args, **kw, **{**vmtk, flag: False})
    assert _differ(a, b)


def test_covered_elsewhere_exists():
    for ref in COVERED_ELSEWHERE.values():
        path, test = ref.split("::")
        assert f"def {test}(" in (Path(__file__).parent / path).read_text(), ref
