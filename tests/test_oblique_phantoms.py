"""The phantom suite's oblique setting, pinned: a Y turned off the lattice onto an anisotropic grid
is traced as well as on the lattice (validation/phantom_suite.py --oblique, docs/validation.md §1)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
import phantom_suite as PS  # noqa: E402


def test_an_oblique_y_on_an_anisotropic_grid():
    segs, truth = PS.y_phantom(60)
    pts = np.concatenate([[s[0], s[1]] for s in segs]).astype(float)
    rmax = max(max(s[2], s[3]) for s in segs)
    m, geo = PS.field_oblique(PS.chain_distance(segs), pts.min(0) - rmax - 4, pts.max(0) + rmax + 4)
    assert not np.allclose(np.linalg.norm(np.asarray(geo.directions), axis=1), 0.7)      # anisotropic
    r = PS.score("y60", m, geo, truth, 0.7, oblique=True)
    assert (r["traced_ends"], r["traced_junctions"]) == (3, 1)
    assert r["axis_error_mm_median"] < 0.02 and r["radius_error_mm_median"] < 0.03
    assert 0.97 < r["equivalent_diameter_over_true_diameter"] < 1.0
    assert all(abs(d - 30.0) < 1.5 for d in r["deflection_deg"])
