"""vtkvmtkPolyBallLine's vectorized pieces: the pruning (candidate_pairs) against brute force."""
import numpy as np
import pytest

from thalweg.vmtk import Centerlines
from thalweg.vmtk import polyball as PB

R = "MaximumInscribedSphereRadius"


def _random_tree(rng, n_lines=6, n=40):
    """Random-walk polylines sharing a root, radii 0.2-2.5 with a few zeros; some zero-length steps."""
    lines, radii = [], []
    for _ in range(n_lines):
        steps = rng.normal(size=(n, 3)) * rng.uniform(0.1, 1.0) + rng.normal(size=3) * 0.5
        steps[rng.random(n) < 0.05] = 0.0                     # coincident consecutive points
        p = np.cumsum(steps, axis=0)
        p[0] = 0.0
        r = rng.uniform(0.2, 2.5) * np.exp(rng.normal(size=n) * 0.2)
        r[rng.random(n) < 0.05] = 0.0
        lines.append(p)
        radii.append(r)
    return Centerlines.from_lines(lines, radii)


def _near_walls(rng, seg, m):
    """Points on the spheres swept along random segments, pushed in or out by up to 5 %."""
    k = rng.integers(0, len(seg), m)
    t = rng.uniform(-0.1, 1.1, m)[:, None]                    # past the ends too: the end caps
    c = seg.p0[k] + np.clip(t, 0, 1) * (seg.p1[k] - seg.p0[k])
    r = seg.r0[k] + np.clip(t[:, 0], 0, 1) * (seg.r1[k] - seg.r0[k])
    u = rng.normal(size=(m, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return c + u * (r * rng.uniform(0.95, 1.05, m))[:, None]


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("block", [1, 5, 16])
def test_candidate_pairs_keep_every_nonpositive_pair(seed, block):
    rng = np.random.default_rng(seed)
    cl = _random_tree(rng)
    seg = PB.tube_segments(cl, R)
    x = np.concatenate([_near_walls(rng, seg, 3000), cl.points, rng.uniform(-15, 15, size=(300, 3))])
    qi, si = PB.candidate_pairs(x, seg, block=block)
    order = qi * len(seg) + si
    assert np.all(np.diff(order) > 0)                          # unique, sorted by (point, segment)
    kept = np.zeros((len(x), len(seg)), bool)
    kept[qi, si] = True
    v = PB.segment_values(x, seg)                              # brute force, every pair
    assert (v <= 0.0).sum() > 1000                             # the test does probe the walls
    missed = (v <= 0.0) & ~kept
    assert not missed.any(), f"{missed.sum()} dropped pairs have a value <= 0"
    assert kept.mean() < 0.6                                   # and it does prune
