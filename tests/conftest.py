"""Shared test fixtures: the vmtk oracles and the case data.

- ``phantom_oracle``: the synthetic tree's vmtk outputs, frozen in tests/fixtures (always present);
- ``case_oracle``: the C3N-00704 subtree's vmtk outputs in $VESSELS_DATA/oracle/<run>/ (skipped when
  absent; regenerate with tests/oracle/vmtk_centerline_oracle.py);
- ``vessels_data``: $VESSELS_DATA itself (default ~/tmp/data/vessels), skipped when absent.

Data never enters git: only the phantom fixtures (a few hundred KB) do. The data tests need, in
$VESSELS_DATA: ``runs/C3N-00704_ctpa0625.lung_vessels.duckn.zip`` and
``runs/MSB-02664_ctape0625_v013.lung_vessels.duckn.zip`` (haversack stores);
``C3N-00704_ctpa0625_{lung_arteries,lung_veins,lung_airways}_centerlines.json`` and
``MSB-02664_ctape0625_v013_lung_arteries_centerlines_voxel.json`` (research/vessels/centerline.py);
``C3N-00704_straight.npz`` (research/vessels/straighten.py); ``oracle/C3N-00704_ctpa0625/``
(tests/oracle/vmtk_centerline_oracle.py). Missing files skip their tests.
"""
import os
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
DATA = Path(os.environ.get("VESSELS_DATA", Path.home() / "tmp/data/vessels"))
CASE_RUN = "C3N-00704_ctpa0625"


class Oracle:
    """One oracle directory: ``oracle["extract"]`` is the stage's npz (loaded once)."""

    def __init__(self, root: Path):
        self.root = root
        self._cache = {}

    def __getitem__(self, stage):
        import numpy as np
        if stage not in self._cache:
            self._cache[stage] = np.load(self.root / f"{stage}.npz")
        return self._cache[stage]

    @property
    def meta(self):
        import json
        return json.loads((self.root / "oracle.json").read_text())


@pytest.fixture(scope="session")
def phantom_oracle():
    return Oracle(FIXTURES / "vmtk_oracle" / "phantom")


@pytest.fixture(scope="session")
def case_oracle():
    root = DATA / "oracle" / CASE_RUN
    if not (root / "oracle.json").exists():
        pytest.skip(f"no case oracle at {root}")
    return Oracle(root)


@pytest.fixture(scope="session", params=["phantom", pytest.param("case", marks=pytest.mark.data)])
def oracle(request):
    """Both oracles in turn (the case one skips when its data is absent)."""
    if request.param == "phantom":
        return Oracle(FIXTURES / "vmtk_oracle" / "phantom")
    root = DATA / "oracle" / CASE_RUN
    if not (root / "oracle.json").exists():
        pytest.skip(f"no case oracle at {root}")
    return Oracle(root)


@pytest.fixture(scope="session")
def vessels_data():
    if not DATA.exists():
        pytest.skip(f"no case data at {DATA}")
    return DATA


# -- the small synthetic vmtk layouts (tests/oracle/vmtk_centerline_oracle.py ``SMALL``) ------------
# Each isolates a splitting / grouping / fallback rule the phantom never reaches; frozen like the
# phantom, so those rules are tested without case data. They skip the per-line stages (see the
# oracle script); a test asking for a stage a fixture lacks is skipped.

SMALL_ORACLES = sorted(p.name for p in (FIXTURES / "vmtk_oracle").iterdir()
                       if p.is_dir() and p.name != "phantom")


class FrozenOracle(Oracle):
    """A fixture oracle whose missing stages skip the test instead of failing it."""

    def __getitem__(self, stage):
        if not (self.root / f"{stage}.npz").exists():
            pytest.skip(f"{self.root.name} has no {stage} stage")
        return super().__getitem__(stage)


@pytest.fixture(scope="session", params=["phantom", *SMALL_ORACLES])
def frozen_oracle(request):
    """The phantom and every small layout in turn (always present)."""
    return FrozenOracle(FIXTURES / "vmtk_oracle" / request.param)


@pytest.fixture(scope="session",
                params=["phantom", pytest.param("case", marks=pytest.mark.data), *SMALL_ORACLES])
def vmtk_oracle(request):
    """Every vmtk oracle: the phantom, the case (skipped when its data is absent), every small layout."""
    if request.param == "case":
        root = DATA / "oracle" / CASE_RUN
        if not (root / "oracle.json").exists():
            pytest.skip(f"no case oracle at {root}")
        return Oracle(root)
    return FrozenOracle(FIXTURES / "vmtk_oracle" / request.param)
