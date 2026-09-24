"""Where the vessel benches keep data: OUTSIDE the repo and outside Dropbox.

medseg lives in Dropbox, so anything written next to these scripts syncs (the decoded
cache alone is ~1 GB). Caches and search results go to $VESSELS_DATA
(default ~/tmp/data/vessels); downloaded DICOM goes to $VESSELS_DICOM
(default ~/tmp/data/idc_vessels). Neither is backed up; both regenerate.
"""
import os
from pathlib import Path

DATA = Path(os.environ.get("VESSELS_DATA", Path.home() / "tmp/data/vessels"))
DICOM = Path(os.environ.get("VESSELS_DICOM", Path.home() / "tmp/data/idc_vessels"))
DATA.mkdir(parents=True, exist_ok=True)
DICOM.mkdir(parents=True, exist_ok=True)
