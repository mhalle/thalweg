"""The thickness ladders: one reference (thinnest) run per patient and its thicker variants.

Every entry: (run name in DATA/runs, image source, nominal slice thickness mm, note). The
image source is a manifest (patient, series number) or a derived file in DICOM/derived.
"""
import csv
from _data import DATA, DICOM

LADDERS = {
    "C3N-00704": [
        ("C3N-00704_ctpa0625", ("C3N-00704", "4"), 0.625, "CTPA, STANDARD"),
        ("C3N-00704_lung125", ("C3N-00704", "5"), 1.25, "LUNG kernel (sharper)"),
        ("C3N-00704_slab2mm", "C3N-00704_slab2mm.nii.gz", 2.0, "boxcar slab average of the 0.625"),
        ("C3N-00704_std375", ("C3N-00704", "6"), 3.75, "STANDARD"),
    ],
    "MSB-02664": [
        ("MSB-02664_ctape0625_v013", ("MSB-02664", "304"), 0.625, "CTA-PE, STANDARD"),
        ("MSB-02664_std125", ("MSB-02664", "302"), 1.25, "STANDARD"),
        ("MSB-02664_slab2mm", "MSB-02664_slab2mm.nii.gz", 2.0, "boxcar slab average of the 0.625"),
        ("MSB-02664_dlir500", ("MSB-02664", "301"), 5.0, "STANDARD, deep-learning recon (DLIR M)"),
    ],
    # the demo: one acquisition only (2 mm slices at 1 mm, kernel B), no thin reference
    "idc-torso1": [
        ("DEMO:idc-torso1", "NIFTI:idc-torso1.nii", 2.0, "demo, nephrogenic phase, kernel B"),
    ],
}


def store(run):
    if run == "DEMO:idc-torso1":
        from pathlib import Path
        return Path.home() / "Dropbox/development/haversack/data/duckn_demo/idc-torso1/lung_vessels.duckn"
    return DATA / "runs" / f"{run}.lung_vessels.duckn.zip"


def image(src):
    if isinstance(src, str) and src.startswith("NIFTI:"):
        from pathlib import Path
        return Path.home() / "tmp/data" / src.split(":", 1)[1]
    if isinstance(src, str):
        return DICOM / "derived" / src
    rows = {(r["patient"], r["series_number"]): r for r in csv.DictReader(open(DICOM / "manifest.csv"))}
    return DICOM / rows[src]["path"]
