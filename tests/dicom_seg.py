"""Build a small DICOM SEG for the tests: synthetic CT slices and a segmentation of them (highdicom)."""
import numpy as np


def ct_series(shape, spacing, origin, row_dir, col_dir, seed=0):
    """``shape`` = (slices, rows, cols) CT slices (pydicom Datasets) on an oblique frame: pixel
    (r, c) of slice k is at origin + c dx row_dir + r dy col_dir + k dz (row_dir x col_dir)."""
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
    row_dir, col_dir = np.asarray(row_dir, float), np.asarray(col_dir, float)
    normal = np.cross(row_dir, col_dir)
    dz, dy, dx = spacing
    study, series, frame = generate_uid(), generate_uid(), generate_uid()
    out = []
    for k in range(shape[0]):
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.MediaStorageSOPClassUID = CTImageStorage
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = generate_uid()
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study, series, frame
        ds.Modality = "CT"
        ds.PatientID, ds.PatientName, ds.PatientBirthDate, ds.PatientSex = "P1", "Test^Phantom", "", "O"
        ds.StudyID, ds.StudyDate, ds.StudyTime, ds.AccessionNumber = "1", "20260930", "120000", ""
        ds.ReferringPhysicianName, ds.SeriesNumber, ds.InstanceNumber = "", 1, k + 1
        ds.Manufacturer = "thalweg tests"
        ds.ImageOrientationPatient = [float(v) for v in (*row_dir, *col_dir)]
        ds.ImagePositionPatient = [float(v) for v in np.asarray(origin, float) + k * dz * normal]
        ds.PixelSpacing = [float(dy), float(dx)]
        ds.SliceThickness = float(dz)
        ds.Rows, ds.Columns = int(shape[1]), int(shape[2])
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.RescaleIntercept, ds.RescaleSlope = 0, 1
        ds.PixelData = np.zeros(shape[1:], np.int16).tobytes()
        out.append(pydicom.dataset.FileDataset("", ds, file_meta=ds.file_meta, preamble=b"\0" * 128))
    return out


def world_points(shape, spacing, origin, row_dir, col_dir):
    """The world position of every pixel of the series, (slices, rows, cols, 3)."""
    row_dir, col_dir = np.asarray(row_dir, float), np.asarray(col_dir, float)
    normal = np.cross(row_dir, col_dir)
    dz, dy, dx = spacing
    k, r, c = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    return (np.asarray(origin, float) + c[..., None] * dx * row_dir + r[..., None] * dy * col_dir
            + k[..., None] * dz * normal)


def write_seg(path, source, masks, labels, fractional=False):
    """A SEG of ``source`` with one segment per (slices, rows, cols) mask (binary, or fractional in
    [0, 1]), labeled ``labels``."""
    import highdicom as hd
    from pydicom.sr.codedict import codes
    from pydicom.uid import generate_uid
    descriptions = [hd.seg.SegmentDescription(
        segment_number=i + 1, segment_label=label, segmented_property_category=codes.SCT.Tissue,
        segmented_property_type=codes.SCT.Tissue, algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL)
        for i, label in enumerate(labels)]
    pixels = np.stack(masks, axis=-1)
    seg = hd.seg.Segmentation(
        source_images=source, pixel_array=pixels.astype(np.float32 if fractional else bool),
        segmentation_type=(hd.seg.SegmentationTypeValues.FRACTIONAL if fractional
                           else hd.seg.SegmentationTypeValues.BINARY),
        segment_descriptions=descriptions, series_instance_uid=generate_uid(), series_number=2,
        sop_instance_uid=generate_uid(), instance_number=1, manufacturer="thalweg tests",
        manufacturer_model_name="phantom", software_versions="0", device_serial_number="0")
    seg.save_as(str(path))
    return path
