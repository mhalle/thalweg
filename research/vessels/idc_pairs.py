# Thin+thick reconstruction pairs in one CT study (IDC v24). Run: uv run --no-project --with idc-index python bench/vessels/idc_pairs.py
from idc_index import IDCClient
import pandas as pd
from _data import DATA
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 40)
c = IDCClient()
for n in ["ct_index", "contrast_index", "volume_geometry_index", "seg_index"]:
    c.fetch_index(n)
q = """
WITH s AS (
  SELECT i.collection_id, i.PatientID, i.StudyInstanceUID, i.SeriesInstanceUID, i.SeriesDescription,
         i.BodyPartExamined, i.license_short_name, i.Manufacturer, i.ManufacturerModelName,
         i.instanceCount, i.series_size_MB, i.crdc_series_uuid,
         TRY_CAST(ct.SliceThickness AS DOUBLE) AS thk, ct.PixelSpacing_row_mm AS px,
         array_to_string(ct.ConvolutionKernel, '/') AS kernel, ct.ReconstructionDiameter AS fov,
         array_to_string(ct.ImageType, '/') AS itype,
         (len(cb.ContrastBolusAgent) > 0 OR len(cb.ContrastBolusRoute) > 0) AS contrast_tag,
         vg.regularly_spaced_3d_volume AS regular
  FROM index i
  JOIN ct_index ct USING (SeriesInstanceUID)
  LEFT JOIN contrast_index cb USING (SeriesInstanceUID)
  LEFT JOIN volume_geometry_index vg USING (SeriesInstanceUID)
  WHERE i.Modality = 'CT' AND vg.regularly_spaced_3d_volume
    AND array_to_string(ct.ImageType, '/') LIKE '%AXIAL%'
    AND i.collection_id <> 'nlst'
)
SELECT t.collection_id, t.PatientID, t.StudyInstanceUID,
       t.SeriesDescription AS thin_desc, t.thk AS thin_thk, t.px AS thin_px, t.kernel AS thin_kernel, t.instanceCount AS thin_n,
       k.SeriesDescription AS thick_desc, k.thk AS thick_thk, k.kernel AS thick_kernel,
       t.BodyPartExamined AS body, t.contrast_tag OR k.contrast_tag AS contrast, t.license_short_name AS lic,
       t.ManufacturerModelName AS scanner, t.series_size_MB AS thin_MB,
       t.SeriesInstanceUID AS thin_uid, k.SeriesInstanceUID AS thick_uid,
       (t.kernel = k.kernel) AS same_kernel, (t.fov = k.fov) AS same_fov
FROM s t JOIN s k ON t.StudyInstanceUID = k.StudyInstanceUID AND t.SeriesInstanceUID <> k.SeriesInstanceUID
WHERE t.thk <= 0.8 AND k.thk >= 2.0
"""
df = c.sql_query(q)
df.to_parquet(DATA / "pairs.parquet")
print("pairs:", len(df), " studies:", df.StudyInstanceUID.nunique())
print(df.groupby("collection_id").agg(studies=("StudyInstanceUID","nunique"), contrast=("contrast","mean"),
      same_kernel=("same_kernel","mean")).sort_values("studies", ascending=False).head(40).to_string())
