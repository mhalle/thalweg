# Every CT series for a WHERE clause, e.g.: uv run --no-project --with idc-index python bench/vessels/idc_study.py "i.PatientID='MSB-02664'"
import sys
from idc_index import IDCClient
import pandas as pd
from _data import DATA
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20); pd.set_option("display.max_colwidth", 34)
c = IDCClient()
for n in ["ct_index", "contrast_index", "volume_geometry_index", "seg_index"]:
    c.fetch_index(n)
where = sys.argv[1]
df = c.sql_query(f"""
SELECT i.collection_id coll, i.PatientID pid, i.StudyDate date, i.SeriesNumber sn, i.SeriesDescription descr,
       TRY_CAST(ct.SliceThickness AS DOUBLE) thk, round(ct.PixelSpacing_row_mm,3) px,
       array_to_string(ct.ConvolutionKernel,'/') kernel, i.instanceCount n,
       round(TRY_CAST(ct.SliceThickness AS DOUBLE)*i.instanceCount) approx_cov_mm,
       array_to_string(cb.ContrastBolusAgent,'/') agent, vg.regularly_spaced_3d_volume reg3d,
       round(i.series_size_MB) MB, i.license_short_name lic,
       (SELECT count(*) FROM seg_index s WHERE s.segmented_SeriesInstanceUID = i.SeriesInstanceUID) segs,
       i.SeriesInstanceUID uid, i.crdc_series_uuid crdc
FROM index i LEFT JOIN ct_index ct USING (SeriesInstanceUID)
LEFT JOIN contrast_index cb USING (SeriesInstanceUID)
LEFT JOIN volume_geometry_index vg USING (SeriesInstanceUID)
WHERE i.Modality='CT' AND {where}
ORDER BY i.PatientID, i.StudyDate, i.SeriesNumber
""")
print(df.drop(columns=["uid","crdc"]).to_string(index=False))
df.to_csv(DATA / (sys.argv[2] if len(sys.argv) > 2 else "study.csv"), index=False)
