"""Centerlines of the airway tree, with the frozen research/vessels/centerline.py unchanged.

    cd explorations/catchments && ../../../haversack/.venv/bin/python airway_centerlines.py [RUN]

centerline.py only knows lung_arteries and lung_veins: _field.fine_field decodes margins for those
two, and its class table has no airway entry. The airway lumen (`lung_airways`) is in the same fine
layer of the lung_vessels store, so this wrapper adds its margin to what fine_field returns and
runs centerline.py's source with the class table extended by one entry, in memory. Everything else
(field connectivity, TEASAR, ridge refinement, segment tree, chord repair) is the reference code.
Writes DATA/<RUN>_lung_airways_centerlines.json.
"""
import sys
from pathlib import Path
import numpy as np
import rankfield as rf

R = Path(__file__).resolve().parents[2] / "research" / "vessels"
sys.path.insert(0, str(R))
import _field                                                          # noqa: E402

run = sys.argv[1] if len(sys.argv) > 1 else "C3N-00704_ctpa0625"
_decode = _field.fine_field


def fine_field(store):
    margins, labels, grid, clip = _decode(store)
    root = _field.open_store(Path(store), "r").root
    segs = root.attrs.asdict()["duckn"]["extensions"]["seg"]["segments"]
    parts = _field.parts_of(root)
    k = len(parts) - 1
    value = {s["name"]: _field._value(s) for s in segs
             if _field._value(s) is not None and not s.get("members") and int(s.get("layer", 0)) == k}
    code = parts[k].field
    margins["lung_airways"] = rf.margin(code, code.labels.index(value["lung_airways"])).astype(np.float32)
    return margins, labels, grid, clip


_field.fine_field = fine_field
src = (R / "centerline.py").read_text()
old = 'code = {"lung_arteries": 3, "lung_veins": 4}[cls]'
assert old in src
src = src.replace(old, 'code = {"lung_airways": 1, "lung_arteries": 3, "lung_veins": 4}[cls]')
sys.argv = [str(R / "centerline.py"), run, "lung_airways"]
exec(compile(src, str(R / "centerline.py"), "exec"), {"__name__": "__main__", "__file__": str(R / "centerline.py")})
