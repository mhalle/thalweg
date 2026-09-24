"""Stage-by-stage timing, vmtk vs the field method, on one machine in one session.

    python bench/vessels/bench_stages.py [RUN] [ROUNDS]

Runs every stage script (each prints one `TIMING {json}` line) in ROUNDS rounds, alternating
which side goes first, then vmtkCenterlines on the whole tree once. Records the 1-minute load
average before each script (other work shares the machine). Writes DATA/<RUN>_bench_stages.json.
"""
import json, os, subprocess, sys, time
from pathlib import Path
from _data import DATA

run = sys.argv[1] if len(sys.argv) > 1 else "C3N-00704_ctpa0625"
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OURS = [str(ROOT.parent / "haversack/.venv/bin/python")]
VMTK = ["uv", "run", "--no-project", "--python", "3.12", "--with", "vmtk", "--with", "scipy", "python"]
ours = ["vmtk_prep.py", "centerline.py", "branch_partition.py", "wall_map.py", "curvature.py", "flow_ext.py"]
vmtk = ["vmtk_run.py", "vmtk_branch.py", "vmtk_mapping.py", "vmtk_curvature.py", "vmtk_flowext.py"]
out = {"runs": []}


def go(side, script, timeout=3600):
    cmd = (OURS if side == "ours" else VMTK) + [str(HERE / script), run]
    load = os.getloadavg()[0]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    wall = time.time() - t0
    timing = {}
    for line in p.stdout.splitlines():                  # vmtk prints progress without a newline: TIMING can be mid-line
        if "TIMING " in line:
            timing.update(json.loads(line.split("TIMING ", 1)[1]))
    with open(DATA / f"{run}_bench_stages.raw.log", "a") as f:
        f.write(f"==== {side} {script}\n{p.stdout}\n")
    rec = dict(side=side, script=script, load=round(load, 2), wall=round(wall, 2), ok=p.returncode == 0, timing=timing)
    out["runs"].append(rec)
    print(json.dumps(rec), flush=True)
    if p.returncode:
        print(p.stderr[-2000:], flush=True)
    json.dump(out, open(DATA / f"{run}_bench_stages.json", "w"), indent=1)


for k in range(rounds):
    order = [("ours", ours), ("vmtk", vmtk)] if k % 2 == 0 else [("vmtk", vmtk), ("ours", ours)]
    for side, scripts in order:
        for s in scripts:
            go(side, s)
go("vmtk", "vmtk_tree_run.py", timeout=2700)
