"""The whole arterial tree as vmtk input: vmtkCenterlines at the same scope as our centerline.py.

    python bench/vessels/vmtk_tree_prep.py [RUN]          (after centerline.py)

The main field component of the artery margin (field connectivity, as centerline.py uses), every
other component floored; marching cubes at 0 (the field's own zero set, unsmoothed, as in
vmtk_prep.py); the largest surface component. Source = our graph's root, targets = every tip.
Writes DATA/<RUN>_vmtk_tree_input.npz for vmtk_tree_run.py.
"""
import json, sys, time
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import marching_cubes
from _data import DATA
from _field import fine_field
from _topo import field_edges, components
from cases import store

run = sys.argv[1] if len(sys.argv) > 1 else "C3N-00704_ctpa0625"
G = json.load(open(DATA / f"{run}_lung_arteries_centerlines.json"))
margins, _, grid, clip = fine_field(store(run))
m = margins["lung_arteries"]
idx, r, c, _, _ = field_edges(m)
_, comp = components(len(idx), r, c)
keep = idx[comp == np.argmax(np.bincount(comp))]
lo = keep.min(0) - 2; hi = keep.max(0) + 3
sub = m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
mask = np.zeros(sub.shape, bool); mask[tuple((keep - lo).T)] = True
near = ndi.binary_dilation(mask, iterations=1)                     # the main component and its wall cells
f = np.pad(np.where(near, sub, -clip), 1, constant_values=-clip)
t0 = time.time()
verts, faces, _, _ = marching_cubes(f, level=0.0)
t_mc = time.time() - t0
# largest surface component
import scipy.sparse as sp
E = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]]])
n, cl = sp.csgraph.connected_components(sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(len(verts),) * 2))
big = np.argmax(np.bincount(cl))
vid = -np.ones(len(verts), int); vid[cl == big] = np.arange((cl == big).sum())
faces = vid[faces[(cl[faces] == big).all(1)]]; verts = verts[cl == big]
world = grid.to_world(verts - 1 + lo)
tips = np.array([nd["point"] for nd in G["nodes"] if nd["kind"] == "tip"])
np.savez(DATA / f"{run}_vmtk_tree_input.npz", verts=world, faces=faces, source=np.array(G["root"]), targets=tips)
print(f"whole tree: {len(world)} vertices / {len(faces)} triangles ({n} surface components, largest kept), {len(tips)} tips; MC {t_mc:.1f} s")
print("TIMING " + json.dumps({"surface_for_vmtk_tree": round(t_mc, 4), "tree_vertices": len(world), "tree_tips": len(tips)}))
