"""The ``thalweg`` command line: verbs over the Python API.

    thalweg structures STORE                              what a store names
    thalweg centerlines STORE -s NAME [-s NAME ...] -o OUT.thalweg.json[.gz]
                        [--ridge-passes N] [--prune length|wall] [--root inlet|deepest]
    thalweg table GRAPH STORE -o BRANCHES.parquet [--stations STATIONS.parquet] [-s NAME] [--step MM]
    thalweg run STORE -o DIR [-s NAME ...]                the batch product: graph, tables, summary, QC
    thalweg export GRAPH STORE -s NAME [--mesh M.vtp] [--vmtk-centerlines C.vtp] [--swc T.swc]
                   [--markups M.mrk.json] [--wall-maps W.npz]
                   [--bifurcation-sections S.parquet] [--zero-d M.json]   at least one output
    thalweg summary GRAPH                                 structures, counts, lengths
    thalweg schema [-o FILE]                              the .thalweg.json JSON Schema
"""
from __future__ import annotations

import json
import sys
import time

import click

from . import __version__
from .errors import ThalwegError


class _Group(click.Group):
    """Every verb: a ThalwegError (a request thalweg cannot satisfy) is a one-line error, not a trace."""

    def invoke(self, ctx):
        try:
            return super().invoke(ctx)
        except ThalwegError as e:
            raise click.ClickException(str(e)) from None


def _method_options(f):
    f = click.option("--ridge-passes", type=click.IntRange(min=1), default=4, show_default=True,
                     help="Coarse-to-fine ridge refinement passes. 4 (the default) removes the one-pass "
                          "refinement's radius deficit (0.03-0.07 mm) and matches vmtk; 1 is the research "
                          "reference, ~1.9x faster to trace. docs/validation.md.")(f)
    f = click.option("--root", type=click.Choice(["inlet", "deepest"]), default="inlet", show_default=True,
                     help="Root each tree at its inlet (the widest end, or an end running off the field "
                          "if at least half as wide) or at the tracer's deepest point (the reference).")(f)
    f = click.option("--prune", type=click.Choice(["length", "wall"]), default="length", show_default=True,
                     help="Spur rule: 'length' (the reference) or 'wall' (also drops terminal branches that "
                          "do not protrude beyond the parent's wall: flat-lumen lobes, and 9-26 % of vessel "
                          "tips).")(f)
    return f


@click.group(cls=_Group, context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="thalweg")
def main():
    """Centerlines, branches and wall geometry of tubular structures, read from a model's field."""


@main.command()
@click.argument("store", type=click.Path(exists=True))
def structures(store):
    """List the structures a ranked store names, with the part each lives in."""
    from .store import open_store
    st = open_store(store)
    click.echo(f"{st.path.name}: labeling scheme {st.labeling_scheme}, parts {st.part_indices}")
    for s in sorted(st.structures, key=lambda s: (s.part, s.label_value)):
        click.echo(f"  part {s.part}  value {s.label_value:4d}  {s.name}")


@main.command()
@click.argument("store", type=click.Path(exists=True))
@click.option("-s", "--structure", "names", multiple=True, required=True,
              help="Structure to trace (repeatable), e.g. lung_arteries.")
@click.option("-o", "--output", required=True, type=click.Path(dir_okay=False),
              help="Output graph, .thalweg.json or .thalweg.json.gz.")
@click.option("--part", type=int, default=None,
              help="Part to read (default: the finest holding the structure).")
@click.option("--graph", type=click.Choice(["field", "voxel"]), default="field", show_default=True,
              help="Connectivity: decided by the field, or the 26-connected labelmap (comparison only).")
@click.option("-q", "--quiet", is_flag=True, help="No progress messages.")
@_method_options
def centerlines(store, names, output, part, graph, quiet, ridge_passes, prune, root):
    """Trace seed-free centerline trees of STORE's structures into one graph file.

    Only the largest connected piece of each structure is traced; the others are listed in the
    structure's statistics (component_lattice_point_counts)."""
    from .centerlines import centerline_graph, combine
    from .store import open_store
    t0 = time.time()

    def log(msg):
        if not quiet:
            click.echo(f"[{time.time() - t0:6.1f}s] {msg}", err=True)

    if len(set(names)) != len(names):
        raise click.UsageError(f"a structure is named twice: {', '.join(names)}")
    try:
        st = open_store(store)
        graphs = []
        for n in names:
            log(f"{n}: decoding and tracing")
            graphs.append(centerline_graph(st, n, part=part, graph=graph, ridge_passes=ridge_passes,
                                           prune=prune, root=root, log=lambda m, n=n: log(f"{n}: {m}")))
        g = graphs[0] if len(graphs) == 1 else combine(graphs)
        g.write(output)
    except ThalwegError as e:
        raise click.ClickException(str(e))
    for s in g.structures:
        st_ = s.statistics
        log(f"{s.name}: {st_['edge_count']} edges, {st_['tip_count']} tips, "
            f"{st_['truncated_end_count']} truncated ends, "
            f"{st_['junction_count']} junctions, {st_['length_mm'] / 10:.1f} cm")
    log(f"wrote {output}")


@main.command()
@click.argument("graph", type=click.Path(exists=True))
@click.argument("store", type=click.Path(exists=True))
@click.option("-o", "--output", required=True, type=click.Path(dir_okay=False),
              help="Branch table, .parquet.")
@click.option("-s", "--structure", "names", multiple=True,
              help="Structures to measure (default: all in GRAPH).")
@click.option("--stations", type=click.Path(dir_okay=False), default=None,
              help="Also write the per-station section profile here (.parquet).")
@click.option("--step", type=click.FloatRange(min=0, min_open=True), default=1.0, show_default=True,
              help="Section spacing along each branch, mm.")
def table(graph, store, output, names, stations, step):
    """Measure every branch of GRAPH's structures in STORE's field: one row per branch (Parquet;
    needs pyarrow, the `tables` extra). Column definitions: thalweg.measure (lobe columns:
    thalweg.lobes). The bronchoarterial pairing columns are written by `thalweg run` only."""
    from .centerlines import check_source
    from .graph import TubeGraph
    from .case import Case
    from .lobes import annotate, lobe_fields, lobe_rows
    from .measure import branch_table, write_table
    from .store import open_store
    g = TubeGraph.read(graph)
    missing = [n for n in names if n not in {s.name for s in g.structures}]
    if missing:
        raise ThalwegError(f"{graph} has no structure {', '.join(missing)}; it has "
                           f"{', '.join(s.name for s in g.structures)}")
    st = open_store(store)
    case = Case(st)
    try:
        lobes = lobe_fields(st)
        g, edge_lobe = annotate(g, lobes)
    except ThalwegError:
        lobes, edge_lobe = None, {}
    rows, prof = [], [] if stations else None
    try:
        for s in g.structures:
            if names and s.name not in names:
                continue
            m, geo, ref = case.margin(s.name, s.source.part)
            check_source(s, geo, ref)
            outer = case.outer(s.name, s.source.part, m)
            mine = branch_table(g, s.name, m, geo, step=step, stations_out=prof, outer=outer)
            if lobes is not None:
                lobe_rows(mine, edge_lobe[s.name])
            rows.extend(mine)
    except ThalwegError as e:
        raise click.ClickException(str(e))
    write_table(rows, output)
    click.echo(f"{len(rows)} branches -> {output}", err=True)
    if stations:
        write_table(prof, stations)
        click.echo(f"{len(prof)} stations -> {stations}", err=True)


@main.command()
@click.argument("store", type=click.Path(exists=True))
@click.option("-o", "--output", required=True, type=click.Path(file_okay=False), help="Output directory.")
@click.option("-s", "--structure", "names", multiple=True,
              help="Structure to process (repeatable). Default: lung_arteries, lung_veins, lung_airways.")
@click.option("--step", type=click.FloatRange(min=0, min_open=True), default=1.0, show_default=True,
              help="Section spacing along each branch, mm.")
@click.option("--no-stations", is_flag=True, help="Skip the per-station profile table.")
@click.option("--branch-volumes", is_flag=True,
              help="Add each branch's volume (the branch partition of the field; about a third more time).")
@click.option("-q", "--quiet", is_flag=True, help="No progress messages.")
@_method_options
def run(store, output, names, step, no_stations, branch_volumes, quiet, ridge_passes, prune, root):
    """The batch product for one case: graph.thalweg.json.gz, branches.parquet, stations.parquet,
    summary.json and qc.json in OUTPUT.

    With a lung_vessels store it also gives: a lobe per branch, airway wall measures and Pi10,
    bronchoarterial pairing, an artery/vein plausibility check, the radius interval per point and
    whole-tree statistics. `thalweg centerlines` alone writes none of those."""
    from pathlib import Path
    from .case import Case
    from .measure import write_table
    t0 = time.time()

    def log(msg):
        if not quiet:
            click.echo(f"[{time.time() - t0:6.1f}s] {msg}", err=True)

    names = names or ("lung_arteries", "lung_veins", "lung_airways")
    if len(set(names)) != len(names):
        raise click.UsageError(f"a structure is named twice: {', '.join(names)}")
    out = Path(output)
    try:
        case = Case.open(store)
        for n in names:
            case.store.ref(n)                                  # every structure exists, before any output
        res = case.run(names, step=step, stations=not no_stations, branch_volumes=branch_volumes, log=log,
                       ridge_passes=ridge_passes, prune=prune, root=root)
    except ThalwegError as e:
        raise click.ClickException(str(e))
    out.mkdir(parents=True, exist_ok=True)
    res["graph"].write(out / "graph.thalweg.json.gz")
    write_table(res["rows"], out / "branches.parquet")
    if res["stations"] is not None:
        write_table(res["stations"], out / "stations.parquet")
    (out / "summary.json").write_text(json.dumps(res["summary"], indent=1) + "\n")
    (out / "qc.json").write_text(json.dumps(res["qc"], indent=1) + "\n")
    for n, q in res["qc"]["structures"].items():
        log(f"{n}: {res['summary'][n]['edge_count']} branches, {q['length_mm'] / 10:.1f} cm, "
            f"{q['truncated_end_count']} truncated ends, "
            f"{q['dropped_components']['component_count']} pieces dropped, "
            f"{q['length_outside_field_mm']:.1f} mm outside")
    log(f"wrote {out}")


def _cap_kinds(ctx, param, value):
    from .export import END_KINDS
    kinds = tuple(k.strip() for k in value.split(",") if k.strip())
    bad = [k for k in kinds if k not in END_KINDS]
    if bad or not kinds:
        raise click.BadParameter(f"{value!r}: give a comma-separated list of {', '.join(END_KINDS)}")
    return kinds


@main.command()
@click.argument("graph", type=click.Path(exists=True))
@click.argument("store", type=click.Path(exists=True))
@click.option("-s", "--structure", "name", required=True, help="The structure to export.")
@click.option("--mesh", type=click.Path(dir_okay=False), default=None,
              help="Closed surface with a flat, named cap at the ends of --cap-kinds (.vtp, cell data "
                   "BoundaryId: 0 the wall, k the k-th cap), plus <mesh>.boundaries.json: each cap's name, "
                   "end node, plane center, outward normal, inscribed radius, area and centroid, and every "
                   "end that got no cap with the reason.")
@click.option("--cap-kinds", default="tip,truncated", show_default=True, callback=_cap_kinds,
              help="Which ends get a cap: tip (a free end), truncated (cut off by the grid's edge; a root "
                   "on the grid's edge counts), root (a degree-1 root: the inlet).")
@click.option("--refine", type=click.IntRange(min=1), default=1, show_default=True,
              help="Mesh the field's trilinear interpolant this many times finer.")
@click.option("--vmtk-centerlines", "vmtk_out", type=click.Path(dir_okay=False), default=None,
              help="Source-to-tip centerlines in vmtk's convention, split into vmtk's groups (.vtp: vmtk's "
                   "centerline attributes + branch extractor), with vmtk's defects corrected unless "
                   "--vmtk-exact.")
@click.option("--vmtk-exact", is_flag=True,
              help="Reproduce vmtk's own numbers, defects included (every vmtk_* flag of the two stages).")
@click.option("--swc", type=click.Path(dir_okay=False), default=None, help="The tree as SWC.")
@click.option("--markups", type=click.Path(dir_okay=False), default=None,
              help="One 3D Slicer curve per branch (.mrk.json).")
@click.option("--curvature", is_flag=True,
              help="With --mesh: the wall's mean curvature from the field, point array MeanCurvature (1/mm).")
@click.option("--distance-to-centerlines", "with_distance", is_flag=True,
              help="With --mesh: point arrays DistanceToCenterlines and CenterlineRadius (mm), vmtk's.")
@click.option("--flow-extensions", "extension_ratio", type=click.FloatRange(min=0, min_open=True),
              default=None,
              help="With --mesh: replace each cap by a flow extension this many ring radii long "
                   "(vmtk's adaptive length; vmtk's own default is 10).")
@click.option("--extension-transition", type=click.FloatRange(0, 1), default=0.25, show_default=True,
              help="The share of an extension's length over which the ring becomes a circle.")
@click.option("--zero-d", "zero_d_out", type=click.Path(dir_okay=False), default=None,
              help="An svZeroDSolver input (.json): Poiseuille vessels, junctions, placeholder boundary "
                   "conditions (CGS units).")
@click.option("--inflow", type=click.FloatRange(min=0, min_open=True), default=1.0, show_default=True,
              help="With --zero-d: the steady inflow at the root, mL/s.")
@click.option("--outlet-resistance", type=click.FloatRange(min=0), default=1000.0, show_default=True,
              help="With --zero-d: every outlet's resistance, dyn s / cm^5.")
@click.option("--bifurcation-sections", "sections_out", type=click.Path(dir_okay=False), default=None,
              help="vmtk's bifurcation sections, cut from the field (.parquet; needs pyarrow).")
@click.option("--distance-spheres", type=click.IntRange(min=1), default=1, show_default=True,
              help="How many touching spheres from each bifurcation its sections lie.")
@click.option("--wall-maps", "wall_maps_out", type=click.Path(dir_okay=False), default=None,
              help="Wall maps r(arc length, angle) of every edge, ray-cast from the field (.npz).")
@click.option("--wall-map-step", type=click.FloatRange(min=0, min_open=True), default=0.5, show_default=True,
              help="Station spacing of the wall maps, mm.")
def export(graph, store, name, mesh, cap_kinds, refine, vmtk_out, vmtk_exact, swc, markups, curvature,
           with_distance, extension_ratio,
           extension_transition, zero_d_out, inflow, outlet_resistance, sections_out, distance_spheres,
           wall_maps_out, wall_map_step):
    """Export one structure: a capped surface for CFD, vmtk-compatible centerlines, SWC, Slicer
    markups and/or wall maps (give at least one output)."""
    from .graph import TubeGraph
    if (extension_ratio is not None or curvature or with_distance) and not mesh:
        raise click.UsageError("--flow-extensions, --curvature and --distance-to-centerlines need --mesh")
    if extension_ratio is None and extension_transition != 0.25:
        raise click.UsageError("--extension-transition needs --flow-extensions")
    if not (mesh or vmtk_out or swc or markups or wall_maps_out or sections_out or zero_d_out):
        raise click.UsageError("nothing to export: give --mesh, --vmtk-centerlines, --swc, --markups, "
                               "--bifurcation-sections, --wall-maps and/or --zero-d")
    g = TubeGraph.read(graph)
    s = g.structure(name)
    if zero_d_out:
        from .solver import write_zero_d_model, zero_d_model
        model = zero_d_model(g, name, inflow=inflow, outlet_resistance=outlet_resistance)
        write_zero_d_model(model, zero_d_out)
        click.echo(f"{zero_d_out}: svZeroDSolver input, {len(model['vessels'])} vessels, "
                   f"{len(model['junctions'])} junctions, {len(model['boundary_conditions']) - 1} outlets "
                   "(placeholder boundary conditions)", err=True)
    if sections_out:
        from .branching import bifurcation_sections, vmtk_branching
        from .centerlines import check_source
        from .measure import write_table
        from .store import open_store
        m, geo, ref = open_store(store).margin(name, s.source.part)
        check_source(s, geo, ref)
        b = vmtk_branching(g, name, vmtk_compatible=vmtk_exact)
        rows = bifurcation_sections(b, m, geo, distance_spheres, vmtk_compatible=vmtk_exact)
        if not rows:
            raise ThalwegError(f"{name}: vmtk's branching finds no bifurcation to section")
        write_table(rows, sections_out)
        closed = sum(r["closed"] for r in rows)
        click.echo(f"{sections_out}: {len(rows)} bifurcation sections ({closed} closed)", err=True)
    if wall_maps_out:
        from .centerlines import check_source
        from .store import open_store
        from .wallmap import ostium, outside_stations, wall_maps, write_wall_maps
        m, geo, ref = open_store(store).margin(name, s.source.part)
        check_source(s, geo, ref)
        maps = wall_maps(g, name, m, geo, step=wall_map_step)
        write_wall_maps(maps, wall_maps_out, name)
        rays = sum(w.radius_mm.size for w in maps.values())
        lost = sum(int(outside_stations(w).sum()) for w in maps.values())
        open_ = sum(int((ostium(w) & ~outside_stations(w)[:, None]).sum()) for w in maps.values())
        click.echo(f"{wall_maps_out}: {len(maps)} edges, {rays} rays, {open_ / max(rays, 1):.1%} through "
                   f"an ostium, {lost} stations outside the structure", err=True)
    if mesh:
        from .centerlines import check_source
        from .export import capped_surface, write_vtp_mesh
        from .store import open_store
        m, geo, ref = open_store(store).margin(name, s.source.part)
        check_source(s, geo, ref)
        msh = capped_surface(g, name, m, geo, kinds=cap_kinds, refine=refine)
        hits = None
        if extension_ratio is not None:
            from .export import extension_collisions, flow_extensions
            msh = flow_extensions(msh, ratio=extension_ratio, transition=extension_transition)
            hits = extension_collisions(msh, m, geo)
        import numpy as np
        tube = np.zeros(len(msh.vertices), bool)                  # vertices of the flow extensions
        for e in msh.extensions.values():
            tube[e["vertices"][0]:e["vertices"][1]] = True
        if curvature:
            from .kernel.curvature import mean_curvature
            h = np.full(len(msh.vertices), np.nan)
            h[~tube] = mean_curvature(m, geo, msh.vertices[~tube], clip=open_store(store).clip(s.source.part))
            msh.point_data["MeanCurvature"] = h                 # the extensions are not the field's surface
        if with_distance:
            from .partition import distance_to_centerlines
            dist, rad = distance_to_centerlines(msh.vertices, g, name)
            msh.point_data["DistanceToCenterlines"] = dist
            msh.point_data["CenterlineRadius"] = rad
        write_vtp_mesh(msh, mesh)
        n_ends = len(msh.caps) + len(msh.skipped)
        more = f", {len(msh.skipped)} not (see below)" if msh.skipped else ""
        click.echo(f"{mesh}: {len(msh.faces)} faces, {n_ends} ends ({'/'.join(cap_kinds)}): "
                   f"{len(msh.caps)} capped{more}", err=True)
        if hits is not None:
            bad = sorted(k for k, v in hits.items() if v)
            click.echo(f"  {len(hits)} flow extensions, {extension_ratio:g} ring radii long; {len(bad)} run "
                       "back into the structure (extension_vertices_inside_structure in the sidecar)",
                       err=True)
        for sk in msh.skipped:
            click.echo(f"  not capped: {sk}", err=True)
    if vmtk_out:
        from .adapters import to_vmtk
        from .export import write_vtp_centerlines
        from .vmtk import VMTK_FLAGS, centerline_attributes, extract_branches
        paths = to_vmtk(g, name)                   # a tree with edges, or ThalwegError
        if paths.centerlines.n_cells == 0:
            raise ThalwegError(f"{name}: no source-to-tip path is long enough for vmtk's centerlines")
        try:
            cl = centerline_attributes(paths.centerlines,
                                       **{f: vmtk_exact for f in VMTK_FLAGS["centerline_attributes"]})
            split = extract_branches(cl, **{f: vmtk_exact for f in VMTK_FLAGS["extract_branches"]})
        except ThalwegError:
            raise
        except ValueError as e:                    # the vmtk ports speak ValueError
            raise ThalwegError(f"{name}: vmtk's branch extraction failed: {e}") from e
        write_vtp_centerlines(split, vmtk_out)
        click.echo(f"{vmtk_out}: {split.n_cells} cells in vmtk's groups", err=True)
    if swc:
        from .export import write_swc
        write_swc(g, name, swc)
        click.echo(f"{swc}: SWC", err=True)
    if markups:
        from .export import write_slicer_markups
        write_slicer_markups(g, name, markups)
        click.echo(f"{markups}: Slicer markups", err=True)


@main.command()
@click.argument("graph", type=click.Path(exists=True))
def summary(graph):
    """Structures, node and edge counts, and total length of a .thalweg.json file."""
    from .graph import TubeGraph
    g = TubeGraph.read(graph)
    click.echo(f"{graph}: format {g.format} {g.version}, {len(g.nodes)} nodes, {len(g.edges)} edges, "
               f"{len(g.points.position)} points")
    for s in g.structures:
        kinds = {}
        for nd in g.nodes:
            if nd.structure == s.name:
                kinds[nd.kind] = kinds.get(nd.kind, 0) + 1
        length = sum(e.length_mm for e in g.edges if e.structure == s.name)
        click.echo(f"  {s.name}: {sum(e.structure == s.name for e in g.edges)} edges, "
                   + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items())) + f", {length / 10:.1f} cm")


@main.command()
@click.option("-o", "--output", type=click.Path(dir_okay=False), default=None,
              help="Write here instead of stdout.")
def schema(output):
    """Print the JSON Schema of the .thalweg.json format."""
    from .graph import json_schema
    text = json.dumps(json_schema(), indent=1) + "\n"
    if output:
        open(output, "w").write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
