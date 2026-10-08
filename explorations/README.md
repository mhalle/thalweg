# explorations

Work that lives with thalweg but is not on the main line: nothing here is ported into
`src/thalweg/` or promised as API. Scripts import the research helpers from `research/vessels/`
and run with haversack's environment, like the research scripts.

- `rendering/render_straight.py` - straightened 3D rendering of a vessel path directly from the
  field (per-sample display-to-world warp, no resampled volume), with an illustration style
  (`STYLE=npr`). Design note §12.5. Whether rendering belongs here or in sdfview is open.
- `dissection/` - aortic dissection from a dual-lumen field: intimal flap as a tie sheet,
  entry tears as topology, branch feed (true / false / both) as a lumen-side query.
  `dissection.py` runs a synthetic arch phantom with known tears and branch stubs (2/2
  tears and 4/4 feeds matched on 2026-09-25); `_dissect.py` holds the phantom;
  `dissection_figure.py` draws the arch, flap line profiles, areas and contacts. See
  `DISSECTION.md`. The real target is TotalSegmentator `aortic_dissection` (Dataset 716)
  on a CTA; no patient has been segmented yet.
