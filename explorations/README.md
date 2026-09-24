# explorations

Work that lives with thalweg but is not on the main line: nothing here is ported into
`src/thalweg/` or promised as API. Scripts import the research helpers from `research/vessels/`
and run with haversack's environment, like the research scripts.

- `rendering/render_straight.py` - straightened 3D rendering of a vessel path directly from the
  field (per-sample display-to-world warp, no resampled volume), with an illustration style
  (`STYLE=npr`). Design note §12.5. Whether rendering belongs here or in sdfview is open.
