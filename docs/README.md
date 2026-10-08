# thalweg documentation

*Index updated 2026-10-08.* Each document is marked as **reference** (describes the code as it is
and is kept current), **record** (a dated account of research or decisions, kept as written, with
a status note at its top), **proposal** (a plan, partly built), or **draft** (parked).

## Start here

| Document | Kind | What it is for |
|---|---|---|
| [`../README.md`](../README.md) | reference | What thalweg is, install, the verbs, the layout |
| [`vmtk-guide.md`](vmtk-guide.md) | reference | thalweg for vmtk users: every vmtk tool and array mapped to thalweg, the differences, and the vmtk port layer for vmtk developers. The fullest description of the system |

## Reference

| Document | What it covers |
|---|---|
| [`format/thalweg-json.md`](format/thalweg-json.md) | The `.thalweg.json` graph format, its invariants and every key written; the tables, mesh sidecar and other outputs beside it. [`format/thalweg-0.1.schema.json`](format/thalweg-0.1.schema.json) is the JSON Schema (`thalweg schema`) |
| [`validation.md`](validation.md) | How thalweg behaves beyond the case it was built on: phantoms, second patients, non-vessel tubes, the vmtk comparisons, the tracer options and the defaults chosen, and every later feature's measurements. The source of the numbers quoted elsewhere |
| [`port-plan.md`](port-plan.md) | vmtk, module by module, and what thalweg carries for each (status: ported / done / todo / out); the data-model requirements for tubes; the structure-specific tier; open and settled decisions |

## Records

| Document | Date | What it records |
|---|---|---|
| [`vmtk-successor.md`](vmtk-successor.md) | 2026-09-23/24 | The design note written during incubation: why a successor, the module-by-module redesign, and the first measurements (§12, the research prototype). The decisions it left open were all taken; see its status note |
| [`vmtk-vs-field-method.md`](vmtk-vs-field-method.md) | 2026-09-24 | A two-page comparison of vmtk and the field method at the end of incubation. Superseded by `vmtk-guide.md` for anyone learning the system; current numbers are in `validation.md` |
| [`slicerheart-opportunities.md`](slicerheart-opportunities.md) | 2026-09-24 | Where the method could plug into SlicerHeart, module by module, with three pilots. Not built |

## Proposals and drafts

| Document | Kind | Status |
|---|---|---|
| [`deliverables.md`](deliverables.md) | proposal | The batch product per case (three tiers). Tier 1 is largely built (`thalweg run`); its status note lists what is missing |
| [`aorta.md`](aorta.md) | draft | Aorta landmark measurements. Parked 2026-10-01 as a research topic; nothing built |

## Elsewhere in the repository

- `../explorations/README.md`: work kept with thalweg but off the main line (straightened
  rendering, vascular catchments and lung segments, aortic dissection), each with its own notes.
- `../research/vessels/`: the incubation scripts, frozen; the reference the kernel ports reproduce.
- `../validation/`: the scripts behind `validation.md`.
- Module docstrings in `../src/thalweg/` hold the algorithm-level detail, including every vmtk
  defect flag and its measured effect (`src/thalweg/vmtk/*.py`).
