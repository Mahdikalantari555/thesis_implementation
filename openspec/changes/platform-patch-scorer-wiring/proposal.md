# Change: platform-patch-scorer-wiring

## Why

Finding 23 (2026-11-18): on the identical finding-18 severity protocol and eval
hash (6186edf8), both patch arms (`patch_cnn_gru`, `patch_pixelwise`) reach
**MDM80 = S1 on all four families** while every field-level arm (B1,
`gru_ae_main`, `gru_pred`) stays censored at **>S6**. The owner's goal is ONE
preliminary patch model trained on the FULL archive, brought up in the
validation platform, so the owner can use the platform's suspected-stress
flags as the field-visit checklist while personally collecting stress-type
data (the L4/A2 blocker).

`docs/platform_score_methods.md` §7 item 9 explicitly reserves this wiring as
a separate openspec change; this is that change. It is authored BEFORE the
G4 tie-break verdict and the full-archive severity results are final, so it
names the **winner-selection rule** rather than a hard-coded arm: the slot
backs the frozen `patch_*_main` checkpoint of the arm that wins the
pre-registered G4 rule ("CNN kept only if it beats patch_pixelwise on MDM80
or localization AUPRC", tasks.md of `patch-level-controlled-trial`) AND
reaches a finite MDM80 on the full-archive dev severity evaluation. If the
full-archive run censors (>S6) — the honest-floor outcome — this change is
NOT applied and the platform stays field-level.

## What Changes

- New platform score slot `patch_spatial` (name fixed at apply time to
  `patch_cnn_gru` or `patch_pixelwise` after the freeze), added to BOTH
  registry mirrors (`scripts/platform_channels.py`, `platform/src/data/channels.ts`)
  with provenance `model` and the trial-stage marker.
- `scripts/export_platform_data.py` gains a patch-scoring path: frozen
  checkpoint + frozen norm + frozen σ table + `patches.zarr`, scored
  causally per cycle bin and attributed to the observation dates inside
  each bin, with the full artifact-hash provenance chain in the payload.
- Payload schema: additive per-observation score array for the new slot (or
  `null` + `absent` where no patch cycle covers the date — 120 fields have
  no admissible cycles and stay absent), plus a per-score `stage` marker
  (`"trial"`) surfaced as a UI badge; contract-v5 additive implications are
  analyzed in design.md (no breaking change to existing slots).
- Suspected-stress flag source stays the frozen `gru_ae_main` until the
  owner approves switching it to the patch slot (separate task in this
  change, owner-gated).
- Persian/English copy for the trial badge via the `persian-writing` skill
  quality bar.

## Impact

- **Affected specs**: `model-output-ingest` (new slot + absent semantics +
  provenance honesty), `field-stress-monitoring` (trial-stage badge; flag
  source unchanged until the owner-gated task runs).
- **NOT affected**: tile server (`serve_tiles.py`), desk payload and desk
  view, legacy checkpoints, locked splits/holdout machinery.
- **Gates**: the 51 contract+bilingual pytest gate, `tsc --noEmit`, vite
  build, runtime screenshot gate; byte-identity of all existing payload
  fields outside the new slot.
- **Cost**: one full exporter run (~33 min) + patch inference over the 799
  admissible cycles; payload grows by one score array.
