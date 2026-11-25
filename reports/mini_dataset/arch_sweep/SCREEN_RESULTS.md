# CPU architecture × spatial-unit screen — 2026-11-25

**Stage:** screen (1 seed 13, 100 optimiser steps per run, equal compute) ·
**Farms:** L02-02 and R12-08 from `3farms-dev.geojson` (provenance alias applied) ·
**Store manifest:** see `screen/sweep.json` · **72 configs, 70 succeeded, 2 failed** ·
wall 518.9 s · claim **L2 screening only**.

Machine-readable: `screen/sweep.json` (per-seed values and aggregates folded in).
Per-run report/checkpoint directories remain local and are regenerable; binaries
never go to GitHub.

## What was varied
- **Spatial units:** single pixel; 5×5 patches of size 1, 3, 5, 7, 9, 11 Landsat pixels; full-field aggregate.
- **Architectures:** all 9 in `MODEL_REGISTRY`: mlp_ae, gru_ae, lstm_ae, cnn_ae, convlstm_ae, mamba_ae, cnn_mamba, barlow_twins, jepa.
- **Controls:** random-init same architecture and a step detector in every run; equal 100-step budget enforced by `ModelConfig.max_steps` (verified `steps_done=100`).

## Ranking by gain over random-init control (top 10 of 70)

| rank | unit__model | synthetic AUPRC | random-init | gain |
|---|---|---|---|---|
| 1 | patch11__mamba_ae | 0.6066 | 0.4936 | +0.1131 |
| 2 | patch7__mamba_ae | 0.6032 | 0.4915 | +0.1117 |
| 3 | patch9__mamba_ae | 0.6033 | 0.4916 | +0.1116 |
| 4 | patch1__mamba_ae | 0.6039 | 0.4942 | +0.1097 |
| 5 | patch3__mamba_ae | 0.6016 | 0.4941 | +0.1075 |
| 6 | patch5__mamba_ae | 0.5997 | 0.4924 | +0.1073 |
| 7 | patch11__gru_ae | 0.6175 | 0.5272 | +0.0903 |
| 8 | patch9__gru_ae | 0.6113 | 0.5253 | +0.0860 |
| 9 | full_field__mamba_ae | 0.6251 | 0.5397 | +0.0854 |
| 10 | patch5__gru_ae | 0.6095 | 0.5247 | +0.0849 |

Patch size has little effect on the control-adjusted gain for mamba_ae
(+0.107 → +0.113 across sizes 1–11); the apparent patch-size effect in raw
AUPRC is mostly a random-init baseline shift.

## Highest absolute AUPRC (not an SSL result)

| unit__model | AUPRC | random-init | gain |
|---|---|---|---|
| pixel__jepa | 0.6876 | 0.7461 | −0.0585 |
| pixel__barlow_twins | 0.6783 | 0.6987 | −0.0204 |
| patch1__barlow_twins | 0.6753 | 0.6912 | −0.0158 |
| patch3__barlow_twins | 0.6590 | 0.6685 | −0.0095 |
| patch5__barlow_twins | 0.6444 | 0.6180 | +0.0263 |

The representation-family models look strongest in absolute terms but their
**random-init controls are stronger still** — an architecture prior, not a
pretraining gain. This is exactly the confound the research report warned about.

## Failures (verbatim)
- `full_field__barlow_twins`: `Expected more than 1 value per channel when training, got input size torch.Size([1, 192])`
- `full_field__jepa`: `'NoneType' object is not subscriptable`

Both are batch-size-1 failures in the full-field arm; not silently dropped.

## Caveats that block any claim
- **Single seed**. No CI, no significance; ranking is provisional.
- **Unequal capacity**: parameters range from ~2.5k (mlp_ae) to ~82k (cnn_mamba) and depend slightly on input width. Equal compute, not equal parameters.
- **Target is a magnitude proxy**: the injected deviation is a −1.0 NDVI drop on held-out windows. The harness report states it is L2 controlled-deviation evidence, **not** phase-aware anomaly or stress detection.
- **Provenance**: canonical 490-field IDs remain shifted; only this mini store carries the verified alias.

## Next
Confirmation stage: top models (mamba_ae, gru_ae, cnn_mamba) × top units
(pixel, patch9, patch11, full-field) × 3 seeds × bigger models
(d_model 64, latent 32, 2 layers, 400 steps) — plus the corrected-store
weak-label probe, which currently shows **no** SSL advantage
(`representation_first_pilot_fixed.json`: SSL AUPRC 0.1108 = random-init 0.1108,
raw features 0.1782; retrieval SSL 0.1533 below chance 0.1800).
