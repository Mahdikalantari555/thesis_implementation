## ADDED Requirements

### Requirement: Patch-level controlled PoC

The system SHALL train a patch-level arm `patch_cnn_gru` (2D CNN stem →
per-location GRU → causal multi-horizon input-space prediction k=1–3 +
masked-reconstruction auxiliary loss, ≈90k parameters, covariates, validity
masks, GroupNorm, no pooled latent) and a CNN-off control `patch_pixelwise`
(same objective and budget, no stem) on patch tensors from patches.zarr,
first on a pre-registered hashed subset of locked train-split fields, with
runtime estimation before any run, per-epoch checkpointing with `--resume`,
progress.jsonl, device auto-select and smoke mode; and SHALL evaluate both
arms through the unchanged severity protocol (S1–S6 × {acute, persistent} ×
{ndvi, savi}, identical injection sets as findings 18/22, 5% cycle-FAR
calibration, MDM80 with ">S6" censoring) on the locked dev subset, with
per-pixel phenophase-binned σ standardization and top-20% field
aggregation feeding the existing alarm rule. The pre-registered PoC gates
(G1 smoke plumbing; G2 overfit-one-batch; G3 subset floor — MDM80 finite
AND better than the best of {b1, gru_ae_main} with a paired bootstrap CI
over fields excluding zero, success = any finite MDM80 while those arms are
censored; G4 CNN kept only if it improves MDM80 or pixel-localization
AUPRC over patch_pixelwise; G5 estimate-before-run and ≤2× gru_ae_main
compute) SHALL be evaluated verbatim; scale-up to the full train set SHALL
happen only if G1–G5 pass; every claim SHALL state ladder rung L1–L2 and
the verdict SHALL be recorded as finding 23 (Persian). Test-split data
SHALL be read only by the locked final-test path.

#### Scenario: Owner runs training on own hardware

- **WHEN** the user launches train_patch.py on CPU or another machine
- **THEN** `--estimate-runtime` prints per-epoch, per-seed, total wall-time
  and disk BEFORE training, `--resume` restores model/optimizer/scheduler/
  RNG with a config-hash check, progress.jsonl and a final wall-time
  summary are written, and no agent-launched long job occurs

#### Scenario: Subset-first scale gate

- **WHEN** the PoC subset run finishes
- **THEN** the five gates are evaluated with the locked severity machinery
  and a written verdict decides whether the full-train run is launched,
  with the honest-negative protocol applying to every gate

#### Scenario: Spatial value decided, not assumed

- **WHEN** patch_cnn_gru and patch_pixelwise are compared at equal budget
- **THEN** the CNN stem is kept only if MDM80 or pixel-localization AUPRC
  improves, and otherwise the pixel-wise variant continues with the
  negative recorded

#### Scenario: Identical-condition comparability

- **WHEN** the severity protocol runs for the patch arms and the existing
  arms
- **THEN** all arms share the same injections (same δ ladder, families,
  hashes), the same calibration rules and the same alarm machinery, and
  results are reported in one table with Wilson CIs and MDM80 censoring
  semantics
