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

### Requirement: Farm-boundary isolation of patch inputs and scores

The patch arms SHALL treat the farm polygon as the analysis boundary: values
of pixels with `in_field = 0` (padding and neighbouring fields inside the
field's bounding box) SHALL be masked — never passed to the stem as values —
so that the CNN stem consumes `concat[X·M_in, M_in]` with
`M_in = pixel_valid ∧ in_field`, and no convolution, norm, or aggregation
SHALL be able to mix observations from different farms. The score of an
in-field pixel SHALL be invariant to any change in out-of-field or
other-field pixels inside the same 32×32 patch. The patch arm's spatial
modules (per-location GRU, GroupNorm, the R3a dilation branches) are
individually justified: GroupNorm is retained because masked pixels corrupt
batch statistics (and, independently, because it carries no cross-field
state); the temporal module runs per pixel location with no spatial
coupling; the multi-scale dilation variant is enabled only under the
invariance test above. Adjacent farms SHALL NOT be treated as spatially
connected even where their boundaries touch.

#### Scenario: Out-of-field pixels cannot influence an in-field score

- **WHEN** the values of pixels with `in_field = 0` are perturbed, or a
  synthetic neighbouring field is placed inside the patch's bounding box
- **THEN** the per-pixel scores of all in-field pixels are byte-identical,
  and a regression test enforces this for the stem, the multi-horizon heads
  and the R3a dilation variant

#### Scenario: The mask is the boundary

- **WHEN** a patch contains a neighbouring field's pixels inside the stored
  bounding box
- **THEN** those values never reach any model computation (masked, not
  zero-as-observed), and the in-field pixel count reported per patch equals
  the polygon mask's pixel count

#### Scenario: Identical-condition comparability

- **WHEN** the severity protocol runs for the patch arms and the existing
  arms
- **THEN** all arms share the same injections (same δ ladder, families,
  hashes), the same calibration rules and the same alarm machinery, and
  results are reported in one table with Wilson CIs and MDM80 censoring
  semantics
