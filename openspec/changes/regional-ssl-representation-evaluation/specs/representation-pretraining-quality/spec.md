# representation-pretraining-quality Specification

## Purpose

Define the Level-1 evidence battery that establishes whether a self-supervised
encoder trained on the regional (Amirkabir–Khuzestan) field-dynamics archive
has learned usable structure — independently of any downstream task verdict and
without touching the locked test split or the holdout-excluded fields.

## ADDED Requirements

### Requirement: Development-only pretraining evidence

The system SHALL execute every Level-1 metric on development clean cycles only,
with the same frozen split manifest (`splits_locked_v2.json`, sha
`30ecbe10…`), with the holdout-excluded fields excluded, with train-split-only
normalization, and with three recorded seeds. The locked test split SHALL NOT
be read by any Level-1 code path.

#### Scenario: Test-split access attempt
- **WHEN** any Level-1 module requests cycles of split `test`
- **THEN** the loader raises an error naming the allowed entry point and the
  run stops with a non-zero exit code

#### Scenario: Split manifest drift
- **WHEN** the split manifest or the cycle file hash differs from the
  recorded one
- **THEN** the pretraining evaluation fails closed and reports the mismatch

### Requirement: Masked reconstruction quality

The system SHALL report masked NRMSE per channel, computed only over observed
(bin, channel) pairs, per seed, with mean ± standard deviation across seeds and
a bootstrap confidence interval over dev cycles.

#### Scenario: Missing bin
- **WHEN** a (bin, channel) pair is unobserved
- **THEN** it contributes to neither the loss, the score, nor any denominator
  of the reported NRMSE

### Requirement: Farm-identity separability probe

The system SHALL report kNN (k ∈ {1, 5}) top-1 and top-5 farm-classification
accuracy of frozen embeddings on dev clean cycles, against chance performance
(1/490 and 5/490) and against kNN on identical-split B1 feature vectors.

#### Scenario: Probe comparability
- **WHEN** the farm-separability probe is run for the SSL encoder and for the
  B1 feature baseline
- **THEN** both use identical cycle subsets, normalization context and metric
  definitions, and the report records the shared subset hash

### Requirement: Phenological-phase information probe

The system SHALL report a linear-probe macro-F1 and accuracy for predicting the
36-bin days-since-harvest (or active `PhaseAxis`) bin from frozen embeddings,
trained on the train split only, compared against the majority baseline and
against the same probe on B1 feature vectors.

#### Scenario: Axis consistency
- **WHEN** the phase axis is not `DSH` (e.g. GDD or landmark, if the
  phenology-method-v2 work selects it)
- **THEN** the probe uses the same configured axis as the training path and
  records it in the output manifest

### Requirement: Embedding temporal stability

The system SHALL report, per field with ≥2 complete clean cycles, the
distribution (median, IQR, tail fraction below a pre-registered floor) of
cosine similarity between cycle-level embeddings, alongside a shuffled-field
null distribution.

#### Scenario: Insufficient cycles for a field
- **WHEN** a field has fewer than two complete clean dev cycles
- **THEN** that field is excluded from the stability statistic and the
  excluded count is reported

### Requirement: Collapse diagnostics

The system SHALL compute per-dimension embedding standard deviation, the
effective rank of the embedding covariance matrix, and alignment/uniformity
statistics on clean dev data. A candidate whose effective rank falls below the
pre-registered threshold, or whose per-dimension std spectrum is degenerate,
SHALL be disqualified at the screening stage and SHALL NOT proceed to
downstream evaluation.

#### Scenario: Degenerate representation detected
- **WHEN** the collapse diagnostic triggers for a screened candidate
- **THEN** the candidate is marked `disqualified-collapse`, the diagnostic
  values are recorded, and downstream probes are not run for that candidate

### Requirement: Provenance and reproducibility

Every Level-1 output SHALL carry the git SHA, config hash, split hashes, seed
list, software versions, device, wall time, metric values with CIs, and the
dataset identifiers required to reproduce it. Outputs SHALL be written as
machine-readable JSON (plus a human-readable Markdown rendering).

#### Scenario: Re-derivation
- **WHEN** a reviewer re-runs the Level-1 suite from the committed
  configuration on the recorded data versions
- **THEN** metric values reproduce within the recorded seed spread and the
  report states the tolerated deviation
