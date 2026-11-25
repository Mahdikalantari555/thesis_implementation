# continual-update-protocol Specification

## Purpose

Update the model with new seasons, farms or cycles from the previous version,
never from scratch by default, with forgetting bounded by a pre-registered gate.

## ADDED Requirements

### Requirement: Versioned, batched updates

Each model version SHALL record weights, config, data and split hashes,
train-only normalization statistics, memory snapshot hashes, git SHA and parent
version; updates SHALL be triggered by season end, a new farm batch or a drift
alarm, never per observation.

#### Scenario: New acquisition arrives
- **WHEN** one new scene is ingested mid-season
- **THEN** it is scored and triaged but weights do not change

### Requirement: Anti-forgetting mechanisms

The update SHALL support stratified experience replay (farm × year × stage ×
pattern), teacher–student distillation from the parent version, optional EWC,
and separate versioned normal and anomaly memories (the anomaly memory with
verified, unresolved and rejected partitions); replay SHALL draw only from
normal memory.

#### Scenario: Warm start with replay and distillation
- **WHEN** an update runs with `--init-from parent`
- **THEN** each batch mixes the configured replay fraction and the distillation loss is logged

### Requirement: Drift detection

The system SHALL compare new-data embeddings to the replay distribution
(MMD or energy distance) and input bands (KS), and SHALL route large drift to
review instead of an automatic update.

#### Scenario: Large drift
- **WHEN** drift exceeds its threshold
- **THEN** the update is held and the drift report is written

### Requirement: Regression-set acceptance gate

A fixed, hash-recorded regression set SHALL be evaluated before acceptance;
a version SHALL be accepted only if forgetting ≤ δ_forget and new-data
performance ≥ the warm-start reference − δ_new, both pre-registered; BWT,
forgetting and forward transfer SHALL be logged for every candidate.

#### Scenario: Update forgets too much
- **WHEN** forgetting exceeds δ_forget
- **THEN** the candidate is rejected and the parent stays active

### Requirement: No silent re-initialization

Training from scratch SHALL require an explicit `--from-scratch` flag with a
recorded reason, legitimate only as a pre-registered baseline arm.

#### Scenario: Compatible checkpoint exists
- **WHEN** a run starts without `--init-from` while a compatible parent exists
- **THEN** it fails closed unless `--from-scratch` is given

### Requirement: Versioned normalization and calibration statistics (r2)

Each model version SHALL bind its normalization statistics (arm, values, data
hash), per-component calibration distributions, sensor-offset table and
feature manifest. On update, normalization statistics SHALL stay frozen at the
parent's values by default; re-normalization SHALL require a drift report
showing a persistent input shift, SHALL create a new major version, and SHALL
re-score the regression set under the new statistics before acceptance.
Calibration statistics SHALL be recomputed per version with old and new
stored, and historical scores SHALL only be re-scored into a new versioned
table.

#### Scenario: Update without drift
- **WHEN** a seasonal update is trained and no persistent drift is reported
- **THEN** the new version uses the parent's normalization statistics unchanged
