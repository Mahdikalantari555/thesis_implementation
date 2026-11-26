# Spec — time2agri-pretexts

### Requirement: Three distinct pretext objectives
The system SHALL expose arms `time2agri_td`, `time2agri_fp`, `time2agri_ff` with
distinct `objective_spec()` estimands (classification of bin-gap class; regression
of dominant NDVI frequencies; reconstruction of masked future bins).

#### Scenario: Objective audit
- **WHEN** `assert_objectives_distinct` is run over the three specs
- **THEN** no two arms share (objective, target_source).

### Requirement: Own loss in mini lab
A model exposing `pretext_loss` SHALL be trained with that loss, not the generic
Barlow loss. Models without it SHALL keep existing behaviour.

#### Scenario: Barlow fallback
- **WHEN** an existing representation model is trained
- **THEN** the Barlow loss is used exactly as before.

### Requirement: Leakage guard for FP
FP frequency targets SHALL be computed only from observed train-period NDVI
samples. A target built from evaluation-period data SHALL be rejected.

#### Scenario: Eval contamination
- **WHEN** an eval-period sample is passed to the FP target builder with
  `allow_eval=False` (default)
- **THEN** a `ValueError` is raised.

### Requirement: Honest alias
The legacy contrastive arm SHALL be named `agri_view_contrastive`; `"time2agri"`
SHALL remain a deprecated alias resolving to it, with no silent semantic change.

### Requirement: L1-only claim
Reports from these arms SHALL be tagged L1 and SHALL NOT contain stress,
phase-anomaly or benchmark-winner claims.
