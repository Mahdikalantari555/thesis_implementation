# generalization-and-claim-semantics Specification

## Purpose

Bound every claim this evaluation can produce: the three generalization levels,
the anomaly → stress-candidate → stress semantic ladder, the pre-registered
decision gates, and the preservation of the repository's frozen evidence base.
Failure of any gate is a recorded scientific result, not an exception to hide.

## ADDED Requirements

### Requirement: Temporal generalization protocol

The system SHALL evaluate year-forward rolls strictly inside train+dev: for
each pre-registered cut year Y, models are refit (normalization included) on
cycles starting ≤ Y and validated on the following 12–24 months; B1 SHALL be
re-run identically for each roll; the locked test split SHALL NOT be read.

#### Scenario: Roll disjointness
- **WHEN** a roll is constructed
- **THEN** every validation cycle starts after every training cycle's cutoff,
  no cycle appears in both partitions, and the roll definition (cut year,
  window) is recorded in the manifest

#### Scenario: Normalization refit
- **WHEN** a temporal roll is trained
- **THEN** normalization statistics are recomputed from that roll's training
  cycles only and stored with the roll artifacts

### Requirement: Spatial generalization protocol

The system SHALL evaluate leave-one-block-out generalization over the 20
locked spatial blocks within train+dev; the 37 holdout-excluded test-block
fields SHALL be excluded from every roll; results SHALL report mean and spread
across rolls, not a single split.

#### Scenario: Holdout exclusion
- **WHEN** a spatial roll partitions blocks
- **THEN** no field from the holdout manifest appears in any partition, and a
  test asserts this for every roll

### Requirement: Geographic scope limitation

The system SHALL declare geographic generalization out of scope: one region,
one crop, one climate regime. Every findings statement quoting a result SHALL
carry the same-region claim ceiling, and no geographic-transfer claim SHALL be
made or implied.

#### Scenario: External-generalization wording
- **WHEN** the capstone report is written
- **THEN** it states that validity beyond the Amirkabir study region is
  untested, requires a new data contract, and is future work

### Requirement: Semantic ladder for outputs

The system SHALL classify every reported flag as (a) anomaly — deviation from
the field's phenology-aware normal at matched phase, supported by controlled
injection evidence; (b) stress candidate — anomalies corroborated by proxy
indicators, reported with the proxy's data-dependence disclosed; (c) stress —
NOT claimable without independent agronomic evidence. Synthetic-anomaly
performance SHALL never be presented as confirmed real-world stress detection.

#### Scenario: Proxy dependence disclosure
- **WHEN** an environmental indicator is evaluated against model flags
- **THEN** the report identifies whether the indicator shares inputs with the
  model (e.g. LST→CWSI_LST, radiation→ETa) and labels the comparison
  accordingly; shared-input comparisons are reported as consistency, never as
  independent validation

#### Scenario: Stress wording prohibition
- **WHEN** no independent agronomic evidence exists for a flagged event
- **THEN** the output uses "anomaly" or "stress candidate", never "stress",
  and the L4 rung is explicitly marked unmet

### Requirement: Validation-ladder tagging on claims

Every quantitative claim in the capstone report SHALL be tagged with its
validation-ladder rung (L1–L5) and the evidence supporting it. Current honest
baseline: L2 not cleared (finding 25); L5 unmet. Tags SHALL NOT be upgraded
without the evidence the rung requires.

#### Scenario: Rung audit
- **WHEN** the report is finalized
- **THEN** each claim line carries (rung, evidence artifact, CI) and the audit
  line is present in the capstone report

### Requirement: Pre-registered decision gates

The system SHALL implement the five pre-registered gates — G-A (pretraining
quality), G-B (label efficiency), G-C (transfer dominance), G-D (retrieval
usefulness), G-E (claim discipline) — with inputs frozen in the committed
pre-registration before any result is observed. Gate outcomes SHALL be
computed mechanically from committed artifacts and SHALL NOT be edited after
the fact; amendments, if any, are recorded as such and named.

#### Scenario: All gates fail
- **WHEN** no candidate passes the gates
- **THEN** the capstone report records the negative result with equal
  prominence, keeps B1 as the reference algorithm, states the SSL status as
  open, and the change may still close once G-E passes

#### Scenario: Post-hoc gate weakening detected
- **WHEN** a gate threshold differs from the committed pre-registration
- **THEN** the report marks the affected result `invalid-posthoc` and the
  reviewer rejects the claim

### Requirement: Preservation of the frozen evidence base

This change SHALL NOT modify, retrain, recalibrate or reinterpret frozen
artifacts (gru_ae_main checkpoints and sidecars, B1, severity tables, locked
splits, holdout manifest, platform payload, findings 00–25). Regression checks
SHALL confirm byte-identical hashes for the artifacts this change references.

#### Scenario: Artifact mutation detected
- **WHEN** an artifact hash deviates from the recorded value at any stage
- **THEN** the run fails closed, the deviation is reported verbatim, and the
  change cannot proceed to the affected stage until resolved or recorded

### Requirement: Honest treatment of incomplete work

Deferred, failed, or not-applicable cells SHALL remain visible with explicit
status and reason; the capstone report SHALL NOT present incomplete coverage as
a complete benchmark; and the final section SHALL state which questions the
evidence cannot answer and what would be required to answer them.

#### Scenario: Budget exhaustion
- **WHEN** the compute budget is exhausted before all cells complete
- **THEN** unfinished cells are marked `deferred` with reasons, the report
  carries a coverage statement, and no gate dependent on a deferred cell may
  report `pass`
