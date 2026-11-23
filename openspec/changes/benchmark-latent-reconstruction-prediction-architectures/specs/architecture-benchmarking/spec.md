# architecture-benchmarking Specification

## Purpose

Define the architecture layer of the benchmark: a common interface across all
supported models, the ten pre-registered configurations (five paradigms × two
backbones), latent-collapse safeguards, hierarchical patch/farm processing, and
the fairness reporting required for any cross-paradigm claim.

## ADDED Requirements

### Requirement: Unified architecture interface

The system SHALL provide a common interface for training, inference, scoring,
checkpointing and evaluation across all supported architectures, so that every
arm can be executed by the same registry, checkpointed with the same layout,
and evaluated by the same code path.

#### Scenario: Cross-architecture reuse
- **WHEN** an architecture is replaced by another A-id in the same experiment cell
- **THEN** only its model class changes; registry, checkpointing, scoring and
  metric code are identical

### Requirement: Ten architecture configurations

The system SHALL implement the five learning paradigms (input reconstruction,
latent reconstruction, future prediction, JEPA, reconstruction+prediction
hybrid) using both GRU and Mamba backbones, producing A1–A10 as defined in the
design; each configuration SHALL have recorded parameter counts within a
pre-registered capacity band (±20 %).

#### Scenario: Mamba feasibility gate
- **WHEN** an Mamba arm is numerically unstable, fails to converge, or exceeds
  twice the wall-time budget of its GRU twin on T4
- **THEN** the arm is marked `Not Applicable` with recorded evidence, and no
  other arm is silently substituted for it

### Requirement: Distinct learning objectives

Input reconstruction, latent reconstruction, future prediction and JEPA latent
prediction SHALL remain mathematically distinct: the objective, its target and
its gradient path SHALL NOT be aliased, and hybrid arms SHALL combine the
documented losses with pre-registered weights.

#### Scenario: Objective audit
- **WHEN** a trained A-id's manifest is inspected
- **THEN** the objective, its target source and (for hybrids) its λ values are
  recorded, and mixed-up objectives fail an automated audit test

### Requirement: Latent representation stability

Latent reconstruction and JEPA arms SHALL include a documented collapse-prevention
mechanism (stop-gradient, EMA target encoder or an equivalent validated design);
the system SHALL monitor per-dimension std, effective rank and alignment/
uniformity; a degenerate candidate SHALL be disqualified and SHALL NOT be
retuned until it passes.

#### Scenario: Collapse detected mid-run
- **WHEN** collapse diagnostics cross the pre-registered thresholds during an
  arm's run
- **THEN** the arm is marked `disqualified-collapse`, its checkpoints are
  retained for audit, and the registry records the triggering diagnostic values

### Requirement: Hierarchical spatial-temporal processing

The system SHALL support patch-level spatial encoding with farm-level
aggregation and SHALL emit both patch-level and farm-level anomaly scores via a
documented, architecture-consistent aggregation. Patch-level cells SHALL be
marked `Deferred` until the v2 patch store and its normalization sidecar are
bound; the field-mean path SHALL remain fully operational regardless.

#### Scenario: Patch prerequisite unmet
- **WHEN** the v2 patch store or its hash binding is absent
- **THEN** every patch-level experiment is `Deferred` with reason `patch-v2-
  unbound`, and no field-mean result is affected

#### Scenario: Hierarchical inference
- **WHEN** a farm contains multiple valid patches
- **THEN** the system produces patch-level scores and a documented aggregated
  farm-level score, and masks invalid patches from both

### Requirement: Fair model comparison

The system SHALL report, per architecture, trainable parameter count, FLOPs (or
a recorded estimation convention), training time, inference latency and peak
VRAM, independently of each other; parameter matching SHALL NOT be presented as
compute matching.

#### Scenario: Reporting completeness
- **WHEN** the architecture comparison report is generated
- **THEN** every completed architecture lists params, FLOPs, time, latency and
  VRAM in a machine-readable table with its measurement context (device, dtype,
  batch shape)
