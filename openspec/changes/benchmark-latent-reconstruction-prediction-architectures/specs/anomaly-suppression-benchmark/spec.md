# anomaly-suppression-benchmark Specification

## Purpose

Define the suppression axis (S0–S3), the synthetic anomaly generator (five
families × three severities) and the rules that keep synthetic contamination
from contaminating training or being mistaken for real stress.

## ADDED Requirements

### Requirement: Four suppression configurations

The system SHALL implement S0 (baseline), S1 (bounded soft suppression),
S2 (masked learning) and S3 (synthetic-anomaly suppression), each with
documented, recorded weights/parameters and non-reinforcing weighting designs.

#### Scenario: Soft-suppression weight bound
- **WHEN** S1 weights are computed
- **THEN** they are bounded within the pre-registered interval and cannot be
  derived solely from the incumbent model's own current residuals without an
  external or detached signal

### Requirement: Explicit architecture applicability

Every (architecture, suppression) pair SHALL carry an explicit status in the
committed applicability matrix: applicable or not-applicable with a recorded,
machine-checkable reason. Invalid combinations SHALL be marked `Not Applicable`
and SHALL NOT be replaced by a substitute loss.

#### Scenario: Invalid combination encountered
- **WHEN** the registry schedules a pair marked not-applicable
- **THEN** the experiment is recorded as `Not Applicable` with the matrix
  reason, no training is launched, and the report lists it as such

### Requirement: Independent synthetic generators

Training contamination generation (S3) and evaluation anomaly generation SHALL
use independent random states and disjoint sample pools; overlap SHALL be
verified by an automated isolation test before any S3 run.

#### Scenario: Contamination isolation
- **WHEN** S3 contamination is drawn
- **THEN** no generated sample appears in the evaluation pools (verified by
  hash comparison) and the isolation check result is recorded in the run manifest

### Requirement: Five anomaly families

The generator SHALL support spectral, temporal, phenological, multivariate and
persistent anomaly families, plus controlled combinations, parameterized by
pre-registered distributions.

#### Scenario: Family combination
- **WHEN** a combined scenario is requested (e.g. persistent spectral +
  multivariate)
- **THEN** the injected mask records all participating families and their
  individual magnitudes

### Requirement: Severity control

Each family SHALL support Weak, Moderate and Strong severities, scaled relative
to field-historical variability and feature-specific distributions; severity
maps SHALL be recorded in the manifest with the scaling source.

#### Scenario: Trivial separability audit
- **WHEN** a generated Weak-severity family is audited
- **THEN** its magnitude distribution is reported against the clean data
  distribution (per MAD) in the manifest realism section, and trivially
  separable configurations are flagged for exclusion, not silently kept

### Requirement: Synthetic ground truth

Every generated event SHALL record farm/patch identifiers, temporal location,
duration, family, severity, injected dimensions, parameters, seed, ground-truth
mask and boundaries; these labels SHALL be used only for controlled evaluation
and designated contamination experiments.

#### Scenario: Evaluation use
- **WHEN** synthetic labels are requested for any purpose other than controlled
  evaluation or S3 contamination
- **THEN** the request is refused by the API and logged

### Requirement: Contamination audit

The system SHALL record, per S3 run, the percentage, type and severity of
synthetic contamination introduced into training, and the report SHALL include
a contamination summary table.

#### Scenario: Contamination summary present
- **WHEN** an S3 run completes
- **THEN** its manifest and the final report list the contamination share by
  family and severity, and the report shows zero overlap with the evaluation
  pools (hash-verified)
