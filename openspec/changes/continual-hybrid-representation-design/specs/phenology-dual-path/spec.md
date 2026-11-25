# phenology-dual-path Specification

## Purpose

Anomaly scoring must work without phenology and improve with it, treating the
growth stage as an uncertain estimate.

## ADDED Requirements

### Requirement: Phenology-agnostic path (r2 definition)

The system SHALL compute an anomaly score that uses no explicit stage label
and does not depend on a correctly extracted growth phase, while preserving
temporal trajectory information: residuals SHALL be computed on order-
preserving windows with Δt and preceding observations, and the normal
reference SHALL be the pixel's or field's own history at comparable temporal
context (trajectory-matched latent neighbours, same-pixel prior years matched
by DOY window), never a pool of observations from arbitrary growth stages
without temporal or contextual matching. Implicit temporal inputs (DOY sin/cos,
Δt, time since series start, days since last large NDVI drop, weather) are
allowed and ablated.

#### Scenario: Stage unavailable
- **WHEN** no stage estimate exists for a window
- **THEN** the agnostic score is still produced from trajectory and temporal context

#### Scenario: Stage estimate is wrong
- **WHEN** the extracted stage is shifted by 32 days
- **THEN** the agnostic score is identical, because it never reads the stage

#### Scenario: No cross-stage pooling
- **WHEN** the reference set for a grand-growth window is built
- **THEN** it contains only trajectory- or DOY-matched observations, not early-tillering or post-harvest observations chosen without temporal context

### Requirement: Uncertainty-aware phenology path

The system SHALL estimate a stage posterior over {tillering, grand growth,
maturity/ripening, fallow/harvest} with a confidence, score as the
posterior-weighted deviation against stage-conditioned normals, and defer to
the agnostic path below a confidence threshold, recording the deferral.

#### Scenario: Low-confidence stage
- **WHEN** stage confidence is below `c_min`
- **THEN** the aware path returns the agnostic score and flags `deferred=true`

### Requirement: Robustness to phenology error

The aware path SHALL be evaluated under stage shifts of ±16 and ±32 days and
10 % random stage swaps, and its degradation SHALL be reported against the
agnostic path with a pre-registered maximum drop.

#### Scenario: Shifted stages
- **WHEN** all stage boundaries are shifted by 32 days
- **THEN** the report shows both paths' metrics before and after the shift

### Requirement: Comparator and prerequisite

The DSH-bin stage arm SHALL be kept as a pre-registered comparator, and
stage-dependent results SHALL NOT be reported as valid until the owner's
cycle-timeline check is recorded.

#### Scenario: Timeline check missing
- **WHEN** the owner check is not recorded
- **THEN** aware-path results are labelled provisional

### Requirement: Independent and combined evaluation (r2)

The agnostic path, the aware path and a confidence-weighted combination
SHALL each be evaluated independently and in combination under identical
splits, calibration and metrics.

#### Scenario: Three-path report
- **WHEN** an E3 report is generated
- **THEN** it contains separate rows for agnostic, aware and combined scores
