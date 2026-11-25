# phenology-dual-path Specification

## Purpose

Anomaly scoring must work without phenology and improve with it, treating the
growth stage as an uncertain estimate.

## ADDED Requirements

### Requirement: Phenology-agnostic path

The system SHALL compute an anomaly score with no stage input, using a
calendar-free normal reference.

#### Scenario: Stage unavailable
- **WHEN** no stage estimate exists for a window
- **THEN** the agnostic score is still produced

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
