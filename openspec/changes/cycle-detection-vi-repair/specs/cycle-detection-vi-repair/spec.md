# cycle-detection-vi-repair Specification

## Purpose

Detect harvest boundaries and assign growth states from NDVI and SAVI alone,
preserve unresolved intervals as evidence, and never synthesize a plant/ratoon
label or force a harvest that the VI trajectory does not support.

## ADDED Requirements

### Requirement: NDVI+SAVI-only harvest score

The harvest score SHALL be the mean of the oriented robust z-scores of NDVI and
SAVI (−z(NDVI), −z(SAVI)), with no LST, GDD, weather or crop-coefficient term.
A date with a missing or non-finite value in either index SHALL have a NaN score
and SHALL NOT become a candidate; surviving channels SHALL NOT be renormalized.

#### Scenario: Weather-only signal
- **WHEN** GDD drops sharply on a winter date while NDVI and SAVI are unchanged
- **THEN** that date is not a harvest candidate

#### Scenario: Missing index
- **WHEN** SAVI is NaN at a date
- **THEN** the score at that date is NaN and no candidate is emitted there

### Requirement: VI-progress growth states

Each cycle row SHALL carry one of four growth states derived from normalized
within-cycle progress: S1 establishment/growth (0–0.25), S2 peak growth
(0.25–0.60), S3 maturity/senescence (0.60–0.85), S4 harvest/dormancy
(0.85–1.00). No calendar stage label, DSH-bin label or plant/ratoon label SHALL
be written to a cycle row.

#### Scenario: Plant/ratoon field absent
- **WHEN** a cycle row is emitted
- **THEN** it has no `kind` field with values `plant` or `ratoon`

#### Scenario: Progress boundary
- **WHEN** normalized progress is 0.62
- **THEN** the growth state is S3 maturity/senescence

### Requirement: Provenance-preserving over-long intervals

A non-censored interval longer than `max_cycle_days` SHALL NOT be deleted. It
SHALL be emitted with `status = "overlong_unresolved"`, its dates, length and
candidate evidence, and SHALL be excluded from splits and stage assignment by
an explicit filter. A confidence threshold SHALL NOT be lowered silently to
split such an interval.

#### Scenario: Long gap remains visible
- **WHEN** an interval of 567 d is produced by two consecutive harvest candidates
- **THEN** a row with `status = "overlong_unresolved"` and length 567 persists in the output

#### Scenario: No silent promotion
- **WHEN** an unresolved interval is reviewed
- **THEN** no code path converts it into a harvest boundary without a recorded owner decision

### Requirement: Greenup cycle start with explicit rest

A cycle SHALL NOT be assumed to have a rest period after every harvest. Rest
SHALL be claimed only when both smoothed NDVI and SAVI stay within `bare_tol`
(default 0.02) of their post-harvest floor for at least `min_rest_obs` (default
3) consecutive acquisitions. Only then, greenup SHALL be the first sustained
rise of at least `rise_tol` (default 0.02) above that floor for
`greenup_min_obs` (default 2) acquisitions; that date is the cycle start and
the preceding days are `rest_days`. If there is no bare-soil plateau, or no
rise before the next boundary, the start SHALL fall back to the harvest with
`cycle_start_source = "harvest"` and `rest_days = 0`. A fraction of full canopy
amplitude SHALL NOT be used as the rise criterion.

#### Scenario: Fallow is not a cycle
- **WHEN** NDVI/SAVI stay near the post-harvest floor for 300 days and then rise
- **THEN** the 300 days are recorded as rest and the cycle length starts at the rise

#### Scenario: No greenup detected
- **WHEN** no sustained rise occurs before the next boundary
- **THEN** the cycle starts at the harvest and `rest_days` is zero

### Requirement: Short bounded interval
- **WHEN** two strong boundaries delimit a bounded interval shorter than `min_cycle_days`
- **THEN** the interval is retained with `status = "short_unresolved"` and excluded from splits instead of being deleted

### Requirement: Reviewable candidate audit

Every candidate SHALL record its score, prominence, channel z-scores and support.
Candidates below the prominence threshold SHALL be retained in a separate
`weak_candidates` table with their evidence, never used as cycle boundaries
without review.

#### Scenario: Weak candidate
- **WHEN** a candidate has score 0.5 and prominence 0.13 at the default threshold 0.3
- **THEN** it appears in `weak_candidates` and does not create a cycle boundary

### Requirement: No claim beyond the review

No rebuilt cycle output SHALL be used for stage-conditioned anomaly claims
until the owner has reviewed the regenerated timeline for the three mini farms
and the six-field stratified sample.

#### Scenario: Stage claim before review
- **WHEN** a report requests a stage-conditioned result before review is recorded
- **THEN** the result is labelled provisional and no L3 claim is made
