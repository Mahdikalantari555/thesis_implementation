# olmoearth-temporal-transfer-validation Specification

## Purpose

Resolve the four issues raised in the owner's 2026-11-27 scientific review of
`evaluate-olmoearth-nano-sugarcane` before its numbers enter the thesis
methodology chapter: quantify how much probe advantage is calendar/cycle time
rather than representation; measure (not assert) the L2→L1 domain shift;
establish unseen-year generalization on 2024–2026; and replace
seed-subsample intervals with field-level bootstrap intervals. Output: an
evidence-based answer to whether OlmoEarth Nano is (a) a temporal encoder,
(b) a transferable agricultural representation, or (c) an anomaly-detection
asset in this archive.

## ADDED Requirements

### Requirement: Probe-arm battery with calendar/cycle-time controls

The system SHALL evaluate five feature arms on identical splits, labels and
classifier: **timestamp-only** (month/day-of-year cyclical encodings; no
cycle-derived information), **cycle-progress-only** (cycle fraction and its
sine/cosine; the label-defining quantity, reported as an oracle-flavoured
upper control), **raw-spectral-only** (8 bin-mean channels), **raw+temporal**
(8 channels + month/day-of-year encodings; cycle fraction excluded as
label-defining, recorded in the report), and **OlmoEarth frozen embeddings**.
Every arm SHALL report macro-F1 and accuracy with **field-level bootstrap
95 % CIs (1,000 resamples over evaluation fields)**, and every OE-vs-baseline
comparison SHALL report a **paired field-level bootstrap delta distribution**.

#### Scenario: Temporal confound quantified

- **WHEN** the battery completes
- **THEN** the report states the OE-minus-timestamp-only and
  OE-minus-raw+temporal paired deltas with their intervals, and the claim
  "embeddings carry phase information beyond raw channels" is retained only
  if OE also exceeds the timestamp-only arm (otherwise the advantage is
  attributed to calendar/cycle time and the wording corrected)

#### Scenario: Seed-subsample intervals retired

- **WHEN** any interval is quoted in this change
- **THEN** it is a field-level bootstrap interval over evaluation fields, and
  the parent change's seed-subsample std is no longer presented as a
  population CI

### Requirement: Unseen-year generalization on 2024–2026

The system SHALL train the stage probe on pre-2024 bins of train-split fields
only, and evaluate per calendar year (2024, 2025, 2026 and pooled ≥2024) on bins of
the **spatially held-out k-means dev/test block fields** (which have zero
overlap with `train_fields`; the v2 protocol's flat `test_fields` overlap
train by design at cycle level, so they are NOT a spatial holdout), with no
adaptation, threshold selection or model choice made on those years. The
report SHALL include an **attrition table** tracing the ≥2024 bins/fields of
the whole patch store (S0) through the flat-split filter (S1) and the
spatial-block filter (S2) to the final evaluation pool after the raw-feature
join (S3); the S2→S3 join SHALL be lossless or the run fails. The raw-spectral arm SHALL be evaluated
under the identical restriction as the control, and a phase-consistency
analysis restricted to ≥2024 cycles SHALL be reported as secondary evidence.

#### Scenario: No holdout leakage

- **WHEN** the holdout-year evaluation runs
- **THEN** the training set contains zero bins with date ≥ 2024-01-01, the
  evaluation sets contain zero bins from train-split fields, and both
  properties are asserted at runtime (hard failure otherwise)

#### Scenario: Sparse years are not over-read

- **WHEN** a holdout year has fewer than 100 evaluation bins OR fewer than
  30 independent evaluation fields (the low-n rule; 2025 qualifies on the
  field axis, 2026 on both)
- **THEN** its per-year metrics carry their n (bins AND fields) and interval,
  are flagged low-n, and are not merged into a headline claim

#### Scenario: Phase margin on the evaluation fields only

- **WHEN** the ≥2024-restricted phase-consistency margin is computed
- **THEN** it yields ONE value per unique evaluation field (the mean over
  that field's cycle pairs; `n_fields` reports unique fields, never cycle
  pairs), carries a field-level bootstrap interval (1,000 resamples), and is
  reported **not estimable** when fewer than 2 evaluable cycle pairs exist
  — the pre-2024-dominated margin of the parent change is not substituted

### Requirement: L2→L1 domain-shift measurement (data-gated)

The system SHALL specify and, once L1 imagery is available, execute a
same-date comparison of Landsat C2 L1 vs L2 products over a designed subset
(≥ 3 scene pairs spanning vegetated, bare/late-season and winter conditions
from path/rows 165-039 and 166-038): per-band distributions of TOA reflectance
vs surface reflectance, brightness temperature vs LST, the resulting
normalized-space distances under the adapter conversion, and the embedding
drift when the same fields are fed through the true-L1 input path vs the
L2-converted path.

#### Scenario: Missing L1 data fails loudly, never silently

- **WHEN** T3 is attempted without same-date L1 scenes on disk
- **THEN** the script exits with an explicit data-gated message and the
  change records the exact owner command to fetch the public C2 L1 scenes
  (Google Cloud Storage public bucket; no credentials) — the gap is never
  bridged by simulation or substituted data

### Requirement: Anomaly-detection re-run with real spatial context (Deferred)

Repeating the injected-deviation battery with true spatial context SHALL be
recorded as **Deferred**, blocked by the HELD openspec change
`date-centric-raw-store`; it SHALL NOT be executed on the field-only patch
store again as if it resolved the limitation.

#### Scenario: Blocked dependency is visible

- **WHEN** the report lists T4
- **THEN** it names `date-centric-raw-store` (0/28, HELD, owner decision) as
  the blocker and contains no T4 numbers

### Requirement: Interpretation verdict

The final output SHALL state which of the three hypotheses the evidence
supports — (a) temporal encoder only, (b) transferable agricultural
representation, or (c) anomaly-detection added value — with the T1/T2/T3
results each mapped to the claim they support, every claim carrying its
validation-ladder rung (L1–L2), and the parent change's ADAPT verdict
confirmed or amended explicitly. Every transferability statement SHALL be
scoped as **within-region, within-crop** (Khuzestan sugarcane; one sensor
family) spatial and temporal transferability; no cross-region, cross-crop or
cross-sensor generalization SHALL be implied by any table, sentence or
summary in this change.

### Requirement: Isolation and artifact discipline

All code SHALL live under `experiments/olmoearth_eval/`, import no
`ssl_anomaly` modules, write only inside that directory, and persist all
tables as JSON with n, years, field counts, split definitions, bootstrap
settings (resamples, seed) and the parent artifact names.
