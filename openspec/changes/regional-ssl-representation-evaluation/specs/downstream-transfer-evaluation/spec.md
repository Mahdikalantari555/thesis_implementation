# downstream-transfer-evaluation Specification

## Purpose

Define the Level-2 evidence battery: the B1–B5 incremental input ladder, the
label-fraction transferability curves (5 / 10 / 25 / 50 / 100 %), the six-arm
comparator protocol, and the few-shot similar-date retrieval benchmark — all
under one identical, FAR-matched evaluation protocol so that learning-based
claims are earned against the simple NDVI+LST control, not against strawmen.

## ADDED Requirements

### Requirement: B1–B5 incremental input ladder

The system SHALL implement the five-rung ladder — B1 (FAR-matched NDVI+LST
rule), B2 (SSL on NDVI+LST only), B3 (SSL on all spectral+thermal channels),
B4 (B3 + daily weather), B5 (B4 + phenological/decomposition context) — under
identical splits, identical calibration subsets, identical anomaly definitions,
identical FAR budgets (5 % cycle-level FAR) and identical aggregation, with
per-family and per-severity cells reported individually.

#### Scenario: FAR-matched control
- **WHEN** the B1 rule is calibrated
- **THEN** its thresholds are chosen on the clean calibration subset to hit the
  same 5 % cycle-FAR budget as the SSL arms, and no evaluation-set information
  is used for threshold selection

#### Scenario: Decomposition features absent
- **WHEN** B5 is configured but the decomposition study has not produced
  admissible trend/seasonal/residual features
- **THEN** the affected B5 inputs are recorded `absent`, B5 runs without them
  and the omission is visible in the report — never silently replaced

### Requirement: Label-fraction transfer curves

The system SHALL evaluate every claim-bearing downstream task at label
fractions 5 / 10 / 25 / 50 / 100 % of the labelled pool, using stratified
subsets hashed and shared across arms, repeated under the three recorded seeds.

#### Scenario: Shared label subsets
- **WHEN** an arm is evaluated at a given label fraction
- **THEN** it uses the identical subset (same hash) as every other arm at that
  fraction, and the subset hashes are recorded in the report

### Requirement: Six-arm comparator protocol

Each downstream cell SHALL compare: A-rule (B1 rule), A-B1 (phase-aware
statistical baseline), A-rand (random-init encoder of the same trunk and
capacity), A-scratch (task-label-only from-scratch model), A-ssl-frozen
(pretrained encoder, frozen + linear head) and A-ssl-ft (pretrained encoder,
fine-tuned, equal budget to A-scratch).

#### Scenario: Capacity control
- **WHEN** A-ssl-ft is trained
- **THEN** its optimizer budget and parameter band equal those of A-scratch,
  and both are reported alongside A-rand so capacity, not initialization
  provenance alone, is controlled

### Requirement: Claim-bearing and exploratory task separation

The system SHALL tag every downstream task as claim-bearing or exploratory.
Claim-bearing: T1 phenology-stage estimation, T2 limited-label injection
detection, T3 few-shot retrieval Track 1, T4 self-supervised trajectory
quality. Exploratory: T3 Track 2 (weak-proxy CWSI_ET/TVDI excursions, labelled
weak), T3 near-match listing, T5 crop-condition / irrigation-pattern analysis.
Exploratory outputs SHALL NOT be used as inputs to any decision gate or thesis
claim.

#### Scenario: Gate input audit
- **WHEN** the decision-gate aggregator runs
- **THEN** it rejects any exploratory artifact as gate input and the rejection
  is recorded

### Requirement: Few-shot similar-date retrieval benchmark

The system SHALL implement the T3 benchmark: query k ∈ {1, 3, 5, 10} labelled
(field, date) points; rank the archive by embedding similarity; report
precision@{1, 5, 10}, mean average precision and, on Track 1, anomaly-family
hit rate; compare against B1-score ranking, A-rule-score ranking, raw
band-vector Euclidean distance and A-rand embeddings.

#### Scenario: Two tracks reported separately
- **WHEN** Track 1 (injection-labelled) and Track 2 (weak-proxy) results are
  written
- **THEN** they appear in separate, clearly labelled sections with separate
  prevalence, and Track 2 carries the explicit note that proxy excursions are
  not ground-truth stress events

#### Scenario: Near-match listing for manual inspection
- **WHEN** T3 runs under the owner decision of 2026-11-23
- **THEN** the system also emits a descriptive natural-data near-match listing
  (top-10 unlabelled archive neighbours per query, with similarity scores,
  field/date and the query answered) for manual owner inspection, labelled
  `exploratory`, containing no ground-truth assertion, and rejected as an
  input to any decision gate, threshold, selection or claim

### Requirement: Metric reporting discipline

Every cell SHALL report AUPRC with prevalence (primary on imbalanced cells) and
AUROC (secondary); F1 SHALL use validation-selected thresholds only; Wilson or
bootstrap intervals SHALL accompany rates; macro-averaging, when shown, SHALL be
supplemented by the full per-cell table.

#### Scenario: Threshold selection
- **WHEN** F1 thresholds are computed
- **THEN** they derive from the validation partition only and the procedure is
  recorded in the manifest

### Requirement: Machine-readable outputs

The system SHALL export `transfer_curves.json`, `retrieval_benchmark.json` and
`b1_to_b5_ladder.json` with full provenance, and a Markdown summary
(`capstone_report.md`) rendering every table including failing cells.

#### Scenario: Incomplete coverage
- **WHEN** any cell cannot be computed (missing data, budget exhaustion,
  disqualified candidate)
- **THEN** the cell is recorded with an explicit status and reason, and the
  report does not present the matrix as complete
