# experimental-validation Specification

## Purpose

Define how benchmark evidence is produced and reported: the validation
protocols, the primary and secondary metrics, the AUPRC macro-averaging
contract, environmental-proxy disclosure, the statistical suite and the
three-winner reporting rule.

## ADDED Requirements

### Requirement: Three validation protocols

The system SHALL implement V1 (temporal holdout within the locked v2 split
discipline), V2 (farm holdout, block-consistent) and V3 (spatiotemporal
holdout within train+dev with holdout exclusions asserted). The locked test
split SHALL NOT be read by any protocol.

#### Scenario: V3 exclusion assertion
- **WHEN** a V3 evaluation set is constructed
- **THEN** an automated check confirms no holdout-manifest field and no
  post-cutoff unseen-period leakage is present, and the check hash is recorded

#### Scenario: Test isolation
- **WHEN** any protocol code path enumerates cycles
- **THEN** split `test` yields an empty admissible set and an explicit notice

### Requirement: Data leakage prevention

Normalization statistics, feature/threshold selection and hyperparameter
selection SHALL use only train/calibration partitions; validation partitions
SHALL be used only for selection/stopping; test partitions SHALL be used for
nothing.

#### Scenario: Selection audit
- **WHEN** an HPO or threshold-selection routine runs
- **THEN** it receives only train/validation data and the audit log records the
  granted splits

### Requirement: AUPRC as primary metric

AUPRC SHALL be computed at event, point, patch and farm levels. The headline
result SHALL be the macro-AUPRC across the five anomaly families × three
severities at a predefined identical test prevalence with equal cell weighting;
per-cell results SHALL always accompany it.

#### Scenario: Headline computation
- **WHEN** the headline is computed
- **THEN** prevalence per cell is recorded, no cell is dropped, and the report
  shows the full 5×3 grid next to the macro value

### Requirement: Secondary metrics

The system SHALL report AUROC, F1 (validation-selected threshold; rule
recorded), detection delay (median days, localized and persistent events),
Spearman and Kendall correlations, Brier score/ECE where applicable, inference
latency, peak VRAM, parameter count and FLOPs estimate.

#### Scenario: Threshold governance
- **WHEN** F1 is reported
- **THEN** the threshold was selected on validation data only and the selection
  procedure is embedded in the report

### Requirement: Independent environmental validation

TVDI, CWSI_ET, CWSI_LST, ETa and ETrF SHALL be evaluated separately with
Spearman and Kendall rank correlation and dependence-aware uncertainty; every
indicator SHALL be labelled with its input overlap relative to the model;
shared-input comparisons SHALL be presented as consistency checks only.

#### Scenario: Proxy dependence label
- **WHEN** CWSI_LST is compared against model scores
- **THEN** the report states that CWSI_LST derives from LST/meteorology and
  shares inputs with thermal model channels, and the comparison is not used for
  calibration, selection or rung promotion

### Requirement: Statistical testing

The system SHALL implement paired bootstrap (farm/event resampling, B ≥ 1,000),
Wilcoxon signed-rank, Friedman with post-hoc comparison and effect-size
estimation, with multiple-comparison correction across cells; confidence
intervals SHALL accompany all reported rates and deltas.

#### Scenario: Dependence-aware resampling
- **WHEN** bootstrap intervals are computed
- **THEN** resampling is over farms and events (not windows), and the method
  is stated in the report

### Requirement: Three random seeds

Every applicable experiment SHALL run seeds {13, 42, 2026}; results SHALL
report mean ± standard deviation; seeds SHALL NOT be treated as independent
farms or populations.

#### Scenario: Seed spread reporting
- **WHEN** results are aggregated across seeds
- **THEN** the report shows per-seed values, the mean ± standard deviation and
  the count of seeds run, and any single-seed result is labelled `n=1-seed`
  rather than presented as replicated

### Requirement: Multiple-comparison correction

Statistical comparisons across cells/architectures SHALL apply Holm–Bonferroni
(default) or Benjamini–Hochberg (screening tables), and the correction used
SHALL be named per table.

#### Scenario: Correction disclosure
- **WHEN** any table of pairwise comparisons is produced
- **THEN** it states the number of comparisons and the correction applied, and
  uncorrected p-values are labelled `unadjusted`

### Requirement: Separate winner reports

The system SHALL emit three independent winner reports — Detection (highest
supported macro-AUPRC), Generalization (stability across V1/V2/V3) and
Multi-objective (Pareto across detection, generalization, compute) — and SHALL
NOT scalarize them into one arbitrary score.

#### Scenario: Winner declaration
- **WHEN** the final report declares a winner in any category
- **THEN** it includes the supporting intervals, effect sizes and the category's
  scope, and notes that the other two categories may legitimately disagree
