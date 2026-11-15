## ADDED Requirements

### Requirement: Generalization matrix (E9)

The system SHALL evaluate gated arms on a generalization matrix with four
cells — in-block, held-out fields (5 contiguous fold-pairs built from the
10 locked spatial blocks), held-out periods (via dev_temporal_cutoff), and
both — with per-fold MDM80/FAR and fold spread, using hash-identical
injection sets across folds and per-fold re-fitted normalization from
in-fold train data only. Every result SHALL be reported raw AND
sensor-stratified (L8-era vs L9-era dates) with the period/sensor confound
stated. An arm's claims SHALL be labelled L3 only when the pre-registered
acceptance holds (fold spread within two standard errors of the dev
estimate, no censoring-status loss in any fold, temporal degradation beyond
one severity level excluded by bootstrap CI); otherwise the failing cells
SHALL be named in an honest negative. Test-block cycles SHALL be read only
by the locked final-test path.

#### Scenario: Folds are spatial and leak-free

- **WHEN** a fold trains and evaluates
- **THEN** the held-out block-pair contributes no training cycles and no
  normalization statistics, and the fold's injection-set hash equals every
  other fold's

#### Scenario: Confound is surfaced, not hidden

- **WHEN** a temporal-axis drop is observed
- **THEN** the sensor-stratified columns show whether the drop tracks the
  L8→L9 era change, and the report states the confound explicitly

#### Scenario: L3 is earned

- **WHEN** an arm's matrix meets the pre-registered acceptance
- **THEN** its claims are upgraded to L3 with the matrix committed under
  reports/; otherwise the claim stays at L2 and the failing cells are
  recorded
