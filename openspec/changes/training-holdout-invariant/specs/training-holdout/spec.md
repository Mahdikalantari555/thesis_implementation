## ADDED Requirements

### Requirement: Frozen training-holdout manifest

The system SHALL maintain a frozen holdout manifest derived from the locked
splits (`9aa5eb11…`) — holdout farms = the test-block fields (whole-field
holdout, all periods), holdout dates = every date from `test_period_start`
(2024-01-01) onward plus all future acquisitions, holdout fields = also any
field registered after the archive freeze — frozen with SHA-256 and
rebuildable only with `--force <reason>`. No new split semantics SHALL be
introduced: the manifest is a derived, frozen statement of the existing
locked assignment.

#### Scenario: Manifest is a pure derivation

- **WHEN** the manifest is built
- **THEN** its farm and date sets are exactly derivable from splits_locked,
  its sha and provenance are recorded, and a rebuild without `--force`
  fails

### Requirement: Fail-closed holdout enforcement in every training path

Every training consumer — `train_main.py`, the patch trainer and its
representation variants, fold retrains and contamination runs — SHALL load
the frozen manifest (hash-verified) and SHALL assert, before training,
that no training input touches a holdout farm or a holdout-period cycle;
violation SHALL raise a named error listing the offending pairs and abort
(fail-closed, never a warning). Per-pixel patch stores MAY contain holdout
fields for evaluation purposes; the enforcement applies at the training
consumer. Subset manifests (PoC subsets, folds, contamination fractions)
SHALL be validated against the holdout manifest before use.

#### Scenario: Holdout leak aborts training

- **WHEN** a training Dataset would include a test-block field's cycle or a
  post-2024 cycle
- **THEN** construction fails with `HoldoutViolationError` naming the
  (field, cycle) pairs, and no training starts

#### Scenario: Clean training passes

- **WHEN** a training Dataset contains only non-holdout farms and
  pre-2024 cycles
- **THEN** the assertion passes and the run records the manifest sha in
  its provenance
