## Design — training-holdout-invariant

### Manifest derivation (pure function of locked splits)

`scripts/build_holdout_manifest.py` reads splits_locked `9aa5eb11…` and
emits `data/processed/holdout_manifest.json`:
- `holdout_farms` = splits_locked `test_fields` (51) — whole-field spatial
  holdout (their pre-2024 cycles are ALSO holdout for training, which the
  split assignment already guarantees by sending every uncensored cycle of
  a test-block field to test).
- `holdout_date_start` = `test_period_start` (2024-01-01) — temporal
  holdout for non-block fields.
- `future_fields_rule` / `future_dates_rule` = "holdout by default" (the
  2026-11-14 product vision: new acquisitions and new fields are reserved
  for validation).
- provenance: splits manifest sha, git sha, created date, own sha256.
Rebuild requires `--force <reason>` (locked-splits discipline). The manifest
is data, derived once — the trainer NEVER re-derives it.

### Enforcement (fail-closed, at the consumer)

`monitor/holdout.py` provides `load_holdout_manifest()` (hash-verified) and
`assert_train_only(field_ids, cycle_start_ends)` raising
`HoldoutViolationError` listing the offending (field, cycle) pairs. Wiring:
- `train_main.py` Dataset construction asserts before the first batch.
- The patch trainer spec (`patch-level-controlled-trial` tasks 1.3/1.4)
  requires the same assertion; `patches.zarr` legitimately contains holdout
  fields — the Dataset filters them out and the assertion proves it.
- Fold retrains, contamination runs, and R2/R3 representation variants
  inherit the assertion through the shared Dataset base.
- A unit test plants a holdout cycle into a Dataset and asserts the error
  (fail-closed proof), plus a test that a clean dataset passes.

### What this does NOT touch

Evaluation/locked-final-test reading rules, dev calibration, the severity
machinery, and the existing split assignment are unchanged — the invariant
only makes an existing structural guarantee explicit, frozen, and enforced.
