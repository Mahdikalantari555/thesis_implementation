# Tasks — patch-generalization-suite

## 1. Fold + matrix machinery (TDD-first)

- [ ] 1.1 Unit tests: fold-pair construction (contiguity, seed-fixed,
      hash-recorded; every block in exactly one fold); per-fold
      normalization re-fit from in-fold train data only (leakage test);
      quadrant-cell cycle selection respects dev_temporal_cutoff and
      dev-eligibility.
- [ ] 1.2 `scripts/run_generalization_matrix.py`: per arm × cell × fold
      training/eval driver with --estimate-runtime, --resume, progress
      output; reuses train_main/train_patch checkpoints where folds permit
      (in-block cell) and retrains for held-out cells.
- [ ] 1.3 Sensor stratification: date→sensor-era mapping (from scene
      metadata); stratified report columns; year-drift quantile-shift
      metric; threshold-stability check (frozen τ/k across years, no refit).

## 2. Runs (user-executed)

- [ ] 2.1 Screening pass: single seed × 5 folds for gated arms; review
      produced reports; 3-seed runs only for arms reaching final claims.
- [ ] 2.2 Matrix report: cells × folds × sensor eras with Wilson CIs,
      prevalence, MDM censoring; committed under reports/.

## 3. Verdict

- [ ] 3.1 Pre-registered L3 acceptance applied verbatim per arm; failing
      cells named; findings updated.
- [ ] 3.2 Persian finding (persian-writing skill): the matrix, the confound
      analysis, and which arms earn L3 — honest negatives explicit.
