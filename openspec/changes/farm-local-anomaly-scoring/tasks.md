# Tasks — farm-local-anomaly-scoring

## 1. Basis classification + independence tests (TDD-first)

- [ ] 1.1 Arm/component basis registry: a single table declaring the basis
      tier (strictly local / locally anchored / population-ranked) for
      `b0_*`, `b1`, `b3`, `gru_ae_main`, `gru_pred`, `patch_cnn_gru`,
      `patch_pixelwise`, `R0.5` arms, and the components `s_pred`, `s_rec`,
      `s_lat`, `s_hyb`, `z_f(t)`, common-mode subtraction, indicator z;
      unit test that every arm name used in `eval_dev.py` has a declaration
      and that a missing declaration fails the report build.
- [ ] 1.2 Independence tests: (a) other-farm perturbation invariance
      (bitwise), (b) field-id permutation invariance, (c) no cross-field
      read (instrumented spy with a whitelist of frozen pooled constants);
      a synthetic population-ranked component is planted to prove the tests
      actually FAIL for it (negative controls).
- [ ] 1.3 Report writers: `basis` column in every arm table
      (`eval_dev.py` reports, `reports/dev/<run>/`); a population-ranked
      component without a stated cross-farm read fails the build.

## 2. Per-field scale tables + arm

- [ ] 2.1 Frozen artifact builder: per-field per-pixel and per-field
      `1.4826 × MAD` of clean dev residuals, shrink weights
      `w_f = n_f/(n_f + κ_σ)`, floor percentile, history tier per farm;
      provenance (residual source run ids, git sha, splits sha, own sha256);
      rebuild requires `--force <reason>`. Unit tests: shrinkage identity
      (`n_f = 0` ⇒ pooled σ), floor behaviour, no injected/test data
      touched, deterministic hashes.
- [ ] 2.2 Scoring integration: `σ` basis switch
      (`--sigma-basis {pooled,farmlocal,shrunk}`) in the patch scorer and
      the `b1`/`b0` paths; τ re-calibrated inside the same 5% cycle-FAR
      budget on clean dev; unit test that `--sigma-basis pooled` reproduces
      the current numbers byte-identically (regression guard).
- [ ] 2.3 Pre-register the comparison rule in a gates module BEFORE any
      run: adopt farm-local only on MDM80 improvement with paired-bootstrap
      CI over fields excluding zero, FAR budget met, and no increase in
      alarm concentration; no goalpost movement.
- [ ] 2.4 Agent smoke on a tiny set of farms (both bases, fallback tiers,
      FAR accounting) — no retraining, residuals frozen; hand the user the
      exact evaluation commands.

## 3. Strata (ratoon / variety)

- [ ] 3.1 `stratum_level` flag on the σ tables
      (`phenophase` / `×ratoon` / `×ratoon×variety`) reusing
      `build_stratum_table` columns and the existing min-n fallback; unit
      tests on fallback coverage and on the untouched default (current
      numbers reproduced).
- [ ] 3.2 Report MDM80 + FAR-achieved per stratum level with coverage per
      stratum; record the decision for the frozen configuration.

## 4. Farm-level anomaly summary

- [ ] 4.1 Summary-record writer: per alarm
      `{field_id, cycle, first_alarm_bin, severity_level, persistence_len,
      affected_pixels, affected_area_frac, n_components, max_component_px,
      max_component_ha, centroid_xy, score, basis_tier, phi_bin}` written as
      per-run JSONL; connected components computed on the in-field mask only
      (unit test: an alarm touching two fields counts components within one
      field only); platform payload asserted unchanged.
- [ ] 4.2 Summary table in the run report: distribution of affected area
      fraction, component counts and extents by severity and phenophase.

## 5. Verdict

- [ ] 5.1 Apply the pre-registered rule verbatim; both bases reported at the
      same FAR with FAR-achieved, MDM80, crossing counts and alarm
      concentration; tier counts stated.
- [ ] 5.2 Hand the decision to `patch-full-run-and-ablations` (the frozen
      final configuration's basis) and label the rows consumed by
      `compare-spatial-representation-strategies`.
- [ ] 5.3 Persian finding (persian-writing skill): is the farm's own history
      a better comparison basis than the population at equal FAR — with the
      honest negative fully written if it is not; the 223/370 history
      limitation and the tier counts stated plainly; L2 rung labelling;
      `00_index.md` row; `todo.md` §D updated.
