# Tasks — gradual-severity-shape

## 1. Injection machinery (TDD-first)

- [ ] 1.1 Unit tests: ramp profile δ·min(1, t/D) per bin for D∈{3,4,6};
      onset sampled so the full ramp fits inside the cycle; partial-δ SAVI
      solver correctness; field-level AND patch-tensor appliers produce
      identical per-bin δ profiles.
- [ ] 1.2 `monitor/perturbations.py` + `monitor/severity_table.py`: gradual
      family registration, new injection-set hashes in the sidecar;
      acute/persistent hashes asserted unchanged (regression test).
- [ ] 1.2b Secondary shapes: peak-magnitude + plateau-move appliers and
      failure-to-recover / slow-recovery appliers (parameters per-injection
      recorded; clipped to admitted bins; covariate consistency tests);
      secondary diagnostic columns in the report writers; regression test
      that primary-gate tables and hashes are untouched by these shapes.
- [ ] 1.3 eval_dev.py / trial_common.py: MDM table gains the gradual
      column(s) + b0_calendar gradual column; censoring semantics
      unchanged; Wilson CIs + prevalence.

## 2. Run + reports

- [ ] 2.1 Agent smoke: gradual sweep on a tiny injection set (shapes,
      hashes, censoring) — no retraining of any frozen checkpoint.
- [ ] 2.2 User-run (exact commands handed over): gradual sweep for the
      available arms (b1, gru_ae_main, b0_calendar, + trained arms that
      exist at apply time).
- [ ] 2.3 Persian finding (persian-writing skill): three-shape MDM curves,
      the B0-gradual result reported as a first-class outcome (expected
      honest negative for local-state models is a finding, not a failure),
      full provenance fields, L2 rung labelling.
