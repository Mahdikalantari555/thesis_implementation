# Tasks: oracle-floor-audit

## 1. Gap realignment (E2 enabler)

- [x] 1.1 `patch_scores._forward`/`predict_pixels` accept `gap: int = 1`;
      gap-h realignment `pred[:, t] = anchor[:, t−h]` for t ≥ h, bins < h NaN;
      h=1 byte-identical to `one_step_ahead_v3` (parity asserted in tests);
      docstring records the causality invariant.
- [x] 1.2 Unit tests: realignment math on a stub forward (all h values),
      h=1 parity, gap out-of-range raises, unscored head bins NaN.

## 2. E2 oracle-floor audit

- [x] 2.1 `scripts/audit_oracle_floor.py`: V-pool builder (monitoring_only ∩
      clean, parquet hash check, no train/dev/test overlap), field-blocked
      V_cal/V_test (rng 13), chunked scoring (clean + one forward per injected
      family), σ_h/α_h/Q_h/δ_z tables per phase group per family, Gaussian
      trailing-window oracle (w=3, FAR 5%, T=36) with numerical threshold
      root-find, gap vs the committed finding-25 cells, (A)/(C)/(B) decision,
      JSON + console report, `--limit` smoke mode.
- [x] 2.2 Pure-function unit tests: oracle power (δ=0 → FAR; δ→∞ → 1;
      monotone in δ), MAD/σ estimator, α/Q arithmetic, V-pool field blocking.
- [x] 2.3 Smoke run green locally (`--limit 8`), then owner handover command
      for the full V=260 run (both arms).

## 3. Q8 matched-NDVI decomposition

- [x] 3.1 `scripts/audit_matched_ndvi.py`: realized per-channel band deltas
      of the SAVI and NDVI solvers at equal nominal Δ on real store cycles,
      band-magnitude ratio ρ per level, NDVI-equivalent Δ* = Δ·ρ, JSON curve.
- [ ] 3.2 DEFERRED 2026-11-22 (design open question): the confirmatory
      matched-Δ one-look cell runs AFTER the E2 decision — the equivalence
      curve (ρ measured, deterministic) already separates the hypotheses;
      implement only if severity re-parameterization stays on the table.
- [x] 3.3 Unit tests for the equivalence math (linear-solver stub).

## 4. Per-injection logging (eval_dev)

- [x] 4.1 Severity cells record `det_flags` (aligned with `injections`) and
      `detection_by_onset` (per onset phase n/rate); additive keys only.
- [x] 4.2 E2E test extended: keys present, phases = {T//6, T//2, 5T//6},
      onset counts sum to the cell n.

## 5. MDE curve

- [x] 5.1 `scripts/build_mde_curve.py`: floor-anchored probit fit per
      arm × family from the six committed metrics JSONs, Δ*(0.8) with
      parametric-bootstrap CI, Δ_req = 0.165 verdict, censored outputs marked
      ">S6" with extrapolation-labelled-as-model; JSON + console table.
- [x] 5.2 Unit tests: fit/inv-sev sanity (monotone, Δ* at power 0.8),
      bootstrap CI contains the point estimate, floor anchoring.

## 6. Verification and handover

- [x] 6.1 Focused tests green (patch_scores, eval_dev, new scripts).
- [x] 6.2 Smoke outputs read and reviewed (no NaN, shapes/counts as
      expected) per AGENTS.md §7 before any handover command.
- [x] 6.3 Handover: exact terminal commands for the full E2 run (both arms),
      the Q8 curve, the MDE build; GPU branch (B) stays gated on the audit
      outcome.
