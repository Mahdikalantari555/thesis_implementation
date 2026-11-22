# Tasks: patch-alarm-statistic-e1

## 1. E1a — wire the field-mean statistic (measured on V)

- [ ] 1.1 `patch_scores`: field-mean alarm path builder (in-field mean of
      standardized residuals, channel-max over touched channels, trailing
      w=3 window) as a selectable statistic; default = production, outputs
      byte-identical when unselected.
- [ ] 1.2 `eval_dev`: `--alarm-statistic` selection; per-statistic LOO
      calibration (paths computed under the identical statistic) +
      calibrate_threshold; loud failure on degenerate τ.
- [ ] 1.3 h* frozen per arm in the run config (pixelwise 8, cnn_gru 1) from
      the committed oracle-floor reports; recorded in the metrics JSON.
- [ ] 1.4 Unit tests: default byte-parity, own-null calibration (a borrowed
      τ raises), field-mean path NaN discipline on unobserved bins.

## 2. E1b — measure pixel-based candidates on V (before any wiring)

- [ ] 2.1 Audit-mode extension: mean-of-pixel-z² and per-channel Tippett
      fusion scored through the per-gap empirical-power leg on V
      (both arms, both indices, same split daf44e65…).
- [ ] 2.2 Report `e1b_pixel_candidates.json` with the family-attribution
      field explicit; wire a candidate only if gap ≥ +0.10 in both families.

## 3. Confirmatory look (once, locked panel)

- [ ] 3.1 Pre-registered gates written into the run config before execution
      (FAR CP-upper ≤ 9.1 % on held-out clean; McNemar gain ≥ +0.10 both
      families; ladder non-flat; honest-negative branch).
- [ ] 3.2 One sweep per adopted statistic on the locked panel (new eval-subset
      hash recorded; historical hashes untouched); metrics schema =
      severity_sweep_patch_v2 + `statistic` + `h_star`.
- [ ] 3.3 Finding 26 written (EN + FA) only from the confirmatory numbers;
      V numbers quoted as design-side evidence, clearly labelled.

## 4. Verification

- [ ] 4.1 Focused tests green (patch_scores, eval_dev, audit).
- [ ] 4.2 Smoke (--limit) of the confirmatory command before handover.
