# Proposal: patch-alarm-statistic-e1

> **Status (2026-11-23): DORMANT.** Unapplied (0/11) child of oracle-floor-audit on pool V (channel-max residual aggregation; empirical power 0.30–0.67 vs committed 0.06–0.46). Not voided by finding 25; blocked on an owner decision to revive any patch direction.
> See `openspec/changes/INDEX.md` (triage 2026-11-23). Nothing here is
> approved for application without an explicit owner decision.

## Why

The oracle-floor-audit change (2026-11-22) measured, on the clean unassigned
pool V (260 cycles, field-blocked V_cal/V_test), that the frozen patch arms'
residuals carry a REAL recoverable gap that the production alarm statistic
does not collect: a field-mean aggregation of the standardized per-pixel
residuals (channel-maxed, trailing window w=3) with gap-h alignment reaches
empirical power 0.298/0.611 (pixelwise ndvi/savi) and 0.450/0.672 (cnn_gru)
at a per-gap τ frozen from the V_cal clean null — vs the committed finding-25
detection rates 0.062/0.206/0.209/0.456. Gaps +0.236/+0.405/+0.242/+0.216,
stable across both field-blocked halves (power_cal ≈ power_test on all four).
The mechanism is a STATISTIC swap on frozen residuals — no retraining — so
the operative branch is C (statistic redesign), and branch B (retrain) stays
gated on this change's confirmatory outcome.

Owner directive (2026-11-22): record this as an openspec change. Scope note
recorded verbatim: the measured gain belongs to the FIELD-MEAN statistic; the
pixel-based production statistic has NO measured alternative yet — candidate
pixel-level aggregations are measured on V inside this change BEFORE any
confirmatory look, and are wired only if they clear the same gates.

## What Changes

- **E1a (measured, ready to wire)**: the field-mean gap-h alarm statistic
  becomes a selectable scoring statistic in the patch eval path
  (`eval_dev.py` + `patch_scores`), with its own leave-one-out empirical-p
  calibration against the same calibration subset (fail-closed, same
  discipline as the production statistic). h* is frozen per arm from V_cal
  (pixelwise h*=8, cnn_gru h*=1) BEFORE the confirmatory run.
- **E1b (unmeasured, measure-first)**: candidate pixel-based aggregation
  alternatives to top-20% — mean-of-pixel-z² over observed in-field pixels,
  and per-channel p-fusion (Tippett) — are measured on V with the same
  audit machinery; wired only if their V-measured gap clears the same gate
  as E1a.
- **ONE confirmatory look on the locked eval panel**: a single eval run
  carries the wired statistics through the full severity sweep with the
  standard calibrated machinery; gates below are pre-registered and the
  honest-negative branch is explicit.
- The MDE curve (finding 25 companion report) stands unchanged as the floor
  document; no committed number is rewritten.

## Capabilities

### New: patch-alarm-statistic-e1
- Every candidate statistic is calibrated on its OWN null (per-gap τ from
  V_cal; LOO empirical-p on the calibration subset inside eval) — never a
  borrowed threshold.
- The confirmatory run happens ONCE on the locked eval panel; V is a design
  surface only (V↔train field sharing, 243/251, is declared).
- A statistic is adopted ONLY if its gate passes; the honest-negative branch
  records the ceiling as measured and closes the branch.

## Impact

- Sources: `scripts/eval_dev.py` (statistic selection), `src/ssl_anomaly/
  monitor/patch_scores.py` (field-mean path + aggregation variants),
  `scripts/audit_oracle_floor.py` (E1b measurement mode reuses the audit).
- Reports: `reports/dev/<run_id>/e1_confirmatory_<statistic>.json`.
- No committed finding-25 number changes; the confirmatory run carries the
  NEW eval-subset hash (the calib/eval partition moved to field-based in
  1f5c823 — 852e1ba3/6186edf8 stay historical).
- GPU: zero (all V work is CPU; the confirmatory run is one CPU sweep per
  statistic, or Colab T4 minutes if preferred).
