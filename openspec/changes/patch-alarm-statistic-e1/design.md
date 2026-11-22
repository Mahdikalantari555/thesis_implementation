# Design: patch-alarm-statistic-e1

## Context

The oracle-floor audit measured, on frozen residuals of the two production
patch arms, that a field-mean aggregation with gap-h alignment collects a
+0.22 to +0.40 detection gain over the production statistic at the same 5 %
cycle-FAR budget — stable across field-blocked splits, achieved with ZERO
retraining. The owner directed an openspec change. The measured gain belongs
to the field-mean statistic; the pixel-based production statistic has no
measured alternative yet. This change wires the measured one and measures the
pixel-based candidates under identical discipline.

## Goals / Non-Goals

- Goals: a selectable alarm statistic per patch run, calibrated on its own
  null; ONE confirmatory look per adopted statistic on the locked panel;
  pre-registered gates with an honest-negative branch; no committed number
  rewritten.
- Non-Goals: no retraining (branch B stays gated on the confirmatory
  outcome); no platform wiring (dead since finding 25 — a later change);
  no new injection families; no partial-extent injections here (they belong
  to the L4-adjacent program).

## Decisions

### D1 — statistic definition (E1a, wired)

For a patch batch scored with gap-h realignment, the alarm evidence per bin
is the channel-max over the injector-touched channels of the in-field-mean
standardized residual, standardized by the SAME σ tables the production
scorer uses, and the alarm path is the trailing w = 3 window mean. The p-value
is the per-bin empirical rank of that path against the calibration subset's
paths computed under the IDENTICAL statistic (same gap, same aggregation) —
never a borrowed null. τ is `calibrate_threshold` on the LOO calibration null
at FAR target 0.05, recorded with `far_achieved` and `tau_calibration`
exactly like the production path.

### D2 — h* frozen per arm before the look

h* is selected on V_cal by empirical power (pixelwise h*=8, cnn_gru h*=1 per
the committed oracle_floor reports) and HARD-FROZEN in the run configuration
before the confirmatory sweep. No post-hoc h shopping.

### D3 — E1b candidates measured, not assumed

Pixel-based candidates: (a) mean-of-pixel-z² over observed in-field pixels
(replacing top-20%), (b) per-channel Tippett fusion of the per-channel
empirical p's. Both are scored through the audit machinery on V with the
oracle-floor-audit's per-gap empirical-power leg; a candidate is wired only
if its V-measured gap ≥ +0.10 in BOTH index families (same gate as the
confirmatory). The Fisher-null caveat (shared reference invalidates Fisher
theory) is moot because τ is empirical — recorded in the report.

### D4 — confirmatory gates (pre-registered, one look)

On the locked eval panel (NEW subset hash — the partition is field-based
since 1f5c823; the historical hashes stay pinned in their findings):

1. **FAR discipline**: `far_achieved` ≤ 0.05 on the calibration null AND the
   held-out clean eval paths' post-onset alarm rate ≤ 0.05 with
   Clopper–Pearson upper ≤ 9.1 %.
2. **Detection gain**: paired (McNemar) detection gain ≥ +0.10 over the
   production statistic in BOTH index families (ndvi and savi) at the level
   where the V-promise was largest.
3. **Ladder liveness**: detection non-flat across S1–S6 with S6 ≥ S1 per
   family (the finding-25 artifact check).
4. **Honest-negative branch**: any gate fails → the statistic is NOT adopted;
   the report records the ceiling and the branch closes (retraining B is then
   evaluated against the confirmed ceiling, not the V promise).

### D5 — bookkeeping and hashes

The confirmatory metrics JSON records: statistic id, h*, gap, τ, far_achieved,
the new eval-subset hash, σ-table sha, git sha, and `tau_calibration` — same
schema as severity_sweep_patch_v2 plus a `statistic` field. V reports stay
under the audit change; nothing in `docs/findings/` is edited until the
confirmatory run lands (then finding 26).

## Risks / Trade-offs

- V↔train field sharing (243/251) inflates design-side optimism; the one-look
  panel result is the only quotable number.
- The field-mean statistic's per-cycle residual correlation (pixelwise
  n_eff ≈ 2.4) makes its empirical null heavier-tailed than iid — the τ from
  129-cycle V_cal and the 170-cycle panel calibration must both hold FAR; if
  the panel calibration cannot reach 5 % without degenerate τ, that IS the
  honest-negative outcome.
- E1b pixel candidates may inherit the top-frac pathologies (heavy-tailed
  null) — they are measured before wiring for exactly this reason.

## Migration Plan

Additive: a `--alarm-statistic` selection (default = production, byte-identical
behavior), new report files. The default path's outputs stay hash-comparable.

## Open Questions

- Whether the confirmatory sweep runs locally (CPU ~35–60 min/statistic) or
  on Colab T4 (~5 min) — decided at implementation time by quota.
