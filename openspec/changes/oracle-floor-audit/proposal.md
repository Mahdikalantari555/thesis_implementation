# Proposal: oracle-floor-audit

> **Status (2026-11-23): DORMANT.** Post-finding-25 analysis of the frozen patch arms on the unassigned pool V (σ-floor verdict). NOT voided — but it restates a direction (patch arms) whose adoption is void under finding 25. Left unmarked deliberately; needs owner review before any follow-up (see INDEX.md).
> See `openspec/changes/INDEX.md` (triage 2026-11-23). Nothing here is
> approved for application without an explicit owner decision.

## Why

Finding 25 (2026-11-21) left the corrected-protocol verdict: MDM80 censored
>S6 for both patch arms, best detection 0.456 (cnn_gru, S6/persistent/savi).
An external consultant panel (Claude, 2026-11-22, answered against our 16-question
deep-research prompt) judged the state "(C), weighted toward (A)": the floor is
mostly a σ floor (measured σ_raw(ndvi) ≈ 0.19–0.25 → S6 ≈ 1.57σ per bin), the
realistic stress range (S1–S4 = 0.10–0.84σ) has zero power at 5% cycle-FAR for
ANY detector, and the recoverable part is the residual definition (absorption
α ≈ 0.45 for one-step baselines) plus the aggregator (GLR/CUSUM vs the trailing
window-mean: predicted +0.10–0.30 power at S6). The consultant's own decisive
recommendation — an oracle-floor audit that measures σ_h and α_h per phase
group and plugs them into a Gaussian oracle power function — gates ALL further
GPU spend. Its two unverifiable numeric assumptions were verified against this
repo on 2026-11-22: σ conversion confirmed (norm = (x−median)/IQR, IQR_ndvi =
0.3867 → S6 = 1.57σ, not 1.7–2.7σ); the design pool corrected to **260 clean
unassigned cycles** (the consultant's 1,033 ignored dev=340/test=773 and the
monitoring_only bucket). Two of its corrections change the gates, not the plan.

## What Changes

- `patch_scores.PatchArmScorer` gains a `gap` realignment parameter
  (forecast of bin t from bins ≤ t−h; h=1 reproduces `one_step_ahead_v3`
  byte-for-byte; causality invariant unchanged — anchor[:, b] uses bins ≤ b).
- NEW `scripts/audit_oracle_floor.py` (E2): measures σ_h (clean residual
  robust σ per phase group) and α_h (absorbed fraction of a persistent S6
  injection, per family) for h ∈ {1,2,3,4,6,8} on the 260-cycle clean
  unassigned pool V (field-blocked V_cal/V_test, seed 13), computes
  Q_h = (1−α_h)·σ_1/σ_h, plugs the best h* into a Gaussian trailing-window
  oracle (w=3, cycle-FAR 5%), and emits the pre-registered (A)/(C)/(B)
  decision per family.
- NEW `scripts/audit_matched_ndvi.py` (Q8): the realized-band-delta
  equivalence curve between the SAVI and NDVI solvers at equal nominal Δ
  (real store baselines, not the consultant's assumed N=0.40/R=0.05), plus an
  optional one-look confirmatory matched-Δ detection cell on the locked eval
  panel.
- `scripts/eval_dev.py` severity cells record `det_flags` per injection and a
  `detection_by_onset` summary (premise-challenge #3: per-onset MDE; enables
  field-blocked bootstrap on future runs).
- NEW `scripts/build_mde_curve.py`: the MDE curve deliverable (floor-anchored
  probit fit over S1–S6, Δ*(0.8) with parametric-bootstrap CI, pre-registered
  Δ_req = 0.165 gate) from the six committed metrics JSONs.

## Capabilities

### New: oracle-floor-audit
- The (A)/(C)/(B) decision must come from measured σ/α on V, never from
  assumed numbers, and must record every input it consumed.
- Gap realignment must preserve the h=1 path byte-for-byte (parity gate).
- The audit is L2 diagnostic machinery: it changes no production scorer
  default, trains nothing, and touches no locked panel hash (V is outside
  train/dev/test; the optional Q8 confirmatory cell is a single pre-registered
  look on the locked eval panel).

## Impact

- Sources: `src/ssl_anomaly/monitor/patch_scores.py` (gap kwarg),
  `scripts/eval_dev.py` (per-injection logging), 3 new scripts.
- Reports: `reports/dev/<run_id>/oracle_floor_{arm}.json`,
  `reports/dev/<run_id>/matched_ndvi_equivalence.json`,
  `reports/dev/<run_id>/mde_curve.json`.
- No behavior change for any existing entry point (gap defaults to 1; the
  severity JSON gains two additive keys).
- GPU spend stays gated: retraining (branch B) only if the audit's gap lands
  ≥ 0.30.
