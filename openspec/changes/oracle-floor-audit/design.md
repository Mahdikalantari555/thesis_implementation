# Design: oracle-floor-audit

## Context

Finding 25's corrected protocol measures detection from the injection onset
with a calibrated τ (FAR 8/170 = 4.71% achieved). The consultant panel's
verdict makes three load-bearing quantitative claims that this change turns
into measured artifacts:

1. **σ floor**: per-bin innovation σ in raw NDVI units ≈ 0.19–0.25 (verified:
   norm = (x − median)/IQR with IQR_ndvi = 0.3867; fitted σ_norm(ndvi) median
   0.658 → σ_raw ≈ 0.254; S6 = 0.40/0.3867/0.658 ≈ 1.57σ). The realistic
   range S1–S4 = 0.10–0.84σ has no power at 5% cycle-FAR for any detector.
2. **Absorption**: a one-step-ahead baseline absorbs a sustained shift
   (α ≈ 0.45 measured for gru_ae at h=1, finding 20). The recoverable lever is
   the conditioning gap h: forecasting bin t from bins ≤ t−h removes the
   shifted neighbours from the causal context (GRU hidden state at t−h cannot
   see them), trading absorption against forecast noise.
3. **Aggregator**: the trailing window-mean of the channel-max path is
   sub-optimal vs GLR/CUSUM by a predicted +0.10–0.30 power at S6.

The audit measures (1) and (2) directly and computes what (3) would buy,
replacing every assumed number with a measured one.

## Goals / Non-Goals

- Goals: measured σ_h/α_h curves per phase group; the pre-registered
  (A)/(C)/(B) decision per family; the SAVI≫NDVI mechanism resolved with real
  baselines; per-onset detection logging; the MDE curve deliverable.
- Non-Goals: no retraining (branch B is only triggered by the audit's gate);
  no change to the locked eval panel hashes; no new severity family; no
  thermal/salinity channels (consultant-excluded to the L4 track — the
  owner-binding index-drop injector cannot move LST, so a thermal arm carries
  zero synthetic signal).

## Decisions

### D1 — gap realignment is a post-hoc view of one forward pass

`pred_anchor[:, b]` forecasts bin b+1 from bins ≤ b (verified invariant).
Forecasting bin t from bins ≤ t−h is therefore `pred[:, t] = anchor[:, t−h]`
— a numpy realignment of the SAME forward output, not a second pass. One
forward per (chunk, clean/injected) serves all h values; the whole E2 audit
needs 3 forwards per chunk per arm (clean + one per injected family).
`gap=1` must reproduce the current `one_step_ahead_v3` output bit-for-bit
(parity asserted in tests and by the audit script itself on its first chunk).

### D2 — V pool and its split

V = clean cycles ∉ train/dev/test = **260 cycles over 251 fields** (verified
2026-11-22 from splits_locked_v2 + cycles_490_v2.parquet; consultant's 1,033
was wrong). V is partitioned into V_cal/V_test by FIELD (rng(13) permutation
of sorted field ids, half/half). Known and declared: 243/251 V fields share
cycles with the train field set — acceptable for a σ/α ceiling audit (the
scorer is frozen; V never selects a model), not for confirmatory gates, which
stay on the locked eval panel with a one-look rule.

### D3 — absorption and σ measured on the scorer's own units

- σ_h(g) = 1.4826 · MAD of clean signed residuals (pred − x) of the ndvi
  channel, per phase group g (the σ table's 18 groups), per gap h, on V_cal.
- shift_h = median(resid_inj − resid_clean) over post-onset bins (ndvi
  channel); α_h = 1 − shift_h/Δ_norm with Δ_norm = Δ_raw/IQR_ndvi.
- Q_h(g) = (1−α_h) · σ_1(g)/σ_h(g) — the consultant's pre-registered SNR
  index; gate: max_h Q_h ≥ 0.72 in ≥ 12/18 phase groups on V_cal, confirmed on
  V_test; honest-negative if max_h Q_h < 0.62 everywhere.
- The oracle power uses a Gaussian trailing-window statistic with the SAME
  window as the production alarm path (w = 3), cycle-FAR 5% over T = 36 bins,
  per-bin shift δ = (1−α_h*)·Δ_norm/σ_h* — closed-form with a numerical
  root-find for the threshold. iid-Gaussian optimism and the channel-max
  multiplicity are recorded as unquantified conservatisms (consultant's own
  flags).

### D4 — Q8 with real baselines

The consultant's SAVI-equivalence arithmetic assumed canopy N=0.40/R=0.05.
The audit measures the solvers on real store patches: realized per-channel
in-field-mean deltas (ΔR, ΔN) of the SAVI and NDVI solvers at each ladder Δ,
the band-magnitude ratio ρ per level, and the NDVI-equivalent Δ* = Δ·ρ. The
confirmatory cell (optional flag) scores Δ* as an NDVI-family injection on
the locked eval panel with the standard calibrated machinery and compares
against the committed cells (SAVI 0.185–0.456 vs NDVI 0.059–0.062): matched ≈
SAVI ⇒ re-parameterize severity in band space; matched ≈ nominal ≪ SAVI ⇒
scorer asymmetry (whiten the (R, NIR) subspace); between ⇒ both levers.

### D5 — per-injection logging is additive

`eval_dev.py` severity cells gain `det_flags` (list, aligned with
`injections` order) and `detection_by_onset` (per onset phase n/rate). No
existing key changes; the fixed-protocol metrics stay hash-comparable.

### D6 — MDE curve is a reporting script, not a gate change

`build_mde_curve.py` fits a floor-anchored probit p(Δ) = floor + (1−floor)·Φ(a
+ b·Δ) per arm × family on the six committed levels, reports Δ*(0.8) with a
parametric-bootstrap CI and the pre-registered Δ_req = 0.165 (top of the
realistic range) verdict. Censored outputs are labelled ">S6" plus an
extrapolation explicitly marked as a model, not a measurement. Field-blocked
bootstrap is deferred until per-injection logging data exists for a run
(this change enables it for all future runs).

## Risks / Trade-offs

- V is small (260 cycles, median 1 cycle/field) → wide CIs on α/σ per group;
  the gates use medians over 18 groups for exactly this reason.
- gap-h scoring puts the model slightly out of its training condition
  (input windows end h−1 bins earlier); masked-pretext training makes this
  in-distribution-ish, and the σ_h curve itself measures the penalty.
- The oracle is iid-Gaussian; a gap ≤ 0.10 verdict could still hide real
  recoverable power under heavy tails. The verdict therefore reports the
  measured σ/α tables alongside the decision so the committee sees the raw
  evidence.

## Migration Plan

Additive only: new kwarg (default preserves behavior), new JSON keys, new
scripts, new reports. Nothing existing is rewritten.

## Open Questions

- Whether the Q8 confirmatory cell should run before or after E2's decision —
  default: after; it only matters if severity re-parameterization is on the
  table (branches A/C).
