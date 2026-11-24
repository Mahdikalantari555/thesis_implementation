# Design — score-semantics-realignment

## Context

Verified code facts (2026-11-20):

- `patch_cnn_gru.py:148-152`: "for every bin `t` and horizon `k`, a prediction
  of bin `t + k - 1` made from bins ≤ `t`" — horizon 1 predicts the bin whose
  values the GRU state has already consumed.
- `hybrid_loss` (`:323-340`): horizon `k=0` compares `pred[:, :, 0]` with
  `target` at the same bin.
- `PatchDeepScorer._forward` (`patch_scores.py:559-562`): score residual =
  horizon-1 prediction minus `x` at the same bin; the class docstring
  (`:436-441`) wrongly asserts "reads bins ≤ t-1".
- Module docstring (`patch_cnn_gru.py:9-14`) asserts "bins < t" — internally
  contradicted.

Owner chose retrain-with-true-alignment over the two cheaper paths; the retrain
is deferred to the v2 store (3,478 clean cycles) so it runs exactly once.

## Decisions

### D1 — Shift at loss + scorer only; `forward()` untouched

The causal property lives in the head/target alignment, not in the recurrence:
the GRU forward is already strictly causal w.r.t. future bins. Concretely:

- `hybrid_loss`: horizon `k` (0-indexed) at bin `t` now targets bin `t+k+1`
  → `pred_k = out["pred"][:, :T-k-1, k]`, `tgt_k = target[:, k+1:]`,
  `obs_k = obs[:, k+1:]`; keep-mask zeroes slots whose target bin does not
  exist.
- Scorer: prediction of bin `t` = `out["pred"][:, t-1, 0]`; residual at bin
  `t` = `x_t − pred_{t|state(t-1)}`; bin 0 → NaN.

Rationale for not restructuring `forward()`: minimal diff, identical parameter
count, the Colab notebook retrains unchanged mechanically; only the semantics
of the existing outputs change.

### D2 — Covariates stay per-bin in the recurrence (state consumes cov ≤ t)

The forecaster predicting bin `t+1` uses covariates of bins ≤ `t`. Target-bin
calendar features (doy/age of `t+1`) are known in advance but are NOT injected
at the head: carrying them in the state across 15-day bins is smooth, and the
alternative (feeding `cov[t+1]` to the head) would also admit the observational
covariates `log1p_n_obs[t+1]` / `days_since_valid[t+1]`, which a pre-observation
forecaster must not see. Mixed per-covariate treatment is recorded as the
rejection rationale; revisit only if the v3 result degrades materially vs the
historical input-inclusive arm.

### D3 — Aux head unchanged

`head_aux` keeps its same-bin reconstruction objective with `aux_weight = 0.3`.
It is a training auxiliary; the score never reads it. Keeping it stabilizes the
stem (dense same-bin signal) without contaminating the scoring semantics.

### D4 — Protocol identity

- `PATCH_TRAIN_PROTOCOL = "patch_train_v3"` (was `patch_train_v2`) in the
  training script; scoring-side contract id bumped equivalently.
- The existing resume guard fails on `config_hash` mismatch; the alignment
  version enters the hashed config so a v2-aligned checkpoint cannot resume or
  be scored under v3 without an explicit error.

### D5 — σ tables refit per checkpoint (existing flow)

Residual distribution changes under the new alignment; the per-run
`calibrate_patch_sigma.py` step already refits per checkpoint and binds
model/norm hashes. No new machinery.

### D6 — Freeze and platform sequencing

Finding 24's frozen winner (`patch_pixelwise`) and the blocked
`platform-patch-scorer-wiring` change are NOT touched here. The v3-protocol
finding (v2 store) re-runs the severity gates + the localization probe; if the
arm ranking flips, the freeze decision is revisited there. Scope discipline:
no platform wiring in this change.

## Risks / trade-offs

- **Risk:** MDM80 may worsen (a true forecaster has strictly less information
  than an input-inclusive residual). *Accepted* — this is the honest early-warning
  number; the input-inclusive arm survives as the historical comparison.
- **Risk:** bin 0 losing its score shortens usable paths by one bin.
  *Mitigation:* negligible (36 bins); `min_pixels`/`min_history` rules
  unchanged.
- **Risk:** horizon weights (1.0/0.5/0.25) were tuned under the old alignment.
  *Mitigation:* unchanged here; retuning is a post-v3-finding decision, not a
  silent knob.

## Migration

1. Land code + tests (this change).
2. `consistent-spectral-injection` lands (independent module).
3. v2 store extraction completes (owner, in-flight chain).
4. Owner-run retrain ×6 (2 arms × 3 seeds) on Colab with the v2 notebook
   pinned to the commit containing both changes; σ ×6; severity sweep ×6 under
   `severity_sweep_patch_v2` / `bands_consistent`.
5. New finding becomes the primary patch-level record; findings 23/24 get the
   superseded annotation; platform freeze decision revisited.
