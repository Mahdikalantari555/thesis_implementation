# Change: score-semantics-realignment

## Why

`PatchCausalGRU`'s horizon-`k` head at bin `t` targets bin `t+k-1` from GRU
state(`t`), which has **consumed** `x_t` (`src/ssl_anomaly/models/patch_cnn_gru.py:148-152`;
loss comparison for horizon 1 at `:325-326`). The deployed score reads horizon 1
at the same bin (`src/ssl_anomaly/monitor/patch_scores.py:559-562`), so the
alarm residual is an **input-inclusive filtered residual**, not a one-step-ahead
forecast. The documentation contradicts the code and itself:

- `patch_scores.py:436-437` claims the horizon-1 output "reads bins ≤ t-1" —
  false.
- `patch_cnn_gru.py:9-14` claims "prediction of bin t from bins < t" —
  contradicted by `:148-152` ("made from bins ≤ t").

Consequences: causal/early-warning claims are unsupported by the implementation
as-is, and the patch arm is scored under a weaker standard than the field-level
`DeepCausalScorer.loo_batch` arms (which withhold the current bin). External
audit 2026-11-20 (verified same day against the local tree).

Owner decision (2026-11-20): **retrain with true alignment** — the doc-only
reframing and the horizon-2 re-score (no retrain) were rejected; the retrain
runs on the v2 store (3,478 clean cycles) after `consistent-spectral-injection`
lands, so training happens exactly once under the corrected protocol.

## What Changes

- **`ssl_anomaly.models.patch_cnn_gru.PatchCausalGRU.hybrid_loss`**: horizon
  alignment shifted so horizon `k` at bin `t` targets bin `t+k+1` from
  state(`t`) (bins ≤ `t`); horizon 1 = prediction of bin `t+1`. Out-of-range
  slots zeroed (keep-mask adjusted). The **aux reconstruction head is
  unchanged** — it is a training auxiliary, never the score. Module and class
  docstrings rewritten to state the true alignment.
- **`ssl_anomaly.monitor.patch_scores.PatchDeepScorer`**: the score residual at
  bin `t` is computed from the prediction of bin `t` made at bin `t-1`
  (state after `t-1`); bin 0 is unscored (NaN). The false "reads bins ≤ t-1"
  docstring claim is replaced by the true alignment statement.
- **Protocol identity**: training/scoring protocol id bumped so v2-aligned
  (input-inclusive) and v3-aligned checkpoints cannot be mixed; the resume
  config-hash guard must fail across the boundary.
- **Tests**: the target-leak invariant (perturbing bin `t`'s values leaves the
  prediction of bin `t` and every score at bins < `t` byte-identical; bin
  `t`'s residual moves only through the observed-vs-forecast comparison — the
  detection signal), a loss-alignment unit test (same-bin identity does NOT
  minimize the horizon loss), bin-0 NaN, docstring/behavior agreement.
- **Docs**: finding-08 mislabel corrected in place (0.7641 vs 0.6013 compared
  two pretext arms, never "SSL beats B1"); findings 23/24 annotated as
  historical under the input-inclusive score + `index_only` basis, superseded
  by the first v3-protocol finding.

## Impact

- **Affected specs:** `model-development-protocol` (ADDED requirement: true
  one-step-ahead scoring alignment).
- **Affected code:** `src/ssl_anomaly/models/patch_cnn_gru.py`,
  `src/ssl_anomaly/monitor/patch_scores.py`, `scripts/train_patch.py`
  (protocol id only), `tests/unit/test_patch_model.py`,
  `tests/unit/test_patch_scores.py`, `docs/findings/08_honest_assessment.md`.
- **Supersession:** findings 23/24 numbers stay on disk as historical records;
  the next full training (v2 store, `patch_train_v3` +
  `bands_consistent`) produces the new primary finding. σ tables are refit
  per-checkpoint by the existing flow (residual distribution changes).
- The finding-24 freeze (`patch_pixelwise`) and the platform wiring stay
  **blocked** until the v3-protocol finding re-confirms the arm choice.
