# Tasks — score-semantics-realignment

## 1. Alignment (TDD-first)

- [x] 1.1 Target-leak invariant test FIRST (red): perturb bin `t` values ⇒
      the prediction of bin `t` and scores at bins < `t` byte-identical (bin
      `t`'s residual moves only through the target side — the detection
      signal) (`tests/unit/test_patch_scores.py`).
- [x] 1.2 `hybrid_loss` shifted alignment: horizon `k` at bin `t` targets bin
      `t+k+1`; keep-mask adjusted; loss-alignment unit test (same-bin identity
      does NOT minimize horizon-1 loss; shifted target does)
      (`tests/unit/test_patch_model.py`).
- [x] 1.3 `PatchDeepScorer` readout shift: bin-`t` residual from state(`t-1`);
      bin 0 NaN; tests for bin-0 NaN + finite from bin 1 + unchanged
      aggregation downstream (top-20 %, CUSUM/EWMA, p-value adapter).
- [x] 1.4 Docstrings: module + class in `patch_cnn_gru.py` state the true
      alignment (prediction of bin `t+k+1` from bins ≤ `t`); `patch_scores.py`
      false claim corrected; terminology "one-step-ahead forecasting residual
      from bins ≤ t-1" everywhere the score is described.
- [x] 1.5 Protocol identity: `patch_train_v3` + scorer contract id in hashed
      config; guard lives at the eval/σ-calibration boundary: `alignment_for_protocol`
      maps the manifest protocol to the readout (None = v2 era, documented),
      unknown protocols raise; `protocol` is hashed into the training config
      so resume across the boundary fails.

## 2. Docs

- [x] 2.1 Finding-08 label correction: appended record note (original text
      preserved) stating 0.7641 vs 0.6013 compared two pretext arms and must
      never be cited as "SSL beats B1".
- [ ] 2.2 Findings 23/24 superseded-annotation ADDED only when the v3-protocol
      finding lands (deferred task; tracked here so it is not forgotten).

## 3. Gates and handover

- [x] 3.1 Patch unit battery green (`test_patch_model.py`,
      `test_patch_scores.py`, `test_patch_perturbations.py`,
      `test_patch_eval.py`); standing verification gates (contract +
      bilingual) green.
- [x] 3.2 `train_patch.py --smoke` end-to-end (via the test-suite CLI battery):
      finite losses, manifest carries `patch_train_v3`; then
      `eval_dev.py --smoke` on the smoke checkpoint: bin-0 NaN observed,
      protocol id recorded, no console/page errors.
- [ ] 3.3 Handover (after consistent-spectral-injection + v2 extraction):
      exact Colab retrain command (2 arms × 3 seeds × 10 epochs,
      `patch_train_v3`, v2 store/norm/splits hashes), σ ×6, severity sweep ×6
      under `severity_sweep_patch_v2` / `bands_consistent` — owner-executed;
      agent reviews outputs against §8 before any claim.
