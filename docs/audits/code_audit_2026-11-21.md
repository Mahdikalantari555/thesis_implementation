# Code audit 2026-11-21 — residual bd65829-class logic defects

Read-only audit of `thesis_implementation` one day after bd65829 (the severity-ladder
fix). Two independent read-only sweeps (monitor/eval engine + data/injection/training),
every critical finding then re-verified line-by-line against the code before acting.
Full context: commit bd65829 (bin-0 NaN poisoning, hardcoded tau, fit_patch_sigma
axis collapse) and the audit commit series below.

## Fixed in this pass (result-corrupting, owner-approved 2026-11-21)

| Finding | Location | Defect | Fix |
|---|---|---|---|
| A1 (P0) | `scripts/calibrate_patch_sigma.py:115-127` | σ fit omitted the per-channel `obs` mask — zero-filled (never-acquired) channels entered `fit_patch_sigma` as fabricated `pred − 0` residuals; MAD inflated, `n` overcounted, σ artifact not the claimed statistic; gate-blind (both parity legs share σ) | 99709b0: `predict_pixels` returns the per-channel `observed` mask; calibration consumes it. **Refit ×6 proved the v2 tables BYTE-IDENTICAL** (same sha256, same n): the v2 store has all-or-nothing per-scene observation — probe across sampled dev cycles found 0 pixel-bins with `0 < channels < 8` — so the defect was contract-real but numerically inert on this store. The fix stands (any future store with partial channel availability — e.g. TIRS failing while OLI succeeds — would have poisoned σ silently). No bundle change resulted. |
| A2 (P1) | `scripts/eval_dev.py:873-876`, `src/ssl_anomaly/monitor/patch_eval.py:150-176` | Patch detection/delay measured from bin 0 (only call site in the repo without onset) → pre-onset false alarms counted as detections (+calibrated-FAR floor per cell, MDM80 boundary risk) and delays +45 d; and every round of a cell was a byte-identical injection (rng consumed only by `spatial="white"`; onset pinned to bin 3) → Wilson CI over pseudo-replicates at `--rounds 2` | ea74a0c: onset read from the per-injection `onset_bin`, delay `(fb − onset)·15` (field-arm convention), `onset_bins` recorded per cell; onset phases rotate per `(sample + round) % 3` and gradual ramps cycle `GRADUAL_RAMP_BINS` per round — exact field-ladder parity |
| A3 (P1) | `scripts/probe_forecast_units.py:176` | Path-C pre-registered budget compared event-FAR (false events per **scored bin**) to the 5 % **per-cycle** target — with ~28-30 scored bins the condition could never bind (one false event in every clean cycle ≈ 0.036 < 0.05) | d4b3cb8: gating condition moved to the cycle-level FAR of the clean evaluation paths at (τ, k); event-FAR kept as a descriptive record; amendment dated in output + docstring. Finding 21 (all-censored NEGATIVE) unaffected |

Gate after fixes: `gate_patch_severity.py` PASS on cpu for both archs (chunking parity +
liveness: tau 0.6464, far_achieved 0.000, S6 detection ≥ 0.8).

**Consequence for results:** the patch-arm numbers produced before 2026-11-21 (MDM80
tables, detection rates, delays, localization AUPRC, findings 18/22/23/24 patch rows,
the ×6 v2 severity runs) are invalidated by the A2 accounting (onset-0 detection,
+45 d delays, duplicate-round CIs) and by the bd65829 tau/NaN artifacts they already
carried — NOT additionally by A1, whose σ contamination is provably inert on the v2
store (refit byte-identical). Field-level arms are NOT affected by A1/A2 (no patch σ,
onset-true accounting already).

## Deferred (recorded, not fixed — none corrupts a current result)

### P1 — defect-in-waiting (no production caller today; fix before first use)

1. **`indicator_z(unit="farm_own")` shrinkage is a no-op** — `src/ssl_anomaly/monitor/indicators.py:211-217`:
   `pooled_scale = own_scale` collapses `scale = w·own + (1−w)·pooled` to `own` for every
   `w`; κ=10 dead. Thin-history fallback (201-209) labelled "pooled fallback" actually
   z-scores against the farm's own 2-4 points. Fix: pass the pooled reference explicitly
   (population or `FieldSigmaTable.pooled`), use it in shrinkage target + fallback, pin
   `n→0 ⇒ pooled` in a test.
2. **`absorption_a` polarity inverted** — `src/ssl_anomaly/train/contamination.py:183-208`:
   formula `a = (pred_cont − pred_clean)/(x_cont − x_clean)` measures how much of the
   injection the OUTPUT follows (a=1 ⇒ absorber), but the docstring says a≈1 = detector
   and `share_absorbed = 1 − a` labels the complement; tests pin the wrong reading
   (`tests/unit/test_contamination.py:130,137`). Siblings correct:
   `monitor/diagnostics.py:141-146`, `scripts/trial_gates.py:24`. Fix before the E7
   patch-contamination report: keep the formula, `share_absorbed = a`, correct docstrings
   + test names.
3. **GRU `weight_hh` orthogonal init covers rows 0:63 of 192** —
   `src/ssl_anomaly/models/patch_cnn_gru.py:217-219`: chunk size `p.shape[1]//3` (=64)
   instead of `p.shape[0]//3` (=192) → only one third of one gate initialized; 67 % of
   the recurrent kernel at PyTorch default. Both patch arms share it; seed-deterministic,
   no crash. Fix: one-line change + retrain, OR record the deviation in manifests
   (existing checkpoints were trained under the partial init).

### P2 — hygiene / latent (fix opportunistically, before the code they touch is next used)

4. `src/ssl_anomaly/monitor/farm_local.py:196` — `other_farm_perturbation_is_invariant`
   tautological: `np.array_equal(base, base)`; the perturbed score `again` is never
   compared; access record is fabricated by `score_field_only`. Dead independence gate.
5. `src/ssl_anomaly/monitor/indicators.py:114-115` — CWSI bad cells (≈0 denominator)
   clipped to 1.0 in `value` (max stress) instead of NaN.
6. `src/ssl_anomaly/monitor/scoring.py:225` — `S[isinf]=NaN` kills the alarm path on
   +inf z² (inf − inf = NaN in the trailing sums), contradicting the line-202 comment
   "±inf stays alarm-worthy". Fix: map +inf to a large finite sentinel before cumsum.
7. `scripts/eval_dev.py:582-594` — `--shapes` without `acute` → KeyError in the AUC-PR
   block (loud crash in an anticipated configuration).
8. `src/ssl_anomaly/monitor/patch_scores.py:116` — out-of-range sensor id silently uses
   sensor 0's σ; should raise (verify v2-store sensor-id coverage first — see the
   recorded "sensor_id unknown" limitation).
9. `src/ssl_anomaly/monitor/farm_local.py:318` — wrong-shaped mask silently ignored when
   fitting per-field σ (`v[m] if m.shape == v.shape else v`); raise instead.
10. `src/ssl_anomaly/monitor/zonal_dist.py:437-440` — `calendar_doy_bins` docstring says
    bin midpoint, code uses bin start (internally consistent; correct the docstring).
11. `src/ssl_anomaly/monitor/normal_bank.py:152-153, 215` — global fallback un-gated by
    `min_class_count`; `_loo_q95` with an empty LOO neighbourhood returns hardcoded 1.0.
12. `src/ssl_anomaly/monitor/scoring.py:372-384` — `partial_auc_pr` breaks ties
    clean-before-injected (slightly conservative AP vs sklearn's group-tie handling);
    document or align.
13. `scripts/eval_dev.py:282-306` — dev calib/eval hash partition is per-cycle, ignoring
    field identity: cycles of one field can straddle the boundary (within-dev spatial
    bleed; threshold-calibration stakes only). Fix: partition by field id. NOTE: changes
    the eval-subset hash — coordinate with any pinned eval_hash records.
14. `scripts/eval_dev.py:1061 vs 1063` — field-level metrics JSON permits NaN literals;
    patch path uses `allow_nan=False`. Pick one policy (explicit null for not-measured).
15. `src/ssl_anomaly/models/patch_cnn_gru.py:262-278` — two stacked docstrings; the bound
    `__doc__` asserts the WRONG horizon alignment (`t+k−1`; code and loss realize
    `t+k+1`). Delete the stale one — alignment confusion is the bd65829 theme.
16. `src/ssl_anomaly/models/patch_cnn_gru.py:352-364` — `hybrid_loss` denominators
    multiply by `B`: loss/grads ∝ 1/batch_size; grad-clip (norm 1.0) and recorded
    `train_loss` are batch-size-dependent (AdamW ratios unaffected). Fix + note that
    checkpoints trained pre-fix carry the old scale.
17. `src/ssl_anomaly/models/masked_tf.py:196-218` — loss masks per-token (`any` channel),
    docstring promises per-channel; unobserved channels are trained toward 0. Arms are
    internally consistent; correct the docstring or change the objective (needs retrain).
18. `src/ssl_anomaly/monitor/inputs.py:283-295 (and 88-90)` — DOY channels hardcode epoch
    `2013-03-30`; nothing asserts `stats.zarr times[0]` equals it. Assert the invariant
    and pass the epoch through instead of the literal.
19. `src/ssl_anomaly/monitor/inputs.py:50-54` — `PHYSICAL_RANGES` has no `"ndvi"` entry;
    NDVI silently skips the bounds sanity check the other bands get.
20. `src/ssl_anomaly/phenology/cycles.py:337-406` — 68 lines of unreachable duplicate
    cycle-builder after `return` (references undefined `harvest_idx`); delete.
21. `src/ssl_anomaly/phenology/cycles.py:249-258` — `_ys_at` fabricates `0.0` when no
    finite smoothed NDVI is near a boundary → `ndvi_at_harvest` provenance violation
    (§2). Return NaN.
22. `src/ssl_anomaly/monitor/patch_perturbations.py:384-389` — `index_only` NDVI path
    records the requested drop as "realized", ignoring its own clipping (the other two
    paths recompute). Overstates realized severity near the NDVI floor (S5/S6).
23. `src/ssl_anomaly/monitor/models.py:483-486` — per-season-group σ in `B1PhaseZCausal.
    fit` skips the `scale_floor` the pooled path applies; near-zero group MAD explodes z.
24. `scripts/train_patch.py:196-200, 532-543` — contamination injects with the default
    `injection_basis=index_only` while the primary evaluation gate is `bands_consistent`;
    `contamination_meta` does not record the basis. Align + record.
25. `src/ssl_anomaly/models/pretexts.py:439-440, 521-522` — contrastive scorers silently
    switch semantics when `ref_embeddings is None` (raw embedding norm / constant-0);
    raise instead.
26. `scripts/extract_patches.py:702` — `counts[b] = min(int(agg.n_obs), 255)` silent
    uint8 clamp; raise if ever exceeded.
27. `scripts/train_patch.py:503`, `scripts/build_patch_norm.py:32`, `scripts/eval_dev.py:981`
    — `load_splits_locked(..., verify_cycle_files=False)` disables the parquet/NPZ drift
    check on all production paths (deliberate for Colab per 36a5fcb); the guard now runs
    only in `build_holdout_manifest.py`. Print a notice when skipped, or opt-in flag.
28. `src/ssl_anomaly/monitor/path_b.py:303-305, 321-327, 332-337, 442` — patch-tensor
    appliers measure/anchor at spatial pixel (0,0), skip a whole bin on any non-finite
    pixel, and compute realized shift over unwritten pixels. Latent (field-level callers
    are correct). Make mask-aware BEFORE any patch-level path-B use, or fail closed.
29. Dead-but-exported machinery (staged, tested, no production caller):
    `scoring.channel_max_from_pred/per_bin_q95_normalizer/normalize_scores`,
    `patch_scores.cusum_path/ewma_path/tail_max/calibrate_cusum_h`,
    `zonal_dist.fit_score_calendar/pathmax_scores/channel_z2`, all of `indicators.py`,
    the `path_b.py` appliers. Each is one silent misuse away from a finding (item 1
    already happened inside `indicators.py`); wire up or annotate in docs.

### Performance opportunities (ranked; none affects results)

| # | What | Where | Est. |
|---|------|-------|------|
| 1 | `output_sha256` materializes the whole store (`root["values"][:]`) to hash; stream row-slabs (byte-identical digest) | `io/patches.py:850` | removes the largest export memory spike |
| 2 | `causal_offset_correct` triple Python loop with per-bin `np.median`; per-(n,c) nanmedian over a `sliding_window_view` of the compressed observed series (bitwise-parity test required) | `scoring.py:123-139` | 10-50× on this function; dominant CPU cost after scoring |
| 3 | `PatchSigmaTable.lookup` rebuilds a (T, C) table per sample; `pixel_scores` loops N — hoist per-sensor (G, C) tables + fancy-index gather | `patch_scores.py:109-121, 216-223` | 5-20× on σ-sweep scoring |
| 4 | `field_paths` (N, T) Python loop with per-cell `np.partition` → vectorized top-k | `patch_scores.py:257-266` | moderate (12k iterations × 24 cells) |
| 5 | `cell_q95_normalizer` (C × T) loop → one masked `nanpercentile` | `scoring.py:142-159` | 5-10× |
| 6 | `build_patch_norm` accumulates float64 lists then concatenates (double peak RAM) | `io/patches.py:653-664` | ~½ peak RAM of the norm build |
| 7 | `normal_bank.score_batch` per-(n, b) loop with dict lookups → group-vectorize | `normal_bank.py:243-254` | 5-20× for bank arms |
| 8 | `eval_dev.py:623-641` re-runs the full S3 injection per phase-summary just to read meta — store meta from the main cell loop | `eval_dev.py` | low-moderate |
| 9 | `patch_cycle_index(root)` / `open_store` re-built multiple times per run — cache per root | `eval_dev.py:780`, `patch_eval.py:48-49` | low-moderate |
| 10 | eval_dev bank arms re-run `apply_norm` on data already normalized inside `score_cycles` | `eval_dev.py:466` | low |

## Files audited clean (no findings of this class)

`monitor/holdout.py` (fail-closed; one design note: `assert_train_only` skips temporal
checks when the caller omits dates — enforcement sits with callers), `monitor/patch_eval.py`,
`monitor/splits.py`, `scripts/build_locked_splits.py`, `scripts/build_holdout_manifest.py`,
`train/run_plan.py`, `models/tcn_challenger.py` (causality genuinely tested),
`io/patches.py` geometry/norm-freeze chain.

## Addendum — owner disposition, 2026-11-22 (doc-sync only, no code changes)

Per the owner's decisions of 2026-11-22, the deferred items above are NOT fixed
before the proposal defense. Dispositions:

- Training-path items **3** (GRU `weight_hh` init), **16** (`hybrid_loss` ×B
  scale) and **24** (contamination basis) land as the first act of the
  post-defense openspec change that next trains; item **17** gets the docstring
  fix only (the per-token objective is kept — provably equivalent on the
  all-or-nothing v2 store per the A1 refit).
- Item **1** (indicator `farm_own` shrinkage no-op) fixes inside the farm-local
  change before first use; item **2** (`absorption_a` polarity) fixes as a
  labels-only change before the E7 report (dated spec amendment: adopt LOWEST
  `a_patch`; intent unchanged, so not goalpost-moving).
- Item **13** (dev calib/eval per-cycle partition) lands with the next
  eval-affecting change (cheap re-eval; eval-subset hash change → finding-25
  amendment).
- Until then a deferral gate in `todo.md` blocks new patch training/eval runs.
  Deviations are recorded in finding 25 ("Known training-path deviations"); no
  retrain was performed — the audit's own judgment stands: none corrupts a
  current result.
