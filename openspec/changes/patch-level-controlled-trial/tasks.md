# Tasks — patch-level-controlled-trial

## 1. Model + trainer (TDD-first)

- [ ] 1.1 Causality and masking tests: x̂_{t+k} depends only on bins ≤ t
      (perturbing future bins changes nothing); state-carry test (GRU state
      unchanged where pixel_valid = 0); NaN safety; GroupNorm regression
      test on degenerate masked batches (no BatchNorm anywhere).
- [ ] 1.2 `src/ssl_anomaly/models/patch_cnn_gru.py`: stem + per-location
      GRU + k=1–3 prediction heads + masked-recon aux head; param-count
      assertion ≈90k ±20% with a hard ≤500k gate; CNN-off flag =
      `patch_pixelwise` (same interface, no stem).
- [ ] 1.3 `src/ssl_anomaly/train/patch_data.py`: Dataset over patches.zarr
      (sample = field-cycle), frozen normalization read, covariate builder
      (sin/cos DOY, cycle age, days-since-last-valid, log1p n_obs, sensor
      one-hot L8/L9), NaN-as-absent masking; holdout filter + fail-closed
      assertion via the shared `monitor/holdout.py` helper
      (`training-holdout-invariant`: holdout farms/dates never reach
      training, even though the store contains them); collate tests.
- [ ] 1.4 `scripts/train_patch.py`: hybrid objective (w = 1, 0.5, 0.25 +
      0.3 aux), 8-bin burn-in, trimmed top-5% after 10-epoch warm-up,
      AdamW/cosine/clip; tqdm + progress.jsonl + final wall-time summary;
      --seed/--epochs/--batch-size/--max-fields/--smoke/--resume/
      --estimate-runtime/--device auto-select; per-epoch checkpoint with
      RNG states + config-hash check; run manifest with seeds, data_source,
      split hashes, git sha, device, wall time (AGENTS.md §7 report fields).

## 2. Scoring + eval integration

- [ ] 2.1 Patch σ tables: 1.4826 × MAD of dev residuals per (channel,
      2-bin phenophase group, sensor), frozen + sha; unit test proving no
      test/dev leakage into scoring-time statistics.
- [ ] 2.2 `PatchDeepScorer` adapter: per-pixel z² → top-20% aggregation →
      (N, 36, C) interface-compatible with `DeepCausalScorer.loo_batch`;
      alarm machinery (τ/k/q95, 5% FAR) untouched; per-pixel score maps
      exported for localization metrics.
- [ ] 2.3 Patch-tensor injection applier in `monitor/perturbations.py`:
      same S1–S6 ladder × {acute, persistent} × {ndvi, savi}, footprint
      1.0 (protocol) / 0.3 (localization probe), the existing
      sum-preserving red/nir SAVI solver per pixel; injection-hash equality
      test against the field-level protocol sets.
- [ ] 2.4 eval_dev.py: scorer choices patch_cnn_gru / patch_pixelwise;
      protocol `severity` unchanged; MDM80 censoring + AUPRC + detection
      delay + pixel-localization AUPRC (footprint-0.3 probe) in the report.

## 3. Subset PoC run (user-executed)

- [ ] 3.1 Pre-register the subset manifest (hashed; default two KMeans
      train blocks ≈100 fields) and freeze the gates in a
      trial_gates-style module BEFORE any training run.
- [ ] 3.2 Agent executes `--smoke` + `--estimate-runtime` +
      overfit-one-batch check; reviews produced files (not console only);
      hands exact user commands (conda env geo, cwd, expected wall time,
      output paths).
- [ ] 3.3 USER runs: subset training seeds {13, 42, 2026} + severity eval.
- [ ] 3.4 Agent reviews produced reports (metrics JSONs, dev_log rows) and
      completes every report field (params, wall time, device, git sha,
      data_source, splits sha, hashes) — missing fields are defects, not
      "done" (AGENTS.md §7).

## 4. Gates + verdict

- [ ] 4.1 Evaluate gates verbatim: G1 smoke · G2 overfit-one-batch (L1) ·
      G3 MDM80 finite AND better than best of {b1, gru_ae_main} (paired
      bootstrap over fields, CI excludes zero; both arms censored ⇒ success
      = any finite MDM80) · G4 CNN kept only if it beats patch_pixelwise on
      MDM80 or localization AUPRC · G5 estimate before run + ≤2×
      gru_ae_main compute. Honest-negative protocol; no goalpost movement
      after the first training run starts.
- [ ] 4.2 Finding 23 (Persian, persian-writing skill): patch-level PoC
      verdict with all gate numbers, n, date ranges, seeds, device, hashes
      and explicit L1–L2 rung labelling; scale-up decision (full train run)
      only if G1–G5 pass; otherwise the floor result is recorded verbatim.
