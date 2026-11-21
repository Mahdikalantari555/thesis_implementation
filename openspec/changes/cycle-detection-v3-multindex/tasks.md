# Tasks — cycle-detection-v3-multindex

## 1. Detector (TDD-first)

- [x] 1.1 `detect_harvest_days` refactor: extract v1's Hampel → Whittaker-CV →
      daily-grid → `detect_harvests` pipeline verbatim (`_prepare_and_detect`);
      `segment_cycles` calls it; existing phenology tests pass unchanged
      (19/19 incl. the new fused battery).
- [x] 1.2 Fusion tests first: corroboration rejection (veg drop alone ⇒ no
      candidate), acceptance (veg drop + LST rise ⇒ candidate), dedup within
      180 d, v1-subset equivalence (no evidence ⇒ identical windows),
      determinism, no-v1-harvest path, vf_min gate, lost-span geometry
      (`tests/unit/test_cycles_fused.py`, 11 tests).
- [x] 1.3 `FusionParams` (frozen D2 values) + `detect_harvest_candidates(...)`
      on the raw observation series (median context windows, vf_min gate).
- [x] 1.4 `segment_cycles_fused(...)`: boundaries = v1 harvests ∪ candidates;
      shared window builder `_build_cycles_from_boundaries` with v1 semantics;
      leading/trailing censoring; replant detection kept for v1 harvests
      (D7 limit recorded).
- [x] 1.5 Full unit battery: 1004 passed; 4 pre-existing failures in
      unrelated modules (test_zonal_dist ×2, test_dev_log_schema,
      test_real_archive) — recorded, out of this change's scope.

## 2. Export chain

- [x] 2.1 `segment_and_export_cycles_490.py`: `--detector {v1,fused}` +
      `--cycles-out/--npz-out` + `--limit` (smoke); fused mode REFUSES v1
      paths — preservation guard; v1⊆v2 guard checks CLEAN keys (candidates
      legitimately split v1's censored edge windows); replaced-count recorded.
- [x] 2.2 Provenance sidecar `cycles_490_v2.provenance.json` (git sha,
      FusionParams hash, params, v1↔v2 comparison, wall time).
- [x] 2.3 `build_locked_splits.py`: `--cycles/--cycles-npz/--out/
      --map-exclusions-from-v1`; `build_splits_locked` gained
      `cycles_npz_path`/`derived_from_sha`/`map_exclusions_from_v1`;
      manifests record `cycle_files` paths; superseding builds REQUIRE a new
      `--out` + the overlap transfer.

## 3. Run + gates

- [x] 3.1 Fused export DONE locally (owner directive 2026-11-20 refined the
      compute-placement rule: only >1 h AND CPU-heavy jobs go to Colab;
      the export ran in ~7 min): `cycles_490_v2.parquet` (4,458 windows /
      **3,478 non-censored** / 490 fields; median length 366 d) +
      `growth_cycles_490_v2.npz` (3,478, 36, 8) + provenance sidecar.
      v1 baseline untouched (799 clean, hash 930a6cde…).
- [x] 3.2 D6 gates 4/4 PASS (`reports/cycle_detection_diagnosis/v2_table_gates.md`):
      length median 366 d / 94.4 % in [200,500]; NEW starts 87.5 % Nov–Mar
      (n=2,679); per-field max 12 ≤ 13 (mean 7.1); v1-clean ⊆ v2-clean
      (0 missing). DATA-quality gates only — no model claims.
- [x] 3.3 `splits_locked_v2.json` (sha 30ecbe10…; same k-means seed/blocks;
      train 2,105 / dev 340 / test 773 / monitoring 1,240; cutoff drops
      6.5 %; exclusions transferred by window overlap: 38 v1 keys → 38 v2
      windows, 0 unresolved) + `holdout_manifest_v2.json` (sha d501bf81…,
      archive-end 2026-09-10, audit green). Extraction smoke on 3 fields:
      bindings green (store→splits 30ecbe10, cycle_source 99a5253f =
      v2 manifest record).

## 4. Owner handover (local chain, ~3–4 h)

- [x] 4.1 DONE 2026-11-20/21 (owner ran the resume chain twice — one mid-run
      stop, both resumed clean; final wall 1h31m49s): full store
      `data/patch_runs/v3/patches.zarr` + `norm_stats_full_v2.json`
      (2105 locked train cycles, inner sha `493f0395…`).
- [x] 4.2 Agent verification 2026-11-21 — ALL GREEN:
      manifest 490 records / 490 unique field_ids, every n_scenes > 0
      (min 65 / median 324 / max 535), Σn_cycles = 3478 = store cycle_valid
      total = v2 clean-window count, 0 zero-obs valid cycles, Σn_obs 149,122
      (~94 % of Σ scene-reads); bindings `99a5253f` / `30ecbe10` /
      `41009147` / epoch 2013-03-30; shapes (490,12,36,8,48,48); provenance
      complete, git_sha `863758f` (docs-only commit during the run).
      Known notes, recorded not hidden: (a) `sensor_id` = 0 (unknown) for
      every slot in BOTH stores — scene-dir naming `landsat_YYYYMMDD_PPP_RRR`
      carries no LC08/LC09 so `sensor_of` always returns unknown;
      normalization is unaffected (single `unknown` stats bucket, exactly as
      v1, so v1↔v2 stays apples-to-apples) but the `sensor_l8`/`sensor_l9`
      covariates are constant-0 in both eras — pre-existing limitation, fix
      would require scene renaming + re-extraction, not worth it for this
      store; (b) fields 1–4 were written by the pre-d8c705d reader (parity
      byte-proven on fields 1–2 by two independent gates; 3–4 same
      deterministic path + store-parity unit tests); (c) the printed
      `fields_written 345` counts only the final resume invocation —
      manifest holds all 490.

## 5. Colab bundle v2

- [x] 5.1 `thesis_data_full_v2.zip` DONE 2026-11-21: 293.5 MB / 12,903
      entries, ZIP CRC all-OK, 3 JSONs byte-identical to disk. Contents ship
      the REAL production path `data/patch_runs/v3/patches.zarr` (no rename —
      arcnames preserve it) + `splits_locked_v2.json` +
      `holdout_manifest_v2.json` + `norm_stats_full_v2.json`; cycle npz files
      deliberately excluded (data only; gates run with
      `verify_cycle_files=False`, same as v1). Size ≈ 4.3× v1's 67.8 MB =
      the 4.35× valid-cycle growth (3,478 vs 799). Local copy at
      `google_drive/thesis_data_full_v2.zip` (gitignored).
- [x] 5.2 Notebook v2 DONE: `notebooks/train_full_archive_colab_v2.ipynb`
      (8 cells, clone flow, pinned-commit note `67b8f82`). Gates cell asserts
      the full v2 binding chain (validate_store 490 fields, provenance↔splits
      manifest, cycle_source↔`cycles_parquet_sha256`, shapes
      (490,12,36,8,48,48), cycle_valid == 3478, norm self-hash +
      splits/holdout hash binding + n_train_cycles 2105) — dry-run of the
      cell PASSED against the real local artifacts (train 2105 / dev 340 /
      test 773 / monitoring 1240). Training cell passes `--splits` and
      `--holdout-manifest` v2 EXPLICITLY — `train_patch.py` defaults are the
      v1 paths, which on a v2 store would bind cycle selection to the wrong
      locked splits (fail-closed trap avoided at the notebook level).
      `--out checkpoints/patch/patch_full_v2_ep10`; same 6-trainings
      protocol, `--resume` preserved.
- [x] **5.3a DEFECT + QUARANTINE (2026-11-21, agent): the first full
      severity ×6 Colab run (patch_full_v2_sigma_eval_colab.zip, git
      73476c7) is scientifically INVALID — detection EXACTLY 44/340 in all
      24 cells × 6 runs, MDM80 censored, delay constant 60 d.** Reproduced
      locally and root-caused to three stacked code defects:
      (1) one_step_ahead_v3 leaves bin 0 unscored → p NaN at bin 0 →
      cumulative_channel_max cumsum NaN-poisoned every downstream bin →
      148/170 eval cycles (season starting at bin 0) lost their alarm path;
      (2) tau=0.05 hardcoded (eval_dev) — clean p uniform ⇒ alarm path ≈0.33
      ≫ 0.05 ⇒ alarm at the ~3rd observed bin of any live cycle, independent
      of severity; (3) fit_patch_sigma boolean indexing collapsed the
      (cycle, bin) axes so block[:,:,c] hit spatial ROW c — σ table 92 %
      NaN, its 35 "fitted" cells were row-7 medians of all channels mixed
      (scoring survived only via the per-channel fallback). The parity gate
      passed because chunked ≡ baseline, both equally poisoned. Fix
      (eval-side only, no retrain/store change): NaN-safe alarm mask in
      cumulative_channel_max (NaN = no data = no alarm), tau calibrated on
      leave-one-out calibration null paths at FAR 0.05
      (loo_empirical_p + calibrate_threshold; per_seed now records
      far_achieved + tau_calibration), integer indexing in fit_patch_sigma,
      gate reworked to current-code full-vs-chunked parity + liveness asserts
      (tau>0.05, far≤target, max S6 detection ≥0.8). 94 patch/eval/diagnostic
      tests pass. **Finding 24 (v1) headline is the SAME uncalibrated-tau
      artifact** (detection 1.000 at S1 AND S6, delay exactly 30 d = 3rd
      observed bin) — v1 escaped the poisoning only because its checkpoints
      were input-inclusive; finding 24 must be amended before thesis use and
      the v1↔v2 comparison must be rebuilt from a fixed-protocol re-run of
      both. v1 σ tables are lost (§6.2 counter-example) so a v1 re-run needs
      σ re-calibration on the v1 store first.
- [ ] 5.3 Findings/dev_log update: v1↔v2 comparison deliverable recorded
      (advisor SAVI hypothesis quantified); validation-ladder rung stated.
      BLOCKED on the fixed-protocol re-run (σ ×6 re-calibration + eval ×6).

#### 4.1 extraction I/O acceptance (no store or resume format change)

- [x] Optional `--tile-cache-mib` caches native aligned windows across fields;
  zero (default) retains the original reader. Compare raw patch and validity
  bytes for three synthetic field windows, including both raster edges.
- [x] Owner gate PASSED 2026-11-20, parity confirmed TWICE independently
  (agent gate + owner gate, both `--fields 1,2 --tile-cache-mib 1024
  --field-batch-size 4 --workers 2` on identical v2 inputs, all dates,
  no `--smoke`): field 1 sha `8a5e1166…` (455 scene-reads) and field 2 sha
  `13bdd036…` (326 scene-reads) byte-identical to the live old-reader run's
  manifest entries. Speed: synthetic 2.3×; agent gate field 1 = 37.3 s vs
  live 91.1 s (2.4×, measured UNDER disk contention from the concurrent
  live run); owner's concurrent gate field 1 = 140.5 s is contention-
  contaminated (both readers competed) — a clean number comes from the
  restarted full run's `--profile-io`. Switch approved; profile shows
  aggregation/write = 45–62 % of batch wall and tile hit-rate only 11 %
  (781/7,029) → further I/O+CPU headroom tracked in the follow-up
  acceleration research (owner-requested prompt 2026-11-20).
- [x] Optional `--field-batch-size 4` stages consecutive pending fields by
  scene (one open per needed band per scene); default 1. `--profile-io` prints
  batch wall, scene reads, remaining time and tile hits/misses/evictions to
  stdout only. Staged float32 patches and validity masks are additional to
  the tile budget (~74 KiB per field-scene at size 48). Compare batch wall
  time and hashes on the same two full fields before any production switch.
- [x] Flush four whole cycle chunks (including empty bins); preserve SHA
  readback. Optional `--workers` partitions scene dates among private threaded
  readers, splitting the tile budget and keeping a single ordered store writer.
  Synthetic 8-field/8-scene median (3 trials, batch 4/cache 32 MiB): 1.597 s
  old writer, 0.752/0.635/0.701 s new writer with 1/2/4 readers. Real-data
  SHA and wall-time gate above still applies.

#### 4.2 further reader acceleration — gated experiment, not approved for production

- [x] Instrument the §4.1 reader (`io_profile` stdout only): batch wall,
  scene phase, aggregation, four cycle writes + readback, LRU occupancy and
  evictions. The remainder includes geometry, metadata and field hashing.
  Parallel per-field read-seconds overlap; do **not** add them as wall time.
- [x] Opt-in `--short-bin-median`: for 1–2 acquisitions, use the same finite,
  QA and polygon tests and float64 pair mean, without a masked-array sort.
  Negative zero and larger bins retain the reference median. Opt-in
  `--native-block-cache`: cache the sparse union of touched GeoTIFF blocks,
  not a dense union bbox spanning unrelated fields. Private per-worker LRU
  budgets, original scene order, single writer, chunking and codec unchanged.
- [x] Three fresh stores per configuration, synthetic 8 fields × 8 dates,
  one acquisition per occupied bin. Local Linux, warm filesystem, no concurrent
  extraction. Old = §4.1 flags without either candidate flag (not pre-d8c705d).
  Median seconds; columns are independent medians. End-to-end times `main`,
  including provenance and validation but not imports, fixture or proof scan.

  | method | end-to-end | batch wall | scene_read | aggregation | write+readback |
  |---|---:|---:|---:|---:|---:|
  | §4.1 baseline | .6489 | .6036 | .1596 | .3417 | .0363 |
  | both flags | .3235 | .2757 | .1630 | .0151 | .0316 |
  | short bins only | .3361 | .2838 | .1666 | .0154 | .0309 |
  | native blocks only | .6865 | .6393 | .1657 | .3520 | .0377 |
  | both, 4 workers | .3517 | .2977 | .1875 | .0154 | .0320 |
  | both, GDAL cache 512 MiB | .3428 | .2942 | .1687 | .0154 | .0332 |
  | both, GDAL decode 2 | .3212 | .2751 | .1567 | .0155 | .0336 |

  Synthetic speed ratio 2.006×; most gain is **aggregation**, not block reads.
  Same raw-array digest in all 21 stores: `852bb963f2528183bd03c867dc30c2c97c45e842bc534a608705118396eb10de`.
  Baseline synthetic resident cache 12 MiB vs native 64 MiB, 0 evictions;
  wider reuse is not free. In the real gate 781 hits equals exactly one per
  scene-read (auto reads red twice); consistent with compulsory misses and no
  cross-field tile reuse, not evidence of budget exhaustion. At uint16, a single
  512² block per band × 8 bands × 607 scenes needs ~2.37 GiB. Examine actual
  hits/misses/evictions/bytes in each batch before increasing memory. Four
  readers increased scene wall here; this alone is not proof of a GIL bottleneck.
  No process pool. Codec probe on one cycle (encode+decode only): default
  full 36 bins Blosc LZ4/5 1.07 ms, LZ4/1 1.05 ms, Zstd/1 3.32 ms;
  two half chunks LZ4/5 1.14 ms. Codec and chunk changes are NOT enabled.
- [x] **Owner real gate RUN 2026-11-20 — FAILED (1.959× < 2×): do not switch.**
  Quiet Windows archive host, fields 1+2, all dates, patch 48, 1024 MiB /
  batch 4 / workers 2, 3 interleaved trials. Byte-identity PASS in all 6
  stores (raw-array fingerprint `19542d9e…` everywhere; anchors exact:
  field 1 = 455 reads / `8a5e1166…`, field 2 = 326 reads / `13bdd036…`).
  Median end-to-end baseline 26.474 s vs candidate 13.512 s = **1.959×**;
  per-trial ratios 2.19 / 1.85 / 1.98 straddle the bar — medians decide,
  fail-closed stands. Gain source: aggregation 13.996 → 3.545 s (−75 %,
  `--short-bin-median` working on real data); scene_read ~unchanged
  (9.382 → 8.608 s). `--native-block-cache` = **no-op on the real archive**:
  scene GeoTIFFs are strip-organized (block 2×496, float64, LZW) so the
  `min(bh, bw) < 16` guardrail in `TileCache._load_blocks` always takes the
  ordinary path — candidate profiles show `cache_bytes 0` and no hit/miss
  counters. The 2.006× synthetic claim came from a tiled fixture. Both flags
  stay OFF; production command unchanged; store, normalization and resume
  untouched.
- [ ] **Switch only after real gate passes and owner approves** — gate
  FAILED, so no switch; the resumed full run continues on §4.1 flags alone.
  Optional follow-up (not this run): re-gate with `--sweep` after the store
  completes if the aggregation share (62–85 % of live batch wall) is worth
  chasing; `--native-block-cache` can be dropped from any future candidate
  (guardrail makes it a structural no-op here). Same original
  full-run command (no `--fields`, production `--out`, all original inputs,
  `--resume --tile-cache-mib 1024 --field-batch-size 4 --workers 2 --profile-io`)
  plus `--short-bin-median --native-block-cache`. Otherwise resume §4.1 without
  these flags. No need to migrate, rebuild or change normalization.
