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

- [ ] 4.1 PowerShell: extraction
      (`extract_patches.py --detector-store v2 --patch-size 48 --splits
      splits_locked_v2.json --cycles cycles_490_v2.parquet --out
      data/patch_runs/v3/patches.zarr`, resume-aware) + `build_patch_norm.py`
      → `norm_stats_full_v2.json`; progress + final summary required.
- [ ] 4.2 After owner reports: agent verifies store (provenance binding to
      splits v2, shapes, valid-cycle count) and norm file.

## 5. Colab bundle v2

- [ ] 5.1 `thesis_data_full_v2.zip` (`patches_v2.zarr` + v2 JSONs, data only).
- [ ] 5.2 Notebook v2 (clone flow, v2 paths, gates cell asserts v2 provenance
      binding); same 6-trainings protocol on the v2 store.
- [ ] 5.3 Findings/dev_log update: v1↔v2 comparison deliverable recorded
      (advisor SAVI hypothesis quantified); validation-ladder rung stated.

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
