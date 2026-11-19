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

- [ ] 2.1 `segment_and_export_cycles_490.py`: `--detector {v1,fused}` +
      `--cycles-out/--npz-out` (defaults = v1 paths, unchanged; fused mode
      REFUSES v1 paths — preservation guard); fused mode emits the v1↔v2
      comparison block. *(code landed 2026-11-19 — smoke run pending)*
- [ ] 2.2 Provenance in the v2 export: git sha, FusionParams hash, wall time,
      v1 reference + sidecar `cycles_490_v2.provenance.json`. *(code landed —
      smoke run pending)*
- [ ] 2.3 `build_locked_splits.py`: `--cycles`/`--cycles-npz`/`--out` args
      (defaults unchanged); `build_splits_locked` needs a `cycles_npz_path`
      param; v2 manifest carries `derived_from` + v2 cycle-file hashes.

## 3. Run + gates (agent, local, read-only inputs)

- [ ] 3.1 Run fused export → `cycles_490_v2.parquet` + `growth_cycles_490_v2.npz`.
- [ ] 3.2 D6 gates on the v2 table (length distribution, season clustering,
      per-field counts, v1⊆v2) — report emitted under
      `reports/cycle_detection_diagnosis/v2_table_gates.md`.
- [ ] 3.3 Build `splits_locked_v2.json` (same k-means seed/blocks; D2
      exclusions by key) + `holdout_manifest_v2.json`
      (`--archive-end 2026-09-10`, fail-closed audit).

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
