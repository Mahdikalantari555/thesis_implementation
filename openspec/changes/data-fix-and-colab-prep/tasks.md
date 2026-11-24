# Tasks — data-fix-and-colab-prep

## 0. Inventory & verification (agent, local)

- [x] 0.1 Inventory canonical `Data/Processed/` and `thesis_implementation/data/processed/`
      (recorded in proposal Why; 538 parquet scenes, zonal_stats 538, cycles/
      splits/holdout/norm v2 only under implementation + bundle).
- [x] 0.2 Confirm `patches.zarr` v3 exists at `thesis_implementation/data/patch_runs/v3/patches.zarr`
      and inside `google_drive/thesis_data_full_v2.zip`.
- [x] 0.3 Confirm absentees: `weather_daily.parquet`, `stl_features.parquet`,
      `patch_sigma.npz`, any `ground_labels*` (verified by find across both roots).
- [ ] 0.4 Record bundle sha256 manifests for `thesis_data_full_v2.zip` and
      `patch_full_v2_eval_bundle*.zip` in `reports/` (command:
      `sha256sum google_drive/*.zip`); compare against owner's recorded values.

## 1. Data fix — weather (owner-gated)

- [ ] 1.1 Owner uploads `google_drive/weather_daily_v2.zip` containing
      `weather_daily.parquet` (daily ERA5 covariates + 3/7/14/30 d windows).
- [ ] 1.2 Extract to `thesis_implementation/data/processed/weather_daily.parquet`
      and mirror to `Data/Processed/weather/weather_daily.parquet`.
- [ ] 1.3 Append an `amendments:` entry to `benchmark_config.yaml` binding
      `artifacts.weather_daily_v2` (date + target + reason + approved_by);
      verify `load_benchmark_config` still loads and the matrix generator runs.
- [ ] 1.4 Re-run the `weather_inputs` cells (not the full matrix) in smoke mode
      on 1 seed; record wall time and confirm finite metrics.
- [ ] **Deferred (reason: owner upload missing)** — ground/confirmed-anomaly
      labels (`ground_labels_v2.zip`). No label parquet exists anywhere; until
      uploaded, all L4 claims stay unsupported and the environmental-correlation
      tables print `absent`/`shared-input` only. Owner action: task 4.

## 2. Data fix — STL features (optional)

- [ ] 2.1 Owner uploads `google_drive/stl_features_v2.zip` (optional, B5-class)
      OR explicitly declines.
- [ ] 2.2 If uploaded: extract + mirror to `Data/Processed/decomposition/stl_features.parquet`;
      bind via amendment. If declined: record `Deferred — owner declined optional
      artifact; benchmark treats as absent` and leave `required_for: optional`.
- [ ] **Deferred (reason: optional, unproduced by default)** — `stl_features`
      is not required for any gate; capstone reports it as `absent`.

## 3. Data fix — patch store sidecar

- [ ] 3.1 Verify a script exists to compute `patch_sigma.npz` from
      `patches.zarr` (search `scripts/` for `patch_sigma` / `build_patch_norm`).
- [ ] 3.2 If present: run in smoke mode on 3 fields; full run on Colab T4 if
      > 15 min; mirror sidecar to `Data/Processed/patch_runs/v3/patch_sigma.npz`.
- [ ] 3.3 If absent: open a follow-up change (do not invent the formula here);
      keep `hierarchical_patch` block labelled `BLOCKED` with reason
      "v2 patch store + norm sidecar unbound".
- [ ] 3.4 Update `benchmark_config.yaml` `patch_store_v2` record with the
      sidecar sha256 prefix once produced (amendment).

## 4. Canonical mirror of v2 artifacts (owner executes; agent verifies)

- [ ] 4.1 Mirror (copy, not move) `cycles_490_v2.parquet`, `splits_locked_v2.json`,
      `holdout_manifest_v2.json`, `norm_stats_full_v2.json` from
      `thesis_implementation/data/processed/` to `Data/Processed/`.
- [ ] 4.2 Mirror `patches.zarr` (v3) from `thesis_implementation/data/patch_runs/v3/`
      to `Data/Processed/patch_runs/v3/` (293 MB bundle; prefer extracting from
      the existing zip rather than re-copying if disk is tight).
- [ ] 4.3 Verify sha256 of every mirrored file matches the source; record in
      `reports/`. Do not alter frozen files.

## 5. Colab prep

- [ ] 5.1 Validate bundle contents (`unzip -l` or Python zipfile) and record
      extraction order in `docs/` or the Colab notebook cell 0.
- [ ] 5.2 Confirm `colab/` (currently empty locally) receives the notebook
      variant that reads from the mirrored canonical paths; no code change to
      the matrix generator.
- [ ] 5.3 Keep `hierarchical_patch` block `BLOCKED` until task 3.2/3.3 resolves.

## 6. WSL skill & tolerance alignment

- [ ] 6.1 Fast-forward WSL clone: `git pull --ff-only` (merge-base `becb6f7b`,
      15 commits behind; no unique WSL commits).
- [ ] 6.2 Build Rust port in WSL geospatial env:
      `PYO3_PYTHON=.../geospatial/bin/python cargo build` (~30 s), then
      `maturin develop`.
- [ ] 6.3 Run parity gates in WSL: `pytest tests/parity/test_backends_parity.py`
      (expect 2 passed) + `pytest tests/unit/test_backends.py` (expect 4 passed).
- [ ] 6.4 Update `rust/README.md` with the build command + Windows-no-toolchain
      limitation + the geospatial env path.
- [ ] 6.5 Align `openspec/specs/optional-rust-acceleration/spec.md` tolerance
      text from `1e-9` to `1e-6` (matching `tests/parity/test_backends_parity.py`);
      record the decision in the change (D6). Do not change the test file.
- [ ] 6.6 (Optional) Note in `AGENTS.md` env section that Rust builds run in WSL.

## 7. Verification gates (run, not assumed)

- [ ] 7.1 `pytest tests/unit/test_platform_data_contract.py tests/unit/test_platform_bilingual.py -q`
      (conda `geo`: `D:/Dev/Miniforge3/envs/geo/python.exe -X utf8`) — expect pass.
- [ ] 7.2 `platform`: `node node_modules/typescript/bin/tsc --noEmit` then
      `TMP/TEMP -> platform/.tmpbuild node node_modules/vite/bin/vite.js build`.
- [ ] 7.3 Runtime gate: start `scripts/serve_tiles.py --port 8765`, then
      `node scripts/capture_platform_screenshots.mjs --out reports/screenshots/audit --locale en`
      (+ `--no-tile-server` degradation run if degradation paths touched).
      Zero console/page errors.
- [ ] 7.4 If tile-server code touched: `python scripts/test_tile_concurrency.py`
      (12-thread burst).
- [ ] 7.5 Rust parity (WSL): as in 6.3.

## Deferred register (unmet, with reason)

- **D-1 ground_labels_v2.zip** — reason: owner upload missing (task 4 of todo);
  no ground/confirmed-anomaly labels exist anywhere. Impact: L4 claims
  unsupported; environmental tables print `absent` only.
- **D-2 weather_daily.parquet** — reason: owner upload missing
  (`weather_daily_v2.zip`); blocks `weather_inputs` cells until 1.1–1.3 done.
- **D-3 stl_features.parquet** — reason: optional, unproduced by default;
  blocks nothing; reported as `absent`.
- **D-4 patch_sigma.npz** — reason: generator availability unverified (task 3.1);
  blocks `hierarchical_patch` patch-level cells until 3.2 done.
- **D-5 canonical mirror of v2 artifacts** — reason: large copy (patches.zarr
  ~293 MB) is an owner/execution step; agent verifies only (task 4).
- **D-6 inspector hardening** — reason: out of scope; recorded as risk (D7).