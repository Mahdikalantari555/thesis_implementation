# Google Drive `thesis_colab` — intake & validation report

- **Date:** 2026-11-20
- **Source:** Drive folder `thesis_colab` (`1FApJHnsTfRDtdx1ouUsUlcxX1xR-hnZ6`)
- **Local copies:** `google_drive/thesis_data_full.zip`, `google_drive/patch_full_ep10_colab_results.zip`
- **Extracted to:** `data/processed/patches.zarr/`, `data/processed/splits_locked.json`, `data/processed/holdout_manifest.json`, `data/processed/norm_stats_full.json`, `data/processed/patch_full_ep10/`
- **Verdict:** ✅ **Usable.** Both archives are intact, internally self-consistent, and the full hash chain (store → splits → holdout → norm → checkpoints → run manifests) reproduces exactly. Three deviations from the locked contract are documented below; none blocks downstream work, but two should be recorded in the thesis limitations.

---

## 1. What was in the folder

| File | Size | Modified (UTC) | Contents |
|---|---|---|---|
| `thesis_data_full.zip` | 67,807,721 B (64.7 MiB) | 2026-11-19 17:55 | `patches.zarr/` (4,266 entries) + the three JSON sidecars |
| `patch_full_ep10_colab_results.zip` | 47,735,119 B (45.5 MiB) | 2026-11-19 19:01 | `patch_full_ep10/` — 6 runs × 15 files |

Nothing else in the folder. Both `unzip -t` runs: **“No errors detected in compressed data.”**

---

## 2. Patch store (`patches.zarr`)

Zarr v3, `cycle_epoch = 2013-03-30`, `patch_size = 48`, 370 fields × 8 cycle slots.

| Array | Shape | dtype |
|---|---|---|
| `values` | (370, 8, 36, 8, 48, 48) | float16 |
| `pixel_valid` | (370, 8, 36, 1, 48, 48) | uint8 |
| `in_field` | (370, 1, 48, 48) | uint8 |
| `n_obs`, `sensor_id` | (370, 8, 36) | uint8 |
| `field_ids` | (370,) | int64 |
| `cycle_ids`, `cycle_start_day`, `cycle_end_day`, `cycle_valid` | (370, 8) | int64 / int64 / int64 / uint8 |
| `bbox_origin` | (370, 2) | int16 |

Channels: `ndvi, blue, green, red, nir, swir1, swir2, lst`. Chunking: one field-cycle per chunk, `zstd`. ~70 MB on disk (~3.9 GB logical fp16).

### Checks that passed

- **`output_sha256` reproduces exactly.** Recomputed from the store with the repo's own algorithm (schema digest + rank-strided 512-value sample + finite count): `2d0aa3f9…248732e` == `provenance.json`. 37,231,336 finite values.
- **`splits_locked.json` self-hash:** `9aa5eb11…e8565` ✅ (matches `manifest_sha256`, and every downstream artifact cites it).
- **`holdout_manifest.json` self-hash:** `39284712…1db21` ✅
- **`norm_stats_full.json` self-hash:** `3cb1ca60…8d0d5` ✅
- All three sidecars' *content* hashes match the values embedded in `provenance.json` and in all 6 `run_manifest.json` files.
- **Repo gates on the live store:** `validate_store`, `validate_locked_patch_store`, `validate_patch_norm_binding`, `assert_patch_count_bounds` all pass; `select_patch_train_cycles` returns **447 cycles / 240 fields / 37 excluded** exactly as reported.
- **Admitted-cycle digest reproduces:** `ab814784…874ff` ✅ (rebuilt the `(key, start_day, end_day)` list independently).
- **Field membership:** 370 unique sorted IDs in [1, 490]; all 240 train, 38 dev, 128 test and 56 PoC fields present.
- **Cycle metadata:** 799 valid cycles (mean 2.16/field, 1–8, min 1); all 2,161 invalid slots have `cycle_id = 0`, zero dates, all-NaN values; dates span 2013-09-03 → 2026-03-20 with no `end <= start`.
- **Validity is coherent:** `pixel_valid` is exactly 1 where `values` is finite (1.000 / 0.000 split, no exceptions); the 11,396 valid-cycle bins with `n_obs == 0` have all-zero `pixel_valid`.
- **No empty fields:** `in_field` counts 25–364 pixels (mean 271.7, median 280, of 2,304).
- **`.manifest.jsonl`:** 370 lines, one per store field, matching `field_ids`.
- **Split coverage:** all 447 train / 69 dev / 171 test keys plus 112 monitoring-only keys present; **zero** keys outside the locked splits; 956 locked keys legitimately absent (those cycles aren't valid in the store).
- **Norms describe this data:** spot-check over the first 10 fields gives NDVI median 0.551 vs 0.531 and LST 309.25 vs 310.0, and `norm_stats_full.json`'s 447 train keys and 240 field IDs are set-equal to `splits_locked.json`.

---

## 3. Training runs (`patch_full_ep10`)

6 runs, `protocol = patch_train_v2`, `input_protocol = patch_inputs_v2`, 10 epochs each, CUDA, torch 2.14.1+cu130, Python 3.13.16.

| Run | Params | Wall | Final loss |
|---|---|---|---|
| `patch_cnn_gru_seed13` | 91,028 | 259.3 s | 0.029013 |
| `patch_cnn_gru_seed42` | 91,028 | 258–259 s | 0.029… |
| `patch_cnn_gru_seed2026` | 91,028 | 258–259 s | 0.029… |
| `patch_pixelwise_seed13` | 36,020 | 235–243 s | 0.0367… |
| `patch_pixelwise_seed42` | 36,020 | 235–243 s | 0.0367… |
| `patch_pixelwise_seed2026` | 36,020 | 235–243 s | 0.0367… |

### Checks that passed

- Every run dir has all 14 expected files; **all 60 epoch checkpoints load cleanly** in torch, with `checkpoint["epoch"]` matching the filename.
- `last.pt` vs `epoch009.pt`: different SHA-256 (expected — the RNG snapshot differs), but **model weights are bit-identical** (max abs diff 0.0 across all 21 tensors), same optimizer state shape, same `config_hash`, same `epoch = 9`. So `last.pt` is safe as the final model.
- No NaN/Inf in any final model tensor.
- All 6 runs share identical `splits_sha256`, `holdout_sha256`, `norm_sha256`, `n_field_cycles = 447`, `n_trainable_fields = 240`, `n_excluded_by_holdout = 37`.
- Per-run `norm_stats.json` hashes equal `norm_stats_full.json`; `unnormalized_channels` is empty (all 8 channels had a usable scale).
- Loss curves are monotone-decreasing across 10 epochs; cosine LR schedule lands on `5e-5` as configured.

---

## 4. Three deviations to record

### 4.1 `sensor_id` is `unknown` (0) everywhere — ⚠️ most consequential

All 106,560 bins are `sensor_id = 0`, including the 17,368 bins that actually have observations. `sensor_of()` in `scripts/extract_patches.py:207` only maps to L8/L9 when the scene *path* contains `LC08` / `_L8` / `LC09` / `_L9`; the Colab scene names evidently don't.

Consequences:
- The `sensor_l8` and `sensor_l9` covariates are identically zero — the model has no sensor signal at all (6 of 8 covariates still carry information, so training is valid, just sensor-blind).
- `norm_stats_full.json` contains a single `unknown` bucket, i.e. L8 and L9 are pooled into one median/IQR rather than normalized separately. Since L8/L9 radiometry is nearly identical in Collection 2 L2 this is a small bias, not a correctness failure.
- **Fix if you want sensor covariates:** make `sensor_of` fall back to reading the `SPACECRAFT_ID`/`LANDSAT_PRODUCT_ID` metadata, or drop the two one-hots and state that sensor is not modelled.

### 4.2 Patch size is 48, not the locked default 32

`src/ssl_anomaly/io/patches.py:40` sets `PATCH_SIZE = 32` (“locked default; `--patch-size` exists but is not swept”), and `docs/patch_inputs.md` describes the frozen contract. The store and every checkpoint use **48**. `validate_store` only enforces squareness, so extraction and training both succeeded — the deviation is silent rather than fatal.

Consequences: ~2.25× the pixels per patch (2,304 vs 1,024), which shows in the 4.3 GB logical footprint and the 4.3 GB CUDA peak. Results remain valid for *this* configuration, but 48 ≠ the preregistered 32. Either re-extract at 32 before the locked run, or amend the contract and note the change.

### 4.3 `n_excluded_by_holdout = 37` is correct — resolved

This looked like a contradiction (0 train cycles sit on holdout farms) but is by design. `select_patch_train_cycles` (`patches.py:528`) computes `keep = train_only_mask(wanted, ...)` over the **whole store field list** (370 fields), not over the locked train fields, and returns those 37 non-admitted fields as `excluded`. I verified: `set(excluded) == holdout_farms ∩ store_fields` → **37**, exact match. Train/dev ∩ holdout is empty; only test-block fields overlap by construction. **No defect.**

---

## 5. Provenance caveats (not data defects)

- `provenance.json` cites git `84184fe5…` and the run manifests cite `36b61613…`. Neither is in this checkout (HEAD `46e7d60`), so the code used on Colab cannot be byte-compared here. Worth pushing that commit or attaching a patch hash.
- `provenance.json` has `holdout_manifest_sha256: null` — the store predates the frozen holdout. Harmless (the holdout is enforced at training time, and it was: all manifests carry `39284712…`), but the store itself isn't holdout-bound.
- `D:/RS/...` Windows paths appear in `provenance.json`. Cosmetic portability wart.
- `tests/unit/test_extract_patches.py` has 15 failures here purely from a missing `rasterio` in this sandbox (the other 3 need `lightning`); the 17 tests that don't need those libs pass. Not an artifact problem.

---

## 6. Housekeeping

- `data/` is gitignored/DVC-tracked, so the extracted 121 MB will **not** be committed; only this report is tracked.
- `google_drive/` (111 MB of zips) is currently untracked and will show up in `git status`. Recommend deleting it or adding it to `.gitignore` — the contents are fully reconstructed under `data/processed/`.
- Recommended next step: decide on 4.1 and 4.2 before the locked held-out run, since both change the input contract and would invalidate these six checkpoints if revised.
