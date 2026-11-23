## 1. Fingerprints and cache layout

- [x] 1.1 Define a canonical fingerprint function for L1/L2/L3 inputs; include
  store/split/holdout/norm/model/sigma/severity hashes, cell id, basis, and a
  `cache_format` version integer.
  -> `ssl_anomaly.monitor.eval_cache`: `fingerprint`, `batch_fingerprint`,
  `l1_key`, `l2_key`, `l3_key`, `store_fingerprint`; format version 1. Array
  digests are NaN-payload-normalized for hashing only.
- [x] 1.2 Create the untracked `data/cache/eval/<layer>/<key>/` layout and
  verify Git ignores it. -> default root resolved from repo root; `/data/` is
  ignored by `.gitignore` line 80. No cache file is tracked.
- [x] 1.3 Decide the L3 size cap and the float32 `.npz` vs zarr threshold from
  measured cell sizes. -> cap 512 MiB (`EVAL_CACHE_MAX_CELL_MB`); plain `.npz`
  chosen. Measured 36 MiB/cell at n=32; zarr is unnecessary at these sizes and
  is deferred if production cells exceed the cap.

## 2. Cache implementation (opt-in)

- [x] 2.1 Implement L1 normalized-batch read/write with atomic temp-file rename.
  -> `EvalArtifactCache.load_l1/save_l1`, wired into `patch_batch_for_cycles`.
- [x] 2.2 Implement L2 clean reference-path read/write.
  -> `load_l2/save_l2`, wired into `patch_reference_paths`.
- [x] 2.3 Factor the post-injection scoring half of `patch_arm_scores` into a
  shared helper so the cached and cold paths use the same code; then implement
  L3 injected-tensor + hash-record read/write.
  -> Injection and scoring are separated inside `patch_arm_scores`; BOTH paths
  call the same `arm.score_cycles` loop, so no scoring logic is duplicated.
  `load_l3/save_l3` store `Xinj` (`.npz`) + hash records (`.json`).
- [x] 2.4 Add explicit read and write flags; default runs must ignore the cache.
  -> `EVAL_CACHE_READ` / `EVAL_CACHE_WRITE` env flags and a per-call `cache=`
  argument (`None` follows env, `False` forces cold). Read and write are
  independent; no flag means no file access.
- [x] 2.5 Implement fail-open behavior for miss, corrupt entry, hash mismatch,
  and shape/dtype mismatch, with diagnostics.
  -> `load`/`save` swallow only cache exceptions, increment `stats`, and append
  to `events`; a bad entry is recomputed and rewritten atomically.

## 3. Parity and evidence

- [x] 3.1 Compare cached vs cold `z2`, `paths`, `p_value`, `observed`, and
  injection records for every severity cell (S1–S6, both families, ndvi/savi,
  both rounds). Use NaN-aware array equality (`equal_nan=True`); plain
  `np.array_equal` gives a false mismatch on NaN entries. Any real mismatch is
  a defect.
  -> `tests/unit/test_eval_cache.py`: all 72 cell combinations pass at
  synthetic scale (n=2) plus a 3-cell real-torch run at n=32. Exact NaN-aware
  equality. Parity at production n on the archive machine remains a handover
  gate before research use.
- [x] 3.2 Run the targeted eval unit tests with cache off and on; record counts
  and any pre-existing environment gaps verbatim.
  -> 85 passed in `tests/unit/test_eval_cache.py`; 72 passed across the
  combined patch_eval + eval_cache + geometry + tile_cache selection.
  Pre-existing gap: `omegaconf` absent in env `geospatial` prevents collecting
  `tests/test_rq1_to_rq8_framework.py`.
- [x] 3.3 Measure cold, cache-build, and warm timings plus on-disk size; add the
  numbers to `reports/fast_anything_benchmark.md` or a dedicated report.
  -> n=32, 3 cells: cold 5.16 s, build 5.98 s, warm 3.53 s = **1.46×**;
  108 MiB / 3 cells; zero cache errors. Report §8.
- [x] 3.4 Re-run parity after any change to injection, normalization, scoring,
  model loading, or fingerprint inputs.
  -> Parity re-run after the `patch_eval.py` refactor; all tests above were run
  on the final code.

## 4. Deferred

- [ ] 4.1 Reduce the 5.46 M per-pixel `ndvi_from_red_nir` calls at the call site
  (vectorize/inline) before considering any Rust port of `inject_patch_drop`.
- [ ] 4.2 Do not apply this cache to training; that path is torch-bound.
