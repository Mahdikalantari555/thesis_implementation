## 1. Opt-in contract

- [x] 1.1 Document the exact commands from `src/`: `fast-anything scan`,
  `port`, `verify`, `build`, `enable <id>`, `disable <id>`, and
  `FAST_ANYTHING_DISABLE=1`. State that a clean clone runs pure Python.
  -> `docs/optional_rust_acceleration.md`.
- [x] 1.2 Add a smoke check that the default path is Python and that an explicit
  opt-in is required for acceleration; verify the opt-out works in the same run.
  -> `tests/unit/test_optional_rust_acceleration.py` (4 tests; subprocess
  checks for both the enabled and `FAST_ANYTHING_DISABLE=1` bindings).
- [x] 1.3 Confirm the two appended swap blocks are reversible and contain no
  edits to the original function bodies; verify after `fast-anything disable`
  that the module matches the pre-enable source.
  -> Static source assertion in the same test: each swap is the last block in
  the file, closed by the tool marker, with the original `def` intact above it.

## 2. Parity gate

- [x] 2.1 Run `fast-anything verify` for each of the three ports and record the
  replay counts and relative tolerance in the evidence report.
  -> ewma 500/500, cusum 500/500, aggregate_bin 204/204; rel-tol 1e-9.
- [x] 2.2 Run the targeted pytest selection with acceleration on and off:
  `pytest tests/unit/test_patch_geometry.py tests/unit/test_patch_tile_cache.py -q`.
  Record the count and any pre-existing environment gaps verbatim.
  -> 72 passed across patch_eval + eval_cache + geometry + tile_cache;
  `test_optional_rust_acceleration.py` adds 4 passed (enabled vs disabled
  digest equality). Pre-existing gap: `omegaconf` absent in env `geospatial`,
  so `tests/test_rq1_to_rq8_framework.py` cannot be collected there.
- [x] 2.3 Add a rule that any edit to a Python function or its Rust crate
  invalidates that port's verification until 2.1 and 2.2 are repeated.
  -> "Promotion gate" section of `docs/optional_rust_acceleration.md`.

## 3. Evidence

- [x] 3.1 Keep the benchmark driver `src/_fa_bench_driver.py` and the report
  `reports/fast_anything_benchmark.md` current; both exist as prototype output
  and this change adopts them as the evidence artifacts.
- [x] 3.2 Record per-workflow A/B results with loop count, environment, and
  machine; keep the honest 1.0× results for eval and training in the report.
  -> Refreshed A/B in report §8 (loops=3, medians of 3).
- [x] 3.3 Confirm binaries, recordings, and profiles remain untracked and that
  Git changes stay text-only.
  -> `src/.fast/.gitignore` plus repository rules; no binary is staged.

## 4. Explicitly deferred (not gating this change)

- [ ] 4.1 Port `flag_runs` and `contiguous_runs` only after their own parity
  gate; expected gain ~13% each of the scoring workload.
- [ ] 4.2 For eval, vectorize or inline the per-pixel `ndvi_from_red_nir` and
  `savi_from_red_nir` calls before considering `inject_patch_drop`.
- [ ] 4.3 Document that training acceleration belongs to `torch.compile` / GPU /
  Colab, not to this Rust path. (Documented as a non-goal; no port work done.)
