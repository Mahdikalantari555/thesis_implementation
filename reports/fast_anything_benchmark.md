# fast-anything benchmark: where this project can be accelerated

Date: 2026-11-23
Tool: `fast-anything` v0.8.1 (MCP + CLI), Python 3.12 conda env `geospatial`
(numpy 2.5.2, scipy 1.18.0, torch 2.14.0+cu130), single-thread BLAS.
Workload: `src/_fa_bench_driver.py` — synthetic, deterministic,
production-like shapes. No real archive data; timings are **relative, not
production extrapolations**. All numbers below are medians from real profiler
and A/B runs, not estimates.

## 1. Headline result

| Workflow (synthetic workload) | Python | Rust ports ON | End-to-end speedup | Hot function speedup |
|---|---:|---:|---:|---:|
| Patch extraction (per-bin median reduce) | 9.56 s | 4.66 s | **2.1×** | `aggregate_bin` 5.39× |
| Field-level scoring / detectors | 11.23 s | 5.18 s | **2.2×** | `ewma_detector` 7.91×, `cusum_detector` 5.86× |
| Patch eval (severity arm sweep) | 20.42 s | 20.29 s | 1.0× | no applicable port yet |
| One training step (patch CNN-GRU) | 8.36 s | 8.45 s | 1.0× | torch-bound; no portable function |

For extraction and detectors the *timed workload itself* (excluding fixture
setup) dropped from ~6.07 s → ~1.27 s and ~7.80 s → ~1.91 s respectively, i.e.
~4.8× and ~4.1×. The end-to-end number is lower because fixture creation is
fixed cost.

## 2. Profile: where the time goes

Shares are from `fast-anything scan`. Nested functions are reported separately,
so shares within one scan do not sum to 100%.

### Patch extraction (11.7 s profiled; 1,200 aggregate calls)
- `ssl_anomaly.io.patches.aggregate_bin` — **59.5%**, 1,200 calls → ported.
- `scripts.extract_patches.short_bin` — ~1.7% (fast 1–2 acquisition path).
- Remainder: driver/fixture setup.

### Eval: patch severity arm sweep (28.4 s profiled)
- `monitor.patch_eval.patch_arm_scores` — **78.1%** (outer frame, includes children).
- `monitor.patch_perturbations.inject_patch_drop` — **39.9%**, 384 calls → **best next port target**.
- `monitor.patch_scores.pixel_scores` — 4.2%; `patch_reference_paths` — 4.3%.
- `monitor.perturbations.ndvi_from_red_nir` — 2.3% but **5,461,710 calls**;
  `savi_from_red_nir` 1.5% / 3,641,140 calls; `_solve_ndvi_drop` 1.5% / 1,820,579 calls.
  These are per-pixel Python calls — the call count, not the arithmetic, is the cost.
- `io.patches.normalize_patch` — 3.0%, 448 calls.

### Training: one forward/backward step (9.9 s profiled)
- `models.patch_cnn_gru.PatchCausalGRU.forward` — 17.4% (torch method; tool blocker: method).
- `PatchStem.forward` — 6.7% (torch method).
- `io.patches.normalize_patch` — 1.8%.
- Tool verdict: *"Nothing portable takes a real share of the run yet."*

### Field-level scoring support (19.0 s profiled, before ports)
- `baseline.score.ewma_detector` — 39.0%, 80,000 calls → ported.
- `baseline.score.cusum_detector` — 26.4%, 80,000 calls → ported.
- `baseline.score.flag_runs` — 6.2% (13.5% after the two ports removed);
  `utils.arrays.contiguous_runs` — 6.0% (13.1% after) → **easy next ports**.
- Earlier whole-pipeline scan: `robust_baseline` 15.2% and `whittaker_batch` 9.7%
  — both are *methods*, so fast-anything cannot port them. The existing PyO3
  `rust/` backend already implements both; use `get_backend("rust")`.

## 3. What was actually ported and verified

| Function | Recorded calls replayed | Result | Micro speedup |
|---|---:|---|---:|
| `baseline.score.ewma_detector` | 500/500 (incl. 300 derived + 100 edge) | pass | 7.91× |
| `baseline.score.cusum_detector` | 500/500 | pass | 5.86× |
| `io.patches.aggregate_bin` | 204/204 | pass | 5.39× |

Parity: fast-anything compares floats at rel-tol 1e-9. Unit tests
`tests/unit/test_patch_geometry.py` + `tests/unit/test_patch_tile_cache.py`:
**25 passed**. One early edge case (uint8 `in_field` mask) was caught by those
tests and fixed; re-verified and re-tested.

## 4. Honest limits

- **Training cannot be sped up this way.** Time sits in torch `forward` methods
  and autograd; fast-anything ports module-level pure functions. Use
  `torch.compile`, mixed precision, GPU/Colab, or kernel fusion for training.
- **Eval is mostly injection + model scoring.** `inject_patch_drop` (~39%) is
  portable in principle but calls `_profile`, `_solve_index_drop_on_bands`,
  `footprint_mask` and builds hash-bearing `PatchInjection` records — a
  careful, separately verified port, not a quick win. The 5.4 M
  `ndvi_from_red_nir` calls should first be reduced at the call site
  (vectorize / inline), which is a Python-side fix, not a Rust port.
- **No scientific claim.** These are engineering speedups on synthetic data.
  Validation-ladder rung: L0 (infrastructure). Detection quality, thresholds
  and findings are unchanged; parity is numerical, not scientific.
- Profiler shares overlap for nested calls. End-to-end A/B numbers are the
  decision-grade figures.

## 5. Recommended next actions (ranked)

1. Port `baseline.score.flag_runs` and `utils.arrays.contiguous_runs`
   (~13% each of the scoring workflow after current ports). Small, pure,
   same pattern as the two detectors — expected ~2–4× on those functions.
2. For eval, vectorize/eliminate the per-pixel `ndvi_from_red_nir` /
   `savi_from_red_nir` calls inside `_solve_index_drop_on_bands` before
   attempting a Rust port of `inject_patch_drop`.
3. If `inject_patch_drop` is ported, replay every severity cell ($S1..S6$,
   acute/persistent/gradual, ndvi/savi, all rounds) against the frozen
   fingerprint suite; a hash mismatch is a defect.
4. Leave training to torch-level optimization; do not spend port effort there.
5. For the already-ported functions, note the swap is appended as a reversible
   block in the Python module — see §6.

## 6. Reproduce / undo

```bash
cd /home/asus/Projects/thesis/thesis_implementation/src
# profile one part
fast-anything scan -- /home/asus/miniforge3/envs/geospatial/bin/python \
  _fa_bench_driver.py --part extract --loops 3
# A/B the current ports
fast-anything bench --runs 3
# write .fast/report.md + .fast/report.svg
fast-anything report
```

- Benchmark driver: `src/_fa_bench_driver.py` (parts: `extract`,
  `eval`, `train`, `detectors`, `affiliation`, `smoothing`, `baseline`, `indices`).
- Tool artifacts: `src/.fast/` (ports, recordings, built `.abi3.so` in
  `src/.fast/lib/`, `report.md`, `report.svg`). Binaries are untracked.
- Source change: two reversible blocks appended in
  `src/ssl_anomaly/baseline/score.py` and `src/ssl_anomaly/io/patches.py`.
- Undo one port: `fast-anything disable <id>` (run from `src/`).
- Undo all for one run: `FAST_ANYTHING_DISABLE=1 <normal command>`.
- Fresh clone needs `fast-anything build` in `src/` to rebuild the `.so` files
  before the swap blocks take effect (they fail open to the Python version).
- Environment gap (pre-existing, not caused by ports): `omegaconf` is absent
  in env `geospatial`, so `tests/test_rq1_to_rq8_framework.py` cannot be
  collected there.

## 7. Eval caching follow-up (scope recorded)

Eval did not benefit from Rust ports (1.0×). An owner-approved follow-up is
specified in `openspec/changes/eval-artifact-cache/`: cache the deterministic
intermediates (normalized batch, clean reference paths, per-cell injected
tensors) under an untracked `data/cache/eval/` keyed by content fingerprints.

A parity-checked probe (3 severity cells, n=32, T=36, C=8, S=32) measured:

| 3 cells | Time |
|---|---:|
| Cold `patch_arm_scores` | 6.17 s |
| One-time cache build | 2.59 s |
| Warm (load + score) | 3.95 s |
| **Warm speedup** | **1.56×** |

Cache size was 108 MiB for 3 cells (~36 MiB/cell), so a full sweep needs a size
cap. `z2`, `paths`, `p_value`, `observed` and injection records matched exactly
using NaN-aware equality. That was a feasibility measurement; the implemented
cache is measured in §8.

## 8. Implemented eval cache (change applied, 2026-11-23)

`openspec/changes/eval-artifact-cache/` is applied. Three opt-in layers are
implemented in `src/ssl_anomaly/monitor/eval_cache.py` and wired into
`monitor/patch_eval.py` behind a `cache=` argument and the independent
`EVAL_CACHE_READ` / `EVAL_CACHE_WRITE` env flags. No flag means no file is
read or written. Entries are content-fingerprinted, atomically written,
fail-open, and capped (512 MiB per cell by default). All artifacts live under
the untracked `data/cache/eval/`.

Same workload as §7 (3 severity cells, n=32, T=36, C=8, S=32, real torch
scorer, single-thread BLAS), measured through the **implemented** code:

| 3 cells | Time |
|---|---:|
| Cold `patch_arm_scores` (`cache=False`) | 5.16 s |
| One-time cache build (first enabled run) | 5.98 s |
| Warm (second enabled run, cache hit) | 3.53 s |
| **Warm speedup** | **1.46×** |

- Cache size: 108 MiB for 3 cells (~36 MiB/cell), unchanged from the probe.
- Cache stats over the run: `hit=3, miss=3, write=3, skip_size=0,
  invalidated=0, error=0`.
- Parity: `z2`, `paths`, `p_value`, `observed` and all injection records were
  identical with NaN-aware equality for the 3 real-torch cells and for all
  72 S1–S6 × family × index × round cells at synthetic scale
  (`tests/unit/test_eval_cache.py`, 85 passed).
- The build pass costs slightly more than a cold run because it also computes
  content fingerprints. The cost amortizes after one additional eval run of
  the same cells. Warm still pays key hashing plus ~50% model scoring, so a
  larger gain from caching alone is not realistic.

## 9. Refreshed A/B after applying both changes

Direct driver timed region (fixture setup excluded), `--loops 3`, median of 3
runs, same env as above. `FAST_ANYTHING_DISABLE=1` selects Python; no flag
selects the enabled Rust ports.

| Part | Python (median) | Rust ON (median) | Speedup |
|---|---:|---:|---:|
| `extract` | 8.98 s | 2.34 s | **3.8×** |
| `detectors` | 2.10 s | 0.53 s | **3.9×** |

The §1 "end-to-end" figures (2.1× / 2.2×) include fixed fixture creation; the
driver's own timed region is the larger ratio. Both are real measurements of
different scopes — neither is a production extrapolation. Eval and training
remain 1.0× under Rust acceleration.

Verification re-run on the final code:

- `fast-anything verify`: ewma 500/500, cusum 500/500, aggregate_bin 204/204
  at relative tolerance 1e-9.
- `pytest tests/unit/test_patch_eval.py tests/unit/test_eval_cache.py
  tests/unit/test_patch_geometry.py tests/unit/test_patch_tile_cache.py -q`:
  72 passed. `tests/unit/test_optional_rust_acceleration.py`: 4 passed.
- `openspec validate optional-rust-acceleration` and
  `openspec validate eval-artifact-cache`: both valid.

## 10. How to use and undo

- Acceleration opt-in / opt-out and the promotion gate:
  `docs/optional_rust_acceleration.md`.
- Eval cache switches (from the repository root):

  ```bash
  EVAL_CACHE_READ=1 EVAL_CACHE_WRITE=1 \
    /home/asus/miniforge3/envs/geospatial/bin/python scripts/eval_dev.py ...
  ```

  Use `EVAL_CACHE_READ=1` alone for a read-only research run that must not grow
  the cache. `EVAL_CACHE_DIR` and `EVAL_CACHE_MAX_CELL_MB` override the root and
  the per-cell cap. Research use at production n still requires the archive
  machine parity gate recorded in the change's tasks.
