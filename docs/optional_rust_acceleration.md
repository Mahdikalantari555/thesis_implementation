# Optional Rust acceleration (validation-ladder rung L0)

This repository is a research artifact. The Python implementations are
**canonical**: they define the semantics and every reported result. Three hot
pure functions have a verified Rust port that is **off by default** and changes
wall time only. It never changes a score, hash, label, count or reported number.

OpenSpec change: `openspec/changes/optional-rust-acceleration/`.
Evidence: `reports/fast_anything_benchmark.md`.

## The three ports

| Function | Module | Verified calls | Micro speedup |
|---|---|---:|---:|
| `aggregate_bin` | `ssl_anomaly.io.patches` | 204 / 204 | 5.39× |
| `ewma_detector` | `ssl_anomaly.baseline.score` | 500 / 500 | 7.91× |
| `cusum_detector` | `ssl_anomaly.baseline.score` | 500 / 500 | 5.86× |

The ports are **swap blocks appended to the same module**. The original Python
`def` stays in place, unchanged. Each block tries to import a compiled `.so`
from `src/.fast/lib/`; if it is missing or unimportable, the Python function is
used. Removing the appended block restores the exact pre-enable source.

## Default behavior

A clean clone has no `src/.fast/lib/*.so` (those binaries are untracked), so it
runs pure Python. No action is needed to stay on the canonical path.

Per-run opt-out works even when the binaries and swap blocks are present:

```bash
FAST_ANYTHING_DISABLE=1 <command>
```

The swap blocks check this variable at import time and leave the Python
function bound.

## Opt-in procedure

All commands run from the repository's `src/` directory with the conda env
`geospatial` active. Python path: `/home/asus/miniforge3/envs/geospatial/bin/python`.

1. **Profile** a real workflow (writes `src/.fast/profile.json`):

   ```bash
   fast-anything scan -- /home/asus/miniforge3/envs/geospatial/bin/python \
       _fa_bench_driver.py --part extract --loops 3
   ```

   Change `--part` to `detectors`, `eval`, `train`, etc. The driver must stay at
   `src/_fa_bench_driver.py`; if it lives outside the scanned directory, module
   ids gain a `src.` prefix and the recorder catches zero calls.

2. **Record and port** a chosen function:

   ```bash
   fast-anything port <function-id>
   ```

   Keyword arguments must be plain data (use the string `"float16"`, not the
   `np.float16` type) or the recorder skips the calls.

3. **Verify** parity before any enablement:

   ```bash
   fast-anything verify <function-id>
   ```

   This replays recorded, edge and derived calls. A port may only be enabled
   when every case passes at relative tolerance 1e-9. The targeted pytest
   selection must also pass both with acceleration on and with
   `FAST_ANYTHING_DISABLE=1`:

   ```bash
   pytest tests/unit/test_patch_geometry.py tests/unit/test_patch_tile_cache.py -q
   ```

4. **Build and enable** the verified port:

   ```bash
   fast-anything build <function-id>
   fast-anything enable <function-id>
   ```

5. **Measure** the end-to-end effect and record it in
   `reports/fast_anything_benchmark.md`:

   ```bash
   fast-anything bench --runs 3
   fast-anything report
   ```

## Disable / revert

```bash
fast-anything disable <function-id>     # revert one port's swap block
FAST_ANYTHING_DISABLE=1 <command>       # one-run opt-out without editing files
```

Both leave the original Python function in place. `disable` must restore the
module to its pre-enable source byte-for-byte.

## Promotion gate (do not skip)

A port may be enabled only when all of the following hold:

1. `fast-anything verify` passed **every** recorded, edge and derived case at
   relative tolerance 1e-9 (state the replay count).
2. The targeted pytest selection passed with acceleration **on and off**.
3. `fast-anything bench` showed a measured end-to-end gain on the committed
   driver; loop count, machine and environment are recorded in the report.
4. Any edit to the Python function or its Rust crate invalidates that port's
   verification until steps 1–3 are repeated.

Nested profiler shares overlap; only the end-to-end A/B numbers justify a port.
Where the benchmark shows 1.0× (currently eval and training), the report says
so and no acceleration is claimed.

## What is deliberately out of scope

- **Training**: time is in torch `forward` + autograd, which this tool cannot
  port. Training acceleration belongs to `torch.compile` / GPU / Colab.
- **Eval injection (`inject_patch_drop`)**: it calls several helpers and builds
  hash-bearing injection records; it needs its own parity suite. See
  `openspec/changes/eval-artifact-cache/` for the disk-cache approach instead.
- **`flag_runs` / `contiguous_runs`**: ~13% each of the scoring workload after
  the current ports; defer until their own parity gate is run.

## Git policy

Binary artifacts stay untracked: `src/.fast/lib/`, recordings, profiles and
target output are ignored by `src/.fast/.gitignore` and repository rules. Git
stays text-only — only the driver, this document and the evidence report are
tracked.
