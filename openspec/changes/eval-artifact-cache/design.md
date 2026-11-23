# Design

## Context

`patch_arm_scores` injects a severity cell into a patch tensor, normalizes the
injected values, runs the model to get per-pixel scores, and converts them to a
one-sided empirical p-value against clean calibration reference paths. All of
these inputs are frozen for a given experiment: the patch store, norm stats,
model checkpoint, sigma table, severity table, and calibration subset do not
change between routine dev reruns. Recomputing them is pure waste.

Rust porting addressed only 5–8% of this workflow. Caching targets the ~47% that
is deterministic preprocessing (injection 39.9% + reference paths 7.3%) plus
the repeated normalization.

## Goals / Non-Goals

- Make repeated eval runs faster without changing any result.
- Make the cache safe by construction: keyed on all inputs, versioned, and
  fail-open.
- Keep the cold path (no cache) as the default and the source of truth.

Non-goals:

- No caching of model weights or of the patch store itself; those already have
  their own artifacts.
- No change to injection math, p-value math, or severity protocol.
- No automatic cache invalidation based on file timestamps alone; hashes bind
  content.
- No claim that cached results are scientifically different from cold results.

## Decisions

1. **Three cache layers, promoted in order.** L1 normalized batch (`X`, `obs`,
   `in_field`, `sensor_id`, `covariates`); L2 clean reference paths `(n_clean,
   T)`; L3 per-cell injected tensor `Xinj` plus the injection hash records.
   L1 and L2 are small and shared by all cells; L3 is large and cell-specific.
2. **Content fingerprints, not paths.** Each layer's key is a SHA-256 over a
   canonical JSON of its input hashes plus a `cache_format` integer. For L3 the
   key also includes level id, family, index, round, onset rotation, and
   injection basis. A change to any input produces a new key.
3. **Storage format.** L1/L2 use `.npz` with plain dtypes; L3 stores `Xinj` as
   float32 `.npz` (or zarr if a cell exceeds a size cap) with compression off
   so bytes survive a round trip. Injection records go beside it as JSON.
   The cache lives in `data/cache/eval/<layer>/<key>/`.
4. **Fail open and self-healing.** Missing key, unreadable file, hash mismatch,
   or shape/dtype mismatch means: compute from the cold path, write atomically
   (temp file + rename), then use the freshly computed value. Never return a
   partially read artifact.
5. **Parity gate before enabling for research.** For every severity cell
   (S1–S6, acute/persistent/gradual, ndvi/savi, both rounds), a warm run must
   reproduce the cold run's `z2`, `paths`, `p_value`, `observed`, and injection
   records exactly. Any mismatch disables the cache for that key and is a
   defect, not a tolerance call.
6. **Opt-in.** Default runs ignore the cache. An explicit environment flag or
   CLI flag enables reads; a separate flag enables writes, so research runs can
   read an existing cache without silently growing it.
7. **Bounded disk use.** Per-cell `Xinj` is roughly
   `n_cycles * T * C * S * S * 4` bytes (≈75 MB for n=64, T=36, C=8, S=32).
   The writer enforces a configurable cap and refuses to cache cells above it;
   oversized cells fall back to the cold path with a diagnostic.
8. **Concurrency.** One process per cache key at write time. Atomic rename
   makes readers either see a complete entry or none; no lock file is required
   for the research workflow, but concurrent writers are acceptable because
   identical inputs produce identical bytes.

## Risks / Trade-offs

- **Stale cache.** Mitigated by full content fingerprints and a format version.
  Timestamps are not trusted.
- **Disk footprint.** L3 can reach hundreds of MB over a full sweep. Mitigated
  by the size cap and by keeping L3 opt-in; L1/L2 are cheap and cover shared work.
- **Byte parity.** float16/float32 casts and injection hashes are sensitive.
  Mitigated by the exact parity gate in decision 5.
- **Non-determinism.** `rng` for injection is seeded per round and phase rotates
  by sample index; a frozen seed set is part of the key. If the sweep changes
  its RNG protocol, keys change.
- **Duplicate logic.** A cached path that reimplements the scoring half of
  `patch_arm_scores` can drift. Prefer factoring the post-injection scoring into
  a shared helper rather than copying it; the parity gate still applies.
- **No model speedup.** The remaining ~50% of eval is torch scoring and is
  unchanged. Expected best case is roughly 1.7× on this workload, not more.

## Feasibility evidence (probe, 2026-11-23)

A parity-checked probe was run before this change was written. It reused the
benchmark driver's synthetic batch (n=32, T=36, C=8, S=32), reproduced
`patch_arm_scores`' injection code, saved `Xinj` + hash records to a temporary
`.npz`/JSON, then loaded them and called the same `arm.score_cycles` path.

| 3 severity cells | Time |
|---|---:|
| Cold (full `patch_arm_scores`) | 6.17 s |
| One-time cache build | 2.59 s |
| Warm (load + score, no injection) | 3.95 s |
| **Warm speedup** | **1.56×** |

- Cache size: **108 MiB for 3 cells** (≈36 MiB/cell at n=32). A full
  24-cell × 2-round sweep would reach several GB; the size cap in decision 7
  is therefore mandatory, and L3 may need to stay opt-in per cell.
- Parity: `z2`, `paths`, `p_value`, `observed` and injection records were
  identical for all three cells using NaN-aware equality (`equal_nan=True`).
  Plain `np.array_equal` reports False only because NaN != NaN; the comparator
  must be NaN-aware.
- The build cost amortizes after one extra eval run of the same cells.

## Evidence still needed before adoption

- Cold, build, and warm times for the **full** severity sweep at production n.
- Cache size at production n and the chosen per-cell cap.
- Exact NaN-aware parity across all severity cells, not just the three probed.
- Re-run after any change to injection, normalization, scoring, or fingerprints.
