# Proposal: Optional cached eval artifacts

## Why

The eval severity sweep is rerun often during development, yet it recomputes
identical intermediate tensors every time. The 2026-11-23 profile
(`reports/fast_anything_benchmark.md`) shows, for the patch arm:

- `patch_arm_scores` = 78.1% of the eval workload (outer frame),
- `inject_patch_drop` = 39.9% (384 calls),
- `patch_reference_paths` = 7.3% (one call, over the frozen calibration subset),
- `normalize_patch` = 3.0%,
- 5,461,710 calls to `ndvi_from_red_nir` at 2.3% (call-count cost, not arithmetic).

Rust porting did not help this workflow (1.0× end-to-end): the time sits in
injection plus model scoring. The owner therefore allows extracting and saving
deterministic intermediates to disk so repeated eval runs can reuse them.

A parity-checked probe (3 cells, n=32) measured 6.17 s cold vs 3.95 s warm,
i.e. **1.56×**, with a 2.59 s one-time build and 36 MiB per cell. All compared
outputs were identical with NaN-aware equality. Details are in design.md.

## What Changes

- Add a **new capability** `eval-artifact-cache`: an opt-in, fingerprinted disk
  cache for deterministic eval intermediates.
- Cache three layers, in this order: (1) the normalized patch batch, (2) the
  clean calibration reference paths, (3) per-severity-cell injected tensors and
  their hash-bearing injection records.
- Bind each cache entry to a fingerprint of every input that can change its
  value: store hash, split/holdout hashes, norm hash, model checkpoint id/hash,
  sigma table hash, severity table hash, cell id (level/family/index/round),
  injection basis, and cache format version.
- Fail open: a miss, mismatch, or corrupt entry triggers recomputation and a
  rewrite. A stale entry is never trusted. The default path stays unchanged.
- Require byte-level parity between a cached run and a cold run for every cell
  before the cache may be enabled for research use.
- Keep the cache directory untracked (`/data/` is already ignored by Git).

## Capabilities

### New Capabilities

- `eval-artifact-cache`: Opt-in reuse of fingerprinted eval intermediates, with
  strict invalidation and a parity gate.

### Modified Capabilities

None. This change complements `optional-rust-acceleration`; it does not alter
its opt-in contract or its ports.

## Impact

- New cache directory under `data/cache/eval/` (untracked). No tracked binary.
- Optional loader/writer helpers and a fingerprint module; no change to the
  scoring, injection, or severity-table semantics.
- Evidence: extend `reports/fast_anything_benchmark.md` with cold-vs-warm and
  cache-build timings, or add a dedicated report beside it.
- No change to data contracts, platform payloads, thresholds, labels, or any
  reported research number. No scientific claim (validation-ladder rung L0).
