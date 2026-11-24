# Change: data-fix-and-colab-prep

## Why

The heavy architecture benchmark contract (`configs/benchmarks/benchmark_config.yaml`,
frozen at version 1) binds every cell to v2 artifacts, but an inventory of
`D:/RS/Thesis/Data/Processed` and `D:/RS/Thesis/thesis_implementation/data/processed`
shows several are absent or misplaced:

- `weather_daily.parquet` (`artifacts.weather_daily_v2`) is absent from both
  canonical `Data/Processed/weather/` and `thesis_implementation/data/processed/`;
  the bundle `google_drive/weather_daily_v2.zip` does not exist either.
- `decomposition/stl_features.parquet` (`artifacts.stl_features`, optional) is
  absent everywhere.
- `patch_runs/v3/patches.zarr` exists under `thesis_implementation/data/` and
  inside `google_drive/thesis_data_full_v2.zip`, but its sidecar
  `patch_sigma.npz` is absent, which blocks the `hierarchical_patch` (patch-level)
  cells.
- `cycles_490_v2.parquet`, `splits_locked_v2.json`, `holdout_manifest_v2.json`,
  and `norm_stats_full_v2.json` live only under `thesis_implementation/data/`
  and the bundle, not under canonical `Data/Processed/`.
- No `ground_labels*.zip` or ground-label parquet exists anywhere (owner task 4).

Without closing these gaps, `weather_inputs` and `patch_level` cells cannot run
and would have to be silently skipped, which violates prime directive 2
("nothing user-visible is synthesized") and research-integrity rule 7 (no
hidden configuration changes). Separately, the Rust acceleration path is
buildable only in WSL (no `cargo`/`rustc` on Windows), and the
`optional-rust-acceleration` spec states a `1e-9` parity tolerance while
`tests/parity/test_backends_parity.py` actually asserts `rtol=1e-6`, so the
gate text and its enforcement disagree.

This change closes the data-placement gaps, prepares and validates the Colab
bundle inputs, and records the WSL build skill plus the tolerance alignment.
It does not touch frozen v2 artifacts, benchmark cell definitions, or the
locked `test` split.

## What Changes

- **Data fix plan**: verify each artifact's presence at canonical and
  implementation paths; mirror v2 artifacts into `Data/Processed/` without
  moving/renaming the originals; generate or explicitly defer the patch sidecar;
  bind `weather_daily` and `stl_features` through `benchmark_config` amendments
  only after an owner upload provides them.
- **Colab prep**: validate the existing `google_drive` bundles (contents + sha256
  manifests), document extraction order for the Colab notebook, and keep the
  `hierarchical_patch` block labelled `BLOCKED` until the sidecar exists.
- **WSL skill**: record the WSL geospatial build command and the
  Windows-no-toolchain limitation in `rust/README.md`; align the spec's stated
  `1e-9` tolerance with the `1e-6` actually asserted (or tighten the tests) with
  a recorded decision; fast-forward the stale WSL clone before any further WSL work.
- **Deferred data**: unmet owner-dependent artifacts (ground labels, weather
  daily, optional STL features) are recorded as `Deferred` with reason, never
  synthesized.

## Capabilities

- New capability: `data-fix-and-colab-prep` (artifact presence, canonical
  placement, Colab bundle validation, WSL build documentation, tolerance
  alignment, deferred-data registry).
- No existing spec is modified; `data-readiness` and
  `optional-rust-acceleration` are referenced but their requirements remain as
  archived.

## Impact

- **Affected paths**:
  - `D:/RS/Thesis/Data/Processed/weather/weather_daily.parquet` (new, owner upload)
  - `D:/RS/Thesis/Data/Processed/decomposition/stl_features.parquet` (new, optional)
  - `D:/RS/Thesis/Data/Processed/satellite/patches.zarr` + `Data/Processed/patch_runs/v3/patch_sigma.npz` (mirror + sidecar)
  - `D:/RS/Thesis/Data/Processed/{cycles_490_v2.parquet, splits/, holdout/, norm v2}` (mirror)
  - `D:/RS/Thesis/thesis_implementation/google_drive/*` bundles (validated, not rebuilt)
  - `configs/benchmarks/benchmark_config.yaml` (amendments only, never silent edits)
  - `rust/README.md`, `openspec/specs/optional-rust-acceleration/spec.md`
- **Frozen / untouched**: `cycles_490_v2.parquet` (3,478 non-censored), `splits_locked_v2.json`
  (sha `30ecbe10…`), `holdout_manifest_v2.json` (sha `d501bf81…`), `patches.zarr`
  (v3), `norm_stats_full_v2.json` (sha `493f0395…`), findings 00–25, the
  platform, and the locked `test` split.
- **Risks**: owner uploads are the gating dependency for weather/labels/STL;
  patch sidecar generation may need a Colab run if done at scale; WSL clone is
  15 commits behind Windows HEAD.

## Prerequisites

- Owner uploads to `thesis_implementation/google_drive/`:
  - `ground_labels_v2.zip` (task 4, owner action)
  - `weather_daily_v2.zip`
  - `stl_features_v2.zip` (optional)
- WSL clone fast-forwarded to Windows HEAD (`349e5ba`) before any WSL work.
- Existing v2 artifacts present (verified: `patches.zarr` v3 + bundles).

## Acceptance

- Every artifact in the inventory is either present at its canonical +
  implementation path, or recorded in `tasks.md` as `Deferred` with a reason and
  owner action.
- `benchmark_config.yaml` amendments are appended (date + target + reason +
  approved_by) for any binding change; no silent edits.
- `rust/README.md` documents the WSL geospatial build command and the
  Windows-no-toolchain limitation; the spec tolerance matches the test assertion
  (`1e-6`) or the tests are tightened to `1e-9`, with the decision recorded.
- Verification: `pytest tests/unit/test_platform_data_contract.py
  tests/unit/test_platform_bilingual.py -q` (conda `geo`) reports pass; tsc +
  vite build for `platform/`; runtime screenshot gate with zero console/page
  errors. Rust parity (WSL only): `tests/parity/test_backends_parity.py` (2
  passed) + `tests/unit/test_backends.py` (4 passed).