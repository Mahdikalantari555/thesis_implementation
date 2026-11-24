# Change: mini-research-dataset

**Change ID:** `mini-research-dataset`
**Project:** Thesis Implementation
**Status:** Proposed
**Priority:** High (enables cheap, broad architecture/SSL benchmarking)
**Target hardware:** laptop (CPU) for extraction + light models; Colab T4 for heavy models
**Created:** 2026-11-24
**Companion farms:** R12-06, R12-08 (primary), R12-10 (block R12, ~25.5 ha, 36×14 px @ 30 m)

---

## Why

The full pipeline (490 farms, ~12–15 h extraction per store, 6 training runs) is too expensive for iterating on architecture, input representation, and SSL strategy choices. A small, self-contained, transferable 3-farm mini dataset lets us benchmark many architectures and SSL methods at low compute cost before committing the full archive. The mini dataset must stay **additive** and **not constrain** the experiment space: all architectures, spatial/temporal representations, weather feature sets, and farm/temporal split protocols must remain runnable.

## What Changes

### A. Mini satellite store (date-centric, per-farm, additive)
- `Data/Processed/mini_research_dataset/satellite.zarr` (zarr **v2**, see design §Zarr decision): per-farm groups, arrays `(D=538, K=7, H_i, W_i)` fp16 + `pixel_valid` (uint8) + `in_field` (uint8), plus per-farm attrs (farm_id, bbox, H, W, n_dates, crs, pixel_size_m).
- **Date list source = `scene_inventory.parquet` acquisition dates (538 real dates, 2013-03-30 → 2026-09-10).** NOT `stats.zarr` day-numbers (those are phantom calendar dates). This fixes a latent bug in `build_raw_satellite_subset.py`.
- Base channels = 7 physical bands only. NDVI/SAVI/EVI/NDMI/NBR/NDWI computed on-demand by feature module.
- Chunking: `(1, 7, H, W)` per farm (date-centric access). Sharding not needed at this scale.
- Resume + manifest (`.manifest.jsonl`) per farm.

### B. Weather store (reuse, no duplication)
- Reuse `weather_shared.parquet` / `weather_aligned.parquet` (57 features, causal rolling windows). Store as parquet under `mini_research_dataset/weather/` (symlink or copy of the aligned subset filtered to the 538 satellite dates) + manifest.
- **Add derived features:** GDD (daily, base 10 °C, from T min/max) and Heat-Stress Hours (hourly T > 35 °C from `point_01_raw_hourly.parquet`). Added as new columns, never overwriting raw.
- Feature-selection / ablation harness: configurable feature subsets (spectral-only, weather-only, spectral+weather, weather-fusion-at-different-stages).

### C. Metadata & validation
- `farms.geojson` (3 farms), `manifest.json` (versions, date list hash, git SHA, source paths).
- `validation.zarr`-style independent indicators (ETa, ETrF, CWSI, TVDI, LAI, METRIC NDVI/LST) aligned by (farm, date) — **not** used as ground truth, only independent evaluation sources.
- Parity check vs `stats_long.parquet` / `zonal_stats` for the 3 farms (per-band median within tolerance).

### D. Experiment infrastructure
- **Protocols (configurable, both supported):** (1) Single-Farm (train on R12-08, test on R12-06 & R12-10), (2) Leave-One-Farm-Out (3 rotations), (3) Intra-Farm Temporal (time split within one farm). All three runnable; no hard-coding.
- **Spatial representations (configurable):** pixel-level, patch-level (configurable sizes, default 3/5/7, no upper bound), full-field. Valid-patch counts computed from `in_field` mask; out-of-farm patches excluded unless explicitly allowed.
- **Temporal representations (configurable):** full series, fixed sliding windows (configurable length/stride), seasonal/growth-cycle windows, variable-length, temporal sampling. Real acquisition date + Δt preserved (irregular cadence).
- **Model-agnostic harness:** MLP-AE, GRU-AE, LSTM-AE, CNN-AE, ConvLSTM-AE, Mamba-AE, CNN-Mamba, Barlow Twins, JEPA (and future). Reconstruction-based and representation-based SSL evaluated on separate metric tracks (different loss/metrics).
- **Tracking:** MLflow for experiment tracking; Optuna for HPO. Controlled comparison: equal compute/param budget per architecture before HPO.
- **Runtime:** CPU-light models run locally; heavy models (Mamba/ConvLSTM/CNN-Mamba) runnable on Colab via the same scripts (`--device auto`). No architecture removed from the framework due to CPU limits.
- **Reproducibility:** fixed seeds, config snapshots, checkpoint + metric + report per run, `--resume`.

### E. Validation split policy (not fixed yet)
- Temporal holdout choice deferred until seasonal structure + coverage + leakage risk are documented (see design §Seasons). Default candidate: last ~2 seasons, but configurable and must be justified per run.

## Capabilities

- New capability: `mini-research-dataset` (3-farm store, weather reuse+derived, experiment protocols, tracking).
- Modified: none of the frozen specs; `data-readiness` and `model-development-protocol` invariants apply.

## Impact

- **Affected (new only):** `Data/Processed/mini_research_dataset/`, new scripts, new tests, docs.
- **Frozen / untouched:** `patches.zarr`, `stats.zarr`, `splits_locked_v2.json`, holdout manifest, checkpoints, findings 00–25, platform, benchmark change.
- **Risks:** 3 farms only → results are Pilot (L1–L2), not generalizable claims. Cloud coverage metric needs clarification before trusting valid_frac.

## Prerequisites

- Zarr v2 vs v3 benchmark completed (see design §Zarr) — v2 selected for portability.
- Date-source bug documented (stats.zarr day-numbers vs scene_inventory real dates).
- Seasonal/coverage/leakage review documented.
- User confirmation of companion farms (done: R12-06, R12-08, R12-10) and primary (R12-08).

## Acceptance

- 3-farm store extracted with 538 real dates, parity vs zonal stats.
- Both Single-Farm and LODO protocols runnable end-to-end (smoke).
- At least one light model (e.g. GRU-AE) + one representation-based (Barlow Twins) run on the mini store on laptop with provenance.
- MLflow run + Optuna study recorded.
- No claim beyond L2.
- Valid-patch counts reported per farm/date.