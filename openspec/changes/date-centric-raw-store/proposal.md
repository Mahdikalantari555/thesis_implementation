# Change: date-centric-raw-store

**Change ID:** `date-centric-raw-store`
**Project:** Thesis Implementation
**Status:** Proposed
**Priority:** High (architectural, post-benchmark)
**Target hardware:** Colab T4 / local geo env for extraction; laptop for tests
**Created:** 2026-11-24

---

## Why

The current v2 store (`patches.zarr`) is **cycle-centric**: 490 fields × 12 cycle slots × 36 bins of 15 days, each bin collapsed to a pixel-median. This discards the real acquisition date, the within-bin temporal spread (up to 14 days), and the pixel distribution. For short stress events (7–14 days) and irregular time-series models this is a hard ceiling, and the analysis in `D:/RS/Thesis/data_architecture_analysis.md` (§2, §8) confirms it.

This change introduces a **date-centric raw satellite store** that keeps **all 538 acquisition dates × 490 fields × 7 base bands × 48×48 pixels**, with cycle/phenology/NDVI becoming **derived views and metadata** — not the storage axis. It is **additive**: the v2 cycle-centric store, splits, checkpoints and reports remain untouched and continue to be the benchmark's input.

## What Changes

### A. Raw Satellite Store (Date-centric) — new, additive
- `raw_satellite.zarr`: per-farm arrays `(D=538, K=7, H_i, W_i) fp16` + `pixel_valid`, `in_field`, `dates`, `field_ids`, `sensor_id`, `bbox_origin`. **Variable spatial size** (farm bounding box, no 48×48 padding) — owner decision 2026-11-24. Each farm gets its own zarr group. NDVI is NOT stored (computed on-demand).
- Base channels = 7 physical bands only: `blue, green, red, nir, swir1, swir2, lst`. **NDVI is NOT stored** (computed on-demand by the feature-engineering module).
- Chunking: `(1, 1, 7, 48, 48)` with sharding for date- and field-centric access.
- Source: METRIC products / `stats_long.parquet` / `zonal_stats/*.parquet` (538 dates). Validation: parity vs `stats_long.parquet`.

### B. Feature Engineering Module — new
- `ssl_anomaly.features.spectral_indices`: `compute_ndvi`, `compute_evi`, `compute_savi`, `compute_ndmi`, `compute_ndwi`, `compute_nbr` on base bands, pixel- and field-level, with versioned formulas and optional cache.

### C. Derived Temporal Metadata (Cycle/Phenology) — derived view
- `cycles_v3.parquet`: `field, cycle_id, start_date, end_date, harvest_date, ratoon_no, kind, length_days, censored, phenology_stage, stage_confidence`.
- Phenology stages extracted from the **multi-band** spectral-temporal curve (NDVI + SAVI + LST + GDD), with stage confidence. **NDVI alone is insufficient** (v2 lesson: missed winter harvests, conflated seasons). CWSI_ET is NOT needed (owner decision 2026-11-24). DSH retained as **temporal conditioning**, not storage axis.
- Classical sugarcane phases (§4 of analysis): Germination (30–45d), Tillering/Canopy (60–90d), Grand Growth (120–180d), Maturation (60–90d), Harvest (event). Ratoon vs plant-cane handled.

### D. Validation Store — independent
- `validation.zarr`: ETa, ETrF, CWSI, TVDI, LAI, METRIC NDVI, LST_metric, aligned by (field_id, date). Never enters training DataLoader by default.

### E. Field Statistics Store — independent
- `field_stats.zarr`: per (field, date) mean/median/std/iqr/quantiles per band, valid_frac, n_pixels.

### F. Weather Store
- `weather.zarr`: daily (5002 days) + aligned to satellite dates (538), causal constraint (only info available up to observation date).

### G. Cycle-detection method comparison (owner-gated)
- Multiple detectors (multi-band NDVI+LST+CWSI peak/valley, change-point on multi-band feature vector, phenology-model fit on NDVI+LST jointly) run on a **farm subset** (selected sensitive farms). Owner visually approves which method detects cycles best on those farms. No single method is imposed; the chosen method + its parameters are recorded. **NDVI-only detection is explicitly rejected** based on the v2 finding (18/19 long cycles were two seasons joined by a missed winter harvest).
- Figures under `D:/RS/Thesis/.scratch/advisor_pack_2026-11-22/figures_perfield/` are the reference for owner review.

### H. Model-training update (additive)
- Tokenizer/decoder gain an optional **date-centric path** (irregular time series with Δt feature + time encoding) alongside the existing cycle-centric path. The cycle-centric path stays the default for the current benchmark.
- A small **farm-subset smoke** (selected farms) runs end-to-end on the owner's laptop to validate the new path without re-running the full benchmark.

## Capabilities

- New capability: `date-centric-raw-store` (raw store, feature engineering, derived phenology, validation/field-stats stores, weather store, cycle-detection comparison, additive model path).
- Modified: none of the frozen specs are changed; `data-readiness` and `model-development-protocol` invariants apply.

## Impact

- **Affected (new only):** `Data/Processed/raw_satellite.zarr`, `Data/Processed/validation.zarr`, `Data/Processed/field_stats.zarr`, `Data/Processed/weather.zarr`, `ssl_anomaly.features.*`, `ssl_anomaly.phenology.*`, new scripts, new tests.
- **Frozen / untouched:** `patches.zarr` (v2), `splits_locked_v2.json`, `holdout_manifest_v2.json`, `norm_stats_full_v2.json`, all checkpoints, findings 00–25, platform, the locked test split, the benchmark change.
- **Risks:** 8.3 GB storage, owner approval for cycle detection, parity tests must pass before any model uses the new store.

## Prerequisites

- Backup of current `Data/Processed/` (Phase 0).
- Owner approval of cycle-detection method on the farm subset.
- Colab T4 for full extraction (8.3 GB).

## Acceptance

- All 8 parity tests (§13 of analysis) pass.
- `raw_satellite.zarr` contains all 538 dates × 490 fields × 7 bands.
- Cycle-detection method chosen and recorded with owner approval on the farm subset.
- Feature engineering module has unit tests for each index.
- Existing v2 benchmark still runs unchanged (regression).
- Farm-subset smoke (selected farms) completes on laptop.
