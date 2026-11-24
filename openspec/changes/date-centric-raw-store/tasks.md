# Tasks — date-centric-raw-store

## 0. Preparation (no code changes yet)
- [ ] 0.1 Full backup of `Data/Processed/` (Phase 0).
- [ ] 0.2 Owner reviews `figures_perfield/` (24 farms) and selects a **farm subset** (5–10 sensitive farms) for method comparison.
- [ ] 0.3 Record the selected farm subset in `configs/benchmarks/farm_subset.yaml` (or a new config).

## 1. Cycle-detection method comparison (owner-gated, no migration yet)
- [ ] 1.1 Implement 3 detectors on the farm subset: (a) multi-band NDVI+SAVI+LST+GDD peak/valley threshold (v2 extended), (b) change-point (PELT/rbocpd) on multi-band feature vector, (c) phenology-model fit (logistic/Gaussian) on NDVI+SAVI+LST jointly. **NDVI-only detection is explicitly rejected** (v2 lesson: missed winter harvests, conflated seasons).
- [ ] 1.2 Render per-farm figures (multi-band curves + detected cycles) for owner review.
- [ ] 1.3 Owner approves the best method per farm (or globally). Record method + params in config.
- [ ] 1.4 Only after approval: run chosen detector on all 490 fields → `cycles_v3.parquet` (draft).

## 2. Raw Satellite Store (Date-centric)
- [ ] 2.1 Create `scripts/build_raw_satellite.py` (extract 538 dates from METRIC products / zonal_stats).
- [ ] 2.2 Build `raw_satellite.zarr` with schema (D, F, K, S, S) fp16 + pixel_valid/in_field/dates/field_ids/sensor_id/bbox_origin.
- [ ] 2.3 Chunking `(1, 1, 7, 48, 48)` + sharding.
- [ ] 2.4 Parity tests (§13 of analysis): Count/Date/Field/Band/Pixel/Sensor/Spatial/parity vs `stats_long.parquet`.

## 3. Feature Engineering Module
- [ ] 3.1 Create `ssl_anomaly/features/spectral_indices.py` (ndvi, evi, savi, ndmi, ndwi, nbr) with unit tests.
- [ ] 3.2 Versioned formulas + optional cache.

## 4. Derived Temporal Metadata
- [ ] 4.1 Build `cycles_v3.parquet` (field, cycle_id, start/end/harvest, ratoon_no, kind, length_days, censored, phenology_stage, stage_confidence).
- [ ] 4.2 Phenology extraction from multi-band curves (NDVI + SAVI + LST + GDD; peak detection, derivative thresholds, stage confidence). **NDVI alone is insufficient** (v2 lesson: missed winter harvests). CWSI_ET not needed.
- [ ] 4.3 DSH as temporal conditioning (not storage axis).

## 5. Validation & Field Stats Stores
- [ ] 5.1 `validation.zarr` (ETa, ETrF, CWSI, TVDI, LAI, NDVI_metric, LST_metric) aligned by (field_id, date); never enters training DataLoader by default.
- [ ] 5.2 `field_stats.zarr` (per (field, date) mean/median/std/iqr/quantiles, valid_frac, n_pixels).

## 6. Weather Store
- [ ] 6.1 `weather.zarr` daily (5002) + aligned to satellite dates (538), causal constraint.

## 7. Model-training update (additive)
- [ ] 7.1 Add optional date-centric path to tokenizer/decoder (irregular time series, Δt feature, time encoding). Keep cycle-centric as default.
- [ ] 7.2 Farm-subset smoke (selected farms) end-to-end on laptop.
- [ ] 7.3 Regression: v2 benchmark still runs unchanged.

## 8. Acceptance
- [ ] 8.1 All 8 parity tests pass.
- [ ] 8.2 raw_satellite.zarr has 538 dates × 490 fields × 7 bands.
- [ ] 8.3 Cycle-detection method chosen + recorded with owner approval on farm subset.
- [ ] 8.4 Feature engineering unit tests pass.
- [ ] 8.5 v2 benchmark regression green.
- [ ] 8.6 Farm-subset smoke completes on laptop.
