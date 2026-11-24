# Tasks — mini-research-dataset

## 0. Preparation (no code yet)
- [ ] 0.1 Confirm companion farms R12-06, R12-08 (primary), R12-10 and record OGC_FIDs.
- [ ] 0.2 Document date-source decision: use scene_inventory acquisition dates (538), not stats.zarr day-numbers.
- [x] 0.3 Cloud metrics resolved (2026-11-24): `cloud_coverage` is METRIC scene-level (0.71–1.0, varies), real; `valid_pixels` (0.41–0.67) is ROI-relevant gate. No blocker.

## 1. Extraction (smoke first, then full)
- [ ] 1.1 Create `scripts/build_mini_dataset.py` reusing `build_raw_satellite_subset.py` patterns, with corrected date source, --farms filter, --out, --resume, --smoke.
- [ ] 1.2 Smoke run: 1 farm × 5 dates → verify shapes, NaN rates, parity vs zonal stats.
- [ ] 1.3 Full run: 3 farms × 538 dates → per-farm zarr groups (v2), fp16, quality mask, in_field mask.
- [ ] 1.4 Parity check: per-band median vs `zonal_stats/*.parquet` for 3 farms (tolerance).
- [ ] 1.5 Manifest: farm list, date list hash, git SHA, source paths, zarr version.

## 2. Weather
- [ ] 2.1 Build `weather_aligned_mini.parquet` (538 rows, left join on scene dates).
- [ ] 2.2 Derive `gdd_daily` (base 10 °C) and `heat_stress_hours` (hourly T>35°C).
- [ ] 2.3 Write `weather_features.json` with feature groups for ablation.

## 3. Feature engineering module
- [ ] 3.1 `ssl_anomaly.features.spectral_indices`: ndvi, savi, evi, ndmi, ndwi, nbr (pixel + field level, versioned formulas).
- [ ] 3.2 Unit tests for each index.

## 4. Validation store
- [ ] 4.1 Align ETa, ETrF, CWSI, TVDI, LAI, METRIC NDVI/LST by (farm, date) into `validation/` (independent indicators, not ground truth).

## 5. Phenology (multi-band)
- [ ] 5.1 Derive sugarcane phases from NDVI+SAVI+LST+GDD curve for the 3 farms.
- [ ] 5.2 Record cycle boundaries + stage confidence; document season structure for holdout decision.

## 6. Experiment harness
- [ ] 6.1 Configurable protocols: single-farm, LODO, intra-farm temporal.
- [ ] 6.2 Configurable spatial repr: pixel, patch (configurable sizes), full-field; valid-patch counting from in_field mask.
- [ ] 6.3 Configurable temporal repr: full series, sliding window, seasonal, variable-length, sampling; Δt preserved.
- [ ] 6.4 Model-agnostic registry: MLP-AE, GRU-AE, LSTM-AE, CNN-AE, ConvLSTM-AE, Mamba-AE, CNN-Mamba, Barlow Twins, JEPA.
- [ ] 6.5 MLflow tracking + Optuna HPO; controlled comparison (equal budget first).
- [ ] 6.6 Train-only normalization fit; no leakage.

## 7. Acceptance (smoke + first runs)
- [ ] 7.1 All protocols runnable end-to-end (smoke) on laptop.
- [ ] 7.2 At least one light model (GRU-AE) + one representation-based (Barlow Twins) run with provenance.
- [ ] 7.3 MLflow run + Optuna study recorded.
- [ ] 7.4 Valid-patch counts reported per farm/date.
- [ ] 7.5 No claim beyond L2.
- [ ] 7.6 Temporal holdout boundary documented + justified per run.