# Mini research dataset — acceptance evidence

Generated 2026-11-24 from real artifacts only. **Claim ceiling: validation
ladder L1/L2.** These are screening runs on three farms; no stress detection,
geographic generalization or baseline-superiority claim is made.

## Store and parity (tasks 1.1–1.5)

- `D:/RS/Thesis/Data/Processed/mini_research_dataset/satellite.zarr` — zarr v2,
  per-farm groups R12-06/OGC_FID 244, R12-08/245, R12-10/246.
- 538 real acquisition dates, sorted, 2013-03-30 → 2026-09-10;
  date-list SHA-256 `0cd6e7202d89…` (manifest).
- Per farm: `data` (538, 7, 34, 12) fp16 chunks (1,7,H,W), `pixel_valid`
  (538,34,12) uint8, `in_field` (34,12) uint8. In-field pixels: 288 / 282 / 282.
- All 538 dates present for all three farms; no synthesized values.
- Parity vs `zonal_stats/*.parquet`: **11,270 / 11,270 checks pass**
  (`parity_full.json`); max reflectance abs diff 0.0177, LST within tolerance.
- `farms.geojson` (3 farms) and `manifest.json` (git SHA `d60963e0…`, source
  paths, date hash, zarr version, wall time 59.66 s) written.
- Provenance note: the canonical processed archive places these farms ~270 m
  east of the current `Farms.geojson` polygons; the store uses the real
  processed pixel grid and records both bboxes in each group's attributes.

## Weather and feature groups (tasks 2.1–2.3)

- `weather/weather_aligned_mini.parquet`: 538 rows, 56 raw features kept
  verbatim + `gdd_daily` + `heat_stress_hours` = 58 features.
- GDD = `max(0, (Tmin+Tmax)/2 − 10)`; heat-stress hours = hourly T2m > 35 °C
  per UTC day. Means: 19.71 °C·d and 9.35 h; max 24 h.
- `weather/weather_features.json` groups: spectral_only, weather_core (12),
  weather_rolling (44), weather_all (56), weather_plus_derived (58).

## Independent validation store (task 4.1)

- `validation/validation_indicators.parquet`: 12,912 rows (3 farms × 538 dates
  × 8 indicators), **12,856 finite** after the owner's 2026-11-24 re-masking
  of `Data/Metric/products/` (was 12,888 before; 56 fewer finite values).
  Coverage is 1,607 finite rows per indicator; the remaining rows are
  `absent_product` / `absent_no_finite_pixels`, never zero or imputed.
- Rebuilt 2026-11-24 from the re-masked METRIC GeoTIFFs;
  `validation_manifest.json` carries refreshed product SHA-256 hashes.
- ETa, ETrF, CWSI_ET, CWSI_LST, TVDI, LAI, METRIC NDVI/LST from real METRIC
  GeoTIFFs. Shared-input indicators flagged in `validation_manifest.json`.
  Never used for training, threshold calibration or model selection.
- Satellite parity re-verified after the re-masking: 11,270/11,270 pass.

## Phenology and holdout (tasks 5.1–5.2, 7.6)

- `phenology/cycles_mini.parquet`: 32 provisional cycles from NDVI+SAVI+LST+GDD
  with stage confidence (0 is possible; absent confidence stays NaN).
- `phenology/season_structure.json`: monthly coverage 23–70 scenes/month,
  no empty month; 2026 partial (44 dates).
- Temporal holdout candidate **2025-01-01**, 427 dates before / 111 after;
  rule = first day of the last complete calendar year before partial 2026.
  Configurable per run; each run's report carries boundary + justification.

## Experiment harness (tasks 6.1–6.6, 7.1, 7.3, 7.4)

- Protocols: single-farm, LODO (3 rotations), intra-farm temporal — all ran
  end to end (`protocols_smoke_report.json`, 6 runs, zero failures).
- Spatial: pixel, patch (3/5/7, no upper bound in code), full-field.
  Valid-patch counts per farm/date/size in `valid_patch_counts.json`
  (e.g. 3×3: min 0 on cloudy dates, max 303, mean ≈302 for R12-06).
- Temporal: full, sliding, seasonal, variable, sampling; Δt preserved.
- Registry: all 9 architectures build and forward-pass (MLP-AE, GRU-AE,
  LSTM-AE, CNN-AE, ConvLSTM-AE, Mamba-AE, CNN-Mamba, Barlow Twins, JEPA).
  Reconstruction and representation models use separate metric tracks.
- Normalization fitted on train partitions only; imputed values are marked
  `observed=False` and counted, never silently treated as measurements.
- MLflow 3.x SQLite backend + Optuna studies recorded under each run directory.

## First runs (tasks 7.2, 7.3, 7.5)

| Run | Family | Seeds | Loss (mean ± std) | Clean score | Deviated score | Synthetic AUPRC | Step reference | Random init |
|---|---|---|---|---|---|---|---|---|
| GRU-AE | reconstruction | 13, 42 | 1.119 ± 0.023 | 1.151 ± 0.029 | 1.170 ± 0.024 | 0.573 ± 0.004 | 0.649 | 0.572 |
| Barlow Twins | representation | 13, 42 | 0.849 ± 0.096 | 0.305 ± 0.014 | 0.378 ± 0.004 | 0.663 ± 0.018 | 0.649 | 0.668 |

- **Negative result, reported verbatim:** at this smoke budget (2 epochs,
  single-farm split, 8 train / 16 test windows) neither trained encoder beats
  the simple step detector; GRU-AE is at random-init level and Barlow Twins is
  marginally below it. This is **not** evidence that SSL works; it is a
  correctly recorded pilot with references. Any adoption decision needs the
  controlled, equal-budget comparison at larger n/seeds, not this run.
- Detector/pretext: one-channel NDVI drop on held-out windows only; clean and
  deviated sets are balanced (prevalence 0.5 by construction). AUPRC here is
  **controlled deviation sensitivity at L2**, not stress accuracy.
- Both reports carry git SHA, data source, split hashes, n, date ranges,
  device, wall time, seeds, config snapshot, holdout boundary and justification.
- No result is reported at L3–L5.

## Reproduce

```bash
D:/Dev/Miniforge3/envs/geo/python.exe -X utf8 scripts/build_mini_dataset.py --check-parity
D:/Dev/Miniforge3/envs/geo/python.exe -X utf8 scripts/build_mini_weather.py
D:/Dev/Miniforge3/envs/geo/python.exe -X utf8 scripts/build_mini_validation.py
D:/Dev/Miniforge3/envs/geo/python.exe -X utf8 scripts/build_mini_phenology.py
D:/Dev/Miniforge3/envs/geo/python.exe -X utf8 scripts/report_mini_patch_counts.py
D:/Dev/Miniforge3/envs/geo/python.exe -X utf8 scripts/run_mini_experiment.py --model gru_ae
D:/Dev/Miniforge3/envs/geo/python.exe -X utf8 scripts/run_mini_experiment.py --model barlow_twins --out D:/RS/Thesis/Data/Processed/mini_research_dataset/experiments_barlow_twins
```
