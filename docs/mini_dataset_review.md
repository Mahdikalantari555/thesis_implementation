# Mini Research Dataset — Technical Review (no proposal yet)

**Date:** 2026-11-24
**Scope:** Review of existing data/infra before any OpenSpec change for the 3-farm mini dataset.
**Companion farms (assumed, configurable):** R12-06, R12-08 (primary), R12-10 — all block R12, ~25.5 ha, 30 m resolution.

---

## 1. Current data infrastructure (verified)

| Asset | Scope | Notes |
|---|---|---|
| `Data/Metric/scenes/` | 607 scene dirs, 2013-03-30 → 2026-09-10 | Landsat L2 per-band GeoTIFFs (blue/green/red/nir/swir1/swir2/lst + qa_pixel) |
| `Data/Processed/satellite/stats.zarr` | 490 fields × 538 dates × 7 bands | date list source (days since 2013-01-01) |
| `Data/Processed/satellite/stats_current.zarr` | 32 fields × 539 dates × 7 bands | v2 subset (fields 1–32) |
| `Data/Processed/satellite/pixels/*.parquet` | 538 dates × 490 farms, 133,822 pixels/date | per-pixel bands + valid mask + row/col/x/y |
| `Data/Processed/weather/weather_shared.parquet` | 5003 days × 57 cols | rolling 3/7/14/30 d aggregates + VPD, already causal |
| `Data/Processed/weather/weather_aligned.parquet` | 539 dates × 57 cols | aligned to satellite dates |
| `Data/Weather/raw/point_01_raw_hourly.parquet` | hourly | enables heat-stress-hours derivation |
| `Data/Processed/interim/field_mask_full.tif` | full 490-farm mask | spatial reference for rasterization |
| `Data/Processed/coordinates/farm_pixels.parquet` | 133,822 rows | pixel→farm mapping |
| `Data/RoI/Farms.geojson` | 490 farms | polygons + metadata (area, variety, age, status) |
| `Data/RoI/Farms_subset.geojson` | 32 farms (R11 block) | existing subset (odd/even) |
| `Data/Processed/raw_satellite_subset.zarr` | 2 farms done (R11-30, R11-32) | partial extraction from earlier run |

## 2. Verified dimensions / coverage

- **Acquisition dates:** 538, irregular (not fixed cadence). Real dates + time-delta must be retained. First 2013-03-30, last 2026-09-10.
- **Farm pixel sizes:** H 10–49, W 8–36 (30 m). Selected 3 farms: 36 × 14 each.
- **Patch counts (36×14, stride 1):** 3×3 → 408, 5×5 → 320, 7×7 → 240. Border padding required for 7×7.
- **Bands:** 7 physical (blue, green, red, nir, swir1, swir2, lst). NDVI/SAVI computed on-demand (per existing decision).
- **Weather columns already present:** T mean/min/max, RH, precip, solar, wind, ET0, soil moisture/temp, VPD, rolling 3/7/14/30 d for all. **Missing in parquet:** GDD, heat-stress hours (derivable from hourly + daily).

## 3. Existing capabilities to reuse (avoid duplication)

- `scripts/build_raw_satellite_subset.py` — extraction pattern (per-farm zarr groups, fp16, quality mask, resume, manifest). Reuse for 3-farm mini store.
- `src/ssl_anomaly/io/landsat.py` — `quality_mask`, `apply_scale_sr/st`, `load_scene`.
- `src/ssl_anomaly/io/zarr_store.py` — zarr v2 writer (xarray-backed).
- `src/ssl_anomaly/models/rst_bt.py` — RST-BT PoC (133k params).
- `src/ssl_anomaly/models/gru_ae.py` — GRU-AE baseline.
- Lightning 2.6.6, mlflow 3.18, torch 2.15 CPU, zarr 3.3.0, xarray 2026.7, sklearn 1.9, scipy 1.18.
- **Missing (user installing):** optuna, tensorboard.

## 4. Zarr version question

- zarr 3.3.0 is installed. Existing code mixes v2 (fields.zarr, via xarray) and v3 (era5/weather, via xarray).
- xarray 2026.7 supports both. For the mini dataset, **v3 is recommended** for new stores (native sharding, consolidated metadata, better for chunked lazy access) while keeping v2 readers for legacy stats.zarr untouched.
- Decision needed: whether mini store is v3-only or dual-readable. Recommend v3 with a small reader shim.

## 5. Leakage-prevention observations

- Weather parquet rolling aggregates are already causal (computed from past days only). Good.
- Normalization must be fit on train split only (existing convention; must be enforced in new code).
- Temporal split: by acquisition date (not row index) to avoid irregular-cadence leakage.
- Farm split: by farm_id; with only 3 farms, leave-one-farm-out gives 3 rotations.
- Weather fusion must use only data available up to observation date (already satisfied by aligned parquet).

## 6. Patch / spatial constraints (verified)

- Farm areas ~25 ha → ~36×14 pixels at 30 m. Very small.
- 7×7 patches need padding (farm smaller than patch in W dimension? W=14 ≥ 7, OK; but border pixels). Recommend masking out-of-farm patches via `in_field` mask.
- Valid-pixel ratio per farm/date must be tracked (cloud mask). `valid_frac` already in stats.
- Number of usable patches per farm/date is small → augmentation and regularization important.

## 7. Compute budget

- Laptop: torch CPU only (IRIS XE not usable for torch — verified earlier). ~15 min/seed, ≤250k params (AGENTS.md §7).
- Models that fit: MLP-AE, GRU-AE, small CNN-AE, Barlow Twins. ConvLSTM/Mamba/CNN-Mamba may exceed budget on CPU → defer to Colab.
- Colab T4 available for heavy runs (user has access historically).

## 8. Open decisions for the user (before any proposal)

1. **Companion farms:** confirm R12-06 + R12-10, or specify others (spatial diversity vs same-block).
2. **Zarr version:** v3 (recommended) or v2 for compatibility with existing xarray pipelines.
3. **Weather feature set:** use existing 57 cols + derive GDD & heat-stress-hours from hourly, or subset to the doc's 15 listed features.
4. **Patch sizes to benchmark:** 3×3, 5×5, 7×7 (doc) — confirm; larger needs padding strategy.
5. **Model scope for laptop:** which architectures to run locally vs defer to Colab (Mamba/ConvLSTM likely Colab-only).
6. **Experiment tracking:** mlflow (available) vs optuna (being installed) — both? Recommend mlflow for tracking + optuna for HPO.
7. **Mini store location:** `Data/Processed/mini_research_dataset.zarr` (new, additive) — confirm naming.
8. **Hold-out:** the doc mentions holding out 1 year for validation — confirm which year (suggest 2025–2026, consistent with existing holdout).

## 9. Verified findings (2026-11-24)

- **Date source bug found:** `stats.zarr["time"]` holds day-numbers since 2013-01-01, NOT acquisition dates. Converting them yields phantom dates. Correct source = `scene_inventory.parquet` acquisition dates (538 real dates, 2013-03-30 → 2026-09-10). `build_raw_satellite_subset.py` must be corrected.
- **538 vs 539:** weather_aligned has 539 = 538 scene dates + `2022-08-07` (no scene). Mini store keeps 538 satellite dates; weather left-joined.
- **Zarr benchmark:** v3 write 0.89s / read 1.30s; v2 write 1.47s / read 1.37s on (538,7,36,14) float16. Both xarray-compatible. **Decision: v2 for portability.**
- **Pixel grid consistency:** verified consistent across dates for a farm (282 px, same row/col range).
- **Monthly coverage:** 27–64 scenes/month, no empty month. Dates/year 23–70 (2026 partial).
- **Hourly weather:** full coverage 2012-12-31 → 2026-09-11. GDD and heat-stress-hours derivable.
- **GDD feasibility:** verified (daily T min/max, base 10 °C).
- **Cloud metric (RESOLVED):** `cloud_coverage` (METRIC-derived, scene-level, 0.71–1.0, varies across 107/538 scenes) is a real metric, not a placeholder. `scene_quality`="LOW_QUALITY" for all 538 is just a label. `valid_pixels` (0.41–0.67) is the ROI-relevant fraction of valid pixels. The two disagree because cloud_coverage is scene-wide while valid_pixels is ROI-focused. Both usable; prefer `valid_pixels` (via `quality_mask` on qa_pixel) as the per-pixel gate. No blocker.
- **OGC_FID:** R12-08 = 245 confirmed; R12-06/10 to verify during extraction.
- **Seasons:** sugarcane plant Mar–Apr, harvest Dec–Jan (Khuzestan). 2025–2026 holdout covers complete + partial cycles. Temporal holdout kept configurable (not hard-coded).

## 10. Recommendation

OpenSpec change `mini-research-dataset` written (proposal/design/tasks). No implementation yet. Awaiting review before applying.<|end|><|start|> assistant to=Bash<|channel|>analysis <|constrain|>xml<|message|><rfl:parameter name="command">
echo written