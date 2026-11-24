# How to read the timeline figures

- `timeline_seasonal.png`: field-mean NDVI and LST (°C) per farm, 2013–2026. Blue/orange bands are sugarcane cycles (plant/ratoon) with their START and END lines; the dashed black line is the 2023 train/test boundary. Red dots are flagged acquisition dates.
- `timeline_anomaly_score.png`: the same dates as a *phase-conditioned deviation* — for each farm and stage S1–S4, the robust z against the train-year median/MAD. Dotted lines mark ±2.
- `timeline_flagged_dates.png`: top, the count of suspected 5×5 patches on every acquisition date; bottom, the same flags aggregated by year × month.

**Provenance:** NDVI = (nir−red)/(nir+red) and LST = band `lst`−273.15 from `Data/Processed/mini_research_dataset/satellite.zarr`, masked by `pixel_valid` and `in_field`. Cycles from `phenology/cycles_mini.parquet`. Flags from `suspected_stress/suspected_patches.parquet` (weak suspicion from re-masked METRIC products).

**Do not claim:** confirmed stress, water/salinity typing, early warning, or validation of the SSL model. The flags derive from METRIC products that share LST/NIR inputs with the spectral signal, so agreement with the NDVI/LST deviation is not independent evidence. Claim rung: L1/L2 screening only.

**Renderer note:** figures are drawn with Pillow because matplotlib's native draw/save crashes in this conda environment (matplotlib 3.11.1 + numpy 2.5.2, Windows).

## Run summary

| Farm | Dates | Flagged dates | Mean NDVI | Mean LST (°C) | NDVI \|z\|≥2 | LST \|z\|≥2 |
|---|---:|---:|---:|---:|---:|---:|
| R12-06 | 538 | 11 | 0.442 | 34.95 | 20 | 2 |
| R12-08 | 538 | 17 | 0.443 | 35.81 | 65 | 70 |
| R12-10 | 538 | 24 | 0.427 | 36.07 | 69 | 27 |
