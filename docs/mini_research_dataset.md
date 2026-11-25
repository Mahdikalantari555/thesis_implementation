# Mini research dataset — preparation record

Scope: the additive three-farm mini store defined by the `mini-research-dataset`
OpenSpec change. This document records the preparation decisions that are
dependencies of extraction; it is not a results report and makes no detection
claim (validation-ladder rung L0 for the data itself, L1/L2 for any model run).

Verified 2026-11-24 with the `geo` conda environment against the real archive.

## 1. Companion farms (task 0.1)

> **Superseded 2026-11-25 (owner decision):** the mini dataset must use the
> farms of `D:\RS\Thesis\Data\RoI\3farms_dev.geojson` (three separated farms).
> The R12-06/R12-08/R12-10 selection below (adjacent strips, FID 244–246) was
> the wrong layer. `scripts/build_mini_dataset.py` now reads the farm set from
> that layer, maps names to archive ids via `Farms.geojson`, refuses to mix farm
> sets in one store, and records `archive_alignment` (IoU of archive pixels vs
> own and ±1 neighbour polygons) to test the ~270 m offset / id off-by-one
> hypothesis. The table below is kept as history.


Source: `D:\RS\Thesis\Data\RoI\Farms.geojson` (490 rows, EPSG:4979, 3D polygons),
reprojected to EPSG:32639 before any pixel geometry.

| Farm | OGC_FID | area (ha) | variety | bbox EPSG:32639 (minx, miny)–(maxx, maxy) | H×W @ 30 m |
|---|---|---|---|---|---|
| R12-06 | 244 | 25.5 | CP48-103 | (237153.69, 3434750.74)–(237519.60, 3435777.23) | 288×14 |
| R12-08 | 245 (primary) | 25.5 | SP70-1143 | (236884.30, 3434783.04)–(237251.88, 3435809.50) | 282×14 |
| R12-10 | 246 | 25.5 | CP73-21 | (236611.91, 3434813.75)–(236982.72, 3435840.53) | 282×14 |

H×W follows the bounded-window convention of
`scripts/build_raw_satellite_subset.py:128-131`. The earlier design note of
"~282" holds for R12-08/R12-10; R12-06 is 288 rows, so the store must keep
per-farm spatial sizes rather than assuming a shared shape.

These three farms are **not** in `Farms_subset.geojson` (the 32-farm file) and
only R12-08 appears in `3farms-dev.geojson`. The full `Farms.geojson` is the
authoritative source.

## 2. Date source (task 0.2)

**Decision:** build the 538-date list from
`D:\RS\Thesis\Data\Processed\satellite\scene_inventory.parquet`
[`acquisition_date`] — 538 unique strings, 2013-03-30 … 2026-09-10, each with a
scene directory under `D:\RS\Thesis\Data\Metric\scenes\`
(`landsat_YYYYMMDD_<path>_<row>/`, 607 directories total, exactly one per
inventory date).

**Rejected source:** day-numbers in `stats.zarr["time"]`. Verified metadata:

```
units: days since 2013-03-30 00:00:00
calendar: proleptic_gregorian
```

The legacy script `build_raw_satellite_subset.py:211-216` assumes a base date of
2013-01-01, shifting every derived date by 88 days and producing phantom dates
that have no scene. design.md §2 also states the wrong base (2013-01-01); the
correct base is **2013-03-30**. Converting with the *recorded* units is valid,
but direct use of the inventory dates is simpler and fails closed when a scene
is missing.

`weather_aligned.parquet` has 539 dates: the 538 scene dates plus 2022-08-07,
which has no scene. The mini weather table keeps the 538 scene dates (left
join); the extra weather-only date is recorded as a join diagnostic, never as
a satellite observation.

## 3. Cloud/validity semantics (task 0.3, resolved 2026-11-24)

`scene_inventory.parquet`: `cloud_coverage` is METRIC scene-level (0.71–1.0,
varies) and real; `scene_quality = LOW_QUALITY` for all 538 is a label, not a
measurement; `valid_pixels` (0.41–0.67) is the ROI-relevant gate. Per-pixel
validity in the mini store comes from the Landsat C2 `qa_pixel.tif` bits
(cloud, cloud shadow, cirrus, snow, fill) implemented in
`ssl_anomaly.io.landsat.quality_mask`, plus a finite-value check because all
607 scene band rasters were re-masked to NaN at QA-bad pixels in the 2026-11-24
cloud fix. `cloud_coverage` stays scene-level metadata and is never used as a
per-pixel gate.

## 4. Weather feature-count correction

`weather_aligned.parquet` is 539 rows × 57 columns: 1 date key
(`acquisition_date`) + **56** weather features, not 57 weather features. The
mini weather table keeps all 56 verbatim and adds `gdd_daily` and
`heat_stress_hours` as two new derived columns (58 feature columns + date).

GDD definition: daily, base 10 °C, from `temperature_min` / `temperature_max`
(`(Tmin + Tmax)/2 − 10`, clipped at 0). This differs from the existing
`scripts/ingest_real.py:312` (`T − 10` from hourly/mean T). Both are recorded
under distinct column names; the mini dataset uses `gdd_daily` and never
silently rewrites the legacy `weather_gdd`.

## 5. Validation indicator availability

- Real per-(farm, date) ETa and CWSI: `thesis_implementation/data/interim/metric_zonal_stats_cache.parquet`
  (`date`, `field_id`, `cwsi_mean`, `eta_mean`; 4375 daily dates, full coverage
  for FIDs 244/245/246).
- `stats_current.zarr` CWSI_ET covers fields 1–32 only, **not** the three target
  farms; it must not be used for them.
- LAI, TVDI, ETrF, METRIC NDVI/LST exist only as per-scene GeoTIFFs under
  `Data/Metric/products/<INDICATOR>/`; they require zonal extraction for the
  three farm polygons and are absent where no raster exists.
- All indicators remain independent evaluation sources. None enters training,
  threshold calibration or model selection; shared-input indicators are
  disclosed as proxy-only.

## 6. Reproducibility

Seeds, config snapshots, date-list hash, farm-list hash, source paths and git
SHA are written to `Data/Processed/mini_research_dataset/manifest.json`. Every
model run additionally writes its own manifest and states its rung (L1/L2
only). No value in the mini store is synthesized: absent values stay absent.
