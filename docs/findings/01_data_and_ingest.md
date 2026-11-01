# یافته ۱ — داده و ingest (Farms_subset, 32 مزرعه)

## داده

- Farms_subset.geojson: ۳۲ مزرعهٔ R11 (۸۱۲.۴ ha).
- ۵۳۹ گذر تمیز لندست ۸/۹ (۲۰۱۳-۰۳ تا ۲۰۲۶-۰۹).
- باندها: ۷ باند خام (SR_B2..B7 + ST_B10/LST) + weather (۵ متغیر ERA5).
- **std** در کنار mean/median در zonal stats.

## خروجی

- `fields.zarr`: (field=32, time=539, band=7) + weather (time, 5).
- نرخ NaN ≈ ۰.۷٪.
- `fields.parquet`: متادیتا.

## یافتهٔ کلیدی

ingest روی subset با باندهای خام + weather + std اجرا شد. Pipeline روی دادهٔ واقعی (نه synthetic) کار می‌کند.