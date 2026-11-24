# Design — date-centric-raw-store

## Principle: Additive, not destructive
The v2 cycle-centric store (`patches.zarr`), splits, checkpoints and reports are **frozen**. This change adds a parallel date-centric raw store and derived views. Nothing is removed; the cycle-centric path remains the default for the current benchmark.

## Phenology in date-centric
Phenology is a **derived metadata layer**, not a storage axis:
- Raw store holds all 538 dates (no phase).
- `cycles_v3.parquet` holds phases + confidence + DSH, extracted from the multi-band
  spectral-temporal curve (NDVI + LST + red-edge proxies + weather), not NDVI alone.
  **Lesson from v2 phenology:** NDVI-only cycle detection missed winter harvests and
  conflated two seasons into one (18/19 long cycles were two seasons joined by a
  missed winter harvest). Multi-band input (NDVI + LST + CWSI_ET + GDD) is required
  for robust harvest detection.
- Classical sugarcane phases (Germination 30–45d, Tillering 60–90d, Grand Growth
  120–180d, Maturation 60–90d, Harvest event). Ratoon vs plant-cane via kind/length_days.
- DSH becomes temporal conditioning in the tokenizer (one-hot stage + DSH), not a bin index.

## Cycle-detection method comparison (owner-gated)
Three detectors on a farm subset selected by the owner from `figures_perfield/`:
1. Multi-band NDVI+LST+CWSI peak/valley threshold (current v2, extended to multi-band).
2. Change-point (PELT / rbocpd) on multi-band feature vector.
3. Phenology-model fit (logistic/Gaussian) on NDVI + LST jointly.

Owner visually approves the best method per farm (or globally). Only then run on all 490
fields. No single method is imposed by default. **NDVI-only detection is explicitly
rejected** based on the v2 finding (missed winter harvests).

## Farm subset for testing
A selected subset of farms (5–10 sensitive farms) runs end-to-end on the owner's laptop, so training/feature pipelines can be validated locally before Colab scale-up.

## Storage schema
See proposal §A. Chunking `(1, 1, 7, 48, 48)` with sharding. ~8.3 GB uncompressed / ~2–3 GB compressed.

## Model path (additive)
Tokenizer/decoder gain an optional date-centric path (irregular time series, Δt feature, time encoding). Cycle-centric remains default. Farm-subset smoke validates it.
