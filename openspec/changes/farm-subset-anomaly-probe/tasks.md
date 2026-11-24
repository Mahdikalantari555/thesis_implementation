# Tasks — farm-subset-anomaly-probe

## 0. Preparation
- [ ] 0.1 Confirm farm subset (32 farms from `Data/RoI/Farms_subset.geojson`).
- [ ] 0.2 Confirm IRIS XE is NOT usable (verified: torch CPU-only, no IPEX/OpenVINO; ONNXRuntime CPU available).

## 0b. Date-centric subset (owner decision: date-centric input)
- [ ] 0b.1 Extract `raw_satellite_subset.zarr` (D=538, F=32, K=7, S=48) from METRIC products for the 32 farms.
- [ ] 0b.2 Parity check vs `stats_long.parquet` for the 32 farms.
- [ ] 0b.3 Fallback to v2 cycle-centric subset if extraction blocked.

## 1. Methods (all 4, on subset, full time range)
- [ ] 1.1 B2 no-NN baseline (phenology + date×neighbor) — σ from residual (train split only). **Phenology uses multi-band (NDVI+LST+CWSI+GDD), not NDVI alone** (v2 lesson).
- [ ] 1.2 Pixel-wise CNN autoencoder (≤250k params) → reconstruction residual = score.
- [ ] 1.3 Barlow Twins (ST-BT style) on spectral-temporal trajectories → representation distance = score.
- [ ] 1.4 GRU-AE baseline (existing) on subset.

## 2. Hybrid scoring output
- [ ] 2.1 Stage 1: field-date anomaly scores (parquet) for 32 farms × 538 dates.
- [ ] 2.2 Stage 2: pixel-level (48×48) score for top-K suspicious (field, date).
- [ ] 2.3 Persistence (≥2 obs, ≥5 d) + cold-LST gate (ΔLST < −2 °C rejected).
- [ ] 2.4 Event-level framing (Q8); no "stress" wording (L2 only).

## 3. Feasibility & runtime
- [ ] 3.1 ≤15 min/seed, ≤250k params (AGENTS.md §7).
- [ ] 3.2 `--resume` support; progress bars + `progress.jsonl`.
- [ ] 3.3 ONNXRuntime CPU for Stage-2 pixel inference.

## 4. Acceptance
- [ ] 4.1 All 4 methods run end-to-end on laptop.
- [ ] 4.2 Anomaly score artifacts with provenance.
- [ ] 4.3 No claim beyond L2.
- [ ] 4.4 Reproducible (seeds fixed).
