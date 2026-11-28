# Tasks — farm-subset-anomaly-probe

## 0. Preparation
- [x] 0.1 Confirm farm subset (32 farms from `Data/RoI/Farms_subset.geojson`). (WP-3 2026-11-28: 32 features verified)
- [x] 0.2 Confirm IRIS XE is NOT usable (verified: torch CPU-only, no IPEX/OpenVINO; ONNXRuntime CPU available). (WP-3: onnxruntime was ABSENT — installed 1.32.0 + onnx 1.23.2 into geo; decision D-2)

## 0b. Date-centric subset (owner decision: date-centric input)
- [ ] 0b.1 Extract `raw_satellite_subset.zarr` (D=538, F=32, K=7, S=48) from METRIC products for the 32 farms. (harness rebuilt + smoke-passed c56c394, incl. date-epoch fix 2013-03-30 and archive-parity membership; FULL extraction = owner handover H-1)
- [ ] 0b.2 Parity check vs `stats_long.parquet` for the 32 farms. (checker in `scripts/probe/parity_subset_stats.py`; smoke-scale PASS — 5,376 stats + 189,945 pixel checks, 0 failures; full-store check = owner step H-2 after H-1)
- [x] 0b.3 Fallback to v2 cycle-centric subset if extraction blocked. (not triggered — extraction feasible; measured 5.3 s smoke / ≈12 min full)

## 1. Methods (all 4, on subset, full time range)
- [x] 1.1 B2 no-NN baseline (phenology + date×neighbor) — σ from residual (train split only). **Phenology uses multi-band (NDVI+SAVI+LST+GDD), not NDVI alone** (v2 lesson). CWSI_ET not needed. (`src/ssl_anomaly/probe/b2.py`; date×block == date effect — all 32 farms are one R11 block, documented; GDD attached as cycle-cumulative feature)
- [x] 1.2 Pixel-wise CNN autoencoder (≤250k params) → reconstruction residual = score. (197,895 params, asserted by test)
- [x] 1.3 Barlow Twins (ST-BT style) on spectral-temporal trajectories → representation distance = score. (canonical `_barlow_loss`, off-diag fix; random-init comparator supported via `BarlowProbe(random_init=True)`)
- [x] 1.4 GRU-AE baseline (existing) on subset. (wraps `models/gru_ae.CausalGRUAE`, d_in=16, ~214k params)

## 2. Hybrid scoring output
- [ ] 2.1 Stage 1: field-date anomaly scores (parquet) for 32 farms × 538 dates. (implemented + smoke 2.0 s end-to-end; full-scale parquet pending owner run H-3)
- [ ] 2.2 Stage 2: pixel-level (48×48) score for top-K suspicious (field, date). (implemented via ONNXRuntime CPU; Stage-1↔Stage-2 smoke parity 3.4e-7; full-scale pending H-3)
- [x] 2.3 Persistence (≥2 obs, ≥5 d) + cold-LST gate (ΔLST < −2 °C rejected). (unit-tested; smoke: 27 cold-LST rejections observed on a real date snapshot)
- [x] 2.4 Event-level framing (Q8); no "stress" wording (L2 only). (unit test enforces; smoke outputs verified clean)

## 3. Feasibility & runtime
- [x] 3.1 ≤15 min/seed, ≤250k params (AGENTS.md §7). (param budgets asserted in tests; full-run estimate 5–8 min/seed — measured at smoke scale only, honest number = provenance wall_s)
- [x] 3.2 `--resume` support; progress bars + `progress.jsonl`. (completeness-verified resume for builder; per-method checkpoints for CNN-AE/Barlow/GRU)
- [x] 3.3 ONNXRuntime CPU for Stage-2 pixel inference. (ort 1.32.0 CPUExecutionProvider; torch/ORT parity test atol 1e-3)

## 4. Acceptance
- [ ] 4.1 All 4 methods run end-to-end on laptop. (smoke end-to-end incl. all four methods' code paths — barlow recorded method-skipped on the 3-date smoke store, no windows; full-scale = H-3)
- [x] 4.2 Anomaly score artifacts with provenance. (provenance.json: git sha, seed, n, date range, data_source, params, thresholds, wall, claim_rung)
- [x] 4.3 No claim beyond L2. (claim_rung recorded; unit test blocks "stress" wording)
- [x] 4.4 Reproducible (seeds fixed). (seed CLI arg, torch/np seeding, checkpoints carry seed)
