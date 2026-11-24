# Change: farm-subset-anomaly-probe

**Change ID:** `farm-subset-anomaly-probe`
**Project:** Thesis Implementation
**Status:** Proposed
**Priority:** High (local, no Colab)
**Target hardware:** Owner laptop, CPU only (IRIS XE not usable for torch; ONNXRuntime CPU for inference)
**Created:** 2026-11-24

## Why

Colab is unavailable. We need **first anomaly-score outputs** on the owner's own machine. The `Farms_subset.geojson` (32 farms, R11) plus the existing v2 store and the (optional) date-centric raw store on a subset make a small-scale probe feasible on CPU (≤250k params, ≤15 min/seed per AGENTS.md §7). This change is a **consumer** of `date-centric-raw-store` (optional) and of `model_improvement_proposals_2026-11-22.md` (P1–P5).

## Feasibility assessment (verified 2026-11-24)
- torch 2.15 CPU-only; `cuda`/`xpu`/`mps` all False; no IPEX/OpenVINO. **IRIS XE cannot accelerate torch training.** ONNXRuntime 1.31 (CPU) is available for inference speed-up.
- 32 farms × whole time range (538 dates) × 48×48 is feasible for small models with batching; pixel-wise CNN on full 48×48 is heavy but tractable with patch sampling.
- Barlow Twins (ST-BT style) on pixel trajectories is cheap (4-layer FC, ~tens of thousands of params).

## What Changes
- **Methods (all on the 32-farm subset, full time range):**
  1. B2 no-NN baseline (phenology + date×neighbor) — σ from residual.
  2. Pixel-wise CNN autoencoder (small, ≤250k params) → reconstruction residual = anomaly score.
  3. Barlow Twins (ST-BT) on spectral-temporal pixel trajectories → representation distance = anomaly score.
  4. GRU-AE baseline (existing, on subset).
- **Output (hybrid, two stages):**
  - Stage 1: anomaly score per (field, date) for all 32 farms × 538 dates (field-date granularity) for triage.
  - Stage 2: for the top-K suspicious (field, date) pairs, pixel-level score on 48×48 (within-field stress map) — ONNXRuntime CPU for speed.
  - Persistence (≥2 obs, ≥5 d apart) + cold-LST gate (ΔLST < −2 °C rejected as cloud/shadow). Event-level framing (Q8).
- **Data input:** date-centric raw store on the 32-farm subset (`raw_satellite_subset.zarr`, D=538, F=32, K=7, S=48). If not ready, fall back to v2 cycle-centric subset (no full migration required for the probe).
- **Phenology note (v2 lesson):** cycle detection and phenology features must use **multi-band** input (NDVI + SAVI + LST + GDD), not NDVI alone. NDVI-only detection missed winter harvests and conflated seasons (18/19 long cycles). CWSI_ET is NOT needed (owner decision 2026-11-24). This applies to the B2 baseline's phenology component and any cycle-derived features.
- **IRIS XE:** document that training stays CPU; inference may use ONNXRuntime CPU. No GPU claim.

## Capabilities
- New capability: `farm-subset-anomaly-probe` (subset selection, anomaly scoring, persistence/gates, output artifacts).

## Impact
- **Affected (new):** `scripts/probe/`, `src/ssl_anomaly/probe/`, `reports/probe/`, config for farm subset + model list.
- **Frozen/untouched:** v2 store, splits, checkpoints, findings, platform, benchmark change.
- **Risks:** CPU runtime (≤15 min/seed); no ground truth (L4) — scores are L2 evidence only; "stress" wording not licensed.

## Prerequisites
- `Data/RoI/Farms_subset.geojson` (32 farms) — present.
- v2 store or date-centric subset store.
- B2 normalization fit on train split only (leakage rule).

## Acceptance
- All 4 methods run end-to-end on the 32-farm subset on the owner's laptop (≤15 min/seed, ≤250k params).
- Anomaly score artifacts produced per (field, date) with provenance (data_source, n, date range, seeds, git SHA).
- Persistence + cold-LST gates applied and reported.
- No claim beyond L2.
- Reproducible (seeds fixed, --resume supported).
