# Design — validate-olmoearth-temporal-transfer

Status: applied for T1/T2 (2026-11-27); T3 data-gated; T4 deferred.

## 1. Shared infrastructure (all read-only, verified)

- Embeddings: `experiments/olmoearth_eval/outputs/embeddings_v1_2_nano.parquet`
  (17,307 obs; default arm of the parent change; provenance in the zarr).
- Raw features: `outputs/raw_bin_features.parquet` (same bins/pixels).
- Splits: `splits_locked_v2.json` sha `30ecbe10…`; patch-store intersection:
  train 297 / dev 53 / test 270 fields.
- Holdout cut: 2024-01-01 = day **3929** from epoch 2013-03-30. Verified
  coverage (attrition chain, owner amendment round 2): **S0** 4,943 bins /
  204 fields (whole store ≥2024) → **S1** 4,740 bins / 195 fields (flat
  dev+test — not a spatial holdout) → **S2** 965 bins / 42 fields (the 73
  k-means block fields; 31 have no ≥2024 coverage) → **S3** 965 bins /
  42 fields (raw-feature join lossless). The evaluation pool is S3;
  per-year fields: 2024: 42 · 2025: 24 · 2026: 1.
- Cycle bounds come from `farmcycle_v1_2_nano.parquet` (store-native;
  `cycle_frac` = (day − cycle_start)/(cycle_end − cycle_start), clipped [0,1]).

## 2. T1 — probe-arm battery (exact feature definitions)

| Arm | Features | Note |
|---|---|---|
| `timestamp_only` | month sin/cos (2), day-of-year sin/cos (2) | pure calendar; no cycle/harvest info |
| `cycle_progress_only` | cycle_frac, sin(2π·frac), cos(2π·frac) | the label-defining quantity — oracle-flavoured control, never used to claim representation quality |
| `raw_only` | 8 bin-mean channels (ndvi..lst) | the parent change's control |
| `raw_plus_temporal` | 8 channels + month sin/cos + doy sin/cos | cycle_frac deliberately excluded (label-defining); recorded |
| `olmoearth` | 128-d frozen embedding | the model arm (contains month/RoPE time information — this is exactly the confound T1 measures) |

Classifier: `StandardScaler → LogisticRegression(C=1.0, max_iter=2000)`,
identical across arms. Split: train = train-fields bins (pre-2024 restriction
NOT applied in T1 — T1 measures the confound on the parent's own protocol;
T2 applies the year discipline), evaluation = dev-fields bins.

Statistics: **field-level bootstrap** — 1,000 resamples of the 53 dev fields
with replacement (seed 2026); per resample compute each arm's macro-F1 on the
resampled bins; report per-arm point estimate + 2.5/97.5 percentiles and
paired deltas (OE − arm) from the same resamples. The 3 training-subsample
seeds of the parent change are kept only as a robustness note, never as CIs.

Interpretation mapping (owner's 3-way question):
- OE ≤ timestamp_only ⇒ hypothesis (a): the probe value is calendar time.
- OE > timestamp_only and OE > raw+temporal, but AD still at B0 level ⇒ (b):
  transferable agricultural representation, not a detector.
- OE > timestamp_only AND detection ≥ B0 (only reachable via T3/T4
  follow-ups) ⇒ (c).

All transferability wording is scoped **within-region, within-crop**
(Khuzestan sugarcane; Landsat 8/9 only): the evaluation fields and years all
come from one agro-industry region, one crop and one sensor family — no
cross-region, cross-crop or cross-sensor generalization is claimed or
implied.

## 3. T2 — unseen-year generalization

- Training set: train-field bins with date < 2024-01-01 (asserted).
- Evaluation: bins ≥ 2024-01-01 of the **k-means spatial block fields**
  (test_block 51 + dev_block 46; 73 in the patch store; zero `train_fields`
  overlap — asserted at runtime). The flat `test_fields` list overlaps
  `train_fields` because the v2 protocol splits temporally at cycle level,
  so it is not a spatial holdout. No adaptation or selection on the eval
  bins (asserted + recorded).
- Arms: `olmoearth`, `raw_only`, `timestamp_only` (time control is essential:
  if the label degrades to calendar time in holdout years, that is part of
  the finding). cycle-progress-only is included as the oracle control.
- Metrics: macro-F1 (+ accuracy) per year and pooled; field-level bootstrap
  CIs; per-year n reported as bins AND fields; **low-n rule = <100 bins OR
  <30 independent fields** (2025 flagged on the field axis: 24 fields;
  2026 on both; unit-tested).
- Secondary margin (≥2024 cycles): ONE value per unique evaluation field
  (mean over its cycle pairs), field-level bootstrap 1,000×; with only 1
  evaluable pair among the block fields it is reported **not estimable** —
  never substituted by the pre-2024-dominated parent margin.
- Secondary: phase-consistency margin (E2 definition) restricted to cycles
  whose bins are ALL ≥ 2024 (no pre-2024 content leaks into the margin).
- Risks recorded in advance: harvest-calendar drift across years changes the
  frac↔phase mapping (this is part of what T2 measures, not a defect);
  2025/2026 cycles may be partially censored in the v1-era store.

## 4. T3 — L1 vs L2 measurement (data-gated)

- Local archive: 607 scene dirs are **L2-only** (`blue/green/red/nir08/
  swir16/swir22/lwir11/qa_pixel` + MTL.json). No L1 product exists locally.
- Fetch path (owner-run): C2 L1 is public on GCS —
  `gs://gcp-public-data-landsat/LC08/02/<path>/<row>/LC08_L1TP_<...>/` —
  no credentials. `download_l1.py` (specified, not executed) fetches the
  same-date L1 scenes for 3+ selected pairs: one peak-growth summer scene,
  one late-season/bare scene, one winter low-biomass scene, from 165-039 and
  166-038; owner approves storage (≈ 1–2 GB/scene unpacked → fetch only the
  needed bands: B1–B7 DN, B10 DN + VVCID/RAD coefficients from MTL).
- Measurement: per-band quantiles of TOA ρ (=DN·2e-5−0.1) vs SR; BT(Planck
  B10) vs LST deltas over land pixels; normalized-space distance
  ||norm(L1) − norm(L2-converted)|| per band; embedding drift for 5 sample
  fields fed through the true-L1 path vs the L2-converted path
  (cosine + probe F1 delta). All labelled as L1–L2 evidence about the
  *adapter*, not the thesis detector.

## 5. T4 — Deferred (blocked)

`date-centric-raw-store` is 0/28 and HELD by owner decision; T4 inherits that
block. No numbers, no re-runs on the field-only store.

## 6. Failure semantics

- Any assertion failure (leakage, missing artifact, hash mismatch) exits
  non-zero with the reason; no silent narrowing.
- Tables land in `outputs/tables/temporal_transfer_{probe_arms,holdout_years}.json`;
  the report amendment lives in the parent REPORT.md §11.
