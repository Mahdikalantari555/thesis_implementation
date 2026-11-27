# Design — evaluate-olmoearth-nano-sugarcane

Status: proposed (2026-11-27). All facts below were verified against the
official repository and the shipped checkpoint by direct inspection + running
probes on this machine (2026-11-27).

## 1. Verified model facts

| Item | Value (verified) |
|---|---|
| Model | OlmoEarth v1.2 Nano (`allenai/OlmoEarth-v1_2-Nano`, HF) |
| Weights | `olmoearth/weights.pth` (owner download), sha256 `2773fca48c238d78adde5e83b7d140a63d36c9e5f73746b8dbadaed743020378`, fp32, 4,225,536 params total = encoder 1,698,096 + decoder + target_encoder (SSL checkpoint) |
| Config | `config.json` sha256 `4cd2888e…` from HF (copied to `model_store/`) |
| Code | `olmoearth_pretrain` cloned @ `6583a536c428ad004bb3c223add0576087d67e92` into `experiments/olmoearth_eval/third_party/` (shallow) |
| License | "OlmoEarth Artifact License" (Ai2): use/modify/derivatives permitted; Ai2 responsible-use restrictions apply. Recorded in the report; thesis use (research, non-competing) is within terms. |
| Architecture | FlexiViT encoder: depth 4, dim 128, 8 heads, MLP 512, RoPE 3D mixed (t,x,y), month embedding (12×32), max_sequence_length 12, patch sizes 1–8 (flexible at inference), single-band-group tokenization for `landsat` = `['B8','B1','B2','B3','B4','B5','B6','B7','B9','B10','B11']` |
| Training data | Landsat Collection 2 **Level-1** (all 11 bands, raw DN), Sentinel-2 L2A, Sentinel-1, + 6 static maps; random band dropout rate 0.2 on `landsat` and `sentinel2_l2a` (bands zeroed **after normalization**, pre patch-embed) |
| Normalization | `data/norm_configs/computed.json`, per band: `(x − (μ−2σ)) / (4σ)` (`Strategy.COMPUTED`, `std_multiplier=2`). Landsat μ/σ computed over raw L1 DN globally (e.g. B2 μ=10493 σ=7872; B10 μ=21347 σ=10957) |
| Inference API | `MaskedOlmoEarthSample(landsat=B×H×W×T×11, landsat_mask=B×H×W×T×1, timestamps=B×T×3 (day, month−1, year))`; `model.encoder(sample, fast_pass=True, patch_size=8)` → `tokens_and_masks.landsat` shape `(B,H',W',T,1,128)`; MISSING tokens are emitted with mask value 3.0 and non-zero content → pooling must filter to mask == 0 (ONLINE) |
| Deps (inference-only) | `einops`, `numpy`, `torch`, `universal-pathlib`, `huggingface_hub`; `olmo_core` NOT required. Installed into `geo` (torch 2.15.0 is newer than the repo's `<2.10` training pin — accepted for inference-only, verified by probe + smoke; any deviation recorded in the report) |
| CPU performance (probe, i7-12700H) | load 1.2 s; forward 0.099 s for 1×48×48×T=3, patch 8 |

## 2. Data source and semantics (verified)

`data/processed/patches.zarr` (370 fields × 8 cycles × 36 bins × 8 channels ×
48×48 px, float16):

- `channel_order = [ndvi, blue, green, red, nir, swir1, swir2, lst]`,
  `band_scale = auto` → SR bands are **physical reflectance** (0–1.2),
  `lst` is **Kelvin** (C2 L2 `apply_scale_st(0.00341802, 149.0)` — same code
  path as `ssl_anomaly.io.landsat`).
- Bins are **15-day** (`BIN_DAYS=15, N_BINS=36`) per-pixel **median** over
  QA-valid acquisitions inside the bin; `n_obs[f,c,b]` = acquisitions
  aggregated (0 ⇒ empty bin); `pixel_valid` per-pixel QA mask;
  `sensor_id` L8=1/L9=2; `cycle_start_day/cycle_end_day` day offsets from
  epoch **2013-03-30** (locked `splits_locked_v2.base_date`).
- QA: `qa_bits = fill, dilated_cloud, cirrus, cloud, cloud_shadow, snow(…)` —
  clouds already excluded upstream.

Splits: `splits_locked_v2.json` (sha `30ecbe10…`). Probing/population
statistics use **train∩patches** fields only; injected-anomaly evaluation uses
**dev∩patches** fields only; **no test-period (≥2024-01-01) claims**.

## 3. Compatibility analysis → adapter decisions

### 3.1 Modality choice: `landsat` (primary)

The model's `landsat` branch is the designed input for Landsat imagery. Our L2
surface-reflectance values differ from the L1 TOA DN the model saw, but the
affine scale maps nearly coincide over the reflectance range that matters:
`DN_L2 = (ρ+0.2)/2.75e-5` vs `DN_L1 = (ρ_TOA+0.1)/2e-5` differ by ≤13 % for
ρ ∈ [0.1, 0.4] (at ρ=0.1: 10909 vs 10000; ρ=0.2: 14545 vs 15000), and their
global DN means (B4 10236, B5 14427) sit almost exactly where our SR-inverted
DNs land (red SR 0.08 → 10182; NIR SR 0.20 → 14545). The residual TOA-vs-SR
gap is a *documented domain shift*, partially mitigated because the model also
trained on S2 L2A (surface reflectance, same semantics as our SR).

Alternative S2-mapping arm (mapping our 6 SR bands onto S2 band slots) is
recorded but not run by default: it injects *wrong-band* normalization stats
(red-edge bands absent) — a larger semantic mismatch than the level gap.

### 3.2 Channel assembly (landsat band order `B8,B1,B2,B3,B4,B5,B6,B7,B9,B10,B11`)

| Slot | Source | Mapping |
|---|---|---|
| B8 (pan) | absent in L2 SR | post-norm fill 0.0 |
| B1 (coastal) | absent in our chips | post-norm fill 0.0 |
| B2 | `blue` | DN = (ρ+0.2)/2.75e-5, then computed.json norm |
| B3 | `green` | same |
| B4 | `red` | same |
| B5 | `nir` | same |
| B6 | `swir1` | same |
| B7 | `swir2` | same |
| B9 (cirrus) | absent in L2 SR | post-norm fill 0.0 |
| B10 | `lst` (Kelvin) | inverse Planck @ λ=10.9 µm → radiance L, DN = (L−0.1)/3.3420e-4 (L8 C2 L1 RADIANCE_MULT/ADD), then norm |
| B11 (TIRS2) | absent | post-norm fill 0.0 (sensitivity arm: duplicate B10) |

Fill value rationale: band dropout zeroes channels **after normalization**
during training → 0.0 (normalized) is the trained "band-missing" value; the
alternative 0.5 (band mean) is a sensitivity arm. Missing-slot fraction = 4/11
bands, but the token itself is one 11-band vector — the model must tolerate
this via its band-dropout training; the report quantifies the effect.

Thermal semantics gap: model saw L1 B10 **TOA brightness** radiance; we feed
L2 **surface temperature** converted through the same radiance scale. LST ≳
BT for clear sky (a few K), so our summer LSTs (300–320 K → DN ≈ 31–37 k)
land at normalized 0.66–0.80, inside the training range (normalized [0,1] ⇔
DN ∈ [−429, 43262] ⇔ BT ≈ [220, 325] K). Documented; quantified via the
probe/AD deltas vs raw-channel baselines.

### 3.3 Spatial handling

- Primary arm: native 48×48 chip interpreted on the model's grid as-is
  (effective GSD 30 m vs training 10 m — a *scale* domain shift).
- Comparison arm: bilinear upsample 48→144 px (10 m-equivalent grid) to test
  whether matching the training GSD matters (token count 18×18×T — still
  trivial for the CPU).
- Per-pixel cloud gaps: `pixel_valid=0` ⇒ `landsat_mask=MISSING` (value 3) at
  those pixel positions; bins with `n_obs==0` are excluded from windows
  entirely (no garbage timesteps).

### 3.4 Temporal windowing

- Valid bins per field-cycle: `n_obs>0` and ≥30 % of in-field pixels QA-valid
  (configurable). Sorted by bin index (phenological order preserved).
- One window per ≤12 consecutive valid bins (`max_sequence_length=12`);
  overflow cycles are covered by additional windows (stride 12, no overlap;
  farm-level aggregation pools across windows).
- Timestamps: bin center date = `epoch + cycle_start_day + b*15 + 7`, passed
  as `(day, month−1, year)`. Real calendar dates → temporal RoPE + month
  embedding operate exactly as trained.

## 4. Embedding artifacts

`experiments/olmoearth_eval/outputs/embeddings_v1_2_nano.zarr` +
`.parquet` companion:

- **Observation level:** one row per (field, cycle, window, bin) —
  128-d pooled embedding (mean over ONLINE spatial tokens × band sets),
  plus `field_id, cycle_id, ratoon_no, bin_index, day_index, date_iso,
  month, year, sensor, n_obs, valid_frac, window_index, t_in_window,
  patch_size, arm`.
- **Farm-cycle level:** one row per (field, cycle, arm) — mean of its
  observation embeddings + `n_windows`.
- Farm ID, timestamps and phenological ordering (cycle_id, ratoon_no, bin
  index, day index) are preserved columns; no re-sorting that would break
  ordering semantics.
- fp32 storage; `provenance.json` in the zarr root (weights sha, config sha,
  code commit, adapter git sha, normalization strategy, band map, seeds,
  `data_source`, created-at).

## 5. Evaluation protocols

### 5.1 Representation analysis (E1–E3)

- PCA (and UMAP if `umap-learn` available, else PCA-only) of observation
  embeddings colored by month / phase class / field; per-field trajectory
  plots ordered by bin day (phenological ordering sanity).
- **Phenological consistency across years:** for each pair of cycles of the
  same field in different years, phase-matched embedding cosine similarity vs
  phase-mismatched similarity within the same year; margin with bootstrap CI
  (1000 resamples, seed fixed).
- **Cross-year / cross-farm similarity:** kNN purity of "same field" and
  "same cycle-year" from embedding distances; distance matrices for a
  rendered subset figure.

### 5.2 Linear probe on phenological stage (E4)

- Derived stage labels (explicitly *derived reference labels* from harvest
  dates + cycle bounds in `cycles_490_v2.parquet`, never "ground truth"):
  4 classes by days-since-cycle-start relative to `length_days`:
  establishment <25 %, tillering/elongation 25–50 %, grand growth 50–80 %,
  maturation >80 % (bounded, deterministic formula recorded in code).
- Probe: logistic regression (sklearn) on frozen embeddings, train fields
  only, stratified split by **field** (spatial split — no neighbouring-field
  leakage across train/test), 3 seeds {13, 42, 2026}, macro-F1 + accuracy
  (mean ± std).
- **Mandatory control:** identical probe on raw-channel features
  (7 channel means + ndvi per bin, the information a statistical baseline
  could use). The claim "embeddings add value" requires OE > raw with
  non-overlapping seed CIs.

### 5.3 Anomaly detection (E5)

Controlled-deviation protocol on **dev∩patches fields** (clearly labelled
L2; injected deviations are synthetic stress-like signatures, never presented
as real stress):

- Injection: choose random bins in clean dev cycles; degrade NDVI by
  δ ∈ {−0.08, −0.15, −0.25} (S1/S2/S3-magnitude, consistent with the
  thesis's ×MAD severities) applied multiplicatively to red/NIR, and raise
  LST by Δ ∈ {+1.5, +3, +6} K. 3 injection seeds; injected bins are scored,
  untouched bins are negatives (imbalance recorded; **AUC-PR**, plus
  recall@FAR=5 %).
- Scores, all calibrated on train fields only (no dev leakage):
  - **OE-emb:** per-dim robust z of the bin embedding vs phase-matched train
    population, channel-max statistic (mirrors E1 finding); also mean-|z|.
  - **B0 spectral:** same robust-z statistic computed on the raw 7-channel
    bin means (the classical spectral baseline).
  - **B1 persistence:** NDVI drop vs the field's own rolling median of the
    3 preceding valid bins (phase-free classical control).
- Real-data concordance (no synthetic data): score all pre-2024 real bins;
  report overlap with the existing `1,437 suspected_stress` flags artifact
  (recomputed on the patch-store subset) and top-K flag listing — labelled
  as *concordance evidence*, not detection accuracy.

### 5.4 Robustness (E6)

- Missing observations: drop k ∈ {1,2,4,6} random bins from extraction
  windows; metric = embedding cosine drift vs full-window + probe macro-F1
  delta.
- Temporal density: re-binned coarser sampling (30-day and 60-day medians);
  same metrics.
- Fill sensitivity: B-fill 0.0 (default) vs 0.5 vs B11=B10.
- Spatial scale: native 48×48 vs bilinear 144×144.
Each arm recorded with its exact config; comparisons against the default arm
under identical seeds.

### 5.5 CPU performance (E7)

Benchmark: per-window forward latency (T ∈ {3,6,12}; 48×48 and 144×144),
batch throughput, peak RSS (measured via `resource.getrusage`), model size on
disk, embedding volume per field. Extrapolated full-store cost (370 fields ×
valid windows) with measured rate; reported with wall-time budget for the
490-field v2 future store.

## 6. Gates (pre-registered before any full run)

| Gate | Criterion (all with seed-CI, not point estimates) |
|---|---|
| G-O1 usability | embeddings finite (zero NaN), temporal ordering preserved; OE stage-probe macro-F1 > raw-channel probe with CI separation |
| G-O2 phase consistency | phase-matched cross-year cosine > phase-mismatched same-year cosine, CI-separated margin |
| G-O3 detection | OE-emb AUC-PR ≥ B0 spectral AUC-PR (CI-separated) on the injected protocol; otherwise the honest verdict is "no added value over raw channels" |
| G-O4 robustness | quantified drift table; no gate — reported (informative for adopt/adapt) |
| G-O5 feasibility | measured full-store extrapolation ≤ 12 h CPU on this machine |

Verdict mapping: **adopt** = G-O1+G-O2+G-O3+G-O5 pass; **adapt** = G-O1 (and
G-O5) pass but G-O2 or G-O3 fail — the representation is usable but not
better than raw channels as-is; **reject** = G-O1 fails or feasibility fails.
Every conclusion sentence states its ladder rung (L1/L2 only).

## 7. Failure semantics and reporting

- Any missing input (checkpoint hash mismatch, store absent, empty split
  intersection) → hard error, never a silent fallback (AGENTS.md §2/§8).
- Injected deviations and derived stage labels are labelled as such in every
  table and the report; no "anomaly = stress" naming before L4.
- Report (`REPORT.md`) carries: data_source, n, date range, seeds, device,
  wall time, git sha, all artifact shas, per-gate verdicts, limitations
  (incl. TOA-vs-SR gap, 4/11 missing bands, GSD mismatch, license), and the
  final recommendation.
- Long runs (full-store extraction if > 2 min measured) are handed to the
  owner as exact commands after unit tests + `--smoke` pass (AGENTS.md §7).
