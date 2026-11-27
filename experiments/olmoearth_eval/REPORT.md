# OlmoEarth v1.2 Nano on the Sugarcane Archive — Evaluation Report

**Change:** `evaluate-olmoearth-nano-sugarcane` (openspec) · **Date:** 2026-11-27
**Machine:** owner's i7-12700H (14C/20T), **CPU only**, torch 2.15.0, `geo` env
**Validation ladder:** every claim below is **L1 or L2**. Nothing here is an
L3+ claim; injected deviations are synthetic; derived phase labels are derived.

---

## 1. Verdict: **ADAPT** (do not adopt as-is; do not reject)

| Gate | Criterion (pre-registered, design.md §6) | Result | Verdict |
|---|---|---|---|
| G-O1 usability | finite embeddings; ordering preserved; OE stage-probe > raw-channel probe, CI-separated | macro-F1 **0.854 ± 0.002** vs raw **0.627 ± 0.000** (Δ +0.227, separated); 17,307/17,307 finite; dates monotone per field-cycle | **PASS** |
| G-O2 phase consistency | cross-year phase-matched cosine > phase-mismatched, CI-separated | matched **0.799** [0.794, 0.804]; margin **+0.197** [0.192, 0.202]; positive for **219/219** fields | **PASS** |
| G-O3 detection ≥ B0 | OE AUC-PR ≥ B0 AUC-PR (CI-separated), injected protocol | overall OE best cell **0.079 ± 0.009** vs B0 **0.075 ± 0.003** (overlapping); per-severity S3: OE 0.063 vs B0 0.043; S1/S2 at base rate for all arms | **FAIL** (pre-registered criterion) |
| G-O4 robustness | reported, no gate | see §7 — robust to missing bins, sensitive to temporal coarsening | reported |
| G-O5 feasibility | full-store extrapolation ≤ 12 h CPU | **79 s** wall for the full 370-field store | **PASS** |

**Recommendation.** Adopt *the representation*, not *the detector*: the frozen
Nano encoder is an excellent **phenology-phase backbone** for this archive
(G-O1, G-O2 both decisive) at negligible CPU cost (G-O5), but its embeddings do
**not** beat the classical phase-matched spectral statistic for anomaly
detection under the pre-registered criterion (G-O3) — consistent with the
thesis's standing finding that NDVI/LST rules are strong competitors. The
honest use is **adaptation**: (a) embeddings as phase/seasonality conditioning
features for the statistical detector (not as the detector); (b) re-evaluation
on the date-centric raw store with true spatial context (the current patch
store is field-only — see §3.3, likely the binding limitation); (c) optionally
the 144×144 (10 m-equivalent) arm, which was the best probe cell (§7).

---

## 2. Provenance (all verified before any run)

| Artifact | Value |
|---|---|
| Model | OlmoEarth v1.2 Nano (`allenai/OlmoEarth-v1_2-Nano`), ViT-Nano, encoder **1,698,096 params** (checkpoint total 4,225,536 = encoder + decoder + target_encoder) |
| Weights | `olmoearth/weights.pth`, fp32, 17.04 MB, sha256 `2773fca48c238d78adde5e83b7d140a63d36c9e5f73746b8dbadaed743020378` (verified at load; mismatch aborts) |
| Config | `config.json` sha256 `4cd2888e79dc543f262cc3d86fcd30d667068fd53a728ca5bd306d5ddb509d1d` |
| Code | `github.com/allenai/olmoearth_pretrain` @ `6583a536c428ad004bb3c223add0576087d67e92`, vendored under `third_party/` (shallow clone, not committed) |
| License | **OlmoEarth Artifact License** (Ai2): use/reproduce/modify/distribute + derivatives permitted, subject to Ai2's use restrictions; thesis research use is within terms. License file ships with the vendored clone. |
| Deps installed into `geo` only | `einops 0.8.2`, `universal-pathlib 0.3.11`, `huggingface_hub 2.5.0` (repo's inference-only set; `olmo_core` NOT required) |
| Data read (read-only) | `data/processed/patches.zarr` (370 fields × 8 cycles × 36 bins × 8 ch × 48×48, fp16; provenance sha `2d0aa3f9…`, git sha `84184fe5…`); `data/processed/splits_locked_v2.json` sha `30ecbe10…`; epoch 2013-03-30 |
| Adapter git sha | recorded in `outputs/embeddings_v1_2_nano.zarr/provenance.json` |
| Splits used | train∩patches **280 fields** (probe/calibration population), dev∩patches **53 fields** (probe-test + injected-AD eval); **no ≥2024-01-01 claims** |

## 3. Compatibility findings and adapter decisions

### 3.1 The model side (verified from code + config)
- Native `landsat` modality: **single 11-band token group**
  `B8,B1,B2,B3,B4,B5,B6,B7,B9,B10,B11` (v1.2 config override), ingested in
  pre-training from **Landsat C2 Level-1 raw DN** (verified: computed.json
  stats, e.g. B2 μ=10493 σ=7872 ⇒ TOA reflectance ≈ DN×2e-5−0.1).
- Normalization: `(x − (μ−2σ)) / (4σ)` per band (`computed.json`).
- Temporal: T ≤ 12 real timestamps `(day, month−1, year)` → calendar-day RoPE
  (3D mixed) + month embedding; patch sizes 1–8 (flexible); trained with
  **random band dropout rate 0.2 on landsat** (channels zeroed post-norm) —
  i.e. the model was explicitly exposed to missing Landsat bands.
- Mask semantics: mask is per **patch-top-left pixel** (stride-8 sampling);
  MISSING tokens are emitted with non-zero content → pooling filters to
  ONLINE tokens only.

### 3.2 The band mapping (unit-tested, 25 tests)
| Slot | Source | Transform |
|---|---|---|
| B8, B1, B9, B11 | absent in L2 SR archive | post-norm fill **0.0** (trained band-dropout value); sensitivity arm fill 0.5 |
| B2..B7 | blue/green/red/nir/swir1/swir2 | SR → DN via inverse C2 L2 scale `(ρ+0.2)/2.75e-5`, then model norm |
| B10 | lst (Kelvin) | inverse Planck @ 10.9 µm → L1 radiance scale `(L−0.1)/3.3420e-4`, then norm |

- The SR→DN mapping lands our values within ≤13 % of the DN positions the
  model saw for equal reflectance over ρ∈[0.1,0.4] (design.md §3.1); the
  residual **L1-TOA vs L2-SR** processing-level gap is a documented domain
  shift (partially mitigated: the model also trained on S2 L2A surface
  reflectance).
- Thermal: model saw TOA brightness; we feed **L2 surface temperature**
  through the same radiance scale. Khuzestan LSTs (295–320 K) land at
  normalized ≈0.6–0.8 — inside the training range (unit-tested).
- Fill semantics split (after store-reality check, §3.3): whole-band absence
  → 0.0 (band-dropout value); per-pixel no-data inside a band with data
  → 0.5 (neutral band mean; the model has no trained per-pixel-missing
  semantics and mask granularity is the patch).

### 3.3 Store reality (material for any future adoption)
- Fields occupy only **~9 % of the 48×48 chip** (214–230 px of 2304);
  **all outside-field pixels are NaN** — the patch store is a *field-isolated
  cutout with no spatial context*.
- `pixel_valid` is marked **only inside the field**; on the model's
  patch-top-left grid only **~3 of 36** patch tokens can be ONLINE per
  timestep. The embedding is therefore a mean of ~3 patch tokens — this is
  the honest consequence of the artifact, and the most likely binding
  limitation for detection performance (§6).
- Bin acceptance: `n_obs>0` AND in-field QA-valid ≥ 30 % AND ≥ 1 online
  patch-top-left grid point (mechanical floor; fraction thresholds are
  unreachable by construction on this store).

## 4. Extraction (deliverable 3)

- **Full default-arm run completed in-session: 79.1 s wall** (22.8 ms/window
  forward), producing **17,307 observation embeddings** (128-d, fp32) over
  **1,741 windows** / **799 farm-cycles** / 370 fields; zero NaN; dates and
  phenological ordering preserved (`field_id, cycle_index, bin_index,
  day_index, date, month, year, sensor, n_obs, valid_frac`).
- Artifacts (`outputs/`): `embeddings_v1_2_nano.{zarr,parquet}` (obs level),
  `farmcycle_v1_2_nano.parquet` (farm-cycle level, mean of obs),
  `provenance.json` (shas, band map, seeds, arm, epoch), per-field
  crash-safe shards + manifest/`progress.jsonl` (resume).
- Robustness arms (seeded 120-field subset, seed 2026): fill05, b11dup,
  up144, drop1/2/4/6, rebin30/60 — same schema, separate suffixes.

## 5. Representation results (E1–E3; L1)

- **E1 (PCA, figures `outputs/figures/`):** PC1+PC2 (38.8 % var) trace a
  smooth **annual phenological ring**; months and derived phases both arrange
  continuously around it; per-field trajectories follow cycle order.
- **E2 cross-year consistency:** phase-matched cross-year same-field cosine
  **0.799** [0.794, 0.804] (bootstrap CI over 219 fields); phase-matched −
  phase-mismatched margin **+0.197** [0.192, 0.202], positive for **219/219**
  fields. The phase structure of the embedding is year-stable.
- **E3 kNN purity (k=5, n=8000):** same-field 0.095 (≈35× chance ≈0.0027),
  same-field-different-year 0.040. Field identity is weakly encoded; the
  space is organized by **phase/state**, not by field — consistent with the
  ring geometry and harmless for phase-conditioning use, but it means the
  embedding is not a field-fingerprint.

## 6. Anomaly detection (E5; L2, synthetic injections)

Protocol: stress-like deviations injected into dev-field bins
(nir×(1−δ), red×(1+δ), lst+Δ; S1/S2/S3 = δ 0.08/0.15/0.25, Δ 1.5/3/6 K);
105 injected bins vs 2,278 dev bins (base rate 0.046); scores calibrated on
train fields only; 3 injection seeds {13, 42, 2026}; AUC-PR primary.

| Score (phase-matched) | AUC-PR (mean±std) | recall @ train-95 % thr | FAR on clean dev @ that thr | recall @ dev 5 % FAR |
|---|---|---|---|---|
| OE_emb_meanabsz | **0.078 ± 0.001** | 0.152 | 0.077 | 0.102 |
| OE_knn_dist (k=20) | 0.079 ± 0.009 | 0.749 | **0.590** | **0.130** |
| OE_emb_maxabsz | 0.061 ± 0.003 | 0.197 | 0.174 | 0.073 |
| B0 raw-channel max-abs-z | 0.075 ± 0.003 | 0.083 | 0.054 | 0.079 |
| B1 NDVI persistence | 0.057 ± 0.002 | 0.054 | 0.046 | 0.054 |

Findings (all honest):
1. **No overall advantage over the classical statistic** (pre-registered
   G-O3 criterion fails): OE mean-abs-z 0.078 vs B0 0.075 with overlapping
   seed stds.
2. **Severity gradient:** at S3, OE mean-abs-z 0.063 > B0 0.043; at S1/S2
   every arm sits at the base rate — the small injections are undetectable
   for everyone (consistent with the thesis's severity analyses: real stress
   sits at 0.5–2.5×MAD while detectors floor at ~6×MAD).
3. **Threshold-transfer asymmetry:** B0's train-calibrated threshold
   transfers cleanly (FAR 0.054 on clean dev); the embedding scores transfer
   badly (OE knn FAR 0.59, OE maxabsz 0.17) — a real train↔dev distribution
   shift in embedding space (dev is a different k-means block).
4. **Not redundant:** flag-set overlap on real pre-2024 dev bins at a 2 %
   budget — OE∩B0 Jaccard 0.115, OE∩B1 0.0, B0∩B1 0.0. The embedding
   detector flags a *different* population than the spectral rules; without
   L4 evidence we cannot say better or worse — only different.

## 7. Robustness (E6; seeded 120-field subset, seed 2026; subset probe baseline 0.870)

| Arm | n obs | cosine to full | probe macro-F1 | ΔF1 |
|---|---|---|---|---|
| fill05 (band-fill 0.5) | 6,358 | 0.892 | 0.879 | +0.010 |
| b11dup (B11←B10) | 6,358 | **0.9998** | 0.871 | +0.001 |
| up144 (10 m-equivalent) | 6,358 | 0.715 | **0.889** | **+0.020** |
| drop1 bins | 6,062 | 0.990 | 0.889 | +0.019 |
| drop2 | 5,767 | 0.981 | 0.871 | +0.001 |
| drop4 | 5,176 | 0.966 | 0.859 | −0.011 |
| drop6 | 4,592 | 0.955 | 0.866 | −0.004 |
| rebin30 | 3,170 | 0.792 | 0.735 | **−0.135** |
| rebin60 | 1,575 | 0.688 | 0.840 | −0.029 |

- **Missing observations: negligible impact** (drop 6 bins → ΔF1 −0.004).
- Fill choice moves embedding geometry (cos 0.89) but not downstream phase
  content; B11 content is irrelevant (cos 0.9998).
- **Temporal coarsening hurts** (rebin30 −0.135): the phase information lives
  in the ~15-day sampling density — the model tolerates irregular dates but
  not coarser ones.
- **up144 (bilinear to 10 m-equivalent grid) is the best probe cell** despite
  very different raw geometry (cos 0.715) — a concrete adaptation lead.

## 8. CPU performance (E7)

- Encoder 1.7 M params; weights 17.04 MB; **peak RSS 1.35 GB** (probe incl.
  Python/torch baseline).
- Latency (48×48 chip, T=3): **4.4 ms/window** batched (B=8), 3.5 ms (B=32);
  T=6 23.8 ms single-window; 144×144 ≈ 68–78 ms/window.
- Full 370-field store: **79 s wall** end-to-end (I/O + forward); a future
  490-field store projects to ≈ 2–3 min. G-O5 trivially met.

## 9. Limitations and failure cases (read before citing any number)

1. **Field-only chips** (~9 % of the tile, no context, ~3/36 online patch
   tokens) — the dominant structural limitation; the date-centric raw store
   (openspec `date-centric-raw-store`) would remove it.
2. **Processing-level gap:** model trained on L1 TOA DN; we feed L2 SR/LST
   mapped into the L1 DN scale (≤13 % band-position deviation over the
   vegetation range; documented, not eliminable without TOA data).
3. **4/11 landsat slots absent** (B8 pan, B1 coastal, B9 cirrus, B11): value
   fills, not measurements; b11dup arm shows B11 content is irrelevant, and
   band-dropout training makes the 0.0-fill partially in-distribution — but
   the token is still a 7-of-11 observation the model never saw exactly.
4. **Spatial scale mismatch:** native 30 m pixels fed on a 10 m grid (the
   up144 arm approximates the training GSD and probes slightly better).
5. **Sampling mismatch:** pre-training used monthly timesteps; we feed
   15-day bins (≤12/window). Robust to missing bins; harmed by coarsening.
6. **Derived labels and synthetic injections:** phase classes are derived
   from cycle bounds (percent-of-cycle), never validated ground truth; AD
   numbers are L2 evidence on synthetic deviations only.
7. **torch 2.15 vs the repo's `<2.10` training pin:** inference-only path
   verified by probes + smoke; no numerical anomaly observed; recorded here
   for reproducibility.
8. **Single machine, no seed spread on extraction** (deterministic encoder;
   injection/probe seeds covered). No GPU claims anywhere.

## 10. Reproducibility

```bash
# geo env; repo root = thesis_implementation/
/home/mahdi/micromamba/envs/geo/bin/python -X utf8 -m pytest \
  tests/unit/test_olmoearth_eval_adapter.py -q                 # 25 passed
/home/mahdi/micromamba/envs/geo/bin/python -X utf8 \
  experiments/olmoearth_eval/extract_embeddings.py --smoke     # end-to-end gate
/home/mahdi/micromamba/envs/geo/bin/python -X utf8 \
  experiments/olmoearth_eval/extract_embeddings.py             # 79 s, 370 fields
/home/mahdi/micromamba/envs/geo/bin/python -X utf8 \
  experiments/olmoearth_eval/analyze_representations.py        # E1-E3 + figures
/home/mahdi/micromamba/envs/geo/bin/python -X utf8 \
  experiments/olmoearth_eval/probe_stage.py                    # E4
/home/mahdi/micromamba/envs/geo/bin/python -X utf8 \
  experiments/olmoearth_eval/detect_anomalies.py               # E5 (~3 min)
/home/mahdi/micromamba/envs/geo/bin/python -X utf8 \
  experiments/olmoearth_eval/robustness.py                     # E6 (~11 min)
/home/mahdi/micromamba/envs/geo/bin/python -X utf8 \
  experiments/olmoearth_eval/benchmark_perf.py                 # E7
# third_party: git clone --depth 1 https://github.com/allenai/olmoearth_pretrain
#   → checkout 6583a536c428ad004bb3c223add0576087d67e92 (not committed here)
```

Machine-readable tables: `outputs/tables/{representations,probe_stage,
detect_anomalies,robustness,benchmark_perf}.json`. Embedding binaries are
regenerable in < 6 min (gitignored; inventory in `outputs/README.md`).

## 11. Amendment 2026-11-27 — owner scientific review and follow-up validation

The owner's review accepted the ADAPT direction and the reproducibility
standard, and raised four issues before thesis use. They are resolved (or
gated) in the follow-up openspec change `validate-olmoearth-temporal-transfer`;
the two runnable batteries were executed the same day:

1. **Probe temporal confound — rejected.** With field-level bootstrap
   (1,000 resamples over the 53 evaluation fields; the seed-subsample std of
   §5.2 is NOT a population CI and is superseded here): timestamp-only probe
   0.613 [0.572, 0.654], raw-only 0.629, raw+temporal 0.665, OlmoEarth
   0.854 [0.827, 0.876]. Paired deltas: OE−timestamp-only **+0.239
   [0.200, 0.279]**, OE−raw+temporal **+0.188 [0.154, 0.225]** (both
   P(δ>0)=1.000). The §5.2 claim survives in a stronger form: the embeddings
   carry phase information beyond raw channels, beyond calendar time, and
   beyond their naive concatenation. The derived labels measure **cycle
   progression** — never read them as validated biological stages.
2. **Unseen-year generalization — holds.** Train = 268 fields / 10,226 bins
   pre-2024; evaluation = 2024–2026 bins of the k-means spatial block fields
   (73 fields, zero train overlap; the flat `test_fields` overlap train by
   design and are NOT a spatial holdout — caught by the leakage assertion).
   Pooled ≥2024: OE 0.825 [0.776, 0.867] vs raw 0.537 → Δ+0.289 [0.242,
   0.336]. Only −0.03 vs the in-sample probe. Calendar time degrades
   sharply in holdout years (0.613→0.367 in 2025) while the embedding
   degrades least — the representation tracks harvest-calendar drift.
   2026 (5 bins) is not interpretable.
3. **L2→L1 domain shift — data-gated.** The ≤13 % figure in §3.2 describes
   DN-scale distance only, not the SR-vs-TOA / LST-vs-brightness semantic
   gap. All 607 local scenes are L2-only; T3 requires same-date public C2
   L1 scenes (fetch command specified in the follow-up change) — owner-run.
4. **AD with real spatial context — deferred** to the date-centric raw
   store (`date-centric-raw-store`, HELD).

**Interpretation verdict (owner's 3-way question):** the evidence supports
**(b) transferable agricultural representation** — more than a temporal
encoder (T1), robust across unseen years and spatial blocks (T2); anomaly-
detection added value remains unproven (G-O3 FAIL; T4 deferred). The ADAPT
verdict of §1 stands unchanged. Tables:
`outputs/tables/temporal_transfer_{probe_arms,holdout_years}.json`.
