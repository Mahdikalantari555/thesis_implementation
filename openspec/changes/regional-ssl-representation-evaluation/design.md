# Design — regional-ssl-representation-evaluation

## Context

Thesis state at `2b479dc` (v2 chain, honest):

| Artifact | Value | Use |
|---|---|---|
| `cycles_490_v2.parquet` | 4,458 windows; 3,478 non-censored cycles; all 490 fields; median length 366 d | Cycle unit of analysis |
| `splits_locked_v2.json` | sha `30ecbe10…`; train 2,105 / dev 340 / test 773 / monitoring 1,240 | Split discipline (unchanged) |
| `holdout_manifest_v2.json` | sha `d501bf81…`; 37 holdout-excluded test-block fields | Excluded from every path here |
| `norm_stats_full.json` | sha `3cb1ca60…` (447 locked train cycles, 240 fields — PoC-era norm; v2 norm to be rebuilt/recorded) | Normalization, train-only |
| Findings 18–25 | MDM80 censored >S6; B1 ≥ SSL; GRU-Pred negative; L2 rung; L5 unmet | Why this change exists |

The advisor analysis (2026-11-23, Persian, relayed by the owner) contributes:
the two-level evaluation frame (Pretraining Quality / Downstream
Transferability), the three generalization levels, the B1–B5 ablation, the
five-advantage hypothesis table, the few-hundred-label transfer protocol, the
semantic ladder (anomaly vs stress-candidate vs stress), the regional
foundation-model reframing ("Learning Generalizable Spatiotemporal
Representations of Agricultural Field Dynamics through Self-Supervised
Learning" — adopted as research framing, **not** as an immediate title change),
and the explicit probability assessment that SSL may still lose on single tasks
while winning on transfer efficiency — both outcomes are reportable.

Data facts (DATA_SUMMARY.md + decomposition-aware-ssl-study): Amirkabir
agro-industrial unit; ~14,000 ha; 490 farm polygons (R01–R16, L02–L08 blocks);
Landsat 7/8/9 Collection-2 Level-2; 607 acquisition dates recorded 2013–2026
(539 METRIC products; 538 usable dates after QC — actual admissible counts are
smaller per split); ERA5-style weather (hourly + daily); METRIC products
(ETaDaily, CWSI_ET, CWSI_LST, ETinst, ETrF, daily interpolation, 4,375 rasters
each); 30 m resolution; EPSG:32639. Single crop (sugarcane; ratoon cycles
length 6–14 months). Farm-mean (zonal) series is the v0 input contract;
patch-level remains out of scope for this change.

## Semantic ladder and claim ceilings (binding)

| Term | Definition | Evidence required | Ceiling here |
|---|---|---|---|
| **Anomaly** | Deviation from the field's own phenology-aware normal at the same growth phase | Controlled injection + clean-cycle behaviour (FAR-matched) | L2 |
| **Stress candidate** | Dates whose observation pattern is *compatible* with anomalous growth (e.g. persistent NDVI drop + LST rise + ETa depression) | Above + proxy agreement, reported **separately** | L3 (proxy-corroborated) |
| **Stress** | Independently verified crop water/salinity stress | Independent agronomic evidence (company records, expert review, in-field measurement); **absent today** | Not claimable |

Rules: synthetic anomalies are controlled evidence, never "real stress";
CWSI is partly circular wherever LST is an input (METRIC CWSI_ET shares
energy-balance inputs, not LST — but ETa shares radiation/meteorology inputs);
ETa/ETrF/TVDI are proxies, not truth; segmentation-derived labels (DSH bins)
come from the cycle segmentation itself and are therefore data-derived, not
model-derived — acceptable for probing, reported as such; no label may be
fabricated — missing evidence is `absent`.

## The five hypotheses (from the advisor analysis, made falsifiable)

| ID | Hypothesis (verbatim intent) | Primary metric | Gate |
|---|---|---|---|
| H1 | SSL learns non-obvious temporal deviations that simple thresholds miss | T2 AUPRC vs B1/B1–B5 at S4–S6 | G-C |
| H2 | Multivariate spectral+thermal+weather fusion adds value over pairs | B3→B4→B5 deltas, paired bootstrap | G-B/G-C |
| H3 | Farm-specific modelling beats global thresholds | Per-field vs pooled calibration FAR/AUC | G-C |
| H4 | SSL detects gradual/short-term/multivariate anomalies missed by rules | Per-family cells (gradual, persistent, multivariate) | G-D |
| H5 | SSL generalizes more stably across unseen years/periods | Temporal roll Δ vs B1, seed spread | G-A/G-D |

Each hypothesis reports against every protocol cell, with effect sizes and
CIs; no point estimate alone supports a conclusion.

## Architecture decision gate (region model v0)

No architecture is pre-approved. The owner's constraint set:

- ≤250k parameters (target ≤100k); CPU-first; ≤15 min per seed on the owner's
  laptop; Colab escalation only after the local gate (AGENTS.md §6).
- Deterministic seeds {13, 42, 2026}; `--resume`; `progress.jsonl`; no NaN
  silent pass; normalization from train split only; mask-aware inputs.
- Interface: frozen-encoder `embed(x) -> z`, plus the existing scoring bridge
  so any candidate can feed the unchanged downstream evaluation.

Candidate list (screening order, all capacity-matched, equal budgets):

1. **C-incumbent:** GRU-AE trunk (8 ch + DOY) reused as a deterministic
   encoder (baseline-of-candidates; zero new architecture risk).
2. **C-BT:** Temporal Barlow Twins on the same trunk (invariance +
   redundancy-reduction objective; no reconstruction head at pretrain time;
   views = pre-registered, phenology-safe augmentations — no inversion of
   anomaly polarity; augmentation strength recorded). Literature: Barlow Twins
   (Zbontar et al., 2021; PMLR 139); Spectral-Temporal Barlow Twins for cloud-
   corrupted satellite time series (2024, Cambridge repository). Augmentation
   sensitivity is a reported axis, not a nuisance.
3. **C-traj:** Self-supervised trajectory predictor (predict bins > t from ≤ t;
   no yield labels; targets are the future observations themselves). This is
   the advisor's "crop trajectory without yield labels" route.
4. **C-hybrid:** Barlow Twins + reconstruction/prediction head
   (`L = L_BT + α·L_rec`, α pre-registered; joint objective, shared encoder).
5. **C-decomp:** Decomposition-conditioned encoder (features from
   `decomposition-aware-ssl-study`: trend/seasonal/residual, STL family) — only
   if that study's ablation report supports the features.

Screening protocol (pre-registered, dev clean cycles only): three seeds, equal
optimizer budget, identical d_model within ±20 % capacity band, selection by
the G-A…G-E aggregates — **not** by lowest reconstruction loss. A candidate
that screens out fails loudly with recorded evidence. If **no** candidate
passes the gates, the capstone records the negative result; the thesis claim
collapses honestly to "B1 ≥ SSL on detection; SSL diagnostics reported
descriptively", consistent with vision.md's pre-committed fallback.

## Level 1 — Pretraining Quality

Executed on **dev clean cycles only**, 3 seeds, with Wilson/bootstrap CIs:

1. **Reconstruction/masked quality** — masked NRMSE per channel (observed
   entries only); reuses existing masking semantics (`specs/model-development-
   protocol`).
2. **Farm-identity separability** — frozen `embed`, kNN (k ∈ {1, 5}) top-1 /
   top-5 farm-classification accuracy vs chance (1/490, 5/490) and vs B1-feature
   kNN on identical splits. (490-way, so absolute numbers are low-variance but
   low-signal; report distribution, not a single threshold.)
3. **Phenological-phase probing** — linear probe (multinomial, train split
   only) predicting DSH bin (36 bins) from frozen z vs majority baseline and vs
   B1-feature probe; report macro-F1 and accuracy.
4. **Embedding temporal stability** — for each field with ≥2 complete clean
   cycles, cosine similarity of aggregate cycle embedding across cycle pairs;
   report median/IQR and the fraction of pairs below a pre-registered floor;
   compare against shuffled-null pairs.
5. **Collapse diagnostics** — per-dimension std spectrum, effective rank of the
   embedding covariance (threshold recorded), alignment/uniformity on clean
   dev; any degenerate candidate is disqualified at screening, not after tuning.

Deliverable: `reports/regional_representation/pretraining_quality.json` (+ MD).

## Level 2 — Downstream Transfer Evaluation

### Common protocol (binding for every arm and cell)

Arms per task:

| Arm | Definition |
|---|---|
| A-rule | FAR-matched NDVI+LST rule (see B1 below) — the advisor's control |
| A-B1 | Incumbent `b1_phasez_causal` (statistical, phase-aware) |
| A-rand | Random-init encoder, same trunk/capacity as the SSL candidate (capacity control) |
| A-scratch | Task-label-only small model (same trunk, init fresh) |
| A-ssl-frozen | Pretrained SSL encoder, frozen, + linear head |
| A-ssl-ft | Pretrained SSL encoder + fine-tuned head (equal budget to A-scratch) |

Label fractions: **5 / 10 / 25 / 50 / 100 %** of the labelled pool, stratified
by farm block and cycle phase; identical index subsets across arms per
fraction (same seeds); three runs per cell; mean ± std and bootstrap CIs.
Metrics per cell: AUC-PR (primary on imbalanced cells), AUROC (secondary),
F1 at validation-selected threshold, precision@k where applicable. Prevalence
reported in every cell; no macro-averaging that hides failing cells (per-cell
table mandatory).

### Tasks (label provenance is binding)

| Task | Operationalisation | Label source | Claim status |
|---|---|---|---|
| **T1 Phenology-stage estimation** | Predict DSH bin (36-way) from z | Cycle segmentation (data-derived, documented) | Claim-bearing (L2/L3 frame) |
| **T2 Limited-label anomaly detection** | Train heads on injection-labelled event pools; AUC-PR per family×severity | Synthetic injection (controlled) | Claim-bearing for controlled evidence only |
| **T3 Few-shot similar-date retrieval** | Query k ∈ {1, 3, 5, 10} labelled dates; rank archive dates by embedding similarity | Track 1: injection-labelled dates · Track 2: weak proxy dates (CWSI_ET / TVDI excursions, explicitly labelled weak) · Descriptive near-match listing of natural-data neighbours (owner decision 2026-11-23) | Track 1 claim-bearing (controlled); Track 2 + near-match listing exploratory |
| **T4 Self-supervised trajectory quality** | Next-1/3-bin forecasting error on clean held-out cycles vs persistence baseline | None (self-supervised) | Claim-bearing (L1 frame) |
| **T5 Crop-condition / irrigation-pattern analysis** | Descriptive clustering + nearest-date listings; ETa response shape | None (descriptive) | **Exploratory only — never claim-bearing** |

### T3 — the owner's stated next task (2026-11-23)

"آخرش اینه که اصلاً عمومیت مدل را بسنجیم، یک مدل پایه بسازیم از محدوده که خیلی
سبک باشد و خیلی سریع روی CPU اجرا شود، و بعد روی تسک‌های مختلف تستش کنیم، مثلاً
همین تنش؛ ببینیم با چند تا لیبل تاریخ، چه تاریخ‌های دیگری را به اسم مشابه
بالا می‌آورد."

Operationalisation (owner decision 2026-11-23: **both weak-proxy track and a
descriptive natural-data near-match listing are included**): with k labelled
(field, date) points, the retriever returns the top-m most similar archive
dates; m ∈ {1, 5, 10}; metrics: precision@m, mean average precision, and
(Track 1) anomaly-family hit rate. Baselines: B1-score ranking, NDVI+LST
rule-score ranking, raw band-vector Euclidean distance, A-rand embeddings. If
SSL retrieval does not beat raw-distance and B1-score ranking on Track 1, the
retrieval claim fails and is reported as such; Track 2 remains descriptive
regardless.

**Track 2 — weak-proxy (explicitly weak, exploratory):** query dates flagged
by CWSI_ET / TVDI excursions on natural dev cycles. Reported separately with
prevalence and the dependence structure (shared inputs with the model) stated;
never claim-bearing, never gate input.

**Near-match listing (descriptive):** for every query used in T3, the top-10
nearest NATURAL-data (unlabelled) archive dates are written to
`near_matches.md`/`near_matches.json` with similarity scores, field/date,
and the query it answered — for manual owner inspection only. This artifact is
explicitly `exploratory`; it SHALL NOT feed any gate, claim, threshold or
selection.

## B1–B5 incremental input ladder (the advisor's ablation)

Identical data, dates, anomaly definition, splits, calibration subset, FAR
budget (5 % cycle-FAR), aggregation and bootstrap for every rung:

| Rung | Inputs | Model | What it isolates |
|---|---|---|---|
| **B1** | NDVI + LST only | Rule: NDVI ≥ a·MAD_field drop **AND** LST ≥ a·MAD_field rise (pre-registered a; report AND-variant and OR-variant, neither selected post hoc), per-field robust thresholds, calibrated on the clean calibration subset | Two direct physical variables, zero learning |
| **B2** | NDVI + LST + mask + DOY | SSL encoder, identical trunk/budget as B3–B5 | Learning value on the same two variables |
| **B3** | All spectral+thermal channels (current 8-ch contract) + mask + DOY | Same | Value of full optical/thermal input |
| **B4** | B3 + daily ERA5 (t2m/tmean, precip, solar radiation, VPD, et0) | Same | Value of meteorological context |
| **B5** | B4 + phenological context (DSH sin/cos, phase bin, STL trend/seasonal/residual as in decomposition-study) | Same | Value of temporal/phenological conditioning |

Evaluation: the finding-18/25-compatible controlled protocol (S1–S6 severity;
acute/persistent; NDVI and SAVI families; eval subset hash recorded) on dev;
plus natural-data behaviour (cycle-FAR, event-FAR); plus the Temporal
generalization rolls (below). Every cell reports per-family × per-severity
results; MDM80 with Wilson CIs; AUPRC with prevalence.

**Reading rule (pre-registered):** if B2 ≈ B1, learning adds nothing on the
two physical variables; if B3 > B2, spectral breadth adds value; if B4 > B3,
weather adds value; if B5 > B4, decomposition context adds value. If none
clear B1 under CI, the honest thesis statement is that the simple algorithm
wins on detection at far lower cost — full stop.

## Generalization protocols

1. **Temporal:** year-forward rolls entirely inside train+dev — for cut
   years Y ∈ {2016, 2018, 2020, 2022} (exact set frozen in
   `configs/evaluation/temporal_rolls.yaml`), train on cycles starting ≤ Y
   (+ normalization refit), validate on the following 12–24 months. Report Δ
   metrics per Y vs B1 run identically. The locked test split is **never**
   read by these rolls.
2. **Spatial:** leave-one-block-out within train+dev using the 20 locked
   blocks (train blocks → val block roll); the 37 holdout-excluded test-block
   fields stay excluded everywhere. Report mean ± spread across rolls.
3. **Geographic:** out of scope — one region, one crop, one climate regime.
   Recorded as an explicit limitation and claim ceiling (L3 same-region). Any
   geographic claim requires a new data contract and is future work.

## Pre-registered decision gates

| Gate | Pass criterion (CI-supported) | Fail consequence |
|---|---|---|
| **G-A** Pretraining quality | SSL beats A-rand on ≥2 of {farm-separability, phase probe, frozen-head T1} AND no collapse criteria triggered | Candidate disqualified; capstone reports diagnostics only |
| **G-B** Label efficiency | A-ssl-frozen or A-ssl-ft matches A-scratch@100 % at ≤50 % label fraction on ≥1 claim-bearing task | "Sample efficiency" claim fails; recorded |
| **G-C** Transfer dominance | A-ssl arms beat A-rule and A-rand on ≥2 claim-bearing tasks at ≥2 label fractions each | "SSL learns superior representations" unsupported on this data |
| **G-D** Retrieval usefulness | A-ssl retrieval beats raw-distance, A-B1-score and A-rule-score ranking on T3 Track 1 at ≥3 of 4 k values | Retrieval claim fails; SSL limited to "descriptive embeddings" |
| **G-E** Claim discipline | Every reported claim carries rung + provenance; negative results published with equal prominence | Report rejected at review; change not closable |

All-gates-fail outcome is explicitly allowed and pre-declared valid: the
thesis then reports the negative result, keeps B1 as the reference algorithm,
and frames SSL's status as open for regions/data with different structure.
No gate may be weakened after results are seen (no motivated reasoning).

## Reproducibility, compute placement, artifacts

- Seeds {13, 42, 2026} everywhere; CPU auto-detect, CUDA used when present;
  laptop budget ≤15 min/seed, ≤250k params (target ≤100k).
- Hand-over rule unchanged: anything > ~2 min is an exact owner command
  (conda env, cwd, flags, expected wall time, outputs); agent runs unit tests +
  `--smoke` first and reads the smoke output; agent never launches long jobs.
- Artifacts: `reports/regional_representation/` — `pretraining_quality.json`,
  `transfer_curves.json`, `retrieval_benchmark.json`, `b1_to_b5_ladder.json`,
  `generalization.json`, `decision_gates.json`, `capstone_report.md` (+ FA).
- Findings numbering: next is **26** (`docs/findings/26_regional_representation_
  evaluation.md` + `.fa.md`). Findings ≤ 25 are immutable.
- GitHub text-only; weights/bundles travel by Drive per AGENTS.md §6.2 naming.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Segmentation labels (T1) leak model-relevant info unfairly | Segmentation is fixed, data-derived, identical for all arms; probe compares encoders, not segmenters; documented |
| Transfer curves overfit dev | Label subsets hashed per fraction; arms share subsets; CI-based gates; temporal rolls as second axis |
| Retrieval gaming via trivial seasonal echoing | Track 1 requires anomaly-family match at matched FAR; raw-distance baseline strong by construction |
| B1 rule mis-specified, rigged low/high | Two pre-registered variants reported; calibration on clean subset; FAR-matched — never tuned on eval |
| Compute creep via B4/B5 encoders | Capacity band ±20 %; one trunk per rung; params reported per rung |
| Claim creep ("SSL wins" from one cell) | G-E gate; per-cell tables; rung tags; no macro-only reporting |
| Scope absorption of the A1–A10 draft | None — it is a separate change (`benchmark-latent-reconstruction-prediction-architectures`, created 2026-11-23 by owner decision); this change consumes only its legitimate measurements and no gate depends on it |

## Open questions for the owner (mirror of proposal)

L4 evidence timeline (company harvest records / expert review) · whether T5
clustering outputs should feed the platform later (out of scope here).
(Resolved 2026-11-23: five-level ladders all kept; screening gate kept for
region model v0; A1–A10 draft created as a separate change; 22 existing
changes triaged superseded with `INDEX.md`; T3 ships both tracks plus the
descriptive near-match listing.)
