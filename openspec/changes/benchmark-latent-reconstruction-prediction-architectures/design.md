# Design — benchmark-latent-reconstruction-prediction-architectures

Converted 2026-11-23 from the owner-supplied draft into repository
conventions; all conflicts with frozen artifacts, the v2 chain and AGENTS.md
were resolved in favor of the repository (this document is the authority).
Draft's non-negotiables §9 (research integrity) are preserved verbatim below.

## Context

Thesis ground truth at `2b479dc`: rubber-meets-road status is L2 (not cleared).
Findings 18–25 established, on the v2 fixed protocol: MDM80 censored >S6 for
all SSL arms (incl. GRU-Pred) and for both patch arms; detection rates rise
only at S5–S6; B1 ≥ SSL everywhere; `patch_pixelwise` frozen winner void; path
C closed. The benchmark therefore tests *paradigms*, not architectures-as-
fashion: reconstruction vs latent reconstruction vs prediction vs JEPA vs
hybrid, under suppression and generalization pressure.

Baseline anchors (all reused, never redefined here): `B1` =
`b1_phasez_causal` phase-aware robust z-score; `gru_ae_main` (frozen, v2);
`ndvi_lst_rule` (FAR-matched simple control from the capstone change —
recommended inclusion as an additional baseline row, owner-configurable);
classical triple (Isolation Forest, One-Class SVM, Robust PCA); random-init
encoder of each trunk (capacity control).

## Dataset and experimental assumptions

| Property | Value |
|---|---|
| Study area | Amirkabir agro-industrial unit, Khuzestan, Iran (~14,000 ha; 490 farm polygons) |
| Satellite | Landsat 7/8/9 C2 L2, EPSG:32639, 30 m |
| Temporal coverage | 2013–2026; 607 acquisitions; 539 METRIC products; 538 usable dates; admissible counts are smaller per split |
| v2 cycle artifacts | `cycles_490_v2.parquet`: 3,478 uncensored cycles / 490 fields; median 366 d; 36 × 15-day bins |
| Splits | `splits_locked_v2.json` sha `30ecbe10…`: train 2,105 / dev 340 / test 773 / monitoring 1,240 (locked; test untouched) |
| Holdout exclusions | `holdout_manifest_v2.json` sha `d501bf81…`: 37 test-block fields excluded everywhere |
| Weather | ERA5-style daily (hourly available but out of v0 scope; daily + windows only) |
| METRIC proxies | ETaDaily, CWSI_ET, CWSI_LST, ETrF, TVDI — see §Environmental validation (none independent) |

Missing observations, quality masks, temporal gaps and differing observation
counts are handled explicitly: mask-aware inputs; masks dropped ⇒ error;
no data ⇒ `absent`, never zero-filled.

## Architecture matrix A1–A10

Five paradigms × two backbones. Distinct mathematical objectives are
preserved; a "hybrid" is a weighted combination, not a re-labelled reconstruction.

| ID | Architecture | Paradigm |
|---|---|---|
| A1 | GRU-AE | Input reconstruction |
| A2 | Mamba-AE | Input reconstruction |
| A3 | GRU-LatentAE | Latent reconstruction |
| A4 | Mamba-LatentAE | Latent reconstruction |
| A5 | GRU Predictor | Future prediction |
| A6 | Mamba Predictor | Future prediction |
| A7 | GRU-JEPA | JEPA latent prediction |
| A8 | Mamba-JEPA | JEPA latent prediction |
| A9 | Hybrid GRU | Reconstruction + prediction |
| A10 | Hybrid Mamba | Reconstruction + prediction |

### Paradigms

- **Input reconstruction** — `L_rec = ‖m ⊙ (x − x̂)‖² / ‖m‖₁` over observed
  entries; learning target is normal input fidelity.
- **Latent reconstruction** — `z = E(x)`; `ẑ = D(context(z))`; `L = ‖z − ẑ‖²`
  with a documented collapse-prevention mechanism (stop-gradient, EMA target
  encoder, or equivalent) — the exact mechanism SHALL be named and justified in
  the config and report. Collapse diagnostics (effective rank, per-dim std,
  alignment/uniformity) are mandatory for A3/A4/A7/A8.
- **Future prediction** — `x̂_{t+1:t+k} = P(x_{1:t})`; error vs actual future
  observations; causal (no ≥ t+1 information at t); distinct from
  reconstruction.
- **JEPA** — context/target encoders (EMA-stabilized target); predict future
  latents; `L = ‖ẑ_future − sg(z_future)‖²`; collapse prevention mandatory;
  representation-content check (probe) required, else the arm is `invalid`.
- **Hybrid** — `L = λ_r L_rec + λ_p L_pred`, λ documented and pre-registered
  (default 1.0/1.0; sensitivity arms {0.5, 1, 2}) with identical backbone
  distinction as other pairs.

### Backbone notes

GRU: established, stable at L=36. Mamba: selective-scan kernels require
CUDA; CPU fallback is allowed for unit tests only. **Feasibility gate:** any
Mamba arm failing numerical stability or the 2× T4 wall-time budget of its GRU
twin is marked `Not Applicable` with recorded evidence; it is never silently
replaced, shrunk, or re-scoped.

## Anomaly suppression matrix S0–S3

| ID | Config | Description |
|---|---|---|
| S0 | Baseline | Standard objective |
| S1 | Soft suppression | Per-observation robustness weights `w_i ∈ [w_min, 1]`, bounded; documented non-reinforcing design (weights may not derive only from current residual — requires an external or detached signal) |
| S2 | Masked learning | Controlled masking of observations/bins; distinct from missing-data handling (which is mandatory anyway) |
| S3 | Synthetic anomaly suppression | Synthetic contamination in training + suppression-aware learning; train generator independent of eval generator (separate seeds, separate samplers) |

**Applicability matrix (binding):** every (Ax, Sy) pair is pre-registered as
`applicable` / `not-applicable` with a mathematical reason where invalid
(e.g., masking conflicts with a pure future-prediction objective at inference
time must be resolved by objective adapters, not by ad-hoc loss edits). The full
matrix is committed before execution; invalid pairs are `Not Applicable`,
never substituted.

## Synthetic anomaly generation

Five families; three severities (Weak/Moderate/Strong), scaled to historical
variability and feature-specific distributions, avoiding trivial separability;
plus controlled family combinations.

1. **Spectral** — band-level shifts; inter-band relation distortion.
2. **Temporal** — sudden jumps/drops; irregular transitions.
3. **Phenological** — shifted timing; abnormal trajectory shape.
4. **Multivariate** — joint spectral + weather distortions; relational breaks.
5. **Persistent** — gradual degradation; multi-date deviations.

Every generated event records: farm id, patch id (where applicable), temporal
location, duration, family, severity, injected dimensions, parameters, seed,
ground-truth mask and boundaries. Synthetic labels are controlled-evaluation
only (and designated contamination training for S3); they are never
interpreted as real stress. Train/test synthetic streams never share samples;
a contamination audit records per-run type/severity/percentage of training
contamination.

## Hierarchical spatial-temporal learning

Pipeline: spatial observations → patch encoder → farm aggregation → temporal
encoder → representation. Patch-level and farm-level scores both produced;
farm scores via documented aggregation (channel-max + trailing window, matching
the frozen semantics; or attention pooling — pre-registered, identical across
applicable architectures). Patch sampling, count, resolution and validity masks
are controlled for fairness.

**Prerequisite flag:** the v1 `patches.zarr` is superseded; the v2 patch
re-extraction (full archive, finding-25-era tooling) is pending an owner action
already tracked in `todo.md`. Until the v2 patch store + norm v2 sidecar are
bound, all patch-level cells are `Deferred` with reason recorded; the
field-mean 1D pipeline is unaffected.

## Weather and phenological conditioning

Input scope (three ablations): **I1** spectral-only · **I2** spectral+weather ·
**I3** spectral+weather+phenological context. Weather fusion: **W1**
conditioning (FiLM-style) · **W2** separate weather encoder → latent fusion.
Daily features only in v0; if hourly is used later it enters via a documented
causal aggregation. Phenological context: DSH-bin/phase embeddings from the
existing `PhaseAxis` machinery (segmentation-derived, data-derived; no
planting-date invention); anything unavailable is `absent`.

## Normality modelling and calibration

Primary paradigm: **hybrid normality** (farm history + phenological context +
environmental conditioning). A farm is never compared primarily to other farms.
Calibration strategies (ablated): **C1** farm-specific · **C2** farm × stage ·
**C3** global. All calibration from train/calibration partitions; test never
used for thresholds or normalization.

## Classical baselines

Isolation Forest, One-Class SVM, Robust PCA — integrated behind the common
score interface; receive the same permitted data; feature/input compatibility
documented (they see curated summary features — their representation difference
is part of the report, not a defect).

## Validation protocols and seeds

| ID | Protocol | Definition |
|---|---|---|
| V1 | Temporal holdout | Chronological train/val/test within the v2 split discipline (dev for selection; test untouched) |
| V2 | Farm holdout | Disjoint farm groups (block-consistent) for train/val/test within train+dev |
| V3 | Spatiotemporal holdout | Unseen farms × unseen periods, within train+dev; holdout-excluded fields excluded |

Three seeds {13, 42, 2026}; mean ± std; seed spread reported separately from
farm and event variability. Overlapping windows are never independent samples.
The locked test split is not read by any protocol here.

## Metrics

**Primary: AUPRC.** Levels: event-level, point-level, patch-level, farm-level.
Headline: macro-AUPRC across the 5 families × 3 severities at a predefined,
identical test prevalence (equal cell weighting; per-cell tables mandatory;
macro-average may not hide a failing cell). Secondary: AUROC; F1 (validation-
selected threshold; threshold rule recorded); Detection Delay (localized and
persistent events; median days); Spearman and Kendall rank correlation;
calibration (Brier score, ECE where applicable); inference latency; peak VRAM;
trainable parameters; FLOPs (or recorded estimate methodology). FLOPs estimates
state their counting convention.

## Environmental validation (proxies, never proof)

TVDI, CWSI_ET, CWSI_LST, ETa, ETrF — each evaluated separately with Spearman
and Kendall, with dependence-aware uncertainty. **Evidence classes (binding
disclosure):**

- *Shared-input:* CWSI_LST and TVDI (LST/NDVI-derived), CWSI_ET and ETa
  (radiation/meteorology + canopy energy balance). These are *consistency
  checks*; using them to "confirm" detection would be circular.
- *Independent:* none fully independent in this archive. ETa/ETrF come closest
  but share meteorological inputs. The report must state, per indicator, its
  input overlap in one sentence.

Indicators are reported as rank-agreement evidence only; calibration,
selection and tuning never use them; they never raise a rung above L3 and
never support L4.

## Statistical analysis

1. Paired bootstrap (resampling whole farms and whole events; preserves
   dependence; B ≥ 1,000).
2. Wilcoxon signed-rank (paired cells).
3. Friedman test across architectures; post-hoc Nemenyi (or Conover) with
   appropriate multiplicity control.
4. Effect size estimation (rank-biserial / Cliff's δ) with confidence
   intervals.
5. Multiple-comparison correction across cells/tests (Holm–Bonferroni default;
   Benjamini–Hochberg for screening tables).
6. Variance decomposition: seed vs farm vs temporal vs synthetic-event
   variability (report all four).
7. **Three winner reports, never a scalarization:** Winner 1 — detection
   (macro-AUPRC with intervals); Winner 2 — generalization (stability across
   V1/V2/V3); Winner 3 — multi-objective (Pareto front across detection,
   generalization, compute). No forced composite score.

## Budget-aware execution

The complete experiment matrix is generated and committed **before** execution
(`configs/benchmarks/matrix.yaml`, hash recorded). Scheduling order:
screening (subset, 1 seed, early-stop on collapse/FAIL criteria) → full
benchmark (3 seeds) → deferred review. Priorities are predefined (incumbent
first, then paradigms adjacent to it). Budget estimation per experiment from
the smoke profile; a global T4 budget (hours) is set by the owner per run
cycle; on exhaustion, remaining cells become `Deferred` — visible in every
report, never dropped.

Registry states (binding): `Pending`, `Running`, `Completed`, `Failed`,
`Interrupted`, `Deferred`, `Not Applicable`. `Not Applicable` requires a
recorded machine-checkable reason (e.g., invalid applicability pair, feasibility
gate trigger). Incomplete matrices are reported as incomplete.

## Equal search budgets

Each architecture receives a predefined search budget (trials, wall-time cap,
search space recorded in config). Hyperparameter selection uses validation
partition metrics only (macro-AUPRC on a validation injection panel); test
data never selects anything. Selected hyperparameters, search duration, trial
count and validation metric are recorded per architecture.

## Capacity fairness

Two comparisons, both mandatory: (a) **parameter-matched** (±20 % band; counts
recorded); (b) **compute-matched** (controlled training FLOPs/steps; recorded
FLOPs estimate with convention). Reports include trainable params, FLOPs
estimate, training time, inference latency, peak VRAM. Parameter matching does
not imply compute equivalence — both are shown.

## Reproducibility manifest (per experiment)

Experiment id; architecture; suppression; input config (I*); weather fusion
(W*); normality strategy; calibration (C*); protocol (V*); seed;
hyperparameters; data versions (cycle file hash, split manifest hash, norm
hash); code git SHA; software versions; hardware; timestamps (start/end);
checkpoint refs; metric outputs (with prevalence); wall time; status. Manifest
sufficiency: re-running from manifest reproduces the cell up to recorded seed
spread.

## The Colab notebook

`notebooks/benchmarks/architecture_comparison_benchmark.ipynb` — sections:
environment setup; hardware detection; dataset verification (hashes); experiment
configuration; split verification; synthetic generation; model initialization;
hyperparameter search; screening; full benchmark; checkpoint management
(`--resume`); resume interrupted; synthetic evaluation; environmental
validation; generalization evaluation; statistical analysis; computational
profiling; winner comparison; report generation; export. It imports implemented
modules (duplication prohibited) and persists to Drive with dated,
self-describing bundle names; local gate first per AGENTS.md §6.1.

## Persistent outputs and report sections

`reports/architecture_benchmark/` — `manifests/`, `metrics/`, `statistical/`,
`figures/`, `findings/`. Exports: CSV results, JSON manifests, checkpoints,
statistical summaries, metric tables, figures, machine logs, final Markdown
(+ FA) report with sections A (architecture comparison) through K (research
conclusions), including limitations and incomplete-coverage statements.

## Research integrity — non-negotiable constraints (binding)

1. No test leakage (locked test unread; selection on validation only).
2. No modification of frozen findings or artifacts.
3. No fabricated phenological labels; segmentation-derived features labelled
   as such.
4. Synthetic anomalies ≠ confirmed stress; proxy ≠ ground truth.
5. No causal attribution of anomalies (water vs salinity).
6. No comparisons on inconsistent splits/dates/FAR.
7. No hidden configuration changes; the committed matrix is the contract.
8. No reporting incomplete coverage as complete.
9. No overlapping windows as independent samples.
10. No winner from a point estimate; intervals + effects mandatory.
11. No silent replacement of mathematically incompatible suppression methods.
12. No mixing shared-input proxies with independent evidence.

## Scientific acceptance — the fourteen questions

The final report SHALL answer, with intervals and rung tags — or explicitly
state insufficiency: (1) highest macro-AUPRC architecture; (2) significance of
the difference; (3) latent vs input reconstruction; (4) prediction vs
reconstruction; (5) JEPA measurable advantage; (6) hybrid improvement;
(7) suppression efficacy; (8) best suppression; (9) weather conditioning value;
(10) phenological-context generalization value; (11) performance loss on
unseen farms; (12) performance loss on unseen periods; (13) best
computational trade-off; (14) whether synthetic detection is corroborated by
proxy evidence (with the circularity caveat printed beside every such table).

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Matrix size exceeds T4 budget | Screening phase; budget estimation from smoke; `Deferred` visibility; owner-set T4 hour budget per cycle |
| Mamba at L=36 underperforms/unstable | Feasibility gate → `Not Applicable` with evidence; GRU twin remains the comparison anchor |
| Latent collapse silently degrading A3/A4/A7/A8 | Mandatory collapse diagnostics; candidate disqualified, not retuned to pass |
| Synthetic anomalies too easy (trivial separability) | Severity scaled to MAD; realism checks reported (injection magnitude distributions vs data distribution) |
| Circularity via CWSI/ETa | Shared-input disclosure table; proxies never select; printed caveat on every proxy table |
| Seed/farm dependence inflating significance | Paired bootstrap at farm/event level; variance decomposition; CIs everywhere |
| Registry drift (configs changed mid-run) | Config hash + git SHA per experiment; resume requires hash match; amendments recorded |
| Drive loss of bundles | §6.2 bundle convention; notebook writes manifest-first, checkpoints second; notebook completion summary lists bundle filename |

## Open questions for the owner

1. v2 patch re-extraction + norm v2 rebuild: approved timing? (Patch-level
   cells stay `Deferred` until bound.)
2. T4 budget (hours) for the first benchmark cycle, and screening breadth
   (1 vs 2 seeds).
3. Whether `ndvi_lst_rule` (capstone B1) is admitted as an extra baseline row
   (recommended: yes).
4. Hourly ERA5 in scope later (would require a documented causal aggregation
   contract)?
5. Findings numbering at completion (next free number; capstone claims ≥ 26).
