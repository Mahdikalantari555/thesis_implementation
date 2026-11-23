# Change: benchmark-latent-reconstruction-prediction-architectures

**Change ID:** `benchmark-latent-reconstruction-prediction-architectures`
**Project:** Thesis Implementation
**Status:** Proposed
**Priority:** High
**Target hardware:** Google Colab NVIDIA T4 (primary), CPU for preprocessing and classical baselines
**Execution strategy:** Budget-aware adaptive benchmark with a complete, pre-registered experiment registry
**Research context:** Self-supervised learning for unlabeled growth-anomaly detection in sugarcane fields using Landsat time series
**Created:** 2026-11-23 (owner decision; converted from the owner-supplied draft of the same date into repository conventions)
**Base:** branch `arena/ed3c4228-thesis-implementation` @ `2b479dc`

---

## Why

The thesis needs evidence — not intuition — about **which learning paradigm, if
any, deserves the central slot** in an unlabeled agricultural growth-anomaly
study. The frozen lineage is honest and negative: finding 20 attributes the
detection failure to the objective/unit, finding 21 rejected the untrained
forecast unit, finding 23 closed the GRU-prediction path, finding 25 voided the
patch verdicts under the v2 fixed protocol (MDM80 censored >S6; detection rates
rising only to 0.456 at S6; L2 rung; L5 unmet). No architecture hypothesis has
been tested fairly and broadly on this archive with identical data, splits,
budgets and statistics.

This change designs and implements a comprehensive, reproducible benchmark of
**reconstruction-based, latent-reconstruction-based, prediction-based, JEPA-
style and hybrid self-supervised architectures** against classical baselines,
under pre-registered protocols, so the field can decide — with uncertainty
quantification — whether any of them beats simple physical rules or the
incumbent B1/GRU-AE stack.

**This change is not expected to rescue SSL, and it is not designed to produce a
winner by construction.** An all-negative or mixed result is a valid,
publishable outcome. The objective is statistically supported evidence about:

1. Which learning paradigm provides the strongest anomaly-detection ranking?
2. Whether anomaly suppression improves robustness under contaminated training?
3. Whether latent-space learning provides advantages over input-space reconstruction?
4. Whether temporal architectures generalize across unseen farms and periods?
5. Whether weather and phenological conditioning improve normality modelling?
6. The trade-offs among detection quality, generalization, complexity and compute?

### Repository grounding (mandatory context)

- **Data:** v2 chain only — `cycles_490_v2.parquet` (3,478 non-censored
  cycles / 490 fields; median cycle 366 d; bin grid 36 × 15-day),
  `splits_locked_v2.json` (sha `30ecbe10…`; train 2,105 / dev 340 / test 773 /
  monitoring 1,240), `holdout_manifest_v2.json` (sha `d501bf81…`; 37
  holdout-excluded test-block fields), `norm_stats_full.json` (sha
  `3cb1ca60…`; v2 sidecar to be rebuilt/recorded). v1 artifacts are not used.
- **Axis semantics:** inputs are mask-aware (8 spectral/thermal channels +
  DOY or the active `PhaseAxis`); detection targets **phenology-phase**
  anomalies, not raw magnitude.
- **Current honest rung: L2 (not cleared); L5 unmet.** All outputs tag claims
  with rung, and synthetic detection is controlled evidence, never stress.
- **Compute policy (owner directive 2026-11-20/21):** heavy jobs run on Colab T4
  after a local gate (unit tests + inspected `--smoke`); the agent never
  launches long jobs; laptop budget ≤15 min/seed, ≤250k parameters for dev runs.
- **Sequence length is 36 bins.** Mamba's selective-scan advantage is designed
  for long sequences; at L=36 the Mamba arms are expected to be dominated by
  simpler backbones. They are kept (owner's request to test them) but carry an
  explicit feasibility gate: if an Mamba arm is numerically unstable or exceeds
  the per-experiment budget, it is marked `Not Applicable` with the recorded
  reason — never silently replaced, downsized without recording, or tuned into
  a different experiment.
- **Matrix-size reality:** the full cross-product is large (10 architectures ×
  suppression × inputs × calibration × 3 seeds). Execution is therefore
  budget-aware with screening first (see below); incomplete cells are reported
  as `Deferred`, never as absent or complete.

### Relation to the companion change

`regional-ssl-representation-evaluation` (the capstone; owner decision
2026-11-23) evaluates a **light, CPU-runnable regional base model**: Level-1
pretraining quality, Level-2 downstream transferability (B1–B5 ladder, label
fractions), few-shot similar-date retrieval, temporal/spatial generalization.
**This change is the heavy counterpart** — it probes the paradigm space broadly
on Colab T4. No gate in the capstone depends on this benchmark completing;
this benchmark's job is to inform it (and the thesis architecture decision)
with controlled evidence. Duplication is avoided: the capstone owns
representation/transfer evaluation and claim semantics; this change owns the
architecture × suppression × synthetic-anomaly comparison.

## What Changes

- **New capability `architecture-benchmarking`** — a common train/infer/
  checkpoint/evaluate interface; the ten architecture configurations
  (A1 GRU-AE, A2 Mamba-AE, A3 GRU-LatentAE, A4 Mamba-LatentAE, A5 GRU predictor,
  A6 Mamba predictor, A7 GRU-JEPA, A8 Mamba-JEPA, A9 hybrid GRU, A10 hybrid
  Mamba) across five learning paradigms; collapse prevention for latent models;
  hierarchical patch→farm processing where the v2 patch store prerequisite is
  met; dual patch/farm scoring.
- **New capability `anomaly-suppression-benchmark`** — S0 baseline, S1 soft
  suppression, S2 masked learning, S3 synthetic-anomaly suppression; explicit
  per-architecture applicability matrix (invalid combos recorded `Not
  Applicable` with reasons); independent synthetic train/test streams.
- **New capability `experimental-validation`** — AUPRC primary (predefined
  macro-average over 5 families × 3 severities, prevalence reported, per-cell
  tables), three protocols (V1 temporal holdout; V2 farm holdout; V3
  spatiotemporal, train+dev only with holdout exclusions — the locked test
  stays untouched), 3 seeds, statistical suite (paired bootstrap preserving
  farm/event dependence, Wilcoxon, Friedman + post-hoc, effect sizes,
  multiple-comparison correction), environmental validation with explicit
  shared-input disclosure (TVDI, CWSI_ET, CWSI_LST, ETa, ETrF), and the three
  winner reports (detection / generalization / multi-objective Pareto).
- **New capability `reproducible-benchmark-execution`** — complete experiment
  registry with explicit states (Pending/Running/Completed/Failed/Interrupted/
  Deferred/Not Applicable); budget-aware scheduling (screening → full),
  checkpointing + resume, manifests for every experiment, Colab T4 notebook,
  persistent Drive outputs with bundle convention, honest completion reporting.
- **Benchmark notebook** `notebooks/benchmarks/architecture_comparison_benchmark.ipynb`
  (loads implemented modules; never duplicates model code).
- **Implementation roots:** `src/ssl_anomaly/{models,suppression,spatial,weather,
  phenology,synthetic,validation,calibration,evaluation,benchmarking}/`,
  `configs/benchmarks/`, `scripts/benchmarks/`, `tests/benchmarking/`,
  `reports/architecture_benchmark/`. Existing modules are reused; frozen
  artifacts (checkpoints, B1, splits, holdout, platform, findings ≤ 25) are
  not modified.
- **Research outputs:** architecture comparison, suppression ablation, synthetic
  anomaly report, input ablation (I1/I2/I3), weather fusion (W1/W2), normality
  and calibration (C1/C2/C3), generalization report, environmental validation
  report, statistical significance report, computational efficiency report, and
  a final research findings document (EN + FA), numbered as the next finding
  after the capstone's (i.e. ≥ 27 at the time of writing).

### Explicit non-goals

- Replacing, retraining or reinterpreting the frozen detection path or any
  frozen artifact; reading the locked test split outside the one locked
  `--final-test` path (which this change does NOT execute).
- Claims of confirmed real-world stress detection or causal attribution
  (water deficiency vs salinity); environmental indicators are proxies.
- Geographic generalization (one region, one crop; recorded as claim ceiling
  and limitation).
- PV/foundational models, multimodal LLMs/VLMs as primary architectures;
  commercial imagery; yield forecasting (no yield labels).
- Platform (`platform/`) changes; new data acquisition; changes to the capstone
  change's scope or pre-registration.
- Reporting an incomplete matrix as a complete benchmark.

## Capabilities

### New Capabilities

- `architecture-benchmarking`
- `anomaly-suppression-benchmark`
- `experimental-validation`
- `reproducible-benchmark-execution`

### Modified Capabilities

- None. `model-development-protocol` invariants (split manifest, test
  isolation, mask-aware inputs, train-only normalization, frozen seeds) apply
  and are referenced, not modified.

## Impact

- **Affected code:** new `ssl_anomaly` subpackages and thin CLIs listed above;
  new tests (`tests/benchmarking/`); new configs; new notebook; new report
  outputs; `todo.md` (status); findings (new EN+FA documents at completion).
- **Not affected:** frozen checkpoints/sidecars; `splits_locked_v2.json`;
  `holdout_manifest_v2.json`; B1; platform; findings 00–25; v1 artifacts; the
  capstone change (`regional-ssl-representation-evaluation`).
- **Compute:** Colab T4 for full runs (handed to the owner with exact cells);
  laptop only for unit tests, `--smoke` and quick single-seed probes
  (≤15 min, ≤250k params). Full protocol assumes Drive-persisted bundles with
  the dated self-describing naming convention; GitHub remains text-only.
- **Known prerequisites:** (a) v2 patch store re-extraction + norm v2 sidecar
  bindings (pending owner action in the v2 chain — required only for the
  hierarchical patch/farm arms, A1–A10's patch-level scoring); (b) daily ERA5
  ingest into the training path for B4-class inputs and I2/I3 (farm-mean daily
  exists; hourly deliberately out of v0 scope); (c) STL features from
  `decomposition-aware-ssl-study` for B5-class inputs (optional; `absent` if
  not produced). If a prerequisite is unmet, dependent cells are `Deferred`
  with the reason recorded — never approximated.

## Acceptance (summary — see design.md §Acceptance)

Implementation: all ten architectures implemented or explicitly blocked with
evidence; four suppression configs with documented applicability; five families
× three severities; hierarchical scores (or documented `Not Applicable` with
prerequisite status); all three protocols; classical baselines; statistical
suite; existing thesis functionality intact (regression tests green).

Benchmark: notebook runs from clean Colab; T4 path verified; manifests complete;
interruption resumable; incomplete cells visible; macro-AUPRC computed per the
predefined protocol; environmental validation separated from synthetic evaluation.

Scientific: the final report answers the fourteen questions in design.md
§Scientific acceptance — or explicitly states when evidence is insufficient. No
model is declared superior solely from a highest point estimate; all claims
carry rung tags, intervals and prevalence.
