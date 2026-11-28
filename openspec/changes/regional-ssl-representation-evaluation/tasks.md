# Tasks — regional-ssl-representation-evaluation

TDD-first. Every implementation task ships with unit tests; nothing reaches a
training run before the local gate (AGENTS.md §6.1). The agent never launches
long jobs — they are handed to the owner as exact commands. Checkpoints and
`progress.jsonl` are mandatory for anything trainable.

## 0. Registration and pre-registration

- [x] 0.1 Record this change in `todo.md` (owner-facing entry) and freeze the
      pre-registration document (`configs/evaluation/pre_registration.yaml`):
      hypotheses H1–H5, gates G-A…G-E, label fractions, k ∈ {1,3,5,10}, m ∈
      {1,5,10}, S1–S6 severity references, eval-subset definition, seeds
      {13, 42, 2026}, capacity band ±20 %, split hashes.
- [ ] 0.2 Verify artifact bindings before any run: `cycles_490_v2.parquet`,
      `splits_locked_v2.json` (sha `30ecbe10…`), `holdout_manifest_v2.json`
      (sha `d501bf81…`), `norm_stats_full.json`; exit non-zero on any mismatch.
- [ ] 0.3 Rebuild/record the v2 normalization sidecar next to the v2 store and
      bind its sha into the pre-registration (train split only).
- [ ] 0.4 Pre-registration git tag or commit hash recorded; any later deviation
      must be recorded as an amendment in the findings document — never
      silently.

## 1. B1–B5 ladder implementation

- [ ] 1.1 `B1` rule arm `ndvi_lst_rule`: per-field robust MAD thresholds,
      AND-variant and OR-variant, calibration on the clean calibration subset
      only, FAR-matched to 5 % cycle-FAR (no eval tuning). Unit tests for
      calibration independence and FAR estimation.
- [ ] 1.2 `B2`/`B3`/`B4`/`B5` SSL input builders wired to the existing v2
      loaders (`8-ch + DOY + mask`; B4 adds daily ERA5 t2m/tmean, precip,
      solar radiation, VPD, et0 with causal masking; B5 adds DSH sin/cos,
      phase bin and STL trend/season/residual features from
      `decomposition-aware-ssl-study` outputs, marked `absent` if that study
      has not produced them).
- [ ] 1.3 Causality test: no feature at bin o references bins > o.
- [ ] 1.4 Capacity accounting: parameter counts per rung recorded in a JSON
      manifest; capacity band enforced.
- [ ] 1.5 Common scoring bridge: every rung exposes the same `score(field,
      bin) -> float` interface for identical evaluation.

## 2. Pretraining quality battery (Capability 1)

- [x] 2.1 Masked reconstruction metric on dev clean cycles (observed entries
      only), 3 seeds.
- [x] 2.2 Farm-identity kNN probe (k ∈ {1,5}, top-1/top-5 vs 1/490 and
      5/490 chance and vs B1-feature kNN).
- [x] 2.3 Phenological-phase linear probe (DSH 36-way; vs majority baseline
      and vs B1-feature probe; macro-F1 + accuracy).
- [x] 2.4 Embedding temporal stability (within-field cycle-pair cosine;
      median/IQR; shuffled-null control).
- [x] 2.5 Collapse diagnostics (per-dim std spectrum; effective rank with
      recorded threshold; alignment/uniformity); hard fail at screening.
- [x] 2.6 `pretraining_quality.json` + unit tests for each probe (synthetic
      embeddings with known properties).

## 3. Candidate screening (region model v0)

- [ ] 3.1 Implement/adapt candidates C-incumbent, C-BT, C-traj, C-hybrid
      (C-decomp only if supported by the decomposition study's ablation).
      Shared trunk where feasible; mask-aware losses; DOY/DSH conditioning
      per candidate declaration.
- [ ] 3.2 Screening runner: dev-only, 3 seeds, equal optimizer budget,
      capacity band enforced; outputs screening_report.json with per-candidate
      gate evidence.
- [ ] 3.3 Selection by G-A…G-E aggregates only; losing candidates recorded
      with reasons; no post-hoc candidate additions.
- [ ] 3.4 SCOPE EXTENSION (recorded 2026-11-28, workflow package WP-4, BEFORE
      implementation; additive only, gates G-A…G-E unchanged): fold the
      research plan's three arms
      (`thesis_research/SSL_Proposal/ssl-hierarchical-agri-model/REPORT.md`
      Level A / C / E) in as additional screening candidates —
      C-stat (Level A: per-field per-date statistical quantile summaries from
      `Data/Processed/satellite/stats.zarr`, a non-learning representation
      arm), C-patch (Level C: small patched transformer with Δt-aware
      positional encoding) and C-film (Level E: frozen backbone + FiLM-style
      per-farm adapters). Discrepancy recorded: the research text says "61-dim"
      summaries and "≤1.5M-param" transformer; the actual store yields 58
      features/field-date and the frozen pre-registration capacity band
      (±20 % of 100 k encoder params, 250 k total hard cap) is AUTHORITATIVE —
      the screening candidate is configured inside the band and the research
      ceilings are recorded in the candidate's manifest, never used to widen
      the band (no post-hoc preregistration amendment).
- [ ] 3.5 Screening runner aggregates over ALL THREE pre-registered seeds
      {13, 42, 2026} (mean ± spread per probe; collapse on any seed
      disqualifies) and writes `screening_report.json` alongside
      `pretraining_quality.json`.
- [ ] 3.6 C-film adapter arm: base encoder frozen, only per-farm FiLM γ/β +
      farm embeddings trainable; adaptation budget recorded; a farm outside
      the adapter table is recorded absent (never routed to a zero adapter).

## 4. Downstream transfer suite (Capability 2)

- [x] 4.1 Label-fraction machinery: stratified 5/10/25/50/100 % subsets per
      task (hashed index lists, shared across arms), 3 repeats per cell.
- [x] 4.2 Arms A-rule, A-B1, A-rand, A-scratch, A-ssl-frozen, A-ssl-ft per
      task; equal budgets for A-scratch vs A-ssl-ft.
- [ ] 4.3 T1 phenology-stage estimation (probe head; claim-bearing).
- [ ] 4.4 T2 limited-label injection detection (AUC-PR per family × severity;
      prevalence reported).
- [ ] 4.5 T3 few-shot retrieval: Track 1 (injection-labelled queries) with
      precision@{1,5,10} / MAP / family-hit-rate; Track 2 (weak-proxy CWSI_ET/
      TVDI excursion queries, explicitly labelled weak, exploration only)
      reported separately; baselines: B1-score, A-rule-score, raw band-vector
      distance, A-rand; **plus** `near_matches.json`/`near_matches.md`: for
      every T3 query, the top-10 nearest NATURAL-data (unlabelled) archive
      dates with similarity scores, field/date and the query answered —
      descriptive, for manual owner inspection, never gate input. (Owner
      decision 2026-11-23.)
- [ ] 4.6 T4 self-supervised next-1/3-bin trajectory quality vs persistence
      baseline (clean held-out dev cycles).
- [ ] 4.7 T5 exploratory clustering/nearest-date listings — descriptive
      report only, labelled `exploratory`, never used in gate evaluation.
- [x] 4.8 `transfer_curves.json`, `retrieval_benchmark.json`,
      `near_matches.json`/`near_matches.md` (+ tests).

## 5. Generalization protocols (Capability 3)

- [ ] 5.1 Temporal rolls inside train+dev (cut years frozen in
      `configs/evaluation/temporal_rolls.yaml`; train ≤ Y → validate following
      12–24 months; B1 run identically per roll).
- [x] 5.2 Leave-one-block-out over the 20 locked blocks within train+dev
      (holdout-excluded fields never appear).
- [ ] 5.3 Geographic limitation recorded; claim ceiling L3 same-region stated
      in every findings section that quotes a number.
- [x] 5.4 `generalization.json` (+ tests for roll disjointness and block
      coverage).

## 6. Decision gates and capstone report

- [x] 6.1 Gate aggregator `decision_gates.json`: per-gate inputs, CIs,
      pass/fail, and pre-declared consequence text.
- [ ] 6.2 Capstone report `docs/findings/26_regional_representation_
      evaluation.md` (+ `.fa.md`): all five-level tables (B1–B5; 5/10/25/50/
      100 %; L1–L5 rung tags; T1–T5; H1–H5), per-cell results, negative
      results with equal prominence, limitations (single region/crop; proxy
      circularity; segmentation-derived labels; geographic scope), explicit
      statements of evidence insufficiency where applicable.
- [ ] 6.3 `todo.md` updated; findings 00–25 verified untouched; platform
      untouched; locked test unread (verified by held `--final-test` path
      audit).

## 7. Verification and hand-over

- [x] 7.1 Unit suite: `pytest tests/evaluation -q` green; full relevant
      regression suite green (state counts).
- [x] 7.2 `--smoke` end-to-end for every new script; smoke outputs read and
      inspected (metrics present, no NaN, shapes/counts expected).
- [ ] 7.3 USER-RUN items: screening (§3), transfer runs (§4), rolls (§5) —
      handed over as exact commands with conda env, cwd, flags, expected wall
      time, output paths; laptop budget ≤15 min/seed and ≤250k params kept or
      the job escalates to Colab per the placement policy (owner decision,
      local gate first).
- [ ] 7.4 Bundle/verify any Colab outputs back per AGENTS.md §6.2 naming;
      GitHub stays text-only.
- [ ] 7.5 Findings FA translation checked with the Persian-writing audit;
      EN/FA number parity reviewed.

## 8. Acceptance (all required to close)

- [ ] 8.1 Every cell of B1–B5 × {dev, temporal rolls, spatial rolls} reported
      with prevalence, CIs and rung tags — no macro-only reporting.
- [ ] 8.2 Gates G-A…G-E recorded (pass or fail) without post-hoc
      modification of the pre-registration.
- [ ] 8.3 Zero reads of the locked test split outside `freeze_main_model.py
      --final-test` (the path remains unexecuted in this change).
- [ ] 8.4 Independent re-derivation check: a second agent/runnable can
      regenerate `decision_gates.json` from committed JSON artifacts and
      arrives at the same verdicts.
