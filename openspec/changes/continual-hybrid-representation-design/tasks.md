# Tasks — continual-hybrid-representation-design

TDD-first. Laptop CPU only (owner directive 2026-11-26). The agent writes code
and unit tests and runs only smoke tests (< ~2 min, AGENTS.md §6); every real
training run is launched by the owner with an exact command and always ends
with a convergence verdict. Synthetic and real results are reported separately.

## 0. Design phase (this session — no code)

- [x] 0.1 Read the owner requirements document (2026-11-26) and the training directive.
- [x] 0.2 Review repository components (models, mini_lab, incremental.py, normal_bank, collapse, phenology, field-id status) — design.md §10.
- [x] 0.3 Write proposal, design and five capability specs with evidence tags.
- [x] 0.3b Revision r2 (owner request 2026-11-26 00:41): anomaly head + weak supervision + anomaly memory; r2 phenology-agnostic definition; multi-signal monitoring and checkpoint selection; input data contract (units, L8/L9, normalization); score calibration arms; decision register. Documentation only.
- [ ] 0.4 Owner answers design.md §12 — answered 2026-11-26 except items 3–4 (δ values after E5, regression-set cells after E1, frozen before E5); this task closes when those two are decided.
- [x] 0.5 Owner records the cycle-timeline visual check (prerequisite for §5 Path B) — recorded 2026-11-26, artifacts at `reports/mini_dataset/figures/timeline_{seasonal,anomaly_score,flagged_dates}.png`. **Verdict: NOT all cycles detected** (`todo.md:44-64`, `docs/audits/cycle_gap_diagnosis_2026-11-26.md`); owner answers 2026-11-26: L02-02 gaps unresolved, R15-18 has multiple missed cycles (detection defect), `overlong_unresolved` flag approved. Path B and stage mapping stay blocked until cycle detection is repaired and re-reviewed.

## 0A. P0 — Input data contract (`input-data-contract`)

- [ ] 0A.1 Unit test pinning SR/ST scale, offset, valid DN ranges and fill to LSDS-1619; test that converted inputs are never re-scaled.
- [ ] 0A.2 Feature manifest (name, unit, conversion, valid range, source) for bands, indices, LST, CWSI, TVDI, ETa, weather, Δt, DOY; loader range assertions; DN-scale rejection test.
- [ ] 0A.3 Band-order assertion per scene; add `sensor_id` to the mini store; L8-only vs L9-only same-phase consistency check using `io/sensor_offset.py`.
- [ ] 0A.4 QA_RADSAT and ST_QA ingestion as quality signals; per-feature observation mask everywhere; no NaN→0 without mask; `min_obs` window exclusion with counts.
- [ ] 0A.5 Normalization arms N0–N4 with training-only, QA-valid, anomaly-excluded statistics; leakage test (2024+ in scope fails); normalization-erasure test; versioned statistics sidecar.

## 1. P0 — Convergence-controlled trainer (`training-convergence-control`)

- [ ] 1.1 `src/ssl_anomaly/train/convergence.py`: `PlateauMonitor` applied per guard signal (each objective + primary probe metric; relative delta, patience, EMA slope window, min_steps, LR drops) + `Verdict` enum incl. `OBJECTIVE_CONFLICT` flag; unit tests on synthetic curves: plateau, still-improving at cap, flat at cap, NaN divergence, collapse, total-flat-but-one-objective-improving, total-improving-while-one-objective-degrades.
- [ ] 1.1b Representation monitors: per-objective val losses, effective rank, dims-above-floor, off-diagonal covariance energy, fixed-budget frozen-encoder probes (phase linear probe, next-NDVI probe, same-field-same-phase kNN, frozen gap reconstruction), linear CKA to previous probe checkpoint, top-K rank stability.
- [ ] 1.1c Eligibility-filtered `best.pt` (primary probe metric among non-collapsed, non-conflicting checkpoints) + `best_loss.pt` + eligibility log in the manifest.
- [ ] 1.2 Checkpoint manager: atomic `last.pt`/`best.pt`, time-based saves, SIGINT handler, full state (RNG, sampler, early-stop, EMA target); test exact resume equality on a tiny model.
- [ ] 1.3 Verbose logger: per-eval line + `progress.jsonl` + end summary with resume command and extra-step extrapolation; test output fields.
- [ ] 1.4 Retrofit `mini_lab/runner.py` (replace fixed epochs/`max_steps` loop; keep `max_steps` as a cap) and `scripts/sweep_mini_architectures.py` (to-plateau + equal-step cut reporting).
- [ ] 1.5 `--resume` / `--init-from` / `--from-scratch` CLI semantics; fail-closed test.
- [ ] 1.6 Throughput probe script: steps/s and peak RAM per arm on the owner laptop (owner runs, < 2 min each); record in `reports/mini_dataset/cpu_budget.json`.

## 2. P0 — Model hygiene

- [ ] 2.1 Make `_mamba` and `_cnn_mamba` distinct or merge; registry duplicate guard + test.
- [ ] 2.2 Fix JEPA full-field `'NoneType' object is not subscriptable` and Barlow batch-size-1 failure (or exclude with recorded reason).
- [ ] 2.3 Per-observation tokens with observation mask and Δt in mini pixel/patch inputs (no NaN→0 without a mask channel).
- [ ] 2.4 Capacity matching utility (±20 % band) + report field.

## 3. P1 — Provenance

- [ ] 3.1 Extend `field-id-provenance-fix`: verify the +1 alias on all 490 fields (IoU per field), fail closed on any exception; owner decides on canonical re-index.

## 4. P1 — Hybrid representation (`hybrid-pixel-context-representation`)

- [ ] 4.1 Context branch: shared masked 2D CNN over k×k (k ∈ {3,5,7,9,11}), pooled token per date; boundary-leak byte-identity test.
- [ ] 4.2 Fusion module (gated add / FiLM / concat; early / late).
- [ ] 4.3 Multi-head module: reconstruction, causal forecast, JEPA latent predictor (EMA target), Barlow projector; configurable λ.
- [ ] 4.4 Augmentation policy + augmentation-erasure test.
- [ ] 4.5 Small-footprint injection (1, 3×3, 9×9 px) in the synthetic harness, labelled `synthetic-injection`.
- [ ] 4.6 Owner runs E1 and E2 on mini to plateau (3 seeds); agent writes the finding (next free number) with both to-plateau and equal-step tables.

## 5. P1 — Statistical discovery and triage (`anomaly-discovery-triage`)

- [ ] 5.1 `src/ssl_anomaly/discovery/statistical.py`: temporal robust z, within-field spatial z + Gi*, harmonic residual, multi-band consistency; independence test (no model import).
- [ ] 5.2 Rank fusion, tiers, quarantine store; test that Suspected never enters update pools.
- [ ] 5.3 Spatial Anomaly Event extraction (connected in-field pixels, ≥2 consecutive acquisitions).
- [ ] 5.4 Active-learning packet generator (uncertainty × diversity × spread, blind, random-normal mix-in) + review record schema.
- [ ] 5.5 Report generator that refuses precision/recall fields without a verified set.
- [ ] 5.6 Anomaly memory (verified / unresolved / rejected partitions, versioned) and its use as an exclusion list for normal memory, replay and normalization statistics.
- [ ] 5.7 Data-category guard: batch assemblers for encoder SSL and anomaly head enforce the design §2.1 table; audit test.
- [ ] 5.8 Score calibration per component (calibration set disjoint from encoder training; p-value / robust z / conformal arms; conditioning arms) + aggregation arms AG-single/AG-max/AG-rank/AG-wcal; refusal on uncalibrated fusion; stability metrics (E6).
- [ ] 5.9 Anti-confirmation-loop audit: model-vs-random selection field in review packets, separate precision reporting, model-surfaced-positive share per update, detector freeze per release.
- [ ] 5.10 Anomaly head (H-none/H-sup/H-cw/H-pu with class-prior sweep), gated on `min_verified`; "insufficient labels" path; Platt/isotonic calibration on held-back Verified.

## 6. P2 — Phenology dual path (`phenology-dual-path`)

- [ ] 6.1 `phenology/stages.py`: Joshi 2024 SOS/EOS pipeline → soft stage posterior with confidence; plant/ratoon from corrected age.
- [ ] 6.2 Agnostic and aware scorers; deferral below `c_min`.
- [ ] 6.3 Robustness harness (±16/±32 day shifts, 10 % swaps); DSH-bin comparator arm.
- [ ] 6.4 Path A per r2: trajectory-preserving residuals, trajectory-matched and DOY-matched own-history references, implicit temporal inputs as ablations; test that Path A output is invariant to stage-estimate changes.
- [ ] 6.5 Combined path (confidence-weighted A+B); three-path report rows.

## 7. P2 — Continual protocol (`continual-update-protocol`)

- [ ] 7.1 Replace `train/incremental.py` stub: versioned model registry (parent, hashes, norm stats, memory snapshots).
- [ ] 7.2 Stratified replay buffer; distillation loss; optional EWC.
- [ ] 7.3 Drift detector (MMD/energy on embeddings, KS on bands).
- [ ] 7.4 Regression set freeze (hash) + acceptance gate (BWT, forgetting, FWT, δ values from 0.4).
- [ ] 7.4b Versioned normalization/calibration/sensor-offset/feature-manifest binding per model version; freeze-on-update default; drift-gated re-normalization with regression-set re-scoring.
- [ ] 7.5 Owner runs E5 on mini (scratch / warm / warm+replay / warm+replay+distill / +EWC) with year-by-year updates 2013→2022.

## 8. P3 — Adaptation and scale

- [ ] 8.1 E4 adaptation arms (frozen, linear probe, partial FT, full FT).
- [ ] 8.1b Owner runs E6 (score aggregation) and E7 (normalization × sensor handling) on mini to plateau.
- [ ] 8.2 Port the selected configuration to the owner's 490-field pixel store (separate change) after 3.1.
- [ ] 8.3 Findings document (EN + FA) with synthetic vs real sections and claim rung.
