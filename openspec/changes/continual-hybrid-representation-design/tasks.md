# Tasks — continual-hybrid-representation-design

TDD-first. Laptop CPU only (owner directive 2026-11-26). The agent writes code
and unit tests and runs only smoke tests (< ~2 min, AGENTS.md §6); every real
training run is launched by the owner with an exact command and always ends
with a convergence verdict. Synthetic and real results are reported separately.

## 0. Design phase (this session — no code)

- [x] 0.1 Read the owner requirements document (2026-11-26) and the training directive.
- [x] 0.2 Review repository components (models, mini_lab, incremental.py, normal_bank, collapse, phenology, field-id status) — design.md §10.
- [x] 0.3 Write proposal, design and five capability specs with evidence tags.
- [ ] 0.4 Owner answers design.md §12 (CPU/RAM, reviewer, δ values, regression set, wall cap).
- [ ] 0.5 Owner records the cycle-timeline visual check (prerequisite for §5 Path B).

## 1. P0 — Convergence-controlled trainer (`training-convergence-control`)

- [ ] 1.1 `src/ssl_anomaly/train/convergence.py`: `PlateauMonitor` (relative delta, patience, EMA slope window, min_steps, LR drops) + `Verdict` enum; unit tests on synthetic curves: plateau, still-improving at cap, flat at cap, NaN divergence, collapse.
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

## 6. P2 — Phenology dual path (`phenology-dual-path`)

- [ ] 6.1 `phenology/stages.py`: Joshi 2024 SOS/EOS pipeline → soft stage posterior with confidence; plant/ratoon from corrected age.
- [ ] 6.2 Agnostic and aware scorers; deferral below `c_min`.
- [ ] 6.3 Robustness harness (±16/±32 day shifts, 10 % swaps); DSH-bin comparator arm.

## 7. P2 — Continual protocol (`continual-update-protocol`)

- [ ] 7.1 Replace `train/incremental.py` stub: versioned model registry (parent, hashes, norm stats, memory snapshots).
- [ ] 7.2 Stratified replay buffer; distillation loss; optional EWC.
- [ ] 7.3 Drift detector (MMD/energy on embeddings, KS on bands).
- [ ] 7.4 Regression set freeze (hash) + acceptance gate (BWT, forgetting, FWT, δ values from 0.4).
- [ ] 7.5 Owner runs E5 on mini (scratch / warm / warm+replay / warm+replay+distill / +EWC) with year-by-year updates 2013→2022.

## 8. P3 — Adaptation and scale

- [ ] 8.1 E4 adaptation arms (frozen, linear probe, partial FT, full FT).
- [ ] 8.2 Port the selected configuration to the owner's 490-field pixel store (separate change) after 3.1.
- [ ] 8.3 Findings document (EN + FA) with synthetic vs real sections and claim rung.
