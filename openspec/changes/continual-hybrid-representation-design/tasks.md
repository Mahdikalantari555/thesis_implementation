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

- [x] 0A.1 Unit test pinning SR/ST scale, offset, valid DN ranges and fill to LSDS-1619; test that converted inputs are never re-scaled. — `src/ssl_anomaly/io/data_contract.py` (constants verified against the guide PDF, Table 6-1/6-4), `tests/unit/test_data_contract.py` (28 tests); literals in `config.py`, `io/landsat.py`, `io/zonal.py`, `scripts/extract_patches.py` consolidated to the pinned source; commit 983d74c.
- [x] 0A.2 Feature manifest (name, unit, conversion, valid range, source) for bands, indices, LST, CWSI, TVDI, ETa, weather, Δt, DOY; loader range assertions; DN-scale rejection test. — `FEATURE_MANIFEST` + `validate_feature_values`/`validate_feature_set` wired into `mini_lab/data.build_dataset` (fail closed); commit 983d74c. ETa ceiling deliberately absent (not pre-registered; owner-open), LST soft window flagged not fatal.
- [x] 0A.3 Band-order assertion per scene; add `sensor_id` to the mini store; L8-only vs L9-only same-phase consistency check using `io/sensor_offset.py`. — `assert_band_order` in `build_dataset`; per-observation `sensor_id` written on all 3 farms (`scripts/add_sensor_ids_mini.py`, 538 dates: 406 L8 / 132 L9, idempotent); measured report `reports/mini_dataset/l8_l9_consistency.json` (LST offsets resolved negative on all farms; reflectance-band offsets flip sign across farms → no global correction); `io/sensor_consistency.py` + tests; commit da84a40.
- [x] 0A.4 QA_RADSAT and ST_QA ingestion as quality signals; per-feature observation mask everywhere; no NaN→0 without mask; `min_obs` window exclusion with counts. — `build_mini_dataset.py` ingests `qa_radsat.tif`/`st_qa.tif` when present (absence recorded via `n_*_dates` attrs; current scenes carry qa_pixel only); `min_obs` exclusion (default 2) with per-unit counts in the run manifest; mask carried structurally (`Split.observed`; 2.3 adds it as an input channel); `_farm_groups` fails closed on empty stores; commit 469c5d0.
- [x] 0A.5 Normalization arms N0–N4 with training-only, QA-valid, anomaly-excluded statistics; leakage test (2024+ in scope fails); normalization-erasure test; versioned statistics sidecar. — `ssl_anomaly/normalization.py` (`train_end` leakage error incl. the 2024+ spec scenario; `exclude_mask` poison-invariance test; `norm-stats/v1` sidecar with data sha256/git sha; erasure record ≥50 % retention on all 5 arms); tests `tests/unit/test_normalization_arms.py` (16); commit cd36003.

## 1. P0 — Convergence-controlled trainer (`training-convergence-control`)

- [x] 1.1 `src/ssl_anomaly/train/convergence.py`: `PlateauMonitor` applied per guard signal (each objective + primary probe metric; relative delta, patience, EMA slope window, min_steps, LR drops) + `Verdict` enum incl. `OBJECTIVE_CONFLICT` flag; unit tests on synthetic curves: plateau, still-improving at cap, flat at cap, NaN divergence, collapse, total-flat-but-one-objective-improving, total-improving-while-one-objective-degrades. — 16 spec-scenario tests; total tracked but never a guard signal; commit 94d3f22.
- [x] 1.1b Representation monitors: per-objective val losses, effective rank, dims-above-floor, off-diagonal covariance energy, fixed-budget frozen-encoder probes (phase linear probe, next-NDVI probe, same-field-same-phase kNN, frozen gap reconstruction), linear CKA to previous probe checkpoint, top-K rank stability. — `train/monitoring.py` (pure NumPy, deterministic sample caps, kNN reports its chance baseline); 18 tests; commit 94d3f22.
- [x] 1.1c Eligibility-filtered `best.pt` (primary probe metric among non-collapsed, non-conflicting checkpoints) + `best_loss.pt` + eligibility log in the manifest. — `train/checkpointing.py::BestTracker` (missing primary metric = ineligible with recorded reason); 6 tests; commit c4fa12f.
- [x] 1.2 Checkpoint manager: atomic `last.pt`/`best.pt`, time-based saves, SIGINT handler, full state (RNG, sampler, early-stop, EMA target); test exact resume equality on a tiny model. — `atomic_torch_save` (tmp+replace, tmp cleaned on failure), `CheckpointManager` (time-based + SIGINT save), full RNG state roundtrip; exact-resume trajectory equality test (spec scenario); commit c4fa12f.
- [x] 1.3 Verbose logger: per-eval line + `progress.jsonl` + end summary with resume command and extra-step extrapolation; test output fields. — `train/progress.py::EvalLogger` + `resume_command`; field-coverage tests; commit c4fa12f.
- [x] 1.4 Retrofit `mini_lab/runner.py` (replace fixed epochs/`max_steps` loop; keep `max_steps` as a cap) and `scripts/sweep_mini_architectures.py` (to-plateau + equal-step cut reporting). — `mini_lab/trainer.py` plateau loop with DEV split (never test: 2023 dev year / intra-train slice recorded in manifest), guard = family objective + season-phase probe; sweep `--plateau` mode with per-seed verdicts and the §4.5 equal-step cut scored from saved eval checkpoints (`score_state_dict`); smoke on the real store (cut step 102); commit 3848692.
- [x] 1.5 `--resume` / `--init-from` / `--from-scratch` CLI semantics; fail-closed test. — resume modes auto (loud fresh start / auto-resume) | resume (fail closed when absent) | init-from (fail closed when missing) | from-scratch (refuse existing last.pt); `config_path` recorded so resume commands cite the exact YAML; fail-closed tests; commit 3848692.
- [x] 1.6 Throughput probe script: steps/s and peak RAM per arm on the owner laptop (owner runs, < 2 min each); record in `reports/mini_dataset/cpu_budget.json`. — `scripts/probe_mini_throughput.py`; agent-run smoke 6/6 arms ok (mamba 12-13 steps/s, peak ~1 GB at T=30 sliding windows); FULL PROBE IS OWNER-RUN (handover in workflow entries); commit 70b3caa.

## 2. P0 — Model hygiene

- [x] 2.1 Make `_mamba` and `_cnn_mamba` distinct or merge; registry duplicate guard + test. — distinct configs (compact d_state=8 vs wide d_state=16/depth≥2); `assert_models_distinct` refuses same-signature collisions (duplicate-injection test); commit 206deab.
- [x] 2.2 Fix JEPA full-field `'NoneType' object is not subscriptable` and Barlow batch-size-1 failure (or exclude with recorded reason). — ROOT CAUSE: `_barlow_loss` with a trailing batch of 1 produced NaN std → NaN weights → NaN scores → None AUPRC → the sweep print crash (finding 28's recorded error was the print, not the model). Fixed: population std + shape check + canonical off-diagonal (diagonal excluded, was double-counted); `BarlowTwinsProjector` batch-1 BN guard (eval-mode BN with visible fallback counter); sweep print reports None honestly. Verified on the real store: JEPA full-field trains finite, AUPRC 0.639. Commits 206deab, 83b7a78. NOTE: representation-family absolute values are not cross-comparable with finding 28 (loss formula + d_in changed); within-run gain-over-random controls unaffected.
- [x] 2.3 Per-observation tokens with observation mask and Δt in mini pixel/patch inputs (no NaN→0 without a mask channel). — input = [values(F) | mask(F) | dt/30(1)] = 2F+1 columns; value 0 ⟺ mask 0 per entry; losses/scores slice to value columns; channel provenance in every report (real store: 26 value → 53 input columns); commit 20e22f0.
- [x] 2.4 Capacity matching utility (±20 % band) + report field. — `mini_lab/capacity.py` (`match_capacity` deterministic grid search + trace; `capacity_report_field` per-arm ratio/in_band); sweep rows carry `capacity` (gru 6.53× mlp correctly flagged out-of-band); commit 1534b31.

## 3. P1 — Provenance

- [x] 3.1 Extend `field-id-provenance-fix`: verify the +1 alias on all 490 fields (IoU per field), fail closed on any exception; owner decides on canonical re-index. — DONE BY WP-1's E0 audit (folded in per the workflow dossier): `docs/audits/e0_cycle_completeness_2026-11-27.md` + `reports/e0_cycle_completeness/e0_results_2026-11-27.json` part B — 490/490 fields verified, IoU own-polygon min 1.0 (n_ge_0d99 = 490, n_lt_0d80 = 0), `best_not_own_fields` empty, verdict PASS, method = best-coverage-date archive pixel membership vs rasterised reference_fid = archive_fid − 1. WP-2 re-read the committed JSON and confirmed all 490 records pass. Canonical re-index stays an OWNER DECISION; the audit itself notes the G0 NaN/cloud-policy and L8/L9-harmonization arms are NOT covered by it.

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
