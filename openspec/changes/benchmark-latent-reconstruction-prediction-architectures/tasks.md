# Tasks — benchmark-latent-reconstruction-prediction-architectures

TDD-first. The agent never launches long jobs; training/evaluation at scale is
handed to the owner as exact Colab cells after the local gate (AGENTS.md §6.1):
unit tests green → `--smoke` run end-to-end → smoke output read and inspected →
exact command/notebook written out → `--resume` verified. Laptop budget for dev:
≤15 min/seed, ≤250k params for probes; anything bigger goes to T4.

## Phase 0 — Repository inspection and protection

- [ ] 0.1 Inspect repository structure, existing `ssl_anomaly` modules, frozen
      artifacts and the v2 chain bindings; record the inventory in the design
      (no new assumptions beyond §Dataset).
- [ ] 0.2 Regression gate: existing unit battery green before any new module
      lands; previously frozen findings/reports hash-verified unchanged.
- [ ] 0.3 Freeze references: findings ≤ 25 read-only; B1/GRU-AE interfaces
      documented as integration constraints.
- [ ] 0.4 Commit the complete experiment matrix (`configs/benchmarks/matrix.yaml`)
      and its hash into the pre-registration before any execution.

## Phase 1 — Experimental foundation

- [ ] 1.1 Common architecture interface (train/infer/score/checkpoint) + tests.
- [ ] 1.2 Config schema + manifest schema (pydantic/Hydra-style, matching repo
      conventions) + validation tests.
- [ ] 1.3 Seed management ({13, 42, 2026}), split loader with hash verification,
      checkpoint interface, logging, `progress.jsonl`, `--resume`.
- [ ] 1.4 Metric interfaces (AUPRC, AUROC, F1 + val-threshold rule, delay,
      Spearman/Kendall, Brier/ECE, latency, VRAM, params, FLOPs estimator) + tests.
- [ ] 1.5 Experiment registry (SQLite or JSONL; states: Pending/Running/
      Completed/Failed/Interrupted/Deferred/Not Applicable) + resume-safe state
      transitions + tests.

## Phase 2 — Architecture implementation (A1–A10)

- [ ] 2.1 GRU-AE (reuse/adapt incumbent contract) and A1 score path.
- [ ] 2.2 Mamba-AE (A2) behind the feasibility gate.
- [ ] 2.3 Latent reconstruction models A3/A4 + collapse prevention (stop-gradient
      or EMA target documented) + collapse diagnostics + tests.
- [ ] 2.4 Future predictors A5/A6 (causal one-or-multi-step; k pre-registered) + tests.
- [ ] 2.5 GRU/Mamba JEPA A7/A8 (EMA target encoder; sg; content-preservation
      probe mandatory) + tests.
- [ ] 2.6 Hybrid A9/A10 (λ_r/λ_p pre-registered; sensitivity arms {0.5,1,2}) + tests.
- [ ] 2.7 Parameter counts recorded; capacity band (±20 %) verified; forward/
      backward passes, checkpoint save/restore tests for all ten.

## Phase 3 — Anomaly suppression (S0–S3)

- [ ] 3.1 S0 baseline objective; 3.2 S1 soft suppression (bounded weights,
      non-reinforcing); 3.3 S2 masked learning (distinct from missing-data
      handling); 3.4 S3 synthetic-anomaly suppression (train generator
      independent of eval generator).
- [ ] 3.5 Architecture×suppression applicability matrix committed (applicable /
      not-applicable + reason); invalid combos marked `Not Applicable` — never
      substituted.
- [ ] 3.6 Suppression diagnostics + loss-component logging + unit tests;
      contamination isolation test.

## Phase 4 — Synthetic anomaly generation

- [ ] 4.1–4.5 Five family generators (spectral, temporal, phenological,
      multivariate, persistent) + 4.6–4.8 severities Weak/Moderate/Strong +
      4.9 combined scenarios.
- [ ] 4.10 Exact ground-truth metadata (mask, boundaries, family, severity,
      seed, parameters) + 4.11 tests + 4.12 train/test stream isolation check +
      4.13 realism report (magnitude distributions vs MAD) in the manifest.

## Phase 5 — Hierarchical spatial-temporal path (gated on v2 patch re-extraction)

- [ ] 5.1 Confirm v2 patch store + norm v2 sidecar bindings (dependency: owner
      action in `todo.md`); if absent → all patch-level cells `Deferred` with
      reason; field-mean path unaffected.
- [ ] 5.2 Patch encoder + documented farm aggregation; 5.3 patch- and
      farm-level dual scoring; 5.4 consistent patch sampling/valid-mask
      control; 5.5 hierarchical inference tests; 5.6 missing/invalid patch tests.

## Phase 6 — Weather and phenology conditioning

- [ ] 6.1 I1/I2/I3 input builders (daily ERA5 causal windows; weather
      conditioning W1; separate weather encoder W2 with latent fusion).
- [ ] 6.2 Phenological context via `PhaseAxis` (DSH default; GDD/landmark if
      selected by the phenology-method work); no fabricated labels.
- [ ] 6.3 Temporal alignment tests (no future information); I-ablation tests.

## Phase 7 — Normality and calibration

- [ ] 7.1 Hybrid normality model wiring (farm history + context); 7.2 C1
      farm-specific, 7.3 C2 farm×stage, 7.4 C3 global; 7.5 train/calibration-
      only verification test; 7.6 calibration diagnostics; 7.7 holdout-split
      calibration checks.

## Phase 8 — Classical baselines

- [ ] 8.1 Isolation Forest, 8.2 One-Class SVM, 8.3 Robust PCA behind the common
      interface; 8.4 GRU-AE/B1 references reused; 8.5 input-compatibility
      documentation; 8.6 common-evaluation smoke.

## Phase 9 — Validation protocols and metrics

- [ ] 9.1 V1 temporal holdout (dev discipline), 9.2 V2 farm holdout
      (block-consistent), 9.3 V3 spatiotemporal (train+dev only; holdout
      exclusions asserted by tests); locked test unread everywhere.
- [ ] 9.4–9.13 Metrics: AUPRC (event/point/patch/farm; macro family×severity
      headline with prevalence), AUROC, F1 (val thresholds), Detection Delay,
      Spearman, Kendall, Brier/ECE, latency, VRAM, params, FLOPs.
- [ ] 9.14–9.17 TVDI, CWSI_ET, CWSI_LST, ETa, ETrF integrations with
      shared-input classification labels; 9.18 dependency-aware uncertainty.

## Phase 10 — Statistical analysis

- [ ] 10.1 Paired bootstrap (B ≥ 1,000; farm/event resampling); 10.2 Wilcoxon;
      10.3 Friedman + post-hoc (Nemenyi/Conover); 10.4 effect sizes (Cliff's δ /
      rank-biserial) with CIs; 10.5 Holm–Bonferroni (+ BH for screening);
      10.6 seed/farm/temporal/event variance decomposition; 10.7 architecture-
      family and suppression interaction analyses; 10.8–10.10 three winner
      reports (detection / generalization / Pareto multi-objective).

## Phase 11 — HPO and fairness

- [ ] 11.1 Per-architecture search spaces + equal budget table committed;
      11.2 validation-only selection verified by test (selection code path
      rejects test split); 11.3 parameter- and 11.4 compute-matched
      configurations; 11.5 FLOPs/runtime recorded; 11.6 no-test-tuning audit.

## Phase 12 — Budget-aware execution

- [ ] 12.1 Matrix generation + prioritization; 12.2 budget estimation (smoke
      profile) ; 12.3 T4 device detection + mixed precision; 12.4 VRAM
      monitoring; 12.5 checkpoint/resume end-to-end test; 12.6 interruption
      resume test; 12.7 budget-exhaustion path → `Deferred` visibility test;
      12.8 incomplete-matrix report test.

## Phase 13 — The Colab notebook

- [ ] 13.1 Create notebook (20 sections per design §The Colab notebook); loads
      implemented modules only; Drive bundle output with dated self-describing
      name; 13.2 dataset/split verification cells; 13.3 screening cells; 13.4
      full benchmark cells with resume; 13.5 reporting/export cells; 13.6 clean-
      Colab dry run (USER-RUN).

## Phase 14 — Research reporting

- [ ] 14.1 Sections A–J reports; 14.2 final findings document (EN + FA; number
      assigned at completion) distinguishing best-detection/generalization/
      trade-off, statistically supported vs inconclusive findings, limitations
      (incl. incomplete-coverage statement); 14.3 `todo.md` updated; 14.4
      machine-readable evidence pack exported (JSON/CSV).

## Phase 15 — Final verification

- [ ] 15.1 Full unit battery green (state counts); 15.2 regression tests for
      previous experiments unchanged; 15.3 leakage safeguard tests (synthetic
      train/test isolation; calibration; test isolation); 15.4 registry audit:
      every experiment has an explicit final status; 15.5 manifests spot-checked
      for completeness; 15.6 report reviewed against AGENTS.md §7 checklist
      (leakage, circularity, seeds, baselines, compute fairness, rung tags,
      claim discipline); 15.7 EN/FA findings parity reviewed.

## USER-RUN (handed to the owner, never launched by the agent)

§2–§13 real runs (screening, full benchmark, rolls) are owner-executed on
Colab T4 after the local gate, or as explicitly scoped laptop probes. Exact
commands/cells are written at hand-over time with cwd, env, flags, expected
wall time and output paths.
