# Tasks — benchmark-latent-reconstruction-prediction-architectures

TDD-first. The agent never launches long jobs; training/evaluation at scale is
handed to the owner as exact Colab cells after the local gate (AGENTS.md §6.1):
unit tests green → `--smoke` run end-to-end → smoke output read and inspected →
exact command/notebook written out → `--resume` verified. Laptop budget for dev:
≤15 min/seed, ≤250k params for probes; anything bigger goes to T4.

## Phase 0 — Repository inspection and protection

- [x] 0.1 Inspect repository structure, existing `ssl_anomaly` modules, frozen
      artifacts and the v2 chain bindings; record the inventory in the design
      (no new assumptions beyond §Dataset).
- [ ] 0.2 Regression gate: existing unit battery green before any new module
      lands; previously frozen findings/reports hash-verified unchanged. *local gate ran 2026-11-23 — `tests/benchmarking` 139 passed. `git diff HEAD` over tracked files is empty (this change adds files only), and `tests/` as a whole carries 51 pre-existing failures / 31 collection errors from environment- and data-dependent modules that do not import `benchmarking`; they are reported, not hidden, and no 'battery green' claim is made for them. Findings ≤ 26 and `configs/evaluation/pre_registration.yaml` are byte-identical to HEAD.*
- [x] 0.3 Freeze references: findings ≤ 25 read-only; B1/GRU-AE interfaces
      documented as integration constraints.
- [ ] 0.4 Commit the complete experiment matrix (`configs/benchmarks/matrix.yaml`)
      and its hash into the pre-registration before any execution. *matrix + `matrix_sha256 03a7b6cd6c4f…` committed to the repo, and every run re-derives the matrix and refuses to start on drift. The 'into the pre-registration' half is deliberately NOT done by the agent: `configs/evaluation/pre_registration.yaml` is a frozen artifact — an owner action to add the benchmark hash there.*

## Phase 1 — Experimental foundation

- [x] 1.1 Common architecture interface (train/infer/score/checkpoint) + tests.
- [x] 1.2 Config schema + manifest schema (pydantic/Hydra-style, matching repo
      conventions) + validation tests.
- [x] 1.3 Seed management ({13, 42, 2026}), split loader with hash verification,
      checkpoint interface, logging, `progress.jsonl`, `--resume`.
- [x] 1.4 Metric interfaces (AUPRC, AUROC, F1 + val-threshold rule, delay,
      Spearman/Kendall, Brier/ECE, latency, VRAM, params, FLOPs estimator) + tests.
- [x] 1.5 Experiment registry (SQLite or JSONL; states: Pending/Running/
      Completed/Failed/Interrupted/Deferred/Not Applicable) + resume-safe state
      transitions + tests.

## Phase 2 — Architecture implementation (A1–A10)

- [x] 2.1 GRU-AE (reuse/adapt incumbent contract) and A1 score path.
- [x] 2.2 Mamba-AE (A2) behind the feasibility gate.
- [x] 2.3 Latent reconstruction models A3/A4 + collapse prevention (stop-gradient
      or EMA target documented) + collapse diagnostics + tests.
- [x] 2.4 Future predictors A5/A6 (causal one-or-multi-step; k pre-registered) + tests.
- [x] 2.5 GRU/Mamba JEPA A7/A8 (EMA target encoder; sg; content-preservation
      probe mandatory) + tests.
- [x] 2.6 Hybrid A9/A10 (λ_r/λ_p pre-registered; sensitivity arms {0.5,1,2}) + tests.
- [x] 2.7 Parameter counts recorded; capacity band (±20 %) verified; forward/
      backward passes, checkpoint save/restore tests for all ten.

## Phase 3 — Anomaly suppression (S0–S3)

- [x] 3.1 S0 baseline objective; 3.2 S1 soft suppression (bounded weights,
      non-reinforcing); 3.3 S2 masked learning (distinct from missing-data
      handling); 3.4 S3 synthetic-anomaly suppression (train generator
      independent of eval generator).
- [x] 3.5 Architecture×suppression applicability matrix committed (applicable /
      not-applicable + reason); invalid combos marked `Not Applicable` — never
      substituted.
- [x] 3.6 Suppression diagnostics + loss-component logging + unit tests;
      contamination isolation test.

## Phase 4 — Synthetic anomaly generation

- [ ] 4.1–4.5 Five family generators (spectral, temporal, phenological,
      multivariate, persistent) + 4.6–4.8 severities Weak/Moderate/Strong +
      4.9 combined scenarios. *(generators + severities done; **4.9 combined scenarios not implemented** — a cycle can carry several families at once, and the eval cell grid treats them independently, but there is no explicit 'designed combined scenario'. Also honest about what a fixture is: the labels are injected, so arm↔index agreement is not evidence about real agriculture.)*
- [x] 4.10 Exact ground-truth metadata (mask, boundaries, family, severity,
      seed, parameters) + 4.11 tests + 4.12 train/test stream isolation check +
      4.13 realism report (magnitude distributions vs MAD) in the manifest.

## Phase 5 — Hierarchical spatial-temporal path (gated on v2 patch re-extraction)

- [x] 5.1 Confirm v2 patch store + norm v2 sidecar bindings (dependency: owner
      action in `todo.md`); if absent → all patch-level cells `Deferred` with
      reason; field-mean path unaffected.
- [x] 5.2 Patch encoder + documented farm aggregation; 5.3 patch- and
      farm-level dual scoring; 5.4 consistent patch sampling/valid-mask
      control; 5.5 hierarchical inference tests; 5.6 missing/invalid patch tests.

## Phase 6 — Weather and phenology conditioning

- [x] 6.1 I1/I2/I3 input builders (daily ERA5 causal windows; weather
      conditioning W1; separate weather encoder W2 with latent fusion).
- [x] 6.2 Phenological context via `PhaseAxis` (DSH default; GDD/landmark if
      selected by the phenology-method work); no fabricated labels.
- [x] 6.3 Temporal alignment tests (no future information); I-ablation tests.

## Phase 7 — Normality and calibration

- [ ] 7.1 Hybrid normality model wiring (farm history + context); 7.2 C1
      farm-specific, 7.3 C2 farm×stage, 7.4 C3 global; 7.5 train/calibration-
      only verification test; 7.6 calibration diagnostics; 7.7 holdout-split
      calibration checks. *C1/C2/C3 + diagnostics + the dev/test-fitting refusal implemented and tested. 7.7 holdout-split calibration is not: the contract selects thresholds on `dev` and reports on `dev`, so a calibrated-holdout check needs an owner-side carve-out. The optimism is therefore recorded inside each `thresholds.json` (`contract` + `note`) instead of being claimed away.*

## Phase 8 — Classical baselines

- [ ] 8.1 Isolation Forest, 8.2 One-Class SVM, 8.3 Robust PCA behind the common
      interface; 8.4 GRU-AE/B1 references reused; 8.5 input-compatibility
      documentation; 8.6 common-evaluation smoke. *IF / OC-SVM / Robust-Covariance PCA + random/chance references behind the common interface, input-compatibility table (`arms.COMPATIBLE_INPUTS`) and smoke-runnable. **8.4 B1 reference is not wired to the incumbent checkpoint** and 8.6's common-evaluation smoke did not include any reference arm — the 58 `B1`/`BL-*` matrix cells are still `Pending`.*

## Phase 9 — Validation protocols and metrics

- [x] 9.1 V1 temporal holdout (dev discipline), 9.2 V2 farm holdout
      (block-consistent), 9.3 V3 spatiotemporal (train+dev only; holdout
      exclusions asserted by tests); locked test unread everywhere.
- [x] 9.4–9.13 Metrics: AUPRC (event/point/patch/farm; macro family×severity
      headline with prevalence), AUROC, F1 (val thresholds), Detection Delay,
      Spearman, Kendall, Brier/ECE, latency, VRAM, params, FLOPs.
- [ ] 9.14–9.17 TVDI, CWSI_ET, CWSI_LST, ETa, ETrF integrations with
      shared-input classification labels; 9.18 dependency-aware uncertainty. *TVDI, CWSI_LST and within-field CWSI implemented with the label-circularity refusal and tested (a uniform index cannot rank above the anomaly). CWSI_ET reports `unavailable` because the archive has no aerodynamic-resistance term, and ETa/ETrF are `absent` until a weather sidecar is bound — none is approximated. 9.18 dependency-aware uncertainty is not implemented.*

## Phase 10 — Statistical analysis

- [ ] 10.1 Paired bootstrap (B ≥ 1,000; farm/event resampling); 10.2 Wilcoxon;
      10.3 Friedman + post-hoc (Nemenyi/Conover); 10.4 effect sizes (Cliff's δ /
      rank-biserial) with CIs; 10.5 Holm–Bonferroni (+ BH for screening);
      10.6 seed/farm/temporal/event variance decomposition; 10.7 architecture-
      family and suppression interaction analyses; 10.8–10.10 three winner
      reports (detection / generalization / Pareto multi-objective). *bootstrap (farm-clustered, B = 2000 contract floor 1000), Wilcoxon, Friedman + Nemenyi, Cliff's δ, Holm, variance decomposition and the three winner reports are in `stats.py`/`report.py` and rendered on a real run. **10.7 interaction analyses are not**: no architecture×family or suppression×architecture term is fitted, and with 1-seed fixture output it would be noise, not a result — this box stays open until a ≥2-seed run exists.*

## Phase 11 — HPO and fairness

- [ ] 11.1 Per-architecture search spaces + equal budget table committed;
      11.2 validation-only selection verified by test (selection code path
      rejects test split); 11.3 parameter- and 11.4 compute-matched
      configurations; 11.5 FLOPs/runtime recorded; 11.6 no-test-tuning audit. *spaces + equal-budget table committed, `d_model` removed from every space (it is the capacity knob), selection-on-forbidden-split refused at selection *and* re-checked by `audit_search`, params/FLOPs recorded. 11.3/11.4 (matched configurations chosen by a *real* search) need the owner's T4 HPO run; `run_hpo.py --fixture --smoke` only proves the plumbing and stamps itself L0.*

## Phase 12 — Budget-aware execution

- [x] 12.1 Matrix generation + prioritization; 12.2 budget estimation (smoke
      profile) ; 12.3 T4 device detection + mixed precision; 12.4 VRAM
      monitoring; 12.5 checkpoint/resume end-to-end test; 12.6 interruption
      resume test; 12.7 budget-exhaustion path → `Deferred` visibility test;
      12.8 incomplete-matrix report test.

## Phase 13 — The Colab notebook

- [ ] 13.1 Create notebook (20 sections per design §The Colab notebook); loads
      implemented modules only; Drive bundle output with dated self-describing
      name; 13.2 dataset/split verification cells; 13.3 screening cells; 13.4
      full benchmark cells with resume; 13.5 reporting/export cells; 13.6 clean-
      Colab dry run (USER-RUN). *notebook generated by `scripts/benchmarks/make_colab_notebook.py` (20 sections, imports only implemented modules, dated self-describing bundle dir, `--check` fails when stale, every code cell parsed at generation). 13.2–13.5 cells exist; **13.6 clean-Colab dry run is USER-RUN** — the agent does not launch GPU jobs, so the notebook has never been executed end-to-end.*

## Phase 14 — Research reporting

- [ ] 14.1 Sections A–J reports; 14.2 final findings document (EN + FA; number
      assigned at completion) distinguishing best-detection/generalization/
      trade-off, statistically supported vs inconclusive findings, limitations
      (incl. incomplete-coverage statement); 14.3 `todo.md` updated; 14.4
      machine-readable evidence pack exported (JSON/CSV). *`report.py` renders coverage-first §0–§8 (three winner tables, secondaries, variance, cost, the 14-question ledger, next-steps) with `report.json` + CSV/JSON evidence (14.4), and 14.3's owner task list is in `todo.md`. **14.2 the results finding is not written** — finding 27 (`docs/findings/27_benchmark_harness_local_gate.md`, EN + FA) records the harness and the gate; the results finding takes the next number when real numbers exist.*

## Phase 15 — Final verification

- [ ] 15.1 Full unit battery green (state counts); 15.2 regression tests for
      previous experiments unchanged; 15.3 leakage safeguard tests (synthetic
      train/test isolation; calibration; test isolation); 15.4 registry audit:
      every experiment has an explicit final status; 15.5 manifests spot-checked
      for completeness; 15.6 report reviewed against AGENTS.md §7 checklist
      (leakage, circularity, seeds, baselines, compute fairness, rung tags,
      claim discipline); 15.7 EN/FA findings parity reviewed. *15.3 (leakage: synthetic train/eval isolation, calibration-split refusal, test-split isolation), 15.5 and 15.7 are done; 15.1/15.2 cannot be claimed for the whole repo (see line 14); 15.4 closes when every executed cell has an explicit status — 277 are legitimately `Pending`; 15.6 is a human review pass on the first real report.*

## USER-RUN (handed to the owner, never launched by the agent)

§2–§13 real runs (screening, full benchmark, rolls) are owner-executed on
Colab T4 after the local gate, or as explicitly scoped laptop probes. Exact
commands/cells are written at hand-over time with cwd, env, flags, expected
wall time and output paths.
