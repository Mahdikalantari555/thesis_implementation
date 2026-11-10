# Tasks: train-and-finalize-main-architecture

**Prerequisite:** `data-and-phenology-readiness` decision record approved (its task 4.2) before sections 2–5 start.

Rule for all sections: the agent runs only unit tests and `--smoke` runs. Full training/evaluation commands are written for the user (AGENTS.md §6).

## 0. Documents and open questions
- [x] 0.1 AGENTS.md §6 (long-running jobs) and §7 (critical ML review checklist)
- [x] 0.2 `docs/methods_explained.md` (Persian teaching document for the metrics and methods)
- [ ] 0.3 Perturbation literature review → moved to `data-and-phenology-readiness` task 1.4
- [ ] 0.4 Open question to advisor (todo.md OQ6): which perturbation types and magnitudes are realistic for Amirkabir sugarcane? Update the interim set in design D6 after the answer
- [x] 0.5 Add `docs/vision.md` note on incremental training + active learning from farmer reports

## 1. Locked splits
- [ ] 1.1 `scripts/build_locked_splits.py`: contiguous blocks from field centroids (k-means, recorded seed), 2024+ test, 2023 dev, train cycles ending before 2024-01-01, monitoring-only set for censored cycles; log counts and dropped cycles
- [ ] 1.2 Write `data/processed/splits_locked.json` with hashes; loader guard refusing test access outside `freeze_main_model.py --final-test`
- [ ] 1.3 Unit tests: disjointness, block exclusivity, hash guard, test-access guard
- [ ] 1.4 User checkpoint: report split sizes; if the temporal cutoff drops more than 15 % of cycles, ask the user

## 2. Inputs
- [ ] 2.1 Mask-aware tensor builder (value + observation mask per band) in `real_archive.py`; remove `nan_to_num` from training paths
- [ ] 2.2 Train-only robust normalization per band × phase bin, saved to sidecar
- [ ] 2.3 Unit tests: no NaN reaches the model, masked entries give zero loss, stats computed from train ids only

## 3. Perturbations and dev metric
- [ ] 3.1 MAD-scaled injection in `synthetic.py`; add growth stall, early senescence, transient drop with recovery; keep onset bin as ground truth
- [ ] 3.2 `scripts/eval_dev.py`: cycle-level FAR calibration on clean dev, detection rate, median delay, partial-cycle AUC-PR, NRMSE, severity curve, per-type and per-phase tables; writes `metrics.json` and appends to `reports/dev_log.csv`
- [ ] 3.3 Unit tests: detection delay on a hand-made cycle, FAR calibration hits 5 % on synthetic normal scores, forbidden-target test (no CWSI/ETa/y_stress/yield in dev paths)

## 4. Models and references
- [ ] 4.1 Causal `b1_phasez_causal` reference with train statistics
- [ ] 4.2 Causal Masked Transformer (attention mask, masked reconstruction on observed bins only, ≤500 k params)
- [ ] 4.3 Causal GRU-AE at matched parameter count (±20 %)
- [ ] 4.4 No-LST variant switch for the main model
- [ ] 4.5 Unit test: causality (appending future bins does not change earlier scores)

## 5. Training script
- [ ] 5.1 `scripts/train_main.py`: device auto-select, seeds `[13, 42, 2026]`, tqdm progress (epoch, losses, ETA), `progress.jsonl`, `--resume`, `--smoke`, early stopping on dev reconstruction loss, wall-time logging
- [ ] 5.2 Agent runs smoke tests for both models; then hands the user the full commands (one per model) with expected time on laptop

## 6. Development loop (run by user, logged)
- [ ] 6.1 Architecture screen: Masked TF vs GRU-AE vs B1 causal, 3 seeds each
- [ ] 6.2 Single-change experiments on the chosen model until plateau; each row in `dev_log.csv`
- [ ] 6.3 LST vs no-LST ablation at the chosen config
- [ ] 6.4 Agent reviews each dev log update with the AGENTS.md §7 checklist before proposing the next change

## 7. Freeze and final test
- [ ] 7.1 `scripts/freeze_main_model.py`: write `checkpoints/<id>_main.pt` + sidecar (config, stats, τ, k, hashes, seeds, dev metrics, params, `ladder_rung`)
- [ ] 7.2 User runs `--final-test` once; report with bootstrap CIs over fields; set `ladder_rung` to L3 only if the target holds on test

## 8. Platform integration (contract v5)
- [ ] 8.1 Registry: add `gru_ae` slot, make `masked_transformer` model-backed, remove CWSI proxy fallback for SSL slots
- [ ] 8.2 Exporter: causal per-observation scores, `stressFlag`, `stressFlagModel`, `ladderRung`
- [ ] 8.3 `channels.ts` / `data.ts` mirror + legacy v4 handling; contract and bilingual tests; tsc and vite build; screenshot runtime gate
- [ ] 8.4 Update `docs/platform_score_methods.md` with exact formulas and file:line references

## 9. Incremental placeholder
- [ ] 9.1 `ssl_anomaly/train/incremental.py` stub raising `NotImplementedError` with docstring
