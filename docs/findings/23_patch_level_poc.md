# Finding 23 — Patch-level PoC (controlled trial, first real runs)

**2026-11-18 · agent-executed under the owner's run mandate ("run all scripts that need data and submit results") · rung L2 · claim noun: synthetic anomaly**

## Environment defects found and fixed first (all committed)

1. **numpy LAPACK × torch MKL duplicate-OpenMP crash** (root cause of the
   five "environmental" `0xc06d007f` crashes recorded since 11-07): any
   process that imports torch and then calls a numpy LAPACK routine
   (`lstsq`, `solve`, …) aborts with `OMP Error #15 (libomp.dll vs
   libiomp5md.dll)`. Fix: `KMP_DUPLICATE_LIB_OK=TRUE` +
   `PATH=$PREFIX/Library/bin` (the geo env is invoked without activation).
   After the fix: test_decomposition_study 20 passed, the four legacy
   crashers 24 passed.
2. **zarr 3 API** in a test fixture (`create_dataset` removed; c2486fd).
3. **Geometry memo correction** (023ccf6): the proposal's 16.7/19.6 px were
   equivalent-square sides (√area), not bounding boxes. Real bboxes (all 490
   fields): median 12×34, max 34×47; 69.8 % of fields exceed 32 px. The
   fail-closed geometry gate caught this on the first extraction. Run default
   **S = 48** set BEFORE the first training run (no goalpost movement).
4. **Holdout manifest v2** (6d06849): the arena enforcement applied the
   2024-01-01 cutoff to every non-test-block field, blocking training on 25
   dev calibration cycles that share ZERO (field, cycle) pairs with the 171
   locked test cycles. Fixed to the locked splits' recorded
   `dev_temporal_cutoff` deviation (design.md: temporal holdout is for
   non-block fields); farm rules stay absolute. Manifest sha 39284712….
5. R0.5 test isolation + reference population (74732dd), `bbox_origin`
   dtype u2→i2 (74732dd), dev_log schema columns (74732dd), Windows AV-lock
   write retries (7417a65), smoke scene-selection overlap (1ba1cd4),
   per-seed patch metrics filenames + `--b1-run-id` + σ-producer
   (`calibrate_patch_sigma.py`, 5f2c085 and follow-ups).

## PoC trial verdict (patch-level-controlled-trial)

Pre-registered subset: blocks 0+1 of the locked train split, 56 fields
(`data/processed/patch_poc_subset.json`, sha 40f5a975…). Stores:
`patches_poc.zarr` (56 train-block fields, 138 cycles) and
`patches_poc_eval.zarr` (94 = subset + 38 dev fields) at S=48, 36 bins ×
15 d, 8 channels. Norm d9fe7438… (105 train cycles). Devices: cpu. Seeds
{13, 42, 2026}; 10 epochs/seed after `--estimate-runtime` (90.9 s/epoch × 30
would break the pre-registered ≤15 min/seed budget; measured 9.0–11.4 min/seed).

**MDM80 at 5 % cycle-FAR (eval subset hash 6186edf8… — identical to the
finding-18 protocol):**

| arm | acute/ndvi | acute/savi | persistent/ndvi | persistent/savi |
|---|---|---|---|---|
| B1 (field-level, finding 18) | >S6 | >S6 | >S6 | >S6 |
| gru_ae_main (field-level) | >S6 | >S6 | >S6 | >S6 |
| gru_pred (field-level, today) | >S6 | >S6 | >S6 | >S6 |
| **patch_cnn_gru ×3 seeds** | **S1** | **S1** | **S1** | **S1** |
| **patch_pixelwise ×3 seeds** | **S1** | **S1** | **S1** | **S1** |

- **G1 smoke: PASS.** G2 overfit-one-batch: PASS with note (loss
  0.534→0.206 monotone on one batch; not zero — stochastic masking + trim +
  wd make 0 the wrong target on this objective).
- **G3 finite MDM80: PASS** — and the sentinel drops from 7 (>S6, all
  field-level arms incl. B1) to **1 (S1)**: the patch representation
  detects the *smallest, literature-realistic* drop (−0.02 NDVI/SAVI) at
  80 % cell detection, median delay 30 d. Both CNN and pixelwise reach S1,
  so the gain comes from the per-pixel spatial representation, not the CNN.
- **G4: INDETERMINATE** — MDM80 ties at S1; the footprint-0.3 localization
  AUPRC probe is not implemented yet (task 2.4 records this honestly).
- **G5: estimate before run PASS; "≤2× gru_ae_main compute" FAILS by
  letter** (~10–20×: patch inputs are 48×48×8 grids vs 18-dim field tokens).
  Recorded verbatim; the ≤15 min/seed laptop budget is the operative
  constraint the estimate was reconciled against.
- **Scale-up decision: NOT YET** — G4 must resolve (localization probe) and
  the S1 result must survive the full-run gates before any production claim.

## gru_pred trial (predictive-ssl-controlled-trial) — honest negative

Trained 3 seeds (~48 s/seed), severity sweep (same hash), frozen
(gru_pred_main, L2, seed 13), parts A–D + locked final test:
- Part A: clean cycle-FAR 0.029/0.0/0.029 — within budget, no concentration.
- Part B: cross-check vs finding-18 EXACT (b1/gru_ae recomputed, 0
  mismatches); MDM80 ">S6" everywhere.
- Part C: **absorption refutes the hypothesis** — gru_pred A = 0.33–0.67
  (absorbs MORE than gru_ae's α 0.38–0.50 at S6/savi: 0.67 vs 0.45).
- Final test (171 test cycles): pre-registered margin −0.0058,
  `a_met_test = false`; gates B non-evaluable (both censored); rung stays L2.
  Path C is dead by its own pre-registered rule (third consecutive negative:
  findings 18, 21, now the predictive arm).

## Same-day supporting runs

- **R0.5 zonal-dist study** (finding 24 input): 4 arms on the 516-cycle
  train+dev population (test never read), FAR achieved 0.043 ≤ target;
  richer distribution stats do not lower FAR (level 30 alarms vs dist 14,
  but top-field share 0.07→0.14). `reports/dev/r05_20261118/`.
- **Spatial-representation benchmark**: materialize-per-field wins 2.58×;
  throughputs R1 2.22 / R2 3.70 / R3 1.75 / pixelwise 3.40 samples/s.
  `reports/bench/spatial_20261118T175350Z.json`. (Bench specs still say
  32/16 px — throughput ratios, not absolute numbers, are the finding.)
- **Decomposition-aware SSL study** (`--stage all`, real archive): 431
  fields, 206,298 band eval cells + 103,149 index eval cells, 6
  method×protocol arms, wall 8862 s. `data/studies/decomposition_aware_ssl/all/summary.json`
  (decomposition task 3.2: source coverage + real paired results recorded;
  architecture-gate extraction from benchmark.json arms is the read-out step).
- Farm-local: library complete (39 tests) — integration runs await tasks
  2.2–5.3 (code, not runs). Generalization matrix: design-blocked pending
  per-fold stores. Full plan (`run_plan.py`): gated on the trial verdict —
  correctly NOT run.

## Provenance

git sha at run time 023ccf6/5f2c085→; splits 9aa5eb11…; holdout 39284712…;
norm d9fe7438…; eval subset 6186edf8…; severity table unchanged (finding 18).
All reports carry params, wall time, device, seeds, hashes.
**Persian write-up of this finding (task 4.2, persian-writing skill): pending.**
