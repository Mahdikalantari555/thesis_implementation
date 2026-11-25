# Possible improvements & next-session roadmap (2026-11-25)

Status of the base result: finding 28 (`docs/findings/28_mini_cpu_architecture_screening.md`) —
**L2 screening only**. Current best control-adjusted architecture: **pixel-level
`mamba_ae`**, +0.171 AUPRC over same-architecture random init, t95 CI
[0.120, 0.221], 3 seeds. The result is on a **magnitude-injection proxy**; the
real weak-label probe is negative. Nothing in this document licenses an L3+ claim.

## 1. Ranked improvements (impact / effort / rung)

| # | Improvement | Why it matters | Effort | Rung it can reach |
|---|---|---|---|---|
| 1 | **Correct the canonical 490-field ID alias at source** | Findings 18–25, splits and every farm attribute are shifted by one; blocks all L3/L4 work | medium, owner-gated (hash-bound splits) | prerequisite |
| 2 | **Phase-aware target instead of −1.0 NDVI step** | The current positive is magnitude-only; AGENTS.md requires phenology-phase anomalies | medium | L2 → L3 |
| 3 | **Incremental / never-from-zero training** (§2) | Saves compute, tests continual adaptation across years/farms, matches the "new data is reserved for validation" vision | high | L3 |
| 4 | **Literature-derived sugarcane stages instead of 36 bins** (§3) | Stage-conditioned baselines are what the literature supports; fixed DSH bins are a convenience artifact | medium | L2/L3 |
| 5 | **Fix `_mamba` vs `_cnn_mamba` duplication** (`models.py:168-193`) | Otherwise "all architectures" overstates the search by one | small | honesty |
| 6 | **Cross-year / ≥3-field roll + phase-aware permutation null** | Needed before any generalization or SSL-advantage wording | medium | L3 |
| 7 | **Independent agronomic modality (L4)** | irrigation/ECe/yield or expert review; without it the thesis caps at L3 | owner-gated | L4 |
| 8 | **Collapse-aware regularization** (VICReg/Barlow-style variance term, logged per-dim variance) | effective rank 1.65–1.31 in probes; near-collapse can fake gains | small | method quality |
| 9 | **Equal-parameter comparison** | mamba ~293k vs gru ~160k; cross-architecture ranking is confounded | small | fairness |
| 10 | **Rule/feature arm (A-rule, A-feat) on the same windows** | Research report: hand-crafted baselines beat SSL at small n; mandatory before L5 | small | L5 |

## 2. Incremental training — "never start from zero"

**Goal.** After the first model is trained, every new year, field, stage or task
starts from the existing checkpoint and improves it, with an explicit guarantee
that earlier competence is not lost. This is a design proposal, not yet implemented.

### 2.1 Core protocol
- **Checkpoint as the unit of work.** Every run keeps `last.pt` (weights +
  optimizer + scheduler + rng), `run_manifest.json`, `progress.jsonl`,
  `norm_stats.json`, data/split hashes and git SHA. Training must accept
  `--init-from <checkpoint>` in addition to `--resume`; resume continues the
  same run, init-from starts a new run from a prior model.
- **No silent re-initialization.** If `--init-from` is missing and a compatible
  checkpoint exists for the same architecture/config, fail closed or require an
  explicit `--from-scratch` flag with a recorded reason. A from-scratch run is
  only legitimate as a pre-registered baseline.
- **Frozen regression set.** Lock a small validation subset (farm×year×stage)
  before incremental updates. After each update the new model must not drop
  more than a pre-registered margin (e.g. 0.02 AUPRC) on it; otherwise reject
  the update ("no-forgetting gate").

### 2.2 Anti-forgetting mechanisms (in order of first implementation)
1. **Replay buffer**: store a stratified sample of old cycles (per farm, year,
   stage, anomaly family) with their embeddings; intermix a fixed fraction in
   every new update. Cheapest and most auditable.
2. **Distillation anchor (LwF)**: keep the previous model frozen as a teacher;
   add KL/feature-matching loss on old inputs so new predictions stay close.
3. **EWC-style penalty**: estimate parameter importance from the old task's
   Fisher diagonal; penalize drift of important weights.
4. **Progressive expansion**: when a new stage/task arrives, grow new branches
   initialised so the old function is preserved exactly at init, then train the
   new branch while old weights are frozen; optional Later re-tuning under the
   no-forgetting gate.
5. **Stage-conditional heads**: keep the temporal trunk and add a small head per
   literature stage; new stages add heads, not a new trunk.

### 2.3 Data-incremental path
- New acquisitions/fields are appended through the manifest-driven ingest path
  (no hard-coded 538-date count), reusing the verified field-ID alias.
- Normalization statistics and σ/thresholds are updated from **train-only**
  data with the previous stats versioned; old runs keep their original sidecars.
- Drift detection: compare new-domain embeddings to the stored replay
  distribution; if the shift is large, route to the expansion path instead of
  fine-tuning blindly.

### 2.4 Experiment to decide whether it works
Pre-register three arms at **equal total optimiser steps**:
`A-scratch` (from zero on all data), `A-warm` (init-from previous checkpoint),
`A-warm+replay` (init-from + replay + distillation). Metrics: target-task AUPRC,
**backward transfer** (regression-set delta), and wall time. Only `A-warm*`
with non-negative backward transfer and a CI excluding zero wins; otherwise
report that incremental training saves compute but does not improve accuracy.

## 3. Phenology stages from the literature — drop the 36-bin method

**What the literature actually supports** (finding 13, Joshi-2024 note,
STS-Net/Albahli/Nie notes):
- **Joshi et al. 2024** (Remote Sensing 16:4244): sugarcane SOS/EOS from
  Landsat/S2 via BISE (30-day window) → 14-day resample → FFT →
  difference-of-Gaussians troughs; FAO two-crop model for growth stages;
  SOS/EOS vs visual labels R² 0.72–0.84, RMSE 17–27 days.
- **Growth stages to use**: `tillering` → `grand growth` → `maturity/ripening`
  (plant/ratoon shift in calendar time; stage conditioning, not DOY, is the
  transferable feature). Stress timing from the perturbation catalogue:
  P1/P2/P3 predominantly **grand growth** (mid-season, Jul–Sep in Khuzestan),
  P4 **early/tillering**, P6 **late maturity** — and P6 is confounded with the
  normal pre-harvest drying-off (OQ6, advisor question).
- Nie/Albahli: two or three well-chosen stage windows beat indiscriminate
  full stacks; condition fusion on developmental stage rather than calendar.

**Method change.** Replace the fixed 36 × 15-day DSH bin assignment with:
1. derive per-field-cycle SOS/EOS and harvest trough (Joshi pipeline, using the
   corrected pixel IDs);
2. map elapsed fraction / thermal time onto the three named stages with
   `plant`/`ratoon` from the (now-corrected) `age` attribute;
3. condition the baseline and the injection on **stage**, not bin index;
4. keep the old DSH-bin arm as a pre-registered comparator during the switch.

## 4. Next session — cycle review with PNGs (owner visual check)

Generated now from the corrected store (real pixels, real cycles, weak flags):
- `reports/mini_dataset/figures/timeline_seasonal.png` — NDVI/LST with cycle
  start/end and suspected flags, per farm;
- `reports/mini_dataset/figures/timeline_anomaly_score.png` — phase-conditioned
  robust deviation (derived, **not** model inference);
- `reports/mini_dataset/figures/timeline_flagged_dates.png` — flagged dates and
  year×month heatmap;
- `reports/mini_dataset/figures/timeline_readme.md` — provenance and caveats.

The owner's question in the next session: **"Are all detectable cycles
detected?"** (missed winter harvests / false merges / phantom cycles). Until
that visual confirmation is recorded, do not rebuild splits or claim the stage
mapping is correct. After confirmation: implement §3, then rerun the
architecture screen with the phase-aware target; then §2 incremental arms.

## 5. Open questions (advisor / owner)
- OQ6: pre-harvest irrigation cut-off timing, burning practice, typical NDVI
  drop magnitude, waterlogging months (needed to bound P6 and P1 severity).
- Does the +1 ID alias hold for all 490 fields? (verified only for 3.)
- Accept batch-size-1 failure of full-field Barlow/JEPA as out-of-scope, or
  rebuild those arms with a larger batch and re-run?

## 6. Commit policy for this work
Text and metrics JSON are committed; checkpoints (`*.pt`) and MLflow DBs under
`reports/mini_dataset/arch_sweep/` are ignored (binaries travel by Drive, never
GitHub). Per-run JSON reports and `sweep.json` stay committed so the numbers are
re-derivable from the recorded hashes.
