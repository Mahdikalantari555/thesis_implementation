# Finding 24 — Patch arms trained on the FULL archive (scale-up verdict)

> **⚠ SUPERSEDED — read before quoting any number below (2026-11-21).**
> The severity results in this finding were produced by the pre-fix protocol
> and are **not valid as detection measurements**: the alarm threshold was
> hardcoded (τ = 0.05 on the −log₁₀ p scale, never calibrated), which clean
> paths exceed by construction, so the flat "MDM80 = S1, detection 1.000,
> delay 30 d" ladder is an artifact of the threshold, not a measured
> detection capability (fixed in commit `bd65829`: calibrated τ + NaN-safe
> alarm paths + σ-table channel fix; see finding 25 for the corrected
> protocol). Additionally, the v1 chain itself (799 valid cycles over 370
> fields) is **superseded by the v2 chain** (3,478 valid cycles over 490
> fields, 4.4× recovery, same downstream architecture): by owner decision
> (2026-11-21) the v1 accuracy comparison was **not** carried out — the v2
> results (finding 25) replace this finding entirely. The v1 artifacts
> remain archived for provenance; do not cite this finding's severity
> numbers in the thesis.

**2026-11-20 · owner-executed training (Colab T4 per the handover notebook) +
agent-executed σ calibration and severity evaluation · rung L2 · claim noun:
synthetic anomaly**

## Headline

The finding-23 result survives the full-archive scale-up **identically**: both
patch arms (`patch_pixelwise`, `patch_cnn_gru`), each trained on the FULL
archive (447 locked-train field-cycles, 240 fields) × 3 seeds, reach
**MDM80 = S1 on all four families** (acute/persistent × ndvi/savi) with
median detection delay 30 d, on the SAME locked dev panel and eval-subset
hash (6186edf8a06f82d3) as findings 18/21/22/23. Every field-level arm
(B1, `gru_ae_main`, `gru_pred`) remains censored **>S6** on the identical
protocol. Per-cell detection at the S1 level: **1.000** (Wilson 95 % CI
[0.947, 1.0], n = 68 per cell = 2 rounds × 34 eval cycles), i.e. the
smallest, literature-realistic drop (−0.02 NDVI/SAVI) is detected in every
injected cell.

## Training provenance (owner-run, Colab T4)

- Command set = the handover PowerShell/notebook protocol exactly:
  `--epochs 10`, seeds {13, 42, 2026}, `--store data/processed/patches.zarr`,
  `--norm data/processed/norm_stats_full.json`, `--out
  checkpoints/patch/patch_full_ep10` (no `--cpu-threads` on GPU).
- Device **cuda** (T4); wall **235–259 s per seed** (~247 s mean; the local
  CPU estimate was 38–49 min/seed — the T4 ran ~10× faster than predicted
  15–40 min total; all 6 runs ≈ 25 min).
- git sha at training time **36b6161** (the pushed handover commit; the
  notebook cloned the repo at this HEAD).
- Results delivered as `patch_full_ep10_colab_results.zip` (47.7 MB, 84
  files = 6 run dirs × 14 files: epoch000–009.pt + last.pt +
  norm_stats.json + progress.jsonl + run_manifest.json) → extracted to
  `checkpoints/patch/patch_full_ep10/` locally.

## Binding verification (agent, fail-closed checks all passed)

- `validate_manifest(m, require_benchmark=False)` → **missing = []** for all
  6 manifests (incl. the top-level `seed` field; device=cuda recorded).
- Config bindings per manifest: `norm_sha256 = 3cb1ca60…` (the full-store
  norm, 447 locked-train cycles) ✓, `holdout_sha256 = 39284712…` ✓,
  `splits_sha256 = 9aa5eb11…` ✓; run-dir `norm_stats.json` self-hash equals
  the recorded one (loaded via `load_patch_norm`) ✓.
- **37 field-cycles excluded by the holdout manifest** per run — the
  standing invariant held on the full store at training time.
- `progress.jsonl`: 10 rows per run, losses finite and decreasing
  (pixelwise 0.041 → 0.037; cnn_gru 0.0375 → 0.0290), no NaN.
- Store binding unchanged from 2026-11-19: 370 fields / 799 recorded cycles,
  S = 48, `cycle_source_sha256 = 930a6cde…` == locked splits.

## Post-training chain (agent-executed)

1. **σ calibration ×6** (`calibrate_patch_sigma.py`, per run dir): dev-clean
   population = 69 locked dev cycles (same partition as every finding since
   18); fitted cells 6.7 % at min_n = 20, fallback covers the rest — the
   same shape as the PoC tables. Each σ npz binds model_sha256 +
   norm_sha256 + input_protocol.
2. **Severity evaluation ×6** (`eval_dev.py`, finding-18 protocol,
   `severity_sweep_patch_v1`, S1–S6 × {acute, persistent} × {ndvi, savi},
   rounds = 2): eval-subset hash **6186edf8a06f82d3** for all 6 — identical
   to the field-level arms' subset. Wall 430–520 s per seed on CPU.

## Result table (MDM80 at 5 % cycle-FAR, sentinel values per seed)

| arm | acute/ndvi | acute/savi | persistent/ndvi | persistent/savi |
|---|---|---|---|---|
| B1 (field-level, finding 18) | >S6 | >S6 | >S6 | >S6 |
| gru_ae_main (field-level) | >S6 | >S6 | >S6 | >S6 |
| gru_pred (field-level, finding 23) | >S6 | >S6 | >S6 | >S6 |
| patch_cnn_gru PoC (subset, finding 23) | S1 | S1 | S1 | S1 |
| patch_pixelwise PoC (subset, finding 23) | S1 | S1 | S1 | S1 |
| **patch_cnn_gru FULL ×3 seeds (today)** | **S1, S1, S1** | **S1, S1, S1** | **S1, S1, S1** | **S1, S1, S1** |
| **patch_pixelwise FULL ×3 seeds (today)** | **S1, S1, S1** | **S1, S1, S1** | **S1, S1, S1** | **S1, S1, S1** |

Median delay 30 d everywhere; zero seed variance at the MDM level. Reports:
`reports/dev/patch_full_ep10/metrics_{arch}_sev_{arch}_s{seed}.json`
(each carries eval hash, σ/σnorm hashes, holdout hash, params, device).

## Gate verdict (pre-registered rules applied verbatim)

- **Full-archive gate (finite MDM80 on the full store): PASS** — sentinel 1
  (S1) for both arms, all families, all seeds; no censoring anywhere in the
  ladder (per-cell detection 1.000 at every level, n = 68).
- **Scale-up rule (platform-patch-scorer-wiring design D2): SATISFIED** —
  G4 resolved against the CNN (finding 23: localization AUPRC gap −0.0003)
  AND the full-archive MDM80 is finite for `patch_pixelwise` → **frozen
  winner = `patch_pixelwise`** (the simpler, CNN-off arm; 36,020 params vs
  the CNN arm's ~91k). The freeze step (model card + sidecar + hashes) and
  the platform wiring remain **BLOCKED on owner approval** of the
  `platform-patch-scorer-wiring` change; nothing is wired yet.
- Notes for the record: the full-run change's wider final criterion (all
  arms incl. B0, gradual shapes, paired-bootstrap vs references) belongs to
  `patch-full-run-and-ablations` and is intentionally NOT evaluated here.

## Honest limitations (unchanged from finding 23, restated for scale-up)

- **Rung stays L2**: synthetic injections on the locked dev panel; no L4
  field evidence exists yet (stress-type data collection is the owner's
  active task). "Anomaly", never "stress".
- **S1 detection is a cycle-level aggregate effect** — the G4 localization
  probe showed per-pixel localization is at chance for S1–S3 (0.29–0.37 vs
  prevalence ≈ 0.30) and only emerges from S4 (0.95–0.97 on SAVI at S6).
  The platform's flags will say *where/when something is off at cycle
  granularity*, not which pixels.
- **Coverage boundary**: training/eval covers the 370 cycle-bearing fields
  (of 478 with any detected cycle, from the 490-field registry); the 108
  excluded fields are a cycle-detector boundary (see the phase-0 diagnosis),
  not a model verdict on them.
- **Single dev panel**: all numbers live on the 34-cycle evaluation subset
  (n = 68 with 2 rounds). The L3 generalization matrix (new years/farms) is
  the next rung and stays gated in `patch-generalization-suite`.
- dev_log rows are intentionally absent for patch arms (the exporter's
  `[dev_log] patch report only` branch is pre-existing behavior — finding 23
  documented the same); this finding file is the record.

## Provenance summary

- Training: Colab T4, cuda, git **36b6161**, seeds {13, 42, 2026}, epochs 10,
  wall 235–259 s/seed; results zip 47.7 MB verified (84 files).
- Store: `data/processed/patches.zarr` (370 × 799, S=48, epoch 2013-03-30,
  cycle_source 930a6cde…); norm **3cb1ca60…**; holdout **39284712…**;
  splits **9aa5eb11…**.
- σ tables: per-checkpoint, dev-clean 69 cycles, min_n 20.
- Evaluation: `severity_sweep_patch_v1`, eval subset **6186edf8a06f82d3**,
  rounds 2, device cpu, agent machine, git **36b6161**.
