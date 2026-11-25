# Finding 28 — CPU architecture / spatial-unit screening on the mini lab

**Date:** 2026-11-25 · **Git SHA:** `7dfeaa64298a2a520267bcfe5df72b249577c770` ·
**Label:** `mini-screening` · **Claim rung:** **L2 (controlled deviation, screening)** — nothing higher.

> **Verdict.** A positive, control-adjusted result exists **on the synthetic
> deviation task**: pixel-level `mamba_ae` beats a random-initialized encoder of
> the *same architecture* by **+0.171 AUPRC** (t95 CI [0.120, 0.221], 3 seeds,
> all positive). It does **not** transfer to the real weak suspected-stress
> labels, where SSL = random-init and raw features are better. This is a
> defensible L2 screening result, **not** a stress-detection, generalization,
> or L4 result.

## 1. Provenance prerequisite (completed first)
The 2026-11-25 rebuild exposed an off-by-one field ID: archive pixels under fid
*k* belong to polygon *k−1* (IoU 1.0 / 0.0 own) for all three `3farms-dev`
farms. Fixed additively in the mini store with measured alias `archive_fid =
reference_fid + 1` (openspec `field-id-provenance-fix`): alignment now verifies
IoU 1.0 on own fid, parity passes **11,242/11,242**, and the earlier two "QA
mismatch" failures vanished (they were the same shift). The canonical 490-field
archive is **not** re-indexed; that remains an owner decision.

## 2. Design (pre-registered before looking at the confirmation numbers)
- **Farms:** L02-02 and R12-08 (different blocks; alias applied). Protocol:
  `single_farm`, train L02-02 / test R12-08; split train ≤2022, dev 2023,
  test 2024+.
- **Screen:** all 9 registered architectures × {pixel, patch 1/3/5/7/9/11,
  full-field}, 1 seed, **100 optimiser steps per run** (enforced by
  `ModelConfig.max_steps`; verified `steps_done=100`).
- **Confirm:** 3 seeds {13, 42, 2026}, bigger models (d_model 64, latent 32,
  2 layers), **400 steps per run**, top units {pixel, patch9, patch11,
  full-field} × {mamba_ae, gru_ae, cnn_mamba}.
- **Mandatory controls in every run:** random-initialized encoder of identical
  architecture, step detector, cluster-aware CIs, checkpoint + progress.jsonl.
- **Target:** the harness's controlled one-channel NDVI drop (−1.0) on held-out
  windows. It is a magnitude proxy, explicitly **not** phase-aware anomaly or
  stress detection.

## 3. Screen (72 configs, 70 succeeded)
Ranking by gain over random-init is led by `mamba_ae` on patches (+0.107 to
+0.113, nearly flat across patch sizes 1–11), then `gru_ae` (+0.08–0.09).
Highest *absolute* AUPRC models were representation-family (JEPA 0.688, Barlow
Twins 0.678), but their random-init controls were higher still (0.746, 0.699):
architecture prior, not pretraining. Failures recorded verbatim:
`full_field__barlow_twins` (`Expected more than 1 value per channel ... torch.Size([1, 192])`)
and `full_field__jepa` (`'NoneType' object is not subscriptable`).
Full table: `reports/mini_dataset/arch_sweep/SCREEN_RESULTS.md`.

## 4. Confirmation (12 configs, all succeeded, 3 seeds, 400 steps)

| unit | model | AUPRC (mean ± sd) | random-init | gain over random-init | t95 CI of gain | all seeds > 0 |
|---|---|---|---|---|---|---|
| **pixel** | **mamba_ae** | **0.6917 ± 0.0205** | 0.5210 ± 0.0042 | **+0.1707** | **[+0.120, +0.221]** | yes |
| pixel | gru_ae | 0.6415 ± 0.0028 | 0.5316 ± 0.0058 | +0.1099 | [+0.085, +0.135] | yes |
| patch11 | mamba_ae | 0.7108 ± 0.0156 | 0.5585 ± 0.0084 | +0.1523 | [+0.086, +0.219] | yes |
| patch11 | gru_ae | 0.6395 ± 0.0265 | 0.5329 ± 0.0069 | +0.1066 | [+0.005, +0.208] | yes |
| patch9 | mamba_ae | 0.6841 ± 0.0537 | 0.5564 ± 0.0088 | +0.1276 | [−0.031, +0.286] | yes |
| patch9 | gru_ae | 0.6465 ± 0.0227 | 0.5308 ± 0.0055 | +0.1157 | [+0.032, +0.200] | yes |
| full_field | mamba_ae | 0.6623 ± 0.0066 | 0.5832 ± 0.0069 | +0.0791 | [+0.056, +0.103] | yes |
| full_field | gru_ae | 0.6283 ± 0.0055 | 0.5529 ± 0.0138 | +0.0754 | [+0.023, +0.128] | yes |

`cnn_mamba` produced **identical numbers to `mamba_ae` in every confirmed cell**.
Cause (file:line): `src/ssl_anomaly/mini_lab/models.py:168-193` — `_mamba` and
`_cnn_mamba` both construct `TinyCNNMamba(d_state=16)` when `d_model=64,
n_layers=2`, so they are the *same model*, not two independent architectures.
Do not count this as a replication.

**Decision on the best architecture / unit.** By the pre-registered rule
(smallest spatial unit not meaningfully worse than the next larger) and by
control-adjusted gain, the screening winner is **pixel-level `mamba_ae`**:
highest gain (+0.171, CI excludes zero), and patch sizes 1–11 were flat in the
screen. `patch11__mamba_ae` has the highest absolute AUPRC (0.711) but a lower
control-adjusted gain (0.152) because its random-init baseline is higher;
`patch9__mamba_ae` is too unstable (CI includes zero).

## 5. Counter-result that bounds the claim
On the corrected store, the weak-label representation probe
(`reports/mini_dataset/representation_first_pilot_fixed.json`, n=480 windows;
train ≤2022 / dev 2023 / test ≥2024) gives:
- retrieval R@1: SSL **0.1533** vs chance 0.1800 and random-init 0.2867 (gain −0.0400);
- suspected-label AUPRC on test: SSL **0.1108** [0.035, 0.177] = random-init
  0.1108, while raw features reach 0.1782 [0.064, 0.275];
- effective rank 1.65 (near-collapse).

So the positive gain is **task-specific**: SSL pretraining helps the controlled
magnitude-injection detector on this pair of fields, and shows no advantage on
the real weak labels. That matches the literature conclusion that SSL gains are
task-dependent and concentrated in label-scarce regimes
(`research/ssl_advantage_2026-11-25/REPORT.md`).

## 6. What this does and does not establish
- **Established (L2):** on two fields, equal compute, 3 seeds, a pretrained
  Mamba-family encoder detects controlled NDVI-drop deviations better than its
  own random-init copy, with CIs excluding zero for pixel, patch11 and full-field.
- **Not established:** stress detection (no L4 evidence), generalization (2 of
  3 fields; no cross-year/farm roll), phase-aware anomaly detection (target is
  magnitude-only), superiority at equal parameter count (mamba ~293k vs gru
  ~160k; within-architecture control is fair, cross-architecture is not), or
  any claim about the canonical 490-field archive (IDs still shifted).
- `mamba_ae` vs `cnn_mamba` is **not** a two-architecture replication under this
  config (see §4).

## 7. Artifacts
- `reports/mini_dataset/arch_sweep/screen/sweep.json` and
  `reports/mini_dataset/arch_sweep/confirm/sweep.json` — include per-seed values
  and all aggregates; per-run report/checkpoint directories are regenerable and
  kept local (gitignored, not shipped to GitHub).
- `reports/mini_dataset/representation_first_pilot_fixed.json` (+ embeddings)
- `reports/mini_dataset/parity.json` (11,242/11,242 pass)
- `docs/audits/mini_rebuild_3farms_probe_2026-11-25.md` (superseded run record)
- `research/ssl_advantage_2026-11-25/REPORT.md` (21-source evidence review)

## 8. Next (owner-gated)
1. Decide whether to correct the canonical 490-field ID mapping (separate change;
   splits/findings are hash-bound) — this blocks any L3+ claim.
2. Fix the `_cnn_mamba`/`_mamba` duplication (e.g. distinct `d_state`) and
   re-run the confirmation if an architecture comparison is to be claimed.
3. Add a phase-aware injection target and an independent agronomic modality
   before any "stress" wording; repeat on ≥3 fields with cross-year roll.
