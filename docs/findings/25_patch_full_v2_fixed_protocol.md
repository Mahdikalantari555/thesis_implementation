# Finding 25 — Patch arms on the v2 chain under the CORRECTED severity protocol

**2026-11-21 · owner-executed evaluation (Colab T4 per the handover notebook) · rung
L2 · claim noun: synthetic anomaly · protocol `severity_sweep_patch_v2` · git
`e46e9a5a`**

This is the finding the corrected protocol produces; it replaces the superseded
severity numbers of finding 24 (supersession banner there, commit `64d5a3a`).
The protocol itself carries the three fix layers: the bd65829 ladder repair
(calibrated τ + NaN-safe alarm mask + σ-table channel fix), the 99709b0 σ fit on
per-channel-observed residuals (A1 — refit proved byte-identical on the v2
store, so these runs used exactly the tables in
`checkpoints/patch/patch_full_v2_ep10/`), and the ea74a0c onset-true
detection/delay with rotating rounds (A2, field-ladder parity). Audit context:
`docs/audits/code_audit_2026-11-21.md`.

## Headline

Under the corrected, honestly calibrated protocol, **MDM80 is censored >S6 for
both patch arms, all four families, and all three seeds**. The ladder is live
(detection rises with severity — no flatness), τ is calibrated per seed at a 5 %
cycle-FAR target (achieved 0.0471 = 8/170 everywhere), and the best measured
detection anywhere in the ladder is **0.456 (cnn_gru, S6/persistent/savi, seed
13; Wilson 95 % [0.404, 0.509], n = 340)** with median delay 30 d from onset.
Nothing reaches the 0.8 MDM80 target inside S1–S6.

The pre-fix headline "MDM80 = S1, detection 1.000, delay 30 d" (findings 23/24)
was an artifact of the hardcoded τ = 0.05 that clean paths exceed by
construction; the corrected protocol removes it, and with it the entire
patch-vs-field MDM80 gap of findings 23/24 disappears: the field-level arms
(B1, `gru_ae_main`, `gru_pred`) were already honestly censored >S6 in finding 18,
and now the patch arms are too.

## Detection table (per-cell rate, n = 340 per cell = 170 eval cycles × 2 rotating rounds)

Rates pooled over the three seeds (per-seed spread quoted separately below):

| family | arm | S1 | S2 | S3 | S4 | S5 | S6 |
|---|---|---|---|---|---|---|---|
| acute/ndvi | patch_pixelwise | 0.012 | 0.012 | 0.012 | 0.012 | 0.018–0.021 | **0.059** |
| acute/ndvi | patch_cnn_gru | 0.015–0.021 | 0.015–0.024 | 0.018–0.026 | 0.021–0.032 | 0.053–0.085 | **0.106–0.218** |
| acute/savi | patch_pixelwise | 0.012 | 0.012 | 0.012 | 0.012 | 0.029–0.032 | **0.185–0.191** |
| acute/savi | patch_cnn_gru | 0.015–0.021 | 0.018–0.024 | 0.018–0.032 | 0.032–0.050 | 0.103–0.188 | **0.315–0.450** |
| persistent/ndvi | patch_pixelwise | 0.012 | 0.012 | 0.012 | 0.015 | 0.018–0.024 | **0.062** |
| persistent/ndvi | patch_cnn_gru | 0.012–0.024 | 0.015–0.024 | 0.021–0.029 | 0.024–0.035 | 0.059–0.082 | **0.109–0.209** |
| persistent/savi | patch_pixelwise | 0.012 | 0.012 | 0.012 | 0.015 | 0.050–0.056 | **0.206–0.209** |
| persistent/savi | patch_cnn_gru | 0.012–0.024 | 0.015–0.026 | 0.021–0.032 | 0.035–0.059 | 0.124–0.203 | **0.324–0.456** |

MDM80 (lowest level with detection ≥ 0.8): **>S6 censored** for every
(arm, family, seed) — recorded sentinel 7.0 in all six metrics JSONs; recomputed
from the cells by the verification script, agrees.

Median delay (from the injection onset, `(fb − onset)·15`): **30 d at S6**
(2 bins) for every family/arm; 15–30 d through S3–S5; the sparse S1–S3
"detections" are late alarms (medians 45–127.5 d) — consistent with the
false-alarm floor, not with signal.

## Calibration record (per seed, from the metrics JSONs)

| arm | seed | τ (normalized −log10 p scale) | far_achieved | wall (T4, s) |
|---|---|---|---|---|
| patch_pixelwise | 13 | 1.5138 | 0.0471 | 298.2 |
| patch_pixelwise | 42 | 1.5138 | 0.0471 | 284.6 |
| patch_pixelwise | 2026 | 1.5055 | 0.0471 | 288.6 |
| patch_cnn_gru | 13 | 1.4221 | 0.0471 | 289.2 |
| patch_cnn_gru | 42 | 1.4871 | 0.0471 | 285.3 |
| patch_cnn_gru | 2026 | 1.4358 | 0.0471 | 284.4 |

τ is chosen by `mscoring.calibrate_threshold` on the leave-one-out calibration
null alarm paths (k = 1, FAR target 0.05, recorded verbatim in
`per_seed.tau_calibration`); the achieved FAR is 8/170 null paths in every run.

## Observations worth carrying forward

1. **SAVI families respond; NDVI families barely.** At S6, SAVI detection is
   2–5× the NDVI detection for both arms (cnn_gru 0.438–0.456 vs 0.109–0.218;
   pixelwise 0.185–0.209 vs 0.059–0.062). The mechanism is NOT established —
   the band-consistent injector realizes the drop on the index channel, and the
   band-space signature of an SAVI-unit drop evidently aligns better with what
   the scorer's residual channels see. Recorded as an open question; do not
   narrate a mechanism the code does not pin down.
2. **cnn_gru ≥ pixelwise from S4 upward** (~2× at S6/SAVI); pixelwise is nearly
   seed-invariant (per-seed rates identical to ±1–2 cycles of 340) while cnn_gru
   has real seed spread (seed 42 weaker: 0.315–0.324 vs 0.438–0.456 at
   S6/SAVI). This reverses the direction of the (invalidated) G4-based winner
   call at the extreme-severity end, but **neither arm reaches MDM80**, so no
   winner is declared here.
3. **The information floor is the story.** Even the largest ladder step
   (S6 = −0.40 NDVI/SAVI units ≈ 6.1× the across-field MAD of 0.066, realized
   bands-consistently on the raw patch) leaves
   the alarm statistic below the calibrated τ for more than half the cycles of
   the best arm. This is the same conclusion the field-level arms reached in
   finding 18 and the deep-research severity analysis anticipated (real stress =
   0.5–2.5×MAD vs a detection floor around 6–8×MAD): the ladder's lower half is
   below what this scorer/calibration can see at a 5 % cycle-FAR budget.

## Consequences (pre-registered rules applied to the corrected numbers)

- **Full-archive gate (finite MDM80 on the full store): FAIL** for both arms —
  the corrected numbers void the finding-24 gate verdict and the
  "frozen winner = patch_pixelwise" scale-up call built on it. Nothing was ever
  wired to the platform (the `platform-patch-scorer-wiring` change was blocked
  on owner approval and stays dead unless a future protocol produces a finite
  MDM80).
- The fixture-level gate (`gate_patch_severity.py`, 20 synthetic cycles) still
  passes with S6 detection ≥ 0.8 — that asserts mechanism liveness, not
  real-data sensitivity; the honest sensitivity number is the real panel's, and
  it is this finding's tables.
- Findings 18/19/21/22 (field-level, censored) stand unchanged; the corrected
  picture is coherent across arms: no arm reaches MDM80 within S1–S6 on the dev
  panel at 5 % cycle-FAR.

## Honest limitations

- **Rung stays L2**: synthetic injections on the locked dev panel; "anomaly",
  never "stress"; no L4 evidence exists.
- **Single panel, small n at the top**: every cell is 170 eval cycles × 2
  rounds (n = 340); the Wilson CIs at S6/SAVI ([0.39–0.41, 0.49–0.51] for
  cnn_gru) are far below 0.8, so the censoring verdict is unambiguous.
- **Post-onset FAR is not separately calibrated**: τ targets the 5 % budget on
  full-path null alarms; post-onset-only FAR is ≤ that by construction
  (conservative direction), which is why S1–S3 rates (1.2–3.5 %) sit below the
  4.71 % achieved budget.
- **Sensor axis structural ceiling**: all v2 cycles are sensor 0, so the σ
  table's sensors 1–2 are empty by construction (33.3 % fitted is the ceiling);
  recorded limitation carried from the σ rebuild.
- dev_log rows are intentionally absent for patch arms (pre-existing exporter
  branch); this finding file is the record.

## v1 supersession (owner scope decision, 2026-11-21)

The v1 chain (799 valid cycles over 370 fields) was replaced entirely by the v2
chain (3,478 valid cycles over 490 fields, 4.4× cycle recovery, same downstream
architecture). By owner decision the v1 accuracy comparison was **not** carried
out — v1 was dropped because it discovered far fewer cycles than v2, not
because of a measured accuracy difference. Finding 24's supersession banner
records this; v1 artifacts remain archived for provenance only.

## Provenance

- Evaluation: owner-run on Colab T4 (device `cuda`), git **e46e9a5a** (the
  notebook clones HEAD; all protocol fixes inside), `severity_sweep_patch_v2`,
  seeds {13, 42, 2026}, wall 284–298 s per run; input protocol
  `patch_inputs_v2`, injection basis `bands_consistent`, scoring alignment
  `one_step_ahead_v3`, 8,160 injections per run (170 cycles × 24 cells × 2
  rotating rounds), eval-subset hash **852e1ba3d2a2a9d0** (identical to the
  field-level arms' panel), splits **30ecbe10…**, holdout **d501bf81…**, norm
  **493f0395…**, severity-table hash **9d720e3b…** (== local code).
- σ tables: the six runs bind, per seed, the canonical σ-table shas of the
  locally rebuilt tables (bd65829 fix + 99709b0 refit-proved-byte-identical) —
  verified per run by the agent's verification script.
- Results zip: `patch_full_v2_sigma_eval_colab_v2.zip`, sha256
  **f4b41ec12be51a2a7aa406e67da5e69af0b9d3ae5e3501c76737995be0357bff** (12,306
  bytes, 6 metrics JSONs, CRC OK), returned 2026-11-21; extracted to
  `reports/dev/patch_full_v2_ep10/metrics_{arch}_sev_{arch}_s{seed}.json`.
- Agent verification (fail-closed, all green): protocol tag, git sha, all
  hashes above, σ self-sha consistency, τ > 0.05, far_achieved ≤ 0.05,
  24 cells × n = 340, ladder non-flat with S6 ≥ S1 in every family, onset
  rotation {6, 18, 30} recorded per cell (= T//6, T//2, 5T//6 with T = 36),
  delays within onset semantics, MDM80 recomputed from cells == recorded.
