# Model-Improvement Proposals — External Review Intake (2026-11-22)

**Status:** PROPOSAL — recorded per AGENTS.md §2.5 (scope discipline). Not approved, not
implemented. Any implementation goes through `openspec new change` first.
**Source:** external consultant review (2026-11-22, "Claude"); artifacts at
`.scratch/claude_stress_detector_2026-11-22/` (moved from repo root for hygiene).
**Independent verification:** ZCode agent reproduced the detector's outputs numerically
identically in the `geo` env on 2026-11-22 (see §Verification record).
**Ladder:** every claim here is L1–L2. No ground truth (L4) exists; the word "stress" is
not licensed yet (AGENTS §3, Q8 event-level framing).

## 0. Verified context

The consultant built a label-free, phenology-aware NDVI episode detector
(`stress_detector.py`: field-grouped out-of-fold GBM on NDVI ~ [phase, cday, clen, doy,
cycle_idx, start_doy] + per-date common-mode removal + per-phase-bin robust z ≤ −2 +
30 d harvest guard) and audited the production `gru_ae_main` flags against it.
Verified numbers (rerun 2026-11-22):

- residual robust SD ladder: phenology-only 0.070 → +date 0.055 → +date×farm-block
  0.048 (consultant; detector's own config measured 0.0555)
- 854 episodes / 366 fields; tiers 523 "A" (ΔLST ≥ +1 °C) / 153 "B" (NDVI-only) /
  178 rejected (ΔLST < −2 °C, cloud/shadow)
- GRU-flag audit (732 in-cycle non-edge flags): median cross-sectional NDVI z = −0.05;
  8.7% z ≤ −2 vs base 5.5%; median ΔLST +0.2 °C
- GRU-flag phase structure (agent-measured): 95.2% in phase 0.5–0.9, 0.0% in 0.3–0.5 —
  structured, not uniform-random
- Corrections to consultant prose: phase < 0.2 episodes = **258** (not 201); A+B mean
  ΔLST = **+4.82 °C, 88% positive** (stronger than quoted "+2.2 / 70%"); 2014-06-13
  flags = documented warm-up/edge artifact (excluded as edge everywhere); ≤3-day flag
  pairs (272/807) = **persistence-k alarm semantics by design**, not duplicates
- GRU-AE input = 8 channels [ndvi, blue, green, red, nir, swir1, swir2, lst]
  (`src/ssl_anomaly/eval/real_archive.py:308`, `io/patches.py:42`) with a causal
  one-step-ahead objective (`models/gru_ae.py:57`). The consultant's audit observes
  only NDVI/LST cross-sectionally (2/8 channels) and cross-field rather than
  own-history, so "GRU flags ≈ random" is NOT established. An 8-channel audit of GRU
  flags (SWIR1/SWIR2/red/nir z at flag dates) remains owed before any verdict.
- Tier label `A_confirmed(NDVI+LST)` must be renamed (e.g. `A_ndvi_lst_concordant`) —
  §8: no "confirmed" before L4.

## P1 — No-NN baseline "B2": phenology + date×neighbor (consultant step 1)

Claim (consultant, on the 77% of observations having a previous cycle):

| model | robust SD |
|---|---|
| phenology only | 0.070 |
| + date common effect (all fields) | 0.055 |
| + date × farm-block (20 blocks ≈ 22 fields) | **0.048** |
| + own-field previous-cycle NDVI path | no gain (0.070 → 0.071) |

Adoption rule proposed: every SSL arm must beat B2 on the MDE curve of the locked
170-cycle panel, otherwise the SSL model is unnecessary.

Agent review conditions before B2 becomes a ladder baseline:

1. Fit B2's normalization on the TRAIN split only — the consultant's quick fit used all
   data (in-sample for the ladder comparison; §8 leakage rule).
2. "Block" = `farm_code` prefix is a management-region proxy, not a spatial block
   (§8: hash-based blocks are not spatial blocks). Use real adjacency from
   `Data/RoI/Farms.geojson` (consultant concurs).
3. Consistency: aligns with LOYO SSL≈B1 and findings 24/25 honest negatives. NOT
   contradicted by the forecast-probe negative (finding 21) — that probe used naive
   carry-forward forecasters (σ_forecast ≈ 0.23 NDVI, "not small"); the cross-sectional
   GBM reaches σ ≈ 0.048–0.073. This is the stronger, untried variant of the gated
   path-C objective (todo.md §D).
4. MDE arithmetic: at σ = 0.048, Δ = 0.08 (S3) ≈ 1.7σ → S1–S3 likely near/below the
   floor even with B2 (consistent with oracle-floor MDE censoring, Δ* 3.3–5× Δ_req).
   Persistence helps only if residual correlation is measured (untested).

## P2 — Neighbor-context inputs (consultant step 2)

Feed each channel relative to the same-date neighbor median (or append the neighbor
median as extra channels). Rationale: removes the date common mode (~24% of residual
variance measured); the current GRU-AE has no path for it.

Conditions: neighbor graph from `Data/RoI/Farms.geojson` adjacency (open question:
touching polygons vs k-NN by centroid distance); causal rule required if early-warning
claims are made (neighbor median at date t may only use observations ≤ t); needs a
derived artifact — data-pipeline change → openspec.

## P3 — Relative-phase time axis (consultant step 3)

Consultant: relative phase beats absolute day-in-cycle (0.107 vs 0.123). Agent
verification: **the v2 store uses absolute 15-day bins** — `io/patches.py:46-48`
(`N_BINS = 36`, `BIN_DAYS = 15`, `MAX_CYCLE_DAYS = 540`), bins indexed by absolute
day-in-cycle and masked beyond cycle length (`patches.py:567`). Variable-length cycles
(median 366 d) occupy only ~24 of 36 bins and misalign across cycle lengths.
**P3 is a real gap in the current chain, not already satisfied.**
Consequence: changing the axis changes the store build (openspec change; eval-subset
hashes change).

## P4 — Contiguous-block pretext (consultant step 4)

Replace random masking with contiguous 30–60 d block masking; reconstruct the block
from phase + neighbor context + surrounding inputs; masked-block residual = anomaly
score. Targets the measured absorption α ≈ 0.45 (finding 20): random masking lets the
model interpolate through continuous drops.

**GATE (agent-added):** do NOT retrain before the E1 wiring decided in the oracle-floor
audit (2026-11-22, Branch C: the gap is achieved by the statistic swap alone on frozen
residuals — no retrain). P4 is the retrain-shaped follow-up only if Branch C/E1 wins
and the 8-channel audit justifies new inputs.

## P5 — Pre-registered scoring gates (consultant step 5)

Persistence (≥ 2 observations ≥ 5 d apart) + cold-LST gate (reject ΔLST < −2 °C as
cloud/shadow; 21% of episodes rejected). Keep event-level framing (Q8). Gate thresholds
must be frozen before evaluation (§8 threshold rule).

## Consultant's constraints (adopt as-is)

- Do not train on, or calibrate thresholds against, the consultant's episodes — they
  derive from the same NDVI (circularity).
- Pixel-level hypothesis (within-field contrast intrinsically removes the date effect)
  is untested but consistent with the patch program's motivation.
- All claims stay at L2; "stress" wording requires independent agronomic evidence;
  `Farms_subset` (R11, 32 fields) is the L4 path.

## Open questions (resolve before any openspec change)

1. Does B2 replace B1 or join as a second no-NN baseline in the ladder? Owner decision.
2. Neighbor graph definition: touching polygons vs k-NN centroids; farm_code prefix
   only as a documented fallback.
3. Can P2/P3 be evaluated without a new extraction (computed at train time from the v2
   store + Farms polygons), or does the store need a rebuild (hashes change)?
4. Exact MDE protocol for B2 on the locked 170-cycle panel — same severity-sweep
   machinery with the Branch-C statistic swap?

## Verification record

- 2026-11-22: `stress_detector.py` rerun in the `geo` env (PATH includes
  `geo/Library/bin`); outputs numerically identical to the consultant's CSV
  (854 / 523 / 153 / 178; SD 0.0555; sensitivity 1514 / 854 / 487; GRU stats as above).
  Artifacts: `.scratch/claude_stress_detector_2026-11-22/` (originals + `rerun_check/`).
