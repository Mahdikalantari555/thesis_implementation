# Representation-first pilot — results and honest reading

**Date:** 2026-11-24 · **Rung:** L1/L2 screening only · **Scripts:**
`scripts/label_mini_suspected_stress.py`, `scripts/probe_mini_suspected_stress.py`.
Machine-readable result: `representation_first_pilot.json`.

## What was actually run

- **Suspected labels from real products:** METRIC `CWSI_ET`, `ETaDaily`, `ETrF`,
  `LAI`, `TVDI` sampled on the exact mini pixel grid (all 538 dates × 3 farms).
  Stage-conditioned (S1–S4), train-years-only ETa baseline (median/MAD), rule
  `CWSI ≥ 0.40 (plant) / 0.35 (ratoon) AND ETa deficit ≥ 2 MAD AND ≥2
  consecutive acquisitions`. Gated by `pixel_valid AND in_field`.
  Result: **3,940 suspected pixels / 419 suspected patch-dates** out of
  455,796 pixel rows (0.86% suspected).
- **Well-trained candidate:** patch-level (5×5) windows of length 30 dates with
  7 bands + NDVI + 12 weather features, normalized on train years only;
  small GRU encoder, 8 epochs, seed 13, CPU. Train = years < 2023,
  test = years ≥ 2023 (2,520 test windows, 370 suspected, prevalence 14.7%).
- **Metrics:** CYSP-RG (same-field/same-stage cross-year R@1 vs raw),
  simple logistic probe on suspected labels (SSL vs raw vs random-init),
  collapse diagnostics.

## Headline numbers

| Quantity | SSL | Raw features | Random init |
|---|---|---|---|
| CYSP-R@1 | **0.0869** | 0.0492 | 0.0476 |
| CYSP-RG (Δ vs raw) | **+0.0377** | — | −0.0016 |
| Suspected-stress AUPRC | 0.3266 | **0.4981** | 0.3014 |
| Effective rank (16-d) | 1.888 | — | — |
| Mean pairwise cosine | 0.0225 | — | — |

## Honest reading — what this does and does not show

1. **A small structural signal exists.** Same-field/same-stage cross-year R@1
   rises from 0.049 to 0.087 after training; random init stays at 0.048. On
   this narrow metric the encoder learned something about field/phase identity
   that the raw vector does not carry. This is an L1 observation, not a benefit.
2. **No benefit on suspected stress.** Raw features beat the frozen SSL
   embedding by a wide margin (0.498 vs 0.327). The representation that helps
   retrieve "same field, same phase" does **not** help discriminate physically
   suspected pixels. This is the same pattern as the earlier detection result.
   Do not claim SSL is useful for stress from this run.
3. **The negative control failed — metric is not yet publishable.** The
   shuffled-year control gave R@1 ≈ 0.999 for both SSL and random embeddings,
   i.e. it did *not* collapse as required. Cause: shuffling which years are
   candidates leaves near-duplicate windows of the same farm/stage available,
   so retrieval is won by identity similarity. The +0.0377 gain is therefore
   **preliminary**: the task must be rebuilt with near-duplicate exclusion
   (same farm/stage/date ± window across years) and a real label-shuffle null.
4. **Collapse risk is present.** Effective rank 1.888 out of 16 dimensions and
   per-dimension std 0.18–0.51 indicate a strongly anisotropic embedding. Under
   a strict pre-registered rank gate this candidate would be disqualified; it
   must not be promoted. Add variance/covariance regularization (VICReg or
   Barlow redundancy term) and re-check before any claim.
5. **Label circularity is disclosed, not solved.** CWSI_ET shares LST and NIR
   inputs with the model; ETaDaily is a METRIC model product, not a measurement.
   The strict arm (model inputs excluding LST/CWSI-derived channels) has not
   been run. "Suspected" is not stress; no water/salinity typing.

## Consequence for the research plan

- Keep the representation-first frame, but **fix the metric before believing
  the gain**: deduplicate cross-year near-duplicates, add a proper null, bootstrap
  CIs by farm/year, and pre-register the collapse threshold.
- Improve the encoder *before* re-testing stress: stage/cGDD conditioning,
  hybrid reconstruction + redundancy reduction, ≥30 epochs with learning-curve
  gate, 3 seeds, matched random-init and harmonic-feature controls.
- Treat suspected-stress discrimination as a **downstream probe**, not the
  objective; if raw keeps winning, the correct thesis statement is that SSL's
  role is embedding/retrieval and physical rules win on stress suspicion.
