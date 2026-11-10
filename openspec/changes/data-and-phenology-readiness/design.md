# Design: Data and Phenology Readiness

## D1. Literature review scope and output

The output is one document, `docs/findings/13_literature_phenology_and_perturbations.md`, with four sections. Each cited paper gets: citation, data (sensor, crop, region), method, what it implies for this thesis, and whether it generalises across crops. Papers already in `thesis_research/SSL_Proposal` (84-paper matrix) are reused first; new searches fill the gaps.

1. **Automatic phenology / season extraction**
   - Methods: TIMESAT (Savitzky-Golay, asymmetric Gaussian, double logistic), phenofit, BFAST / BFAST-monitor, CCDC, shape-model fitting, DTW-based alignment, learned season detection.
   - Criteria: works with irregular Landsat sampling and cloud gaps; can run causally (in-season) or only retrospectively; handles multiple seasons per year and perennial or ratoon crops; reported accuracy against ground truth.
2. **Time-axis representation for SSL**
   - Options: fixed DSH bins (current), normalised cycle progress (0–1), GDD axis, landmark registration, and irregular tokens with timestamp/position encoding (e.g. SITS-BERT, Presto, SITS-Former, TimeMatch-style).
   - For each: whether it supports causal monitoring, missing-data handling, and crop independence.
3. **Realistic sugarcane stress signatures in optical + thermal time series**
   - Stresses: water deficit, salinity, waterlogging, pests and disease, frost, lodging, early ripening or flowering.
   - For each: bands affected, phase, onset speed, duration, typical magnitude, and recovery. This produces the perturbation catalogue with sources.
4. **SSL for irregular, gappy satellite series (short)**
   - Purpose: confirm or challenge the Masked Transformer + GRU-AE choice at this data scale (about 800 cycles).

**Stopping rule**: about 25–40 relevant papers in total, enough for a decision. This is not an exhaustive review.

## D2. Cycle audit

- **Sample**: 60 cycles, stratified by year (2013–2026), cycle length tercile, and field block, plus every cycle with a sanity flag. The seed is recorded.
- **Sanity flags** (automatic, all cycles):
  - length below 240 or above 480 days (proposed)
  - no clear NDVI drop at the start
  - fewer than 6 valid observations
  - overlap with the neighbouring cycle
  - peak NDVI below 0.4 (proposed)
- **Review**: one plot per cycle (NDVI + LST + valid_frac, detected harvest dates marked). The agent pre-labels each plot. The user confirms a subset of 20 on the platform. Error classes: correct, harvest shifted by more than 30 days, split cycle, merged cycles, not sugarcane / fallow, too sparse.
- **Output**: estimated error rate with a Wilson 95 % CI, per class and per year.
- **Decision rule (proposed)**:
  - Error rate below 10 %: proceed, flagged cycles excluded from dev/test metrics.
  - 10–25 %: tune segmentation parameters, as a separate fix proposal.
  - Above 25 %: segmentation becomes a blocking item before training.
- **Ground truth**: if planting or harvest records arrive, segmentation error in days is computed against them. This is optional and never blocking.

## D3. Data quality report

- Per year and per field block:
  - observations per cycle
  - fraction of valid observations
  - LST values outside 15–55 °C
  - bands with NaN fraction
  - acquisitions with `valid_frac` below 0.5
- **Count check**: confirms the 538-vs-539 date discrepancy (todo OQ5).

## D4. Decision record

After D1–D3, a short section added to `train-and-finalize-main-architecture/design.md` records:
- the input time axis (keep DSH 36×15 or change)
- the segmentation parameters
- the final perturbation set and magnitudes, with sources
- the cycle exclusion rule

Each decision cites the review or audit section it rests on. The user approves this record before any training code is written.
