# Change: cycle-detection-vi-repair

**Change ID:** `cycle-detection-vi-repair` · **Created:** 2026-11-26 ·
**Status:** PROPOSED — implementation in progress · **Owner:** Mahdi ·
**Source:** owner verdict on `reports/mini_dataset/figures/timeline_seasonal.png`,
`docs/audits/cycle_gap_diagnosis_2026-11-26.md`, and owner UI answers 2026-11-26.

## Why

The owner's visual check found **not all cycles were detected**: L02-02 has two
unresolved long gaps and R15-18 has multiple missed cycles (two NDVI peaks in
2019–2020 with one missed, plus the 2018→2019 cycle missed, apparently from a
low NDVI peak). `build_cycle_rows` silently deletes any non-censored interval
longer than 540 d, so the evidence disappears from `cycles_mini.parquet`.
Splits, stage mapping and the cycle-correctness claim stay blocked until this
is repaired and re-reviewed (`todo.md` cycle-gap item).

## What Changes

1. **Harvest score uses NDVI + SAVI only.** LST is dropped from the score; GDD is
   dropped from the score (owner decision 2026-11-26 after explanation: GDD is a
   shared single-station thermal series and belongs to a separate thermal-time
   axis ablation, findings 13/15). No channel is imputed: a date missing either
   index stays unscored.
2. **Growth states from the VI trajectory.** Four normalized within-cycle
   progress states replace the current calendar fractions and the plant/ratoon
   split: S1 establishment/growth (0–0.25), S2 peak growth (0.25–0.60), S3
   maturity/senescence (0.60–0.85), S4 harvest/dormancy (0.85–1.00). These are
   the states documented in finding 17; the owner asked for "main phases only"
   and supplied this VI-only framework as the chosen interpretation.
3. **Remove `kind = plant|ratoon`** from cycle rows (owner: "looks wrong, no need
   to that now"). No plant/ratoon inference is performed.
4. **Retain over-long intervals** with `status = "overlong_unresolved"`, dates,
   length and evidence; exclude them explicitly from splits and stage mapping.
   No threshold is silently lowered and no harvest is forced inside those gaps
   (owner decision: leave L02-02 gaps unresolved).
5. **Stratified review sample:** six additional fields selected across blocks
   and years, with visual review artifacts generated for the owner.

No split, holdout, finding number or canonical artifact is rewritten by this
change; the rebuilt mini phenology output is additive and versioned.

## Capabilities

- New: `cycle-detection-vi-repair` — NDVI+SAVI harvest scoring, VI-progress
  growth states, provenance-preserving over-long intervals, reviewable candidates.

## Relation to other changes

- `continual-hybrid-representation-design` §5/§6.4: supplies the repaired cycle
  and phase basis; its Path B stays blocked until the owner re-reviews.
- `cycle-detection-v3-multindex`: separate 490-field producer, untouched here.
- `field-id-provenance-fix`: prerequisite for any archive-wide extension.

## Impact

- **Affected:** `src/ssl_anomaly/phenology/mini.py`,
  `scripts/build_mini_phenology.py`, `tests/unit/test_mini_phenology.py`,
  `todo.md`, `docs/audits/cycle_gap_diagnosis_2026-11-26.md` (addendum).
- **Frozen:** findings 00–28, splits/holdout manifests, canonical rasters,
  capstone gates.
- **Risk:** NDVI+SAVI-only scoring changes candidate ranks; the weak R15-18
  candidates must be reviewed, never silently promoted.
