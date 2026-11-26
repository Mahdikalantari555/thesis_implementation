# Design — cycle-detection-vi-repair

**Date:** 2026-11-26 · **Status:** PROPOSED (implementation in progress) ·
**Claim ceiling:** L1 screening until owner re-review.

Evidence tags: **[R]** research-backed · **[C]** exists in code · **[D]** new
development · **[DEC]** owner decision · **[HYP]** hypothesis.

## 1. Decisions

| # | Decision | Basis |
|---|---|---|
| 1 | Score = mean(−z NDVI, −z SAVI); drop LST and GDD from the score | [DEC] owner; one-station thermal series is not field-specific; GDD is deferred to a thermal-time-axis ablation [R findings 13/15, Nyborg TPE, T³S] |
| 2 | Growth states S1–S4 from normalized within-cycle progress | [DEC] owner supplied the VI-only framework; finding 17 states; main phases only |
| 3 | No `kind` plant/ratoon field | [DEC] owner: "looks wrong, no need to that now" |
| 4 | Over-long non-censored intervals kept as `overlong_unresolved`, excluded from splits | [DEC] owner approved the flag |
| 5 | Weak candidates stored separately, never silently promoted | [DEC] no confirmation loop; audit requirement |
| 6 | Six-field stratified review sample selected by agent, reviewed by owner | [DEC] owner |
| 7 | Short bounded intervals (< 180 d) kept as `short_unresolved`; minimum NOT lowered | [DEC] owner; 135 d is shorter than any supported sugarcane cycle |
| 8 | Weak 2016-01-26 peak stays in `weak_candidates`; no precise-cycle machinery. Stage comparison uses trajectory-conditioned matching (design §5 Path A), not exact harvest dates | [DEC] owner |
| 9 | Rest is claimed **only** for a genuine bare-soil plateau: both smoothed indices within 0.02 of their post-harvest floor for ≥3 acquisitions. Greenup is the first sustained rise ≥0.02 above that floor (≥2 acquisitions); otherwise the cycle starts at the harvest. No fraction of full canopy amplitude is used | [DEC] owner, correcting the over-broad first rule |

With this corrected rule, 14 of 36 mini cycles have a bare-soil rest (median
80.5 d); the others start at harvest. L02-02's former 720-day gap becomes a
96-day bare period followed by a 624-day cycle; field_318 starts 2018-09-05 as
the owner identified. Three over-long and one short interval remain, all
visible as unresolved. This is a screening heuristic, not phenology extraction;
owner review remains mandatory and unresolved intervals are excluded from splits.

**Design caveat:** `growth_state_at_midpoint` is necessarily S2 because a cycle
midpoint has progress 0.5. It is a legacy field, not a stage-conditioned
scoring basis. Per-date states need a separate table; do not read this field as
evidence of stage assignment.

## 2. Algorithm

1. Compute field-median NDVI and SAVI series, 5-date centred median smoothing
   (unchanged from `scripts/build_mini_phenology.py`) [C].
2. `vi_harvest_score(ndvi, savi)`: oriented robust z (−NDVI, −SAVI), fail closed
   on either missing channel, mean of the two [D].
3. `detect_harvests`: `find_peaks` on the score with pre-registered `prominence`
   and real-day `distance`; candidates above threshold are harvest boundaries;
   candidates below threshold go to the `weak_candidates` table [D].
4. `build_cycle_rows`: consecutive boundaries form cycles; growth state from
   progress fraction; intervals `> max_cycle_days` (non-censored) get
   `status="overlong_unresolved"`; `status="normal"` otherwise; censored tail
   keeps its existing flag [D].
5. Splits and stage assignment filter on `status == "normal"` [D].

## 3. Expected effect on the known gaps

- L02-02 (567 d, 600 d): no positive NDVI/SAVI mid-gap signal in the audit, so
  no boundary is forced; both intervals become visible as `overlong_unresolved`.
- R15-18: the 2019-02 signal had NDVI z +1.32 / SAVI z +1.35; without LST/GDD
  its score rises from 0.543 to ~1.33, so it is expected to become a strong
  candidate. The 2018-03 signal (z +0.91/+0.78, prominence 0.13) is expected to
  stay in `weak_candidates` pending owner review. These are predictions, not
  results; the rebuilt output must be read before claiming them.

## 4. Validation and risks

- Unit tests: orientations, NaN fail-closed, two-boundary detection, weak
  candidate separation, over-long retention, no `kind`, S1–S4 boundaries,
  explicit split exclusion [D].
- Smoke: rebuild mini phenology for the three farms, read the output, compare
  candidate count with the audit. No long run is launched by the agent.
- Visual review artifacts: three mini farms + six stratified fields from the
  490 archive; owner verdict recorded before any stage-conditioned claim.
- Risk: a two-index score is more sensitive to soil background in sparse
  canopy; SAVI mitigates but does not eliminate it. Risk: 6 fields are weak
  evidence for cross-field generalization; no L3 claim follows from them.

## 5. Frozen

Findings 00–28, `splits_locked_v2.json`, holdout manifests, canonical rasters,
capstone gates G-A…G-E. The rebuilt `cycles_mini.parquet` is a new versioned
artifact; nothing is overwritten silently.
