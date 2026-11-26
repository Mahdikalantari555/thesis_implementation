# Cycle-gap diagnosis — 2026-11-26

Owner verdict on `reports/mini_dataset/figures/timeline_seasonal.png`: **R12-08
fully detected; L02-02 two misses; R15-18 one miss.** This note diagnoses those
three misses. No detector, split or stage code was changed.

## 1. What the misses are

All three are intervals between consecutive harvest candidates that were
**silently dropped** by `build_cycle_rows`
(`src/ssl_anomaly/phenology/mini.py:168-172`) because
`length_days > max_cycle_days` (540 d). The gap is visible as a missing
`cycle_id` in `phenology/cycles_mini.parquet`:

| farm | dropped interval | days | missing `cycle_id` |
|---|---|---|---|
| L02-02 | 2018-08-20 → 2020-03-09 | 567 | 6 |
| L02-02 | 2023-06-30 → 2025-02-19 | 600 | 10 |
| R15-18 | 2017-11-12 → 2019-06-27 | 592 | 5 |

R12-08 has no non-censored interval over 540 d (its longest is 536 d), so
nothing was dropped there except the expected censored 2026 tail — consistent
with the owner's "fully detected" verdict.

## 2. Signal inside the three gaps

Method: the same field-median NDVI/SAVI/LST and daily GDD channels, 5-date
centred median smoothing, and the same oriented multi-band harvest score used
by the detector. **All three gaps have zero NaN scores**, so the fail-closed
missing-channel rule is not the cause.

### L02-02, 2018-08-20 → 2020-03-09 (567 d)
NDVI rises monotonically from 0.147 (post-harvest Aug 2018) to 0.712
(Jul 2019), stays high through Nov 2019, then falls to 0.189 by Feb 2020 — that
fall is the already-accepted 2020-03-09 harvest. The strongest local peak
inside the gap is 2019-10-01: prominence **0.128** (threshold 0.3), score
**−0.457**, channel support **0/4**. There is no positive mid-gap harvest
signal.

### L02-02, 2023-06-30 → 2025-02-19 (600 d)
NDVI rises from 0.107 (Jun 2023) to 0.699 (Jul–Sep 2024), then drops to 0.163
in Jan 2025 — the already-accepted 2025-02-19 harvest. Strongest local peak
2024-10-15: prominence **0.103**, score **−0.510**, support **0/4**. Again a
single long growth cycle, not a second harvest event.

### R15-18, 2017-11-12 → 2019-06-27 (592 d)
Here the signal is genuinely ambiguous. NDVI goes 0.169 (Nov 2017–Feb 2018)
→ 0.505 (Aug 2018) → 0.448 (Dec 2018) → **0.160 (Jan 2019)** → 0.219
(Apr–May 2019) → 0.148 (Jun 2019). Two sub-threshold signals exist:

- **2018-03-29**: score 0.618, prominence **0.130** (< 0.3), support **3/4**
  (oriented z: NDVI +0.91, SAVI +0.78, LST +0.13, GDD +0.70).
- **2019-02-12**: score 0.543, support **3/4** (NDVI +1.32, SAVI +1.35,
  LST −1.64, GDD +1.75), but it sits on the rising ramp toward the accepted
  2019-06-27 peak (score 1.292), so `find_peaks` never selects it.

This gap most likely contains a **missed harvest around Jan–Feb 2019** that was
merged into the broad June 2019 low-NDVI state. The 120-day separation rule is
not the blocker; low prominence / no local maximum is.

## 3. Interpretation (honest, not a preset conclusion)

- The two L02-02 gaps are **not missed harvest events under the current
  signal** — each is one ~19–20 month growth cycle. Whether that is
  agronomically real (fallow or a two-season ratoon) or a detector merge
  requires the owner's judgment; the data alone cannot decide.
- The R15-18 gap **does** contain a plausible missed harvest signal, but it is
  weak (score 0.54, 3/4 support, no local peak). Promoting it needs an explicit
  threshold decision, not a silent patch.
- None of this licenses a cycle-correctness claim. The stage-mapping work
  (Joshi stages / DSH bins) stays blocked until the owner resolves these gaps.

## 4. Independent code defect

Dropping an interval without recording it is a provenance defect regardless of
classification: `build_cycle_rows` removes it from `cycles_mini.parquet` with no
flag. **Proposed, not applied:** keep over-long intervals with
`status = "overlong_unresolved"` plus their dates and length, and exclude them
explicitly from splits and stage assignment. This preserves the evidence while
keeping them out of training. No threshold change, rebuild or split change has
been made.

## 5. Open questions for the owner

1. L02-02 `2018-08-20 → 2020-03-09` and `2023-06-30 → 2025-02-19`: real long
   cycles / fallow, or should a second harvest be forced?
2. R15-18: accept a Jan–Feb 2019 harvest despite score 0.54 and 3/4 support?
3. Approve keeping over-long intervals as `overlong_unresolved` (§4) instead of
   deleting them?

---

## 6. Repair addendum — 2026-11-26 (implementation, not yet owner-reviewed)

Owner decisions: L02-02 gaps stay unresolved (no forced harvest);
`overlong_unresolved` flag approved; LST and GDD dropped from the harvest score;
growth states S1–S4 from VI progress; `kind` plant/ratoon removed; short bounded
intervals flagged `short_unresolved`; weak 2016-01-26 peak stays weak; stage
comparison will use trajectory-conditioned matching, not exact cycle dates.

**Root cause of the missed R15-18 Feb-2019 boundary:** the old `find_peaks`
distance was computed in *indices* from the median sampling interval. With
irregular acquisitions it suppressed a peak only 12 indices away despite being
**135 real days** from the stronger June peak. The repaired detector enforces
the 120-day separation in real days; 2019-02-12 now has prominence 0.406 and is
a strong boundary.

**Smoke output** (`/tmp/opencode/phenology_smoke3`, 0.94 s): 39 cycle rows —
33 `normal`, 4 `overlong_unresolved`, 1 `short_unresolved`; no NaN in
length/status; no `kind` or GDD columns. R15-18 retains two unresolved
over-long intervals (2015-03-12→2017-01-12 = 672 d;
2022-04-17→2023-11-21 = 583 d) and the 135 d interval is now visible as
`short_unresolved`. The R15-18 2016-01-26 candidate remains in
`weak_candidates` (prominence 0.174). L02-02 still has its two long gaps
(720 d, 663 d), as the owner chose.

**Evidence for review:** three mini farms via
`/tmp/opencode/vi_review/mini/timeline_seasonal.png`; six stratified archive
fields (12, 87, 164, 241, 318, 455) via
`/tmp/opencode/vi_review/fields/field_*.png` and `review_summary.json`.
These are local artifacts, not committed. **No cycle-correctness, stage or
stress claim is made until the owner reviews them.**

### 6.1 Greenup start — corrected after owner review (2026-11-26)

The owner pointed out that a field may rest before the next cycle's greenup,
but **not after every harvest**. The first version (20% of full canopy
amplitude) claimed rest too broadly and started field_318 in 2019-01 instead of
the observed rise in 2018-09. Corrected rule: rest is claimed only when both
smoothed indices stay within 0.02 of their post-harvest floor for ≥3
acquisitions (bare soil); greenup is the first sustained rise ≥0.02 above that
floor (≥2 acquisitions); otherwise the cycle starts at the harvest.

Rebuilt mini smoke (`/tmp/opencode/phenology_smoke5`): 36 rows = 32 normal,
3 `overlong_unresolved`, 1 `short_unresolved`; 14 cycles have `rest_days`
(median 80.5 d). Field_318 now starts 2018-09-05 as the owner identified.
Regenerated plots with distinct S1–S4 colours, labels, legend and gray rest
spans are at `/tmp/opencode/vi_review4/{mini,fields}/`. This is a screening
heuristic, not phenology extraction; no cycle-correctness or stage claim is
made before the owner's review.
