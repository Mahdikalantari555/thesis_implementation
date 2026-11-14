# Finding 14 — Cycle Audit (D2) — SUPERSEDED DRAFT, PENDING RE-RUN

> **Status (2026-11-14):** the results below are **void**. The previous draft
> reported a 96.7 % `overlap_with_neighbour` rate and applied the D2 decision
> rule to conclude "BLOCKING — do not proceed to model training". That verdict
> was an artifact of an implementation bug, not evidence about segmentation
> quality. The audit artifacts were gitignored as "deprioritized" on the same
> day; on 2026-11-14 the user chose to fix the tooling and finish the audit
> properly instead. The final error rate, Wilson CI and D2 rule application
> will be written here after the `--all` re-run and the user's confirmation of
> 20 sampled cycles on the platform (tasks 2.2/2.3).

## What was wrong (fixed 2026-11-14)

1. **`overlap_with_neighbour` compared the wrong cycles.**
   `scripts/audit_cycles.py` flagged a cycle when it overlapped in calendar
   time with a cycle of a **different field in the same block**
   (`audit_cycles.py` pre-fix lines 153–160). Design D2 defines the flag as
   overlap with the field's own **neighbouring cycle** (the split/merged-cycle
   error class). Sugarcane harvest is staggered across fields, so calendar
   overlap between neighbouring fields is normal agronomy — nearly every cycle
   was guaranteed to "overlap" some other field's cycle. Direct evidence from
   the void run: `audit.csv` rows field 52 (cycle ≈ 2024-02 → 2025-01) and
   field 53 (cycle ≈ 2019-09 → 2020-12), both flagged in block L08, years
   apart from each other. The same-field check was entirely absent (the
   field's own cycles were explicitly skipped). The fixed script checks
   same-field overlap only; since `segment_cycles` tiles cycles between
   consecutive harvests, a non-zero count now means a segmentation regression,
   and 0 is the expected clean result.

2. **The cycle recount silently returned 0** (companion defect in
   `scripts/data_quality_report.py`). The recount filtered `stats_long` on
   `band == "ndvi"`, but `stats_long` carries only
   `blue, green, lst, nir, red, swir1, swir2` — every field was skipped and
   `reports/data_quality.json` recorded `cycle_recount_non_censored: 0` as if
   it were a real count. The R8 recount (1704 vs 799 discrepancy, required
   before test lock) therefore never actually ran. Fixed: NDVI is computed
   from red + nir per field; per-field skips/errors are reported in
   `cycle_recount_details` so a 0 can no longer masquerade as a count.
   Verified on real data (bounded run): 40 fields → 70 non-censored cycles,
   0 errors, 0 skips.

## What replaces the void numbers

- Sampling now covers the **full non-censored population** (no 240-cycle scan
  cap) and stratifies per D2: start year × cycle-length tercile × field block
  (seed 20261111). Every cycle with a sanity flag joins the reviewed set.
- Per-cycle plots are rendered with PIL (matplotlib `savefig` segfaults with
  0xC0000005 in this environment): NDVI + LST(°C) + valid_frac with the
  cycle's harvest dates marked — usable for agent pre-labeling (task 2.2).
- Regression tests: `tests/unit/test_cycle_audit_data_quality.py`
  (8 tests, green 2026-11-14), including the cross-field-must-not-flag
  regression and the recount-from-red+nir test.

## Pending (order matters)

1. User runs `scripts/audit_cycles.py --all` and
   `scripts/data_quality_report.py --all` (commands in the handoff; smoke runs
   green 2026-11-14).
2. Agent pre-labels the 60 stratified + flagged cycles from
   `reports/cycle_audit/plots/`.
3. User confirms 20 sampled cycles on the platform (task 2.2).
4. This document is finalized: per-class and per-year error rate with Wilson
   95 % CI, D2 decision rule applied (<10 % proceed / 10–25 % tune / >25 %
   block), flagged-cycle exclusion list for the locked splits.
5. Recount result recorded → resolves OQ7 (1704 vs 799) before test lock.

## Metadata (superseded run, kept for the record)

- Script: `scripts/audit_cycles.py` (pre-fix version, untracked)
- Generated: 2026-11-11 · Git SHA `41f87f1c6d87a61dddc66de248f2b74f605f0ec2`
- Seed `20261111` · Data source `D:\RS\Thesis\Data\Processed`
- Void result: `overlap_with_neighbour` 58/60 (96.7 %) — cross-field artifact;
  other flags: `length_out_of_range` 2/60, others 0.
- Per-cycle PNG plots were skipped in that run (matplotlib segfault).
