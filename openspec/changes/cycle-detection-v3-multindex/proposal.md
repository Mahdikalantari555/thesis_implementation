# Change: cycle-detection-v3-multindex

## Why

Phase-0 diagnosis (`scripts/diagnose_cycle_recovery.py`, 2026-11-19, commit
36b6161) measured the cost of the frozen NDVI-only detector: v1 yields 799
clean cycles over 478 cycle-bearing fields / 13 years (~2.2 per field) while
1,819 lost spans hold **5,625.7 field-years**. Classifying every span shows
the bottleneck is NOT missing imagery — only 2 spans / 1.1 field-years are
observation holes; winter observations exist almost everywhere
(39–55 % of 15-day bins carry ≥1 valid observation vs 87–94 % in summer).
The binding constraints are v1's NDVI-only criteria (field-GLOBAL
`plateau_frac` 0.7 × ymax, rise-within-180 d on a sparse winter series).

A transparent fusion rule (NDVI-or-SAVI drop ≥ 0.20 corroborated by an
LST rise ≥ 3 K or a SWIR1 rise ≥ 0.05, ≥180 d dedup, ≥30 d from known
boundaries) recovers **3,416 candidate clean cycles (+2,617 ≈ 4.3× v1)**,
stable across the full 18-combo threshold grid (v2 ∈ [3,059, 3,927]).
The owner approved going to production v3 directly (2026-11-19): manual
candidate labeling is not feasible for the owner, the numbers are stable,
and the v1↔v2 comparison is itself a keepable scientific deliverable
(advisor's SAVI hypothesis, quantified).

## What Changes

- **`ssl_anomaly.phenology.cycles`**: refactor the v1 harvest pipeline into
  a reusable `detect_harvest_days(...)` (verbatim behavior), add
  `FusionParams` and `detect_harvest_candidates(...)` (the measured fusion
  rule on the raw observation series), and `segment_cycles_fused(...)`
  building cycles from the union boundary set with the SAME 180–540 d
  validity rule, leading/trailing censoring, and v1 replant handling for
  v1 harvests. v1 behavior is a strict subset (no candidate → identical
  output).
- **`scripts/segment_and_export_cycles_490.py`**: `--detector {v1,fused}`
  and `--cycles-out/--npz-out` flags; `fused` writes
  `data/processed/cycles_490_v2.parquet` +
  `data/interim/growth_cycles_490_v2.npz` with a v1↔v2 comparison block in
  the log/report. Defaults unchanged.
- **Split/freeze chain v2, all at NEW paths** (nothing overwritten):
  `build_locked_splits.py --cycles --out` → `splits_locked_v2.json` (same
  k-means seed/blocks, same D2 exclusion list, temporal cutoffs);
  `build_holdout_manifest.py --splits-path --out --archive-end 2026-09-10`
  → `holdout_manifest_v2.json`; `extract_patches.py --patch-size 48
  --splits splits_locked_v2.json … --out data/patch_runs/v3/patches.zarr`;
  `build_patch_norm.py` → `norm_stats_full_v2.json`.
- **Pre-extraction gates on the v2 table** (fail-closed before the 3–4 h
  store build): cycle-length distribution vs v1 (365 ± 49 d reference),
  harvest-month clustering (Nov–Mar dominance), per-field cycle counts,
  v1↔v2 counts table.
- **Owner handover**: PowerShell for the local chain (export → splits v2 →
  holdout v2 → extraction → norm v2); agent then verifies, builds the v2
  data zip + Colab notebook (same 6-trainings protocol on the v2 store).

## Impact

- **Specs**: new capability `cycle-detection` (fused detection, artifact
  preservation, provenance, gates).
- **Preserved**: `cycles_490.parquet` (930a6cde…), `splits_locked.json`
  (9aa5eb11…), `holdout_manifest.json`, `patches.zarr`, `norm_stats_full.json`,
  and every checkpoint/report trained on them stay frozen and are the v1
  baseline of the comparison. No in-place migration anywhere.
- **Known limits (recorded, not hidden)**: candidates are an upper bound —
  no manual labels were applied; the unresolved class (374 spans / 539 fy)
  mixes "no harvest truth" with "criteria still miss it"; candidate ratoon
  numbering does not re-run replant detection (v1 harvests keep v1 behavior).
  External harvest records remain the strongest future validator (L4).
