# Design — cycle-detection-v3-multindex

## D1: Hybrid union detector (v1 ∪ candidates), not a rewrite

v3 harvest boundaries = v1's detected harvests (verbatim pipeline) ∪ fusion
candidates from the measured rule. Rationale: the phase-0 numbers
(3,416 candidates, threshold-grid stability) were measured with exactly this
union semantics; v1 behavior remains a strict subset (v2 boundaries ⊇ v1);
the v1↔v2 comparison stays meaningful because only ADDITIONS change. A
from-scratch rewrite would re-derive everything and detach the result from
the measured diagnosis.

## D2: Frozen thresholds (the measured primary point)

`veg_drop 0.20, lst_rise 3.0, swir_rise 0.05, ctx_days 120, min_gap 5,
vf_min 0.5, edge_guard 30, dedup_days 180, season check none` — frozen in
`FusionParams` with a recorded hash in the v2 table provenance. The
threshold grid lives in `reports/cycle_detection_diagnosis/report.json`;
changing any threshold re-opens the diagnosis, not this change.

## D3: Candidate evidence on the raw observation series (deliberate)

Candidates use median context windows over the OBSERVATION series
(`[t−120, t−5)`), not the Whittaker daily grid. This matches the diagnosis
exactly, avoids re-smoothing choices, and is robust to single outliers by
construction (medians). v1 detection itself keeps its exact internal
pipeline (Hampel → Whittaker CV → daily grid → `detect_harvests`).

## D4: Refactor, not fork

`segment_cycles`' harvest pipeline is extracted verbatim into
`detect_harvest_days(...)`; `segment_cycles` calls it (behavior unchanged —
existing phenology tests must pass byte-identically). The fused path reuses
the same helper plus `_build_cycles_from_boundaries` shared window logic so
the two detectors cannot drift.

## D5: Preservation and naming

Every new artifact gets an explicit v2 name; nothing is written to a v1
path: `cycles_490_v2.parquet`, `growth_cycles_490_v2.npz`,
`splits_locked_v2.json`, `holdout_manifest_v2.json`,
`patches_v2.zarr` (under `data/patch_runs/v3/`), `norm_stats_full_v2.json`.
The v2 splits manifest records `derived_from` (v1 manifest sha) and the v2
cycle-file hashes; the holdout v2 freeze records the same policy
(test-block farms, post-2024 cutoff, `--archive-end 2026-09-10`).

## D6: Fail-closed pre-extraction gates

Before the 3–4 h store build, the v2 table must pass:
1. cycle-length distribution consistent with annual sugarcane
   (median 300–420 d; share of 200–500 d cycles ≥ 90 %);
2. harvest starts clustered in the Khuzestan season (≥ 70 % of NEW
   (candidate-only) boundaries in Nov–Mar);
3. per-field cycle counts sane (max ≤ 13, i.e., one per archive year);
4. v1↔v2 counts table emitted (v1 ⊆ v2 by construction).
A gate failure blocks the handover and is reported verbatim.

## D7: Candidate limits recorded in provenance

`ndvi_at_harvest` for candidate boundaries uses the firing channel's value
at the candidate observation; replant re-detection is NOT run for candidates
(ratoon numbering continues v1's chain) — recorded as a known limit; the
audit-exclusion list (D2 finding-14) applies to v2 cycles by key as in v1.

## D8: Training protocol unchanged

The v2 store trains with the SAME `train_patch.py` protocol (6 trainings,
seeds 13/42/2026, 10 epochs, `--patch-size 48` store geometry) — only
`--store/--norm/--splits/--holdout-manifest` point at v2 paths. Colab
bundle v2 = `patches_v2.zarr` + v2 JSONs; notebook v2 = the clone flow with
v2 paths and gates.
