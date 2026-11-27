# E0 cycle-completeness audit — 2026-11-27 (WP-1, workflow_2026-11-27)

**Gate:** E0/G0 completeness arm — "cycle-completeness audit BEFORE any
training … No modeling before G0" (roadmap
`thesis_research/SSL_Proposal/ssl-hierarchical-agri-model/REPORT.md:185,249`).
**Scope of this audit:** (a) completeness accounting over the training cycle
tables v1/v2, the ~3.6 cycles/field expectation, unresolved-flag accounting;
(b) the `archive_fid = reference_fid + 1` alias verification on ALL 490 fields
(IoU per field, folded in from `continual-hybrid-representation-design`
task 3.1). The NaN/cloud-policy and L8/L9-harmonization arms named in the same
roadmap row are **NOT part of this audit and were not run here** — G0 is not
fully closed by this document alone.

**Method / evidence:** read-only audit script
`.tmp/wp1_e0/e0_cycle_completeness.py` (run 2026-11-27, 4.6 s wall,
`OMP_NUM_THREADS=4`, no writes outside `reports/e0_cycle_completeness/`);
full machine-readable results
`reports/e0_cycle_completeness/e0_results_2026-11-27.json` (committed).
Every number below is copied from that JSON or from the cited artifact.

## 1. Data facts

| artifact | location | verified |
|---|---|---|
| v1 cycles (frozen 2026-11-06) | `thesis_implementation/data/processed/cycles_490.parquet` | 1,755 rows / 478 fields |
| v2 cycles (fused, created 2026-11-20, git cfa57d86) | `thesis_implementation/data/processed/cycles_490_v2.parquet` | 4,458 rows / 490 fields |
| v2 provenance | `data/processed/cycles_490_v2.provenance.json` | detector `fused`, fusion hash b73c5517a2fc5594, supersedes v1 |
| splits | `data/processed/splits_490.json` | 370 fields / 799 cycles / 20 spatial blocks / 13 LOYO years, sha256 0315a240… |
| reference polygons | `Data/RoI/Farms.geojson` | 490 unique `OGC_FID` (0–489), reprojected EPSG:32639 |
| archive pixel membership | `Data/Processed/satellite/pixels/*.parquet` (538 dates, 2013-03-30 → 2026-09-10, sensors L8/L9, path/row 165039+166038) | per-field pixel rows, `field_id` 1–490 |

Location discrepancy recorded: the package ask said "locate the cycle tables
under `/home/mahdi/RS/Thesis/Data/Processed` (v1 and v2 parquet)" — they are
not there; the canonical training cycle tables live in the **impl repo**
`data/processed/` (reconciliation ledger fact #4; confirmed by `ls` this
session; `Data/Processed/` holds the pixel/stat artifacts instead).

Field-id space: both cycle tables use **archive ids 1–490** (v1 min 1 / max
490; v2 min 1 / max 490; no ids outside 1–490 in either table — verified in
the committed JSON `v1_extra_field_ids_outside_1_490` / `v2_…` = empty).

## 2. Completeness accounting (Part A)

| metric | v1 | v2 |
|---|---|---|
| rows | 1,755 | 4,458 |
| fields with ≥1 cycle | 478 / 490 (12 zero-cycle) | 490 / 490 (0 zero-cycle) |
| zero-cycle archive field ids | 97, 127, 145, 171, 193, 198, 215, 217, 231, 260, 387, 402 | — |
| non-censored | 799 | 3,478 |
| censored | 956 | 980 |
| cycles/field mean (median, min–max) | 3.672 (3, 2–10) | 9.098 (9, 3–14) |
| non-censored per field | 1.672 | 7.098 |
| date range (`t_start`→`t_end`, epoch 2013-03-30) | 2013-03-30 → 2026-09-11 | 2013-03-30 → 2026-09-11 |
| `status` retention-flag column | **absent** | **absent** |

Consistency: v1 non-censored (799) == `splits_490.json` `n_cycles` (799) — the
frozen 370-field split is a subset of v1's clean cycles.

### The ~3.6 cycles/field expectation

Roadmap `REPORT.md:122,185`: "1,755/490 ≈ 3.6 cycles/field vs expected ~10–12
for an annual crop … pass = cycles ≈ 10–12/field **or a documented reason**".

- **v1 measured 3.672/field.** Documented reason for the shortfall: the v1
  detector keeps only intervals with 180 ≤ length ≤ 540 days and **silently
  skips the rest** (`src/ssl_anomaly/phenology/cycles.py:298-301`), and fields
  with <30 usable observations return no cycles at all
  (`src/ssl_anomaly/phenology/cycles.py:165-167`). 12 fields have zero cycles.
  This is a provenance defect (silent deletion), not a data absence.
- **v2 measured 9.098/field (7.098 non-censored).** v2 = v1 harvests + the
  multi-index fusion detector (provenance JSON, git cfa57d86, created
  2026-11-20) recovering censored/skipped windows: 3,478 non-censored cycles
  over all 490 fields. The same 180–540 d window rule still applies, so the
  count stays below the naive 10–12 expectation; the measured median clean
  length (366 d, provenance JSON `v2_clean_len_median`) is consistent with
  annual sugarcane, so the remaining shortfall is attributable to censoring at
  series edges and detector merges — **documented reason satisfied for v2**.

### Unresolved-flag accounting

- v1 and v2 have **no `status` column**: intervals outside 180–540 d were
  dropped at production time and are **not recorded** in either table. Absent
  is reported as absent, not zero. Retrofitting retention flags into the 490
  producer belongs to `cycle-detection-v3-multindex` (separate change; its
  production path is OFF) and was **not** done by this audit.
- The retention flags (`overlong_unresolved`, `short_unresolved`) exist only
  in the repaired **mini** chain: the canonical `cycles_mini.parquet` is still
  the pre-repair 34-row schema (owner-gated rebuild, see
  `docs/audits/cycle_gap_diagnosis_2026-11-26.md` §7); the regenerated review
  run (2026-11-27) shows 32 normal / 3 `overlong_unresolved` /
  1 `short_unresolved` over 36 rows
  (`/tmp/opencode/vi_review4/mini/review_summary.json`).
- As of commit b9d5583 every mini cycle-table consumer fails closed on the
  pre-flag schema and surfaces unresolved-row counts (never silently deletes).

## 3. Alias verification `archive_fid = reference_fid + 1` (Part B)

**Verdict: PASS — 490/490 fields verified.**

- Method: per-field archive pixel membership from `pixels/*.parquet`
  (best-coverage date per field over all 538 dates; in practice one
  cloud-free date, 2013-03-30, covers all 490 fields), rasterized against the
  reference polygons of `reference_fid = archive_fid − 1` and its ±1
  neighbours on the field's own 30 m window; IoU per field.
- Result: `iou_own_polygon = 1.0` for **every** field (min = p05 = median =
  mean = 1.0; 490/490 ≥ 0.99); `best_reference_fid == reference_fid` for all
  490; neighbour IoU 0.0 (no overlap with adjacent polygons). The alias
  measured on the 3 mini farms (`scripts/build_mini_dataset.py:73-78`) holds
  for the whole archive.
- Fail-closed guards, all active: 30 m affine-fit check per date (slope
  tolerance 1e-3), affine residual < 1 m, unknown `field_id` in any parquet,
  zero-coverage fields, coverage < 490 — any hit aborts with the exception
  verbatim and writes partial results.
- Convention note (provenance): the pixels parquet stores `x`/`y` as pixel
  **corner** coordinates. The first script run used a half-pixel (center)
  shift and produced a systematic IoU ≈ 0.86 with a symmetric ±1-px
  boundary ring; the corner convention reproduces the recorded mini evidence
  (IoU exactly 1.0) and was adopted. Both runs are documented here; the
  committed JSON is the corrected run.

## 4. Go / no-go for training

- **v1 chain (`cycles_490.parquet`): NO-GO as a completeness basis for new
  modeling.** 3.672 cycles/field vs the ~10–12 expectation with 12 zero-cycle
  fields and silently dropped out-of-window intervals (unrecorded). Existing
  frozen artifacts hashed to v1 (e.g. `splits_490.json`, 799 cycles) keep
  their historical provenance; nothing is rewritten here.
- **v2 chain (`cycles_490_v2.parquet`): GO for the completeness arm.** All 490
  fields covered, 3,478 non-censored cycles, shortfall vs the naive
  expectation documented above, and the field-id alias verified 490/490 with
  IoU 1.0. Training planned on v2 bindings (as in
  `benchmark-latent-reconstruction-prediction-architectures` 0.2) may proceed
  **with respect to cycle completeness**.
- **G0 overall is NOT closed by this document**: the roadmap E0 row also
  requires the quality.parquet NaN/cloud-policy check and the L8/L9
  harmonization-shift check (REPORT.md:185), which are outside this package
  and were not run here.
- No model claim is made anywhere in this audit (pre-training data
  accounting; no validation-ladder rung applies).

## 5. Reproduction

```
cd /home/mahdi/RS/Thesis/thesis_implementation
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
/home/mahdi/micromamba/envs/geo/bin/python -X utf8 .tmp/wp1_e0/e0_cycle_completeness.py
```
Expected: "Part A done: v1 1755 rows / 478 fields; v2 4458 rows / 490 fields",
"Part B done: PASS | IoU own min 1.0 …", ~5 s; rewrites only
`reports/e0_cycle_completeness/e0_results_2026-11-27.json`.
