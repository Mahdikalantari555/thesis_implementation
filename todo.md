# todo.md — thesis platform & research backlog

Owner-driven requirement list. **Rule:** each numbered item becomes its own
openspec change when we start it (per user, 2026-11-07). Items ✅ are done;
▶ is in progress; everything else is queued by priority.

## Done (2026-11-07, change `wire-platform-to-real-data`, commits b667e5a / 20a28f5)

- ✅ Tile server survives concurrent browser load (per-handle locks; burst + headless-pan gates)
- ✅ Real weather wired (ERA5 join, 100% coverage on the then-monthly grid); null ⇒ `absent`
- ✅ Real farm registry metadata; zero synthesized strings
- ✅ Real model generations exported **with** checkpoints (9,639 model-provenance cells)
- ✅ Honest copy (PipelineDataBadge, banner, footers); stressEvent machinery removed; `mock.ts` → `data.ts`
- ✅ Map Explorer UX: Hide = polygons only; opacity floor 0; **single date slider** drives fields + raster; raster opacity persists
- ✅ No fabricated numbers (score 0.5 fallback removed; complete-case Moran's I; CSV empty cells)
- ✅ Presentation audit (`docs/platform_presentation_audit.md`) + precise methods doc (`docs/platform_score_methods.md`)
- ✅ MSc wording (was PhD) in en/fa
- ✅ RGB tile **seams fixed** — per-scene contrast window (`SceneStretchCache` in `serve_tiles.py`) instead of per-tile stretch; verified on stitched adjacent tiles

## Done (second wave, 2026-11-07)

- ✅ **1. Full-archive observation axis** — every acquisition date (538; 263,620 obs; multi-acquisition months kept). Contract v4 compact payload (292 MB → 29.8 MB, RLE provenance, derived formula ids). Charts: adaptive ticks; scatters stride-sampled for display (stats on full data). Registry envelopes widened for real winter medians (ndvi −0.399 observed). Gates green (up + down).
- ✅ RGB tile seams (per-scene contrast window), compact provenance chips, tick fonts, CDP send-timeout + watchdog in the gate harness.

## In progress

- ▶ **`rerun-rq-suite-on-real-490`** (proposed 2026-11-10) — RQ1–RQ8 + data-scarcity results in findings 11–12 came from `simulate_490_field_archive`, not the real 490-field archive; re-run on real data, fail closed, provenance in every report.
  - ✅ Task 1: real_archive.py loader + hash check + provenance + --source arg (tests/unit/test_real_archive.py 4/6 pass)
  - ✅ Task 2: all 8 run_rq*.py + run_data_scarcity.py refactored with --source {real490,real32,simulated}; simulated → reports/simulated/
  - ✅ Task 3.1: data/processed/splits_490.json built (370 fields, 799 cycles, 20 spatial blocks, SHA-256 recorded)
  - ✅ Task 3.2: proxies absent on real490 (no 32-farm cwsi_et borrow); test passes
  - ✅ Task 4: smoke tests all pass on real490 (run_rq1–rq7_rq8 + run_data_scarcity all EXIT 0)
  - ✅ Task 5: findings 08/10/11/12 updated; real490 status + sim superseded sections added
  - ✅ Task 6: tests added (test_provenance_complete, test_fail_closed, test_no_32farm_borrow, test_sim_output_path); spec R1–R5 verified; change archived to openspec/changes/archive/
  - See `openspec/changes/rerun-rq-suite-on-real-490/`.

## Queued — openspec candidates (one change each, user-approved direction)

1. **`visualize-metric-products`** — wire METRIC products into the platform:
   - ✅ Interim visualization DONE (user, 2026-11-10): `data/interim/metric_interpolation_timeseries.png` — ticked 2026-11-14; platform wiring still pending
   - `Data/Metric/products/ETaDaily/` — 271 scene rasters (mm/day, 30 m, UTM39N, AOI 496×472)
   - `Data/Metric/products/interpolation/{ETaDaily,CWSI_ET}/` — **4,375 daily interpolated rasters each** (2013-03-31 → …)
   - Zonal stats per field per date → new payload channels (ETa_METRIC, CWSI_ET) → Temporal/Map views + raster tile layers
   - Decide: which product feeds which view (daily interpolation for continuous sliders; scene rasters for truth)
2. **`two-cwsi-channels`** — the two CWSI variants as separate channels:
   `CWSI_ET` (energy-balance) vs `CWSI_LST` (thermal) from `products/`; current heuristic CWSI stays labelled `proxy` or is replaced. Use CWSI-ET + ETaDaily as **proxy references for how the anomaly models perform** (agreement view), while stating plainly that magnitude agreement is NOT phase-anomaly validation.
3. **`phenology-cycle-visualization`** — visualize the segmentation:
   - Export cycle spans per field (`t_start/t_end/harvest_day` → real dates) into the payload
   - Temporal view: shaded cycle bands + harvest markers + the 36×15-day phase-binned series (DSH axis) beside the DOY axis
   - Plain-language doc section (what a "cycle" is and why models score per phase) — expand `docs/platform_score_methods.md` §4.1
3b. **`data-and-phenology-readiness`** — ✅ agent tasks complete (literature review findings 13, cycle audit script, data quality report, stage-conditioned anomaly detection script + findings 14, openspec D5 spec + D11 decision record). ✅ User approved task 4.2 — training sections 2–5 of item 4 unblocked.
4. **`retrain-and-improve-models`** → openspec change `train-and-finalize-main-architecture` — ▶ UNBLOCKED (user approved 3b task 4.2) — training sections 2–5 can now start:
   - D11 decision record adopted (R1–R8 + findings 14); sin/cos DOY auxiliary input (R3) applies to all candidate models; outputs/reports must be publication-ready with full provenance.
   - Tooling (new): save per-seed weights for resume, run_manifest.json, run_dev_matrix.py, check_dev_env.py, Data/Scripts/README.md, dev_log schema test, TDD-first causality/mask tests.
   - ML hardening (new): gradient clipping + NaN guard, deterministic seeding, DOY ablation harness, causal-mask test, FAR calibration with bootstrap CI, model card sidecar.
   - Academic/repro (new): scholar-evaluation rubric, literature-review/citation-management, scientific-writing, benchmark-methodology/eval-harness, mle-workflow, install selected skills.
   - Current checkpoints are deliberately small (`d_model=32`, ≤ ~10⁵ params, 15 epochs, single-shot, trained 2026-11-04)
   - More epochs/parameters/time once data wiring is final; re-train on the final cycle export; keep formula-id/versioning so the payload records *which* checkpoint scored it
   - Include masked_transformer (needs a cycle-compatible artifact) and JEPA scoring
5. **`validation-ladder`** — multi-stage validation as the acceptance frame:
   - L1 learns growth patterns → L2 detects controlled deviations → L3 generalizes across years/farms → L4 agrees with independent agronomic evidence → L5 beats baselines
   - Map existing views/tests to rungs; build what's missing (L2 controlled-anomaly injection, L3 year/field holdouts, L4 expert/ground-truth review — ties to the 4 paused growth-validation tasks)
   - **OPEN QUESTION (user will send the full explanation):** the complete 5-level table + criteria — wait for it before writing this spec.
6. **`era5-covariates`** — wire the unused weather columns: `et0_sum`, VPD, soil moisture, solar radiation, wind, 3/7/14/30-day windows (30-day precip = the right drought signal; ET0 enables a water-balance ETa).
7. **`physical-cwsi`** — Idso/Jackson CWSI `(Tc − Ta)/(Tc_upper − Ta_lower)` using the already-joined ERA5 air temperature; replaces the hard-coded (20 °C, 25 °C) heuristic.
8. **`per-pixel-heterogeneity`** — pixel parquets exist; within-field views (deficit-irrigated corners invisible today).
9. **`score-uncertainty`** — per-bin error curves already computed; show score stability/confidence on the inspector + temporal charts.
10. **`literature-benchmark-view`** — surface `reports/literature_models_benchmark.json` (exists, no view).
11. **`registry-aware-neighbors`** — neighbour graphs from block/office/management instead of centroid kNN.

## Open questions & blockers (restructured 2026-11-14 after the severity/objective deep-research)

Full evidence: `D:\RS\Thesis\research\ssl-stress-advantage\REPORT.md` (27 sources) +
`docs/findings/16_development_protocol_results.md`. Blockers grouped by who unblocks them.

### A. Advisor-gated (Dr. Hamzeh) — blocks the model/protocol track (finding-16 follow-ups)

- **A1. OQ6 — realistic stress signatures & magnitudes for Amirkabir sugarcane** (water deficit,
  salinity, waterlogging, pests/disease, early ripening, lodging; sub-questions from findings 13 §3.3:
  pre-harvest irrigation cut-off timing/magnitude; pre-harvest burning NDVI/LST signature; transient vs
  chronic disturbances; typical mid-season NDVI drop; waterlogging occurrence/months).
  **PARTIALLY ANSWERED 2026-11-14 (user): vegetation-index drops (NDVI, SAVI) are the signals that
  matter — other signature categories (LST rise, waterlogging, burning, …) the user cannot answer.**
  Remaining sub-question = drop MAGNITUDE + duration. Fallback if the advisor is also silent (recorded):
  use the deep-research literature table as the interim default — NDVI/SAVI drops −0.05/−0.10/−0.15/−0.25
  (2–6 bins duration), labelled interim/literature-derived in D6 + model card; severity table is config,
  advisor can adjust later without invalidating infrastructure.
  **Blocks:** D6 perturbation-set update; task 6.2 tuning resumption; final severity-scale decision
  (across-field MAD vs model-residual) and FAR level (cycle vs event vs bin).
- **A2. L4 evidence availability** — can irrigation records, per-field yield, rootzone-ECe/salinity
  surveys, or expert review be obtained for even a subset of the 490 fields? **Blocks:** any L4
  validation plan; without it the thesis caps at L2/L3 + proxy agreement (published SSL-EO norm,
  cf. SAED-ViT SPEI r=−0.909; STS-Net's imposed treatments = the gold standard we don't have).
- **A3. Sign-off on the honest-negative framing + the 3-move plan** (M1 severity protocol /
  M2 normal-bank scorer / M3 event-FAR + POT declustering) — **blocks** committing the normal-bank
  arm as the official task-6.2 direction (more reconstruction tuning would optimize noise, finding 16).
- **A4. Validation-ladder full write-up** (5 levels + criteria) — user will send; blocks item 5's spec. [existing]

### B. User-gated (repo / product / workflow decisions)

- **B1.** METRIC products: which interpolated product is authoritative for daily values (item 1) —
  scene rasters vs `interpolation/` semantics. **Visualization step DONE (user, 2026-11-10):
  `data/interim/metric_interpolation_timeseries.png`**; platform wiring + the authoritative-product
  decision still open. **Blocks:** `visualize-metric-products` (platform part).
- **B2.** Full-archive axis: keep the completeness-stat semantics (invalid acquisitions = missing) as-is? [existing]
- **B3.** `rust/` crate (added by coding-agent commits; `32.7×` claim unreproduced) — keep+verify or remove? [existing]
- **B5.** Tasks 11.1–11.6 (skill adoption: scholar-evaluation, literature-review/citation-management,
  scientific-writing, benchmark-methodology/eval-harness, mle-workflow; 11.6 installs skills locally) — approval pending. [existing]

### C. Technical ambiguities (agent-resolvable; decision recorded here)

- **C1. Severity-unit computation (pairs with A1):** full per-band model-residual distribution
  (residual MAD vs across-field MAD) on the 490-field archive — agent can compute on request; the
  deep-research conversion used sidecar medians only (first-order). **Blocks:** final severity-scale choice.
- **C2. Threshold rule:** adopt SHAZAM's per-DOY score-climatology threshold (fitted mean + 1.64 SD,
  circular DOY regression) as a second calibrated rule alongside the per-(bin,channel) q95 — testable
  in ~1 day; needs a protocol decision.
- **C3. Sugarcane-specific spectral magnitudes:** NDVI/SAVI drop under controlled water stress and
  NDVI-vs-ECe salinity slopes not retrievable from the open web (MDPI/S2 blocked) — row-crop analogues
  (maize/cotton/wheat) + FAO 29 used meanwhile; institutional access to Gómez Flores 2022
  (Remote Sensing 14:3389) would fill the salinity side. **Blocks:** exact salinity-side severity anchoring only.
- **C4. TSB-AD-S injection amplitudes unread** (OpenReview bot-gated; CATS repo 404) — nice-to-have,
  blocks nothing critical.
- **C5. M2 design details** (bank granularity: per-bin vs per-window latents; distance metric;
  fusion rule with the channel-max alarm statistic) — agent's call inside the M2 openspec change,
  flagged here as a recorded decision point.
- **C6. Legacy adapter for deleted `fields.zarr`** (test_ablation) — user-paused 2026-11-07;
  pre-existing failure documented.
- **C7. Environmental 0xc06d007f crashes** (test_daily_anomaly_map / test_eval /
  test_baseline_comparators / test_data_scarcity) — environment/load-order, not code; suite runs with
  `--ignore` for these 5 files → 203 passed 1 failed (=real_archive data-state, C6 family).

### D. Follow-up research paths from finding 20 (user-directed; gated on explicit approval before any openspec change)

Evidence: `docs/findings/20_reconstruction_invariance.md` (signal-path decomposition + absorption,
2026-11-14). Both paths are DESIGN-ONLY until the user approves; no implementation, no retraining,
no architecture edits may start from them without a new openspec change.

- **D1. Path B — re-examine the anomaly definition:** multivariate/relational and phenology-shape
  injection families (the current single-index absolute-drop family is the finding-20 scope limit).
  Design must be informed by the diagnostic (which stages pass/fail there is unknown for relational
  families). User priority: A > B > C (A = behavior analysis = finding 20 itself, now DONE).
- **D2. Path C — forecasting/predictive-coding objective arm (finding-20 decision-matrix remedy):**
  stage-1 dominance in 24/24 cells (signal born at 0.36–2.68σ in train-per-bin-σ units vs τ needing
  ≈3.5–4.9σ per bin) + measured absorption (gru_ae prediction tracks ≈half of persistent drops,
  α≈0.38–0.50 at S4–S6, while frozen B1 absorbs nothing) point at the objective/unit change, not at
  scoring fixes or an architecture swap. A forecasting objective re-anchors σ to temporal
  predictability and makes sustained level shifts structurally unpredictable. Advisor OQ6/A3 sign-off
  still applies before any retraining decision.

### Resolved

- **2026-11-14 — OQ5 / B4 (date count) RESOLVED (user):** one date's metadata was corrupted —
  `data/DATA_SUMMARY.md` §5 names it: **2022-08-07 metadata GeoJSON missing** → **538 usable dates is
  the canonical count** (539 scene files on disk). Counts updated in data/README.md,
  data/DATA_SUMMARY.md, docs/KNOWLEDGE_BASE.md §13, docs/findings/01, docs/findings/14,
  docs/final_plan.md, and the data-and-phenology-readiness change notes (archived change docs left as
  historical record).
- **2026-11-14 — product vision recorded (user):** the pipeline must accept newly added data
  (future acquisitions/fields) without redesign — new data will be reserved for validation; keep the
  current 538-date archive fixed. Any future openspec change must keep the ingest path
  data-refresh-friendly (manifest-driven, no hard-coded date counts).
- **2026-11-14 — desk implemented + archived:** `add-halqeh-anomaly-desk` implemented as the ninth
  platform view (commits 136028c / 9fbde4f / 871e0f1 / a0b1dfb incl. payload export + gate evidence)
  and archived (7e00f17, spec `platform-anomaly-desk`); the working branch was merged — history now on
  `main`.
- **2026-11-14 — `wire-platform-to-real-data` archived** (ff3ccd6; specs: `platform-real-data` created,
  `platform-visualization` updated). Carried reminder cleared.
- **2026-11-14 — pushed:** `main` == `origin/main` (0 ahead / 0 behind); the 2026-11-13/14 session
  commits (contract v5, finding 16, archive, desk) are all on origin. GitHub default branch: verify
  `main` is default after the push (was `kilo/noble-drum-b7o`).
- **2026-11-14 (was: 1704 vs 799 cycle-count discrepancy):** recount (fixed `data_quality_report.py
  --all`) = 800 non-censored cycles, matching the archive 799 within 1 trailing-censoring cycle;
  findings-02 count is stale. Segmentation decision rule applied via the cycle audit — see finding 14
  (exclusion list `reports/cycle_audit/exclusion_cycles.csv`) and finding 15 (method write-up).
  Remaining follow-up: optional winter-harvest segmentation tuning.

### Not blocked (startable on user approval alone)

- M2 normal-bank scorer as an *experimental* arm (openspec change; final adoption gated on A1/A3).
- M3 event-FAR + POT declustering re-computation of finding-16 metrics (analysis-only, no retraining).
- Items 1–3, 6–11 openspec candidates once their user-gated questions (B1) are answered.

## Thesis-level reminders

- Platform = research validation dashboard for pipeline output; claims must match the validation-ladder rung.
- Models must detect **phenology-phase anomalies** — the proxy rule is magnitude-only (CWSI affine); only checkpoint models are phase-aware. Keep this distinction in every validation claim.
- ~~`openspec archive wire-platform-to-real-data` pending~~ — archived 2026-11-14 (ff3ccd6).
8. **After the OQ6 advisor answer (carried over from the archived `2026-11-14-train-and-finalize-main-architecture`):**
   — the authoritative, expanded list of open questions/blockers now lives in
   **"Open questions & blockers" above (A1–A4 / B1–B8 / C1–C7)**; this item tracks the execution side only.
   - Update the interim perturbation set (design D6) per the answer; the extended-severity diagnostic in finding 16 says detectable anomalies start at ~6–8×MAD at cycle-level FAR 5% — decide severity scale (across-field MAD vs model-residual scale) and whether FAR stays cycle-level or moves to bin-level.
   - Resume task 6.2 (single-change model tuning until plateau) — paused because the primary metric had no dynamic range at 1×MAD; tuning on it optimizes noise.
   - Tasks 11.1–11.6 (skill adoption: scholar-evaluation, literature-review/citation-management, scientific-writing, benchmark-methodology/eval-harness, mle-workflow) — pending user approval (11.6 installs skills locally).
   - Optional: winter-harvest segmentation tuning (findings 14/15 follow-up); `openspec archive wire-platform-to-real-data` still pending after its item 1 lands.
