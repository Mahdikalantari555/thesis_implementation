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

- (nothing — next: pick from Queued)

## Queued — openspec candidates (one change each, user-approved direction)

1. **`visualize-metric-products`** — wire METRIC products into the platform:
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
4. **`retrain-and-improve-models`** — after the pipeline settles:
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

## Open questions (waiting on user)

1. **Validation ladder full write-up** (the 5 levels) — send whenever ready; blocks item 5's spec.
2. METRIC products: confirm which interpolated product is authoritative for daily values (item 1) — scene rasters vs `interpolation/` semantics.
3. For the full-archive axis: keep the completeness stat semantics (invalid acquisitions count as missing) as-is?

## Thesis-level reminders

- Platform = research validation dashboard for pipeline output; claims must match the validation-ladder rung.
- Models must detect **phenology-phase anomalies** — the proxy rule is magnitude-only (CWSI affine); only checkpoint models are phase-aware. Keep this distinction in every validation claim.
- Unpushed commits on `main`; `openspec archive wire-platform-to-real-data` pending after item 1 lands.
