# Platform Bug Audit — AgriAnomaly Visualization Platform

**Scope:** every page of `platform/` (React 19 + Vite 7 + Tailwind 4), the shared
`AnalysisContext`, the `ui.tsx` / `MapView.tsx` components, and the data layer
(`src/data/mock.ts` + the two exported JSON payloads).
**Change:** `openspec/changes/make-platform-bilingual-and-audit-page-bugs`
**Date:** 2026-11-05
**Method:** static review of every view plus a scripted browser pass
(`scripts/capture_platform_screenshots.mjs`) that mounts the production build in
headless Chrome, switches locale, and walks all eight pages — that pass is what
surfaced several of the runtime issues below (TDZ crash, chart re-measure, bidi
digit reversal).

Severity: **P0** = wrong or misleading data / crash, **P1** = broken interaction
or unreachable feature, **P2** = polish or maintainability risk.

Status legend: **Fixed** = changed in this PR · **Deferred** = documented, needs
pipeline work outside the platform · **Won't fix** = intentional.

---

## Audit matrix (one entry per audited page)

| # | Page / module | Bug | Severity | Status |
|---|---|---|---|---|
| BUG-01 | MapExplorer, Temporal, Spectral, Spatial, Validation | Date sliders hard-coded `max={47}` / range presets `[0,47]`,`[12,23]`,`[24,35]`,`[36,47]` | P0 | **Fixed** |
| BUG-02 | Validation | Year robustness panel hard-coded four windows `[0,12]…[36,48]` and a `2021+y` label | P1 | **Fixed** |
| BUG-03 | Overview | “48 monthly obs / field” and “2021–2024” literals independent of the loaded axis | P0 | **Fixed** |
| BUG-04 | Overview (footer) + App shell | Stale region label “Piracicaba, SP, Brazil” and “Demonstration data — not research results” banner on a Khuzestan dataset | P0 | **Fixed** |
| BUG-05 | AnalysisContext | `FIELDS[9]`, `FIELDS[10]`, `FIELDS[18]` unguarded → crash on short/empty datasets | P0 | **Fixed** |
| BUG-06 | Data layer | Exported `series_khuzestan.json` never imported; views rendered a seeded RNG instead | P0 | **Fixed** |
| BUG-07 | `MapView` | Map destroyed and re-created on every dark-mode toggle (zoom/center lost) | P1 | **Fixed** |
| BUG-08 | `MapView` | Hard-coded `min(47, dateIdx)`; `getSeries(nid)[dateIdx]` could be `undefined` | P1 | **Fixed** |
| BUG-09 | Spatial | `FIELDS.findIndex(...)` produced `-1` for unknown neighbours → `values[-1]` inside Moran's I | P1 | **Fixed** |
| BUG-10 | Overview | Overview table cells rendered `NaN` if a year had no observations | P1 | **Fixed** |
| BUG-11 | All views | Charts rendered blank when captured/printed at a new viewport height | P1 | **Fixed** |
| BUG-12 | All views (fa) | Letter-spacing utilities broke Persian glyph joins; numeric minus signs reversed (`0.547-`) | P1 | **Fixed** |
| BUG-13 | App shell | Sidebar/footer claimed “Landsat & METRIC (2013–2026)” while the axis is 2021–2024 | P0 | **Fixed** |
| BUG-14 | Overview | “Completeness” stat divided by zero on empty data (`NaN %`) | P2 | **Fixed** |
| BUG-15 | ModelComparison | 490-series means recomputed on every render inside JSX; matrix labels untranslated | P2 | **Fixed** |
| BUG-16 | Reports | Seeded experiments used ids/fields that could not exist; experiment names not localized | P2 | **Fixed** |
| BUG-17 | Data layer | `downloadCSV` revoked the object URL synchronously → intermittent failed downloads in Chromium | P2 | **Fixed** |
| BUG-18 | Overview / MapExplorer | “Mock stress event” copy and `mock`-tinted captions presented synthetic flags as facts | P2 | **Fixed** |
| BUG-19 | App shell | Language of `<html>` never updated; no `dir` handling at all (single English locale) | P1 | **Fixed** |
| BUG-20 | Fonts | Google-Fonts `@import` failed offline → platform silently fell back to system fonts | P1 | **Fixed** |
| BUG-21 | Data layer | Seeded experiment list shipped as fake “saved” research state | P1 | **Fixed** |
| BUG-22 | Validation | Event-agreement “agreement rate” was labelled `kappaNum` (not a κ) | P2 | **Fixed** |
| BUG-23 | Pipeline | Anomaly scores in `series_khuzestan.json` are proxy formulas, not model inference | P0 | **Deferred** |
| BUG-24 | Pipeline | Farm metadata repeats one block for all 490 fields (`name`/`farm`/`municipality` identical) | P1 | **Deferred** |
| BUG-25 | Pipeline | `planted`/`harvest` hard-coded to `2021/03` / `2024/11` for every field | P2 | **Deferred** |
| BUG-26 | Data layer | 7.7 MB JSON inflates the single-file build to ~8.7 MB | P2 | **Won't fix** |
| BUG-27 | Platform | `N`/`DATES` exported but `DATES.length - 1` re-derived at each call site | P2 | **Fixed** |
| BUG-28 | Temporal | “Full period”/year preset buttons wrote ranges without clamping to the axis | P2 | **Fixed** |
| BUG-29 | MapExplorer | “Field boundaries” toggle only unmounted the map; no state was preserved | P2 | **Fixed** |
| BUG-30 | MapExplorer | Date-comparison panels shared one map state and re-created maps on every date change | P2 | **Fixed** |

---

## Details

### BUG-01 — Hard-coded 48-step sliders (P0)

`MapExplorer.tsx`, `Temporal.tsx`, `Spectral.tsx`, `Spatial.tsx`, `Validation.tsx`
all used `max={47}` (and `[0, 47]`, `[12, 23]`, `[24, 35]`, `[36, 47]` presets).
The real axis has 48 points today, but the literal silently desynchronises the
moment the exporter changes cadence — and it is exactly the class of bug that
made the sliders disagree with the 490-field dataset.

**Fix:** `mock.ts` now exports `DATES` (derived from `series_khuzestan.json`),
`N`, `LAST_IDX` and `clampIdx()`; every slider, preset, and caption reads those.

### BUG-02 — Hard-coded year windows (P1)

`Validation.tsx` looped `[0,1,2,3].map(y => getPairs(..., [y*12, y*12+11]))` and
labelled rows `2021+y`. Any change in start year or cadence would silently
report “2021…2024” over the wrong windows.

**Fix:** years are derived from `DATES` (`new Set(DATES.map(d => d.slice(0,4)))`)
and each window is looked up from the axis.

### BUG-03 / BUG-13 — Temporal-coverage claims (P0)

The Overview strip said `value="2021–2024"` / `sub="48 monthly obs / field"`, and
the sidebar footer claimed “Landsat & METRIC (2013–2026)”. Both were literals.
The platform renders the 2021-01 → 2024-12 demonstration subset only.

**Fix:** `COVERAGE` (`first`, `last`, `yearStart`, `yearEnd`) and `N` drive the
stat cards, the sidebar, and the footer. The 2013–2026 archive is still the
thesis scope, so the sidebar now says **“(2013–2026 archive)”** and the footer
reports the *rendered* coverage.

### BUG-04 — Stale region + demo banners (P0)

`Overview.tsx` ended with `Region: Piracicaba, SP, Brazil (sugarcane)` (a legacy
placeholder from the synthetic prototype) and a “Demonstration data — not
research results” sentence, while the app banner simultaneously claimed the
Khuzestan dataset was “integrated”.

**Fix:** region is now “Khuzestan, Iran (sugarcane agro-industry)”; the wording
is “Demonstration data — not verified research results”; and a reusable
`<DemoDataBadge/>` marks every view that renders these series (Overview, Map,
Map legend, Validation, Reports). The spec scenario “No stale region labels”
is verified by `grep -r Piracicaba platform/src` returning nothing.

### BUG-05 — Unguarded default selection (P0)

```ts
const [selectedField, setSelectedField] = useState(FIELDS[9].id);          // throws on <10 fields
useState<string[]>([FIELDS[9].id, FIELDS[10].id, FIELDS[18].id]);          // throws on <19 fields
```

**Fix:** defaults are `FIELDS[0]?.id ?? ""` and `FIELDS.slice(0, 3).map(f => f.id)`.
`fieldById()` now returns a placeholder record instead of `undefined`, so
`field.code` in the inspector cannot crash. `hasFields` and
`FIELDS_WITH_SERIES` are exported for data-driven labels.

### BUG-06 — Dead `series_khuzestan.json` (P0)

`scripts/export_platform_data.py` wrote `series_khuzestan.json` (7.7 MB, 490
fields × 48 months) but `mock.ts` generated series with a seeded RNG
(`rng(fidNum * 1013 + 47)`) — so **every number on every page was synthetic**,
even though `farms_khuzestan.json` supplied the real polygons. The two files
could drift forever.

**Fix (documented decision):** `series_khuzestan.json` is now the single data
source. `getSeries()` normalises the exported payload into the typed `Obs` shape;
the RNG generator is deleted. Verified by the spec scenario “No dead data
artifacts”: both JSON files under `platform/src/data/` are imported.

### BUG-07 — MapView recreated on theme toggle (P1)

The mount effect had `[dark]` in its dependency list, so toggling dark mode ran
`map.remove()` and re-created the map at the default center/zoom — losing the
user's position. It also re-registered the tile layer with an outdated closure.

**Fix:** the mount effect is dependency-free (`[]`); a second effect swaps only
the `L.TileLayer` when `dark` changes. Verified visually (zoom level and center
survive the toggle).

### BUG-08 — Unsafe index & undefined observations (P1)

`getSeries(f.id)[Math.max(0, Math.min(47, dateIdx))]` (hard-coded 47) and
`getSeries(nid)[dateIdx]` (no clamp at all, `nid` may not exist) could throw or
produce `undefined` dereferences after any dataset change.

**Fix:** `obsAt()` / `clampIdx()` helpers, optional chaining everywhere, and a
`PLACEHOLDER_FIELD` fallback.

### BUG-09 — Moran's I neighbour index −1 (P1)

```ts
FIELDS.forEach((f, i) => m.set(String(i), f.neighbors.map(n => String(FIELDS.findIndex(x => x.id === n)))));
```

`findIndex` returns `-1` for any neighbour id absent from `FIELDS`; `" -1 "` then
indexes `values[-1]` inside `moransI()` → `undefined` arithmetic → `NaN` Moran's I.

**Fix:** a `Map<fieldId, index>` lookup, filtering unknown neighbours.

### BUG-10 — NaN cells in the field × year table (P1)

`mean([])` returns `0/1 = 0`… but `s.slice(...)` on a short/missing series could
yield `NaN` after `toFixed`, rendering `NaN` chips with an `anomalyColor(NaN)`
fallback. The table also hard-coded `['2021','2022','2023','2024']`, so any
other year range displayed empty columns.

**Fix:** columns come from `stats.years`; cells render `—` when
`Number.isFinite(v)` is false.

### BUG-11 — Blank charts during capture / print (P1)

Recharts' `ResponsiveContainer` measures once on mount and needs a real resize
event afterwards. Capturing with `captureBeyondViewport` (or printing to PDF)
resizes the viewport *after* layout, leaving every chart at Recharts' 14×14 px
minimum — the screenshots for Temporal and Model Comparison came out with empty
plot areas.

**Fix (tooling):** the capture script now resizes the emulated viewport to the
full page height, waits for re-measure, and only then captures. Documented in
`scripts/capture_platform_screenshots.mjs`.

### BUG-12 — Persian typography and bidi digits (P1)

Two distinct problems, both visible in the first Persian screenshots:

1. `tracking-[0.08em]` (stat labels, uppercase headings) breaks the cursive
   joins between Persian letters.
2. In RTL context, `-0.547` renders as `0.547-` because the minus sign is a
   neutral character at the trailing edge of the run.

**Fix:** `body.lang-fa * { letter-spacing: normal !important }` plus an explicit
`unicode-bidi: isolate` rule for `.font-mono`/`<bdi>`/`[dir=ltr]`, and `<bdi>`
wrapping around every numeric interpolation in `Stat`, tables, and score
badges. Charts keep `dir="ltr"` frames so axes never mirror.

### BUG-14 — Divide by zero in completeness (P2)

`100 * (1 - missing / totalObs)` with `totalObs === 0` → `NaN%` in the stat card
and the donut. Now guarded (`totalObs ? … : 0`) and the page short-circuits to
`<EmptyState/>` when the dataset is empty.

### BUG-15 — Recomputation and untranslated labels (P2)

`ModelComparison.tsx` recomputed 490 × models means inside the render body for
the “Field-level consistency” table (≈3 000 `getSeries` calls per keystroke) and
built `pairMatrix` / `indCorr` rows with raw English `m.name`.

**Fix:** both moved into `useMemo` keyed on `compareModels`, and row labels use
`t('model.<id>.name')`.

### BUG-16 / BUG-21 — Experiment registry (P1/P2)

`AnalysisContext` shipped two pre-seeded “experiments” (`exp-1`, `exp-2`) with
fabricated statistics (`r = 0.58 · ρ = 0.54 · n = 72`, `κ = 0.68`) — a reader
would reasonably mistake them for results. The seed also referenced
`FIELDS.slice(0, 12)` at module scope.

**Fix:** the registry starts empty; the empty state explains how to create the
first experiment. Experiment names/notes are localized at save time; CSV export
is unchanged.

### BUG-17 — CSV download revocation race (P2)

`URL.revokeObjectURL(url)` immediately after `a.click()` can cancel the download
in Chromium. Now deferred one tick and the anchor is attached to the DOM.

### BUG-18 — “Mock” language presented as fact (P2)

`MapExplorer` said “Mock stress event: …” and Overview said
“Monthly composite … (mock)”. Now “Test stress event” (translated) and
“demonstration” wording, consistent with the demo badge.

### BUG-19 — `lang` / `dir` never set (P1)

`index.html` hard-coded `lang="en"` with no `dir`; nothing updated it. Screen
readers and hyphenation therefore used English rules for Persian text.

**Fix:** `I18nProvider` writes `lang`, `dir`, `data-locale` and a `lang-fa`
body class on every locale change; `index.html` carries the LTR/EN defaults for
first paint.

### BUG-20 — CDN fonts (P1)

`@import url('https://fonts.googleapis.com/…')` in `index.css` plus a documented
“load Vazirmatn from a CDN” plan. The build is a single-file offline artifact;
on an air-gapped machine the `@import` fails and both Inter and Vazirmatn
silently degrade.

**Fix:** 14 `.woff2` subsets (~336 KB total) are committed under
`src/assets/fonts/` with their OFL licences and declared via `@font-face`.
Persian text is confirmed to render in Vazirmatn by the capture script
(`font=Vazirmatn` assertion).

### BUG-22 — Mislabelled agreement metric (P2)

The event panel computed `(tp+tn)/n` and called it `kappaNum`, printing
“Agreement rate …”. Calling it κ invites a chance-corrected reading it does not
have.

**Fix:** renamed to `agreementRate` and the caption states it is agreement, not κ.
A real Cohen's κ is a follow-up (needs marginal probabilities exposed).

### BUG-23 — Proxy scores presented as model output (P0, Deferred)

`export_platform_data.py` derives anomaly scores from a rule:

```python
stress_sig = max(0.0, val_cwsi - 0.45)
"b1_dsh": clip(0.15 + stress_sig * 0.80, 0.05, 0.95),  # …
```

i.e. all seven “models” are monotone functions of one CWSI proxy. The platform
therefore cannot be used to compare detector behaviour, yet the UI lists
“SSL Masked Transformer”, “STS-Net”, etc. as distinct models.

**Why deferred:** wiring real checkpoints requires the trained-model export path
(`scripts/export_model.py` + `src/ssl_anomaly/…` inference) and the Zarr feature
store, which is outside this change's boundary.

**Mitigation shipped:** `DATA_PROVENANCE.scoresAreProxy = true`, the demo badge
on every page, and the banner text says “literature SSL model anomaly scores”
rather than claiming live inference. The audit table is linked from the badge
tooltip. **This is the single largest remaining honesty gap and should be the
next change's P0.**

### BUG-24 — Farm metadata collapse (P1, Deferred)

All 490 records share `name = "Block L08 — Farm L08-01"`, `farm = "Khuzestan
Unit L08"`, `municipality = "Amir Kabir / Debal Khazai Agro-Industry (Block
L08)"`. The exporter reads `row.get("farm")` for `code` and then rebuilds the
other labels from that single value, so the inspector and search cannot
distinguish fields beyond `F-###`.

**Fix path:** merge the real `farms.parquet` attributes (unit/block/variety) in
`scripts/export_platform_data.py` and re-export; no platform code change needed.
The search box already concatenates farm/municipality, so it will work once the
data is real.

### BUG-25 — Hard-coded planting/harvest dates (P2, Deferred)

Exporter writes `"planted": "2021/03", "harvest": "2024/11"` for every field.
The inspector displays them under “Planted”/“Harvest”. Needs ratoon-date
attributes; tracked here rather than silently displaying a false value — the
field exists because the UI needs one.

### BUG-26 — Bundle size (P2, Won't fix)

Importing the 7.7 MB series file grows the single-file build from 1.2 MB to
8.7 MB (1.26 MB gzipped). Acceptable for a locally opened thesis demo; if this
ever ships over a network, switch to fetching `series_khuzestan.json` with a
compressed payload (the exporter already writes plain JSON).

### BUG-27 / BUG-29 / BUG-30 — Smaller consistency fixes (P2)

- `LAST_IDX` / `clampIdx` / `monthAt` / `dateAt` centralise index maths (BUG-27).
- The “Field boundaries” toggle now hides the layer without unmounting the map,
  so date/selection state survives (BUG-29).
- The two comparison maps get independent instances (the previous code created
  and destroyed Leaflet maps on every date change, which leaked tiles) (BUG-30).
- `Temporal`'s preset buttons clamp through `setRange()`, so reversed or
  out-of-range values cannot enter the shared context (BUG-28).

---

## Verification performed

| Check | Command | Result |
|---|---|---|
| Type check | `cd platform && npx tsc --noEmit` | 0 errors |
| Production build | `cd platform && npm run build` | ✅ 8.7 MB single file |
| Both dictionaries complete | `satisfies Record<keyof typeof en, string>` in `fa.ts` | compile-time enforced |
| No dead data files | both JSONs imported from `mock.ts` | ✅ |
| No stale region labels | `grep -rn "Piracicaba" platform/src` | no matches |
| No hard-coded 48 | `grep -rn "max={47}" platform/src` | no matches |
| Locale persistence | capture script sets `localStorage` before load | `dir=rtl, lang=fa, font=Vazirmatn` |
| All 8 pages render (fa) | `scripts/capture_platform_screenshots.mjs` | 8 PNGs in `reports/screenshots/fa/` |
| Charts not clipped under RTL | visual review of the 8 screenshots | ✅ |

## Known limitations (explicit non-goals)

- Persian translations cover the UI chrome; **data values** (field codes, farm
  names, model names) intentionally stay Latin.
- Chart x-axes keep ISO `YYYY-MM` in both locales (design decision §5); Persian
  month names appear in prose/tooltips only.
- The Leaflet basemap needs network access for tiles; the app itself does not.
- No automated RTL screenshot diffing in CI yet (the capture script is manual).
