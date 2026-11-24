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

- ▶ **compute-placement policy: local → Colab escalation (owner directive, 2026-11-20)** —
  From now on, **every long job runs on Colab, not just training**: training, full
  evaluation, σ calibration, and any export expected to exceed the local ~2 min budget.
  The rule is *validate locally first, then escalate*:
  1. **Local gate (mandatory, before any Colab run):** unit tests covering the touched
     code green; the script's `--smoke` mode run end to end; smoke output read (metrics
     present, no NaN, shapes/counts as expected); the exact command recorded. A pipeline
     that has not passed this gate is NOT handed to Colab.
  2. **Colab run:** the owner executes the handed-over command/notebook.
  3. **Return path:** Colab → Google Drive (dated, well-named bundle), owner downloads,
     hands it to the Arena agent → the Arena agent continues the work → **the Arena agent
     bundles its own outputs back to Drive** and names the exact file for the owner to
     download → owner carries it to the **local agent (Zcode) on their own machine**.
  - **Bundles must be complete, not just checkpoints.** Finding-24 lesson: the 6
    `patch_sigma.npz` files were written into the Colab run dirs and therefore never
    reached Drive/GitHub → the six committed `reports/dev/patch_full_ep10/metrics_*.json`
    quote `patch_sigma_sha256` values whose artifact no longer exists, so those results
    cannot be re-derived. Any bundle SHALL include every artifact its reports hash-bind:
    checkpoints (`last.pt` + epoch files), `run_manifest.json`, `progress.jsonl`,
    `norm_stats.json`, **`patch_sigma.npz`**, and the metrics JSON.
  - **Naming:** Drive bundles are dated and self-describing, e.g.
    `patch_ep10_colab_2026-11-19.zip`, `sigma_eval_colab_2026-11-20.zip` — never a bare
    `results.zip`. Both directions use the same convention so the owner always knows
    which file to download and where it belongs when unpacked.
  - **GitHub stays text-only.** `.gitignore` already blocks `/data/` (line 80),
    `checkpoints/` (33) and `reports/figures/*.png|html` (56–58); no `.pt`/`.ckpt` has
    ever been tracked and the repo is 55.5 MB (largest blob 4.14 MB) vs GitHub's 50 MB
    warn / 100 MB hard limit. **Nothing needs changing there** — weights never go to
    GitHub; only metrics JSON/CSV + findings go, and the binaries go to Drive.
  - Deferred, owner-optional: a `scripts/bundle_run_artifacts.py` that packs a run plus a
    SHA-256 manifest so a missing sidecar (the σ case) fails at bundle time instead of at
    eval time. Not written yet — the manual naming convention above covers the need for now.
- ▶ **`phenology-method-v2` (owner-stated current focus, 2026-11-20)** — the new phenology
  method is the thing being finished now; the immediate downstream consequence is a
  **second heavy GRU training run on Colab** once it is implemented. Status today: the
  phenology `PhaseAxis` contract exists (`config.py:18` → `doy`/`dsh`/`gdd`/`landmark`,
  default `DSH`) and the alternative axes have working implementations
  (`phenology/align.py`: `compute_dsh`, `compute_gdd`, `phase_bin`;
  `baseline/b2_phenology_axes.py`: `progress_axis`, `bin_by_progress`, `gdd_progress`,
  `landmark_progress`), but **only `DSH` is wired into the training path** — `PhaseAxis`
  is referenced nowhere outside `config.py`, so choosing GDD/landmark currently changes
  nothing in `train_ssl.py` / `patch_data.py`. Remaining work to call the method
  "implemented": (a) make the axis selection actually reach tokenization/bin assignment in
  the training path, (b) tests for the non-DSH axes in the train path (today only
  `test_align.py` + `test_baseline_comparators.py` cover the math), (c) smoke run +
  inspected output, (d) **then** the Colab GRU retrain per the compute-placement policy
  above, (e) σ + eval on the same eval-subset hash so the result stays comparable with
  findings 18–24. Blocking question for the owner: does "new phenology method" mean
  wiring an existing axis (GDD/landmark), or a new axis/alignment not in the repo yet?
  Do not start the Colab retrain before that is answered.
- ▶ **2026-11-19 full-archive patch run + G4 probe (owner run mandate)** —
  - ✅ Pre-flight: patch unit battery 285 tests → 2 failures fixed (`test_holdout_is_required` was not hermetic once the real `data/processed/holdout_manifest.json` existed — `load_holdout_manifest` default now resolves at call time; `test_patch_severity_cli...` still expected the pre-`_s{seed}` metrics filename from before the 08c969f collision fix).
  - ✅ Extraction default-size correction: `P.DEFAULT_PATCH_SIZE` is 32 (locked schema default), NOT 48 as assumed — the first full-extraction attempt failed the fail-closed geometry gate at field 6 (bbox 33×12 > 32). Relaunched with explicit `--patch-size 48` (the PoC protocol; store attrs prove the PoC used it) + `--mask` + `--splits`. **Full store DONE 2026-11-19: `data/processed/patches.zarr`, 370 fields × 799 recorded cycles, S=48, 36 bins × 8 ch, provenance bound to splits 9aa5eb11, wall ≈ 3.6 h** (370 fields: 120 fields have only censored cycles and are absent by design; 37 test-block fields present but read-gated).
  - ✅ Full-store norm: `data/processed/norm_stats_full.json` sha **3cb1ca60** (447 locked train cycles, 240 fields; the PoC norm d9fe7438 stays at `norm_stats.json` — the trainer/eval default resolves next-to-store, so the owner's commands carry `--norm` explicitly).
  - ✅ G4 localization probe (`patch-level-controlled-trial` task 2.4 + 4.1): `localization_auprc` + `patch_localization_scores` in `monitor/patch_eval.py` (TDD, 8 tests) + `scripts/probe_patch_localization.py`; 6 PoC checkpoints on the same eval subset (6186edf8). **Verdict AGAINST the CNN**: cnn_gru 0.4504 ± 0.0137 vs pixelwise 0.4507 ± 0.0053 (gap −0.0003) → per the pre-registered rule **patch_pixelwise is the scale-up candidate**; both arms localize at chance for S1–S3 (detection at S1 is a cycle-level aggregate effect). Finding 23 EN+FA updated; gate table complete.
  - ✅ `platform-patch-scorer-wiring` openspec change PROPOSED (validate OK) — registry slot + exporter patch-scoring path + trial-stage badge + owner-gated flag-source switch; BLOCKED on owner approval + frozen winner.
  - ✅ Persian finding 23 (`docs/findings/23_patch_level_poc.fa.md`, persian-writing audit clean).
  - ✅ `--cpu-threads` heat throttle added to `train_patch.py` (owner request): caps torch CPU pools, rides the config hash, recorded in the manifest; smoke verified identical loss at 4 threads.
  - ✅ Full-store plumbing smoke chain: train `--smoke` → `calibrate_patch_sigma` → `eval_dev --smoke` all green (§7 seed manifest gap fixed; smoke artifacts removed so the real `--out` is free).
  - ✅ **HANDOVER EXECUTED + FULL-ARCHIVE VERDICT (2026-11-20, finding 24)**: owner trained all 6 runs on Colab T4 (cuda, ~247 s/seed, git 36b6161, results zip verified 84 files, bindings green: norm 3cb1ca60 / holdout 39284712 / splits 9aa5eb11 / 37 holdout-excluded); agent ran σ ×6 + severity eval ×6 on the same panel (eval hash 6186edf8) → **MDM80 = S1, all four families × both patch arms × 3 seeds, detection 1.000 (CI [0.947,1.0], n=68), delay 30 d** — field-level refs all stay >S6. Full-archive gate PASS; scale-up rule satisfied → **frozen winner = patch_pixelwise**; freeze + platform wiring stay BLOCKED on owner approval of `platform-patch-scorer-wiring`. Record: `docs/findings/24_patch_full_archive.md` (+ .fa).
  - ⚠ **CORRECTION (2026-11-22, doc-sync to finding 25):** the verdicts in the two
    entries above are **void as measurements** — the finding-23/24 patch numbers
    (G4 localization AUPRC, "Full-archive gate PASS", "MDM80 = S1, detection
    1.000, delay 30 d") were produced by the pre-2026-11-21 patch protocol
    (onset-0 detection accounting + duplicate-round CIs, fixed `ea74a0c`; on top
    of the `bd65829` uncalibrated-τ/NaN artifacts). Corrected verdict =
    **finding 25** (v2 chain, 3,478 cycles/490 fields, eval hash 852e1ba3…):
    MDM80 **censored beyond S6** for both arms × 4 families × 3 seeds;
    full-archive gate **FAIL**; **no frozen winner exists** (the finding-24
    "frozen winner = patch_pixelwise" call is void; `platform-patch-scorer-wiring`
    stays dead). Supersession banners added to findings 23/24; amendment in
    `patch-level-controlled-trial/tasks.md`.
  - ✅ **GATE UPDATE (2026-11-22, later the same day — owner ordered immediate
    implementation of the whole backlog): the training-path fixes are LANDED,
    gate lifted for training.** Landed: #3/#15/#16 (`b082283`), #24 (`e9221cd`),
    #17 (`cbeff9f`), #2 + dated E7 amendment (`d7432a9`), #1 (`adae60d`), and
    hygiene #4/#5/#6/#8/#10/#11/#12/#19/#20/#21/#22/#23/#25/#26/#28 (this
    commit). 103 targeted unit tests green before commit; full battery recorded
    with the commit. Future training runs start from the FIXED code (a new
    generation — the finding-25 checkpoints remain as recorded; the deviations
    section stays a true record of how they were trained).
    **Still pending:** #7/#13/#14/#18/#27 in `scripts/eval_dev.py` (file held
    by the parallel writer) — #13's fix changes the eval-subset hash, so the
    finding-25 numbers remain bound to the recorded 852e1ba3… protocol until a
    re-eval; do not re-run eval before #13 lands. Plus #9 (farm_local mask
    raise) and #29 (staged-machinery annotations).
- ▶ **`cycle-recovery-diagnosis` (user-requested phase 0, 2026-11-19)** — quantify how many of the
  ~5,600 lost field-years (missed winter harvests → censored/skipped spans in `cycles_490.parquet`)
  a multi-index detector could recover, vs the frozen NDVI-only v1 (`segment_cycles`, prom 0.15 /
  180–540 d, `plateau_frac` 0.7 vs field-global ymax). Advisor hypothesis: SAVI (soil-adjusted
  winter floor) + LST (opposite-polarity harvest jump) + SWIR1 corroboration.
  **Read-only phase 0**: reproduces v1 verbatim, classifies every lost span (obs-hole /
  fusion-recoverable / unresolved), counts v2-candidate valid cycles from inserted harvests,
  reports a threshold-sensitivity grid. NOTHING frozen is touched; outputs only under
  `reports/cycle_detection_diagnosis/`. The v1↔v2 comparison is itself a keepable scientific
  deliverable (owner decision); any real detector-v3 (splits/store/holdout rebuild) stays a
  separate openspec change gated on these numbers.
  - ✅ **Phase 0 DONE (2026-11-19, `scripts/diagnose_cycle_recovery.py`, wall 288 s)**: v1
    reproduced verbatim (799/799 clean, 1755/1755 windows, 0 mismatches; harvest list captured
    via a `detect_harvests` shim — cycle objects alone lose silently-skipped harvests).
    1,819 lost spans = 5,625.7 field-years (all 478 cycle-bearing fields affected;
    lead 1,401 + gap 2,987 + trail 1,238). **Class breakdown kills the obs-hole hypothesis:
    only 2 spans / 1.1 field-years are observation holes** — winter obs exist almost everywhere;
    the binding constraint is v1's NDVI-only criteria (global-ymax 0.7 plateau, rise-within-180 d).
    Fusion (SAVI drop ∨ NDVI drop + LST rise ∨ SWIR rise, ≥180 d dedup) yields
    **3,416 candidate clean cycles at the primary point (+2,617 over v1 ≈ 4.3×)**, stable across
    the full 18-combo threshold grid (v2 ∈ [3,059, 3,927], strictest corner +2,260). CANDIDATES
    are an upper bound — precision unvalidated until a reviewed subsample + (ideally) company
    harvest records. Unresolved 374 spans / 539 fy mix "no harvest truth" with "criteria miss".
    Next: reviewed-precision subsample → owner gate → separate v3 openspec change.
  - ✅ **UPDATE 2026-11-20: owner SKIPPED the precision subsample (cannot label manually) and
    directed straight to v3** — openspec change `cycle-detection-v3-multindex` (validated);
    fused detector implemented + 12 tests (`segment_cycles_fused`, FusionParams frozen at the
    measured point, `map_exclusions_by_window` for the finding-14 verdicts — v1 cycle_id
    numbering does not carry over). **Local gate §6.1 passed** (smoke 8 fields inspected);
    full fused export run locally (~7 min, owner refined compute-placement: only >1 h AND
    CPU-heavy → Colab): **`cycles_490_v2.parquet` = 4,458 windows / 3,478 non-censored /
    ALL 490 fields, median length 366 d** (v1 799/478 fields untouched — 4.4× recovery).
    D6 gates 4/4 PASS (`reports/cycle_detection_diagnosis/v2_table_gates.md`).
    **`splits_locked_v2.json` sha 30ecbe10** (same k-means blocks; train 2,105 / dev 340 /
    test 773 / monitoring 1,240; exclusions transferred by overlap 38→38) +
    **`holdout_manifest_v2.json` sha d501bf81** (audit green). Extraction smoke green
    (store↔splits↔cycle bindings verified). **NEXT: owner runs the local extraction
    handover (~overnight; estimator 83 min is optimistic — v1 scaled ≈ 12–15 h) + norm v2;
    then agent verifies store → `thesis_data_full_v2.zip` + notebook v2 → Colab training
    of the SAME 6-trainings protocol on the v2 store.** The v1↔v2 comparison remains a
    standing thesis deliverable; candidate precision stays unvalidated (company harvest
    records remain the L4 validator).
- ✅ **`decomposition-aware-ssl-study`** — ARCHIVED 2026-11-23 (spec `decomposition-aware-anomaly-study` merged). Isolated runner and [runbook](docs/decomposition_aware_ssl_study.md). Note: real benchmark/ablation numbers for its STL features still require the canonical polygon zonal store; smoke is not empirical evidence (applies to the capstone's B5/C-decomp reuse).
- ▶ **`rerun-rq-suite-on-real-490`** (proposed 2026-11-10) — RQ1–RQ8 + data-scarcity results in findings 11–12 came from `simulate_490_field_archive`, not the real 490-field archive; re-run on real data, fail closed, provenance in every report.
  - ✅ Task 1: real_archive.py loader + hash check + provenance + --source arg (tests/unit/test_real_archive.py 4/6 pass)
  - ✅ Task 2: all 8 run_rq*.py + run_data_scarcity.py refactored with --source {real490,real32,simulated}; simulated → reports/simulated/
  - ✅ Task 3.1: data/processed/splits_490.json built (370 fields, 799 cycles, 20 spatial blocks, SHA-256 recorded)
  - ✅ Task 3.2: proxies absent on real490 (no 32-farm cwsi_et borrow); test passes
  - ✅ Task 4: smoke tests all pass on real490 (run_rq1–rq7_rq8 + run_data_scarcity all EXIT 0)
  - ✅ Task 5: findings 08/10/11/12 updated; real490 status + sim superseded sections added
  - ✅ Task 6: tests added (test_provenance_complete, test_fail_closed, test_no_32farm_borrow, test_sim_output_path); spec R1–R5 verified; change archived to openspec/changes/archive/
  - See `openspec/changes/rerun-rq-suite-on-real-490/`.

## Active openspec changes (2026-11-23) — derived from the change folders; full task lists live in each `tasks.md`

**`regional-ssl-representation-evaluation`** — PROPOSED, 16/42. The capstone. Code built and green
(`pytest tests/evaluation` 355 passed; six scripts smoke-pass), but **no real number exists yet** —
every artifact needs owner-side data (v2 parquet/splits/holdout, T4 hour budget, v2 patch store) and
owner-run training. Next actions (in order):
1. Phase 0 registration: freeze `configs/evaluation/pre_registration.yaml` (H1–H5, G-A…G-E, fractions, seeds) on owner approval to apply;
   verify artifact bindings (`cycles_490_v2.parquet`, `splits_locked_v2.json` sha 30ecbe10…, `holdout_manifest_v2.json` sha d501bf81…);
   rebuild/record the v2 normalization sidecar; commit the pre-registration hash.
   *Frozen in-tree* (`configs/evaluation/pre_registration.yaml`, gates G-A…G-E with executable
   thresholds, `bind_artifacts` rejects a weakened criterion); the v2 bindings are still owner-side.
2. B1–B5 ladder first (Phase 1) — the FAR-matched `ndvi_lst_rule` control must exist before any SSL claim;
   capacity manifest; scoring bridge; causality tests.
3. Level-1 pretraining-quality battery (Phase 2) → architecture screening of region model v0 (Phase 3),
   dev-only, 3 seeds, selection by gates only.
4. Level-2 transfer suite + T3 retrieval with near-match listing (Phase 4) → generalization rolls (Phase 5)
   → decision gates + findings 26 (Phase 6). Capstone claims nothing above L3; applied ambition (vision.md item 9) is vision-stage.
- Blocked by: owner approval to run `/opsx:apply`; v2 store re-extraction/norm v2 for the patch-dependent parts stays out of scope here.

**`benchmark-latent-reconstruction-prediction-architectures`** — IN PROGRESS, 26/37 (ticked 2026-11-23; 11 boxes annotated with what is missing).
Harness built and green (26/37 boxes ticked, each open box annotated with what is missing): `src/ssl_anomaly/benchmarking/` (config, matrix, registry, bindings, inputs,
splits, synthetic, capacity, collapse, interface, arms+suppression, metrics, calibration, scoring,
checkpoints, stats, protocols, indices, hpo, runner, report, bundle), `configs/benchmarks/`
(**`matrix.yaml` committed** — 357 cells, `matrix_sha256 03a7b6cd6c4f…`, Pending 297 / Deferred 30
`patch-v2-unbound` / Not Applicable 30), `scripts/benchmarks/{run_benchmark,run_hpo,report_benchmark,
bundle_benchmark,make_colab_notebook}.py`, `notebooks/benchmarks/architecture_comparison_benchmark.ipynb`
(20 generated sections), `tests/benchmarking` **139 passed**, fixture+smoke end-to-end verified twice
(18/20 and 16/20 Completed; the difference is the collapse gate firing on noise — see
`docs/findings/27_benchmark_harness_local_gate.md`). **No scientific claim**: every manifest from a
plumbing run carries `stamp = SMOKE FIXTURE` and `ladder_rung = L0`.
1. Owner runs the notebook §9 (screening, T4) then §10 (full block, 3 seeds, `--resume`); the agent
   never launches it (§6.1 gate is complete and reproducible).
2. Bundle back with `--collect` (complete-or-defect, `SHA256SUMS.txt`); report regenerates from the
   bundle alone.
3. The 14 questions are answered only after ≥2 seeds and ≥8 farms in common; on the fixture the cell
   factor explains ~4.5× the variance of the model factor, so a single-seed ranking is declared
   meaningless **before** looking at real numbers.
- Blocked by: owner-side data (list below); T4 hour budget for the first cycle; v2 patch re-extraction
  decision (patch cells stay `Deferred` until the store + norm sidecar are bound). Findings numbering:
  27 assigned to the harness/gate record; the results finding gets the next number at completion.

### Owner data-prep tasks — architecture benchmark (2026-11-23)

Everything below is a **Drive upload or a repo binding decision**; no code is needed from the owner.
Items 1–2 already exist and need no action beyond confirmation; 3–6 are the gaps that currently make
parts of the matrix unrunnable, and each names what unblocks.

| # | task | exact target | what it unblocks |
|---|---|---|---|
| 1 | Confirm the v2 cycle table is the one to read | `MyDrive/thesis_colab/thesis_data_full_v2.zip` (293,475,431 B, present) contains `data/processed/cycles_490_v2.parquet` + `splits_locked_v2.json` + `holdout_manifest_v2.json` + `norm_stats_full_v2.json` | the whole farm-level matrix; the runner pins the hashes of the files it actually reads, so no separate checksum upload is needed |
| 2 | Do **not** upload `thesis_data_full.zip` as a benchmark source | v1 stays as-is (historical comparison only) | keeps the v1↔v2 distinction honest — the benchmark reads v2 only |
| 3 | Export and upload a **daily ERA5/weather sidecar** per farm-cycle (the W-axis input): `et0` (grass-reference ET), plus the fields already used by the platform channel list, at the same 15-day bin grid as the cycles | `MyDrive/thesis_colab/weather_daily_v2.zip` (naming to match `artifacts.weather_daily_v2`), or an added file inside the v2 zip + a one-line `amendments:` entry | I2/I3 with real weather, W1 vs W2 (gated late fusion), and the ETa/ETrF/CWSI_ET reference indices — today CWSI_ET is reported `unavailable` and ETa/ETrF fall back to `absent` because no ET0 series is bound |
| 4 | Upload **labelled confirmed-anomaly / harvest-loss records** (field id, date range, type, source), if any exist outside the remote sensing archive | `MyDrive/thesis_colab/ground_labels_v2.zip` | the only way to test "synthetic anomaly ≠ confirmed stress": without it every index/arm agreement stays a remote-sensing-internal comparison, and the independence checks (`assert_not_circular`) have no real label source to point at |
| 5 | Re-extract the **v2 patch store** + its `patch_sigma.npz`-class norm sidecar, or say "not this cycle" | `data/patch_runs/v3/patches.zarr` (+ sidecar) inside the v2 bundle | 30 `Deferred` patch-level cells (`reason = patch-v2-unbound`). Binding them is a config edit + matrix re-derivation, not code work; leaving them Deferred is a legitimate answer and the report states it |
| 6 | Provide a **standalone `cycles_490_v2.parquet`** (or confirm the in-zip copy is authoritative) if the run is to be repeated from a laptop instead of Colab | `MyDrive/thesis_colab/cycles_490_v2.parquet` + `sha256` recorded in the run notes | local re-planning of the exact cells (the `--plan` path runs without data; a local *execute* needs the table) |
| 7 | Decide the **T4 hour budget for the first launch** (contract says 6 h) | reply or edit `configs/benchmarks/benchmark_config.yaml → budget.t4_hours` **as a recorded amendment** | how much of the 297 pending cells fits per launch; overruns become `Deferred` with a reason, never silent truncation |
| 8 | STL/decomposition feature table (finding-20 follow-up), if the phase-axis alternative is to be scored | `MyDrive/thesis_colab/stl_features_v2.zip` | the `PhaseAxis` alternative inputs (GDD/landmark) named in 6.2 — the DSH default needs nothing extra, so this is optional |
| 9 | Bind the **B1 / GRU-AE incumbent checkpoint** (+ its `norm_stats.json` and config) if the incumbent is to appear as a reference arm | its path recorded in the run's `bindings.json` — the benchmark refuses a missing binding rather than retraining a stand-in | 8.4 reference arms + the 58 `B1`/`BL-*` matrix cells that are still `Pending` |
| 10 | Confirm the **T4 hour budget** for the screening launch (contract: 6 h) | `budget.t4_hours` edit *as a recorded amendment* | how much of the 297 pending cells fits per launch; the excess becomes `Deferred` with a reason, never silent truncation |



**Blocking minimum: 3 and 5** (item 9 only decides whether the incumbent appears as a reference row). With neither, the run still produces the primary
paradigm/backbone/suppression comparison on I3/W0–W2-farm-level data, and the report prints the
missing-coverage rows; nothing is silently substituted. Rows 1–2 need no action beyond a confirmation; 3–10 are decisions or uploads.


**`cycle-detection-v3-multindex`** — ACTIVE, 26/27. The v2 producer; keep as-is. Only open item is the optional `--sweep`
re-gate follow-up (gate FAILED → no switch; production path stays off). Owner call: pursue after the v2 store completes or close it.

**DORMANT — need an explicit owner decision:** `oracle-floor-audit` (14/15; pool-V σ-floor audit of the voided patch arms)
and `patch-alarm-statistic-e1` (0/11; channel-max residual aggregation follow-up). Revive with a fresh data/provenance contract, or reject.
Nothing superseded is to be applied (see `openspec/changes/INDEX.md`); archived 2026-11-23:
`decomposition-aware-ssl-study`, `training-holdout-invariant`, `separate-verification-gates`
(main specs created: `decomposition-aware-anomaly-study`, `training-holdout`, `research-verification`).

## Queued — openspec candidates (one change each, user-approved direction)

- **2026-11-23 triage (owner decision):** the v1-era patch line and the GRU-prediction
  line are SUPERSEDED by the v2 chain + finding 25; `openspec/changes/INDEX.md` added;
  superseded folders marked (not archived); `oracle-floor-audit` and
  `patch-alarm-statistic-e1` left DORMANT for owner review. Archive candidates were
  archived 2026-11-23 (see "Active openspec changes" above).
- **`regional-ssl-representation-evaluation`** (PROPOSED 2026-11-23) — the capstone:
  Level-1 pretraining quality + Level-2 downstream transferability of a light
  **regional** SSL representation of field dynamics (B1–B5 input ladder, 5/10/25/50/100%
  label fractions, T1–T5 tasks, temporal rolls + leave-block-out generalization,
  pre-registered gates G-A…G-E). Architecture (incl. Barlow Twins candidate) is NOT
  pre-approved — pre-registered screening decides. Semantics: anomaly ≠ stress
  candidate ≠ stress; synthetic = controlled evidence only; geographic scope out.
- **`benchmark-latent-reconstruction-prediction-architectures`** (PROPOSED 2026-11-23,
  owner decision) — heavy counterpart, standalone Colab T4 benchmark: 5 paradigms ×
  GRU/Mamba (A1–A10) × suppression (S0–S3) × 5 synthetic anomaly families × 3 severities
  × 3 validation protocols; AUPRC primary; budget-aware execution with Deferred
  visibility; three winner reports; feeds the capstone but no capstone gate depends on it.
- **`madeline-yield-embeddings-init`** (TRACKED, not created 2026-11-23; owner decision) — Madeline's path: initialise/fine-tune the region base model from external early-season yield-forecasting-model embeddings. Self-supervised trajectory use needs no yield labels, but the external artifact itself needs a data, licence and provenance contract before an openspec change is written. Not implemented; not scheduled.
0. **`code-audit-2026-11-21-deferred`** — the deferred defect list from the post-bd65829
   audit (`docs/audits/code_audit_2026-11-21.md`): the 3 result-corrupting bugs are FIXED
   (99709b0 / ea74a0c / d4b3cb8, σ refit + severity re-run required); the remaining
   3 P1 latent defects (farm_own shrinkage no-op, absorption polarity, GRU init slice),
   26 P2 hygiene items and 10 perf opportunities are RECORDED there and must be picked
   up as small fixes before the code they touch is next used. Nothing on this list
   corrupts a current result.
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
- **C8. Temporal selection window (owner validation guide §2, 2026-11-15):** the guide proposes
  train ≤2021 / val 2022–2023 / test 2024–2026; the LOCKED splits (9aa5eb11) use test start
  2024-01-01 with dev cycles cut at that boundary — no 2021/2022 boundary exists. Agent
  recommendation (recorded): KEEP the locked splits for comparability with findings 18/21/22;
  adopt 2022–2023 as the SELECTION sub-window within dev for the final architecture decision IF
  the dev 2022–23 cycle count supports it (counting the dev cycles by period = the decision's
  first task). Changing the locked splits mid-program would break the pre-registered trials.

### D. Follow-up research paths from finding 20 (user-directed; changes CREATED 2026-11-14, awaiting execution)

Evidence: `docs/findings/20_reconstruction_invariance.md` (signal-path decomposition + absorption,
2026-11-14). Owner approval 2026-11-14: both changes below were written and validated; execution is
the next step. Owner prohibitions recorded verbatim: do NOT re-develop Normal Bank, do NOT bring
foundation models into the main path, do NOT trial multiple architectures in parallel — only these
two experiments.

- **D1. Path B — re-examine the anomaly definition:** multivariate/relational and phenology-shape
  injection families (the current single-index absolute-drop family is the finding-20 scope limit).
  Decided as the FALLBACK path: taken only if the predictive trial's synthetic-detection gate fails
  (see `predictive-ssl-controlled-trial` decision gates). No implementation exists yet.
- **D2. Path C — forecasting/predictive-coding objective (finding-20 decision-matrix remedy).**
  Two changes created and validated 2026-11-14, to run in order:
  1. **`forecast-unit-probe`** (zero-training, agent-runnable in minutes): pre-registered untrained
     causal forecasters (persist/med3/trend4) inside the exact production statistic; same eval
     subset + injections; MDM80 + event-FAR + waterfall. Success = non-censored MDM80 ≤ S5 on
     persistent/ndvi → path C justified with a measured target floor; all censored → honest negative.
     → finding 21.
     **OUTCOME (2026-11-14, run complete): honest negative — ALL FOUR ARMS censored ">S6" on
     persistent/ndvi** (persist/med3/trend4 + B1 reference recomputed in-harness; B1 crosscheck
     vs finding-18 = 0 mismatches). Mechanism: the clean one-step residual scale is NOT small —
     σ_forecast ≈ 0.23 NDVI (≈1.3× the across-cycle spread; the design's 0.01–0.02 estimate was
     ~10× off) — so the S6 drop is only 1.2–1.7 forecast-σ vs a ~3.6–4.0σ per-bin requirement;
     no untrained arm even reaches gru_ae's requirement ratio (best persist 0.44 vs gru_ae 0.48);
     carry-forward forecasters absorb the level (A ≈ 0.75–0.81) — the naive-forecaster unit
     upside is disproven. Evidence: `docs/findings/21_forecast_unit_probe.md`,
     `reports/dev/forecast_probe_20261114/probe_summary.json`. The trained `gru_pred` trial
     below still runs with its own five gates (a trained model could shrink σ_forecast); weight
     shifts toward path B unless it passes.
  2. **`predictive-ssl-controlled-trial`** (trains ONE new arm `gru_pred` — same trunk/inputs/budget
     as gru_ae_main, one-step-ahead objective only; seeds {13,42,2026}; user runs training from
     agent-provided commands): four parts — A natural-data FAR behavior, B the unchanged finding-18
     severity protocol vs B1+gru_ae_main, C the absorption experiment (error at three checkpoints +
     A ratio vs gru_ae's α≈0.45), D agreement with independent indicators (CWSI/ETa/ETrF/TVDI —
     proxy evidence, never stress proof; unavailable sources = `absent`). Pre-registered decision
     gates decide: predictive = main path, or fall back to D1. → finding 22.
     **STATUS (2026-11-14, implementation COMPLETE, awaiting user-run training):** gru_pred arm
     implemented + smoke-green (commit 60fc792: `DeepPredictiveScorer` — naturally causal single
     pass, per-(bin,channel) masked smooth-L1 β=1.0 pre-registered in
     `train_main.PREDICTIVE_OBJECTIVE`; 174,008 params = gru_ae's exact count; 10 unit tests).
     Parts A–D scripts built + smoke-green (commit 521bb94: `trial_part_{a,b,c,d}_*.py` via
     `trial_common.py`; part B recomputes all three arms on trial-scoped artifact copies and
     crosschecks B1/gru_ae vs the recorded finding-18 rates at 0-mismatch; part D availability
     runtime-verified — **CWSI_ET PRESENT as a 32-field subset in
     `Data/Processed/satellite/stats_current.zarr` (cwsi_et, field×539 dates)**, CWSI_LST/ETa/ETrF
     absent as processed field stats, TVDI derived proxy from existing per-scene LST/NDVI stats).
     Five-gate evaluator ready (commit 5ee5feb, `scripts/trial_gates.py` — operationalizations
     recorded before the full run). NEXT: user runs the three training commands (~1–2 min/seed,
     handed off 2026-11-14) → `trial_part_b_severity.py` → A/C/D → `trial_gates.py` → finding 22.
  Advisor OQ6/A3 sign-off still applies before any retraining decision beyond this trial's scope.

- **D3. Patch-level program (created 2026-11-15; 11 changes PROPOSED, all valid, NONE applied;
  2026-11-16 audit added `farm-local-anomaly-scoring` and extended the comparison study).**
  Owner directive (notes.md 2026-11-15 + validation guide same day): add patch-level analysis,
  subset-first, with runtime estimation + checkpoint/resume built in; the owner request AMENDS the
  "no parallel architecture trials" prohibition for THIS program only. Architecture source:
  `research/Architecture_Selection_for_Self_Supervised_Spatiotemporal_Anomaly.md` (2D CNN stem →
  per-location GRU, prediction primary + masked-recon aux, ~90k params, no pooled latent).
  Progress 2026-11-16: **R0.5 zonal distribution arm IMPLEMENTED**
  (`monitor/zonal_dist.py` + `scripts/run_zonal_dist_study.py`): the three-way
  prompt structure Zonal Mean / Zonal Distribution Statistics / Patch now has
  its middle rung, built from the statistics already in `stats.zarr` with no
  training and no extraction (presets level=8ch ≡ b1 / lower=8ch p10-only /
  spread=14ch / dist=45ch; the `level` arm is proven to reproduce `b1`
  exactly). 27 tests. NEXT for this arm: the severity/MDM80 sweep on the locked
  evaluation panel (user data), the `b0_calendar` DOY variant and the
  `p10`-only path-max scorer.
  The audit follow-up verifies locked patch extraction, the train-only norm
  sidecar and causal train/scoring inputs on fixtures. See the
  [patch input runbook](docs/patch_inputs.md). Next: audit the **existing**
  freeze on the archive machine, then verify a fresh bound export and review
  the preregistered patch PoC; do not refreeze or recompute frozen checkpoints.
  Fold-matrix execution stays blocked until fold-aware calibration and scoring
  are wired. The checkout suite does not certify the strict real-data gate.
  Execution order: 0 `training-holdout-invariant` (STANDING invariant, owner
  2026-11-15: holdout farms/dates frozen BEFORE implementation, fail-closed in
  every training path, pixel-level included) → 1 `patch-data-pipeline` (patches.zarr) → 2 `patch-level-controlled-trial`
  (subset PoC, gates G1–G5, finding 23) → 3 `compare-spatial-representation-strategies`
  (representation ladder R0 zonal mean / **R0.5 zonal distribution statistics (std, p10, iqr —
  already in stats.zarr, no training)** / R1 patch-32 / R2 sub-crops / R3 multi-scale, plus the
  heterogeneity battery (static-pattern control, white-noise null, per-pixel alarm frequency), the
  extent sweep {0.1,0.3,0.6,1.0}, the paired detectability-crossing table and required 30 m
  resolution arithmetic; gated on PoC gates; owns the scale sweep delegated from E5; finding 24) +
  `patch-scoring-ensemble` (E8 + C1/C2/M3; basis labels) +
  `farm-local-anomaly-scoring` (NEW 2026-11-16: farm-local vs population-ranked basis labelling,
  per-field shrunk σ arm, cross-farm independence tests, ratoon/variety scale strata, farm-level
  anomaly summary; answers the owner's "compare each patch with the same farm's own history"
  requirement) +
  `gradual-severity-shape` (3rd temporal shape + secondary peak/recovery shapes) → 4
  `patch-contamination-absorption` (E7) +
  `patch-full-run-and-ablations` (scale-up + E2–E6 + TCN/Mamba challengers + SSL-off control +
  classical baselines + error analysis + pre-registered final criterion) → 5
  `patch-generalization-suite` (L3 matrix) + `patch-independent-indicators` (E10 + physical CWSI +
  weather/neighbor external consistency) + `patch-xai-attribution` (patch-level XAI, new
  capability). Conditional fallback: `anomaly-definition-path-b` — pre-registered, applies ONLY if
  BOTH path-C trials fail their detection gates.

### Resolved

- **2026-11-16 — spatial prompts audited against the written changes (agent):** the owner's two
  spatial research prompts (controlled representation comparison; farm-internal detection) were
  mapped requirement-by-requirement in `docs/audits/spatial_prompts_vs_openspec_2026-11-16.md`.
  Outcome: prompt 1 ≈80% covered, prompt 2 ≈60% covered, nothing implemented (all changes open).
  Amendments applied the same day: `compare-spatial-representation-strategies` gained R0.5
  distribution statistics, the heterogeneity/noise battery, the extent axis, the crossing table,
  honest R0 pairing and the resolution-report requirement; `patch-level-controlled-trial` gained
  the farm-boundary input-masking requirement + invariance tests; `patch-scoring-ensemble` gained
  basis labelling with a farm-local default; `patch-independent-indicators` gained the
  population-referenced indicator-z label and a farm-own variant; `patch-full-run-and-ablations`
  E4 gained the ratoon/variety cell; and the new change `farm-local-anomaly-scoring` owns the
  per-field σ arm, independence tests and farm-level summary. Remaining known limits recorded in
  the audit (223/370 fields have ≥2 cycles ⇒ partial pooling is forced, not chosen).
- **2026-11-15 — B6 (five-shape protocol) DECIDED (user):** recommendation adopted —
  peak-disruption + recovery-disruption promoted as pre-registered SECONDARY diagnostic shapes
  (folded into `gradual-severity-shape`; primary gates stay on the index-drop family, no
  goalpost movement); growth-delay stays inside path-B gating (D1 unchanged). With the three
  primary shapes + two secondary shapes, the validation guide's five synthetic types are fully
  mapped.
- **2026-11-15 — A4 (validation-ladder write-up) DELIVERED (user):** the full validation guide
  arrived and is mapped onto the existing L1–L5 ladder (identical levels): claims weighted L1–L3
  primary + L4 external consistency + L5 benchmark; synthetic detection = sensitivity evidence,
  never stress proof; External Consistency Analysis (weather + neighbors) added to
  `patch-independent-indicators`; model selection without GT = sensitivity/stability/FAR-burden
  balance; novelty = the COMBINATION, held as hypothesis (queued item 12). Guide content stored
  in agent memory (validation-guide-2026-11-15); operational pieces landed in the D3 changes.
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
- **Item 12 `novelty-positioning-verification`** (owner guide §10, 2026-11-15): the claim "first
  unsupervised agricultural anomaly detection" is NOT defensible — 2017 unsupervised RS anomaly
  paper, 2024 aerial-MIM AD (already in corpus: Shikhar & Sobti 2404.08931), Oct-2026
  Mediterranean water-use screening (Scientific Reports). Verify the 2017 + 2026 papers against
  the 84-paper matrix, then phrase novelty as the COMBINATION (long-term per-farm Landsat +
  per-farm historical pattern learning + phenology-cycle awareness + SSL temporal representation +
  label-free growth-deviation detection + multi-level validation) — hypothesis pending
  verification, not a proven claim — in method_design_ssl.md and the thesis text.

## Thesis-level reminders

- **Standing holdout invariant (owner, 2026-11-15):** farms and dates completely
  excluded from training are identified and frozen BEFORE any implementation —
  holdout = test-block farms (whole-field, all periods) + post-2024 dates + all
  future acquisitions/fields; fail-closed enforcement in EVERY training path,
  pixel-level included (`training-holdout-invariant`). Always observed.
- Platform = research validation dashboard for pipeline output; claims must match the validation-ladder rung.
- **Claims weighting (owner validation guide, 2026-11-15):** main thesis claims on L1–L3; L4 =
  external consistency evidence (weather + neighbors + indicators); L5 = benchmark comparison.
  Synthetic detection is sensitivity evidence, never stress proof — the thesis deliverable is a
  trustworthy GROWTH-ANOMALY detection system, not a stress classifier.
- Models must detect **phenology-phase anomalies** — the proxy rule is magnitude-only (CWSI affine); only checkpoint models are phase-aware. Keep this distinction in every validation claim.
- ~~`openspec archive wire-platform-to-real-data` pending~~ — archived 2026-11-14 (ff3ccd6).
8. **After the OQ6 advisor answer (carried over from the archived `2026-11-14-train-and-finalize-main-architecture`):**
   — the authoritative, expanded list of open questions/blockers now lives in
   **"Open questions & blockers" above (A1–A4 / B1–B8 / C1–C7)**; this item tracks the execution side only.
   - Update the interim perturbation set (design D6) per the answer; the extended-severity diagnostic in finding 16 says detectable anomalies start at ~6–8×MAD at cycle-level FAR 5% — decide severity scale (across-field MAD vs model-residual scale) and whether FAR stays cycle-level or moves to bin-level.
   - Resume task 6.2 (single-change model tuning until plateau) — paused because the primary metric had no dynamic range at 1×MAD; tuning on it optimizes noise.
   - Tasks 11.1–11.6 (skill adoption: scholar-evaluation, literature-review/citation-management, scientific-writing, benchmark-methodology/eval-harness, mle-workflow) — pending user approval (11.6 installs skills locally).
   - Optional: winter-harvest segmentation tuning (findings 14/15 follow-up); `openspec archive wire-platform-to-real-data` still pending after its item 1 lands.
