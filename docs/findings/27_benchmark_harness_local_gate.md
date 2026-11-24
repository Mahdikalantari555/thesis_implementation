# Finding 27 — Architecture-benchmark harness: what the local gate proved, and what it did not

**2026-11-23 · agent-built harness, owner-executed runs still pending · rung L0 (fixture/smoke:
plumbing only) · claim noun: synthetic anomaly · change
`benchmark-latent-reconstruction-prediction-architectures` · branch `arena/867db619-thesis-implementation`
(off `e9c5079`)**

This document records **no comparison result**. Nothing in it says that a reconstruction arm beats a
prediction arm, or that suppression helps, or that a Mamba backbone transfers. Those are the questions
the change exists to answer, and the answers require the Colab runs, which have not happened. What this
finding *does* report is the state of the machinery that will answer them, because five defects found
today would each have produced a plausible-looking wrong answer in a first real run — and a defect that
survives into a results table is the expensive kind.

## 1. Scope of what now exists

`src/ssl_anomaly/benchmarking/` is the harness; the notebook (`notebooks/benchmarks/
architecture_comparison_benchmark.ipynb`) and `scripts/benchmarks/*` are its only two front ends, and
neither contains a line of experiment logic:

| layer | module | what it guarantees |
|---|---|---|
| contract | `config.py` | frozen `benchmark_config.yaml` + `suppression_applicability.yaml`; every config field consumed exactly once by `validate_config`, which returns `[]` |
| design | `matrix.py` | 357 cells committed as `configs/benchmarks/matrix.yaml` (`matrix_sha256 03a7b6cd6c4f…`), each cell hashed; the runner refuses to start against a drifted matrix |
| state | `registry.py` | exactly 7 states, append-only log, illegal transitions raise (a `Completed` cell cannot be quietly re-run) |
| arms | `models/bench/arms.py` + `suppression/` | A1–A10 behind one interface, objectives audited distinct, `masking_applied_at="train_only"` |
| data | `inputs.py`, `splits.py`, `synthetic.py` | I1/I2/I3 + W0/W1/W2, admissible-split guard (a `test`/`monitoring` request raises), 5 families × 3 severities with disjoint train/eval event pools |
| scoring | `metrics.py`, `calibration.py`, `collapse.py`, `indices.py` | macro-AUPRC over the 15 cells with prevalence recorded, per-farm scores, ECE with equal-frequency bins, collapse disqualification + content probe, TVDI/CWSI_LST/ETa/ETrF references (CWSI_ET declared unavailable, not approximated) |
| inference | `stats.py`, `report.py`, `protocols.py` | farm-clustered paired bootstrap (B ≥ 1,000), Wilcoxon, Friedman + Nemenyi, Cliff's δ, Holm within family, seed/farm/cell/residual variance decomposition (**no temporal/event factor — see §4**); three separate winner tables; V1/V2/V3 with farm-block consistency asserted |
| fairness | `capacity.py`, `hpo.py` | the contract's ±20 % band enforced on every arm (spread 1.22 on the fixture); equal trial/wall budget per arm; the hidden knobs are what the search tunes; selection refused on any forbidden split, and the audit re-reads the trace rather than trusting the caller |
| transport | `bundle.py` | complete-or-defect bundles with `SHA256SUMS.txt`, verified after transfer |

## 2. The gate, and the five defects it caught

AGENTS.md §6.1 requires: unit tests → `--smoke` end-to-end → read the smoke output → write the exact
command → verify `--resume`. All five steps ran today. Every item below was found *by* that sequence,
each with a test now pinned to it:

1. **A missing key read as a null result.** `report.py` aggregated `metrics.macro["auprc"]` while
   `macro_summary` writes `macro` — so the winner table printed "no usable cells" next to a registry
   showing 18 completed cells. An empty table is indistinguishable from an honest absence, which is
   precisely why this class of bug survives review. `PRIMARY_MACRO` now reads the key that exists, and
   the table is asserted non-empty against a real manifest.
2. **A statistics API invented instead of read.** `holm()` returns `p_adjusted`/`reject_at_0.05`; the
   table read `p_holm`/`reject` and printed `—` for every adjusted p-value. Same for
   `n_cells` → `n_cells_mean_per_farm`. Both columns now render, and the Holm column is checked against
   a direct `stats.holm({...})` call.
3. **Global cell intersection silently destroying every paired comparison.** A farm carries a
   (family, severity) cell only where its cycles actually received that event, so intersecting the cells
   of all 20 farms yields the empty set and each comparison degraded to "no farms in common". Pairing
   is now per (farm, cell) and averaged per farm; on the fixture that is the difference between
   `Δ +0.0918 [+0.0728, +0.1111], δ = 0.910` and nothing at all.
4. **`NaN` written into JSON manifests.** `json.dumps` emits the bare token `NaN`, which `jq`,
   `JSON.parse` and strict pandas reject — an unparseable manifest makes a whole bundle a defect.
   `_jsonable` now maps non-finite floats to `null` at the serialisation boundary.
5. **A resume that extended training, and a hash that no run could match.** The training loop ran
   `range(start, start + epochs)`, so a resumed cell received *more* epochs than its never-crashed twin
   while both reported `epochs: N`; it is now `range(start, max(start, epochs))` with
   `epochs_this_invocation` and `resumed_from` in the manifest. Separately, the runner hashed the
   applicability table by reading a `sha256` key the file never declares: every cell carried an empty
   applicability hash, and a matrix committed with the real one was refused at start-up. One helper
   (`config.applicability_sha256_of`) is now the single source of truth, and
   `verify_committed_matrix` recomputes it instead of trusting the committed file — which is how a stale
   pin is caught before any GPU hour is spent.

6. **A run directory that could not describe itself.** `bundle.RUN_REQUIRED` lists `matrix.yaml`, but
   the runner only ever *read* the matrix from `configs/benchmarks/` — so `bundle_benchmark.py --audit`
   returned `complete: false` with **zero cell defects**, which reads like a data problem and is a
   packaging one: a bundle is supposed to be readable without the repo next to it. The runner now
   mirrors the exact matrix it bound to into the out dir, and a test asserts both the mirror and the
   audit's refusal of an unauditable run.

7. **Three baselines that were one baseline.** `_run_reference` picked its estimator with
   `transfer.get("fitter") or "robust_centre"` — and `assert_transfer` never returns a `fitter` key. So
   Isolation Forest, One-Class SVM and robust PCA all fitted the same centre-and-spread rule, every
   searched knob was ignored, and the fixture HPO run proved it: three arms, three different search
   spaces, and *identical* scores to five decimals (0.02751). The estimator is now declared once, in
   `suppression/classical.FITTERS`, next to the estimators; `FITTER_CONSUMED` states which knobs each
   fitter reads; a selection key that no code consumes is a `RunPlanError` rather than a dropped
   argument; and `run_hpo.py` skips an arm whose space its fitter cannot use and writes the reason into
   `hpo_selection.json`. After the fix BL-IF = 0.0272 with `n_estimators=200, max_samples=0.8,
   contamination=0.05` reaching the actual `IsolationForest`, BL-OCSVM = 0.0379, and BL-RPCA is recorded
   as `SKIPPED — the 'rpca_pca' fitter reads ['contamination', 'n_components'] and nothing else, so
   ['iters', 'lam'] could not be searched`. That last one is **an open contract problem, not a passing
   test**: either the robust-PCA estimator grows a λ/iteration path or the space is amended; a silent
   no-op search was the only thing worse than saying so.
8. **A handoff the notebook promised and the code did not have.** Section 8 of the generated notebook
   tells the owner that the winning configs join the run via `--selection-file`; no such flag existed,
   and `RunOptions` had no selection field, so HPO output could only have been pasted by hand into the
   frozen contract — which the contract forbids. `--selection-file` now exists end to end: it refuses a
   file whose own split audit did not pass, refuses `d_model` (the capacity band is not searchable), and
   applies each arm's config through the runner — structural keys into `build_arm`, optimisation keys
   into a per-cell view of `training` (so the manifest's `optimizer.lr` is the lr that ran), reference
   keys into the fitter — while every cell manifest records `selection_applied`. The selection is a
   *layer*, recorded per cell; the contract keeps its hash.

Two more were caught at the boundary between modules: `metrics.cell_metrics` had no per-farm output at
all (so the bootstrap had nothing farm-level to resample), and `TensorFrame` carried no cycle dates (so
V1's temporal-holdout assertion could not be evaluated — it now derives a monotone proxy from row order
and *states* that it did).

## 3. Fixture-smoke evidence (L0 — plumbing, not results)

`python scripts/benchmarks/run_benchmark.py --execute --yes --fixture --smoke --tier screening
--block screening` on this machine, no CUDA, no v2 archive:

* `cells_planned 20 · cells_executed 20 · wall 66.8 s · by_status {Completed: 18, Failed: 2}`;
  registry after the run `{Completed 18, Failed 2, Deferred 30, Not Applicable 30, Pending 277}` —
  i.e. **no cell vanished**, and the 30 patch-level cells are `Deferred` with
  `reason = patch-v2-unbound` rather than quietly absent.
* The `Failed` rows are latent-arm collapses (`A7.S0/S1` in the first run, and `A3` as well on a
  repeat: **16 Completed / 4 Failed vs 18 / 2**). On 2 epochs of synthetic noise the effective-rank
  ratio of a JEPA/latent arm falls below the contract floor (0.1), so the cell is **disqualified, not
  scored low**. The unstable *count* is the expected behaviour of a gate on noise, not a flaky test — and
  it is recorded here rather than citing the friendlier of the two runs. That the gate fires at all is
  the evidence wanted; which arm tripped it says nothing about real data.
* Objective audit: `problems []` (no two arms optimise the same thing); capacity band: all 10 arms in
  0.8–1.2× of 120,000 params, spread 1.22 (A1 120,456 → A7 100,986); per-arm hour estimates recorded
  from this run (A1 0.00429 h … A8 0.01102 h), which is what makes the T4 budget *predictive* instead of
  aspirational.
* Report: `report_benchmark.py` renders §0–§8 (11 `## ` headings: what-this-is, coverage, primary
  metric, the three factor tables, secondaries, variance, cost, the 14 questions, what-would-change-it),
  writes `REPORT.md` + `report.json` next to the run, and exports the evidence pack. `evidence/results.csv`
  has **637 data rows over 44 columns**: all 357 matrix cells appear — the 20 executed ones as
  15 rows each (one per (family × severity) cell, *including* the disqualified `Failed` arms) and the
  other 337 as one status row each, so the CSV cannot be mistaken for a table of 637 results.
  `evidence/statistical.json` carries the bootstrap/Holm/Friedman payload; the report also prints
  "no cell is citable (fixture/smoke or incomplete)" on stderr, which is the only correct summary here.
  `warnings: ['fixture/smoke run: no scientific claim in this document']`, and every manifest carries
  `stamp = "SMOKE FIXTURE — plumbing evidence only, not a scientific result"` plus
  `ladder_rung = "L0 (fixture/smoke: plumbing only)"`.
* Statistics rendered on that fixture (numbers listed **only** to prove the paths compute): paradigm
  macro-AUPRC `hybrid 0.0998 / input_reconstruction 0.0997 / future_prediction 0.0994 / jepa 0.0455 /
  latent_reconstruction 0.0187`; `future_prediction` vs `jepa`: 20 farms, Δ +0.0918,
  95 % CI [+0.0728, +0.1111], Wilcoxon p = 0.000, Holm p = 0.0000, Cliff's δ = 0.910 (large);
  Friedman χ² = 175.6 (df 4) over 174 complete (farm, cell) blocks, Nemenyi CD 0.462; variance shares
  `seed 0.000 · farm 0.024 · cell 0.452 · model 0.070 · residual 0.454`.
  The `seed 0.000` row is the point of the whole exercise: **one seed cannot separate models here — the
  cell (family × severity) explains ~4.5× more variance than the architecture does.** Whether that
  survives on real data is Q1/Q2, unmeasured.
* Bundle: `--collect` → 180 files, `SHA256SUMS.txt` + `BUNDLE_MANIFEST.json`, `verify` clean; flipping
  one byte in a scored cell's `norm_stats.json` made `verify` return `ok: false` naming that file. The
  second fixture run (`--out /tmp/cli_run2`) audited `complete: true` with `matrix.yaml` among the run
  files, and `collect → verify` is now a test, not a transcript.

Unit battery: `tests/benchmarking/` **142 passed** (`--no-cov`), including the
`collect → verify → tamper` bundle round trip and the run-dir self-description check. The rest of `tests/` is unchanged by
this work — `git diff HEAD` over tracked files is empty; every line of it is new files. (In this sandbox
51 pre-existing failures / 31 collection errors remain in modules that do not import `benchmarking`;
they are environment- and data-dependent and are reported rather than hidden.)

## 4. What the harness refuses, on purpose

* A run never reads `test`/`monitoring` rows: the only row-returning path calls
  `splits.assert_admissible`, and `protocols.build_protocol` counts and records the excluded rows.
* HPO may not move `d_model` (`run_hpo.py` drops it and says why): the parameter band is what makes the
  arms comparable, so a search that could widen it would let an arm win by being larger.
* Calibration records which partition it was fitted on and the contract's `sources`, because the
  pre-registered rule *is* `thresholds_selected_on: dev` — dev is simultaneously the threshold-selection
  and the reported partition, and that optimism is written into the threshold record instead of being
  dressed up as independence. It is also why the contract's rung ceiling (L3) cannot be reached on dev
  alone.
* An index whose inputs are the label's source raises (an ET index scored on ET-derived labels is a
  tautology), and CWSI_ET reports `available: false` because the archive lacks the aerodynamic term.
* The variance decomposition reports the four factors it can actually separate — seed, farm, cell,
  residual. The task asks for a temporal/event factor too; that needs a run with real cycle dates
  (fixture rows carry synthetic dates by construction), so the factor is absent from the table rather
  than faked. Interaction terms (architecture × family, suppression × architecture) are likewise not
  fitted anywhere yet.
* The ±20 % capacity band is **fixed by the contract and not searchable**. HPO tunes
  `lr / weight_decay / batch_size / epochs / horizon_k / dropout / loss weights` instead; allowing it
  to move `d_model` would let an arm win by being bigger, so `run_hpo.py` drops the key from every
  space and says so in `hpo_selection.json`.
* A partial bundle is a defect. There is no "upload it anyway and note the gaps later". Two
  corollaries were needed to make that rule true rather than merely strict: the run directory carries
  its own copy of the bound `matrix.yaml` (so a bundle describes itself without the repo), and
  `meta.json`/`last.pt` are required as a **pair** only where a checkpoint can exist — a random-init
  control or a fitted reference estimator trains nothing, so "no checkpoint" is a recorded design fact,
  while a sidecar without its checkpoint is a truncated write and stays a defect.
* `--plan` does not need a GPU (it trains nothing); requiring one made the laptop planning path die on
  a device error before it could report the *real* blocker, which is the missing v2 artifact bindings.
* `RunOptions.hpo` raises if set: the runner does not tune, so the flag would otherwise record an
  unsearched arm as searched.

## 5. Not yet done, and what each item needs

| item | state | what it needs |
|---|---|---|
| 14 scientific questions | answered only as "insufficient" | the real runs (§6); Q1/Q2 in particular need ≥2 seeds and ≥8 farms in common |
| Phase 5 (patch level) | 30 cells `Deferred`, reason `patch-v2-unbound` | the owner re-extracting the v2 patch store + binding its hash in an `amendments:` entry; then the cells become `Pending` by themselves |
| I2/I3 with ERA5, W1/W2 covariates, C3 chosen by dev Brier | implemented, unexecuted on real weather | an ERA5/weather sidecar in the bundle (absent from `thesis_colab` today) |
| Figures | not generated | real data; a fixture figure would be a decorative lie, so `report_benchmark.py` exports CSV/JSON only |
| Colab screening + full blocks | commands written, never launched by the agent | owner-run, per AGENTS.md §6 |
| `reports/architecture_benchmark/` content | dirs + README only | first real bundle |

## 6. Handover commands (exact, with expected wall time)

```bash
# laptop, no GPU, ~2 min each — plumbing proof, safe to repeat
python scripts/benchmarks/run_benchmark.py --plan --fixture --smoke --tier screening --block screening --out /tmp/plan
python scripts/benchmarks/run_benchmark.py --execute --yes --fixture --smoke --tier screening --block screening --out /tmp/bench
python scripts/benchmarks/report_benchmark.py --out /tmp/bench
python scripts/benchmarks/bundle_benchmark.py /tmp/bench --audit

# Colab T4 (owner-run; the notebook mirrors these) — screening first, then the full block
python scripts/benchmarks/run_hpo.py --trials 8 --epochs 3 --out reports/architecture_benchmark/hpo
python scripts/benchmarks/run_benchmark.py --execute --yes --tier screening --block screening \
    --data-root /content/thesis_implementation --out /content/bench_screen
python scripts/benchmarks/run_benchmark.py --execute --yes --tier full --block full_benchmark \
    --seeds 13,42,2026 --data-root /content/thesis_implementation \
    --out /content/drive/MyDrive/thesis_colab/bench_full --resume
python scripts/benchmarks/report_benchmark.py --out /content/drive/MyDrive/thesis_colab/bench_full \
    --copy-to /content/drive/MyDrive/thesis_colab/reports
python scripts/benchmarks/bundle_benchmark.py /content/drive/MyDrive/thesis_colab/bench_full \
    --collect --dest /content/drive/MyDrive/thesis_colab/bundles
```

Screening is expected to fit inside one T4 session (`budget.t4_hours = 6`); the full block
(297 pending cells × 3 seeds) is not a single launch — `--resume` plus the per-arm hour estimates
decide how much of it fits, and whatever does not fit stays `Pending` in the registry where the report
can count it.

## 7. Reading rules for whoever picks this up

1. A number with `stamp = SMOKE FIXTURE` is plumbing evidence. Not a result, not a preview of one.
2. Every claim in the final report carries a rung tag; the ceiling for anything measured on dev is L3
   ("within this study"), and L5 (prospective operational validation) is unmet.
3. Coverage is stated before any result: `coverage.md` and §1 of `REPORT.md` are the authority on what
   ran, not the presence of a table.
4. A `Not Applicable` cell is a design statement from the applicability matrix, and a `Deferred` cell
   names the artifact it waits for. Neither is a failure, and neither may be dropped from a total.
5. The committed matrix hash is the contract. If `--verify-only` fails, the fix is a recorded amendment,
   never a re-run that redefines the cells.
