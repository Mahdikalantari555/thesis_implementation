# Mini rebuild on 3 separated farms + corrected probe — run record

**Date:** 2026-11-25 · **Git SHA:** `7dfeaa64298a2a520267bcfe5df72b249577c770` ·
**Env:** `/home/mahdi/micromamba/envs/geo` (Python 3.11, numpy 2.4.6, zarr 3.1.6, torch 2.15 CPU) ·
**Layer:** `/home/mahdi/RS/Thesis/Data/RoI/3farms-dev.geojson` (L02-02, R12-08, R15-18 — three blocks, non-adjacent).

> **SUPERSEDED (2026-11-25, change `field-id-provenance-fix`).** The off-by-one
> field ID described in Finding A below was fixed **for the mini store**: the
> builder now uses `archive_fid = reference_fid + 1` with rasterize-verified
> evidence (IoU 1.0 own / 0.0 neighbours for all three farms). The store and all
> downstream artifacts were rebuilt; parity now passes **11,242/11,242** (the
> two "QA mismatch" failures below were another symptom of the shift). The
> numbers below were produced on the **shifted store** and remain void as
> measurements. The canonical 490-field archive is still uncorrected and needs a
> separate owner-gated decision. See the new run record written after the fix.

---

## 1. Bugs fixed before the run

| # | File | Fix |
|---|---|---|
| 1 | `scripts/probe_mini_suspected_stress.py:160` | Broken control (`train, test = perm_years < 2023, years < 2023`, queries from train, R@1≈0.999) replaced by test-year queries + candidate `(farm, stage)` label-permutation null (200 permutations), with empirical chance level and p97.5. Legacy key `shuffled_year_control` now carries the corrected null mean. |
| 2 | `configs/mini/run_smoke.yaml:18`, `src/ssl_anomaly/mini_lab/config.py:36,101`, `scripts/build_mini_phenology.py:144` | Holdout boundary `2025-01-01` → `2024-01-01`; split is now train ≤ 2022 / dev 2023 / test 2024+ per accepted plan §9. Dev excluded from retrieval and probes. |
| 3 | `scripts/probe_mini_suspected_stress.py` | NaN→0 removed: values keep NaN, an explicit `observed` mask is a model input; standardization uses train-only observed values. |
| 4 | `scripts/probe_mini_suspected_stress.py` | Patches sampled on the global 5×5 label grid (phase offset applied), spread over the whole field instead of `coords[:80]` top rows. |
| 5 | `scripts/probe_mini_suspected_stress.py` | Windows crossing a 2023/2024 boundary dropped (24 dropped); year from window, not first date. |
| 6 | `scripts/probe_mini_suspected_stress.py` | Logistic probe changed from 200 un-converged GD steps to `sklearn.LogisticRegression(lbfgs, max_iter=3000)`, with farm×year cluster-bootstrap 95 % CI. |
| 7 | `src/ssl_anomaly/utils/paths.py` (new) | Windows archive defaults `D:/RS/Thesis/...` resolve to the Linux mount; no-op on Windows, overridable by `THESIS_ROOT`. Applied to all mini builders and the mini-lab config. |

**Environment note:** `optuna` was missing from the `geo` env and was installed
(`optuna==5.1.0`, `colorlog==6.12.0`) so the runner's smoke path could execute.

## 2. Commands run (all with `PYTHONPATH=src`, `python -X utf8`)

```
scripts/build_mini_dataset.py --farms-geojson <3farms-dev.geojson> --check-parity
scripts/build_mini_weather.py
scripts/build_mini_phenology.py
scripts/build_mini_validation.py
scripts/label_mini_suspected_stress.py
scripts/run_mini_experiment.py --smoke --config configs/mini/run_smoke.yaml
scripts/probe_mini_suspected_stress.py
```

The previous store was moved aside to
`Data/Processed/.mini_research_dataset_old_2026-11-25` (the builder refuses mixed farm sets).

## 3. Finding A — off-by-one field IDs (highest priority)

`archive_alignment` IoU of archive pixels vs reference polygons:

| Layer farm | Archive `ogc_fid` | Best-matching fid | Farm of best match | IoU (best / own) |
|---|---|---|---|---|
| L02-02 | 300 | **299** | R10-31 | 1.0000 / 0.0000 |
| R12-08 | 245 | **244** | R12-06 | 1.0000 / 0.0000 |
| R15-18 | 17 | **16** | R15-20 | 1.0000 / 0.0000 |

The archive pixels for field id *k* are the polygon of id *k−1*. This is an **ID
off-by-one, not a georeferencing shift** (the polygon shapes match perfectly one
id away). The three selected physical fields are R10-31, R12-06 and R15-20 —
three different blocks, so the non-adjacency requirement still holds.

Consequences (not fixed here; the builder deliberately preserves parity with the
frozen archive):

- Farm names, variety, age, management/office/block attributes in the 490-field
  archive are shifted by one relative to the pixels.
- The suspected-stress crop kind (`plant`/`ratoon` from `Age`) built from layer
  attributes is applied to the neighbouring field's pixels in this store.
- Any L4 linkage to company records (harvest, yield, ECe) is invalid until the
  mapping is corrected; all variety/age strata in prior findings are suspect.
- Requires an owner decision and a separate provenance-fixing change — do not
  silently re-index the canonical archive.

## 4. Finding B — parity (2 of 11,256 checks failed)

`reports/mini_dataset/parity.json`: 11,254 pass, 2 fail, both **L02-02 / 2020-09-01**:
`swir1` stored 0.357788 vs zonal 0.280672 (abs 0.077116), `swir2` 0.308960 vs
0.218523 (abs 0.090437). Investigation: the zonal row used `n_pixels = 129`
while the store has 84 quality-masked valid pixels that date; all seven bands are
shifted the same way (only SWIR exceeds the tolerance). Pre-existing canonical
zonal/pixel QA-mask mismatch, reported verbatim, not silently ignored.

## 5. Smoke experiment (harness plumbing only — not evidence)

24 units, 26 features, `gru_ae`, 1 seed, 2 epochs, CPU, 6.25 s, boundary 2024-01-01.
`synthetic_auprc 0.5555` vs `random_init_auprc 0.5540` vs `step_reference_auprc
0.5752`. Single seed, two epochs: no performance claim. Report:
`Data/Processed/mini_research_dataset/experiments/report.json`.

## 6. Corrected representation probe

`reports/mini_dataset/representation_first_pilot.json` (overwrote the void
`d60963e` report). 384 windows, 20 features, 126 suspected (prevalence 0.3281).
Split: train 240 / dev 24 / test 120; 24 boundary-crossing windows dropped.

| Measure | SSL | Random init | Raw features | Chance / null |
|---|---|---|---|---|
| R@1 (farm+stage, n=120) | 0.2083 | 0.1583 | 0.1500 | chance 0.1275; permutation null mean 0.1317, p97.5 **0.2917** |
| AUPRC, test ≥2024 (n=120, 39 positives, 6 clusters) | 0.4038, CI [0.3242, 0.5992] | 0.3487, CI [0.2935, 0.4222] | 0.1886, CI [0.1363, 0.2340] | prevalence 0.325 |

Collapse diagnostics: effective rank **1.31** (participation ratio), mean pairwise
cosine 0.0508, per-dim std 0.044–0.221 → near-collapse; an anti-collapse arm is
needed before any representation claim.

**Interpretation:** SSL R@1 (0.2083) is above chance (0.1275) but **inside the
permutation null band** (p97.5 0.2917) → not significant. SSL AUPRC exceeds the
raw-feature AUPRC, but its CI overlaps the random-init control's CI → the
architecture itself is not shown to help. The old report's "SSL gain +0.0377"
and "AUPRC SSL 0.327 < raw 0.498" are both void. Rung: **L1/L2 screening**, and
provisional because of Finding A.

## 7. Gates

- `pytest tests/unit/test_platform_data_contract.py tests/unit/test_platform_bilingual.py -q` → **51 passed**.
- Mini unit suite (`test_mini_representation_probe.py`, `test_mini_farm_source.py`, `test_mini_lab.py`) → **16 passed**.
- Builder smoke (5 dates, 3 farms) inspected before the full run; full run 53.27 s.
- Probe run 7.02 s; all metrics finite; no NaN in reported fields.

## 8. Next actions (owner-gated)

1. Decide how to correct the 490-field ID mapping (re-index pixels, or re-label
   polygons) — this blocks all attribute/variety/L4 work and creates a new
   openspec change; do not patch it silently.
2. Decide the 2020-09-01 QA-mask parity mismatch (accept with a documented
   exception, or fix the canonical zonal stats).
3. Only then: add an anti-collapse candidate and repeat the patch-size sweep and
   L1-a…L1-e battery from `docs/mini_lab_regional_eval_plan.md` §8.
