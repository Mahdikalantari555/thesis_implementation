# `reports/architecture_benchmark/` — where the benchmark's evidence lives

**Status: no result artefacts here yet.** The directories are committed so the shape of an
evidence pack is fixed *before* any number exists to fill it — and so that an empty
`figures/` is a visible fact rather than a missing folder someone assumes was never planned.

| directory | written by | contents |
|---|---|---|
| `runs/` | `scripts/benchmarks/run_benchmark.py --out …` | one subdirectory per cell: `run_manifest.json`, `meta.json`, `last.pt`, `progress.jsonl`, `norm_stats.json`, `metrics.json`, `bin_scores.npz`, plus `collapse.json` / `feasibility.json` / `classical_fit.json` where the cell shape has them |
| `runs/matrix.yaml` | mirrored by the runner | the exact matrix the run bound to (a bundle must be readable without the repo) |
| `runs/registry.json`, `registry.log.jsonl` | runner | every cell's final status + the append-only transition log |
| `runs/coverage.md` | runner | what ran, what did not, and why — the authority on coverage |
| `runs/REPORT.md`, `runs/report.json` | `report_benchmark.py` | the three winner tables, secondary metrics, variance decomposition, cost, and the 14-question ledger |
| `manifests/`, `metrics/`, `statistical/` | `report_benchmark.py --copy-to` | copies for citation: per-cell manifests, metric tables, `statistical.json` (bootstrap CIs, Wilcoxon, Friedman/Nemenyi, Holm, Cliff's δ, variance shares) |
| `findings/` | hand-written, from the above | the finding document(s); every claim tagged with its validation-ladder rung |
| `hpo/` | `run_hpo.py` | per-arm search traces + `hpo_selection.json` (a selection *layer*, never an edit to the frozen contract) |
| `figures/` | not generated | deliberately empty until real data exists: a figure drawn from a fixture run is a decorative lie |

## Bundle rules (AGENTS.md §6.2)

A transfer is **complete or a defect**. `bundle_benchmark.py --collect` audits every cell and
*refuses* (exit 2) if a required file or manifest field is missing, then writes
`SHA256SUMS.txt` + `BUNDLE_MANIFEST.json` into a temp directory and moves it into place, so a crash
mid-write cannot leave a half bundle with a plausible name. Verify before use:

```bash
cd <bundle> && sha256sum -c SHA256SUMS.txt
python scripts/benchmarks/bundle_benchmark.py <bundle> --audit   # cell-level completeness
```

`LATEST_BUNDLE.md` inside each bundle names the bundle, its source run, and the verification
commands. There is no `LATEST_BUNDLE.md` here yet because no bundle has been published yet — the
first one lands after the Colab screening run.

## Reading rules

1. Any manifest with `stamp = "SMOKE FIXTURE — plumbing evidence only, not a scientific result"` is
   plumbing evidence. Not a result and not a preview of one.
2. A cell missing from `coverage.md` was not run. Do not describe the set as complete.
3. `Not Applicable` is a design statement from the applicability matrix; `Deferred` names the artifact
   it waits for. Neither is a failure, and neither may be dropped from a total.
4. The rung ceiling for anything measured on dev is **L3** (“within this study”); L5
   (prospective operational validation) is unmet, so no sentence here may read as field-validated.
