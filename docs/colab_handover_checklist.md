# Colab Handover Checklist — benchmark-latent-reconstruction-prediction-architectures

Prepared 2026-11-24. Companion change: `openspec/changes/data-fix-and-colab-prep`.
This is a handover document only: the agent does not launch long jobs; the
owner runs the commands below on Colab (T4 GPU).

## 0. Bundle verification (run locally before upload)

Bundles present (verified 2026-11-24):

| Bundle | Size | SHA256 |
|---|---|---|
| `google_drive/thesis_data_full_v2.zip` | 280 MB | `bed90d70ade04af94c8808c70befc85854a319613b18c2730b1fad5f5c7bdec3` |
| `google_drive/patch_full_v2_eval_bundle.zip` | 4.3 MB | `bed5d2cbe369dac06b23989617e2709de1f44c2b6f48dac0c233988e048e32ad` |
| `google_drive/patch_full_v2_eval_bundle_v2.zip` | 4.3 MB | `77377d2450641bdf974626394a0b90371b270e7966fa321776fb6629b022042d` |

Verify with:
```bash
cd D:/RS/Thesis/thesis_implementation
sha256sum google_drive/*.zip
python -X utf8 -c "import zipfile; print(zipfile.ZipFile('google_drive/thesis_data_full_v2.zip').namelist()[:5])"
```

Bundle contents: `thesis_data_full_v2.zip` holds `patches.zarr` (v3, ~12.9k entries),
`splits_locked_v2.json`, `holdout_manifest_v2.json`, `norm_stats_full_v2.json`.
It does NOT hold `cycles_490_v2.parquet`, `weather_daily.parquet`, `stl_features.parquet`,
`patch_sigma.npz`, or any `ground_labels*`.

## 1. Missing prerequisites (Deferred — do not synthesize)

- ERA5/weather sidecar: `weather_daily.parquet` (`artifacts.weather_daily_v2`) absent.
  Owner must upload `google_drive/weather_daily_v2.zip`. Blocks `weather_inputs` cells.
- Ground/confirmed-anomaly labels: none anywhere (owner task 4). L4 claims stay unsupported.
- STL features: `decomposition/stl_features.parquet` optional, unproduced. Reported as `absent`.
- Patch sidecar: `patch_sigma.npz` absent next to `patches.zarr`. Blocks `hierarchical_patch` cells.
- Canonical mirror: v2 artifacts live under `thesis_implementation/data/` and the bundle,
  not `Data/Processed/`. Mirror step is owner/execution; agent verifies only.

Deferred markers: `reports/data_gaps/*.deferred.json` and
`openspec/changes/data-fix-and-colab-prep/deferred/*.deferred.json`.

## 2. Environment (Colab)

- Python 3.10+ (Colab default), install deps:
  `pip install -q numpy scipy pandas scikit-learn pyyaml pyarrow torch`
- GPU: T4 (cuda). The notebook detects device via `ssl_anomaly.benchmarking.environment`.
- Clone repo:
  ```bash
  git clone https://github.com/Mahdikalantari555/thesis_implementation.git /content/thesis_implementation
  cd /content/thesis_implementation
  ```
- Place bundle in Drive at `/content/drive/MyDrive/thesis_colab/thesis_data_full_v2.zip`.

## 3. Notebook §9 — screening smoke (exists and runnable)

Notebook: `notebooks/benchmarks/architecture_comparison_benchmark.ipynb`, section
`## ۹. غربالگری — fixture/smoke: لوله‌بندی، نه نتیجه` (cell 18).

Command (exact, from the notebook):
```bash
python scripts/benchmarks/run_benchmark.py --execute --yes \
  --fixture --smoke --tier screening --block screening \
  --out /content/bench
```
Verified locally 2026-11-24: RC=0, ~84 s wall, 17 Completed / 3 Failed
(failures are "latent collapse" design outcomes, not crashes). No NaN in summary.

## 4. Full benchmark run (owner executes)

```bash
cd /content/thesis_implementation
python scripts/benchmarks/run_benchmark.py --execute --yes \
  --tier full --block <BLOCK> --out /content/drive/MyDrive/thesis_colab/bench_full \
  --data-root /content/data_v2 --resume
```
Expected wall time: up to ~6 T4-hours for the full matrix (see `t4_hours` in
`run_summary.json`). Use `--resume` after any disconnect.

## 5. Outputs

- Screening: `/content/bench` (or `BENCH` in notebook).
- Full: `/content/drive/MyDrive/thesis_colab/bench_full`.
- Each run dir contains `run_manifest.json`, `run_summary.json`, `REPORT.md`,
  `matrix.yaml` (mirrored), `SHA256SUMS` (section 20).
- Evidence pack (CSV/JSON) written under the run dir; copy back to
  `reports/architecture_benchmark/` on the local repo.

## 6. Degradation / blocked cells

- `weather_inputs` cells: Deferred (no `weather_daily.parquet`).
- `hierarchical_patch` (patch-level) cells: Deferred/BLOCKED (no `patch_sigma.npz`).
- Keep these labels; do not silently skip or synthesize values.

## 7. Verification gates (run before declaring done)

```bash
D:/Dev/Miniforge3/envs/geo/python.exe -X utf8 -m pytest tests/unit/test_platform_data_contract.py tests/unit/test_platform_bilingual.py -q
```
Platform (if touched): `node node_modules/typescript/bin/tsc --noEmit` then
`TMP/TEMP -> platform/.tmpbuild node node_modules/vite/bin/vite.js build`.
