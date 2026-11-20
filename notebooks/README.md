# Notebooks

Colab notebooks for the thesis. Each notebook is self-contained and assumes the
data bundle `thesis_data_full.zip` (≈68 MB) is available in `../colab/` or
uploaded to the Colab runtime.

## Placement policy

- **Colab** = full experiment pipeline + evaluation (training, σ computation,
  MDM80 severity, event FAR, report generation). GPU runtime. Reproducible
  end-to-end on the same bundle.
- **Arena / local** = code-only work that does not need the data bundle
  (openspec changes, tests, platform wiring, XAI, bug fixes). Heavy
  data-dependent steps are deferred to Colab.

## Data

Training data lives in `../colab/thesis_data_full.zip` (gitignored). The raw
METRIC archive (≈44 GB under `Data/Metric/`) stays local and only feeds the
extraction pipeline that produces the bundle. Checkpoints are gitignored; only
reports and findings are committed.

## Conventions

- One notebook per run family; name with date suffix (e.g.
  `train_full_archive_colab.ipynb`).
- Record git SHA, data bundle hash, seeds, device and wall time in the final
  cell and in the committed report.
