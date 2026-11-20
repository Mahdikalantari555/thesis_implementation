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

Training data is hosted on Google Drive under the `thesis_colab/` folder
(`thesis_data_full.zip`, ≈68 MB). Mount Drive in the notebook and read from
`/content/drive/MyDrive/thesis_colab/`. The raw METRIC archive (≈44 GB under
`Data/Metric/`) stays local and only feeds the extraction pipeline that
produces the bundle. Checkpoints are gitignored; only reports and findings are
committed.

## Output naming

Every notebook run must write its outputs with a date suffix so it is
unambiguous which run produced them, e.g.
`report_2026-11-20.json`, `checkpoint_2026-11-20.pt`,
`findings_2026-11-20.md`. Include the git SHA and data-bundle hash alongside
the date in the final cell.

## Conventions

- One notebook per run family; name with date suffix (e.g.
  `train_full_archive_colab.ipynb`).
- Record git SHA, data bundle hash, seeds, device and wall time in the final
  cell and in the committed report.
