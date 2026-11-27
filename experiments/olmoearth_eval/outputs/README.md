# outputs/ — artifact inventory

Committed (evidence): `tables/*.json` (machine-readable results),
`figures/*.png` (E1 figures), `raw_bin_features.parquet` (B0/B1 feature
cache), `REPORT.md` at the experiment root.

Regenerable in < 6 min on this machine (gitignored, see REPORT.md §10):
- `embeddings_v1_2_nano{,_smoke,fill05,b11dup,up144,drop1..6,rebin30,rebin60}.parquet`
  + matching `.zarr` (obs-level embeddings + provenance.json)
- `farmcycle_v1_2_nano{...}.parquet` (farm-cycle level means)
- `shards*/`, `manifest*.jsonl`, `progress*.jsonl` (crash-safe resume state)
- `embeddings_v1_2_nano_smoke.*` (the `--smoke` gate artifacts)

Nothing here is written to `Data/Processed/`; this experiment stays isolated
under `experiments/olmoearth_eval/`.
