# Tasks — Integrate Real Model Outputs into Platform Data Pipeline

## 1. Checkpoint loading & inference
- [ ] 1.1 Implement `scripts/load_model_checkpoints.py` — discover `.pt`/`.ckpt` checkpoints under `checkpoints/`, load config + normalization stats sidecar files, return model + normalizer or `None` per model ID
- [ ] 1.2 Build per-field cycle input tensor `(phase_bins, bands)` from `stats.zarr` time series (reuse `src/ssl_anomaly/io/` cycle adapter)
- [ ] 1.3 Run inference per field per model; write real anomaly scores into observation `scores` map
- [ ] 1.4 Add per-score `score_provenance` (`"model"`/`"proxy"`/`"absent"`) and `score_formula_id` to each observation

## 2. Exporter integration
- [ ] 2.1 Replace proxy CWSI score block (`export_platform_data.py:193-202`) with checkpoint-backed inference + fallback
- [ ] 2.2 When checkpoints missing, emit proxy scores with `"provenance": "proxy"` (do not crash)
- [ ] 2.3 Log which models loaded as `"model"` vs `"proxy"` at export time
- [ ] 2.4 Re-run exporter; regenerate `platform/src/data/series_khuzestan.json` with real scores

## 3. Contract & parity
- [ ] 3.1 Bump contract version to v3 in `scripts/platform_channels.py` and `platform/src/data/channels.ts`
- [ ] 3.2 Add score provenance registry fields to both channel registries
- [ ] 3.3 Add tests to `tests/unit/test_platform_data_contract.py`: provenance values are `model`|`proxy`|`absent`; when checkpoints present scores are `"model"`, when absent `"proxy"`
- [ ] 3.4 Run `tsc --noEmit` → 0 errors; run pytest → all pass

## 4. Docs & commit
- [ ] 4.1 Add proxy-vs-real score finding to `docs/platform_bug_audit.md`
- [ ] 4.2 Commit all changes on current branch
- [ ] 4.3 Verify platform renders real scores with provenance labels
