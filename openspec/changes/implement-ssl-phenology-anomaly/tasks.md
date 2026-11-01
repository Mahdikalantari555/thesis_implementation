# Tasks

## 1. Project scaffolding

- [ ] 1.1 Create Python package structure under `src/ssl_anomaly/`
- [ ] 1.2 Add `pyproject.toml`, lockfile, and base Hydra configuration
- [ ] 1.3 Configure pytest, ruff, and pyright
- [ ] 1.4 Add DVC pipeline skeleton

## 2. Data ingest

- [ ] 2.1 Implement field mask construction
- [ ] 2.2 Implement per-scene zonal statistics with quality masks
- [ ] 2.3 Implement SQLite manifest checkpoint and resume
- [ ] 2.4 Implement resource throttling utilities
- [ ] 2.5 Implement Zarr aggregation into `fields.zarr`
- [ ] 2.6 Add CLI script and tests

## 3. Phenology and cycles

- [ ] 3.1 Implement Whittaker-Eilers smoothing
- [ ] 3.2 Implement harvest detection and cycle segmentation
- [ ] 3.3 Implement replant detection and ratoon tracking
- [ ] 3.4 Implement phase axes and binning
- [ ] 3.5 Add synthetic unit tests and golden regression tests

## 4. Anomaly baseline

- [ ] 4.1 Implement stratum assignment and fallback
- [ ] 4.2 Implement robust baseline statistics
- [ ] 4.3 Implement iterative normal-cycle trimming
- [ ] 4.4 Implement anomaly scoring and runs
- [ ] 4.5 Add ablation support for LST and weather

## 5. Self-supervised models

- [ ] 5.1 Implement cycle tokenization and data loading
- [ ] 5.2 Implement GRU autoencoder
- [ ] 5.3 Implement masked Transformer
- [ ] 5.4 Implement trimmed loss and masking strategies
- [ ] 5.5 Implement reconstruction and embedding anomaly scores
- [ ] 5.6 Add training, checkpointing, and evaluation hooks

## 6. Evaluation

- [ ] 6.1 Implement synthetic anomaly injection
- [ ] 6.2 Implement proxy-based validation
- [ ] 6.3 Implement stability tests
- [ ] 6.4 Implement statistical comparison utilities
- [ ] 6.5 Add MLflow tracking integration

## 7. Export and interfaces

- [ ] 7.1 Implement SafeTensors export and loading
- [ ] 7.2 Save config and normalization statistics
- [ ] 7.3 Define phenology backend interface
- [ ] 7.4 Add model versioning metadata

## 8. Documentation and tests

- [ ] 8.1 Document setup and usage
- [ ] 8.2 Add golden regression fixtures
- [ ] 8.3 Add parity test placeholders for Rust
- [ ] 8.4 Validate OpenSpec change
