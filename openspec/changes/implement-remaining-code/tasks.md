# Tasks — Implement Remaining Code

## 1. Per-day anomaly
- [ ] 1.1 Ensure compute_daily_anomaly.py uses per-bin z-score (not cycle-level)
- [ ] 1.2 Update make_map_custom.py to use daily_anomaly.parquet
- [ ] 1.3 Verify map colors change per date

## 2. Rust backend
- [ ] 2.1 Implement PyO3 binding in rust/src/lib.rs
- [ ] 2.2 Add maturin build config
- [ ] 2.3 Compile in WSL
- [ ] 2.4 Add parity test (rtol=1e-6)
- [ ] 2.5 Document build steps in rust/README.md

## 3. Ablation
- [ ] 3.1 Create scripts/run_ablation.py (input.drop multirun)
- [ ] 3.2 Run weather on/off, LST on/off
- [ ] 3.3 Report metric change with CI

## 4. E-INTRA
- [ ] 4.1 Create scripts/run_eintra.py (pixel-level harvest detection)
- [ ] 4.2 Run on 32 subset farms
- [ ] 4.3 Report harvest-date scatter within field

## 5. B0/B2/B3 baselines
- [ ] 5.1 Implement B0 (calendar z-score) in baseline/
- [ ] 5.2 Implement B2 (GDD/landmark axis)
- [ ] 5.3 Implement B3 (IsolationForest/ECOD)
- [ ] 5.4 Add comparison script

## 6. GRU-AE (M-A)
- [ ] 6.1 Test existing GRU-AE code
- [ ] 6.2 Add to train_ssl.py as model option
- [ ] 6.3 Compare with Transformer

## 7. Documentation
- [ ] 7.1 Update findings with new results
- [ ] 7.2 Update 00_index.md