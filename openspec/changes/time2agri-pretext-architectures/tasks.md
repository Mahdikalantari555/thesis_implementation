# Tasks — time2agri-pretext-architectures

## Phase 1 — faithful models
- [x] 1.1 `models/time2agri_archs.py`: `TinyViT1DEncoder` + TD/FP/FF arms with
      `pretext_loss` and `objective_spec` (10,260 / 11,334 / 10,970 params).
- [x] 1.2 Unit tests: shapes, distinct objectives, FP leakage guard, no NaN
      (7 tests in `tests/unit/test_time2agri_archs.py`).
- [x] 1.3 Renamed legacy arm to `agri_view_contrastive` with deprecated `time2agri`
      alias (old manifests/checkpoints still resolve).

## Phase 2 — mini integration
- [x] 2.1 `mini_lab/models.py`: registered the three arms with `family="pretext"`.
- [x] 2.2 `mini_lab/runner.py`: uses `model.pretext_loss` when family is `pretext`
      (including the Optuna probe path); embedding-distance scoring and random-init
      reference enabled for pretext arms; FP targets built from train-period NDVI.
- [x] 2.3 Short mini smoke (3 × ~4 s, CPU, seed 13, 2 epochs) run and output read:
      `/tmp/opencode/t2a_smoke/{time2agri_td,time2agri_fp,time2agri_ff}/report.json`,
      all tagged `claim_rung=L1`. Losses: TD 1.431→1.408, FP 0.0544→0.0332,
      FF 1.052→1.050; no NaN in loss, finite params. **No anomaly claim:** the
      2-epoch / 8-training-unit smoke AUPRC (TD 0.533, FP 0.551, FF 0.580 vs
      random-init 0.608 for FF) is not evidence at this scale.

## Phase 3 — deferred (not absent)
- [ ] 3.1 Benchmark registry arms A11–A13 behind `BaseArm` with `score_bins`.
- [ ] 3.2 2D patch/ViT path from the date-centric raw store.
