# Tasks — training-holdout-invariant

## 1. Manifest (TDD-first)

- [ ] 1.1 Unit tests: manifest derivable from splits_locked (farm/date set
      equality), sha recording, rebuild refusal without --force.
- [ ] 1.2 `scripts/build_holdout_manifest.py` →
      `data/processed/holdout_manifest.json` (farms, date start, future
      rules, provenance, sha256); build it once and freeze.

## 2. Enforcement (TDD-first)

- [ ] 2.1 Unit tests: `assert_train_only` catches planted holdout farms/
      post-2024 cycles (named error with offending pairs); passes clean
      input; hash-verified manifest load; provenance recording.
- [ ] 2.2 `src/ssl_anomaly/monitor/holdout.py`: loader + assertion helper
      (shared Dataset base so every future trainer inherits it).
- [ ] 2.3 Wire into `train_main.py` Dataset construction (assert before
      first batch; manifest sha into the run manifest).

## 3. Program wiring

- [ ] 3.1 `patch-level-controlled-trial` tasks 1.3/1.4 amended: the patch
      Dataset filters holdout via the shared helper and asserts before
      training (reference this change).
- [ ] 3.2 `patch-data-pipeline` proposal note amended: extraction covers all
      fields, training consumers exclude holdout via this change.
- [ ] 3.3 todo.md Thesis-level reminders: the standing invariant recorded
      (holdout fixed BEFORE implementation, always).
