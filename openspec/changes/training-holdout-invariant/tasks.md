# Tasks — training-holdout-invariant

## 1. Manifest (TDD-first)

- [x] 1.1 Unit tests: manifest derivable from splits_locked (farm/date set
      equality), sha recording, rebuild refusal without --force
      (`tests/unit/test_holdout_manifest.py`, 15 tests; synthetic split
      manifests so no data archive is needed).
- [x] 1.2 `scripts/build_holdout_manifest.py` →
      `data/processed/holdout_manifest.json` (farms, date start, future
      rules, provenance, sha256); build it once and freeze. **Code + CLI
      verified; the freeze itself is a user run on the machine that holds
      `data/processed/splits_locked.json`** (see "User action still needed").

## 2. Enforcement (TDD-first)

- [x] 2.1 Unit tests: `assert_train_only` catches planted holdout farms/
      post-2024 cycles (named error with offending pairs); passes clean
      input; hash-verified manifest load; provenance recording
      (`tests/unit/test_holdout_enforcement.py`, 22 tests incl. planted
      negative cases, the `+N more` cap, and a static assertion that the
      assertion precedes the first data use in `train_main.py`).
- [x] 2.2 `src/ssl_anomaly/monitor/holdout.py`: loader + assertion helper
      (shared Dataset base so every future trainer inherits it) —
      `build_holdout_manifest`, `save/load_holdout_manifest`,
      `assert_train_only`, `filter_train_only`, `train_only_mask`,
      `HoldoutGuard` (Dataset mixin), `HoldoutViolationError`.
- [x] 2.3 Wire into `train_main.py` Dataset construction (assert before
      first batch; manifest sha into the run manifest) — the assertion runs
      right after `select_splits` and BEFORE `fit_robust_norm`; the manifest
      sha and the holdout-farm count reach `norm_stats` provenance.

## 3. Program wiring

- [x] 3.1 `patch-level-controlled-trial` tasks 1.3/1.4 amended: the patch
      Dataset filters holdout via the shared helper and asserts before
      training (reference this change) — the task text already required it;
      the helper it names now exists and is importable
      (`from ssl_anomaly.monitor import holdout`).
- [x] 3.2 `patch-data-pipeline` proposal note amended: extraction covers all
      fields, training consumers exclude holdout via this change — recorded
      in that change's proposal (already present) and now backed by the
      module.
- [x] 3.3 todo.md Thesis-level reminders: the standing invariant recorded
      (holdout fixed BEFORE implementation, always).

## User action still needed (agent never runs long jobs / touches data)

- [ ] U.1 On the machine holding the data archive, build the frozen manifest
      ONCE and commit it:

      ```bash
      python scripts/build_holdout_manifest.py --dry-run   # inspect first
      python scripts/build_holdout_manifest.py \
          --archive-end 2026-09-10
      ```

      `--archive-end` is the last acquisition date (538-date archive); it
      makes trailing future dates holdout-by-default. Omit it only if you
      prefer not to check dates at all — it is recorded as `null`, never
      guessed.

- [ ] U.2 Then re-run the training smoke to confirm the wiring in situ:

      ```bash
      python scripts/train_main.py --model gru_ae --smoke
      ```

      Expected: one `[holdout] manifest … verified; 0 holdout farms / cycles
      in the training inputs` line before the first batch. A
      `HoldoutViolationError` here means the split assignment and the
      manifest disagree — stop and investigate rather than filtering.

## Notes recorded during implementation

- The derivation deliberately does NOT reuse `split_of_cycle` for enforcement:
  holdout is decided by farm membership + cycle start date, so a farm that is
  unknown at freeze time (future registration) is caught even though it is
  absent from every split list.
- `filter_train_only` is provided for constructing subsets, but the
  enforcement point is `assert_train_only` on the RESULT — silent filtering is
  how leaks hide, so the spec's fail-closed proof is the assertion, not the
  filter.
- `assert_train_only` accepts `cycle_keys` alone (field ids parsed from the
  key) so a patch Dataset that carries only keys is still enforceable.
- `--skip-cycle-file-check` exists solely so `--dry-run` inspection works on a
  machine without the cycle files; it is refused without `--dry-run` and must
  never be used for a real freeze.
