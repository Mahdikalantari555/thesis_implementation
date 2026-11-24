# Tasks: platform-patch-scorer-wiring

**BLOCKED until BOTH: (a) the owner approves applying this change, and
(b) the winner is frozen** (G4 verdict + full-archive MDM80 finite, D2 in
design.md). Nothing below starts before those.

## 1. Registry + contract

- [ ] 1.1 TDD: extend `tests/unit/test_platform_data_contract.py` parity
      test for the new slot (`patch_*_main`, provenance `model`, stage
      `trial`) on both mirrors; watch it fail.
- [ ] 1.2 Add the slot to `scripts/platform_channels.py` and
      `platform/src/data/channels.ts` (order after `time2agri`; name fixed
      at freeze time); green the parity tests.
- [ ] 1.3 Payload-schema contract tests: new per-obs score array present
      when the artifacts exist; all-`absent` when they do not; v5 payload
      files still load with the slot absent (degradation path asserted).

## 2. Exporter patch-scoring path

- [ ] 2.1 TDD: unit tests for the artifact loader (checkpoint + norm + σ +
      store hash binding, fail-closed mismatch abort, missing-artifact →
      all-absent warning path).
- [ ] 2.2 TDD: unit tests for the bin→date attribution rule (D1: half-open
      grid, epoch from locked splits, multi-acquisition dates share the bin,
      out-of-cycle dates absent) on a synthetic store + cycles table.
- [ ] 2.3 Implement the exporter patch-scoring path reusing
      `PatchDeepScorer` + `PatchArmScorer` unchanged (calibration paths from
      the store's train+calibration cycles ONLY — dev/test cycles are never
      calibration input; assert via the existing holdout machinery).
- [ ] 2.4 Wire the payload arrays + provenance RLE + registry meta
      (`stage: "trial"`, artifact hashes) into `export_platform_data.py`;
      existing slots untouched (byte-identity assertion test on a fixture
      export).

## 3. Platform UI

- [ ] 3.1 TDD: bilingual tests for the trial badge strings (EN + FA through
      the `persian-writing` skill quality bar).
- [ ] 3.2 Render the badge wherever the slot appears; no pixel-level UI.

## 4. Gates + evidence

- [ ] 4.1 pytest contract+bilingual (51+ tests) green; `tsc --noEmit`
      clean; vite build green (TMP/TEMP redirected to `platform/.tmpbuild`).
- [ ] 4.2 Runtime screenshot gate (serve_tiles + capture script) with the
      new slot visible and zero console/page errors; evidence committed
      under `reports/screenshots/audit-*`.
- [ ] 4.3 `--desk-only` re-run byte-identical (md5) BEFORE the full export;
      then the full export (~33 min + patch inference) as an owner-approved
      long run.

## 5. Docs + close-out

- [ ] 5.1 `docs/platform_score_methods.md`: move patch arms from §7 item 9
      (unwired) to the wired-slots section with the exact formula id and
      file:line cites; note the trial stage and the flag-source decision.
- [ ] 5.2 Update `docs/findings/23_patch_level_poc.md` + the G4/full-run
      findings with the wiring record (slot name, checkpoint sha, payload
      hash).
- [ ] 5.3 Owner-gated (separate decision, may become its own change):
      switch `stressFlagModel` to the patch checkpoint after the owner
      reviews the platform; until then gru_ae flags stand.
