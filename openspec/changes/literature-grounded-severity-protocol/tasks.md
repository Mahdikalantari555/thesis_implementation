# Tasks — literature-grounded-severity-protocol

## 1. Severity table as verified config

- [ ] 1.1 `src/ssl_anomaly/monitor/severity_table.py`: the S1–S6 ladder as a frozen dataclass/dict
      with `source` (citation+DOI), `verification` ∈ {verified_full_text, verified_abstract,
      design_choice, rejected}, `confidence`, `delta_ndvi`, `duration_bins`, `persistent` flag; hash
      function for the sidecar. Unit test: hash stability + `rejected` levels refuse to load.
- [ ] 1.2 Extend `perturbations.py` with the index-drop family: absolute-Δ injection on ndvi/savi
      (separate axes), persistent variant (4–6 bins), per-field % of own grand-growth median computed
      at injection time, physical-range floor, never erasing observations. Unit tests: onset = ground
      truth; masks unchanged; ranges respected; SAVI axis independent.
- [ ] 1.3 Retire MAD-scaled D6 magnitudes from the headline path (keep the code path for the
      finding-16 continuity report, flagged `legacy`); `lst_rise` stays runnable for diagnostics but
      is excluded from the primary metric per the owner decision (2026-11-14).

## 2. MDM sweep + event-level FAR

- [ ] 2.1 `scoring.py`: event declustering (gap ≤ 2 bins → one event) + `event_false_alarm_rate`
      (events hitting no injection window / time span; extra alarms inside a hit window not
      double-counted). Unit tests with synthetic alarm paths (adjacent bins, gap-2, gap-3).
- [ ] 2.2 `eval_dev.py`: severity sweep S1–S6 × {acute, persistent} × {ndvi, savi} → per-level
      detection rates (Wilson CI) + `mdm80` per family per index; independent-eval-cycle subset
      (disjoint from calibration subset when dev size permits) with recorded hash; B1 identical sweep.
- [ ] 2.3 `metrics.json` schema extension (mdm80, event_far, severity_table_hash) +
      `test_dev_log_schema.py` update; dev_log rows carry the sweep hash.
- [ ] 2.4 Rerun the dev matrix for the three frozen architectures + B1 (CPU, ≤15 min/seed budget
      unchanged; agent runs smoke only — full run handed to the user per AGENTS §7).

## 3. Freeze/final-test integration (gated)

- [ ] 3.1 `freeze_main_model.py`: sidecar gains `severity_table_sha256` + `mdm80` fields; the 5 %
      cycle-level calibration rule and τ/k semantics unchanged.
- [ ] 3.2 `--final-test`: one locked test run of the sweep on the test split (same isolation rules);
      bootstrap CIs over fields for MDM80 and per-level detection; report event-level FAR on test.
- [ ] 3.3 Finding 18 (`docs/findings/18_severity_reanchor_results.md`) with verified-source table,
      MDM curves, honest statement of which levels remain undetectable; finding 16 cross-ref.

## 4. Gates

- [ ] 4.1 Unit tests for perturbations/scoring/severity table (TDD-first, per repo convention).
- [ ] 4.2 Platform untouched → contract/bilingual gates not required; if any payload field changes,
      run the §4 gates before done.
