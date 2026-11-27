# Tasks — cycle-detection-vi-repair

TDD-first. Agent writes code, unit tests and smoke checks; owner runs any long
rebuild and performs the visual review. No stage-conditioned claim before review.

## 1. P0 — Detector

- [x] 1.1 Replace `multi_band_harvest_score` with `vi_harvest_score(ndvi, savi)` in `src/ssl_anomaly/phenology/mini.py`; unit-test orientations, NaN fail-closed and absence of LST/GDD terms.
- [x] 1.2 Extend `detect_harvests` to return strong boundaries plus a `weak_candidates` audit list; enforce the 120-day gap in **real days** (index-distance suppression was the cause of the missed R15-18 Feb-2019 peak).
- [x] 1.3 Replace `phase_for_date` with S1–S4 VI-progress states; unit-test all four boundaries. Note: `growth_state_at_midpoint` is necessarily S2 (midpoint progress 0.5); per-date state assignment remains a separate table if needed.
- [x] 1.4 `build_cycle_rows`: remove `kind`, add `status` (`normal` / `overlong_unresolved` / `short_unresolved`), retain over-long and short bounded intervals with evidence.
- [x] 1.4b Greenup start only after a confirmed bare-soil plateau (0.02 tolerance, ≥3 acquisitions; rise 0.02 sustained ≥2), with `rest_days` and `cycle_start_source`; otherwise fall back to harvest; unit-tested (9 tests in this file).
- [x] 1.5 Update `scripts/build_mini_phenology.py` to the new API and write `weak_candidates.parquet`; `mini_lab/data.py` and `label_mini_suspected_stress.py` filter on `status == "normal"`.

## 2. P1 — Verification and review

- [x] 2.1 Unit tests: 34 passed across mini phenology, cycles, cycle audit and mini lab. Smoke rebuild: 0.94 s, 39 cycle rows (33 normal, 4 over-long, 1 short), no NaN in length/status, no `kind`/GDD columns.
- [x] 2.2 Review artifacts regenerated with the corrected bare-soil rule, distinct S1–S4 colours/labels/legend and gray rest spans: `/tmp/opencode/vi_review4/mini/{L02-02,R12-08,R15-18}.png` and `/tmp/opencode/vi_review4/fields/field_*.png` + `review_summary.json` (local, not committed).
- [ ] 2.3 Owner visual review of those artifacts; cycle-gap block closes only if the verdict is positive, otherwise Path B stays blocked.
- [ ] 2.4 Add the owner verdict to `docs/audits/cycle_gap_diagnosis_2026-11-26.md` and decide whether to rebuild the canonical mini phenology output.

## 3. Workflow 2026-11-27 addendum (WP-1 package; scope extension recorded before implementation)

- [x] 3.1 `/tmp/opencode/vi_review4/` was lost with the tmp directory; regenerated 2026-11-27 via
      `scripts/visualize_vi_cycle_review.py` (new `--skip-archive-fields` so the documented `mini/`
      vs `fields/` layout regenerates exactly): `mini/{L02-02,R12-08,R15-18}.png` (detection re-run
      on the fly; 36 rows = 32 normal / 3 overlong / 1 short), `fields/field_{012,087,164,241,318,455}.png`,
      `review_summary.json` per subdir + a mechanical unedited merge at the top. Dated addendum §7 in
      `docs/audits/cycle_gap_diagnosis_2026-11-26.md`; **owner verdict PENDING** (2.3/2.4 stay open).
- [x] 3.2 Retention flags surfaced in every mini cycle-table consumer, never silently deleted:
      `split_normal_cycles` in `src/ssl_anomaly/phenology/mini.py` (fail closed on the pre-flag
      schema); `mini_lab/data.py` seasonal/variable selection + run-manifest surfacing;
      `label_mini_suspected_stress.py` stage assignment; `visualize_mini_timeline.py` status
      accounting/README note. Commit b9d5583; 8 new tests in `tests/unit/test_cycle_status_consumers.py`
      (48 mini-chain unit tests green).
- [x] 3.3 E0 cycle-completeness gate (research, folded in from the roadmap G0 + continual-hybrid task 3.1):
      `docs/audits/e0_cycle_completeness_2026-11-27.md` — v1 1,755/478 (3.672 cyc/field, 12 zero-cycle,
      silent 180–540 d skip) vs v2 4,458/490 (3,478 non-censored, 9.098 cyc/field, documented reason);
      alias `archive_fid = reference_fid + 1` verified IoU 1.0 on ALL 490 fields, fail closed.
      Verdict: v2 GO / v1 NO-GO for the completeness arm; G0 NaN/harmonization arms not covered here.
      Commit 9816aaa.
- [ ] 3.4 PARKED (owner-gated): literature-stage replacement of the VI-progress states (tillering →
      grand growth → maturity/ripening; Joshi 2024 SOS/EOS via BISE→FFT→DoG; FAO two-crop model) with
      the current states kept as a pre-registered comparator, then architecture/target sweep rerun
      with phase-aware injection. Pre-registered plan:
      `.scratch/workflow_2026-11-27/entries/WP-1-report.md` §Pre-registered plan.
