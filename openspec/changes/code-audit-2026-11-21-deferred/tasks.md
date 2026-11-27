# Tasks — code-audit-2026-11-21-deferred

## 1. Scope recording

- [x] 1.1 Record the audit deferred list in this change (manual creation —
      `openspec` CLI not installed on this machine; verified via
      `bunx openspec --version` failure in the workflow ledger §13). Evidence:
      `openspec/changes/code-audit-2026-11-21-deferred/{proposal,tasks}.md`
      created 2026-11-27 (WP-9); `todo.md` §"code-audit-2026-11-21-deferred"
      updated to match.

## 2. P1 latent defects — fix before first use (audit 2026-11-21 items 1–3)

> Fixed on 2026-11-22 by adae60d / d7432a9 / b082283; **re-verified 2026-11-27
> (WP-9)**: the pinned tests pass at HEAD (88 passed in
> `tests/unit/test_indicators.py tests/unit/test_contamination.py
> tests/unit/test_patch_model.py`) and each test set FAILS on the pre-fix
> parent commit (defect reproduced in detached temporary worktrees, run with
> `PYTHONPATH=<worktree>/src`).

- [x] 2.1 **Item 1 — `indicator_z(unit="farm_own")` shrinkage no-op**
      (`src/ssl_anomaly/monitor/indicators.py`). Fix adae60d: explicit
      `pooled_reference` argument used for both the κ-shrinkage target and the
      thin-history fallback; missing `pooled_reference` raises (fail-closed).
      Tests: `test_farm_own_kappa_shrinkage_pulls_toward_the_pooled_scale`,
      `test_farm_own_zero_history_falls_back_to_the_pooled_reference`,
      `test_farm_own_unit_requires_the_pooled_reference`,
      `test_farm_own_z_uses_the_fields_own_history`,
      `test_farm_own_falls_back_and_says_so_when_history_is_thin`
      (`tests/unit/test_indicators.py:96-166`). Reproduction on parent
      d7432a9: 5 farm_own tests failed —
      `TypeError: indicator_z() got an unexpected keyword argument
      'pooled_reference'` (the no-op API could not even accept an explicit
      pooled reference).
- [x] 2.2 **Item 2 — `absorption_a` polarity inverted**
      (`src/ssl_anomaly/train/contamination.py:183-208` at audit time). Fix
      d7432a9: formula unchanged, labels flipped — docstring states a≈1 ⇒ the
      output reproduces the injection (absorber), `share_absorbed = a`; E7 rule
      amended to LOWEST `a_patch` (dated amendment in the archived
      patch-contamination-absorption design). Tests:
      `test_a_patch_is_one_when_the_output_follows_the_injection_absorber`,
      `test_a_patch_is_zero_when_the_output_stays_clean_detector`
      (`tests/unit/test_contamination.py:131-143`). Reproduction on parent
      cbeff9f: both polarity tests failed (pre-fix code returned the
      complement).
- [x] 2.3 **Item 3 — GRU `weight_hh` orthogonal init covered rows 0:63 of 192**
      (`src/ssl_anomaly/models/patch_cnn_gru.py:_init_weights`). Fix b082283:
      gate slices use `p.shape[0]//3` so all three gate blocks are covered.
      Test: `test_weight_hh_orthogonal_init_covers_all_rows`
      (`tests/unit/test_patch_model.py:182-200`, gram ≈ I per gate block).
      Reproduction on parent 0486753:
      `AssertionError: gru.weight_hh_l0 gate block 0 is not orthonormal (init
      did not cover it)`. Checkpoint caveat per the audit OR-branch: existing
      checkpoints were trained under the partial init — deviation recorded in
      finding 25 (`docs/findings/25_patch_full_v2_fixed_protocol.md:148-152`),
      no retrain performed or claimed; the first post-fix training run uses
      the corrected init.

## 3. Deferred backlog — recorded, not scheduled (audit items 4–29 P2 + perf 1–10)

- [ ] 3.1 **P2/perf pickup** — fix each remaining item as a small conventional
      commit when its file is next touched, then tick it here with the commit
      hash. The audit doc (`docs/audits/code_audit_2026-11-21.md` §P2/§Performance)
      is the source of truth for the item list; do not re-survey silently.
      **Status note (2026-11-27, WP-9):** the 2026-11-22 sweep commits report
      the following P2 items fixed — f51fd43: 4, 5, 6, 8, 9, 10, 11, 12, 19,
      20, 21, 22, 23, 25, 26, 28, 29; b082283: 15, 16; cbeff9f: 17. WP-9
      spot-checked in code and CONFIRMS: #4 (tautological independence gate
      gone, `farm_local.py:229-266` compares base vs again), #5 (CWSI bad
      cells → NaN, `indicators.py:121`), #6 (+inf → finite sentinel,
      `scoring.py:320`), #9 (mask-shape raise, `farm_local.py:390-394`), #12
      (tie policy documented, `scoring.py:490`), #24 (contamination basis =
      bands_consistent and recorded, `train_patch.py:207,547`), PERF-2/PERF-5
      vectorized forms present with a parity oracle
      (`scoring.py:141,211,246`; not benchmarked by WP-9). Items NOT verified
      and still open per the audit text: **#7, #13, #14, #18, #27** (all
      `scripts/eval_dev.py` / loader paths — the todo.md deferral gate on
      `eval_dev.py` re-runs stays until they land) and **perf 1, 3, 4, 6, 7,
      8, 9, 10**. The workflow ledger's claim that #9/#29 were still pending
      is stale (contradicted by f51fd43 and the code greps above).

## Constraints (binding on pickup)

- One small conventional commit per item, referencing the audit item number.
- `scripts/eval_dev.py` must NOT be re-run before #7/#13/#14/#18/#27 land
  (todo.md deferral gate); landing them changes the eval-subset hash (#13) →
  finding-25 amendment required.
- Item 13's partition fix and item 24's basis alignment are already reflected
  in finding 25's deviation record; keep that record current with any change.
- Nothing here may trigger a retrain or an eval re-run without the owner's
  long-job handover (AGENTS §6).
