# Change: code-audit-2026-11-21-deferred

> **Note (2026-11-27):** created manually — the `openspec` CLI is not installed
> on this machine (verified in the workflow ledger §13), so no CLI-generated
> scaffolding exists. This change records the deferred-defect backlog from
> `docs/audits/code_audit_2026-11-21.md` as the durable openspec home that
> `todo.md` §"code-audit-2026-11-21-deferred" already named.

## Why

The post-bd65829 code audit (`docs/audits/code_audit_2026-11-21.md`) found three
result-corrupting bugs (fixed same day: 99709b0 / ea74a0c / d4b3cb8) and a
deferred list of three P1 latent defects, 26 P2 hygiene items and 10 performance
opportunities. None of the deferred items corrupts a current result, but the
audit's rule stands: **each item is fixed as a small change before the code it
touches is next used.** Later work packages build on `monitor/indicators.py`,
`train/contamination.py` and `models/patch_cnn_gru.py`, so the three P1 defects
had to be fixed before that use. The owner disposition recorded in the audit
addendum (2026-11-22) assigned each deferred item to the change that next
touches it, with checkpoint-affecting deviations recorded in finding 25 instead
of an immediate retrain.

## What Changes

- **P1 item 1 — `indicator_z(unit="farm_own")` shrinkage no-op** — the pooled
  scale aliased the farm's own scale, collapsing `w·own + (1−w)·pooled` to
  `own` for every κ, and the thin-history "pooled fallback" z-scored the farm
  against its own 2–4 points. Fixed in adae60d: an explicit `pooled_reference`
  argument (population values or `FieldSigmaTable.pooled`), shrinkage target +
  fallback both use it, `pooled_reference` is mandatory (fail-closed), tests pin
  κ-acting, `n→0 ⇒ pooled`, and the fail-closed ValueError
  (`tests/unit/test_indicators.py:96-166`).
- **P1 item 2 — `absorption_a` polarity inverted** — the formula
  `a = (pred_cont − pred_clean)/(x_cont − x_clean)` measures how much of the
  injection the output follows (a≈1 ⇒ absorber), but docstrings/tests labelled
  a≈1 as detector. Fixed in d7432a9 (labels-only, formula unchanged):
  docstring states the measurement, `share_absorbed = a`
  (`train/contamination.py`), tests flipped to the audit-sanctioned reading
  (`tests/unit/test_contamination.py:131-143`), E7 decision rule amended to
  LOWEST `a_patch` with a dated amendment in the (archived)
  patch-contamination-absorption design. Siblings
  (`monitor/diagnostics.py:141-146`, `scripts/trial_gates.py:24`) already
  measured the same polarity.
- **P1 item 3 — GRU `weight_hh` orthogonal init covered rows 0:63 of 192** —
  the gate slice used `p.shape[1]//3` (= 21–64) instead of `p.shape[0]//3`
  (= 64), leaving ≈67 % of the recurrent kernel at PyTorch default. Fixed in
  b082283 (rows sliced, all 192 covered; `models/patch_cnn_gru.py::_init_weights`),
  pinned by `tests/unit/test_patch_model.py::test_weight_hh_orthogonal_init_covers_all_rows`
  (all three gate blocks orthonormal). Per the audit's OR-branch and the owner
  disposition, existing checkpoints trained under the partial init are covered
  by the deviation record in finding 25
  (`docs/findings/25_patch_full_v2_fixed_protocol.md:148-152`, "Known
  training-path deviations … no retrain was performed"); the first post-fix
  training run uses the corrected init.
- **Deferred backlog recorded, not scheduled:** the 26 P2 hygiene items (audit
  items 4–29) and 10 performance opportunities stay in the audit doc as the
  source of truth; task 3.1 tracks their pickup. A 2026-11-22 hygiene sweep
  (f51fd43, plus b082283 items 15/16 and cbeff9f item 17) reports most P2 items
  fixed; WP-9 spot-checked #4/#5/#6/#9/#12/#24 and PERF-2/PERF-5 in code on
  2026-11-27 (see tasks.md §3 for what is verified vs. cited).

## Impact

- **Affected specs:** none new — `indicator_z`'s `farm_own` contract and the
  `absorption_a` reading live in the archived 2026-11-24 patch changes; this
  change records the audit-sanctioned corrections without re-opening those
  (superseded) specs.
- **Affected code:** `src/ssl_anomaly/monitor/indicators.py`,
  `src/ssl_anomaly/train/contamination.py`,
  `src/ssl_anomaly/models/patch_cnn_gru.py` (+ their tests). No production
  caller exists for `farm_own`/`absorption_a` (verified by grep 2026-11-27), so
  the signature change breaks nothing; the GRU init change affects the next
  training run, whose checkpoints were pre-fix (deviation recorded, finding 25).
- **Validation ladder:** all claims ≤ L1 — code-correctness fixes pinned by
  unit tests; no model-performance claim is made or implied. A retrain that
  benefits from item 3 is a future change's business, per the finding-25 record.
