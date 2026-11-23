# Change: consistent-spectral-injection

> **Status (2026-11-23): SUPERSEDED.** The band/index-consistency injection fix remains valid code and is retained; the evidence context it audited (finding 24, S1=1.000) is void per finding 25, and the protocol was re-anchored on the v2 chain (eval hash 852e1ba3).
> See `openspec/changes/INDEX.md` (triage 2026-11-23). This change is kept as
> historical record only; it SHALL NOT be applied, archived as complete, or
> cited as current evidence.

## Why

The 8-channel patch input stores NDVI (channel 0) alongside the raw Red (channel 3)
and NIR (channel 4) bands, but the two severity-sweep injection arms break the
band↔index identity:

- `index="ndvi"` shifts only channel 0 and leaves Red/NIR untouched
  (`src/ssl_anomaly/monitor/patch_perturbations.py:255-271`).
- `index="savi"` solves Red/NIR with the sum-preserving solver and leaves the
  stored NDVI channel stale (`patch_perturbations.py:272-303`).

A detector can therefore satisfy the severity gates by flagging band/index
*inconsistency* rather than a spectrally realistic vegetation change, and finding
24's S1-at-1.000 result cannot separate the two explanations. External audit
(2026-11-20) flagged this as the methodological issue with the widest claim
impact; the finding-24 training runs used `contaminate_fraction = 0.0`, so
injections are evaluation-only and the fix re-scores without retraining.

Owner decision (2026-11-20): **band-consistent injection becomes the primary
gate protocol; the legacy behavior is kept as a labeled sensitivity row** so
findings 18/22/23/24 stay reproducible and comparable.

## What Changes

- **`ssl_anomaly.monitor.patch_perturbations`**: new `injection_basis` parameter
  ∈ `{"bands_consistent", "index_only"}` on `inject_patch_drop` and
  `inject_patch_recovery`.
  - `bands_consistent` (primary): the requested index drop is realized through
    the raw Red/NIR bands (sum-preserving solve for both indices) and the stored
    NDVI channel is **recomputed** from the post-injection bands. Meta records
    the basis, the realized per-bin drop of the requested index AND the realized
    side-effect drop of the other index, plus floored-value counts.
  - `index_only` (sensitivity): today's behavior verbatim, regression-pinned by
    a byte-identity test.
- **`ssl_anomaly.monitor.patch_eval`** (`patch_arm_scores`,
  `patch_localization_scores`): basis plumbed through; every severity-cell
  report carries `injection_basis`.
- **`scripts/eval_dev.py`**: `--injection-basis` flag; primary sweeps run
  `bands_consistent` under protocol id `severity_sweep_patch_v2`; `index_only`
  keeps id `severity_sweep_patch_v1`. Provenance records the basis and id.
- Field-level `monitor/perturbations.py` is NOT modified (design D3).

## Impact

- **Affected specs:** `model-development-protocol` (ADDED requirement:
  band-consistent injection basis).
- **Affected code:** `src/ssl_anomaly/monitor/patch_perturbations.py`,
  `src/ssl_anomaly/monitor/patch_eval.py`, `scripts/eval_dev.py`,
  `tests/unit/test_patch_perturbations.py`.
- Eval-side only: no checkpoint, σ-table, store or platform artifact changes.
  Findings 23/24 remain historical records under `index_only`. The first v3
  finding (score-semantics-realignment + v2-store retrain) uses
  `bands_consistent` as its primary protocol.
