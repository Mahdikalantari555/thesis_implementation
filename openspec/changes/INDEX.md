# OpenSpec changes index — triage 2026-11-23, updated 2026-11-24

Maintained by the owner's triage decision of 2026-11-23 (recorded in
`todo.md`): **the v1-era patch line and the GRU-prediction line are
superseded by the v2 fixed-protocol chain** (`cycles_490_v2.parquet` —
3,478 non-censored cycles / 490 fields; `splits_locked_v2.json` sha
`30ecbe10…`; `holdout_manifest_v2.json` sha `d501bf81…`) **and by finding 25**
(`docs/findings/25_patch_full_v2_fixed_protocol.md`), which voided the v1
verdicts (finding 23/24 headlines) and froze the patch direction. Findings
numbers are immutable; superseded folders are kept as historical record only.

**Superseded ≠ completed.** `openspec/changes/archive/` also contains
superseded changes archived as historical records with explicit owner approval;
those changes and their delta specs are not current evidence or requirements.
**Dormant** = not voided, but blocked on an explicit owner decision before any
application.

## Active / proposed

| Change | Tasks | Status | Notes |
|---|---|---|---|
| `regional-ssl-representation-evaluation` | 15/42 | PROPOSED — the capstone | Level-1/Level-2 evaluation of a light regional SSL representation; five-level ladders; pre-registered gates G-A…G-E; owner decisions registered 2026-11-23 (advantage = undecided; priority = both + applied ambition; Madeline path = tracked future change) |
| `benchmark-latent-reconstruction-prediction-architectures` | 0/37 | PROPOSED — heavy counterpart | Architecture/suppression/synthetic breadth benchmark (Colab T4, budget-aware); feeds the capstone, no capstone gate depends on it |
| `cycle-detection-v3-multindex` | 26/27 | ACTIVE — v2 producer; nearly closed | Only open item is an optional follow-up: the gate FAILED, so no switch; production path stays off; owner call whether to re-gate with `--sweep` after the v2 store completes |

## Dormant — unapplied, await owner decision

| Change | Tasks | Notes |
|---|---|---|
| `oracle-floor-audit` | 14/15 | Post-finding-25 pool-V audit (σ-floor verdict); analysis stands, but it restates the voided patch direction; needs owner review before any follow-up |
| `patch-alarm-statistic-e1` | 0/11 | Unapplied child of oracle-floor-audit (channel-max residual aggregation; power 0.30–0.67 vs committed 0.06–0.46); blocked on a patch-revival decision |

## Archive (`openspec/changes/archive/`)

19 changes archived previously (2026-11-07 → 2026-11-15). Archived
2026-11-23 (tasks verified complete; main specs created/updated by
`/opsx:archive`):

| Change | Archived spec (new) |
|---|---|
| `2026-11-23-decomposition-aware-ssl-study` | `decomposition-aware-anomaly-study` (+7 requirements) |
| `2026-11-23-training-holdout-invariant` | `training-holdout` (+2 requirements) |
| `2026-11-23-separate-verification-gates` | `research-verification` (+1 requirement) |

Archived 2026-11-24 with owner approval: gating work completed; the deferred
items remain unchecked in the archived task files.

| Change | Archived spec (new) | Deferred tasks |
|---|---|---|
| `2026-11-24-optional-rust-acceleration` | `optional-rust-acceleration` (+6 requirements) | 3 (9/12 complete) |
| `2026-11-24-eval-artifact-cache` | `eval-artifact-cache` (+5 requirements) | 2 (12/14 complete) |

## Superseded historical changes archived 2026-11-24

At the owner's request, these changes were archived as historical records,
without syncing their superseded delta specs. Their task counts do not imply
that unfinished work was completed; they SHALL NOT be applied or cited as
current evidence. The original proposal, specs, and tasks remain in each archive.

| Change | Tasks | Supersession note |
|---|---|---|
| `2026-11-24-patch-level-controlled-trial` | 17/17 | v1 PoC; verdict MDM80=S1 void per finding 25 |
| `2026-11-24-patch-full-run-and-ablations` | 4/15 | v1 full run; finding 24 void |
| `2026-11-24-patch-data-pipeline` | 9/12 | v1 `patches.zarr` superseded; pipeline code lives on for v2 re-extraction |
| `2026-11-24-patch-generalization-suite` | 2/8 | Never completed; re-specced (temporal rolls / leave-block-out) in the capstone and the benchmark change |
| `2026-11-24-patch-independent-indicators` | 4/7 | Voided patch context; environmental validation re-specced (proxy-only) |
| `2026-11-24-patch-scoring-ensemble` | 2/12 | Voided patch context |
| `2026-11-24-patch-contamination-absorption` | 2/8 | Voided patch context; field-level absorption closed (findings 20/23) |
| `2026-11-24-patch-xai-attribution` | 3/6 | Voided patch context |
| `2026-11-24-farm-local-anomaly-scoring` | 6/14 | Voided patch context |
| `2026-11-24-platform-patch-scorer-wiring` | 0/15 | Blocked on owner approval + a frozen winner; the winner is void (finding 25) |
| `2026-11-24-consistent-spectral-injection` | 9/10 | Code fix retained; evidence context void |
| `2026-11-24-gradual-severity-shape` | 4/7 | Gradual family absorbed into the benchmark change |
| `2026-11-24-anomaly-definition-path-b` | 4/7 | Trigger condition met (both path-C trials failed); absorbed into the benchmark change |
| `2026-11-24-compare-spatial-representation-strategies` | 2/24 | Never completed; void patch premise |
| `2026-11-24-predictive-ssl-controlled-trial` | 6/13 | gru_pred negative (findings 22/23); path C closed; v1 protocol |
| `2026-11-24-score-semantics-realignment` | 8/10 | Residual-semantics fix for the voided patch path; audit trail |
