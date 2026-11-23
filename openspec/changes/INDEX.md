# OpenSpec changes index — triage 2026-11-23

Maintained by the owner's triage decision of 2026-11-23 (recorded in
`todo.md`): **the v1-era patch line and the GRU-prediction line are
superseded by the v2 fixed-protocol chain** (`cycles_490_v2.parquet` —
3,478 non-censored cycles / 490 fields; `splits_locked_v2.json` sha
`30ecbe10…`; `holdout_manifest_v2.json` sha `d501bf81…`) **and by finding 25**
(`docs/findings/25_patch_full_v2_fixed_protocol.md`), which voided the v1
verdicts (finding 23/24 headlines) and froze the patch direction. Findings
numbers are immutable; superseded folders are kept as historical record only.

**Superseded ≠ completed.** `openspec/changes/archive/` is reserved for
genuinely completed changes (via `/opsx:archive`); nothing superseded is
archived here, so the archive ledger stays honest. **Dormant** = not voided,
but blocked on an explicit owner decision before any application.

## Active / proposed

| Change | Tasks | Status | Notes |
|---|---|---|---|
| `regional-ssl-representation-evaluation` | 0/42 | PROPOSED — the capstone | Level-1/Level-2 evaluation of a light regional SSL representation; five-level ladders; pre-registered gates; this implementation's anchor change |
| `benchmark-latent-reconstruction-prediction-architectures` | 0/37 | PROPOSED — heavy counterpart | Architecture/suppression/synthetic breadth benchmark (Colab T4, budget-aware); feeds the capstone, no capstone gate depends on it |
| `cycle-detection-v3-multindex` | 26/27 | ACTIVE — live, v2 producer | Fused multi-index detector; produced `cycles_490_v2.parquet` (4,458 windows / 3,478 clean) |
| `oracle-floor-audit` | 14/15 | DORMANT — analysis stands | Post-finding-25 pool-V audit; restates the (void) patch direction; owner review required before follow-up |
| `patch-alarm-statistic-e1` | 0/11 | DORMANT — unapplied | Child of oracle-floor-audit (channel-max residual aggregation; power 0.30–0.67 vs 0.06–0.46 committed); blocked on patch-revival decision |

## Complete — `/opsx:archive` candidates (touch nothing until owner approves)

| Change | Tasks | Notes |
|---|---|---|
| `decomposition-aware-ssl-study` | 8/8 | Completed; its features feed the capstone B5 rung / C-decomp candidate |
| `training-holdout-invariant` | 13/13 | Standing invariant; remains binding for all future changes |
| `separate-verification-gates` | 2/2 | Checkout-vs-real-data verification selections; live practice |

## Superseded (historical record; SHALL NOT be applied or cited as current)

| Change | Tasks | Supersession note |
|---|---|---|
| `patch-level-controlled-trial` | 17/17 | v1 PoC; verdict MDM80=S1 void per finding 25 |
| `patch-full-run-and-ablations` | 4/15 | v1 full run; finding 24 void |
| `patch-data-pipeline` | 9/12 | v1 `patches.zarr` superseded; pipeline code lives on for v2 re-extraction |
| `patch-generalization-suite` | 2/8 | Never completed; re-specced (temporal rolls / leave-block-out) in the capstone and the benchmark change |
| `patch-independent-indicators` | 4/7 | Voided patch context; environmental validation re-specced (proxy-only) |
| `patch-scoring-ensemble` | 2/12 | Voided patch context |
| `patch-contamination-absorption` | 2/8 | Voided patch context; field-level absorption closed (findings 20/23) |
| `patch-xai-attribution` | 3/6 | Voided patch context |
| `farm-local-anomaly-scoring` | 6/14 | Voided patch context |
| `platform-patch-scorer-wiring` | 0/15 | Blocked on owner approval + a frozen winner; the winner is void (finding 25) |
| `consistent-spectral-injection` | 9/10 | Code fix retained; evidence context void |
| `gradual-severity-shape` | 4/7 | Gradual family absorbed into the benchmark change |
| `anomaly-definition-path-b` | 4/7 | Trigger condition met (both path-C trials failed); absorbed into the benchmark change |
| `compare-spatial-representation-strategies` | 2/24 | Never completed; void patch premise |
| `predictive-ssl-controlled-trial` | 6/13 | gru_pred negative (findings 22/23); path C closed; v1 protocol |
| `score-semantics-realignment` | 8/10 | Residual-semantics fix for the voided patch path; audit trail |

## Archive

`archive/` holds the 19 previously archived changes (2026-11-07 → 2026-11-15);
untouched by this triage.
