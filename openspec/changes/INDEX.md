# OpenSpec changes index — triage 2026-11-23, archived 2026-11-23

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
| `regional-ssl-representation-evaluation` | 0/42 | PROPOSED — the capstone | Level-1/Level-2 evaluation of a light regional SSL representation; five-level ladders; pre-registered gates G-A…G-E; owner decisions registered 2026-11-23 (advantage = undecided; priority = both + applied ambition; Madeline path = tracked future change) |
| `benchmark-latent-reconstruction-prediction-architectures` | 0/37 | PROPOSED — heavy counterpart | Architecture/suppression/synthetic breadth benchmark (Colab T4, budget-aware); feeds the capstone, no capstone gate depends on it |
| `cycle-detection-v3-multindex` | 26/27 | ACTIVE — v2 producer; nearly closed | Only open item is an optional follow-up: the gate FAILED, so no switch; production path stays off; owner call whether to re-gate with `--sweep` after the v2 store completes |
| `optional-rust-acceleration` | 9/12 | APPLIED — gating tasks done; 3 deferred | Pure Python stays canonical; Rust ports opt-in via `fast-anything build` + `enable`; opt-out `FAST_ANYTHING_DISABLE=1`; docs + smoke test in `docs/optional_rust_acceleration.md`; no scientific claim (L0) |
| `eval-artifact-cache` | 12/14 | APPLIED — gating tasks done; 2 deferred | Opt-in fingerprinted L1/L2/L3 disk cache of normalized batch, clean reference paths and per-cell injected tensors; exact NaN-aware parity (72 cells); untracked `data/cache/eval/`; warm 1.46× at n=32 |

## Dormant — unapplied, await owner decision

| Change | Tasks | Notes |
|---|---|---|
| `oracle-floor-audit` | 14/15 | Post-finding-25 pool-V audit (σ-floor verdict); analysis stands, but it restates the voided patch direction; needs owner review before any follow-up |
| `patch-alarm-statistic-e1` | 0/11 | Unapplied child of oracle-floor-audit (channel-max residual aggregation; power 0.30–0.67 vs committed 0.06–0.46); blocked on a patch-revival decision |

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

## Archive (`openspec/changes/archive/`)

19 changes archived previously (2026-11-07 → 2026-11-15). Archived
2026-11-23 (tasks verified complete; main specs created/updated by
`/opsx:archive`):

| Change | Archived spec (new) |
|---|---|
| `2026-11-23-decomposition-aware-ssl-study` | `decomposition-aware-anomaly-study` (+7 requirements) |
| `2026-11-23-training-holdout-invariant` | `training-holdout` (+2 requirements) |
| `2026-11-23-separate-verification-gates` | `research-verification` (+1 requirement) |
