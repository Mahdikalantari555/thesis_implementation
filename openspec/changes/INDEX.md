# OpenSpec changes index — triage 2026-11-23, updated 2026-11-27

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
| `continual-hybrid-representation-design` | 5/55 | PROPOSED — design r2 (2026-11-26) | Owner requirements: SSL + continual representation learning and anomaly discovery; hybrid pixel+context, dual-path phenology, statistical discovery + active learning, regression-set gated updates; plateau-driven CPU training with mandatory checkpoints and verdicts. Feeds the capstone; does not edit its gates |
| `regional-ssl-representation-evaluation` | 15/42 | PROPOSED — the capstone | Level-1/Level-2 evaluation of a light regional SSL representation; five-level ladders; pre-registered gates G-A…G-E; owner decisions registered 2026-11-23 (advantage = undecided; priority = both + applied ambition; Madeline path = tracked future change) |
| `benchmark-latent-reconstruction-prediction-architectures` | 26/37 | PROPOSED — heavy counterpart | Architecture/suppression/synthetic breadth benchmark (Colab T4, budget-aware); feeds the capstone, no capstone gate depends on it |
| `cycle-detection-v3-multindex` | 26/27 | ACTIVE — v2 producer; nearly closed | Only open item is an optional follow-up: the gate FAILED, so no switch; production path stays off; owner call whether to re-gate with `--sweep` after the v2 store completes |
| `cycle-detection-vi-repair` | 11/14 | ACTIVE — artifacts regenerated 2026-11-27; owner verdict PENDING | NDVI+SAVI detector repair landed 2026-11-26 with `overlong_unresolved`/`short_unresolved` flags kept. WP-1 2026-11-27: vi_review4 artifacts regenerated (§7 addendum in `docs/audits/cycle_gap_diagnosis_2026-11-26.md`), retention flags surfaced fail-closed in all mini cycle consumers (b9d5583), E0 cycle-completeness audit written (`docs/audits/e0_cycle_completeness_2026-11-27.md`; alias verified IoU 1.0 on all 490 fields; v2 GO/v1 NO-GO for completeness). Open: 2.3 owner visual review, 2.4 verdict + canonical mini phenology rebuild decision, 3.4 PARKED literature-stage replacement (owner-gated). Path B stays blocked until 2.3 closes positively |
| `time2agri-pretext-architectures` | 7/8 | ACTIVE — phases 1–3.1 done; 3.2 deferred | Faithful Gupta & Sobti 2025 TD/FP/FF arms + benchmark adapters A11–A13 landed; L1 pretext-only gate. Open item 3.2 (2D patch/ViT path) deferred — blocked on the date-centric raw store (HELD) |
| `data-fix-and-colab-prep` | 3/33 | ACTIVE | Data fixes, Colab bundle prep and platform/Rust verification gates. Rust tasks 6.x are stale (WSL-era wording superseded by native Fedora builds, AGENTS §13 2026-11-26); `platform/node_modules` absent → `bun install` before tsc/vite gates |
| `date-centric-raw-store` | 0/28 | PROPOSED — HELD (owner decision; do not build) | Full-temporal/pixel-fidelity raw store with cycle-centric view derived. Phase-0 gates are owner-side: full `Data/Processed/` backup, owner reviews `figures_perfield/` and selects the 5–10 farm subset. Consumers stay deferred (`farm-subset-anomaly-probe`, time2agri 3.2) |
| `farm-subset-anomaly-probe` | 0/20 | PROPOSED — active | First anomaly-score outputs on the owner's machine (CPU-only) validated on a farm subset; 0b.3 fallback to the v2 cycle-centric subset if blocked. Optional consumer of the date-centric raw store |
| `evaluate-olmoearth-nano-sugarcane` | 16/16 | COMPLETE (2026-11-27) — external model evaluation | Verdict **ADAPT**: strong phase backbone (probe 0.854 vs raw 0.627; cross-year margin +0.197) at trivial CPU cost (79 s full store), but no anomaly-detection advantage over the spectral baseline (AUC-PR 0.078 vs 0.075); report at `experiments/olmoearth_eval/REPORT.md`; L1–L2 only; integration would need a future change |
| `validate-olmoearth-temporal-transfer` | 10/13 | ACTIVE — T1/T2 executed + owner amendments round 2 applied 2026-11-27: temporal-confound REJECTED (OE−timestamp +0.239 [0.200,0.279]); unseen-year generalization holds WITHIN-REGION/WITHIN-CROP only (pooled ≥2024 Δ+0.289 [0.242,0.336]; 42 block fields after attrition 4,943/204→965/42; 2025 low-n by field count; ≥2024 margin not estimable — 1 pair); field-level bootstrap throughout; T3 data-gated; T4 Deferred (`date-centric-raw-store` HELD) |
| `code-audit-2026-11-21-deferred` | 4/5 | ACTIVE — 3 P1 latent defects fixed + re-verified 2026-11-27 (WP-9); P2/perf pickup task 3.1 open | Deferred list of `docs/audits/code_audit_2026-11-21.md`. P1: farm_own shrinkage no-op → adae60d (explicit `pooled_reference`, κ acts, fail-closed); absorption polarity → d7432a9 (`share_absorbed = a`, E7 rule = LOWEST `a_patch`); GRU `weight_hh` init slice → b082283 (pre-fix checkpoints: deviation recorded in finding 25, no retrain). Defects re-reproduced on the pre-fix parents (2026-11-27); 88 tests pass at HEAD. P2/perf backlog recorded in the change; still open: #7/#13/#14/#18/#27 + perf 1,3,4,6,7,8,9,10; the `eval_dev.py` deferral gate stands |

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

Archived 2026-11-27 (tasks verified complete before the move; specs synced
manually into `openspec/specs/` — the openspec CLI is unavailable on this
machine):

| Change | Archived spec (new) | Note |
|---|---|---|
| `add-thesis-inspector` | `thesis-inspector` (+12 requirements) | Local read-only research-artifact inspector; 25/25 |
| `field-id-provenance-fix` | `field-id-provenance` (5 requirements) | 6/6. **Follow-up stays OWNER-GATED:** whether to correct the canonical 490-field archive's off-by-one field id (its splits/findings are hash-bound — no silent re-index; todo.md "field-id-provenance-fix" entry) |
| `mini-research-dataset` | `mini-research-dataset` (+12 requirements) | 28/28. Executed lab farms are **L02-02/R12-08/R15-18** (`RoI/3farms-dev.geojson`, 538 scene dates); task 0.1's R12-06/08/10 list described the proposal, not the executed selection |

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
