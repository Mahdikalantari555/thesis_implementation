# Change: regional-ssl-representation-evaluation

**Change ID:** `regional-ssl-representation-evaluation`
**Project:** Thesis Implementation
**Status:** Proposed
**Priority:** High
**Research context:** Self-supervised learning for unlabeled agricultural growth
anomaly detection in sugarcane fields using Landsat time series.
**Base:** branch `arena/ed3c4228-thesis-implementation` @ `2b479dc`
**Evidence date:** 2026-11-23 (advisor analysis relayed by the owner)

---

## Why

The frozen experimental lineage does not support the thesis's strongest possible
claim. Under the v2 fixed protocol (findings 23–25), the SSL arms (`gru_ae_main`,
`gru_pred`, the normal-bank branches and both patch-level arms) fail to beat the
phase-aware statistical baseline `B1` on the controlled severity protocol: MDM80
is censored beyond S6 in every model × family × seed cell, the L2 rung is not
cleared, and L5 (SSL beats baselines) is unmet. Finding 20 attributes the failure
to the objective/unit, finding 21 rejected the untrained forecast unit, finding 23
closed path C (`gru_pred`) as a negative result, and finding 25 records an honest
"archive gate red / frozen winner void" verdict on the v2 chain.

The advisor analysis (2026-11-23) reframes the contribution:

> The thesis is not required to produce the best stress-detection algorithm. It
> exists to build and **measure a reusable, lightweight, regional
> self-supervised representation of agricultural field dynamics** — a regional
> base model for the Amirkabir (Khuzestan) sugarcane archive — and to test,
> without assuming the answer in advance, whether that representation carries
> information that simple physical rules do not.

Two facts make this reframing mandatory rather than cosmetic:

1. A simple NDVI + LST rule is a genuinely strong competitor: both variables are
   nearly direct physical readouts of water stress (vegetation vigour; surface
   thermal response). Any claim that a deep model is *scientifically* better
   must be earned against FAR-matched, capacity-honest controls — not against a
   strawman.
2. The honest, already-sanctioned fallback (vision.md, G2) is that SSL's role is
   limited to embedding/retrieval if it does not beat B1. This change makes that
   fallback a **measured, first-class result** instead of an apologia.

This change operationalizes the advisor-endorsed reframing as a single capstone
evaluation framework. It defines *what must be measured and under which
protocol* — it does not bless any architecture in advance (the owner's
warning stands: "architecture may not be right yet"), and it does not modify,
reopen or reinterpret any frozen artifact or prior finding.

### The five-level structure (explicit)

This change is deliberately built around five-level ladders, so the owner can
see at a glance that all of them are covered:

1. **B1–B5 incremental input ladder** — the advisor's ablation:
   B1 = FAR-matched NDVI+LST rule · B2 = SSL on NDVI+LST only ·
   B3 = SSL on all spectral+thermal channels · B4 = B3 + weather ·
   B5 = B4 + phenological/temporal decomposition context.
2. **Label fractions 5 / 10 / 25 / 50 / 100 %** — the transferability ladder
   for every label-scarce downstream benchmark.
3. **L1–L5 validation ladder** — every reported claim is tagged with its rung
   (L1 learned growth patterns → L2 controlled deviations → L3
   generalization across years/farms → L4 independent agronomic evidence →
   L5 beats baselines). Current honest rung: **L2 (not cleared)**.
4. **Five downstream task families** — phenology-stage estimation (T1),
   limited-label synthetic anomaly detection (T2), few-shot similar-date
   retrieval (T3, the owner's stated next task 2026-11-23), self-supervised
   trajectory quality (T4), and exploratory crop-condition / irrigation-pattern
   analysis (T5, descriptive only).
5. **Five potential-advantage hypotheses H1–H5** — the advisor's research-
   question table (temporal behaviour, multivariate relations, farm-specific
   modelling, complex/gradual/multivariate anomalies, generalization), each
   converted into a falsifiable, pre-registered question with a pass/fail gate.

In addition, the two evaluation levels (**Level 1 — Pretraining Quality**,
**Level 2 — Downstream Transferability**) and the three generalization levels
(**Temporal / Spatial / Geographic**) are implemented as separate, separately
reportable sections. Geographic generalization is **out of scope** for data
reasons and is recorded as a claim ceiling, not as a result.

## What Changes

- **New capability `representation-pretraining-quality`** — a Level-1 metric
  battery executed on development clean cycles only: masked reconstruction
  quality, farm-identity separability, phenological-phase linear probing,
  embedding temporal stability, and collapse diagnostics, all with seeds, spread
  and confidence intervals.
- **New capability `downstream-transfer-evaluation`** — a Level-2 benchmark
  suite: the B1–B5 incremental ladder under identical, FAR-matched conditions;
  label-fraction transfer curves (5/10/25/50/100 %) for five arms per task;
  the few-shot similar-date retrieval benchmark with two strictly separated
  tracks (injected-labelled vs weak-proxy) plus a descriptive natural-data
  near-match listing for manual owner inspection.
- **New capability `generalization-and-claim-semantics`** — Temporal
  (year-forward rolls), Spatial (leave-one-block-out) and Geographic
  (out-of-scope, documented) generalization protocols; the anomaly →
  stress-candidate → stress semantic ladder with claim ceilings; and the
  pre-registered decision gates G-A…G-E whose outcomes (including total
  failure) are publishable results.
- **New evaluation harness** under `src/ssl_anomaly/evaluation/` (metric
  probes, transfer-runner, retrieval benchmark, B1–B5 runner, generalization
  protocols, gate aggregator, report builder). New code only; zero edits to
  frozen training/eval/scoring paths.
- **A lightweight reference encoder** ("region model v0") defined by explicit
  constraints (≤250k parameters, CPU-first, ≤15 min/seed laptop budget,
  deterministic, resume-safe), with the architecture itself decided by an
  in-change, pre-registered screening gate over a candidate list — never by
  taste. Candidates include the incumbent GRU-AE trunk (reused as encoder),
  a Temporal Barlow Twins variant, a self-supervised trajectory predictor
  (no yield labels required), a hybrid, and a decomposition-conditioned
  encoder. No candidate is blessed in this proposal.
- **Capstone findings document 26** (EN + FA) and machine-readable reports
  under `reports/regional_representation/` with full provenance (hashes, seeds,
  git SHA, split ids, data_source, wall time).
- **`todo.md` updated** with the outcome; findings ≤ 25 remain immutable.

### Explicit non-goals

- Replacing, retraining or reinterpreting the frozen detection path
  (`gru_ae_main` v2 sidecars, severity protocol, B1, calibration, locked test).
- Any claim of **stress detection** (L4/L5) — this change reports screening
  evidence and proxies only; agronomic validation remains external and pending.
- The broad architecture-competition benchmark (`benchmark-latent-reconstruction-
  prediction-architectures`, A1–A10 × Mamba/JEPA/suppression matrix), created
  2026-11-23 (owner decision) as a **separate change**. This change does not
  depend on it; it consumes whatever that benchmark legitimately measures. If it
  is deferred, this change still closes.
- Foundation models / LLMs / VLMs as primary architectures; commercial imagery;
  yield forecasting (no yield labels exist); new satellite or weather data.
- Madeline's path (initializing region model v0 from external early-season
  yield-forecasting-model embeddings): not in scope — no such artifact exists
  in-repo and no yield labels exist; if wanted later it is a separate change
  with its own data contract.
- Platform (`platform/`) changes; spec-target edits after the fact.

## Capabilities

### New Capabilities

- `representation-pretraining-quality`
- `downstream-transfer-evaluation`
- `generalization-and-claim-semantics`

### Modified Capabilities

- None. `model-development-protocol` is referenced (invariants, test isolation,
  provenance rules) but not modified.

## Impact

- **Affected code:** new modules under `src/ssl_anomaly/evaluation/`,
  `configs/evaluation/` (pre-registration), `scripts/run_*.py` probes (thin
  CLIs), `tests/evaluation/` (TDD-first), `docs/findings/26_*.md(+ .fa.md)`,
  `reports/regional_representation/`, `todo.md`.
- **Not affected:** frozen checkpoints/sidecars; `splits_locked_v2.json`
  (`30ecbe10…`); `holdout_manifest_v2.json` (`d501bf81…`); `B1`; platform;
  locked final-test path; findings 00–25; v1 artifacts (quarantine rules of
  finding 25 stand).
- **Compute:** CPU-first laptop budget; anything longer than the AGENTS.md §6
  threshold (≈2 min) is handed to the owner as exact commands; Colab is used
  only after the local gate (unit tests + `--smoke` inspected) per AGENTS.md
  §6.1. The agent never launches long jobs itself.
- **Data:** v2 chain artifacts only (e.g. `cycles_490_v2.parquet` — 3,478
  uncensored cycles / 490 fields, median 366 d; `norm_stats_full.json`
  `3cb1ca60…`). v1 artifacts are not used for new evidence.
- **Claim labels:** every number in the capstone report carries rung tags and a
  provenance line; unlabelled or unavailable evidence is `absent`, never
  estimated.

## Owner decisions (2026-11-23)

1. **"Five levels" — all ladders kept.** The change covers B1–B5, 5/10/25/50/
   100 % label fractions, L1–L5, five downstream tasks and H1–H5 as written.
2. **Region model v0 architecture — screening gate kept.** No architecture
   pre-approved; the pre-registered in-change screening decides. Barlow Twins
   remains one candidate, not the plan.
3. **A1–A10 draft — created as a separate change**
   (`benchmark-latent-reconstruction-prediction-architectures`).
4. **The 22 existing changes — triaged as superseded** (v2 chain / finding 25);
   `openspec/changes/INDEX.md` added; superseded folders marked, not archived;
   the two post-finding-25 patch-* folders (`oracle-floor-audit`,
   `patch-alarm-statistic-e1`) left DORMANT for owner review.
5. **L4 evidence path — still required externally**; no change in this change.
6. **T3 retrieval — both tracks included, plus a descriptive natural-data
   near-match listing** (top-10 neighbours per query, explicitly descriptive,
   never gate input) for manual owner inspection.
7. **Advisor's closing question 1 — expected main advantage: UNDECIDED
   (owner decision 2026-11-23).** No single advantage is pre-asserted; H1–H5
   remain fully pre-registered and the capstone gates (G-A…G-E) are the
   mechanism that decides which advantage the evidence supports. The thesis
   narrative is scoped only after the results exist — no motivated direction.
8. **Advisor's closing question 2 — priority: BOTH frames, plus a registered
   applied ambition (owner decision 2026-11-23; owner's words, English gloss
   verbatim):** «پژوهش علمی معتبر و همچنین دقت بالا در حل مسائل کشاورزی،
   مثل تشخیص تنش زودهنگام، دادن راهکار، تبدیل شدن به مدل مرکزی یک سیستم هوش
   مصنوعی برای آبیاری هوشمند و کشاورزی دقیق» — scientifically valid research
   AND high accuracy on real agricultural problems: early stress detection,
   giving recommendations, and becoming the central model of an AI system for
   smart irrigation and precision agriculture. Registered as aspiration with
   hard ceilings:
   (i) *Early stress detection* is an unmet claim today — finding 25 records
   detection only from onset and only at S6; it stays a gate outcome, never a
   promise, until the capstone evidence says otherwise.
   (ii) *Recommendations («دادن راهکار»)*, *decision support* and *being the
   central AI of an irrigation system* are vision-stage capabilities recorded
   in `docs/vision.md` as later-stage items; they require agronomic
   validation (L4) and a product/decision-support contract; no artifact of
   this change is presented as irrigation advice, and no platform copy may
   imply it.
   (iii) The scientific frame keeps full force: negative results are published
   with equal prominence (G-E); competitive performance claims still need
   L5 evidence.
9. **Madeline's path (external early-season yield-forecasting-model embeddings
   as initialization) — tracked as a FUTURE separate change (owner decision
   2026-11-23).** Not implemented here: no such artifact exists in-repo, no
   yield labels exist, and any external weights need their own data, licence
   and provenance contract before an openspec change is even written. Added
   to `todo.md` as tracked-not-created.

## Open Questions (still open)

5. **L4 evidence path** — company harvest records / expert review: timeline and
   willingness to provide labels? Required for any L4 claim (and therefore for
   every part of the registered applied ambition above).
6. **Whether T5 clustering/near-match outputs should feed the platform later**
   (out of scope here).
