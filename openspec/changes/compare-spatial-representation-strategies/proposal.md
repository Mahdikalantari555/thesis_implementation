# Change: compare-spatial-representation-strategies

## Why

Owner proposal (2026-11-15, `compare-spatial-representation-strategies`): a
controlled study of spatial representation strategies — zonal statistics vs
patch sampling vs multi-scale — for the label-free growth-anomaly task, with
the owner's clarification that nothing in the proposal text is definitive
and the ESSENTIAL deliverable is a **transparent comparison of detection
accuracy across spatial scales/representations**. Owner refinement of the
same-day notes.md request; the architecture document's E1/E5 fragments are
elevated from scattered optional ablations to a first-class study.

**What is ALREADY covered elsewhere (no duplication here):** R0 zonal arms
exist and stay (b0_calendar, b1, b3, gru_ae_main, gru_pred); R1 full-field
32×32 patches = `patch-level-controlled-trial` primary; the pixel-vs-patch
question = that trial's E1 CNN-off control; patches-within-polygons, QA
masking, leakage discipline, equal-condition evaluation, farm-level scores +
spatial maps, independent indicators — all owned by the existing program.

**What is genuinely NEW here:** (a) the sub-crop representation (K small
crops per farm + aggregation — the sugar-beet design lineage, absent from
the program), (b) multi-scale representation, (c) the elevation of the
scale question into ONE pre-registered, capacity-matched, transparent
comparison with a single decision table, and (d) measured (not assumed)
data-access and patch-size decisions.

## What Changes

- **Representation ladder (pre-registered):**
  - **R0 — zonal statistics** (existing arms; results REUSED, never re-run).
  - **R1 — full-field patch 32×32** (the PoC's `patch_cnn_gru`).
  - **R2 — sub-crop representation**: K pre-drawn 16×16 crops fully inside
    the field polygon (default fixed hashed crop sets; K ∈ {4, 6, 8} swept
    at single seed first), same PatchCNNGRU trunk at 16×16 input, field
    score = aggregation over crops (top-q mean; majority-vote variant as
    flag). Crops are DERIVED by cropping the materialized 32×32 store — no
    re-extraction. Fields smaller than 16×16 (measured 7.3%) get a recorded
    fallback, not silence.
  - **R3 — multi-scale, two variants in cost order**: (a) multi-dilation
    stem at 32×32 (dilation branches 1/2/4 — broader context, zero extra
    data, ~+11k params); (b) true two-scale fusion (16-crop branch + 32
    branch) built ONLY if R2 or R3a shows broader context matters.
- **Geometry-first scale memo** — 64×64 measured against the real field
  geometry (median side 16.7 px, max 19.6 px ⇒ a 64×64 window is ~93%
  padding/other fields for the median farm): excluded from training by
  default, available behind one flag if the owner still wants it, decision
  documented with the numbers.
- **Data-access decision by measurement, not assumption** — one-time
  micro-benchmark: Zarr lazy random-read throughput vs materialized-store
  preload (the store is ≈1.0 GB fp16, RAM-preloadable). The proposal's lazy
  loading is adopted ONLY if the measurement favors it; recommendation on
  record: materialized, for CPU-bound training.
- **Equal-budget protocol** — same objective (prediction primary + masked
  aux), bins, channels, seeds {13, 42, 2026}, injection sets, σ tables and
  alarm machinery across all representations; capacity-matched (param count
  in every table; >±20% deviation flagged); same farms/cycles (paired).
- **Transparent deliverable (owner's core ask)** — ONE table + figure:
  representation × {MDM80 per temporal shape, event-FAR, detection delay,
  pixel-localization AUPRC, params, epoch time, peak RAM} — detection
  quality, spatial sensitivity, temporal consistency and computational
  efficiency side by side. → finding 24 (Persian).
- **Pre-registered adoption rule** — a representation replaces R1 only if
  MDM80 improves with a paired-bootstrap CI (over fields) excluding zero at
  ≤2× compute; multi-scale is adopted only if it beats the best single
  scale with CI excluding zero. Honest negative otherwise.
- **Explicit non-goals:** no 64×64 training by default; no lazy-pipeline
  rewrite unless the benchmark favors it; no supervised objectives; no
  platform changes; no new injection families; no re-running of R0 arms.

## Sequencing

Gated on the PoC gates (G1–G5) — it consumes patches.zarr and the trained
R1 checkpoints. Feeds the `patch-full-run-and-ablations` final criterion:
that change's E5 scale sweep is delegated HERE (inner-buffer/burn-in
ablations stay there).
