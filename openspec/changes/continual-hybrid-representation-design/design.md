# Design — continual-hybrid-representation-design

**Date:** 2026-11-26 · **Status:** PROPOSED · **Claim ceiling:** none (design only)

Evidence tags used on every proposal:
**[R]** research-backed (source named) · **[C]** exists in code (file named) ·
**[D]** needs development · **[H]** experimental hypothesis, not yet supported.
Nothing here is a result. No architecture is declared a winner.

---

## 0. Scope and framing

Working title (framework, not the approved thesis title): *Self-supervised,
continual representation learning of sugarcane field dynamics and anomaly
discovery from Landsat time series.*

The framework has four layers, each independently testable:

1. **Representation** — a self-supervised encoder of per-pixel spectral–temporal
   behaviour with spatial context (§1–§3); phenology paths in §5.
2. **Discovery** — model scores and model-independent statistical evidence,
   fused into a ranked, uncertainty-tiered candidate list (§6).
3. **Verification** — active learning over candidates; only independently
   verified items become anomalies (§6).
4. **Continual update** — controlled, gated updates as new seasons, farms or
   cycles arrive (§7).

The training policy (§4) applies to every trainable component.

---

## 1. Proposed architecture and data flow

```
 Landsat stack per farm (D dates × 7 bands + LST, pixel_valid, in_field)
 + Δt (days since previous obs) + DOY + shared weather (context only)
                     │
          ┌──────────┴───────────┐
          │  window sampler      │  pixel-centred windows, L obs, irregular Δt,
          │  (no cross-boundary  │  observation mask carried, never NaN→0
          │   windows; split-safe)│
          └──────────┬───────────┘
        ┌────────────┴─────────────┐
        ▼                          ▼
  PIXEL PATH  [C/D]          CONTEXT PATH  [D]
  x_pix (L, C+mask+Δt)       x_ctx (L, C+mask, k, k) masked to in_field
  per-obs MLP tokenizer      shared 2D CNN per date (k ∈ {3,5,7,9,11})
        │                          │  → pooled context token per date
        ▼                          ▼
  temporal backbone  ◄──── fusion F (gated add / FiLM / concat) [H]
  (Mamba-lite | GRU control)       per date, before the temporal backbone
        │
        ▼
  h_t (per-obs latent), z (window latent)
        │
  ┌─────┼──────────────┬─────────────────┬──────────────────┐
  ▼     ▼              ▼                 ▼                  ▼
 REC   FCST          JEPA             BARLOW            MEMORY
 head  head          predictor        projector         (normal / anomaly)
 masked next-obs     latent of        invariance +      kNN distance
 obs   features      masked/future    redundancy        in z-space
 recon               spans vs EMA     reduction (reg.)
                     target encoder
  │     │              │                 │                  │
  └─────┴──────┬───────┴─────────────────┘                  │
               ▼                                            ▼
      per-pixel per-date residual scores        memory distance score
               └───────────────┬────────────────────────────┘
                               ▼
     phenology-agnostic score  +  phenology-aware score (§5)
                               ▼
      rank fusion with statistical detectors (§6) → tiers → active learning
```

Key properties:
- **Pixel is the scoring unit**; context only conditions it, so a 1-pixel
  anomaly is never averaged away by a patch pool [H, tested in E2].
- **Out-of-field pixels are masked before the first convolution**, as already
  done in `models/patch_cnn_gru.py` [C].
- Weather is a context covariate, identical for all three mini farms (one
  station series); it cannot explain between-farm differences.

---

## 2. Role of every encoder, head and objective

| Component | Role | Status | Evidence / note |
|---|---|---|---|
| Per-obs tokenizer (MLP over bands + mask + Δt) | handle irregular sampling and missing obs explicitly | [C] partly (`models/tokenizer.py`), [D] for mask+Δt in mini | Presto-style irregular inputs [R: Presto] |
| Temporal backbone: Mamba-lite (`TinyCNNMamba`) | long-range temporal dependency, O(L), pure PyTorch, CPU-feasible | [C] `models/cnn_mamba.py` | finding 28: best control-adjusted gain on synthetic target (L2 only) |
| Temporal backbone control: GRU-AE | simpler recurrent control at matched params | [C] `models/gru_ae.py` | required for fairness (capacity band ±20 %) |
| Temporal backbone option: small Transformer | attention baseline | [C] `models/masked_tf.py`, `models/jepa.py` encoder | optional arm |
| Context CNN (2D, per date, shared weights) | neighbourhood spatial patterns | [C] stem in `models/patch_cnn_gru.py`; [D] as a fusable branch | UTAE +2.0 OA on PASTIS but rank reversal under year shift [R: Vincent 2024] |
| Fusion F | combine pixel and context tokens | [D] | gated-add vs FiLM vs concat is an ablation [H] |
| Reconstruction head | reconstruct masked observations (real cloud gaps + random masks) | [C] AE decoders | basis of L1-a gap reconstruction |
| Forecast head | predict next observation's features from past only (causal) | [C] `patch_cnn_gru` horizon heads; [D] in mini | anomaly = surprise vs forecast [R: reconstruction/forecast TSAD survey, `thesis_research/notes/yahya_*`] |
| JEPA objective | predict latent of masked/future spans from context; EMA target | [C] `models/jepa.py` (note: full-field failure in finding 28) | objective, **not** a backbone (owner rule) |
| Barlow Twins objective | invariance to view perturbations + redundancy reduction | [C] `models/rst_bt.py`, `mini_lab/runner.py::_barlow_loss` | regularizer only; never the anomaly score (owner rule) [R: Barlow 2021; ST-BT note] |
| Variance/covariance monitor (VICReg-style) | detect collapse | [C] `benchmarking/collapse.py` | effective rank 1.31–1.65 observed in mini probes → mandatory |
| Normal memory | reference set of confident-normal embeddings | [C] `monitor/normal_bank.py` (immutable, cycle-based) | [D] versioned, pixel-window entries |
| Anomaly memory | verified anomalies only | [D] | used for evaluation and few-shot heads, never for SSL training |

**Barlow views (augmentation policy).** Allowed: random observation dropout,
temporal sub-sampling, Δt jitter within one revisit, small radiometric noise
within sensor noise. **Forbidden:** level shifts, band scaling beyond noise,
temporal warping that moves a phenological transition, smoothing over
>1 revisit — these can erase real stress or phenology [H; owner rule 3.3].
An "augmentation-erasure test" (§8) checks that a synthetic NDVI drop stays
separable after the augmentation.

**Loss composition.** L = λ_rec·L_rec + λ_fc·L_fc + λ_jepa·L_jepa + λ_bt·L_bt.
Number of heads/losses is decided only by ablation (E1), starting from the
single-loss arms that already exist.

**Anomaly score.** S = robust-z of {recon residual, forecast residual, JEPA
latent error, normal-memory kNN distance}, combined by a pre-registered rule
(max of robust z per component, or rank mean). Barlow never contributes a
score term.

---

## 3. Pixel + spatial context fusion (owner decision 2.3)

- Context is a k×k in-field-masked neighbourhood per date, encoded by a shared
  2D CNN, pooled to one token, fused with the pixel token **before** the
  temporal backbone (early fusion) or **after** (late fusion: two temporal
  streams) — both are arms [H].
- k ∈ {3, 5, 7, 9, 11}; finding 28 found gains flat across patch 1–11 on the
  synthetic target, and the literature supports 5×5 as a standard
  [R: Presto 5×5; LULC 5×5 > 7×7 > 64×64, abstract-level]. 48×48 is dropped
  (larger than a ~400 m field).
- Small-anomaly sensitivity test: inject synthetic anomalies of footprint
  1, 3×3, 9×9 pixels; the pixel path must keep detecting the 1-pixel case at the
  pre-registered level; a hybrid that loses it is rejected even if better on
  large footprints.

---

## 4. Training policy (owner directive 2026-11-26)

Laptop CPU only. Every trainable arm uses one shared trainer.

### 4.1 Plateau-driven stopping
- Evaluate every `eval_every` optimizer steps on a **validation split that is
  never the test split** (mini: dev year 2023; train ≤2022; test ≥2024).
- Monitored metric: validation total loss (and each component logged).
- **Plateau** = no relative improvement > `min_rel_delta` (default 0.5 %) of the
  best value for `patience` consecutive evaluations (default 8), **and** the
  slope of an EMA-smoothed validation curve over the last `window` evals
  (default 10) is ≥ −`min_rel_delta`/window. Both conditions avoid stopping on
  noise.
- Hard budget cap: `max_steps` and `max_wall_minutes` (default unbounded steps,
  wall cap set by the owner per session).
- `min_steps` before plateau can trigger (warm-up).
- On plateau, an optional LR drop (×0.5) with `lr_patience` before final stop;
  final stop only after `max_lr_drops` is reached.

### 4.2 Verdicts (written to `run_manifest.json` and printed)
| Verdict | Condition | Message to owner |
|---|---|---|
| `CONVERGED_PLATEAU` | plateau rule met | stopped early at step N |
| `BUDGET_EXHAUSTED_STILL_IMPROVING` | cap hit while the slope is still negative beyond threshold | "needs more training"; prints exact `--resume` command and an extrapolated number of extra steps |
| `BUDGET_EXHAUSTED_FLAT` | cap hit, curve flat but patience not reached | likely converged; optional resume |
| `DIVERGED` | NaN/inf loss or val loss > `diverge_factor`×best | stop, keep best checkpoint |
| `COLLAPSED` | effective rank < threshold or per-dim std below floor for `patience` evals | stop, flag representation collapse |

### 4.3 Checkpointing (always)
- `last.pt` every eval and every `ckpt_every_minutes` (default 5): model,
  optimizer, scheduler, step, epoch, RNG states (python/numpy/torch), data
  sampler position, early-stopping state, EMA/target encoder, config hash,
  data/split hashes, git SHA. Atomic write (tmp + rename).
- `best.pt` whenever the monitored metric improves.
- SIGINT/Ctrl-C writes `last.pt` before exit.
- `--resume` continues the same run bit-for-bit; `--init-from` starts a new
  run from a checkpoint (continual learning, §7); neither silently
  re-initializes.

### 4.4 Verbose output
- One line per eval: step, epoch, train loss (and components), val loss, best,
  relative improvement, patience counter k/P, smoothed slope, LR, effective
  rank, steps/s, RAM, ETA to cap.
- `progress.jsonl` with the same fields (machine-readable).
- End-of-run summary block: verdict, best step, final vs best val, wall time,
  params, and the resume command when relevant.

### 4.5 Fairness under plateau stopping
Plateau stopping gives arms unequal compute. Every comparison reports **both**
(a) the to-plateau result with steps/wall/params and (b) an equal-step cut
taken from checkpoints at the smallest arm's plateau step [R: He et al. 2018,
pretraining vs convergence]. Random-init controls are evaluated at the same
step as their trained twin.

### 4.6 CPU and memory budget (estimates, to be measured in task 1.6)
- Mini: 3 farms × ~280 in-field pixels × 538 dates × 8 channels fp16 ≈ 7 MB of
  pixel series; context windows k=11: B=64, L=60, C=9 → ~15 MB per batch fp32.
- 490 farms: ~135 k pixels × 538 × 8 fp16 ≈ 1.2 GB on disk; must be streamed
  from zarr, not loaded whole.
- Params: Mamba-lite d=64/2 layers ≈ 293 k, GRU-AE ≈ 160 k (finding 28) — the
  capacity band ±20 % needs GRU widened or Mamba narrowed.
- BLAS threads pinned via `utils/throttle.py` [C].
- Owner laptop core count and RAM are unknown here; task 1.6 measures
  steps/s per arm before any long run.

---

## 5. Phenology dual path (owner decision 2.2)

- **Path A — phenology-agnostic** [D]: score from §2 with a calendar-free
  normal reference (pooled normal memory; same-pixel previous years at any
  stage). Works with no stage estimate.
- **Path B — phenology-aware** [D/H]: a stage posterior p(s | window) over
  literature stages {tillering, grand growth, maturity/ripening} plus
  fallow/harvest, with a confidence c. Score = Σ_s p(s)·dev_s, where dev_s
  compares against the stage-conditioned normal memory. If c < c_min, Path B
  defers to Path A and records it.
- Stage estimation: Joshi et al. 2024 pipeline (BISE → 14-day resample → FFT →
  DoG troughs; FAO two-crop growth stages) [R: Joshi 2024, finding 13];
  plant/ratoon from the **corrected** age attribute (after the id fix).
  Uncertainty from the SOS/EOS RMSE (17–27 days) → soft stage boundaries.
- The existing S1–S4 DSH bins stay as a pre-registered comparator arm [C]
  (`phenology/mini.py`).
- **Robustness test:** shift stage estimates by ±16 and ±32 days, and randomly
  swap 10 % of cycles; Path B must degrade gracefully (pre-registered maximum
  drop) and never fall below Path A by more than a set margin.
- Blocker: the owner's visual check of the cycle timelines
  (`reports/mini_dataset/figures/timeline_seasonal.png`) — "are all detectable
  cycles detected?" — must be recorded before stage-dependent results count.

---

## 6. Statistical discovery + active learning (owner decision 2.1)

### 6.1 Model-independent detectors [D; methods R]
1. **Temporal robust z** — same pixel, same stage (or same DOY window) in prior
   years; median/MAD.
2. **Within-field spatial outlier** — pixel vs field median at the same date
   (robust z), and local spatial autocorrelation (Getis-Ord Gi*) for clusters.
3. **Breakpoint / harmonic residual** — BFAST-style or harmonic seasonal model
   residual [R: Verbesselt 2012 note].
4. **Multi-band consistency** — a candidate must deviate coherently in ≥2
   independent bands/indices (e.g. NIR-based and SWIR-based), not only in one.
CWSI, ETa (METRIC) and TVDI are **auxiliary evidence** with flagged dependence
on model inputs (LST, NIR) — never ground truth.

### 6.2 Fusion and tiers
- Rank fusion (mean rank or reciprocal-rank) of model score and statistical
  detectors; disagreement between them is an uncertainty signal.
- Tiers: **Normal-confident** (all sources low), **Suspected** (any source high
  or high disagreement), **Verified** (independent confirmation only: expert
  blind review or independent records — irrigation, ECe, yield, field reports).
- Suspected items enter **quarantine**: excluded from normal memory, from replay,
  and from SSL updates until resolved.

### 6.3 Active learning
- Query score = uncertainty (disagreement + score near threshold) ×
  diversity (k-center in embedding space) × spread across farm/year/stage.
- Batch of K candidates per review session; reviewer blind to the score and
  source; a fixed share of random Normal-confident items is mixed in to
  estimate the false-normal rate.
- Reviewer outcomes: verified anomaly / normal / unsure (unsure stays
  quarantined).

### 6.4 Avoiding the confirmation loop
- Statistical-detector outputs are **never** used as positive training targets
  for the encoder; the model must not learn to imitate them.
- SSL updates use only Normal-confident data, weighted by confidence.
- A held-back share of verified items is reserved for evaluation only.
- Report agreement between model and detectors separately from agreement with
  verified labels.

### 6.5 Evaluation unit and claims
- Units: **Pixel × time window** and **Spatial Anomaly Event** (connected
  in-field pixels flagged on ≥2 consecutive acquisitions).
- Without a valid verified set: report only stability, cross-detector
  agreement, temporal/spatial coherence, cross-year recurrence and robustness
  to missing data — **no Precision/Recall/"stress detected"**.
- Precision@K only on a valid blind-reviewed set.

---

## 7. Memory and continual-learning protocol (owner section 4)

### 7.1 Versioning
Each model version stores: weights, config, data manifest hash, split hash,
normalization statistics (train-only), memory snapshot hashes, git SHA, parent
version. Old runs keep their own sidecars.

### 7.2 Update triggers
Updates are batched — season end, new farm batch, or drift alarm — never per
observation [owner rule]. New data first goes to the scorer, then to the
triage, then (only Normal-confident) to the update pool.

### 7.3 Mechanisms (implementation order)
1. **Experience replay** [R; D] — stratified buffer by farm × year × stage ×
   pattern type; fixed replay fraction per batch.
2. **Teacher–student distillation (LwF / feature matching)** [R; D] — previous
   version frozen as teacher; L_distill on replay + new inputs.
3. **EWC** [R; D, optional] — Fisher-diagonal penalty; experimental arm.
4. **Drift detection** [R; D] — MMD or energy distance between new and replay
   embeddings; KS on input bands; large drift routes to review instead of a
   blind update.
5. **Normal memory / anomaly memory** — separate, versioned; normal memory
   only from Normal-confident; anomaly memory only Verified.
6. **Anomaly quarantine** — §6.2.
`train/incremental.py::fine_tune` already sketches trimming the top-scoring
fraction [C stub]; it is replaced, not extended.

### 7.4 Acceptance gate
- A fixed **Regression Set** (farm × year × stage stratified; pre-registered,
  hash-recorded, never trained on after freezing).
- Metrics: L1 metrics on the regression set; performance on new data;
  **Backward Transfer (BWT)**, **Forgetting** (best past − current), Forward
  Transfer.
- Accept a new version only if forgetting ≤ δ_forget (pre-registered, e.g.
  relative 2–5 % of the regression metric) **and** new-data performance ≥ the
  warm-start-only reference − δ_new. Otherwise reject and keep the parent.
- Zero forgetting is not promised; forgetting is bounded and measured.

---

## 8. Experiments, ablations, metrics

All experiments: leakage control, time split (≤2022 / 2023 / ≥2024), farm
holdout where possible (LOFO on 3 farms is weak evidence), seeds {13, 42, 2026},
cluster bootstrap CIs at farm × year, random-init same-architecture control,
rule/feature baselines (A-rule, A-feat), collapse diagnostics, **synthetic
results reported apart from real results** (label `synthetic-injection` vs
`real-discovery`).

| ID | Axis | Arms | Primary metrics | Ablations |
|---|---|---|---|---|
| E1 | Representation learning | Recon, Barlow+recon, JEPA, JEPA+recon, Hybrid (all four) | L1-a real-gap reconstruction vs linear/Whittaker/harmonic; L1-b forecast vs persistence/DOY climatology; stage linear probe; effective rank; synthetic injection AUPRC (secondary) | number of heads/losses; λ weights; augmentation erasure |
| E2 | Spatial–temporal modelling | pixel-only, context-CNN-only, hybrid early fusion, hybrid late fusion; k ∈ {3,5,7,9,11} | same as E1 + small-footprint injection sensitivity (1, 3×3, 9×9 px) | fusion type; k |
| E3 | Real anomaly discovery | best 2–3 encoders from E1/E2 × Path A/B × statistical detectors | stability across seeds; cross-detector agreement; spatial/temporal coherence; cross-year recurrence; Precision@K **only** with blind review | detector subsets; fusion rule |
| E4 | Adaptation | frozen encoder, linear probe, partial FT, full FT | stage probe, retrieval, few-shot head on verified set (if any) | label count {1,2,4,8,all cycles} |
| E5 | Continual learning | scratch, warm start, warm + replay, warm + replay + distill, (+EWC) | BWT, forgetting, new-data perf, wall time, acceptance-gate pass rate | replay fraction; distill weight |

**Sanity tests with known answers:** permuted temporal order must hurt
L1-b; random-init must sit near DOY climatology on L1-c; an S6-strength
perturbation must be found by every arm; label permutation null for any
probe; augmentation-erasure test (§2).

**Model selection rule** (pre-registered before running): the smallest model /
fewest losses / smallest k not meaningfully worse (CI of difference includes
0 or within margin) than the next larger, decided on the validation split only.

---

## 9. Limitations, risks, unproven assumptions

- [H] SSL pretraining improves real anomaly discovery here — **unsupported**;
  finding 28 shows the opposite on weak labels. A null result must be
  publishable (literature has no precedent for an SSL advantage on this task:
  `research/ssl_advantage_2026-11-25/REPORT.md`).
- 3 mini farms cannot establish spatial generalization; 490-field runs need the
  pixel store and the archive-wide id fix.
- Single weather series for the whole area.
- No independent agronomic modality yet → claims cap at L3.
- METRIC-derived CWSI/ETa share inputs with the model.
- Plateau stopping and equal compute conflict (§4.5).
- CPU throughput may make E5 on 490 fields infeasible locally; then E5 runs on
  the mini lab and a sampled 490 subset.
- `_cnn_mamba` and `_mamba` are the same model under the confirm config
  (`mini_lab/models.py:168-193`); they must be made distinct or merged before
  any architecture comparison.
- JEPA full-field arm failed (`'NoneType' object is not subscriptable`) and
  Barlow full-field failed at batch size 1; both need fixing or exclusion.

---

## 10. Comparison with the current repository

| Requirement | Current code | Gap |
|---|---|---|
| CNN spatial features | `models/patch_cnn_gru.py` (masked 2D stem, 32×32 patches, causal GRU) [C] | needs to become a fusable context branch with k ≤ 11; pixel scoring kept |
| Mamba | `models/cnn_mamba.py` `TinyCNNMamba` [C] | duplicate registry entry; add mask+Δt tokens |
| JEPA | `models/jepa.py` (EMA target, masked-token latent prediction) [C] | full-field failure; future-span prediction mode [D] |
| Barlow Twins | `models/rst_bt.py`, mini `_barlow_loss` [C] | use as regularizer on views from §2; batch-size guard |
| Reconstruction / forecast heads | AE decoders [C]; horizon heads in patch model [C] | unified multi-head module [D] |
| Benchmark arms | `models/bench/arms.py`, `benchmarking/registry.py` (490 path) [C]; `mini_lab/models.py` 9-arch registry [C] | one trainer for both; avoid rewriting arms |
| Collapse diagnostics | `benchmarking/collapse.py` [C] | wire into trainer verdicts |
| Normal memory | `monitor/normal_bank.py` (immutable, cycle bins) [C] | versioned, pixel-window entries, quarantine [D] |
| Contamination / trimming | `train/contamination.py` [C] | reuse for quarantine experiments |
| Continual learning | `train/incremental.py` stub [C stub] | everything in §7 [D] |
| Training loop | mini: fixed epochs/`max_steps`, `last.pt` per epoch [C]; 490: Lightning with `early_stop_patience` [C] | plateau verdicts, wall-time checkpoints, RNG/sampler state, verbose summary [D] |
| Phenology | `phenology/cycles.py` (harvest troughs), `phenology/mini.py` (S1–S4 DSH bins) [C] | literature stages with uncertainty [D]; owner visual check pending |
| Field-id mapping | mini alias `archive_fid = reference_fid + 1` verified on 3 farms [C] | archive-wide verification on 490 [D] |
| Synthetic vs real separation | finding 28 separates them by text [C] | enforce via report label field [D] |

---

## 11. Prioritized roadmap (implementation, next sessions)

1. **P0 — Trainer** (`training-convergence-control`): plateau + verdicts +
   checkpoints + verbose; retrofit mini runner and sweep script. Unit tests
   with synthetic curves (plateau, still-improving, diverge, collapse).
2. **P0 — Hygiene**: de-duplicate `_cnn_mamba`; fix JEPA/Barlow full-field;
   mask + Δt tokens; capacity matching.
3. **P1 — Archive-wide id verification** (extend `field-id-provenance-fix`).
4. **P1 — E1 + E2 on mini** (CPU, owner runs, to plateau).
5. **P1 — Statistical detectors + tiers + quarantine** (no model needed).
6. **P2 — Phenology Path B** after the owner's timeline check.
7. **P2 — E3 discovery report + blind-review packet** (owner/expert).
8. **P2 — Continual protocol** (replay → distill → drift → EWC) and E5.
9. **P3 — E4 adaptation**; scale to the 490 pixel store when it exists.

## 12. Open questions for the owner

1. Laptop CPU cores and RAM (sets batch size, k and the wall cap).
2. Who can do blind review, and how many candidates per session (K)?
3. δ_forget and δ_new values for the acceptance gate.
4. Regression-set composition (which farm × year × stage cells).
5. Default wall-time cap per training session.
