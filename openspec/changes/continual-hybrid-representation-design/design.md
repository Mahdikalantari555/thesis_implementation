# Design — continual-hybrid-representation-design

**Date:** 2026-11-26 · **Revision:** r2 (owner revision request 2026-11-26 00:41) ·
**Status:** PROPOSED · **Claim ceiling:** none (design only)

Evidence tags used on every proposal:
**[R]** research-backed (source named) · **[C]** exists in code (file named) ·
**[D]** needs development · **[H]** experimental hypothesis, not yet supported.
Nothing here is a result. No architecture is declared a winner.

Decision status labels (r2): **[DEC]** confirmed owner decision ·
**[HYP]** experimental hypothesis/arm · **[OPEN]** unresolved question.
§13 is the decision register.

**r2 changes:** §1A (input data contract: units, L8/L9 consistency,
normalization) — new; §2.1 (learning streams and data categories) — new;
§2.2 (anomaly score calibration and aggregation arms) — replaces the single
"Anomaly score" paragraph; §4.1 (multi-signal monitoring and checkpoint
selection) — rewritten; §5 Path A — redefined; §6.4 and §6.6–§6.7 (anomaly
head, weak supervision, anomaly memory) — extended; §7.3 item 5 and §7.5
(memories and normalization versioning) — extended; §8 E3/E6/E7 — added;
§12 and §13 — updated.

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

## 1A. Input data contract — Landsat 8/9 units, consistency, normalization

Landsat 8 OLI/TIRS and Landsat 9 OLI-2/TIRS-2 Collection 2 Level-2 only.
**Landsat 7 harmonization is out of scope [DEC]** and is not a requirement.

### 1A.1 Physical units and scale [C partly; D verification]

| Feature | Source | Conversion | Unit | Valid range (after conversion) | Fill / invalid |
|---|---|---|---|---|---|
| SR_B2…SR_B7 | C2 L2 SR | ρ = DN × 0.0000275 − 0.2 | unitless reflectance | ≈ 0 … 1 (DN 7273–43636); slightly negative values allowed only within a pre-registered tolerance, else invalid | DN 0 → invalid |
| ST_B10 (LST) | C2 L2 ST | T = DN × 0.00341802 + 149.0 | Kelvin | ≈ 250 … 345 K plausibility window for this site (product valid DN 293–61440 → ≈150–359 K) | DN 0 → invalid |
| NDVI, NDWI, NDMI | derived from converted ρ | normalized difference | unitless | −1 … 1; denominator ≈ 0 → invalid | propagated |
| CWSI, TVDI | derived (METRIC/LST) | as defined in `docs/methods_explained.md` | unitless | 0 … 1 (clip flagged, not silent) | propagated |
| ETa | METRIC | — | mm day⁻¹ | 0 … site maximum (pre-registered) | propagated |
| Weather covariates | station / ERA5 | — | documented per variable (°C, mm, m s⁻¹, kPa) | per variable | NaN + mask |
| Δt, DOY | dates | — | days | Δt ≥ 1; DOY 1…366 | — |

- The scale factors, offsets, valid DN ranges and fill value above match the
  USGS *Landsat 8-9 Collection 2 Level 2 Science Product Guide* (LSDS-1619
  v6.0, 2024, table of band designations: SR UINT16 valid 7273–43636, fill 0,
  ×0.0000275 −0.2; ST_B10 UINT16 Kelvin valid 293–61440, fill 0,
  ×0.00341802 +149) [R]. **Only the four scale/offset constants are pinned in
  code** (`config.py::IngestCfg:124-127`) [C]; the valid DN ranges (SR
  7273–43636, ST 293–61440) and fill value 0 are **not** pinned anywhere —
  the literals are duplicated in `io/landsat.py`, `io/zonal.py` and
  `scripts/extract_patches.py` [D]. Task 0A.1 consolidates all of them in one
  place, adds a unit test against the guide, and asserts that no code path
  re-applies them to already-converted data.
- The mini builder reads METRIC scene rasters described as "reflectance 0..1
  and LST Kelvin, already scaled" (`scripts/build_mini_dataset.py`) [C]; task
  0A.2 asserts this with a range check, so raw DNs can never be mixed with
  converted values (a DN-scale value is ≫ 1 and fails the check).
- Every store, cache and model input carries a **feature manifest**: name,
  unit, conversion applied (yes/no and which), valid range, source product
  [D]. A loader refuses features whose manifest is missing or whose observed
  range violates it beyond a tolerance.
- QA: QA_PIXEL bits (fill, dilated cloud, cirrus, cloud, cloud shadow, snow)
  are already configured [C, `config.py::QualityCfg`]; QA_RADSAT
  (saturation) and ST_QA (LST uncertainty) are added as quality signals [D].
  ST_QA becomes an optional per-observation uncertainty weight [HYP].

### 1A.2 Landsat 8 vs Landsat 9 consistency

- One product guide covers both sensors with identical band designations,
  units and scaling (LSDS-1619) [R]; the pipeline fixes one band order and
  asserts it per scene (task 0A.3).
- Literature: L8/L9 underfly cross-calibration found band-level differences of
  about 1 % [R: Gross et al. / underfly studies, summarized in GIScience &
  Remote Sensing 2024, doi 10.1080/15481603.2024.2318071]; over China
  landscapes, SR RMSE 0.010–0.025 for VNIR/SWIR and NDVI RMSE 0.042 [R:
  Remote Sensing 15:4948, 2023]; differences grow slightly in multiband
  products (+0.2–0.5 percentage points, JAG 2023) and are larger in dark/bright
  targets due to 14-bit vs 12-bit quantization [R]. These are small relative to
  stress signals of interest but **not negligible** relative to a subtle
  anomaly score, hence the measured decision below. Exact numbers are to be
  re-read from the papers before they are quoted in the thesis [OPEN].
- Sensor id is parsed **per scene** (`io/landsat.py` parses LC08/LC09 [C]);
  `io/patches.py` stores a **majority sensor per bin**, not per observation [C];
  the mini store does not carry it [D]. Task 0A.3 therefore adds a
  per-observation `sensor_id` channel rather than surfacing an existing one.
- Systematic difference is **measured, not assumed**: `io/sensor_offset.py`
  [C] estimates per-band median offsets from near-coincident L8/L9 pairs
  (≤ 3 days) with MAD CIs and refuses unresolved offsets. Decision rule
  [DEC, reusing the existing code's rule]: apply a correction only if it is
  resolved (CI excludes 0) **and** reduces the sensor-era score gap by ≥ 25 %
  without raising the L8-only alarm rate; otherwise no transformation.
- Arms [HYP]: (a) no correction, (b) offset correction, (c) sensor id as an
  input token. Chosen by E7 on validation only.

### 1A.3 Normalization [HYP arms; DEC on the constraints]

Constraints [DEC]:
- statistics from the **training split only** (time ≤ 2022 and training farms
  only), never from validation/test years or held-out farms;
- computed on **Normal-confident, QA-valid** observations; Suspected and
  Verified-anomaly observations are excluded so anomalies cannot distort the
  reference;
- **missing/invalid values stay missing**: an observation mask channel is
  always carried; no NaN→0 filling without the mask (finding of the real490
  review);
- normalization is invertible and versioned (§7.5).

Arms compared in E7 (no arm is presumed best):
| Arm | Definition | Expected strength | Risk |
|---|---|---|---|
| N0 z-score per feature | (x − μ)/σ global | simple baseline | σ inflated by outliers/clouds |
| N1 robust per feature | (x − median)/(1.4826·MAD) or IQR | outlier-resistant | small MAD on flat features |
| N2 physical-scale fixed | reflectance kept in ρ units (optionally ×k), LST as (T − T_ref)/s with fixed physical constants | preserves magnitude, no data dependence, trivially stable across updates | features on different scales |
| N3 robust + clipped | N1 then clip to ±c, clip rate logged | bounded inputs | hides extreme real anomalies if c too small |
| N4 per-field or per-pixel centring | subtract training median of that field/pixel | removes static soil/variety offsets | can remove real persistent anomalies; leak risk if computed on test years |

- Indices (bounded −1…1) are left unnormalized or only scaled [HYP].
- Any arm must keep meaningful magnitude changes detectable: the
  augmentation-erasure style test (§8) is repeated as a "normalization
  erasure" test.

### 1A.4 Missing values and masks
- Inputs: value channel (finite or 0 placeholder) **plus** a binary
  observation mask per feature; Δt token carries irregular sampling.
- Losses are computed only on observed entries; masked-out entries never
  enter reconstruction targets or normalization statistics.
- A window with fewer than `min_obs` valid observations is excluded and
  counted.

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
| Anomaly memory | verified anomalies + unresolved Suspected (separate partitions), versioned | [D] | used for the anomaly head (Verified only), active learning and evaluation; never for encoder SSL (§6.7) |
| Anomaly head | separate small head on the (frozen or slowly updated) encoder that learns from Verified anomalies | [D] | weak-supervision arms are [HYP] (§6.6) |

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

### 2.1 Learning streams and allowed data categories [DEC]

Four streams with **separate protocols**; a stream may only use the data
categories marked ✓.

| Data category | Encoder SSL (representation) | Normal memory | Anomaly discovery (scoring) | Anomaly head (adaptation) | Calibration (score normalization) | Evaluation |
|---|---|---|---|---|---|---|
| Normal-confident (QA-valid, all sources low) | ✓ (confidence-weighted) | ✓ | ✓ | ✓ as negatives | ✓ | ✓ |
| Unscreened unlabeled (before triage, training years) | ✓ only in the first pretraining pass, with robust/trimmed loss [HYP] | ✗ | ✓ | ✓ as **unlabeled** in PU arms only [HYP] | ✗ | ✗ |
| Suspected (quarantined) | ✗ | ✗ | ✓ | ✓ as **unlabeled, down-weighted** in weak-supervision arms only [HYP]; never as positives | ✗ | ✓ (unresolved count) |
| Unsure after review | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ |
| Verified anomaly (train part) | ✗ | ✗ | ✓ | ✓ as positives | ✗ | ✗ |
| Verified anomaly (held-back eval part) | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ only |
| Verified normal (reviewed) | ✓ | ✓ | ✓ | ✓ as negatives | ✓ | ✓ |
| Synthetic injections | ✗ for SSL; ✓ only in explicitly labelled synthetic arms | ✗ | ✓ | ✓ only in synthetic arms, never mixed with real positives | ✗ | ✓ labelled `synthetic-injection` |
| Statistical detector outputs | ✗ | ✗ | ✓ as **evidence** | ✗ as targets | ✗ | ✓ agreement only |
| Model's own scores / pseudo-labels | ✗ | ✗ | ✓ | ✗ as targets | ✗ | ✓ stability only |

- **Encoder SSL protocol:** self-supervised objectives only; no anomaly labels,
  no detector outputs; data = Normal-confident (+ verified normal), weighted by
  normal confidence.
- **Anomaly head protocol:** encoder frozen (default) or partially unfrozen in
  the E4 arms; trained on Verified positives vs Normal-confident/verified
  normal negatives; weak-supervision arms add unlabeled data (§6.6).
- The anomaly head's gradients never update the encoder in the default arm;
  an arm that lets them is [HYP] and must pass the forgetting gate (§7.4).

### 2.2 Anomaly score calibration and aggregation [HYP arms]

Raw residual magnitudes from different objectives are **not comparable**
[DEC]. Each component c ∈ {recon residual, forecast residual, JEPA latent
error, memory kNN distance} is calibrated before fusion:

1. **Per-component reference distribution** from the calibration set:
   Normal-confident observations of training years, held out from the data the
   encoder was trained on (separate calibration farms/windows where possible).
2. **Conditioning** [HYP arms]: none / per-field / per-DOY-window / per
   observation-count class (sparse windows have larger residuals).
3. **Calibrated value**: empirical CDF → upper-tail p-value (default) or
   robust z = (r − median)/(1.4826·MAD); conformal p-values as an arm.
4. Calibration statistics are versioned with the model (§7.5).

Aggregation arms (all kept, none presumed):
| Arm | Rule | Note |
|---|---|---|
| AG-max | max over calibrated components (or min p, with Bonferroni/Simes correction) | sensitive to one strong signal; more false alarms |
| AG-rank | mean (or reciprocal-rank) of per-component ranks | robust to scale; loses magnitude |
| AG-wcal | Σ w_c·z_c with weights fit on a **calibration objective that uses no test data** — e.g. maximizing separation on Verified-train only, or inverse-variance / stability weights when no verified set exists | needs a verified set for anything beyond stability weights |
| AG-single | each component alone | ablation baseline |

Score stability evaluation: rank correlation (Spearman/Kendall) of the top-K
across seeds, across adjacent checkpoints, under observation dropout (10/30 %)
and under phenology shifts; Jaccard overlap of top-K events. Final ranking
validation: only against the held-back Verified set and blind review
(Precision@K, AP) when it exists; otherwise stability and coherence only
(§6.5). Barlow never contributes a score term [DEC].

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
- **r2: a lower weighted total loss is not evidence of a better
  representation** [DEC]. Each evaluation logs a monitoring vector:
  - **objective signals** — each validation loss separately (recon, forecast,
    JEPA, Barlow on/off-diagonal terms), never only their sum;
  - **collapse signals** — effective rank (entropy of normalized singular
    values) and per-dimension std are already implemented [C
    `benchmarking/collapse.py:89-130`]; off-diagonal covariance energy
    (VICReg-style) and JEPA target–prediction variance are **new monitors** [D]
    to be added by task 1.1b;
  - **frozen-encoder quality** (every `probe_every` evals, cheaper than
    eval): linear-probe R² for DOY/season-phase and NDVI-next-obs,
    kNN retrieval of same-field-same-phase windows, and L1-a gap reconstruction
    from the frozen latent; all on the validation split with a fixed small
    probe budget;
  - **stability** — CKA/linear-CKA between the current and previous probe
    checkpoints, and top-K score-rank correlation (§2.2).
- **Plateau (r2)** is declared on a **guard set**, not on the total loss:
  each objective signal and the primary probe metric are tracked with the
  same rule (relative improvement > `min_rel_delta`, default 0.5 %, for
  `patience` evals, default 8, plus EMA slope over `window`, default 10).
  Stop when **all** guard signals have plateaued, or when the primary probe
  metric has plateaued and no objective is still improving beyond threshold.
- **No hiding:** if any objective degrades by > `max_obj_regress` (default
  5 % relative to its best) while the total improves, the evaluation is
  flagged `OBJECTIVE_CONFLICT`, printed, and that checkpoint cannot become
  `best.pt`.
- **Checkpoint selection (r2):** `best.pt` = best primary probe metric among
  checkpoints that pass the eligibility filter (no collapse flag, no objective
  conflict, every objective within `max_obj_regress` of its best). Also kept:
  `best_loss.pt` (lowest total loss, for diagnosis) and `last.pt`. Which
  probe metric is primary is pre-registered per experiment [OPEN default:
  linear-probe R² on season phase].
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
| `OBJECTIVE_CONFLICT` (flag, not a stop) | one objective regresses > `max_obj_regress` while total improves | printed; checkpoint ineligible for `best.pt`; persistent conflict over `patience` evals stops the run for owner review |

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
- One line per eval: step, epoch, train loss (and components), each objective's val loss, total val loss, best, primary probe metric (when run), eligibility flags,
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

- **Path A — phenology-agnostic (r2 definition)** [DEC definition; D]:
  *agnostic* means **no explicit stage labels and no dependence on a correctly
  extracted phase** — it does **not** mean comparing observations from
  different growth stages without temporal context (that definition in r1 is
  withdrawn).
  - The score is conditioned on the **pixel's own trajectory**: the residuals
    of recon / forecast / JEPA are computed on windows that keep the temporal
    order, Δt and the preceding observations, so "what is normal next" is
    learned from the trajectory itself.
  - The **reference** for memory distance and for calibration is the pixel's
    (or field's) **own history at comparable temporal context**, found
    implicitly: nearest neighbours in the encoder's latent of the preceding
    window (trajectory-matched analogues), plus same-pixel prior years matched
    by DOY window and by days-since-last-detected-harvest when available —
    using these as soft context features, never as hard stage labels.
  - Allowed implicit temporal inputs: DOY (sin/cos), Δt, time since series
    start, days since last large NDVI drop (a raw signal, not a stage label),
    shared weather. Each is an ablation item [HYP] in E1.
  - Works when phenology extraction is missing, incomplete or wrong, because
    nothing in Path A reads the extracted stage.
- **Path B — phenology-aware** [D/H]: a stage posterior p(s | window) over
  literature stages {tillering, grand growth, maturity/ripening} plus
  fallow/harvest, with a confidence c. Score = Σ_s p(s)·dev_s, where dev_s
  compares against the stage-conditioned normal memory. If c < c_min, Path B
  defers to Path A and records it.
- Stage estimation: Joshi et al. 2024 pipeline (BISE → 14-day resample → FFT →
  DoG troughs; FAO two-crop growth stages) [R: Joshi 2024, finding 13];
  plant/ratoon from the **corrected** age attribute (after the id fix).
  Uncertainty from the SOS/EOS RMSE (17–27 days) → soft stage boundaries.
- The existing four DSH-fraction phases (`PHASE_ORDER` = germination,
  tillering, grand growth, maturation; `phenology/mini.py:124`) stay as a
  pre-registered comparator arm [C]. "S1–S4" is **not** a phenology label in
  code — it denotes severity levels in findings — so the comparator's naming
  is decided before E1 [OPEN].
- **Combination [HYP]:** Path C = calibrated fusion of A and B (§2.2 arms),
  with B's weight scaled by stage confidence c. A, B and C are evaluated
  independently and in combination (E3).
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
  and from SSL updates until resolved. They are stored in the anomaly memory's
  *unresolved* partition (§6.7) with their evidence vector, tier history and
  the model version that flagged them.
- Tier thresholds are pre-registered per release; an item can move only
  Suspected → Verified/Normal/Unsure by review or independent record, never by
  a model or detector score alone [DEC].

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
- **r2 — explicit anti-confirmation-loop rules [DEC]:**
  1. No model score, detector score, or pseudo-label is ever written into a
     label field; labels come only from review or independent records.
  2. A sample flagged by version v and reviewed is used to train the anomaly
     head of version v+1 only if it was reviewed blind; its review packet
     records whether it was model-selected or randomly selected.
  3. Active-learning selection bias is measured: Precision@K on randomly
     selected reviewed items vs on model-selected items is reported
     separately; positives are never only model-selected (fixed random share).
  4. Weak-supervision arms (§6.6) treat uncertain candidates as *unlabeled*
     with bounded weight, never as positives; their outputs never re-enter the
     triage as evidence for the same items (no circular evidence).
  5. The statistical detectors are frozen per release and never fitted to
     model outputs; the model is never trained to predict detector outputs
     (no distillation from detectors).
  6. Audit: per update, the fraction of anomaly-head positives that were first
     surfaced by the model itself is logged; a rising share with flat random
     precision triggers an owner review.

### 6.5 Evaluation unit and claims
- Units: **Pixel × time window** and **Spatial Anomaly Event** (connected
  in-field pixels flagged on ≥2 consecutive acquisitions).
- Without a valid verified set: report only stability, cross-detector
  agreement, temporal/spatial coherence, cross-year recurrence and robustness
  to missing data — **no Precision/Recall/"stress detected"**.
- Precision@K only on a valid blind-reviewed set.

### 6.6 Anomaly learning — separate anomaly head and weak supervision

- **Separation [DEC]:** representation learning (encoder, SSL, §2.1) and
  anomaly learning (anomaly head) have separate data, losses, schedules and
  version numbers. The anomaly head never trains before at least
  `min_verified` (pre-registered, [OPEN] default 20 events across ≥ 2 farms
  and ≥ 2 years) Verified positives exist; until then the system is discovery
  only.
- **Head [D]:** small MLP (or logistic) on frozen encoder features + calibrated
  score components; output = anomaly probability with calibration
  (Platt/isotonic on held-out Verified).
- **Learning arms [HYP]:**
  - H-sup: Verified positives vs Normal-confident negatives, class-weighted.
  - H-cw: confidence-weighted — negatives weighted by normal confidence;
    Suspected excluded.
  - H-pu: Positive-Unlabeled learning (nnPU) with Verified as P and
    Suspected + unscreened as U; class prior π estimated (e.g. from the
    random-review share) and varied in a sensitivity sweep. Output never
    becomes a label.
  - H-none: no head; discovery score only (baseline).
- **Evaluation:** only on the held-back Verified set and blind review; with
  fewer than `min_verified` held-back events, report "insufficient labels",
  not a metric.

### 6.7 Anomaly memory [D]

- Separate from normal memory; versioned with the model (§7.5).
- Partitions: **verified** (event id, pixels, dates, evidence vector, reviewer,
  independent source, review date), **unresolved** (Suspected and Unsure, with
  evidence and the flagging model version), **rejected** (reviewed as normal;
  kept to audit false positives).
- Uses: training positives for the anomaly head (verified, train part);
  active-learning pool (unresolved); evaluation (verified, held-back part);
  exclusion list for normal memory, replay and normalization statistics
  (verified + unresolved).
- Never used for encoder SSL.

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
   only from Normal-confident; anomaly memory holds Verified plus the
   unresolved/rejected partitions (§6.7).
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

### 7.5 Versioning of normalization and calibration statistics [DEC]

- Each model version binds: normalization statistics (arm, values, the data
  hash they were computed on), per-component calibration distributions
  (§2.2), sensor-offset table (if applied, §1A.2), feature manifest (§1A.1).
- **Default on update:** normalization statistics are **frozen** at the
  parent's values so historical and new observations stay in one input space;
  new statistics are only computed and compared, not applied.
- **Re-normalization** is allowed only when a drift report (§7.3) shows a
  persistent input shift; it creates a new major version, and the regression
  set is re-scored under the new statistics before acceptance. Encoder
  re-adaptation after re-normalization follows the same acceptance gate.
- Calibration statistics **are** recomputed per version (residual scales change
  with weights) on the calibration set, and both old and new are stored.
- Old scores are never silently re-labelled: re-scoring history under a new
  version writes a new score table with its version id.

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
| E3 | Real anomaly discovery | best 2–3 encoders from E1/E2 × Path A/B/C × statistical detectors × anomaly-head arms (H-none/H-sup/H-cw/H-pu, only once `min_verified` is met) | stability across seeds; cross-detector agreement; spatial/temporal coherence; cross-year recurrence; Precision@K **only** with blind review; random-vs-model-selected review precision | detector subsets; fusion rule; PU prior π sweep |
| E4 | Adaptation | frozen encoder, linear probe, partial FT, full FT | stage probe, retrieval, few-shot head on verified set (if any) | label count {1,2,4,8,all cycles} |
| E5 | Continual learning | scratch, warm start, warm + replay, warm + replay + distill, (+EWC) | BWT, forgetting, new-data perf, wall time, acceptance-gate pass rate | replay fraction; distill weight; frozen vs re-computed normalization |
| E6 | Score calibration and aggregation | AG-single ×4, AG-max, AG-rank, AG-wcal; conditioning arms | top-K stability (seeds, checkpoints, dropout, phenology shift); synthetic injection AUPRC (labelled synthetic); Precision@K only with verified set | calibration distribution (p-value vs robust z vs conformal) |
| E7 | Input pipeline | normalization N0–N4 × sensor handling (none / offset / sensor token) | L1-a/L1-b, probe metrics, L8/L9 era score gap, normalization-erasure test, clip rate | indices normalized vs raw; ST_QA weighting |

**Monitoring (r2):** every experiment reports per-objective curves,
effective rank, probe metrics and the `best.pt` eligibility log, not only the
total loss.

**Sanity tests with known answers:** permuted temporal order must hurt
L1-b; a DN-scale feature must be rejected by the loader; L8-only vs L9-only
windows of the same field and phase must not differ beyond the measured
sensor offset; random-init must sit near DOY climatology on L1-c; an S6-strength
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
| Training loop | mini: fixed epochs/`max_steps`, `last.pt` per epoch [C]; 490: Lightning with `early_stop_patience` on loss [C] | multi-signal plateau, eligibility-filtered `best.pt`, per-objective curves, probes, verdicts, wall-time checkpoints, RNG/sampler state [D] |
| Units / scale | scale/offset constants in `config.py::IngestCfg:124-127` [C]; mini builder reads pre-scaled METRIC rasters [C]; DN ranges and fill **absent** [D] | pin all constants in one place, unit test, range assertions, feature manifest [D] |
| L8/L9 | `io/landsat.py` per-scene sensor parse, `io/patches.py` majority-per-bin `sensor_id`, `io/sensor_offset.py` measured offsets with refusal rule [C] | per-observation `sensor_id` in mini store; E7 arms [D] |
| Normalization | mini: global mean/std over finite training values (`mini_lab/data.py:422-449`) [C]; 490: `norm_stats_full.json` [C] | robust arms, anomaly-excluded statistics, versioning and freeze-on-update [D] |
| Anomaly head / PU | none | §6.6 [D] |
| Score calibration | robust-z in `monitor/indicators.py:151` and `benchmarking/calibration.py:95`; `monitor/scoring.py` uses q95/causal-median normalizers [C] | per-component calibration + AG arms [D] |
| Phenology | `phenology/cycles.py` (harvest troughs), `phenology/mini.py` (four named DSH-fraction phases, `PHASE_ORDER:124`) [C] | literature stages with uncertainty [D]; owner visual check pending |
| Field-id mapping | mini alias `archive_fid = reference_fid + 1` verified on 3 farms [C] | archive-wide verification on 490 [D] |
| Synthetic vs real separation | finding 28 separates them by text [C] | enforce via report label field [D] |

---

## 11. Prioritized roadmap (implementation, next sessions)

0. **P0 — Input data contract** (§1A): pinned constants test, range
   assertions, feature manifest, sensor id in the mini store, mask channel.
1. **P0 — Trainer** (`training-convergence-control`): multi-signal plateau +
   eligibility-filtered checkpoint selection + verdicts + checkpoints +
   verbose; retrofit mini runner and sweep script. Unit tests with synthetic
   curves (plateau, still-improving, diverge, collapse, objective conflict).
2. **P0 — Hygiene**: de-duplicate `_cnn_mamba`; fix JEPA/Barlow full-field;
   mask + Δt tokens; capacity matching.
3. **P1 — Archive-wide id verification** (extend `field-id-provenance-fix`).
4. **P1 — E1 + E2 on mini** (CPU, owner runs, to plateau).
5. **P1 — Statistical detectors + tiers + quarantine + anomaly memory**
   (no model needed); score calibration and AG arms (E6); normalization and
   sensor arms (E7).
6. **P2 — Phenology Path B** after the owner's timeline check.
7. **P2 — E3 discovery report + blind-review packet** (owner/expert); the
   anomaly head (§6.6) only after `min_verified` is met.
8. **P2 — Continual protocol** (replay → distill → drift → EWC) and E5.
9. **P3 — E4 adaptation**; scale to the 490 pixel store when it exists.

## 12. Owner questions and answers (2026-11-26)

Items 3 and 4 stay **[OPEN]** by the owner's explicit decision to defer.

1. **CPU/RAM — answered (measured in this workspace):** Fedora Linux, 12th-gen
   Intel Core i7-12700H, 14 physical / 20 logical cores, 23 GiB RAM (~18 GiB
   available), torch sees no GPU (`cuda=False`); micromamba env `geo` is Python
   3.11.18. The owner confirmed this is their **only** system (no Windows, no
   dual boot, no WSL), so old Windows-only paths in `AGENTS.md` are stale.
2. **Blind review — answered:** owner only, K=10 candidates per session, with
   the fixed random-Normal mix-in required by §6.3/§6.4.
3. **δ_forget / δ_new — OPEN:** owner chooses to decide **after E5**, from the
   measured baseline spread; the gate (§7.4) cannot be pre-registered until then.
4. **Regression-set composition — OPEN:** owner chooses to decide **after E1**;
   the set must still be hash-frozen before E5.
5. **Wall cap — answered with a new run-placement directive:** default session
   cap 30 min. The design must be CPU-friendly — do not use all cores, handle
   RAM carefully, and do not run hot/overwhelming jobs for long. Longer runs,
   even beyond 2 h, are available **only with owner permission** at session
   endpoints or where results are needed to continue, and only after the local
   gate plus a successful smoke test. This supersedes the old ≤15 min/seed and
   Colab-first defaults for this single-machine setup; `AGENTS.md` must be
   amended to match (owner approved 2026-11-26).
6. **min_verified — answered:** 20 Verified events across ≥ 2 farms and ≥ 2
   years (the proposed default); below that, discovery-only.
7. **Primary probe metric — answered:** linear-probe R² on season phase.
8. **Independent record — answered:** none available (no irrigation logs, ECe
   or yield records) beyond owner blind review; claims therefore stay capped
   at **L3**, as §9 already states.

**Cycle-timeline check (task 0.5) — recorded, but verdict negative:** the owner
found missed cycles (`todo.md:44-64`,
`docs/audits/cycle_gap_diagnosis_2026-11-26.md`). Owner answers 2026-11-26:
L02-02 gaps stay **unresolved** (flagged, no forced harvest); R15-18 shows
**multiple** missed cycles (two peaks in 2019–2020, one detected, plus the
2018→2019 cycle missed, apparently from low NDVI peaks) — a detection defect,
not one disputed boundary; the `overlong_unresolved` flag is **approved**.
Path B, stage mapping and the cycle-correctness claim remain blocked until
cycle detection is repaired and re-reviewed.

## 13. Decision register (r2)

**Confirmed decisions [DEC]** (owner requirements 2026-11-26 and revision r2):
- Framework = SSL + continual representation learning + anomaly discovery, not only a stress detector.
- Hybrid statistical discovery + active learning; Normal-confident / Suspected / Verified tiers; quarantine; no confirmation loop (§6.4 rules 1–6).
- Separate protocols: encoder SSL vs anomaly head (§2.1 data table).
- Dual-path phenology with the r2 definition of phenology-agnostic (§5).
- Hybrid pixel + spatial context; pixel is the scoring unit.
- JEPA and Barlow are objectives, not backbones; Barlow never scores.
- Continual updates are batched and gated by a fixed regression set; forgetting bounded and measured, not promised zero.
- Laptop CPU training to plateau, mandatory checkpoints, verbose verdicts; plateau and checkpoint selection on multiple signals (§4.1 r2).
- Raw residuals are calibrated before fusion (§2.2).
- Landsat 8/9 C2 L2 only; no Landsat 7 harmonization; units and ranges contract (§1A.1); train-only, anomaly-excluded normalization statistics; masks always carried.

**Experimental hypotheses / arms [HYP]:** Mamba-lite vs GRU vs Transformer; fusion type and k; head/loss count; aggregation AG-max/AG-rank/AG-wcal; calibration form; normalization N0–N4; sensor handling; anomaly-head arms H-sup/H-cw/H-pu; PU class prior; Path C fusion; CL mechanisms (replay, distillation, EWC); unscreened data in the first pretraining pass; ST_QA weighting; that SSL pretraining helps real anomaly discovery at all (currently unsupported, finding 28).

**Confirmed 2026-11-26 (owner UI answers):** owner-only blind review with K=10;
no independent agronomic record exists, so L4 is unavailable and claims cap at
L3; `min_verified` = 20 events / ≥2 farms / ≥2 years; primary probe = linear
R² on season phase; 30 min default session cap with CPU-friendly, RAM-aware,
non-saturating design, and longer runs only with owner permission after the
local gate (§12 items 1, 2, 5–8). The cycle-timeline visual check is recorded but its verdict is negative
(three missed cycles; `todo.md:35-52`; §0 task 0.5), so Path B stays blocked.

**Unresolved [OPEN]:** §12 items 3–4 (δ values after E5; regression-set cells
after E1, frozen before E5); archive-wide field-id alias (490); amendment of
the stale `AGENTS.md` environment and run-placement rules; whether
METRIC-derived CWSI/ETa/TVDI are independent enough of model inputs to count
as evidence.

