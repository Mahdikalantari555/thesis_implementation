# Change: continual-hybrid-representation-design

**Change ID:** `continual-hybrid-representation-design` · **Created:** 2026-11-26 ·
**Status:** PROPOSED (design only, no code in this change's first phase) ·
**Revision:** r2 2026-11-26 00:41 (owner revision request; direction approved) ·
**Owner:** Mahdi · **Source:** owner requirements document
"Research & Architecture Requirements — Next Design Iteration" (2026-11-26, chat),
plus the owner's training directive of 2026-11-26 (train to plateau, always
checkpoint, verbose convergence verdict, laptop CPU only).

## خلاصه‌ی فارسی

این چنج چارچوب پایان‌نامه را از «تشخیص‌دهنده‌ی تنش» به **یادگیری بازنمایی
خودنظارتی و مستمر رفتار طیفی‌ـ‌زمانی نیشکر + کشف آنومالی** تغییر می‌دهد. سه
تصمیم قطعی مالک ثبت می‌شود: (۱) کشف آماری مستقل از مدل + یادگیری فعال، با
قرنطینه‌ی نمونه‌های مشکوک؛ (۲) فنولوژی دومسیره (بی‌نیاز از فنولوژی / آگاه از
فنولوژی با عدم‌قطعیت)؛ (۳) ترکیب پیکسل + زمینه‌ی مکانی. یادگیری مستمر الزام
اصلی است و هر به‌روزرسانی فقط با گذر از Regression Set ثابت پذیرفته می‌شود.
همه‌ی آموزش‌ها روی CPU لپ‌تاپ مالک، تا رسیدن به فلات، با چک‌پوینت دائمی و
گزارش verbose انجام می‌شوند و اگر بودجه تمام شود ولی مدل هنوز بهتر می‌شود،
صریحاً اعلام می‌کند که آموزش بیشتر لازم است.
فاز اول این چنج فقط طراحی است؛ پیاده‌سازی از جلسه‌ی بعد با `tasks.md` شروع می‌شود.

## Why

- Finding 28 (L2, mini) shows a pretrained Mamba-family pixel encoder beats its
  own random init on a synthetic magnitude injection (+0.171 AUPRC, CI
  [0.120, 0.221]) but **not** on real weak labels (SSL = random 0.1108; raw
  0.1782). A stress detector trained against weak METRIC-derived labels is
  therefore not a defensible thesis core.
- The owner and advisor want a **framework**: a representation of sugarcane
  spectral–temporal behaviour and field spatial patterns that generalizes and is
  **updated as new seasons arrive without uncontrolled forgetting**.
- `train/incremental.py` is a stub that raises `NotImplementedError`; there is
  no continual-learning protocol, no regression-set gate, no drift detection.
- The mini training loop (`mini_lab/runner.py`) runs a fixed number of
  epochs/steps; it has no validation-driven plateau stop and cannot say whether
  a run needed more training.
- The canonical 490-field archive carries an off-by-one field-id alias
  (confirmed on 3 farms, `field-id-provenance-fix`); every farm attribute and
  any L4 linkage is invalid until it is resolved archive-wide.

## What Changes

1. **Design document** (`design.md`) answering the owner's ten required
   outputs: architecture + data flow, role of every encoder/head/objective,
   pixel+context fusion, dual-path phenology, statistical discovery + active
   learning, memory + continual protocol, experiments/ablations/metrics, risks
   and unproven assumptions, comparison with the current repository, and a
   prioritized roadmap. Every proposal carries an evidence tag:
   **[R]** research-backed, **[C]** exists in code, **[D]** needs development,
   **[H]** experimental hypothesis.
2. **Five capability specs** in the original design phase (ADDED):
   `training-convergence-control`, `hybrid-pixel-context-representation`,
   `phenology-dual-path`, `anomaly-discovery-triage`,
   `continual-update-protocol`; **r2 adds a sixth**, `input-data-contract`.
3. **Tasks** for the implementation phase (owner implements from next session,
   CPU laptop only, TDD, agent never runs long jobs).

No code, data, split, checkpoint or finding is modified by this change's design
phase. No architecture is declared a winner.

**r2 additions:** separate anomaly head trained only on Verified anomalies,
confidence-weighted and PU learning as experimental arms, versioned anomaly
memory (verified / unresolved / rejected), strengthened anti-confirmation-loop
rules; phenology-agnostic redefined as "no explicit stage labels" while
keeping trajectory and temporal context; multi-signal monitoring and
eligibility-filtered checkpoint selection; Landsat 8/9 input data contract
(units, valid ranges, sensor consistency by measurement, leak-free
anomaly-excluded robust normalization arms, versioning); per-component score
calibration with max / rank / calibrated-weighted aggregation arms; decision
register separating confirmed decisions, hypotheses and open questions.
Landsat 7 harmonization is explicitly out of scope.

## Capabilities

- New: `training-convergence-control` — plateau-driven training, mandatory
  checkpointing and resume, verbose convergence verdict.
- New: `hybrid-pixel-context-representation` — pixel temporal path + masked
  spatial-context path + fusion; objectives as losses, not backbones.
- New: `phenology-dual-path` — phenology-agnostic and phenology-aware scores,
  stage as a probability with confidence.
- New: `anomaly-discovery-triage` — model-independent statistical detectors,
  rank fusion, Normal / Suspected / Verified tiers, quarantine, active learning.
- New: `continual-update-protocol` — versioned updates, replay, distillation,
  EWC (optional), drift detection, regression-set acceptance gate; r2:
  versioned normalization and calibration statistics.
- New (r2): `input-data-contract` — Landsat 8/9 C2 L2 units, valid ranges,
  feature manifest, sensor consistency, masks, leak-free normalization.

## Relation to other changes

- `regional-ssl-representation-evaluation` (capstone, 15/42): **unchanged**.
  This change supplies candidate architectures and protocols to it; its
  pre-registered gates G-A…G-E are not edited here. Any change to them needs a
  separate, pre-run amendment.
- `field-id-provenance-fix`: **hard prerequisite** for any 490-field run
  (archive-wide verification of the alias).
- Owner's planned 490-field **pixel store** change: prerequisite for scaling
  the pixel path beyond the mini lab.
- `mini-research-dataset`: the 3-farm lab (`3farms-dev`: L02-02, R12-08,
  R15-18) is where every arm is first screened on CPU.

## Impact

- **Affected (implementation phase):** `src/ssl_anomaly/mini_lab/` (runner,
  models), new `src/ssl_anomaly/train/convergence.py`,
  `src/ssl_anomaly/continual/`, `src/ssl_anomaly/discovery/`,
  `src/ssl_anomaly/phenology/stages.py`, configs under `configs/mini/`,
  `train/incremental.py` (stub replaced).
- **Frozen:** findings 00–28, `splits_locked_v2.json`, holdout manifest,
  canonical `pixels/` and `zonal_stats/`, capstone pre-registration.
- **Risk:** plateau-based stopping makes compute unequal across arms; the
  design reports both the to-plateau result and an equal-step cut (design §4.5).
