# Change: validate-olmoearth-temporal-transfer

**Change ID:** `validate-olmoearth-temporal-transfer`
**Project:** Thesis Implementation
**Status:** Proposed → partially applied (T1/T2 runnable on existing artifacts)
**Priority:** Medium
**Parent evidence:** `evaluate-olmoearth-nano-sugarcane` (COMPLETE 2026-11-27,
commits `cbf8f15`→`b100c01`, REPORT.md)
**Owner request:** scientific review 2026-11-27 (this change is the owner's
explicit proposal, reproduced in their review message)
**Base:** `main` @ `b100c01`

---

## Why

The owner's review of the OlmoEarth Nano evaluation accepted the ADAPT
direction and the reproducibility standard, but identified four issues that
must be resolved before any of its numbers enter the thesis methodology
chapter:

1. **Probe temporal confound (major).** The stage probe gave OlmoEarth real
   timestamps (month embedding, calendar-day RoPE) while the raw-channel
   baseline had only 8 spectral features — and the derived label is itself
   percent-of-cycle. Part of the 0.854-vs-0.627 advantage may therefore be
   *calendar/cycle-time information*, not representation quality.
   Required arms: timestamp-only, cycle-progress-only, raw-only,
   raw+temporal, alongside OlmoEarth.
2. **L2→L1 conversion is numerically, not semantically, justified.**
   SR→DN inversion maps onto the model's DN scale but does not turn surface
   reflectance into TOA reflectance; L2 LST is not L1 brightness temperature.
   The "≤13 %" figure describes DN-scale distance only. Required: same-date
   L1 vs L2 comparison over several scenes/conditions to measure the real
   domain shift.
3. **Cross-year consistency ≠ temporal generalization.** The E2 result shows
   similar phase structure across cycles of the same field; it does not show
   the representation holds on years never used in any evaluation. Required:
   evaluation on 2024–2026 with no adaptation or model selection on those
   years. The owner explicitly authorizes 2024–2026 use for this purpose in
   this change (2026-11-27).
4. **Statistical interpretation.** The 3 probe "seeds" are three subsamples
   of the *training* fields, not independent samples of the field
   population; the reported CI separation is not a full population CI.
   Required: field-level bootstrap intervals everywhere in this change.

Additionally, anomaly detection was evaluated without spatial context
(field-only chips); its re-evaluation is deferred until the date-centric raw
store exists.

## What Changes

- **New isolated validation battery** under
  `experiments/olmoearth_eval/` (same isolation rules as the parent change):
  - **T1 probe-arm battery** (runnable now on existing artifacts): five
    feature arms on identical splits/labels with field-level bootstrap CIs
    and paired deltas.
  - **T2 unseen-year generalization** (runnable now): probe trained on
    pre-2024 train-field bins only, evaluated per-year on 2024–2026 bins of
    dev+test fields; raw-channel arm as control; ≥2024-restricted phase
    consistency as secondary.
  - **T3 L1-vs-L2 domain-shift measurement** (data-gated — the 607 local
    scenes are L2-only; a downloader for public C2 L1 imagery on GCS is
    specified and handed to the owner).
  - **T4 anomaly-detection re-run with real spatial context**
    (Deferred — blocked by `date-centric-raw-store`, which is HELD).
- Amended limitations note in the parent `REPORT.md` pointing here.
- OpenSpec registration (INDEX.md, todo.md).

## Out of scope

- Any fine-tuning or continued pre-training of OlmoEarth.
- Changing the thesis's frozen findings 20–25, the capstone gates, or the
  v2 protocol artifacts.
- Integrating OlmoEarth into the production pipeline or making it the main
  thesis detector (ADAPT verdict stands: representation use only, decision
  on final architecture after this change).
- Any L3+ claim; injected deviations remain synthetic; derived phase labels
  are never presented as biological ground truth (they measure cycle
  progression).

## Impact

- **Affected specs:** new capability `olmoearth-temporal-transfer-validation`.
- **Affected code:** additions only under `experiments/olmoearth_eval/`
  (`temporal_transfer.py`, optional `download_l1.py`); one REPORT.md
  amendment; no production files touched.
- **Decision this change informs** (owner's 3-way question): is OlmoEarth
  (a) just a good temporal encoder, (b) a transferable agricultural
  representation, or (c) genuinely valuable for anomaly detection here.
