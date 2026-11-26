# Change: time2agri-pretext-architectures

**Change ID:** `time2agri-pretext-architectures`
**Created:** 2026-11-26 · **Status:** PROPOSED → APPLYING (mini phase) · **Owner:** Mahdi
**Source:** owner request 2026-11-26 + UI answers; Gupta & Sobti 2025 (arXiv:2507.04366)

## Why

`src/ssl_anomaly/models/pretexts.py` contains an arm named `time2agri`, but it is
**not** the architecture in the paper: it is a contrastive arm with calendar-shift,
cloud-gap and canopy-jitter augmentations (lines 160–194, 268–279). The paper's
three agriculture-specific pretexts — TD (time-difference classification), FP
(dominant NDVI frequency regression) and FF (future-frame prediction) — are not
implemented. Shipping a mislabeled substitute would violate honest provenance.

## What Changes

1. New faithful, testable implementation of **all three** pretexts on a shared
   tiny ViT-style 1D encoder over the existing `(B,T,C)` cycle tokens (1D first:
   CPU-runnable; 2D patch/ViT path deferred to the date-centric raw store).
2. Each arm exposes its **own** `pretext_loss`; the mini-lab runner stops forcing
   the generic Barlow objective onto representation arms.
3. Existing contrastive arm **kept and renamed** `agri_view_contrastive`; the old
   name remains a deprecated alias so existing manifests/checkpoints still resolve.
4. First test level is **L1 pretext learning only** — convergence, no collapse,
   distinct objectives. No anomaly/stress claim and no benchmark winner claim.

## Non-goals / guardrails

- FP frequency targets MUST be computed from train-period acquisitions only;
  a target derived from eval-period data is leakage and invalidates the arm.
- The paper uses RGB+NIR Sentinel-2 chips with no thermal band; the 1D adaptation
  uses the thesis spectral channels and does not inherit the paper's water-stress
  interpretation.
- "FF prediction error = anomaly signal" is an L1/L2 hypothesis in the paper
  (argued, not measured); it is NOT a stress claim here.
- Benchmark registry ids A11–A13 and the 2D patch path are **deferred**, not absent.

## Capabilities

- New: `time2agri-pretexts` — TD/FP/FF pretext arms with distinct estimands, own
  losses, deterministic seeding and L1 diagnostics.
