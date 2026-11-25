# hybrid-pixel-context-representation Specification

## Purpose

Learn per-pixel spectral–temporal representations that use in-field spatial
context without averaging away small anomalies.

## ADDED Requirements

### Requirement: Pixel is the scoring unit; context conditions it

The encoder SHALL produce a per-pixel, per-observation latent from a pixel path
(per-observation tokens carrying bands, observation mask and Δt) and a context
path (shared 2D CNN over a k×k in-field-masked neighbourhood, k ≤ 11), fused
by a configurable fusion (gated add, FiLM or concat; early or late).

#### Scenario: Out-of-field pixels cannot leak
- **WHEN** values outside the farm mask are perturbed
- **THEN** in-field latents and scores are byte-identical

#### Scenario: Small anomalies survive
- **WHEN** a synthetic anomaly with a 1-pixel footprint is injected
- **THEN** the hybrid arm's detection is reported next to the pixel-only arm, and a hybrid that falls below the pre-registered 1-pixel sensitivity is rejected

### Requirement: Objectives are losses, not backbones

JEPA and Barlow Twins SHALL be implemented as objectives on a shared backbone;
Barlow Twins SHALL NOT contribute an anomaly-score term; the number of heads
and losses SHALL be chosen by ablation.

#### Scenario: Barlow-only arm
- **WHEN** an arm is trained with the Barlow objective alone
- **THEN** its anomaly score comes from reconstruction/forecast heads or memory distance trained on the frozen encoder, never from the Barlow loss

### Requirement: Augmentations preserve real change

View augmentations SHALL be limited to observation dropout, temporal
sub-sampling, Δt jitter within one revisit and sensor-level noise, and an
augmentation-erasure test SHALL confirm that a synthetic NDVI drop stays
separable after augmentation.

#### Scenario: Erasure test fails
- **WHEN** an augmentation makes injected and clean windows inseparable
- **THEN** that augmentation is disabled and the result recorded

### Requirement: Capacity-matched comparisons

Compared arms SHALL lie within a ±20 % parameter band or report the mismatch,
and duplicate registry entries (e.g. identical `_mamba` / `_cnn_mamba`) SHALL
be removed or made distinct before comparison.

#### Scenario: Registry duplicate
- **WHEN** two registry names build the same module under a config
- **THEN** the sweep refuses to run them as separate arms
