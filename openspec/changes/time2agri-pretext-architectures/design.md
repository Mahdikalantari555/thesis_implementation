# Design — time2agri-pretext-architectures

## Decision register

| # | Decision | Basis |
|---|---|---|
| 1 | Implement TD + FP + FF, faithful to Gupta & Sobti 2025 | [DEC] owner, UI 2026-11-26 |
| 2 | 1D cycle tokens first; 2D patch/ViT path later from date-centric raw store | [DEC] owner |
| 3 | Wire into mini_lab for smoke; benchmark registry A11–A13 deferred | [DEC] owner; time cap 15 min this session |
| 4 | First gate = L1 only; no anomaly claim | [DEC] owner |
| 5 | Existing contrastive arm kept + renamed `agri_view_contrastive`, old name = deprecated alias | [DEC] owner |
| 6 | FP spectrum from train-period observed NDVI only | [C] leakage guard, AGENTS §7 |

## Architecture (1D adaptation)

Shared `TinyViT1DEncoder`: patchify `(B,T,C)` into non-overlapping `patch_size=2`
patches → linear projection + CLS token + learned position + optional month/year
temporal encoding → 2 pre-LN transformer blocks → `(cls, seq)`.

- **TD**: sample two patch frames with a known bin gap; gap bucket ∈ 4 classes
  (2/4/6/8+ bins ≈ 30/60/90/120+ d). 3-layer MLP on `[h_a; h_b]` → logits;
  symmetric CE over (a,b) and (b,a). Chance = 0.25.
- **FP**: target = top-3 dominant FFT frequencies (+ amplitudes) of the observed
  NDVI series after Savitzky–Golay smoothing and linear interpolation; 6-d
  normalized target; MLP from CLS + temporal enc → 6; MSE. Input = two frames only
  (target context is wider than the input, matching the paper's bitemporal setup).
- **FF**: first half of the bins = origin, second half = target; 2-layer time
  translator maps origin CLS + target time encoding → d_model; MLP decoder →
  `(T/2, C)`; masked normalized MSE over observed target bins.

## Distinctness

`objective_spec()` returns (objective, target_source, gradient_path, horizon) per
arm: TD = CE / bin-gap class / CLS pair / 0; FP = MSE / FFT spectrum / CLS / full
series; FF = masked MSE / future observed bins / origin tokens / T/2. No arm may
alias another's estimand.

## Provenance and limits

1D token vectors are per-field medians; there is no per-pixel FFT map as in the
paper — documented as a deliberate adaptation, not a claim of equivalence.
FP is computed from each sample's own train-period series; eval data never enters
the target. No thermal channel is present in the paper; LST stays a thesis axis.
