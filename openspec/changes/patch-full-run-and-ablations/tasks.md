# Tasks — patch-full-run-and-ablations

## 1. Plan + plumbing

- [ ] 1.1 Pre-registered run plan file: every planned run (factor, seeds,
      venue, estimate, output paths) BEFORE the first launch; pruning rule
      (single-seed screening → 3 seeds for finalists) written in.
- [ ] 1.2 Benchmarking fields in train/eval manifests (psutil RAM, CUDA
      VRAM, MACs, per-field-cycle inference); unit test on manifest
      completeness (AGENTS.md §7).
- [ ] 1.3 Scale-up run (user-executed, estimate first): winning PoC config
      on all train fields, seeds ×3; agent reviews produced reports.
- [ ] 1.4 TCN challenger module behind a flag (causal dilated conv, kernel
      3, dilations 1/2/4/8, mask-aware) + causality tests — built ONLY if
      the E3 trigger fires (GRU slow/unstable per the plan's criteria).

## 2. Ablations (each: agent smoke → user runs → agent reviews)

- [ ] 2.1 E2 aux-weight {0, 0.3}; decision: keep aux only if MDM improves.
- [ ] 2.2 E6 LST-off checkpoint (feeds patch-independent-indicators).
- [ ] 2.3 E4 covariates ladder (+weather last, conditioned AND
      unconditioned reported). The ladder gains ONE cell for the
      between-farm differences the owner prompt names: ratoon/cycle kind and
      variety group (already present in `cycles_490.parquet` and used by
      `baseline/robust.py` strata, absent from every model covariate set) —
      added to the patch covariates and compared against the no-stratum arm;
      whether the SCALE should depend on ratoon/variety is separately owned
      by `farm-local-anomaly-scoring`.
- [ ] 2.4 E5 inner-buffer + burn-in variants (the scale/representation sweep
      lives in `compare-spatial-representation-strategies`).
- [ ] 2.5 Sensor-offset estimation script (near-coincident L8/L9 pairs over
      the AOI) + applied/not-applied comparison.
- [ ] 2.6 E3 challengers if triggered: TCN + TinyCNNMamba; record the
      CNN-Mamba title implication for the advisor discussion.
- [ ] 2.7 SSL-off random-init control (same architecture/scoring) +
      optional One-Class SVM arm next to the existing B3 (IsolationForest +
      ECOD); each method's temporal-information level reported (guide §5
      fairness).

## 3. Final decision

- [ ] 3.1 Comparison table: all arms × {acute, persistent, gradual} MDM80,
      Wilson CIs, prevalence, paired-bootstrap arm-vs-reference
      differences; criterion evaluated verbatim.
- [ ] 3.2 Winner frozen (model card + sidecar + hashes) → hands off to
      patch-generalization-suite (L3 matrix) and patch-xai-attribution.
- [ ] 3.3 Error analysis (guide §9 step 8): FP review with weather/neighbor
      context; FN decomposition by family × severity × phenophase; causes
      recorded.
- [ ] 3.4 Persian finding (persian-writing skill): the decision record —
      criterion result, per-cell decisions, benchmark table, error-analysis
      chapter; honest negative fully written if the criterion fails (the
      floor as the finding), L-rung labelling throughout.
