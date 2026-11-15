# Tasks — patch-xai-attribution

## 1. Attribution modules (TDD-first)

- [ ] 1.1 Unit tests: band occlusion (median baseline, Δ map shape/values
      on a synthetic checkpoint), temporal occlusion, gradient×input vs
      finite-difference gradient agreement on a tiny case.
- [ ] 1.2 `src/ssl_anomaly/xai/patch_attribution.py`: the four methods +
      stage aggregation (SUGARCANE_STAGE_BINS reuse); artifact writers
      (band table, temporal table, spatial arrays).
- [ ] 1.3 Faithfulness adapter: deletion/insertion AUC on the patch grid,
      randomization sanity, sign concordance — adapted from
      `xai/faithfulness.py`; pre-registered sample definition (50 events,
      stratified by family/severity, hash-recorded).

## 2. Runs (eval-only, user-executed with estimates)

- [ ] 2.1 Agent smoke on a tiny event set; review; hand over the sampled
      attribution runs (natural alarms from gated checkpoints).
- [ ] 2.2 Attribution artifacts per event + summary tables under reports/.

## 3. Deliverable

- [ ] 3.1 Persian finding (persian-writing skill): which bands/windows/
      spatial patterns drive the scores, faithfulness results (including
      failures), the synthetic-vs-natural framing, L2 wording only.
