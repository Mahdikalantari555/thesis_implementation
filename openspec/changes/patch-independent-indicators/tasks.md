# Tasks — patch-independent-indicators

## 1. Indicators (TDD-first)

- [ ] 1.1 Physical CWSI module: Idso/Jackson formula, ERA5 Ta join at
      scene time, coefficient registry with sources; unit tests on known
      values; `absent` propagation when a coefficient or Ta is missing.
- [ ] 1.2 Indicator z-score pipeline: per-bin aggregation, phenophase-matched
      robust z, exceedance threshold; unit tests on synthetic series. Both
      z-units are produced and labelled: across-field (population-referenced,
      current behaviour) AND farm-own (per-field spread, with the pooled
      fallback and tier counts of `farm-local-anomaly-scoring`); a report row
      without the unit label fails the build.
- [ ] 1.3 Availability checker: CWSI_ET / TVDI / physical CWSI / ETa-ETrF
      with coverage counts; `absent` semantics test.
- [ ] 1.4 External-consistency module: ERA5 event context (precip deficit,
      Tmax anomaly, ET0, VPD vs historical reference) + neighbor/same-block
      deviation share (block adjacency v1; registry graphs when todo item
      11 lands); alarm classification (local-consistent /
      regional-common-mode / local-unexplained) + candidate causes; unit
      tests on synthetic contexts; `absent` propagation.

## 2. Agreement (runs after gated checkpoints exist)

- [ ] 2.1 Patch part-D adapter: flagged (field, bin) events + per-pixel
      event maps vs indicator exceedance maps; Spearman/enrichment/timing
      with CIs; per-arm table incl. LST-on vs LST-off (E6 dependency) with
      the consistency-vs-corroboration labels.
- [ ] 2.2 Reports + dev_log rows complete (provenance, n, CIs, coverage).

## 3. Verdict

- [ ] 3.1 Persian finding (persian-writing skill): agreement/enrichment/
      timing per arm, the shared-input caveat applied, disagreements
      reported as information, ceiling stated (L3 + proxy; L4 gated on A2).
