# Tasks — anomaly-definition-path-b

## 1. Family machinery (verdict-independent, TDD-first)

- [ ] 1.1 Relation model: phenophase-binned robust fits (NDVI→LST;
      swir1|NDVI) on train cycles only, frozen + sha; leakage tests.
- [ ] 1.2 Relational appliers: decoupled NDVI-drop (LST held/ε-lowered) and
      band-relation break, severity in relation-residual MAD units,
      field-level AND patch-tensor appliers; unit tests on profiles.
- [ ] 1.3 Phenology-shape appliers: green-up delay {2,4} bins, grand-growth
      compression {20,40}%, early senescence {2,4} bins; clip-to-admitted-
      bins; covariate consistency tests.
- [ ] 1.4 Gate file: the both-gates-failed condition encoded and tested
      (tasks §2 refuse to run without it).

## 2. Evaluation (HARD-GATED on both path-C failures)

- [ ] 2.1 Injection sets + hashes for the new families; B0/B1-class
      references recomputed in the same runs.
- [ ] 2.2 User-run sweeps through the unchanged protocol; per-pixel
      localization for relational families on patch scorers.
- [ ] 2.3 Persian finding (persian-writing skill): does the deep advantage
      appear on relational/shape anomalies where the document and
      deep-research predict it; anomaly-vs-stress wording discipline; L2
      ceiling; verdict = finding + owner decision, not silent adoption.
