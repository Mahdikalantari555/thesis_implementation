# Tasks: data-and-phenology-readiness

## 1. Literature review
- [ ] 1.1 Reuse `thesis_research/SSL_Proposal` paper matrix; list what already covers D1 sections 1–4
- [ ] 1.2 Search and read new sources for gaps (phenology extraction, time axis, sugarcane stress signatures, SSL on irregular series)
- [ ] 1.3 Write `docs/findings/13_literature_phenology_and_perturbations.md` (per-paper table + implications + crop-generalisation note)
- [ ] 1.4 Perturbation catalogue with sources; mark assumptions; link to advisor question (todo OQ6)
- [ ] 1.5 Add plain-language sections to `docs/methods_explained.md` for the methods adopted

## 2. Cycle audit
- [ ] 2.1 `scripts/audit_cycles.py`: sanity flags for all cycles, stratified sample (seed recorded), per-cycle plots
- [ ] 2.2 Agent pre-labels sampled cycles; user confirms 20 on the platform
- [ ] 2.3 Write `docs/findings/14_cycle_audit.md`: error rate with Wilson CI, per class, per year; apply D2 decision rule
- [ ] 2.4 If planting/harvest records become available: segmentation error in days (optional, non-blocking)

## 3. Data quality
- [ ] 3.1 `scripts/data_quality_report.py` → `reports/data_quality.json` (observation density, valid fraction, LST range, NaN fractions, 538 vs 539 dates)

## 4. Decision record (gate)
- [ ] 4.1 Add the decision record to `train-and-finalize-main-architecture/design.md` (time axis, segmentation params, perturbation set, exclusion rule), citing sections of 13/14
- [ ] 4.2 User approval of the decision record
