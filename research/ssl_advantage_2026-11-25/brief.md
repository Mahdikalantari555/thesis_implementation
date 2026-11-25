# Research brief — what makes an SSL-advantage claim defensible in small-n field-level EO anomaly detection

**Date:** 2026-11-25 · **Depth:** standard · **Decision at stake:** how to design and judge a CPU-only architecture/patch-size sweep on 1–2 sugarcane fields from a 3-field Landsat archive before claiming any self-supervised-pretraining advantage.

## Refined question
Under what conditions do published studies find a genuine advantage from self-supervised pretraining over (a) hand-crafted features/rules, (b) random-init models of the same architecture, and (c) supervised-from-scratch models, for time-series anomaly detection on individual agricultural fields at 30 m Landsat resolution — and which controls, sample sizes and effect reporting are required for such a claim to survive review?

## Scope
**In:** peer-reviewed/primary papers and preprints 2020–2026 on SSL for remote-sensing time series, anomaly/change detection, crop phenology; spatial-unit (pixel/patch/field) comparisons; collapse and evaluation controls; negative results.
**Out:** generic vision SSL without time-series/EO relevance; commercial product claims; non-peer-reviewed blog opinion unless attributed to a named practitioner with a concrete method.
**Assumptions:** 3 separate sugarcane fields, 538 Landsat dates (2013–2026), 30 m pixels, ~25 ha fields, CPU-only training, n ≈ 1–2 fields for the local sweep, small number of independent field-years. The local probe found SSL retrieval above chance but inside a label-permutation null, and SSL probe AUPRC overlapping a random-init control — no advantage established yet.

## Angles
1. **F1 — Evidence for SSL advantage:** which downstream tasks and quantitative gains are actually reported for SSL-pretrained EO time-series encoders, against which baselines, with what n and date ranges?
2. **F2 — Spatial unit:** what do studies say about pixel vs small patch vs field-aggregate representations for 30 m crop monitoring, and what patch sizes are used/justified?
3. **F3 — Defensibility controls:** collapse diagnostics, random-init and rule/feature controls, permutation nulls, pre-registration, and published negative results — what turns a positive number into a defensible claim?

## Output
`REPORT.md` synthesized by theme, cited inline, with open questions. The report informs (but does not predetermine) the local sweep's pre-registration; a positive result is not assumed.
