# external-foundation-model-evaluation Specification

## Purpose

Define the protocol for evaluating an external geospatial foundation model
(OlmoEarth v1.2 Nano) on the thesis sugarcane archive with frozen embeddings:
provenance, adapter correctness, artifact schema, evaluation batteries
(representation, probing, anomaly detection vs baselines, robustness, CPU
feasibility), and the evidence-based adopt/adapt/reject verdict. Evaluation
only — this capability never modifies the production pipeline.

## ADDED Requirements

### Requirement: Artifact provenance and license recording

The evaluation SHALL verify and record, before any run: model weights sha256,
config sha256, vendored model-code commit, adapter git sha, model license
name and a one-line terms summary, dependency versions, and the data artifact
names + shas it reads (`patches.zarr` provenance, `cycles_490_v2.parquet`,
`splits_locked_v2.json` sha `30ecbe10…`).

#### Scenario: Checkpoint mismatch fails loudly

- **WHEN** the weights file hash does not match the recorded sha256
- **THEN** extraction exits non-zero with the mismatch printed — it never
  proceeds with unidentified weights and never substitutes another checkpoint

### Requirement: Landsat adapter correctness

The adapter SHALL assemble the model's 11-band `landsat` token group from the
patch store's 8 physical channels using the documented mapping (SR bands
inverted through the C2 L2 scale to raw-DN space, LST Kelvin converted via
inverse Planck to B10 radiance DN, absent bands filled post-normalization at
0.0), then normalized with the model's computed.json statistics
(`(x − (μ−2σ)) / (4σ)`). The mapping SHALL be unit-tested against hand-computed
values, including the inverse-Planck conversion and the fill semantics.

#### Scenario: Missing bands are in-distribution fills

- **WHEN** a landsat slot has no source channel (B8, B1, B9, B11)
- **THEN** the normalized slot is filled with 0.0 (the trained band-dropout
  value), the fill is recorded in the artifact provenance, and the report
  lists the 4/11 missing-slot fraction as a limitation

#### Scenario: Invalid pixels use the model's MISSING mechanism

- **WHEN** a pixel is QA-invalid (`pixel_valid = 0`) inside a valid bin
- **THEN** the adapter sets that pixel's mask to `MaskValue.MISSING` rather
  than fabricating reflectance values, and pooling excludes MISSING tokens

### Requirement: Temporal windowing preserves phenology

The adapter SHALL pack only valid bins (`n_obs > 0` and ≥ 30 % in-field
QA-valid pixels, configurable) into windows of ≤ 12 consecutive bins, pass
real calendar timestamps (bin-center date) to the model, and preserve
`field_id, cycle_id, ratoon_no, bin_index, day_index, date` on every
observation-level embedding.

#### Scenario: Empty bins never enter the model

- **WHEN** a 15-day bin has `n_obs = 0`
- **THEN** it is skipped and never passed as a timestep (no fabricated
  observations), and the window's metadata records the skipped-bin count

### Requirement: Embedding artifact schema

Extraction SHALL write observation-level and farm-cycle-level embeddings as
Zarr plus a Parquet companion under `experiments/olmoearth_eval/outputs/`,
fp32, with a `provenance.json` carrying all provenance fields of this spec,
and SHALL record per-run progress (`progress.jsonl`) and a final summary line
with wall time and output paths.

#### Scenario: Smoke mode proves the chain

- **WHEN** `extract_embeddings.py --smoke` runs
- **THEN** it extracts a small field subset end to end, asserts finite
  embeddings with the documented shapes, and prints the summary line — any
  NaN, shape mismatch, or silent partial work is a defect

### Requirement: Evaluation batteries and mandatory baselines

The evaluation SHALL run: (a) representation analysis; (b) cross-year
phenological consistency with bootstrap CIs; (c) cross-year/cross-farm kNN
purity; (d) a linear stage probe with a **mandatory raw-channel control**
under identical splits and seeds; (e) controlled-deviation anomaly detection
on dev-split fields with at least the two classical baselines (robust-z on
raw channels; NDVI persistence) under train-only calibration, scored by
AUC-PR and recall@FAR=5 % over ≥ 3 seeds; (f) robustness arms (missing bins,
coarser sampling, fill sensitivity, spatial scale); (g) a CPU performance
benchmark with measured latency, RSS, model size, and a full-store
extrapolation.

#### Scenario: Probe claims require beating the control

- **WHEN** the report claims the embeddings carry stage information
  "beyond raw channels"
- **THEN** the OE probe exceeds the raw-channel probe macro-F1 with
  non-overlapping seed CIs, and otherwise the claim is written as a negative
  result

#### Scenario: Injected deviations are never named stress

- **WHEN** controlled deviations are injected for the detection battery
- **THEN** every table and sentence labels them as synthetic controlled
  deviations at L2, and real-data concordance is reported separately and
  never merged into detection accuracy

### Requirement: Pre-registered gates and verdict

Gates G-O1 (usability + probe-over-control), G-O2 (phase consistency margin),
G-O3 (detection AUC-PR ≥ B0), G-O4 (robustness — reported, no gate), and
G-O5 (feasibility ≤ 12 h CPU extrapolation) SHALL be evaluated as specified
in the design and the final report SHALL state one verdict — adopt / adapt /
reject — mapped from the gate outcomes, with every claim stating its
validation-ladder rung (L1–L2 only for this capability).

#### Scenario: Metrics never appear without context

- **WHEN** the report quotes any performance number
- **THEN** it is accompanied by n, date range, seeds, mean ± std, protocol
  name, and data_source — stale or context-free numbers are defects

### Requirement: Isolation from the production pipeline

All experimental code SHALL live under `experiments/olmoearth_eval/`, import
no `ssl_anomaly` modules, write no files outside its own directory
(excluding read-only shared data), and leave `Data/Processed/` untouched.

#### Scenario: Adoption does not silently integrate

- **WHEN** the verdict is "adopt"
- **THEN** integration happens only through a future, separately proposed
  change — this capability ends at the report and its artifacts
