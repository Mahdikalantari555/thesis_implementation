## ADDED Requirements

### Requirement: Representation ladder comparison

The system SHALL compare spatial representations under an equal-budget
protocol — R0 zonal statistics (reused results), R1 full-field 32×32 patch,
R2 sub-crop 16×16 with field-level aggregation, R3 multi-scale (multi-
dilation stem; two-scale fusion only behind its pre-registered trigger) —
holding objective, bins, channels, seeds, injection sets, σ tables and
alarm machinery constant, with param counts reported per representation and
>±20% capacity deviation flagged. All representations SHALL be evaluated on
the same farms and cycles through the unchanged severity protocol (3 shapes
+ secondary shapes), pixel-localization on partial-footprint injections,
and the benchmark metrics (epoch time, peak RAM, throughput). The study
SHALL publish ONE transparent table + figure across the four dimensions
(detection quality, spatial sensitivity, temporal consistency,
computational efficiency) as finding 24 (Persian), and a 64×64 window SHALL
be excluded from training by default with the geometry memo (median field
side 16.7 px, max 19.6 px) documented, available only behind an explicit
flag.

#### Scenario: Equal budget holds

- **WHEN** two representations are compared
- **THEN** they share farms, cycles, dates, injections, seeds and
  calibration, their param counts appear in the same table, and any
  >±20% capacity deviation is flagged next to the result

#### Scenario: Scale decisions are measured, not assumed

- **WHEN** the study begins
- **THEN** the 64×64 exclusion memo and the lazy-vs-materialized data-access
  decision are recorded from measurements (geometry + the benchmark JSON)
  before any training run

### Requirement: Sub-crop representation mechanics

R2 crops SHALL lie fully inside the field polygon (in-field fraction 1.0),
drawn as fixed seed-hashed sets recorded in the run manifest (per-epoch
resampling = flag, default off); fields admitting fewer than K complete
windows SHALL use a recorded reduced K and fields with no valid window a
recorded padded fallback; field-level scores SHALL aggregate per-crop
scores via top-q mean (primary) and majority vote (comparison variant) into
the unchanged alarm machinery.

#### Scenario: Crop sets are reproducible

- **WHEN** R2 trains or evaluates
- **THEN** the crop sets are byte-identical across runs from the manifest
  hash, and every reduced-K or padded-fallback field is listed in the
  report

### Requirement: Pre-registered adoption rule

A representation SHALL replace R1 only if its MDM80 at 5% cycle-FAR improves
with a paired-bootstrap CI over fields excluding zero at ≤2× R1 compute;
the multi-scale variant SHALL be adopted only if it beats the best single-
scale representation with a CI excluding zero. If no representation beats
R1, the honest negative SHALL be recorded with the full table and R1
continues as the patch representation.

#### Scenario: Adoption is decided verbatim

- **WHEN** the comparison completes
- **THEN** the pre-registered rule is applied to the committed table, the
  winner (or honest negative) is named in finding 24, and the decision
  flows into the final-architecture criterion

### Requirement: Benchmark module

A reusable benchmark module SHALL measure, before any training run: Zarr
lazy random-read throughput, materialized preload time and resident RAM,
per-representation epoch forward/backward throughput, CPU utilization,
peak RAM and disk I/O, and SHALL emit per-epoch and total runtime estimates
that gate the user-run training commands. Results SHALL be committed as a
JSON artifact with the recorded data-access decision.

#### Scenario: Estimates precede training

- **WHEN** a representation's training is handed to the user
- **THEN** its per-epoch and total estimates come from the measured
  benchmark artifact, not assumptions, and the lazy-vs-materialized decision
  cites the same artifact
